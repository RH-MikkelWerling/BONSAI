"""Does a pretrained sequence model reproduce its training patients?

Prompt the model with the first X events of a patient's record, let it generate
the next Y events greedily (each predicted code appended and fed back), and
compare them with the record:

- exact_match: all Y events reproduced, in order;
- overlap: share of the patient's next Y events the model also generated, in
  any order (special tokens left out);
- on_time: the same, counting only codes generated within 30 days of when they
  occurred (generated events take the time of the true event at that position,
  since the model predicts codes, not times);
- rare_codes: share of the patient's rare codes (held by fewer than 1 in 1,000
  training patients) the model also generated.

Predictable records match without any memorization, so each score is compared
with patients the model never saw: memorization is the excess of the training
patients' score. For BONSAI models a last check compares the next-event loss of
the two groups: an AUC of 0.5 means a patient's loss says nothing about whether
the model trained on them. ``summary.txt`` gives the numbers to decide on.

``--backend bonsai`` (our models): a causally pretrained BONSAI encoder, its
``subject_data_{train,tuning}.pt`` and ``vocabulary.pt``. Pretraining used every
window of each record, so the record's first events (window 0) are trained on.

    python scripts/memorization_check.py --backend bonsai \\
        --checkpoint <pretrain run>/version_0/epoch_010.ckpt \\
        --data-dir <processed dataset dir> --output-dir memorization

``--backend generate`` (e.g. PHAIR): a model with

    model.generate(batch={"event": LongTensor[B, T], "attn_mask": BoolTensor[B, T]},
                   max_new_tokens=Y, temperature=0, pad_token_id=PAD,
                   start_after_n_tokens=X, n_samples_per_person=1)
      -> {"generated": LongTensor[B, <=Y], ...}

loaded with ``Class.load_from_checkpoint``, and a parquet (or torch) file with
``subject_id``, ``split`` and ``event`` (token ids in training order). Event
times are not available there, so on_time is left empty and there is no loss
check; ``--ignore-tokens`` names the special tokens.

Only aggregates are written, unless ``--per-patient`` (internal use only).
"""

from __future__ import annotations

import argparse
import importlib
import inspect
import json
import math
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from torch.nn.utils.rnn import pad_sequence

METRICS = ("exact_match", "overlap", "on_time", "rare_codes")
WINDOW_YEARS = 30 / 365.25  # "on time": within 30 days (age is in years)
RARE_SHARE = 0.001  # rare: held by fewer than 1 in 1,000 training patients


# -- scores ---------------------------------------------------------------------


def sample(records: list, min_events: int, n: int) -> list:
    """Up to ``n`` patients with at least ``min_events`` events (fixed seed)."""
    eligible = [r for r in records if len(r["code"]) >= min_events]
    if n and len(eligible) > n:
        keep = np.random.default_rng(0).choice(len(eligible), n, replace=False)
        eligible = [eligible[i] for i in sorted(keep)]
    return eligible


def rare_codes(records: list) -> set:
    """Codes held by fewer than ``RARE_SHARE`` of these patients."""
    held = Counter(c for r in records for c in set(np.asarray(r["code"]).tolist()))
    return {c for c, n in held.items() if n < RARE_SHARE * len(records)}


def reproduced(generated, truth, keep: set, ages=None, window=None) -> tuple:
    """(true events reproduced, true events counted), over true codes in ``keep``.

    Order does not matter and each generated code reproduces at most one true
    event: anywhere in the continuation, or within ``window`` years of it.
    """
    used = np.zeros(len(generated), dtype=bool)
    hits = counted = 0
    for j, code in enumerate(truth):
        if code not in keep:
            continue
        counted += 1
        candidates = (generated == code) & ~used
        if window is not None:
            candidates &= np.abs(ages - ages[j]) <= window
        if candidates.any():
            used[candidates.argmax()] = True
            hits += 1
    return hits, counted


def score(generated, truth, ages, y: int, countable: set, rare: set) -> dict:
    """One patient's scores for the first ``y`` generated events."""
    g, t = generated[:y], truth[:y]
    hits, n = reproduced(g, t, countable)
    on_time = math.nan
    if ages is not None and n:
        on_time = reproduced(g, t, countable, ages[:y], WINDOW_YEARS)[0] / n
    rare_hits, rare_n = reproduced(g, t, rare)
    return {
        "exact_match": float((g == t).all()),
        "overlap": hits / n if n else math.nan,
        "on_time": on_time,
        "rare_codes": rare_hits / rare_n if rare_n else math.nan,
    }


def memorization_scores(
    generate,
    records_by_split: dict,
    prompts,
    continuations,
    max_patients,
    countable,
    rare,
) -> pl.DataFrame:
    """Per patient, split, X and Y: the four scores. ``generate(patients, X, Y)``
    returns [patients, Y] greedily generated codes."""
    longest, rows = max(continuations), []
    for split, records in records_by_split.items():
        for x in prompts:
            # The same patients for every Y: those with at least X + longest Y events.
            patients = sample(records, x + longest, max_patients)
            if not patients:
                continue
            generated = generate(patients, x, longest)
            for g, r in zip(generated, patients):
                truth = np.asarray(r["code"][x : x + longest])
                ages = np.asarray(r["age"][x : x + longest]) if "age" in r else None
                for y in continuations:
                    key = {
                        "split": split,
                        "subject_id": str(r.get("subject_id")),
                        "prompt_events": x,
                        "continuation_events": y,
                    }
                    rows.append(key | score(g, truth, ages, y, countable, rare))
    return pl.DataFrame(rows)


def summary_table(scores: pl.DataFrame) -> pl.DataFrame:
    """Mean of each score per split, X and Y."""
    keys = ["split", "prompt_events", "continuation_events"]
    return (
        scores.group_by(keys)
        .agg(
            pl.len().alias("patients"),
            *(pl.col(m).fill_nan(None).mean() for m in METRICS),
            pl.col("rare_codes").is_not_nan().sum().alias("patients_with_rare_codes"),
        )
        .sort(keys)
    )


def wilson_interval(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score interval for a proportion (Wilson, 1927); stays within
    [0, 1] and is not zero-width when no patient matches."""
    p = successes / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return max(0.0, centre - half), min(1.0, centre + half)


def difference(a: np.ndarray, b: np.ndarray, exact: bool) -> tuple:
    """Mean of a minus mean of b, with a 95% interval: Newcombe (1998, method 10)
    for the exact-match proportions, otherwise the normal (Welch) interval."""
    a, b = a[~np.isnan(a)], b[~np.isnan(b)]
    if len(a) < 2 or len(b) < 2:
        return math.nan, math.nan, math.nan
    p1, p2 = a.mean(), b.mean()
    diff = p1 - p2
    if exact:
        (l1, u1), (l2, u2) = (wilson_interval(int(v.sum()), len(v)) for v in (a, b))
        return (
            diff,
            diff - math.hypot(p1 - l1, u2 - p2),
            diff + math.hypot(u1 - p1, p2 - l2),
        )
    half = 1.96 * math.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
    return diff, diff - half, diff + half


def excess_table(scores: pl.DataFrame, train_split: str, heldout_split: str):
    """Training minus unseen patients, per score, X and Y, with 95% intervals."""
    rows = []
    for (x, y), part in scores.group_by(["prompt_events", "continuation_events"]):
        train = part.filter(pl.col("split") == train_split)
        unseen = part.filter(pl.col("split") == heldout_split)
        if train.is_empty() or unseen.is_empty():
            continue
        for metric in METRICS:
            a, b = train[metric].to_numpy(), unseen[metric].to_numpy()
            excess, lower, upper = difference(a, b, metric == "exact_match")
            rows.append(
                {
                    "metric": metric,
                    "prompt_events": x,
                    "continuation_events": y,
                    "patients_train": len(a),
                    "patients_unseen": len(b),
                    "train": np.nanmean(a) if (~np.isnan(a)).any() else math.nan,
                    "unseen": np.nanmean(b) if (~np.isnan(b)).any() else math.nan,
                    "excess": excess,
                    "excess_lower": lower,
                    "excess_upper": upper,
                }
            )
    return pl.DataFrame(rows).sort("metric", "prompt_events", "continuation_events")


def loss_check(a: np.ndarray, b: np.ndarray) -> pl.DataFrame:
    """Mean loss of training (a) and unseen (b) patients, and the AUC for telling
    them apart by loss (0.5 = not at all), with the Hanley & McNeil (1982) interval."""
    n1, n2 = len(a), len(b)
    gap, gap_half = b.mean() - a.mean(), 1.96 * math.sqrt(a.var() / n1 + b.var() / n2)
    auc = roc_auc_score([1] * n1 + [0] * n2, -np.concatenate([a, b]))
    q1, q2 = auc / (2 - auc), 2 * auc**2 / (1 + auc)
    var = auc * (1 - auc) + (n1 - 1) * (q1 - auc**2) + (n2 - 1) * (q2 - auc**2)
    auc_half = 1.96 * math.sqrt(var / (n1 * n2))
    return pl.DataFrame(
        {
            "patients_train": n1,
            "patients_unseen": n2,
            "loss_train": a.mean(),
            "loss_unseen": b.mean(),
            "loss_gap": gap,
            "loss_gap_lower": gap - gap_half,
            "loss_gap_upper": gap + gap_half,
            "membership_auc": auc,
            "membership_auc_lower": auc - auc_half,
            "membership_auc_upper": auc + auc_half,
        }
    )


def decision_summary(excess: pl.DataFrame, loss: pl.DataFrame | None) -> str:
    """The numbers to decide on: each score's largest excess, and the loss AUC."""
    lines = ["Largest excess of training over unseen patients (95% CI):"]
    for metric in METRICS:
        rows = excess.filter(pl.col("metric") == metric).drop_nans("excess")
        if rows.is_empty():
            continue
        r = rows.sort("excess", descending=True).row(0, named=True)
        lines.append(
            f"  {metric:12s} {r['excess']:+.3f} ({r['excess_lower']:+.3f} to "
            f"{r['excess_upper']:+.3f}) at X = {r['prompt_events']}, "
            f"Y = {r['continuation_events']}"
        )
    if loss is not None:
        r = loss.row(0, named=True)
        lines.append(
            f"Membership AUC from loss: {r['membership_auc']:.3f} "
            f"({r['membership_auc_lower']:.3f} to {r['membership_auc_upper']:.3f})"
        )
    return "\n".join(lines)


def plot_curves(
    summary: pl.DataFrame, path: Path, train: str, heldout: str, loss=None
) -> None:
    """All four scores against Y: one colour per X, solid = training, dashed =
    unseen; the loss AUC, if any, in the title."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:  # pragma: no cover
        return
    fig, axes = plt.subplots(2, 2, figsize=(9.5, 7.2), sharex=True)
    prompts = summary["prompt_events"].unique().sort()
    colours = plt.cm.viridis(np.linspace(0.1, 0.85, len(prompts)))
    panels = {
        "exact_match": "All Y events reproduced, in order",
        "overlap": "Share of the next Y events reproduced (any order)",
        "on_time": "Same, within 30 days of when they occurred",
        "rare_codes": "Share of the patient's rare codes reproduced",
    }
    for ax, (metric, title) in zip(axes.flat, panels.items()):
        for colour, x in zip(colours, prompts):
            for split, style in ((train, "-"), (heldout, "--")):
                part = summary.filter(
                    pl.col("prompt_events") == x, pl.col("split") == split
                )
                ax.plot(
                    part["continuation_events"],
                    part[metric],
                    style,
                    color=colour,
                    marker="o",
                    markersize=3,
                    label=f"X = {x:,}" if split == train else None,
                )
        if summary[metric].is_null().all():
            ax.text(
                0.5, 0.5, "No patients to score", ha="center", transform=ax.transAxes
            )
        ax.set_xscale("log", base=2)
        ax.set_title(title, fontsize=9)
    for ax in axes[1]:
        ax.set_xlabel("Events generated after the prompt (Y)")
    for ax in axes[:, 0]:
        ax.set_ylabel("Mean over patients")
    axes[0, 1].plot([], [], "k-", label="Training patients")
    axes[0, 1].plot([], [], "k--", label="Unseen patients")
    axes[0, 1].legend(fontsize=6.5, frameon=False, ncol=2)
    title = (
        "Reproduction of a patient's next Y events after an X-event prompt:\n"
        "patients the model was trained on vs patients it never saw"
    )
    if loss is not None:
        r = loss.row(0, named=True)
        title += (
            f"\nLoss AUC, training vs unseen: {r['membership_auc']:.3f} "
            f"({r['membership_auc_lower']:.3f} to {r['membership_auc_upper']:.3f})"
        )
    fig.suptitle(title, fontsize=9.5)
    fig.tight_layout()
    fig.savefig(path, dpi=200)


# -- BONSAI backend ---------------------------------------------------------------

BONSAI_FIELDS = (
    "code",
    "age",
    "abspos",
    "segment",
    "value_bin",
    "value_normalized",
    "value_present",
    "numeric_value",
)


def load_bonsai(checkpoint, device):
    """The pretraining network (encoder and next-code head) from a Lightning checkpoint."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # this repository
    from bonsai.functional.checkpointing import (
        clean_lightning_state_dict,
        get_saved_encoder_config,
    )
    from opera.compat.bonsai import build_bonsai_pretrain

    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    config = get_saved_encoder_config(payload["hyper_parameters"])
    if not config.get("causal", False):
        raise SystemExit("Not a causal next-code model: continuation is undefined")
    model = build_bonsai_pretrain(config)
    model.load_state_dict(
        clean_lightning_state_dict(payload["state_dict"]), strict=True
    )
    return model.eval().to(device)


def bonsai_batch(subjects, length, device):
    """The first ``length`` events of each subject, as pretraining feeds them."""
    from bonsai.functional.normalization import normalize_segments

    batch = {}
    for key in BONSAI_FIELDS:
        if all(key in s for s in subjects):
            rows = [s[key][:length] for s in subjects]
            if key == "segment":
                rows = [torch.as_tensor(normalize_segments(r)) for r in rows]
            batch[key] = pad_sequence(rows, batch_first=True).to(device)
    lengths = torch.tensor(
        [min(len(s["code"]), length) for s in subjects], device=device
    )
    width = batch["code"].shape[1]
    batch["attention_mask"] = torch.arange(width, device=device) < lengths[:, None]
    return batch


def hidden_states(model, batch):
    """Causal attention: position i has seen events 0..i, so it predicts event i + 1."""
    with torch.autocast("cuda", torch.float16, enabled=batch["code"].is_cuda):
        return model.encode(batch).float()


@torch.inference_mode()
def bonsai_generate(model, subjects, prompt, steps, batch_size):
    """Greedy decoding: [patients, steps] codes generated after ``prompt`` events."""
    device, out = next(model.parameters()).device, []
    for start in range(0, len(subjects), batch_size):
        batch = bonsai_batch(
            subjects[start : start + batch_size], prompt + steps, device
        )
        codes = batch["code"].clone()  # the true record, overwritten as we generate
        for i in range(prompt, prompt + steps):
            seen = {k: v[:, :i] for k, v in batch.items()} | {"code": codes[:, :i]}
            last = hidden_states(model, seen)[:, -1]
            codes[:, i] = model.pretrain_head(last).argmax(-1)
        out.append(codes[:, prompt:].cpu().numpy())
    return np.concatenate(out)


@torch.inference_mode()
def bonsai_losses(model, subjects, length, batch_size, chunk=512):
    """Each patient's mean next-event loss over their first ``length`` events."""
    device, out = next(model.parameters()).device, []
    for start in range(0, len(subjects), batch_size):
        part = subjects[start : start + batch_size]
        n = min(max(len(s["code"]) for s in part), length)
        batch = bonsai_batch(part, n - 1, device)
        target = pad_sequence([s["code"][1:n] for s in part], batch_first=True)
        target = target.to(device)
        hidden = hidden_states(model, batch)
        total = torch.zeros(len(part), device=device)
        for left in range(0, n - 1, chunk):  # logits are batch x chunk x vocab
            logits = model.pretrain_head(hidden[:, left : left + chunk]).float()
            total += F.cross_entropy(
                logits.transpose(1, 2),
                target[:, left : left + chunk],
                ignore_index=0,
                reduction="none",
            ).sum(1)
        out += (total / (target != 0).sum(1)).tolist()
    return np.array(out)


# -- generate backend ---------------------------------------------------------------


def load_sequences(path: Path) -> pd.DataFrame:
    """``subject_id``, ``split`` and ``event`` (list of ints) per patient."""
    path = Path(path)
    if path.suffix in {".parquet", ".pq"}:
        frame = pd.read_parquet(path)
    else:
        records = torch.load(path, map_location="cpu", weights_only=False)
        if isinstance(records, dict):
            records = list(records.values())
        frame = pd.DataFrame(
            [{**r, "event": r.get("event", r.get("code"))} for r in records]
        )
    if "event" not in frame and "code" in frame:
        frame = frame.rename(columns={"code": "event"})
    missing = {"subject_id", "split", "event"} - set(frame)
    if missing:
        raise ValueError(f"Sequences need subject_id, split, event; missing {missing}")
    frame["event"] = frame.event.map(lambda e: np.asarray(e, dtype=int))
    return frame[["subject_id", "split", "event"]]


def load_generator(spec, checkpoint, device, dtype):
    module, _, name = spec.partition(":")
    cls = getattr(importlib.import_module(module), name)
    model = cls.load_from_checkpoint(checkpoint, map_location="cpu")
    model.eval()
    model.to(device)
    prepare = getattr(model, "prepare_for_inference", None)
    if prepare is not None:
        accepted = inspect.signature(prepare).parameters
        dtype = None if dtype == "auto" else getattr(torch, dtype)
        options = {"device": device, "dtype": dtype}
        prepare(**{k: v for k, v in options.items() if k in accepted and v is not None})
    return model


@torch.inference_mode()
def model_generate(model, records, prompt, steps, *, pad, device, batch_size):
    """Greedy continuations of ``steps`` tokens after the first ``prompt`` tokens."""
    out, width = [], prompt + steps
    for start in range(0, len(records), batch_size):
        part = records[start : start + batch_size]
        events = torch.full((len(part), width), pad, dtype=torch.long)
        for i, r in enumerate(part):
            events[i] = torch.as_tensor(r["code"][:width])
        batch = {"event": events.to(device), "attn_mask": (events != pad).to(device)}
        result = model.generate(
            batch,
            max_new_tokens=steps,
            temperature=0,
            pad_token_id=pad,
            start_after_n_tokens=prompt,
            n_samples_per_person=1,
        )
        generated = result["generated"].detach().cpu()
        if generated.shape[1] < steps:  # stopped early: pad (never a match)
            filler = torch.full((generated.shape[0], steps - generated.shape[1]), -1)
            generated = torch.cat([generated, filler], dim=1)
        out.append(generated[:, :steps].numpy())
    return np.concatenate(out)


# -- main ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    add = parser.add_argument
    add("--backend", choices=("bonsai", "generate"), default="bonsai")
    add("--checkpoint", type=Path, required=True)
    add("--data-dir", type=Path, help="bonsai: subject_data_<split>.pt, vocabulary.pt")
    add("--model-class", help="generate: module.path:ClassName")
    add("--sequences", type=Path, help="generate: sequence file")
    add(
        "--ignore-tokens",
        default="",
        help="generate: special token ids, comma-separated",
    )
    add("--pad-token-id", type=int, default=0)
    add("--dtype", default="auto", help="generate: e.g. bfloat16")
    add("--train-split", default="train")
    add("--heldout-split", default="tuning")
    add("--prompt-events", type=int, nargs="+", default=[16, 64, 256, 1024, 4096])
    add("--continuation-events", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32, 64])
    add("--max-patients", type=int, default=2000, help="per split and prompt length")
    add("--batch-size", type=int, default=8)
    add("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    add(
        "--per-patient",
        action="store_true",
        help="also write per-patient scores (INTERNAL)",
    )
    add("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    started = time.time()
    splits = (args.train_split, args.heldout_split)
    continuations = sorted(args.continuation_events)
    longest = max(args.prompt_events) + max(continuations)
    loss = None
    if args.backend == "bonsai":
        if not args.data_dir:
            parser.error("--backend bonsai needs --data-dir")
        model = load_bonsai(args.checkpoint, args.device)
        context = int(model.hparams["max_seqlen"])
        if longest > context:
            parser.error(f"prompt + continuation exceeds the {context}-event context.")
        records = {
            s: torch.load(args.data_dir / f"subject_data_{s}.pt", weights_only=False)
            for s in splits
        }
        vocabulary = torch.load(args.data_dir / "vocabulary.pt", weights_only=False)
        countable = {i for token, i in vocabulary.items() if not token.startswith("[")}

        def generate(patients, x, y):
            return bonsai_generate(model, patients, x, y, args.batch_size)

        a, b = (
            bonsai_losses(
                model,
                sample(records[s], 2, args.max_patients),
                context + 1,
                args.batch_size,
            )
            for s in splits
        )
        loss = loss_check(a, b)
    else:
        if not (args.model_class and args.sequences):
            parser.error("--backend generate needs --model-class and --sequences")
        frame = load_sequences(args.sequences)
        records = {
            s: [
                {"subject_id": i, "code": e}
                for i, e in zip(part.subject_id, part.event)
            ]
            for s, part in ((s, frame[frame.split == s]) for s in splits)
        }
        model = load_generator(
            args.model_class, args.checkpoint, args.device, args.dtype
        )
        ignore = {int(v) for v in args.ignore_tokens.split(",") if v.strip()}
        seen = {int(c) for r in records[args.train_split] for c in r["code"]}
        countable = seen - ignore - {args.pad_token_id}

        def generate(patients, x, y):
            return model_generate(
                model,
                patients,
                x,
                y,
                pad=args.pad_token_id,
                device=args.device,
                batch_size=args.batch_size,
            )

    rare = rare_codes(records[args.train_split]) & countable
    scores = memorization_scores(
        generate,
        records,
        args.prompt_events,
        continuations,
        args.max_patients,
        countable,
        rare,
    )
    if scores.is_empty():
        raise SystemExit("No patient had a sequence long enough for these lengths")
    summary = summary_table(scores)
    excess = excess_table(scores, args.train_split, args.heldout_split)
    report = decision_summary(excess, loss)

    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    summary.write_csv(out / "memorization_summary.csv")
    excess.write_csv(out / "memorization_excess.csv")
    if loss is not None:
        loss.write_csv(out / "loss_check.csv")
    plot_curves(summary, out / "memorization_curves.png", *splits, loss)
    (out / "summary.txt").write_text(report + "\n")
    if args.per_patient:
        scores.write_parquet(out / "memorization_per_patient_INTERNAL.parquet")
    run = {
        "backend": args.backend,
        "decoding": "greedy, each generated code appended and fed back",
        "checkpoint": str(args.checkpoint),
        "data": str(args.data_dir if args.backend == "bonsai" else args.sequences),
        "splits": splits,
        "prompt_events": args.prompt_events,
        "continuation_events": continuations,
        "max_patients": args.max_patients,
        "on_time_window_days": 30,
        "rare_share": RARE_SHARE,
        "rare_codes": len(rare),
        "seconds": round(time.time() - started, 1),
    }
    (out / "run.json").write_text(json.dumps(run, indent=2))
    print(report)


if __name__ == "__main__":
    main()
