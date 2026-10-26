"""Does a pretrained sequence model reproduce the records it was trained on?

We give the model the first X events of a patient's record and let it continue
the record for Y events, one event at a time, always taking its most likely next
code. Then we compare what it wrote with what really happened next, using four
scores:

    exact_match  all Y events reproduced, in the right order
    overlap      share of the next Y true events that the model also wrote,
                 in any order (special tokens are not counted)
    on_time      as overlap, but a written code only counts if the true event
                 happened within 30 days of the event it stands in for (the
                 model writes codes, not times)
    rare_codes   share of the patient's rare codes (held by fewer than 1 in
                 1,000 training patients) that the model also wrote

A predictable record can be continued correctly without any memorization, so
every score is computed both for patients the model was trained on and for
patients it never saw. Memorization shows up as an *excess*: training patients
scoring higher than unseen ones. For BONSAI models we also compare the
next-event loss of the two groups: if the loss cannot tell them apart (AUC near
0.5), the model has not memorized who it was trained on.

summary.txt holds the numbers to decide on. It only uses (X, Y) combinations
with at least --min-patients scored patients in each group; long prompts leave
few patients, and their noisy differences would otherwise dominate.

Our own models (``--backend bonsai``): a causally pretrained BONSAI checkpoint
and the processed data directory (subject_data_<split>.pt and vocabulary.pt).
Pretraining used every window of each record, so the start of every training
record was trained on.

    python scripts/memorization_check.py --backend bonsai \\
        --checkpoint <pretrain run>/version_0/epoch_010.ckpt \\
        --data-dir <processed dataset dir> --output-dir memorization

Other models (``--backend generate``, e.g. PHAIR): a Lightning class with a
``generate`` method,

    model.generate(batch={"event": LongTensor[B, T], "attn_mask": BoolTensor[B, T]},
                   max_new_tokens=Y, temperature=0, pad_token_id=PAD,
                   start_after_n_tokens=X, n_samples_per_person=1)
      -> {"generated": LongTensor[B, <=Y], ...}

and a parquet (or torch) file with ``subject_id``, ``split`` and ``event``
(token ids in order). There are no event times, so on_time stays empty and the
loss check is skipped. ``--ignore-tokens`` lists the special tokens.

Only aggregates are written, unless --per-patient is given (internal use only).
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

SCORES = ("exact_match", "overlap", "on_time", "rare_codes")
ON_TIME_DAYS = 30
ON_TIME_YEARS = ON_TIME_DAYS / 365.25  # ages in the records are in years
RARE_CODE_SHARE = 0.001
MIN_PATIENTS = 50


# -- Choosing patients and rare codes ---------------------------------------------


def patients_with_enough_events(records, min_events, max_patients):
    """Up to ``max_patients`` patients with at least ``min_events`` events.

    The random draw uses a fixed seed, so reruns score the same patients.
    """
    eligible = [record for record in records if len(record["code"]) >= min_events]
    if not max_patients or len(eligible) <= max_patients:
        return eligible
    chosen = np.random.default_rng(0).choice(len(eligible), max_patients, replace=False)
    return [eligible[i] for i in sorted(chosen)]


def find_rare_codes(records):
    """Codes that fewer than ``RARE_CODE_SHARE`` of these patients have."""
    patients_per_code = Counter()
    for record in records:
        patients_per_code.update(set(np.asarray(record["code"]).tolist()))
    limit = RARE_CODE_SHARE * len(records)
    return {code for code, patients in patients_per_code.items() if patients < limit}


# -- Scoring one patient -------------------------------------------------------------


def count_reproduced(generated, truth, codes_to_count, ages=None, window=None):
    """How many true events the model reproduced, out of how many counted.

    Only true events whose code is in ``codes_to_count`` are counted. Order does
    not matter, but each generated code can stand in for one true event only.
    With ``ages`` and ``window`` the generated code must also sit within
    ``window`` years of the true event (a generated event takes the age of the
    true event at its position).
    """
    already_used = np.zeros(len(generated), dtype=bool)
    reproduced = counted = 0
    for position, code in enumerate(truth):
        if code not in codes_to_count:
            continue
        counted += 1
        matches = (generated == code) & ~already_used
        if window is not None:
            matches &= np.abs(ages - ages[position]) <= window
        if matches.any():
            already_used[matches.argmax()] = True
            reproduced += 1
    return reproduced, counted


def score_patient(generated, truth, ages, y, countable_codes, rare_codes):
    """The four scores for the first ``y`` generated events of one patient."""
    generated, truth = generated[:y], truth[:y]

    reproduced, counted = count_reproduced(generated, truth, countable_codes)
    overlap = reproduced / counted if counted else math.nan

    on_time = math.nan
    if ages is not None and counted:
        in_window, _ = count_reproduced(
            generated, truth, countable_codes, ages[:y], ON_TIME_YEARS
        )
        on_time = in_window / counted

    rare_reproduced, rare_counted = count_reproduced(generated, truth, rare_codes)
    rare = rare_reproduced / rare_counted if rare_counted else math.nan

    return {
        "exact_match": float((generated == truth).all()),
        "overlap": overlap,
        "on_time": on_time,
        "rare_codes": rare,
    }


def score_all_patients(
    generate, records_by_split, prompts, continuations, max_patients,
    countable_codes, rare_codes,
):
    """One row per patient, split, prompt length X and continuation length Y.

    ``generate(patients, x, y)`` returns an array [patients, y] of the codes
    the model wrote after each patient's first ``x`` events. Each X is
    generated once at the longest Y; shorter Ys score the start of it.
    """
    longest = max(continuations)
    rows = []
    for split, records in records_by_split.items():
        for x in prompts:
            # The same patients for every Y: those with X + longest Y events.
            patients = patients_with_enough_events(records, x + longest, max_patients)
            if not patients:
                continue
            generated = generate(patients, x, longest)
            for written, record in zip(generated, patients):
                truth = np.asarray(record["code"][x : x + longest])
                ages = None
                if "age" in record:
                    ages = np.asarray(record["age"][x : x + longest])
                for y in continuations:
                    scores = score_patient(
                        written, truth, ages, y, countable_codes, rare_codes
                    )
                    rows.append({
                        "split": split,
                        "subject_id": str(record.get("subject_id")),
                        "prompt_events": x,
                        "continuation_events": y,
                        **scores,
                    })
    return pl.DataFrame(rows)


# -- Training against unseen patients ------------------------------------------------


def mean_per_split(scores):
    """Mean of each score per split, X and Y, with patient counts."""
    keys = ["split", "prompt_events", "continuation_events"]
    means = [pl.col(score).fill_nan(None).mean() for score in SCORES]
    return (
        scores.group_by(keys)
        .agg(
            pl.len().alias("patients"),
            *means,
            pl.col("rare_codes").is_not_nan().sum().alias("patients_with_rare_codes"),
        )
        .sort(keys)
    )


def wilson_interval(successes, n, z=1.96):
    """95% Wilson interval for a proportion (Wilson 1927). Unlike the normal
    interval it stays inside [0, 1] and is not empty when no patient matches."""
    p = successes / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half_width = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return max(0.0, centre - half_width), min(1.0, centre + half_width)


def difference_in_means(train, unseen, is_proportion):
    """Mean of ``train`` minus mean of ``unseen``, with a 95% interval.

    Exact match is a yes/no score, so its interval is Newcombe's (1998, method
    10), built from the two Wilson intervals; the other scores use the normal
    (Welch) interval. Patients without a score (NaN) are left out.
    """
    train, unseen = train[~np.isnan(train)], unseen[~np.isnan(unseen)]
    if len(train) < 2 or len(unseen) < 2:
        return math.nan, math.nan, math.nan
    p_train, p_unseen = train.mean(), unseen.mean()
    difference = p_train - p_unseen
    if is_proportion:
        low_train, high_train = wilson_interval(int(train.sum()), len(train))
        low_unseen, high_unseen = wilson_interval(int(unseen.sum()), len(unseen))
        lower = difference - math.hypot(p_train - low_train, high_unseen - p_unseen)
        upper = difference + math.hypot(high_train - p_train, p_unseen - low_unseen)
        return difference, lower, upper
    standard_error = math.sqrt(
        train.var(ddof=1) / len(train) + unseen.var(ddof=1) / len(unseen)
    )
    return difference, difference - 1.96 * standard_error, difference + 1.96 * standard_error


def excess_over_unseen(scores, train_split, unseen_split):
    """Per score, X and Y: training minus unseen patients, with 95% intervals.

    The patient counts are the patients who had that score (rare_codes, for
    example, only exists for patients with a rare code among their next Y events).
    """
    rows = []
    for (x, y), part in scores.group_by(["prompt_events", "continuation_events"]):
        in_train = part.filter(pl.col("split") == train_split)
        in_unseen = part.filter(pl.col("split") == unseen_split)
        if in_train.is_empty() or in_unseen.is_empty():
            continue
        for score in SCORES:
            train = in_train[score].to_numpy()
            unseen = in_unseen[score].to_numpy()
            train, unseen = train[~np.isnan(train)], unseen[~np.isnan(unseen)]
            excess, lower, upper = difference_in_means(
                train, unseen, is_proportion=(score == "exact_match")
            )
            rows.append({
                "metric": score,
                "prompt_events": x,
                "continuation_events": y,
                "patients_train": len(train),
                "patients_unseen": len(unseen),
                "train": train.mean() if len(train) else math.nan,
                "unseen": unseen.mean() if len(unseen) else math.nan,
                "excess": excess,
                "excess_lower": lower,
                "excess_upper": upper,
            })
    return pl.DataFrame(rows).sort("metric", "prompt_events", "continuation_events")


def membership_from_loss(train_losses, unseen_losses):
    """Can a patient's loss tell whether the model trained on them?

    Returns the mean loss of each group, their gap, and the AUC for separating
    training from unseen patients by loss (0.5 = not at all), with the Hanley
    and McNeil (1982) interval.
    """
    n_train, n_unseen = len(train_losses), len(unseen_losses)
    gap = unseen_losses.mean() - train_losses.mean()
    gap_half_width = 1.96 * math.sqrt(
        train_losses.var() / n_train + unseen_losses.var() / n_unseen
    )
    is_train = [1] * n_train + [0] * n_unseen
    auc = roc_auc_score(is_train, -np.concatenate([train_losses, unseen_losses]))
    q1, q2 = auc / (2 - auc), 2 * auc**2 / (1 + auc)
    variance = (
        auc * (1 - auc) + (n_train - 1) * (q1 - auc**2) + (n_unseen - 1) * (q2 - auc**2)
    ) / (n_train * n_unseen)
    auc_half_width = 1.96 * math.sqrt(variance)
    return pl.DataFrame({
        "patients_train": n_train,
        "patients_unseen": n_unseen,
        "loss_train": train_losses.mean(),
        "loss_unseen": unseen_losses.mean(),
        "loss_gap": gap,
        "loss_gap_lower": gap - gap_half_width,
        "loss_gap_upper": gap + gap_half_width,
        "membership_auc": auc,
        "membership_auc_lower": auc - auc_half_width,
        "membership_auc_upper": auc + auc_half_width,
    })


# -- Report and figure ---------------------------------------------------------------


def write_summary(excess, loss, min_patients):
    """The numbers to decide on: each score's largest excess among (X, Y)
    cells with at least ``min_patients`` scored patients per group, and the
    loss AUC."""
    lines = [
        "Largest excess of training over unseen patients (95% CI), among "
        f"X, Y with at least {min_patients} scored patients in each group:"
    ]
    for score in SCORES:
        cells = excess.filter(
            pl.col("metric") == score,
            pl.col("patients_train") >= min_patients,
            pl.col("patients_unseen") >= min_patients,
        ).drop_nans("excess")
        too_small = excess.filter(pl.col("metric") == score).height - cells.height
        if cells.is_empty():
            lines.append(f"  {score:12s} no cell has enough patients")
            continue
        top = cells.sort("excess", descending=True).row(0, named=True)
        lines.append(
            f"  {score:12s} {top['excess']:+.3f} "
            f"({top['excess_lower']:+.3f} to {top['excess_upper']:+.3f}) "
            f"at X = {top['prompt_events']}, Y = {top['continuation_events']} "
            f"(n = {top['patients_train']} / {top['patients_unseen']}; "
            f"{too_small} smaller cells left out)"
        )
    if loss is not None:
        row = loss.row(0, named=True)
        lines.append(
            f"Membership AUC from loss: {row['membership_auc']:.3f} "
            f"({row['membership_auc_lower']:.3f} to {row['membership_auc_upper']:.3f})"
        )
    return "\n".join(lines)


def plot_curves(means, path, train_split, unseen_split, loss=None):
    """Each score against Y: one colour per X, solid for training patients and
    dashed for unseen ones."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:  # pragma: no cover
        return
    titles = {
        "exact_match": "All Y events reproduced, in order",
        "overlap": "Share of the next Y events reproduced (any order)",
        "on_time": f"Same, within {ON_TIME_DAYS} days of when they occurred",
        "rare_codes": "Share of the patient's rare codes reproduced",
    }
    prompts = means["prompt_events"].unique().sort()
    colours = plt.cm.viridis(np.linspace(0.1, 0.85, len(prompts)))
    fig, axes = plt.subplots(2, 2, figsize=(9.5, 7.2), sharex=True)
    for ax, (score, title) in zip(axes.flat, titles.items()):
        for colour, x in zip(colours, prompts):
            for split, line in ((train_split, "-"), (unseen_split, "--")):
                curve = means.filter(pl.col("prompt_events") == x, pl.col("split") == split)
                ax.plot(
                    curve["continuation_events"], curve[score], line,
                    color=colour, marker="o", markersize=3,
                    label=f"X = {x:,}" if split == train_split else None,
                )
        if means[score].is_null().all():
            ax.text(0.5, 0.5, "No patients to score", ha="center", transform=ax.transAxes)
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
        row = loss.row(0, named=True)
        title += (
            f"\nLoss AUC, training vs unseen: {row['membership_auc']:.3f} "
            f"({row['membership_auc_lower']:.3f} to {row['membership_auc_upper']:.3f})"
        )
    fig.suptitle(title, fontsize=9.5)
    fig.tight_layout()
    fig.savefig(path, dpi=200)


# -- BONSAI models -------------------------------------------------------------------

BONSAI_FIELDS = (
    "code", "age", "abspos", "segment",
    "value_bin", "value_normalized", "value_present", "numeric_value",
)


def load_bonsai(checkpoint, device):
    """The pretraining network (encoder and next-code head) from a checkpoint."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # this repository
    from bonsai.functional.checkpointing import (
        clean_lightning_state_dict,
        get_saved_encoder_config,
    )
    from opera.compat.bonsai import build_bonsai_pretrain

    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    config = get_saved_encoder_config(saved["hyper_parameters"])
    if not config.get("causal", False):
        raise SystemExit("Not a causal next-code model, so it cannot continue a record")
    model = build_bonsai_pretrain(config)
    model.load_state_dict(clean_lightning_state_dict(saved["state_dict"]), strict=True)
    return model.eval().to(device)


def bonsai_batch(subjects, length, device):
    """The first ``length`` events of each subject, padded as in pretraining."""
    from bonsai.functional.normalization import normalize_segments

    batch = {}
    for field in BONSAI_FIELDS:
        if not all(field in subject for subject in subjects):
            continue
        rows = [subject[field][:length] for subject in subjects]
        if field == "segment":
            rows = [torch.as_tensor(normalize_segments(row)) for row in rows]
        batch[field] = pad_sequence(rows, batch_first=True).to(device)
    lengths = torch.tensor([min(len(s["code"]), length) for s in subjects], device=device)
    positions = torch.arange(batch["code"].shape[1], device=device)
    batch["attention_mask"] = positions < lengths[:, None]
    return batch


def encode(model, batch):
    """Hidden states. Attention is causal: position i has seen events 0..i and
    predicts event i + 1."""
    with torch.autocast("cuda", torch.float16, enabled=batch["code"].is_cuda):
        return model.encode(batch).float()


@torch.inference_mode()
def bonsai_generate(model, subjects, prompt, steps, batch_size):
    """Greedy continuation: [patients, steps] codes written after ``prompt`` events.

    Each written code replaces the true one before the next step, so the model
    only ever sees its own continuation. Times, ages and values stay those of
    the true record.
    """
    device = next(model.parameters()).device
    written = []
    for start in range(0, len(subjects), batch_size):
        batch = bonsai_batch(subjects[start : start + batch_size], prompt + steps, device)
        codes = batch["code"].clone()
        for position in range(prompt, prompt + steps):
            seen = {field: values[:, :position] for field, values in batch.items()}
            seen["code"] = codes[:, :position]
            last_hidden = encode(model, seen)[:, -1]
            codes[:, position] = model.pretrain_head(last_hidden).argmax(-1)
        written.append(codes[:, prompt:].cpu().numpy())
    return np.concatenate(written)


@torch.inference_mode()
def bonsai_losses(model, subjects, length, batch_size, chunk=512):
    """Each patient's mean next-event loss over their first ``length`` events."""
    device = next(model.parameters()).device
    losses = []
    for start in range(0, len(subjects), batch_size):
        part = subjects[start : start + batch_size]
        n = min(max(len(subject["code"]) for subject in part), length)
        hidden = encode(model, bonsai_batch(part, n - 1, device))
        targets = pad_sequence([s["code"][1:n] for s in part], batch_first=True).to(device)
        total = torch.zeros(len(part), device=device)
        # In chunks: the logits are batch x positions x vocabulary.
        for left in range(0, n - 1, chunk):
            logits = model.pretrain_head(hidden[:, left : left + chunk]).float()
            total += F.cross_entropy(
                logits.transpose(1, 2), targets[:, left : left + chunk],
                ignore_index=0, reduction="none",
            ).sum(1)
        losses += (total / (targets != 0).sum(1)).tolist()
    return np.array(losses)


def bonsai_backend(args):
    """Records, generate function, countable codes and loss check for BONSAI."""
    model = load_bonsai(args.checkpoint, args.device)
    context = int(model.hparams["max_seqlen"])
    if max(args.prompt_events) + max(args.continuation_events) > context:
        raise SystemExit(f"Prompt plus continuation exceeds the {context}-event context")

    splits = (args.train_split, args.heldout_split)
    records = {
        split: torch.load(args.data_dir / f"subject_data_{split}.pt", weights_only=False)
        for split in splits
    }
    vocabulary = torch.load(args.data_dir / "vocabulary.pt", weights_only=False)
    countable = {i for token, i in vocabulary.items() if not token.startswith("[")}

    def generate(patients, x, y):
        return bonsai_generate(model, patients, x, y, args.batch_size)

    train_losses, unseen_losses = [
        bonsai_losses(
            model,
            patients_with_enough_events(records[split], 2, args.max_patients),
            context + 1,
            args.batch_size,
        )
        for split in splits
    ]
    loss = membership_from_loss(train_losses, unseen_losses)
    return records, generate, countable, loss


# -- Other models (generate backend) ---------------------------------------------------


def load_sequences(path):
    """``subject_id``, ``split`` and ``event`` (an integer array) per patient."""
    path = Path(path)
    if path.suffix in {".parquet", ".pq"}:
        frame = pd.read_parquet(path)
    else:
        records = torch.load(path, map_location="cpu", weights_only=False)
        if isinstance(records, dict):
            records = list(records.values())
        frame = pd.DataFrame(
            [{**record, "event": record.get("event", record.get("code"))} for record in records]
        )
    if "event" not in frame and "code" in frame:
        frame = frame.rename(columns={"code": "event"})
    missing = {"subject_id", "split", "event"} - set(frame)
    if missing:
        raise ValueError(f"Sequences need subject_id, split and event; missing {missing}")
    frame["event"] = frame.event.map(lambda events: np.asarray(events, dtype=int))
    return frame[["subject_id", "split", "event"]]


def load_generator(class_path, checkpoint, device, dtype):
    """A Lightning model given as ``module.path:ClassName``."""
    module, _, name = class_path.partition(":")
    model_class = getattr(importlib.import_module(module), name)
    model = model_class.load_from_checkpoint(checkpoint, map_location="cpu")
    model.eval()
    model.to(device)
    prepare = getattr(model, "prepare_for_inference", None)
    if prepare is not None:
        accepted = inspect.signature(prepare).parameters
        options = {"device": device}
        if dtype != "auto":
            options["dtype"] = getattr(torch, dtype)
        prepare(**{key: value for key, value in options.items() if key in accepted})
    return model


@torch.inference_mode()
def model_generate(model, records, prompt, steps, *, pad, device, batch_size):
    """Greedy continuation of ``steps`` tokens after each record's first ``prompt``."""
    width = prompt + steps
    written = []
    for start in range(0, len(records), batch_size):
        part = records[start : start + batch_size]
        events = torch.full((len(part), width), pad, dtype=torch.long)
        for row, record in enumerate(part):
            events[row] = torch.as_tensor(record["code"][:width])
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
        if generated.shape[1] < steps:
            # The model stopped early: pad with -1, which never matches a code.
            filler = torch.full((generated.shape[0], steps - generated.shape[1]), -1)
            generated = torch.cat([generated, filler], dim=1)
        written.append(generated[:, :steps].numpy())
    return np.concatenate(written)


def generate_backend(args):
    """Records, generate function and countable codes for other models."""
    frame = load_sequences(args.sequences)
    records = {}
    for split in (args.train_split, args.heldout_split):
        part = frame[frame.split == split]
        records[split] = [
            {"subject_id": subject, "code": events}
            for subject, events in zip(part.subject_id, part.event)
        ]
    model = load_generator(args.model_class, args.checkpoint, args.device, args.dtype)

    ignored = {int(token) for token in args.ignore_tokens.split(",") if token.strip()}
    seen = {int(code) for record in records[args.train_split] for code in record["code"]}
    countable = seen - ignored - {args.pad_token_id}

    def generate(patients, x, y):
        return model_generate(
            model, patients, x, y,
            pad=args.pad_token_id, device=args.device, batch_size=args.batch_size,
        )

    return records, generate, countable, None


# -- Command line ----------------------------------------------------------------------


def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--backend", choices=("bonsai", "generate"), default="bonsai")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path,
                        help="bonsai: folder with subject_data_<split>.pt and vocabulary.pt")
    parser.add_argument("--model-class", help="generate: module.path:ClassName")
    parser.add_argument("--sequences", type=Path, help="generate: the sequence file")
    parser.add_argument("--ignore-tokens", default="",
                        help="generate: special token ids, comma-separated")
    parser.add_argument("--pad-token-id", type=int, default=0)
    parser.add_argument("--dtype", default="auto", help="generate: e.g. bfloat16")
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--heldout-split", default="tuning")
    parser.add_argument("--prompt-events", type=int, nargs="+",
                        default=[16, 64, 256, 1024, 4096])
    parser.add_argument("--continuation-events", type=int, nargs="+",
                        default=[1, 2, 4, 8, 16, 32, 64])
    parser.add_argument("--max-patients", type=int, default=2000,
                        help="per split and prompt length")
    parser.add_argument("--min-patients", type=int, default=MIN_PATIENTS,
                        help="scored patients per group for a cell to enter summary.txt")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--per-patient", action="store_true",
                        help="also write per-patient scores (INTERNAL)")
    args = parser.parse_args()
    if args.backend == "bonsai" and not args.data_dir:
        parser.error("--backend bonsai needs --data-dir")
    if args.backend == "generate" and not (args.model_class and args.sequences):
        parser.error("--backend generate needs --model-class and --sequences")
    args.continuation_events = sorted(args.continuation_events)
    return args


def main():
    args = parse_args()
    started = time.time()
    splits = (args.train_split, args.heldout_split)

    if args.backend == "bonsai":
        records, generate, countable, loss = bonsai_backend(args)
    else:
        records, generate, countable, loss = generate_backend(args)

    rare = find_rare_codes(records[args.train_split]) & countable
    scores = score_all_patients(
        generate, records, args.prompt_events, args.continuation_events,
        args.max_patients, countable, rare,
    )
    if scores.is_empty():
        raise SystemExit("No patient had a record long enough for these lengths")
    means = mean_per_split(scores)
    excess = excess_over_unseen(scores, *splits)
    summary = write_summary(excess, loss, args.min_patients)

    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    means.write_csv(out / "memorization_summary.csv")
    excess.write_csv(out / "memorization_excess.csv")
    if loss is not None:
        loss.write_csv(out / "loss_check.csv")
    plot_curves(means, out / "memorization_curves.png", *splits, loss)
    (out / "summary.txt").write_text(summary + "\n")
    if args.per_patient:
        scores.write_parquet(out / "memorization_per_patient_INTERNAL.parquet")
    run = {
        "backend": args.backend,
        "decoding": "greedy, each generated code appended and fed back",
        "checkpoint": str(args.checkpoint),
        "data": str(args.data_dir if args.backend == "bonsai" else args.sequences),
        "splits": splits,
        "prompt_events": args.prompt_events,
        "continuation_events": args.continuation_events,
        "max_patients": args.max_patients,
        "min_patients_in_summary": args.min_patients,
        "on_time_window_days": ON_TIME_DAYS,
        "rare_code_share": RARE_CODE_SHARE,
        "rare_codes": len(rare),
        "seconds": round(time.time() - started, 1),
    }
    (out / "run.json").write_text(json.dumps(run, indent=2))
    print(summary)


if __name__ == "__main__":
    main()
