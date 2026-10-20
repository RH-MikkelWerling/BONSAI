#!/usr/bin/env python
"""How much does a pretrained sequence model reproduce its training patients?

Discoverable memorization (Carlini et al., 2022), adapted to health records:
give the model the first X tokens of a patient's real sequence, let it generate
the next Y tokens greedily, and compare them with what the patient's record
actually contains next. A 1-1 match of all Y tokens is the strict memorization
event; per-token agreement and the length of the correct prefix are reported
alongside.

Records are predictable (separators, routine codes, repeated prescriptions),
so some continuations match without any memorization. The control is the same
measurement on patients the model never trained on: memorization is the excess
of the training patients' match rate over the unseen patients' rate.

Two backends.

``--backend bonsai`` (our models): a BONSAI encoder pretrained causally on the
next code, loaded from its Lightning checkpoint, with the pretraining split
files ``subject_data_{train,tuning}.pt``. The model predicts the code at
position i from positions before it, so greedy generation reproduces the next
Y true codes exactly if and only if its top prediction is the true code at
each of those Y positions given the true history. One forward pass over the
real sequence therefore gives the exact match and the correct prefix, exactly
as greedy generation would, for every X and Y. The event times of the
continuation are the patient's own (the model needs a time for every code;
using the true ones makes the check, if anything, generous to memorization).
Token agreement is reported as next-code accuracy given the true history.

    python scripts/memorization_check.py --backend bonsai \
        --checkpoint <pretrain run>/version_0/best.ckpt \
        --data-dir <processed dataset dir with subject_data_train.pt> \
        --train-split train --heldout-split tuning --output-dir memorization

``--backend generate`` (e.g. PHAIR): the model must provide

    model.generate(batch={"event": LongTensor[B, T], "attn_mask": BoolTensor[B, T]},
                   max_new_tokens=Y, temperature=0, pad_token_id=PAD,
                   start_after_n_tokens=X, n_samples_per_person=1)
      -> {"generated": LongTensor[B, <=Y], "generation_lengths": LongTensor[B], ...}

as GenerativeNanoEncoder.generate does, and is loaded with
``Class.load_from_checkpoint`` (then ``prepare_for_inference`` if present).
Sequences: a parquet file with ``subject_id``, ``split`` and ``event`` (list of
token ids, in training order), or a torch file holding a list of such dicts
(``code`` is accepted for ``event``). Use the sequences the model was trained
on (same tokenisation and cutoff); ``--train-split`` names the patients it
trained on, ``--heldout-split`` patients it did not (e.g. the tuning split).

    python scripts/memorization_check.py \
        --model-class phair.models.generative:GenerativeNanoEncoder \
        --checkpoint /path/model.ckpt --sequences /path/sequences.parquet \
        --train-split train --heldout-split tuning \
        --prompt-lengths 32,64,128,256,512 --new-tokens 1,4,8,16,32,64 \
        --patients 2000 --output-dir memorization

Writes ``memorization_summary.csv`` (aggregates only: patients, exact-match
rate with 95% Wilson interval, mean token agreement and correct-prefix length
per split, X and Y), ``memorization_excess.csv`` (training minus unseen, Newcombe
95% interval), ``memorization_curves.png`` and ``run.json``. No patient-level
values are written unless ``--per-patient`` is given (keep those internal).
"""

from __future__ import annotations

import argparse
import importlib
import inspect
import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch


# -- data ----------------------------------------------------------------------

def load_sequences(path: Path) -> pd.DataFrame:
    """``subject_id``, ``split`` and ``event`` (list of ints) per patient."""
    path = Path(path)
    if path.suffix in {".parquet", ".pq"}:
        frame = pd.read_parquet(path)
    else:
        records = torch.load(path, map_location="cpu", weights_only=False)
        if isinstance(records, dict):
            records = list(records.values())
        frame = pd.DataFrame([{**r, "event": r.get("event", r.get("code"))} for r in records])
    if "event" not in frame and "code" in frame:
        frame = frame.rename(columns={"code": "event"})
    missing = {"subject_id", "split", "event"} - set(frame)
    if missing:
        raise ValueError(f"Sequences need columns subject_id, split, event; missing {missing}")
    frame["event"] = frame.event.map(lambda e: [int(t) for t in (e.tolist() if hasattr(e, "tolist") else e)])
    return frame[["subject_id", "split", "event"]]


def sample_patients(frame, split, minimum_length, n, seed):
    """Patients of one split long enough for the largest prompt + continuation."""
    eligible = frame.loc[(frame.split == split) & (frame.event.map(len) >= minimum_length)]
    if n and len(eligible) > n:
        eligible = eligible.sample(n=n, random_state=seed)
    return eligible.reset_index(drop=True)


# -- scoring --------------------------------------------------------------------

def score(generated, truth, ignore=frozenset()):
    """Per patient: exact match of all Y tokens, token agreement, correct prefix.

    ``generated`` and ``truth`` are arrays [n, Y]. ``ignore`` token ids (e.g.
    separators) are left out of the content agreement only; the exact match
    always compares every position.
    """
    generated, truth = np.asarray(generated), np.asarray(truth)
    equal = generated == truth
    exact = equal.all(axis=1)
    first_miss = np.where(~equal, np.arange(equal.shape[1]), equal.shape[1]).min(axis=1)
    content = ~np.isin(truth, list(ignore)) if ignore else np.ones_like(equal)
    with np.errstate(invalid="ignore", divide="ignore"):
        content_agreement = (equal & content).sum(1) / content.sum(1)
    return pd.DataFrame({"exact": exact, "token_agreement": equal.mean(axis=1),
                         "content_agreement": content_agreement,
                         "correct_prefix": first_miss})


def wilson(successes, n, z=1.96):
    if n == 0:
        return (math.nan, math.nan, math.nan)
    p = successes / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return p, max(0.0, centre - half), min(1.0, centre + half)


def summarise(scores: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (split, x, y), part in scores.groupby(["split", "prompt_tokens", "new_tokens"]):
        rate, low, high = wilson(int(part.exact.sum()), len(part))
        rows.append({"split": split, "prompt_tokens": x, "new_tokens": y, "patients": len(part),
                     "exact_match": rate, "exact_match_lower": low, "exact_match_upper": high,
                     "token_agreement": part.token_agreement.mean(),
                     "content_agreement": part.content_agreement.mean(),
                     "correct_prefix_mean": part.correct_prefix.mean()})
    return pd.DataFrame(rows)


def excess(summary: pd.DataFrame, train: str, heldout: str) -> pd.DataFrame:
    """Training minus unseen exact-match rate per X and Y (Newcombe interval).

    Built from the two Wilson intervals, so it stays sensible at rates of 0 or 1.
    """
    keys = ["prompt_tokens", "new_tokens"]
    a = summary.loc[summary.split == train].set_index(keys)
    b = summary.loc[summary.split == heldout].set_index(keys)
    j = a.join(b, lsuffix="_train", rsuffix="_heldout", how="inner")
    p1, l1, u1 = j.exact_match_train, j.exact_match_lower_train, j.exact_match_upper_train
    p2, l2, u2 = j.exact_match_heldout, j.exact_match_lower_heldout, j.exact_match_upper_heldout
    diff = p1 - p2
    return pd.DataFrame({"exact_match_train": p1, "exact_match_heldout": p2, "excess": diff,
                         "excess_lower": diff - np.sqrt((p1 - l1) ** 2 + (u2 - p2) ** 2),
                         "excess_upper": diff + np.sqrt((u1 - p1) ** 2 + (p2 - l2) ** 2),
                         "patients_train": j.patients_train,
                         "patients_heldout": j.patients_heldout}).reset_index()


# -- generation -----------------------------------------------------------------

def continuations(model, sequences, prompt, new, *, pad, device, batch_size):
    """Greedy continuations of ``new`` tokens after the first ``prompt`` tokens.

    Each batch is cut to prompt + new tokens, so the model sees only the
    prompt; ``generate`` writes into a copy.
    """
    out = []
    width = prompt + new
    for start in range(0, len(sequences), batch_size):
        chunk = sequences[start:start + batch_size]
        events = torch.full((len(chunk), width), pad, dtype=torch.long)
        for i, seq in enumerate(chunk):
            events[i, :width] = torch.tensor(seq[:width])
        batch = {"event": events.to(device), "attn_mask": (events != pad).to(device)}
        with torch.inference_mode():
            result = model.generate(batch, max_new_tokens=new, temperature=0,
                                    pad_token_id=pad, start_after_n_tokens=prompt,
                                    n_samples_per_person=1)
        generated = result["generated"].detach().cpu()
        if generated.shape[1] < new:  # generation stopped early: pad (never a match)
            filler = torch.full((generated.shape[0], new - generated.shape[1]), -1)
            generated = torch.cat([generated, filler], dim=1)
        out.append(generated[:, :new].numpy())
    return np.concatenate(out) if out else np.empty((0, new), dtype=int)


def run(model, frame, *, splits, prompts, news, n, pad, device, batch_size, seed=2026,
        ignore=frozenset()):
    """Score every split, prompt length and continuation length."""
    longest = max(news)
    rows = []
    for split in splits:
        for prompt in prompts:
            patients = sample_patients(frame, split, prompt + longest, n, seed)
            if patients.empty:
                continue
            seqs = patients.event.tolist()
            generated = continuations(model, seqs, prompt, longest, pad=pad, device=device,
                                      batch_size=batch_size)
            truth = np.array([s[prompt:prompt + longest] for s in seqs])
            for new in news:  # greedy: the first Y tokens are the Y-token continuation
                scored = score(generated[:, :new], truth[:, :new], ignore)
                rows.append(scored.assign(split=split, prompt_tokens=prompt, new_tokens=new,
                                          subject_id=patients.subject_id.to_numpy()))
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()



# -- BONSAI: causal next-code models, one forward pass ---------------------------

BONSAI_FIELDS = ("code", "age", "abspos", "segment", "value_bin", "value_normalized",
                 "value_present", "numeric_value")


def load_bonsai(checkpoint, device):
    """The pretraining network (encoder and next-code head) from a Lightning checkpoint."""
    import sys

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
    model.load_state_dict(clean_lightning_state_dict(payload["state_dict"]), strict=True)
    return model.eval().to(device)


def load_bonsai_subjects(data_dir, split):
    return torch.load(Path(data_dir) / f"subject_data_{split}.pt", map_location="cpu",
                      weights_only=False)


def bonsai_batch(subjects, width, device):
    """The first ``width`` events of each subject, as the pretraining dataset feeds them."""
    from bonsai.functional.normalization import normalize_segments

    batch = {}
    for key in BONSAI_FIELDS:
        if all(key in subject for subject in subjects):
            rows = [subject[key][:width] for subject in subjects]
            if key == "segment":
                rows = [normalize_segments(row) for row in rows]
            batch[key] = torch.stack(rows).to(device)
    batch["attention_mask"] = torch.ones_like(batch["code"], dtype=torch.bool)
    return batch


def next_code_hits(model, subjects, width, *, device, batch_size, chunk=128):
    """[n, width - 1]: is the top prediction at position i - 1 the code at i?"""
    out = []
    use_amp = str(device).startswith("cuda")
    for start in range(0, len(subjects), batch_size):
        batch = bonsai_batch(subjects[start:start + batch_size], width, device)
        codes = batch["code"]
        hits = torch.zeros(codes.shape[0], width - 1, dtype=torch.bool)
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16,
                                                     enabled=use_amp):
            hidden = model.encode(batch)
            for left in range(0, width - 1, chunk):  # logits are batch x chunk x vocab
                right = min(left + chunk, width - 1)
                predicted = model.pretrain_head(hidden[:, left:right]).argmax(-1)
                hits[:, left:right] = (predicted == codes[:, left + 1:right + 1]).cpu()
        out.append(hits.numpy())
    return np.concatenate(out) if out else np.empty((0, width - 1), dtype=bool)


def score_hits(hits, truth, ignore=frozenset()):
    """Per patient from next-code hits over the Y continuation positions."""
    hits, truth = np.asarray(hits, dtype=bool), np.asarray(truth)
    first_miss = np.where(~hits, np.arange(hits.shape[1]), hits.shape[1]).min(axis=1)
    content = ~np.isin(truth, list(ignore)) if ignore else np.ones_like(hits)
    with np.errstate(invalid="ignore", divide="ignore"):
        content_agreement = (hits & content).sum(1) / content.sum(1)
    return pd.DataFrame({"exact": hits.all(axis=1), "token_agreement": hits.mean(axis=1),
                         "content_agreement": content_agreement,
                         "correct_prefix": first_miss})


def run_bonsai(model, subjects_by_split, *, prompts, news, n, device, batch_size,
               seed=2026, ignore=frozenset()):
    """Score every split, prompt length X and continuation length Y."""
    longest = max(news)
    limit = int(model.hparams.get("max_seqlen", 10**9))
    rows = []
    for split, subjects in subjects_by_split.items():
        for prompt in prompts:
            width = prompt + longest
            if width > limit:
                raise SystemExit(f"X + Y = {width} exceeds the model's {limit} events")
            eligible = [s for s in subjects if len(s["code"]) >= width]
            if not eligible:
                continue
            if n and len(eligible) > n:
                pick = np.random.default_rng(seed).choice(len(eligible), n, replace=False)
                eligible = [eligible[i] for i in sorted(pick)]
            hits = next_code_hits(model, eligible, width, device=device,
                                  batch_size=batch_size)
            truth = np.stack([s["code"][prompt:width].numpy() for s in eligible])
            ids = np.array([s.get("subject_id") for s in eligible])
            for new in news:
                scored = score_hits(hits[:, prompt - 1:prompt - 1 + new], truth[:, :new],
                                    ignore)
                rows.append(scored.assign(split=split, prompt_tokens=prompt,
                                          new_tokens=new, subject_id=ids))
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def plot(summary, path, train, heldout):
    """Exact match by continuation length: colour = prompt length, style = split."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D
    except ImportError:  # pragma: no cover
        return
    fig, ax = plt.subplots(figsize=(6.6, 3.9))
    prompts = sorted(summary.prompt_tokens.unique())
    colors = plt.cm.viridis(np.linspace(0.1, 0.85, len(prompts)))
    for color, prompt in zip(colors, prompts):
        for split, style in ((train, "-"), (heldout, "--")):
            part = summary.loc[(summary.prompt_tokens == prompt)
                               & (summary.split == split)].sort_values("new_tokens")
            if len(part):
                ax.plot(part.new_tokens, part.exact_match, style, color=color, marker="o",
                        markersize=3)
                ax.fill_between(part.new_tokens, part.exact_match_lower,
                                part.exact_match_upper, color=color, alpha=0.08, lw=0)
    ax.set_xscale("log", base=2)
    ax.set_xlabel("Events to reproduce after the prompt (Y)")
    ax.set_ylabel("Exact match with the patient's record")
    ax.set_title("Does the model reproduce the patients it was trained on?", fontsize=9)
    by_prompt = [Line2D([], [], color=c, marker="o", markersize=3, label=f"{p:,}")
                 for c, p in zip(colors, prompts)]
    by_split = [Line2D([], [], color="0.3", linestyle="-", label="Training patients"),
                Line2D([], [], color="0.3", linestyle="--",
                       label="Unseen patients (never trained on)")]
    first = ax.legend(handles=by_prompt, title="Prompt length (X events)", fontsize=6.5,
                      title_fontsize=7, frameon=False, loc="upper right")
    ax.add_artist(first)
    ax.legend(handles=by_split, fontsize=6.5, frameon=False, loc="upper center")
    fig.tight_layout()
    fig.savefig(path, dpi=200)


def load_model(spec, checkpoint, device, dtype):
    module, _, name = spec.partition(":")
    cls = getattr(importlib.import_module(module), name)
    model = cls.load_from_checkpoint(checkpoint, map_location="cpu")
    model.eval()
    model.to(device)
    prepare = getattr(model, "prepare_for_inference", None)
    if prepare is not None:
        accepted = inspect.signature(prepare).parameters
        options = {"device": device, "dtype": None if dtype == "auto" else getattr(torch, dtype)}
        prepare(**{k: v for k, v in options.items() if k in accepted and v is not None})
    return model


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--backend", choices=("bonsai", "generate"), default="generate")
    parser.add_argument("--model-class", help="generate: module.path:ClassName")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--sequences", type=Path, help="generate: sequence file")
    parser.add_argument("--data-dir", type=Path,
                        help="bonsai: processed dataset with subject_data_<split>.pt")
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--heldout-split", default="tuning")
    parser.add_argument("--prompt-lengths", default="32,64,128,256,512")
    parser.add_argument("--new-tokens", default="1,4,8,16,32,64")
    parser.add_argument("--patients", type=int, default=2000, help="Per split and prompt length")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--pad-token-id", type=int, default=0)
    parser.add_argument("--ignore-tokens", default="",
                        help="Token ids left out of content agreement (e.g. separators)")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", default="auto", help="e.g. bfloat16; auto leaves the model's own")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--per-patient", action="store_true",
                        help="Also write per-patient scores (INTERNAL; do not export)")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    prompts = [int(v) for v in args.prompt_lengths.split(",")]
    news = sorted(int(v) for v in args.new_tokens.split(","))
    ignore = frozenset(int(v) for v in args.ignore_tokens.split(",") if v.strip())
    started = time.time()
    splits = [args.train_split, args.heldout_split]
    if args.backend == "bonsai":
        if not args.data_dir:
            parser.error("--backend bonsai needs --data-dir")
        model = load_bonsai(args.checkpoint, args.device)
        subjects = {split: load_bonsai_subjects(args.data_dir, split) for split in splits}
        scores = run_bonsai(model, subjects, prompts=prompts, news=news, n=args.patients,
                            device=args.device, batch_size=args.batch_size, seed=args.seed,
                            ignore=ignore)
    else:
        if not (args.model_class and args.sequences):
            parser.error("--backend generate needs --model-class and --sequences")
        frame = load_sequences(args.sequences)
        model = load_model(args.model_class, args.checkpoint, args.device, args.dtype)
        scores = run(model, frame, splits=splits, prompts=prompts, news=news,
                     n=args.patients, pad=args.pad_token_id, device=args.device,
                     batch_size=args.batch_size, seed=args.seed, ignore=ignore)
    if scores.empty:
        raise SystemExit("No patient had a sequence long enough for these lengths")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary = summarise(scores)
    summary.to_csv(args.output_dir / "memorization_summary.csv", index=False)
    excess(summary, args.train_split, args.heldout_split).to_csv(
        args.output_dir / "memorization_excess.csv", index=False)
    plot(summary, args.output_dir / "memorization_curves.png", args.train_split,
         args.heldout_split)
    if args.per_patient:
        scores.to_parquet(args.output_dir / "memorization_per_patient_INTERNAL.parquet")
    (args.output_dir / "run.json").write_text(json.dumps({
        "backend": args.backend,
        "method": ("one teacher-forced pass; exact match and correct prefix equal greedy "
                   "generation with the patient's own event times"
                   if args.backend == "bonsai" else "greedy generation"),
        "checkpoint": str(args.checkpoint),
        "data": str(args.data_dir if args.backend == "bonsai" else args.sequences),
        "train_split": args.train_split, "heldout_split": args.heldout_split,
        "prompt_lengths": prompts, "new_tokens": news, "patients_per_cell": args.patients,
        "decoding": "greedy", "ignore_tokens": sorted(ignore),
        "seconds": round(time.time() - started, 1)}, indent=2))
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
