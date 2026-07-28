"""Extract the last-clinical-token alternative to cls_last.

``_pool_cls_last`` (``opera/run/extract_outcome_transfer_embeddings.py:482-494``)
reads the hidden state at the appended ``[CLS]``/predict-token position.
AUDIT_FINDINGS.md Part A established that this position never occurs in any
pretraining sequence -- it is appended only by ``FinetuneDataset``/extraction
code. The token immediately before it -- the patient's last real clinical
event before the index date -- has, in this causal model, attended over the
entire preceding window and did receive gradient under the pretraining
next-token objective.

This is new, additive code: it does not modify
``extract_outcome_transfer_embeddings.py``, ``extract_patient_embeddings.py``,
pretraining code, or any config. It reuses the same index-table, censoring,
and dynamic-padding conventions as those scripts, so the resulting NPZ is
directly comparable, patient-for-patient, to the existing cls_last NPZ.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from opera.compat.bonsai import FinetuneDataset, dynamic_padding, encoder_hidden_state
from opera.evaluation.outcome_transfer_evaluation import file_hash
from opera.modules.datamodules.OutcomeFinetuneDataModule import load_subject_pool
from opera.run.extract_outcome_transfer_embeddings import (
    OutcomeTransferExtractionError,
    _reference_records,
    _resolve_device,
    load_frozen_encoder,
)
from opera.run.extract_patient_embeddings import (
    _subject_paths,
    prepare_prediction_origins,
    read_index_table,
)

SPLIT_ORDER = ("train", "tuning", "held_out")


def assert_cls_is_last_token(
    batch: Mapping[str, torch.Tensor], cls_token_id: int
) -> None:
    """Verify the appended [CLS]/predict token sits at ``attention_mask.sum()-1``.

    This is the load-bearing assumption behind last-clinical-token pooling:
    if [CLS] is not the last real token for some row, ``lengths - 1`` is not
    "the last clinical event" for that row. Raises rather than silently
    extracting a wrong position.
    """
    lengths = batch["attention_mask"].sum(dim=1) - 1
    if torch.any(lengths < 0):
        raise OutcomeTransferExtractionError(
            "Encountered an empty sequence while verifying the [CLS] position."
        )
    batch_arange = torch.arange(batch["code"].shape[0], device=batch["code"].device)
    observed = batch["code"][batch_arange, lengths]
    mismatched = observed != cls_token_id
    if torch.any(mismatched):
        bad_rows = mismatched.nonzero(as_tuple=True)[0].tolist()
        bad_subjects = (
            batch["subject_id"][mismatched].detach().cpu().tolist()
            if "subject_id" in batch
            else bad_rows
        )
        raise OutcomeTransferExtractionError(
            f"[CLS] is not at the last real-token position for {len(bad_rows)} "
            f"row(s) in this batch; subject_ids={bad_subjects}. The "
            "lengths-1 assumption behind last-clinical-token pooling does not "
            "hold for these rows -- stopping rather than extracting the wrong "
            "position."
        )


def _pool_last_clinical(
    encoder: torch.nn.Module,
    batch: Mapping[str, torch.Tensor],
    *,
    cls_token_id: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pool the hidden state immediately before the appended [CLS] token.

    Mirrors ``extract_outcome_transfer_embeddings._pool_cls_last`` exactly,
    except it reads ``clamp(lengths - 1, min=0)`` instead of ``lengths``.
    Returns ``(pooled, clamped_mask)``; ``clamped_mask`` marks rows where the
    clamp actually changed the index, i.e. rows where [CLS] is the *sole*
    token in the sequence (``lengths == 0``), for whom this falls back to the
    [CLS] position itself.

    Note this is a narrower condition than "no clinical event before index":
    every real patient carries background tokens (DATA_FORMAT.md), so a
    patient with background tokens but zero clinical events lands on their
    last background token (index >= 0) and does NOT trip this clamp. This
    flag only catches the more degenerate case of no token at all -- which
    should not occur on real data -- not "clinical-content-free but has
    background tokens."
    """
    assert_cls_is_last_token(batch, cls_token_id)
    hidden = encoder_hidden_state(encoder(batch))
    lengths = batch["attention_mask"].sum(dim=1) - 1
    if torch.any(lengths < 0):
        raise OutcomeTransferExtractionError(
            "Encountered an empty sequence during extraction."
        )
    clamped_mask = (lengths - 1) < 0
    clamped_lengths = (lengths - 1).clamp(min=0)
    pooled = hidden[
        torch.arange(hidden.size(0), device=hidden.device),
        clamped_lengths,
    ]
    return pooled, clamped_mask


def extract_last_clinical_embeddings(
    encoder: torch.nn.Module,
    *,
    reference: pd.DataFrame,
    subject_split_paths: Mapping[str, str | Path],
    vocabulary: Mapping[str, int],
    split_keys: Mapping[str, str],
    max_len: int,
    batch_size: int,
    num_workers: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[int], dict[str, int]]:
    """Pool physical SSL shards, selecting last-clinical-token embeddings.

    Mirrors ``extract_shared_split_embeddings``'s subject selection, split
    order, and duplicate/coverage checks exactly, so the returned
    ``subject_ids``/``splits`` ordering matches the existing cls_last
    extraction for the same inputs.
    """
    if "[CLS]" not in vocabulary:
        raise OutcomeTransferExtractionError(
            "Vocabulary must contain '[CLS]' for prediction-origin extraction."
        )
    cls_token_id = int(vocabulary["[CLS]"])

    physical_paths = list(
        dict.fromkeys(Path(path) for path in subject_split_paths.values())
    )
    for source in physical_paths:
        if not source.exists():
            raise FileNotFoundError(
                f"Physical subject-data shard does not exist: {source}"
            )
    try:
        subject_pool = load_subject_pool([str(path) for path in physical_paths])
    except ValueError as exc:
        raise OutcomeTransferExtractionError(str(exc)) from exc

    subject_ids: list[np.ndarray] = []
    embeddings: list[np.ndarray] = []
    splits_out: list[np.ndarray] = []
    clamped_subject_ids: list[int] = []
    split_counts: dict[str, int] = {}

    for split in SPLIT_ORDER:
        records = _reference_records(reference, str(split_keys[split]))
        if not records:
            raise OutcomeTransferExtractionError(
                f"Reference outcome has no patients for split {split!r}."
            )
        selected = [
            subject for subject in subject_pool if int(subject["subject_id"]) in records
        ]
        selected_ids = {int(subject["subject_id"]) for subject in selected}
        missing = sorted(set(records) - selected_ids)
        duplicate = len(selected_ids) != len(selected)
        if missing or duplicate:
            raise OutcomeTransferExtractionError(
                f"Pooled physical shards do not exactly cover temporal split "
                f"{split!r}; missing={len(missing)} examples={missing[:10]}, "
                f"duplicates={duplicate}."
            )
        background_length = int((selected[0]["segment"] == 0).sum())
        dataset = FinetuneDataset(
            selected,
            outcomes=records,
            predict_token_id=cls_token_id,
            background_length=background_length,
            max_len=int(max_len),
        )
        loader = DataLoader(
            dataset,
            batch_size=int(batch_size),
            num_workers=int(num_workers),
            shuffle=False,
            drop_last=False,
            collate_fn=dynamic_padding,
        )
        split_ids: list[np.ndarray] = []
        split_embeddings: list[np.ndarray] = []
        with torch.inference_mode():
            for batch in loader:
                device_batch = {
                    key: value.to(device) if isinstance(value, torch.Tensor) else value
                    for key, value in batch.items()
                }
                pooled, clamped_mask = _pool_last_clinical(
                    encoder, device_batch, cls_token_id=cls_token_id
                )
                if clamped_mask.any():
                    clamped_subject_ids.extend(
                        device_batch["subject_id"][clamped_mask].detach().cpu().tolist()
                    )
                split_ids.append(device_batch["subject_id"].detach().cpu().numpy())
                split_embeddings.append(
                    pooled.detach().cpu().numpy().astype(np.float32)
                )
        if not split_ids:
            raise OutcomeTransferExtractionError(
                f"No batches were emitted for split {split!r}."
            )
        split_subject_ids = np.concatenate(split_ids)
        if len(np.unique(split_subject_ids)) != len(split_subject_ids):
            raise OutcomeTransferExtractionError(
                f"Duplicate subject IDs were emitted while extracting split {split!r}."
            )
        subject_ids.append(split_subject_ids)
        splits_out.append(np.full(len(split_subject_ids), str(split_keys[split])))
        embeddings.append(np.concatenate(split_embeddings))
        split_counts[split] = int(len(split_subject_ids))

    all_ids = np.concatenate(subject_ids)
    all_splits = np.concatenate(splits_out)
    all_embeddings = np.concatenate(embeddings)
    if len(np.unique(all_ids)) != len(all_ids):
        raise OutcomeTransferExtractionError(
            "A subject appeared in multiple shared subject splits; extraction is unsafe."
        )
    return all_ids, all_splits, all_embeddings, clamped_subject_ids, split_counts


def assert_matches_reference_npz(
    subject_ids: np.ndarray, reference_npz: str | Path
) -> None:
    """Fail loudly unless ``subject_ids`` exactly matches the reference NPZ's,
    in both content and order."""
    with np.load(reference_npz) as artifact:
        if "subject_ids" not in artifact.files:
            raise OutcomeTransferExtractionError(
                f"{reference_npz} has no subject_ids array to compare against."
            )
        reference_ids = artifact["subject_ids"]
    if reference_ids.shape != subject_ids.shape or not np.array_equal(
        reference_ids, subject_ids
    ):
        raise OutcomeTransferExtractionError(
            f"last_clinical subject_ids do not exactly match {reference_npz}'s "
            "subject_ids/order; refusing to write a misaligned NPZ."
        )


def spectral_summary(path: str | Path) -> tuple[float, float, int]:
    """Participation ratio, effective rank, and n-components-for-90%-variance."""
    with np.load(path) as artifact:
        embeddings = artifact["embeddings"].astype(np.float64)
    centered = embeddings - embeddings.mean(0)
    singular_values_sq = np.linalg.svd(centered, compute_uv=False) ** 2
    variance_fraction = singular_values_sq / singular_values_sq.sum()
    participation_ratio = float(
        singular_values_sq.sum() ** 2 / (singular_values_sq**2).sum()
    )
    effective_rank = float(
        np.exp(-(variance_fraction * np.log(variance_fraction + 1e-12)).sum())
    )
    n90 = int(np.searchsorted(np.cumsum(variance_fraction), 0.90) + 1)
    return participation_ratio, effective_rank, n90


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Extract last-clinical-token frozen patient embeddings (the token "
            "immediately before the appended [CLS]), as an alternative to the "
            "existing cls_last extraction."
        )
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--checkpoint-metadata", default=None)
    parser.add_argument("--vocabulary", required=True)
    parser.add_argument("--subject-data-dir", required=True)
    parser.add_argument("--index-table", required=True)
    parser.add_argument("--subject-col", default="subject_id")
    parser.add_argument("--index-date-col", default="index_date")
    parser.add_argument("--split-col", default="split")
    parser.add_argument("--train-key", default="train")
    parser.add_argument("--tuning-key", default="tuning")
    parser.add_argument("--held-out-key", default="held_out")
    parser.add_argument("--tuning-start-date", default="2022-01-01")
    parser.add_argument("--held-out-start-date", default="2023-01-01")
    parser.add_argument(
        "--expected-training-stage",
        default=None,
        help="Fail unless checkpoint metadata records this training stage.",
    )
    parser.add_argument("--max-len", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--attention-backend",
        choices=["auto", "checkpoint", "sdpa", "flash"],
        default="auto",
    )
    parser.add_argument(
        "--reference-npz",
        required=True,
        help=(
            "Existing daly_care_pretrain.npz (cls_last) to assert an "
            "identical subject_id/order match against before writing, and "
            "to compare spectral statistics against."
        ),
    )
    parser.add_argument(
        "--output",
        default=None,
        help=(
            "Output NPZ path. Defaults to "
            "$EMB_DIR/daly_care_pretrain__last_clinical.npz."
        ),
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.output is not None:
        output = Path(args.output)
    else:
        emb_dir = os.environ.get("EMB_DIR")
        if not emb_dir:
            raise OutcomeTransferExtractionError(
                "Neither --output nor $EMB_DIR is set; cannot resolve an output path."
            )
        output = Path(emb_dir) / "daly_care_pretrain__last_clinical.npz"

    device = _resolve_device(args.device)
    encoder, vocabulary, checkpoint_metadata = load_frozen_encoder(
        args.checkpoint,
        vocabulary_path=args.vocabulary,
        device=device,
        attention_backend=args.attention_backend,
        checkpoint_metadata_path=args.checkpoint_metadata,
    )
    recorded_stage = checkpoint_metadata.get("training_stage")
    if (
        args.expected_training_stage is not None
        and recorded_stage != args.expected_training_stage
    ):
        raise OutcomeTransferExtractionError(
            "Checkpoint training stage does not match the requested stage: "
            f"recorded={recorded_stage!r}, expected={args.expected_training_stage!r}."
        )
    raw_index_table = read_index_table(args.index_table)
    reference, split_keys = prepare_prediction_origins(
        raw_index_table,
        subject_col=args.subject_col,
        index_date_col=args.index_date_col,
        split_col=args.split_col,
        split_keys={
            "train": args.train_key,
            "tuning": args.tuning_key,
            "held_out": args.held_out_key,
        },
        tuning_start_date=args.tuning_start_date,
        held_out_start_date=args.held_out_start_date,
    )
    max_len = int(args.max_len or encoder.hparams["max_seqlen"])
    if max_len <= 0 or args.batch_size <= 0 or args.num_workers < 0:
        raise OutcomeTransferExtractionError(
            "max-len and batch-size must be positive; num-workers non-negative."
        )
    paths = _subject_paths(args.subject_data_dir)

    start_time = time.perf_counter()
    subject_ids, splits, embeddings, clamped_subject_ids, split_counts = (
        extract_last_clinical_embeddings(
            encoder,
            reference=reference,
            subject_split_paths=paths,
            vocabulary=vocabulary,
            split_keys=split_keys,
            max_len=max_len,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            device=device,
        )
    )
    wall_time_seconds = time.perf_counter() - start_time

    print(
        f"Hit the clamp (no token, clinical or background, before [CLS]) for "
        f"{len(clamped_subject_ids)} patient(s)."
    )
    if clamped_subject_ids:
        print(f"Clamped subject_ids: {clamped_subject_ids}")

    assert_matches_reference_npz(subject_ids, args.reference_npz)

    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        subject_ids=subject_ids,
        embeddings=embeddings,
        splits=splits,
    )

    metadata: dict[str, Any] = {
        "checkpoint_metadata": checkpoint_metadata,
        "embedding_extraction": {
            "checkpoint": str(Path(args.checkpoint)),
            "checkpoint_hash": file_hash(args.checkpoint),
            "vocabulary": str(Path(args.vocabulary)),
            "index_table": str(Path(args.index_table)),
            "index_table_hash": file_hash(args.index_table),
            "pooling": "last_clinical",
            "encoder_frozen": True,
            "max_len": max_len,
            "attention_backend": str(encoder.hparams["attn_type"]),
            "split_counts": split_counts,
            "n_subjects": int(len(subject_ids)),
            "embedding_dim": int(embeddings.shape[1]),
            "n_clamped": len(clamped_subject_ids),
            "clamped_subject_ids": clamped_subject_ids,
            "reference_npz_checked": str(Path(args.reference_npz)),
            "wall_time_seconds": wall_time_seconds,
        },
    }
    sidecar = output.with_suffix(".metadata.json")
    sidecar.write_text(json.dumps(metadata, indent=2, default=str) + "\n", encoding="utf-8")

    new_stats = spectral_summary(output)
    reference_stats = spectral_summary(args.reference_npz)

    print(
        f"Wrote {len(subject_ids):,} last_clinical frozen embeddings "
        f"({embeddings.shape[1]} dimensions) to {output}; "
        f"wall_time={wall_time_seconds:.1f}s; sidecar={sidecar}."
    )
    print(
        "Spectral summary (participation_ratio, effective_rank, n90):\n"
        f"  last_clinical ({output}): {new_stats}\n"
        f"  cls_last      ({args.reference_npz}): {reference_stats}"
    )


if __name__ == "__main__":
    main()
