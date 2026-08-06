"""Extract prediction-origin patient embeddings from a native BONSAI checkpoint.

Unlike ``extract_outcome_transfer_embeddings``, this entry point is not tied to
the focused DAPT/OPERA transfer manifest. It reuses the same strict checkpoint
loader and first-line censoring machinery, but takes an explicit index table
containing one prediction origin and prospective split per patient.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from opera.evaluation.outcome_transfer_evaluation import file_hash
from opera.functional.outcomes import attach_prediction_censor_abspos
from opera.run.extract_outcome_transfer_embeddings import (
    OutcomeTransferExtractionError,
    _resolve_device,
    extract_shared_split_embeddings,
    load_frozen_encoder,
)


def read_index_table(path: str | Path) -> pd.DataFrame:
    """Read CSV or Parquet prediction origins."""
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(f"Index table does not exist: {source}")
    if source.suffix.lower() in {".parquet", ".pq"}:
        return pd.read_parquet(source)
    if source.suffix.lower() in {".csv", ".txt"}:
        return pd.read_csv(source)
    raise OutcomeTransferExtractionError(
        "Index table must be CSV or Parquet."
    )


def prepare_prediction_origins(
    frame: pd.DataFrame,
    *,
    subject_col: str = "subject_id",
    index_date_col: str = "index_date",
    split_col: str = "split",
    split_keys: Mapping[str, str] | None = None,
    tuning_start_date: str = "2022-01-01",
    held_out_start_date: str = "2023-01-01",
) -> tuple[pd.DataFrame, dict[str, str]]:
    """Validate and standardize one index date and split per patient."""
    required = {subject_col, index_date_col}
    missing = required - set(frame.columns)
    if missing:
        raise OutcomeTransferExtractionError(
            f"Index table is missing required columns: {sorted(missing)}."
        )
    keys = dict(
        split_keys
        or {"train": "train", "tuning": "tuning", "held_out": "held_out"}
    )
    if set(keys) != {"train", "tuning", "held_out"}:
        raise OutcomeTransferExtractionError(
            "split_keys must define train, tuning, and held_out."
        )
    if len(set(keys.values())) != 3:
        raise OutcomeTransferExtractionError(
            f"Split labels must be distinct: {keys}."
        )

    columns = [subject_col, index_date_col]
    if split_col in frame.columns:
        columns.insert(1, split_col)
    reference = frame[columns].copy()
    reference = reference.rename(
        columns={
            subject_col: "subject_id",
            split_col: "split",
            index_date_col: "index_date",
        }
    )
    if reference["subject_id"].isna().any():
        raise OutcomeTransferExtractionError("Index table contains missing subject IDs.")
    if reference["subject_id"].duplicated().any():
        duplicates = (
            reference.loc[reference["subject_id"].duplicated(), "subject_id"]
            .head(10)
            .tolist()
        )
        raise OutcomeTransferExtractionError(
            "Index table must contain one row per patient; "
            f"duplicate examples={duplicates}."
        )
    reference["index_date"] = pd.to_datetime(
        reference["index_date"], errors="coerce"
    )
    if reference["index_date"].isna().any():
        raise OutcomeTransferExtractionError(
            "Index table contains missing or invalid index dates."
        )
    if "split" not in reference:
        tuning_start = pd.Timestamp(tuning_start_date)
        held_out_start = pd.Timestamp(held_out_start_date)
        if tuning_start >= held_out_start:
            raise OutcomeTransferExtractionError(
                "tuning_start_date must be earlier than held_out_start_date."
            )
        reference["split"] = keys["train"]
        reference.loc[
            reference["index_date"] >= tuning_start, "split"
        ] = keys["tuning"]
        reference.loc[
            reference["index_date"] >= held_out_start, "split"
        ] = keys["held_out"]
    observed = set(reference["split"].astype(str))
    expected = set(keys.values())
    unknown = sorted(observed - expected)
    absent = sorted(expected - observed)
    if unknown or absent:
        raise OutcomeTransferExtractionError(
            "Index table split labels do not match the configured prospective "
            f"splits; unknown={unknown}, absent={absent}, expected={sorted(expected)}."
        )
    reference["split"] = reference["split"].astype(str)
    reference = attach_prediction_censor_abspos(reference)
    return reference, keys


def _subject_paths(subject_data_dir: str | Path) -> dict[str, Path]:
    root = Path(subject_data_dir)
    paths = {
        "ssl_train": root / "subject_data_train.pt",
        "ssl_validation": root / "subject_data_tuning.pt",
    }
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Required physical subject-data shards are missing: " + ", ".join(missing)
        )
    return paths


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Extract frozen [CLS] patient embeddings at explicit prediction origins."
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
        "--pooling",
        choices=("cls_last", "mean_last_128"),
        default="cls_last",
        help="Pooling used by the checkpoint's downstream objective.",
    )
    parser.add_argument(
        "--attention-backend",
        choices=["auto", "checkpoint", "sdpa", "flash"],
        default="auto",
    )
    parser.add_argument("--output", required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
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
    split_was_derived = args.split_col not in raw_index_table.columns
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
    subject_ids, embeddings, sequence_lengths, split_counts = extract_shared_split_embeddings(
        encoder,
        reference=reference,
        subject_split_paths=paths,
        vocabulary=vocabulary,
        split_keys=split_keys,
        max_len=max_len,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=device,
        pooling=args.pooling,
    )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    split_by_subject = reference.set_index("subject_id")["split"]
    extracted_splits = np.asarray(
        [split_by_subject.loc[subject_id] for subject_id in subject_ids]
    )
    np.savez_compressed(
        output,
        subject_ids=subject_ids,
        embeddings=embeddings,
        splits=extracted_splits,
        sequence_length=sequence_lengths,
    )
    metadata: dict[str, Any] = {
        "checkpoint_metadata": checkpoint_metadata,
        "embedding_extraction": {
            "checkpoint": str(Path(args.checkpoint)),
            "checkpoint_hash": file_hash(args.checkpoint),
            "vocabulary": str(Path(args.vocabulary)),
            "index_table": str(Path(args.index_table)),
            "index_table_hash": file_hash(args.index_table),
            "pooling": args.pooling,
            "encoder_frozen": True,
            "max_len": max_len,
            "attention_backend": str(encoder.hparams["attn_type"]),
            "subject_data_paths": {
                name: str(path) for name, path in paths.items()
            },
            "split_keys": split_keys,
            "split_strategy": (
                {
                    "type": "derived_from_index_date",
                    "tuning_start_date": args.tuning_start_date,
                    "held_out_start_date": args.held_out_start_date,
                }
                if split_was_derived
                else {"type": "provided_column", "column": args.split_col}
            ),
            "split_counts": split_counts,
            "n_subjects": int(len(subject_ids)),
            "embedding_dim": int(embeddings.shape[1]),
        },
    }
    sidecar = output.with_suffix(".metadata.json")
    sidecar.write_text(
        json.dumps(metadata, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    print(
        f"Wrote {len(subject_ids):,} frozen patient embeddings "
        f"({embeddings.shape[1]} dimensions) to {output}; sidecar={sidecar}."
    )


if __name__ == "__main__":
    main()
