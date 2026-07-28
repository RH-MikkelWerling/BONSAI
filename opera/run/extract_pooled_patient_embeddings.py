"""Multi-layer, multi-pooling frozen patient embeddings (AUDIT_FINDINGS.md Part B).

Runs ONE forward pass per batch with all hidden states captured, then
computes a grid of (layer, pooling) variants from that single pass. This is
a new, additive extraction path: it does not modify
``opera/run/extract_patient_embeddings.py`` or
``opera/run/extract_outcome_transfer_embeddings.py``, and reuses the same
index-table parsing, censoring, prediction-token append, and dynamic-padding
conventions as those scripts so every variant is directly comparable,
patient-for-patient, to the existing cls_last NPZ.

Layers: final, and the layers nearest 3/4, 1/2, and 1/4 depth
(``opera.functional.pooled_extraction.resolve_target_layers``).
Poolings: cls (reference), mean, last, mean_last_128, max — all mask-aware
and, except ``cls``, computed only over tokens other than the appended
prediction/CLS position (see ``pooled_extraction`` module docstring for why).
``attn_cls`` is not implemented: this architecture's attention
(``bonsai/modules/networks/components/mha.py``) uses
``torch.nn.functional.scaled_dot_product_attention``, a fused kernel that
does not expose attention probabilities, so retrieving them cheaply is not
possible without reimplementing the attention math by hand.

Output schema matches the existing cls_last NPZ exactly:
``subject_ids``, ``embeddings``, ``splits``. One NPZ per (layer, pooling),
named ``{output-prefix}__L{layer}__{pooling}.npz``.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from opera.compat.bonsai import FinetuneDataset, dynamic_padding
from opera.evaluation.outcome_transfer_evaluation import file_hash
from opera.functional.pooled_extraction import (
    POOLING_NAMES,
    pool_variants,
    predict_token_mask,
    resolve_target_layers,
)
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


def extract_pooled_variants(
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
    last_k: int = 128,
) -> tuple[np.ndarray, np.ndarray, dict[tuple[str, str], np.ndarray], dict[str, Any]]:
    """Pool physical SSL shards once, computing every (layer, pooling) variant.

    Mirrors ``extract_outcome_transfer_embeddings.extract_shared_split_embeddings``
    exactly in its subject selection, split order, and duplicate/coverage
    checks, so the returned ``subject_ids``/``splits`` ordering matches the
    existing cls_last extraction for the same inputs.
    """
    if "[CLS]" not in vocabulary:
        raise OutcomeTransferExtractionError(
            "Vocabulary must contain '[CLS]' for prediction-origin extraction."
        )
    predict_token_id = int(vocabulary["[CLS]"])
    num_layers = int(encoder.hparams["num_layers"])
    target_layers = resolve_target_layers(num_layers)
    layer_index_to_labels: dict[int, list[str]] = {}
    for label, idx in target_layers.items():
        layer_index_to_labels.setdefault(idx, []).append(label)

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

    variant_keys = [
        (label, pooling) for label in target_layers for pooling in POOLING_NAMES
    ]
    subject_ids_per_split: list[np.ndarray] = []
    splits_per_split: list[np.ndarray] = []
    embeddings: dict[tuple[str, str], list[np.ndarray]] = {
        key: [] for key in variant_keys
    }
    split_counts: dict[str, int] = {}

    peak_bytes = 0
    start_time = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

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
            predict_token_id=predict_token_id,
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
        split_variant_embeddings: dict[tuple[str, str], list[np.ndarray]] = {
            key: [] for key in variant_keys
        }
        with torch.inference_mode():
            for batch in loader:
                device_batch = {
                    key: value.to(device) if isinstance(value, torch.Tensor) else value
                    for key, value in batch.items()
                }
                final_hidden, layer_hidden_states = encoder.encode(
                    device_batch, output_hidden_states=True
                )
                hidden_by_index: dict[int, torch.Tensor] = {num_layers: final_hidden}
                for idx in layer_index_to_labels:
                    if idx != num_layers:
                        hidden_by_index[idx] = layer_hidden_states[idx - 1]
                hidden_by_label = {
                    label: hidden_by_index[idx] for label, idx in target_layers.items()
                }
                predict_mask = predict_token_mask(device_batch["code"], predict_token_id)
                variants = pool_variants(
                    hidden_by_label,
                    attention_mask=device_batch["attention_mask"],
                    predict_mask=predict_mask,
                    last_k=last_k,
                )
                split_ids.append(device_batch["subject_id"].detach().cpu().numpy())
                for key, tensor in variants.items():
                    split_variant_embeddings[key].append(
                        tensor.detach().cpu().numpy().astype(np.float32)
                    )
                if device.type == "cuda":
                    peak_bytes = max(peak_bytes, torch.cuda.max_memory_allocated(device))
        if not split_ids:
            raise OutcomeTransferExtractionError(
                f"No batches were emitted for split {split!r}."
            )
        split_subject_ids = np.concatenate(split_ids)
        if len(np.unique(split_subject_ids)) != len(split_subject_ids):
            raise OutcomeTransferExtractionError(
                f"Duplicate subject IDs were emitted while extracting split {split!r}."
            )
        subject_ids_per_split.append(split_subject_ids)
        splits_per_split.append(
            np.full(len(split_subject_ids), str(split_keys[split]))
        )
        for key in variant_keys:
            embeddings[key].append(np.concatenate(split_variant_embeddings[key]))
        split_counts[split] = int(len(split_subject_ids))

    all_ids = np.concatenate(subject_ids_per_split)
    all_splits = np.concatenate(splits_per_split)
    if len(np.unique(all_ids)) != len(all_ids):
        raise OutcomeTransferExtractionError(
            "A subject appeared in multiple shared subject splits; extraction is unsafe."
        )
    all_embeddings = {key: np.concatenate(value) for key, value in embeddings.items()}

    stats = {
        "wall_time_seconds": time.perf_counter() - start_time,
        "peak_gpu_memory_bytes": int(peak_bytes),
        "device": str(device),
        "split_counts": split_counts,
        "resolved_layers": target_layers,
        "layer_index_aliases": {
            str(idx): labels
            for idx, labels in layer_index_to_labels.items()
            if len(labels) > 1
        },
        "attn_cls_skipped_reason": (
            "SDPA fused kernel (torch.nn.functional.scaled_dot_product_attention) "
            "does not expose attention probabilities; not cheaply retrievable "
            "without reimplementing the attention math."
        ),
    }
    return all_ids, all_splits, all_embeddings, stats


def _assert_matches_reference_cls_npz(
    subject_ids: np.ndarray, reference_cls_npz: str | Path
) -> None:
    with np.load(reference_cls_npz) as artifact:
        if "subject_ids" not in artifact.files:
            raise OutcomeTransferExtractionError(
                f"{reference_cls_npz} has no subject_ids array to compare against."
            )
        reference_ids = artifact["subject_ids"]
    if reference_ids.shape != subject_ids.shape or not np.array_equal(
        reference_ids, subject_ids
    ):
        raise OutcomeTransferExtractionError(
            "Pooled-variant subject_ids do not exactly match "
            f"{reference_cls_npz}'s subject_ids/order; comparisons across "
            "variants would not be patient-for-patient apples-to-apples."
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Extract multi-layer, multi-pooling frozen patient embeddings from "
            "one forward pass per batch, to screen pooling alternatives to the "
            "existing cls_last extraction without retraining."
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
    parser.add_argument("--last-k", type=int, default=128)
    parser.add_argument(
        "--reference-cls-npz",
        default=None,
        help=(
            "Existing cls_last NPZ (e.g. from extract_patient_embeddings.py) "
            "to assert an identical subject_id/order match against before "
            "writing any output."
        ),
    )
    parser.add_argument("--output-prefix", default="daly_care_pretrain")
    parser.add_argument("--output-dir", required=True)
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
    if args.last_k <= 0:
        raise OutcomeTransferExtractionError("--last-k must be positive.")
    paths = _subject_paths(args.subject_data_dir)

    subject_ids, splits, embeddings, stats = extract_pooled_variants(
        encoder,
        reference=reference,
        subject_split_paths=paths,
        vocabulary=vocabulary,
        split_keys=split_keys,
        max_len=max_len,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=device,
        last_k=args.last_k,
    )

    if args.reference_cls_npz is not None:
        _assert_matches_reference_cls_npz(subject_ids, args.reference_cls_npz)
        stats["reference_cls_npz_checked"] = str(Path(args.reference_cls_npz))
    else:
        print(
            "WARNING: no --reference-cls-npz supplied; subject_id/order match "
            "against the existing cls_last NPZ was not asserted."
        )

    target_layers = stats["resolved_layers"]
    layer_index_to_labels: dict[int, list[str]] = {}
    for label, idx in target_layers.items():
        layer_index_to_labels.setdefault(idx, []).append(label)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    written: list[str] = []
    for layer_idx, labels in sorted(layer_index_to_labels.items()):
        representative_label = labels[0]
        for pooling in POOLING_NAMES:
            key = (representative_label, pooling)
            variant_embeddings = embeddings[key]
            if len(np.unique(subject_ids)) != len(subject_ids) or len(
                subject_ids
            ) != len(variant_embeddings):
                raise OutcomeTransferExtractionError(
                    f"Variant {key} has an embeddings/subject_ids length mismatch; "
                    "refusing to write a misaligned NPZ."
                )
            name = f"{args.output_prefix}__L{layer_idx}__{pooling}.npz"
            output_path = output_dir / name
            np.savez_compressed(
                output_path,
                subject_ids=subject_ids,
                embeddings=variant_embeddings,
                splits=splits,
            )
            written.append(str(output_path))

    metadata: dict[str, Any] = {
        "checkpoint_metadata": checkpoint_metadata,
        "pooled_extraction": {
            "checkpoint": str(Path(args.checkpoint)),
            "checkpoint_hash": file_hash(args.checkpoint),
            "vocabulary": str(Path(args.vocabulary)),
            "index_table": str(Path(args.index_table)),
            "index_table_hash": file_hash(args.index_table),
            "encoder_frozen": True,
            "max_len": max_len,
            "attention_backend": str(encoder.hparams["attn_type"]),
            "num_layers": int(encoder.hparams["num_layers"]),
            "resolved_layers": target_layers,
            "layer_index_aliases": stats["layer_index_aliases"],
            "poolings": list(POOLING_NAMES),
            "attn_cls_skipped_reason": stats["attn_cls_skipped_reason"],
            "last_k": args.last_k,
            "split_counts": stats["split_counts"],
            "n_subjects": int(len(subject_ids)),
            "wall_time_seconds": stats["wall_time_seconds"],
            "peak_gpu_memory_bytes": stats["peak_gpu_memory_bytes"],
            "device": stats["device"],
            "reference_cls_npz_checked": stats.get("reference_cls_npz_checked"),
            "outputs": written,
        },
    }
    sidecar = output_dir / f"{args.output_prefix}__pooled_extraction.metadata.json"
    sidecar.write_text(json.dumps(metadata, indent=2, default=str) + "\n", encoding="utf-8")
    print(
        f"Wrote {len(written)} pooled-variant NPZ files "
        f"({len(subject_ids):,} subjects each) to {output_dir}; "
        f"wall_time={stats['wall_time_seconds']:.1f}s, "
        f"peak_gpu_memory={stats['peak_gpu_memory_bytes'] / 1e9:.2f}GB; "
        f"sidecar={sidecar}."
    )


if __name__ == "__main__":
    main()
