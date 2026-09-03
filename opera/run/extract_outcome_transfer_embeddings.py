"""Extract target-safe frozen embeddings for the focused OPERA transfer study.

The transfer evaluator consumes a single frozen embedding table per
``representation × seed``.  This command creates that table directly from a
DAPT or OPERA contrastive checkpoint and the shared subject split files.  It
uses the same pre-index censoring, appended prediction token, dynamic padding,
and ``cls_last`` pooling convention as downstream BONSAI finetuning.

No target outcome labels, outcome dates, or test performance are used to fit
or tune the encoder here.  Outcome files are read only to obtain and verify a
common prediction origin (``index_date``) across the transfer target union.
If target endpoints do not share that origin, the command fails rather than
silently making generic embeddings with post-index information.
"""

from __future__ import annotations

import argparse
import json
import string
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from bonsai.functional.checkpointing import (
    clean_lightning_state_dict,
    get_saved_encoder_config,
)
from opera.compat.bonsai import (
    FinetuneDataset,
    build_bonsai_encoder,
    dynamic_padding,
    encoder_hidden_state,
)
from opera.evaluation.outcome_transfer_evaluation import (
    DAPT_REPRESENTATION,
    _filtered_outcome_frame,
    _load_membership,
    _split_keys,
    file_hash,
    resolve_registry,
)
from opera.functional.outcomes import attach_prediction_censor_abspos
from opera.functional.outcome_transfer import (
    DEFAULT_MANIFEST,
    resolve_transfer_manifest,
)


class OutcomeTransferExtractionError(ValueError):
    """Raised when a checkpoint or subject split is unsafe for extraction."""


def _resolve_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(value)


def _read_checkpoint_metadata(
    checkpoint: Path,
    checkpoint_payload: Mapping[str, Any],
    metadata_path: str | Path | None = None,
) -> dict[str, Any]:
    source = (
        Path(metadata_path)
        if metadata_path is not None
        else checkpoint.parent / "checkpoint_metadata.json"
    )
    if source.exists():
        payload = json.loads(source.read_text(encoding="utf-8"))
        if not isinstance(payload, Mapping):
            raise OutcomeTransferExtractionError(
                f"Checkpoint sidecar {source} must contain a mapping."
            )
        metadata = payload.get("checkpoint_metadata", payload)
        if not isinstance(metadata, Mapping):
            raise OutcomeTransferExtractionError(
                f"Checkpoint sidecar {source} has non-mapping checkpoint_metadata."
            )
        return dict(metadata)
    hparams = checkpoint_payload.get("hyper_parameters", {})
    metadata = (
        hparams.get("checkpoint_metadata", {}) if isinstance(hparams, Mapping) else {}
    )
    if metadata and not isinstance(metadata, Mapping):
        raise OutcomeTransferExtractionError(
            f"Checkpoint {checkpoint} has non-mapping checkpoint_metadata."
        )
    return dict(metadata or {})


def _extract_backbone_state_dict(
    state_dict: Mapping[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Extract native encoder weights from DAPT or OPERA Lightning state dicts."""
    cleaned = clean_lightning_state_dict(dict(state_dict))
    if not cleaned:
        raise OutcomeTransferExtractionError(
            "Checkpoint contains no model state dictionary."
        )
    encoder_prefix = "encoder."
    if any(name.startswith(encoder_prefix) for name in cleaned):
        result = {
            name[len(encoder_prefix) :]: value
            for name, value in cleaned.items()
            if name.startswith(encoder_prefix)
        }
    else:
        # A DAPT/pretraining Lightning module wraps BonsaiPretrain directly,
        # so its clean state is already a backbone plus pretraining heads.
        excluded = (
            "head.",
            "decoder.",
            "cls.",
            "classifier.",
            "pretrain_head.",
            "value_head.",
            "value_bin_head.",
            "finetune_head.",
        )
        result = {
            name: value
            for name, value in cleaned.items()
            if not name.startswith(excluded)
        }
    if not result:
        raise OutcomeTransferExtractionError(
            "Checkpoint did not expose a native encoder namespace."
        )
    return result


def load_frozen_encoder(
    checkpoint_path: str | Path,
    *,
    vocabulary_path: str | Path,
    device: torch.device,
    attention_backend: str = "auto",
    checkpoint_metadata_path: str | Path | None = None,
) -> tuple[torch.nn.Module, dict[str, Any], dict[str, Any]]:
    """Reconstruct a DAPT/OPERA backbone and return checkpoint provenance."""
    checkpoint = Path(checkpoint_path)
    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint}")
    vocabulary_file = Path(vocabulary_path)
    if not vocabulary_file.exists():
        raise FileNotFoundError(f"Vocabulary does not exist: {vocabulary_file}")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping) or "state_dict" not in payload:
        raise OutcomeTransferExtractionError(
            f"Checkpoint {checkpoint} is not a Lightning checkpoint with state_dict."
        )
    hparams = payload.get("hyper_parameters")
    if not isinstance(hparams, Mapping):
        raise OutcomeTransferExtractionError(
            f"Checkpoint {checkpoint} is missing mapping hyper_parameters."
        )
    model_config = get_saved_encoder_config(dict(hparams))
    vocabulary = torch.load(vocabulary_file, map_location="cpu", weights_only=False)
    if not isinstance(vocabulary, Mapping):
        raise OutcomeTransferExtractionError(
            "Vocabulary must be a token-to-ID mapping."
        )
    saved_vocab_size = int(model_config.get("vocab_size", -1))
    if saved_vocab_size != len(vocabulary):
        raise OutcomeTransferExtractionError(
            f"Checkpoint vocab_size={saved_vocab_size} does not match vocabulary size={len(vocabulary)}."
        )
    saved_attention = str(model_config.get("attn_type", ""))
    if attention_backend not in {"auto", "checkpoint", "sdpa", "flash"}:
        raise OutcomeTransferExtractionError(
            "attention_backend must be auto, checkpoint, sdpa, or flash."
        )
    if attention_backend == "sdpa" or (
        attention_backend == "auto"
        and device.type == "cpu"
        and saved_attention == "flash"
    ):
        model_config["attn_type"] = "sdpa"
    elif attention_backend == "flash":
        model_config["attn_type"] = "flash"
    # 'auto' on CUDA and 'checkpoint' preserve the saved backend.
    encoder = build_bonsai_encoder(model_config, vocab_size=len(vocabulary))
    try:
        encoder.load_state_dict(
            _extract_backbone_state_dict(payload["state_dict"]), strict=True
        )
    except RuntimeError as exc:
        raise OutcomeTransferExtractionError(
            f"Could not strictly load encoder weights from {checkpoint}: {exc}"
        ) from exc
    encoder = encoder.to(device)
    encoder.eval()
    metadata = _read_checkpoint_metadata(
        checkpoint,
        payload,
        metadata_path=checkpoint_metadata_path,
    )
    model_init = hparams.get("model_init_config", {})
    if isinstance(model_init, Mapping) and model_init.get("pooling") is not None:
        # This is stored outside stage metadata by OperaContrastiveModule, so
        # preserve it for the extraction guard without treating it as a user
        # supplied transfer-provenance field.
        metadata["_checkpoint_pooling"] = model_init["pooling"]
    return encoder, dict(vocabulary), metadata


def _checkpoint_source_dapt(metadata: Mapping[str, Any]) -> str | None:
    """Return DAPT provenance from new or legacy contrastive sidecars."""
    for field in ("source_dapt_checkpoint", "source_checkpoint"):
        value = metadata.get(field)
        if (
            isinstance(value, str)
            and value.strip()
            and value.strip().lower() not in {"none", "null"}
        ):
            return value.strip()
    return None


def _legacy_seed_is_path_bound(
    *,
    checkpoint_path: str | Path | None,
    template: str | None,
    seed: int,
) -> bool:
    """Verify an unseeded legacy-full checkpoint is explicitly seed-bound.

    A historic full-OPERA checkpoint may predate a ``seed`` metadata field.
    It is still usable only when the operator supplies a template containing
    ``{seed}`` that resolves to this exact checkpoint, matching the launcher
    contract.  Filename guessing (for example looking for ``seed_42``) is
    deliberately not treated as provenance.
    """
    if checkpoint_path is None or not template:
        return False
    try:
        fields = [field for _, field, _, _ in string.Formatter().parse(template)]
        if "seed" not in fields or any(field not in (None, "seed") for field in fields):
            return False
        rendered = Path(template.format(seed=seed))
    except (KeyError, IndexError, ValueError):
        return False
    return rendered.resolve(strict=False) == Path(checkpoint_path).resolve(strict=False)


def _validate_checkpoint_identity(
    *,
    representation: str,
    seed: int,
    metadata: Mapping[str, Any],
    plan: Mapping[str, Any],
    checkpoint_path: str | Path | None = None,
    legacy_full_checkpoint_template: str | None = None,
) -> str | None:
    """Validate source checkpoint provenance and return its DAPT origin.

    Existing full-OPERA checkpoints are allowed only through the narrow legacy
    branch below.  Their old ``source_checkpoint`` field is normalized into
    the transfer sidecar's ``source_dapt_checkpoint`` field, so downstream
    evaluation still sees one strict provenance schema.
    """
    if representation == DAPT_REPRESENTATION:
        # Do not let an OPERA adaptation checkpoint be supplied under the
        # DAPT baseline key.  Historic DAPT sidecars may not have a condition
        # field, but their training stage and upstream pretraining source are
        # still mandatory.
        recorded_condition = metadata.get("condition")
        if recorded_condition not in (None, DAPT_REPRESENTATION):
            raise OutcomeTransferExtractionError(
                "DAPT extraction received a checkpoint tagged as "
                f"{recorded_condition!r}, not 'dapt'."
            )
        if metadata.get("training_stage") != "hematology_domain_adaptation":
            raise OutcomeTransferExtractionError(
                "DAPT extraction requires training_stage="
                "'hematology_domain_adaptation'."
            )
        source_checkpoint = metadata.get("source_checkpoint")
        if (
            not isinstance(source_checkpoint, str)
            or not source_checkpoint.strip()
            or source_checkpoint.strip().lower() in {"none", "null"}
        ):
            raise OutcomeTransferExtractionError(
                "DAPT extraction requires a non-empty source_checkpoint."
            )
        return None

    if representation != DAPT_REPRESENTATION:
        actual = metadata.get("condition")
        legacy_full = representation == "opera_full" and actual is None
        expected = plan["conditions"][representation]
        if legacy_full:
            expected_outcomes = list(expected["training_outcomes"])
            observed = list(metadata.get("outcome_set", []))
            if (
                metadata.get("training_stage") != "opera_contrastive_adaptation"
                or set(observed) != set(expected_outcomes)
                or len(observed) != len(expected_outcomes)
            ):
                raise OutcomeTransferExtractionError(
                    "Legacy opera_full reuse requires training_stage="
                    "'opera_contrastive_adaptation' and an outcome_set equal to the "
                    "canonical full OPERA panel."
                )
            recorded_seed = metadata.get("seed")
            if recorded_seed is None:
                if not _legacy_seed_is_path_bound(
                    checkpoint_path=checkpoint_path,
                    template=legacy_full_checkpoint_template,
                    seed=seed,
                ):
                    raise OutcomeTransferExtractionError(
                        "Legacy opera_full checkpoint lacks seed provenance. Supply a "
                        "matching --legacy-full-checkpoint-template containing {seed}, "
                        "or use a checkpoint with a matching seed metadata field."
                    )
            else:
                try:
                    seed_matches = not isinstance(recorded_seed, bool) and int(
                        recorded_seed
                    ) == int(seed)
                except (TypeError, ValueError):
                    seed_matches = False
                if not seed_matches:
                    raise OutcomeTransferExtractionError(
                        f"Legacy opera_full checkpoint seed={recorded_seed!r} does not "
                        f"match requested seed={seed}."
                    )
        elif actual is None:
            raise OutcomeTransferExtractionError(
                f"OPERA checkpoint for {representation!r} lacks condition metadata."
            )
        elif str(actual) != representation:
            raise OutcomeTransferExtractionError(
                f"Checkpoint condition={actual!r} does not match requested {representation!r}."
            )
        if not legacy_full:
            exact = {
                "training_stage": "opera_contrastive_adaptation",
                "condition": representation,
                "transfer_level": expected["transfer_level"],
                "seed": int(seed),
                "registry_hash": plan["registry_hash"],
                "manifest_hash": plan["manifest_hash"],
                "base_contrastive_config_hash": plan["base_contrastive_config_hash"],
                "included_outcomes": list(expected["training_outcomes"]),
                "excluded_outcomes": list(expected["training_excluded_outcomes"]),
                "evaluation_outcomes": list(expected["evaluation_outcomes"]),
                "related_retained_outcomes": list(
                    expected["related_retained_outcomes"]
                ),
                "direct_dependencies_excluded": list(
                    expected["direct_dependencies_excluded"]
                ),
                "selection_outcomes": list(expected["training_outcomes"]),
                "split_contract": plan["split_contract"],
                "split_contract_hash": plan["split_contract_hash"],
            }
            missing = [field for field in exact if field not in metadata]
            if missing:
                raise OutcomeTransferExtractionError(
                    f"OPERA checkpoint for {representation!r} has incomplete provenance; "
                    f"missing {sorted(missing)}."
                )
            for field, expected_value in exact.items():
                if metadata[field] != expected_value:
                    raise OutcomeTransferExtractionError(
                        f"Checkpoint {field} does not match the resolved transfer plan."
                    )
            observed = metadata.get("outcome_set")
            expected_outcomes = list(expected["training_outcomes"])
            if (
                not isinstance(observed, (list, tuple))
                or set(observed) != set(expected_outcomes)
                or len(observed) != len(expected_outcomes)
            ):
                raise OutcomeTransferExtractionError(
                    f"Checkpoint outcome_set does not match {representation!r}'s "
                    "resolved training panel."
                )
        source_dapt_checkpoint = _checkpoint_source_dapt(metadata)
        if source_dapt_checkpoint is None:
            raise OutcomeTransferExtractionError(
                f"OPERA checkpoint for {representation!r} lacks a non-empty DAPT "
                "source checkpoint (source_dapt_checkpoint or legacy source_checkpoint)."
            )
        return source_dapt_checkpoint
    raise AssertionError("DAPT branch should have returned above.")


def validate_common_prediction_origins(
    plan: Mapping[str, Any],
    registry: Mapping[str, Any],
    *,
    reference_outcome: str,
) -> pd.DataFrame:
    """Verify all transfer targets share a reference index date and split.

    Only ``subject_id``, ``split``, and ``index_date`` are read for this
    validation.  Events/outcome dates never enter the representation extractor.
    The returned reference frame includes ``censor_abspos`` used to truncate
    sequences at that common prediction origin.
    """
    membership = _load_membership(registry)
    reference, _, _ = _filtered_outcome_frame(registry, reference_outcome, membership)
    reference = reference.copy()
    reference["index_date"] = pd.to_datetime(reference["index_date"], errors="coerce")
    if reference["index_date"].isna().any():
        raise OutcomeTransferExtractionError(
            f"Reference outcome {reference_outcome!r} has missing index_date values."
        )
    reference = attach_prediction_censor_abspos(reference)
    if reference["_subject_key"].duplicated().any():
        raise OutcomeTransferExtractionError(
            "Reference outcome must contain one row per subject after eligibility filtering."
        )
    reference_index = reference.set_index("_subject_key")[["split", "index_date"]]
    targets = list(plan.get("evaluation_target_union", []))
    if not targets:
        raise OutcomeTransferExtractionError(
            "Resolved transfer plan has no evaluation_target_union."
        )
    for target in targets:
        target_frame, _, _ = _filtered_outcome_frame(registry, str(target), membership)
        candidate = target_frame[["_subject_key", "split", "index_date"]].copy()
        candidate["index_date"] = pd.to_datetime(
            candidate["index_date"], errors="coerce"
        )
        if candidate["index_date"].isna().any():
            raise OutcomeTransferExtractionError(
                f"Transfer target {target!r} has missing index_date values."
            )
        joined = candidate.set_index("_subject_key").join(
            reference_index,
            how="left",
            rsuffix="_reference",
            validate="one_to_one",
        )
        if joined["split_reference"].isna().any():
            missing = joined.index[joined["split_reference"].isna()].tolist()[:10]
            raise OutcomeTransferExtractionError(
                f"Transfer target {target!r} has patients absent from the reference "
                f"prediction-origin outcome; examples={missing}."
            )
        same_split = joined["split"].astype(str) == joined["split_reference"].astype(
            str
        )
        same_date = joined["index_date"] == joined["index_date_reference"]
        if not (same_split & same_date).all():
            bad = joined.loc[~(same_split & same_date)].head(5)
            raise OutcomeTransferExtractionError(
                f"Transfer target {target!r} does not share the reference prediction "
                "origin. One generic frozen embedding table would be unsafe; "
                f"examples={bad.reset_index().to_dict('records')}."
            )
    return reference


def _reference_records(
    reference: pd.DataFrame, split_key: str
) -> dict[int, dict[str, Any]]:
    selected = reference.loc[reference["split"].astype(str) == str(split_key)].copy()
    records: dict[int, dict[str, Any]] = {}
    for row in selected.itertuples(index=False):
        try:
            subject_id = int(row.subject_id)
        except (TypeError, ValueError) as exc:
            raise OutcomeTransferExtractionError(
                "Frozen subject data extraction currently requires integer subject IDs, "
                "matching BONSAI's outcome binarization contract."
            ) from exc
        records[subject_id] = {"label": 0, "censor_abspos": float(row.censor_abspos)}
    return records


def _pool_cls_last(
    encoder: torch.nn.Module, batch: Mapping[str, torch.Tensor]
) -> torch.Tensor:
    hidden = encoder_hidden_state(encoder(batch))
    lengths = batch["attention_mask"].sum(dim=1) - 1
    if torch.any(lengths < 0):
        raise OutcomeTransferExtractionError(
            "Encountered an empty sequence during extraction."
        )
    return hidden[
        torch.arange(hidden.size(0), device=hidden.device),
        lengths,
    ]


def _pool_prediction_origin(
    encoder: torch.nn.Module,
    batch: Mapping[str, torch.Tensor],
    pooling: str,
) -> torch.Tensor:
    """Pool the censored sequence exactly as the selected OPERA representation."""
    if pooling == "cls_last":
        return _pool_cls_last(encoder, batch)
    if pooling not in {"last", "mean", "mean_last_128"}:
        raise OutcomeTransferExtractionError(
            "Prediction-origin extraction supports cls_last, last, mean, and "
            "mean_last_128."
        )
    hidden = encoder_hidden_state(encoder(batch))
    # FinetuneDataset appends [CLS] at `lengths`. The pooling modes below
    # deliberately exclude that token, which is important for checkpoints
    # whose pretraining objective never trained a [CLS] representation.
    lengths = batch["attention_mask"].sum(dim=1) - 1
    if torch.any(lengths <= 0):
        raise OutcomeTransferExtractionError(
            f"{pooling} requires at least one token before [CLS]."
        )
    if pooling == "last":
        return hidden[
            torch.arange(hidden.size(0), device=hidden.device), lengths - 1
        ]
    positions = torch.arange(hidden.size(1), device=hidden.device).unsqueeze(0)
    starts = (
        torch.zeros_like(lengths)
        if pooling == "mean"
        else (lengths - 128).clamp_min(0)
    ).unsqueeze(1)
    clinical = (positions >= starts) & (positions < lengths.unsqueeze(1))
    weights = clinical.unsqueeze(-1).to(hidden.dtype)
    return (hidden * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)


def extract_shared_split_embeddings(
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
    pooling: str = "cls_last",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, int]]:
    """Pool physical SSL shards, then select temporal outcome splits."""
    if "[CLS]" not in vocabulary:
        raise OutcomeTransferExtractionError(
            "Vocabulary must contain '[CLS]' for prediction-origin extraction."
        )
    from opera.modules.datamodules.OutcomeFinetuneDataModule import load_subject_pool

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
    sequence_lengths: list[np.ndarray] = []
    counts: dict[str, int] = {}
    for split in ("train", "tuning", "held_out"):
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
                f"Pooled physical shards do not exactly cover temporal split {split!r}; "
                f"missing={len(missing)} examples={missing[:10]}, duplicates={duplicate}."
            )
        background_length = int((selected[0]["segment"] == 0).sum())
        dataset = FinetuneDataset(
            selected,
            outcomes=records,
            predict_token_id=int(vocabulary["[CLS]"]),
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
        split_lengths: list[np.ndarray] = []
        with torch.inference_mode():
            for batch in loader:
                device_batch = {
                    key: value.to(device) if isinstance(value, torch.Tensor) else value
                    for key, value in batch.items()
                }
                pooled = _pool_prediction_origin(encoder, device_batch, pooling)
                split_ids.append(device_batch["subject_id"].detach().cpu().numpy())
                split_embeddings.append(
                    pooled.detach().cpu().numpy().astype(np.float32)
                )
                split_lengths.append(
                    device_batch["attention_mask"].sum(dim=1).detach().cpu().numpy()
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
        embeddings.append(np.concatenate(split_embeddings))
        sequence_lengths.append(np.concatenate(split_lengths))
        counts[split] = int(len(split_subject_ids))
    all_ids = np.concatenate(subject_ids)
    all_embeddings = np.concatenate(embeddings)
    if len(np.unique(all_ids)) != len(all_ids):
        raise OutcomeTransferExtractionError(
            "A subject appeared in multiple shared subject splits; extraction is unsafe."
        )
    return all_ids, all_embeddings, np.concatenate(sequence_lengths), counts


def _subject_paths(args: argparse.Namespace) -> dict[str, Path]:
    explicit = {
        "train": args.train_subject_data,
        "tuning": args.tuning_subject_data,
        "held_out": args.held_out_subject_data,
    }
    if args.subject_data_dir:
        root = Path(args.subject_data_dir)
        configured = {
            "ssl_train": Path(args.train_subject_data)
            if args.train_subject_data
            else root / "subject_data_train.pt",
            "ssl_validation": Path(args.tuning_subject_data)
            if args.tuning_subject_data
            else root / "subject_data_tuning.pt",
        }
        if args.held_out_subject_data:
            configured["legacy_third_shard"] = Path(args.held_out_subject_data)
        return configured
    supplied = [value for value in explicit.values() if value is not None]
    if not supplied:
        raise OutcomeTransferExtractionError(
            "Provide --subject-data-dir or at least one physical subject-data shard."
        )
    return {f"physical_{index}": Path(value) for index, value in enumerate(supplied)}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Extract cls_last frozen embeddings for focused OPERA outcome-transfer probes."
    )
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--registry", default=None)
    parser.add_argument("--base-config", default=None)
    parser.add_argument(
        "--representation",
        required=True,
        help="dapt or a resolved OPERA transfer condition.",
    )
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--checkpoint-metadata", default=None)
    parser.add_argument(
        "--legacy-full-checkpoint-template",
        default=None,
        help=(
            "Required only when reusing a legacy unseeded opera_full checkpoint; "
            "a seed-aware template (for example /ckpts/full/seed_{seed}/best.ckpt) "
            "that resolves to --checkpoint."
        ),
    )
    parser.add_argument("--vocabulary", required=True)
    parser.add_argument("--subject-data-dir", default=None)
    parser.add_argument("--train-subject-data", default=None)
    parser.add_argument("--tuning-subject-data", default=None)
    parser.add_argument("--held-out-subject-data", default=None)
    parser.add_argument("--reference-outcome", default=None)
    parser.add_argument("--max-len", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--attention-backend",
        choices=["auto", "checkpoint", "sdpa", "flash"],
        default="auto",
    )
    parser.add_argument(
        "--output", required=True, help="Output NPZ embedding artifact."
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    plan = resolve_transfer_manifest(
        args.manifest,
        registry_path=args.registry,
        base_config_path=args.base_config,
    )
    if (
        args.representation != DAPT_REPRESENTATION
        and args.representation not in plan["conditions"]
    ):
        raise OutcomeTransferExtractionError(
            f"representation must be 'dapt' or one of {sorted(plan['conditions'])}."
        )
    if args.seed not in plan["seeds"]:
        raise OutcomeTransferExtractionError(
            f"seed={args.seed} is not in the canonical transfer seeds {plan['seeds']}."
        )
    # ``plan['registry']`` is repository-relative in the checked-in manifest.
    # Use the evaluator's shared resolver so extraction remains runnable from
    # a server results directory rather than depending on the launch CWD.
    registry = resolve_registry(args.registry or plan["registry"])
    device = _resolve_device(args.device)
    encoder, vocabulary, checkpoint_metadata = load_frozen_encoder(
        args.checkpoint,
        vocabulary_path=args.vocabulary,
        device=device,
        attention_backend=args.attention_backend,
        checkpoint_metadata_path=args.checkpoint_metadata,
    )
    source_dapt_checkpoint = _validate_checkpoint_identity(
        representation=args.representation,
        seed=args.seed,
        metadata=checkpoint_metadata,
        plan=plan,
        checkpoint_path=args.checkpoint,
        legacy_full_checkpoint_template=args.legacy_full_checkpoint_template,
    )
    saved_pooling = checkpoint_metadata.get(
        "_checkpoint_pooling"
    ) or checkpoint_metadata.get("pooling")
    if saved_pooling not in (None, "cls_last"):
        raise OutcomeTransferExtractionError(
            f"Transfer extraction supports only cls_last pooling; checkpoint records {saved_pooling!r}."
        )
    reference_outcome = args.reference_outcome or str(registry["death_outcome"])
    reference = validate_common_prediction_origins(
        plan,
        registry,
        reference_outcome=reference_outcome,
    )
    max_len = int(args.max_len or encoder.hparams["max_seqlen"])
    if max_len <= 0 or args.batch_size <= 0 or args.num_workers < 0:
        raise OutcomeTransferExtractionError(
            "max-len and batch-size must be positive; num-workers non-negative."
        )
    paths = _subject_paths(args)
    subject_ids, embeddings, sequence_lengths, split_counts = extract_shared_split_embeddings(
        encoder,
        reference=reference,
        subject_split_paths=paths,
        vocabulary=vocabulary,
        split_keys=_split_keys(plan["split_contract"]),
        max_len=max_len,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=device,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        subject_ids=subject_ids,
        embeddings=embeddings,
        sequence_length=sequence_lengths,
    )
    resolved_condition_metadata: dict[str, Any] = {}
    if args.representation != DAPT_REPRESENTATION:
        condition = plan["conditions"][args.representation]
        resolved_condition_metadata = {
            "training_stage": "opera_contrastive_adaptation",
            "condition": args.representation,
            "transfer_level": condition["transfer_level"],
            "included_outcomes": list(condition["training_outcomes"]),
            "excluded_outcomes": list(condition["training_excluded_outcomes"]),
            "evaluation_outcomes": list(condition["evaluation_outcomes"]),
            "related_retained_outcomes": list(condition["related_retained_outcomes"]),
            "direct_dependencies_excluded": list(
                condition["direct_dependencies_excluded"]
            ),
            "selection_outcomes": list(condition["training_outcomes"]),
            # Contrastive modules sort internal head names, so preserve the
            # canonical plan order in the extraction sidecar and let the
            # validator compare this field as a set plus cardinality.
            "outcome_set": list(condition["training_outcomes"]),
            "source_dapt_checkpoint": source_dapt_checkpoint,
        }
    metadata = {
        "checkpoint_metadata": {
            **checkpoint_metadata,
            **resolved_condition_metadata,
            "condition": args.representation,
            "seed": int(args.seed),
            "checkpoint_hash": file_hash(args.checkpoint),
            "registry_hash": plan["registry_hash"],
            "manifest_hash": plan["manifest_hash"],
            "base_contrastive_config_hash": plan["base_contrastive_config_hash"],
            "split_contract": plan["split_contract"],
            "split_contract_hash": plan["split_contract_hash"],
        },
        "embedding_extraction": {
            "representation": args.representation,
            "seed": int(args.seed),
            "checkpoint": str(Path(args.checkpoint)),
            "vocabulary": str(Path(args.vocabulary)),
            "reference_outcome": reference_outcome,
            "prediction_origin_verified_for_targets": list(
                plan["evaluation_target_union"]
            ),
            "pooling": "cls_last",
            "encoder_frozen": True,
            "max_len": max_len,
            "subject_split_paths": {name: str(path) for name, path in paths.items()},
            "split_counts": split_counts,
            "n_subjects": int(len(subject_ids)),
            "embedding_dim": int(embeddings.shape[1]),
            "registry_hash": plan["registry_hash"],
            "manifest_hash": plan["manifest_hash"],
            "base_contrastive_config_hash": plan["base_contrastive_config_hash"],
            "split_contract_hash": plan["split_contract_hash"],
        },
    }
    sidecar = output.with_suffix(".metadata.json")
    sidecar.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(
        f"Wrote {len(subject_ids):,} cls_last frozen embeddings ({embeddings.shape[1]} dimensions) "
        f"to {output}; sidecar={sidecar}."
    )


if __name__ == "__main__":
    main()
