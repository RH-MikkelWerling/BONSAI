"""Frozen-probe evaluation primitives for the focused OPERA transfer study.

This module intentionally has a narrow contract.  It never trains an encoder
and it never selects an OPERA checkpoint.  It consumes one already-extracted,
frozen patient embedding table for each ``(representation, seed)`` pair and
uses the canonical outcome files only *after* contrastive training is over.

Embedding artefacts accepted by :func:`load_embedding_artifact` are either
CSV/Parquet tables with ``subject_id`` plus numeric ``embedding_*`` columns,
or NPZ files containing ``subject_ids`` and a two-dimensional ``embeddings``
array.  A representation may have additional patients, but it must include
every eligible labelled patient for every comparison it is asked to make.  A
missing patient is a denominator mismatch and is an error, not a reason to
silently shrink a test set.

The evaluator is deliberately independent from the broad OPERA sweep.  Its
only downstream model is a standardized logistic-regression probe fitted on
pan-hematology training patients, selected on pan-hematology tuning patients,
and scored once on the locked held-out population.  Grouped-cohort rows are
then re-stratifications of those same saved held-out predictions.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from opera.compat.bonsai import binarize_outcomes
from opera.evaluation.metrics import calibration_intercept_slope
from opera.evaluation.treatment_embeddings import embedding_columns
from opera.functional.outcomes import (
    filter_outcome_eligibility,
    filter_registry_eligible_outcomes,
)
from opera.run.generate_sweep_configs import load_registry


ALL_HEMATOLOGY = "all_hematology"
DAPT_REPRESENTATION = "dapt"
NO_OUTCOME_GUIDED_ADAPTATION = "no_outcome_guided_adaptation"
REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
METRIC_NAMES = (
    "auroc",
    "auprc",
    "brier_score",
    "calibration_intercept",
    "calibration_slope",
)
DEFAULT_C_GRID = (0.01, 0.1, 1.0, 10.0, 100.0)

# These schemas remain present even when a real target has no fit-able probe.
# A schema-stable empty CSV is an explicit unsupported-result artefact, not a
# silently missing output that downstream aggregation might misinterpret.
_LABEL_COUNT_COLUMNS = (
    "n_train",
    "n_train_events",
    "n_train_non_events",
    "n_train_competing_events",
    "n_train_censored_before_horizon",
    "n_tuning",
    "n_tuning_events",
    "n_tuning_non_events",
    "n_tuning_competing_events",
    "n_tuning_censored_before_horizon",
    "n_test",
    "n_test_events",
    "n_test_non_events",
    "n_test_competing_events",
    "n_test_censored_before_horizon",
    "n_competing_events",
    "n_censored_before_horizon",
    "prevalence",
)
TRANSFER_RESULT_COLUMNS = (
    "condition",
    "comparison_condition",
    "seed",
    "target_outcome",
    "target_family",
    "transfer_level",
    "primary_horizon_days",
    "evaluation_level",
    "evaluation_group",
    "direct_target_seen",
    "same_family_seen",
    "matched_lower_grade_seen",
    "included_outcome_count",
    "excluded_outcome_count",
    "checkpoint_hash",
    "checkpoint_hash_source",
    "embedding_artifact_hash",
    "registry_hash",
    "manifest_hash",
    "test_denominator_hash",
    "selected_probe_c",
    "tuning_auroc_for_selection",
    "probe_type",
    "encoder_frozen",
    "n_test_metric_rows",
    *_LABEL_COUNT_COLUMNS,
    "metric",
    "value",
)
TRANSFER_PREDICTION_COLUMNS = (
    "subject_id",
    "_subject_key",
    "split",
    "cohort_grouped",
    "label",
    "event",
    "time_days",
    "censor_abspos",
    "probability",
    "representation",
    "condition",
    "comparison_condition",
    "seed",
    "target_outcome",
    "target_family",
    "transfer_level",
    "primary_horizon_days",
    "direct_target_seen",
    "same_family_seen",
    "matched_lower_grade_seen",
    "included_outcome_count",
    "excluded_outcome_count",
    "checkpoint_hash",
    "checkpoint_hash_source",
    "embedding_artifact_hash",
    "registry_hash",
    "manifest_hash",
)
TRANSFER_PROBE_STATUS_COLUMNS = (
    "condition",
    "comparison_condition",
    "seed",
    "target_outcome",
    "primary_horizon_days",
    "status",
    "reason",
    "selected_probe_c",
    "tuning_auroc_for_selection",
    "probe_type",
    "encoder_frozen",
    *_LABEL_COUNT_COLUMNS,
)
TRANSFER_FAILURE_COLUMNS = (
    "stage",
    "condition",
    "comparison_condition",
    "seed",
    "target_outcome",
    "primary_horizon_days",
    "failure_type",
    "message",
)


class OutcomeTransferEvaluationError(ValueError):
    """Base error for an invalid frozen-probe transfer evaluation."""


class DenominatorParityError(OutcomeTransferEvaluationError):
    """Raised when representations would be compared on different patients."""


class ProbeSupportError(OutcomeTransferEvaluationError):
    """Raised when a pan-hematology probe cannot be fit or tuned honestly."""


@dataclass(frozen=True)
class EmbeddingArtifact:
    """A validated frozen representation for one condition and seed."""

    representation: str
    seed: int
    path: Path
    frame: pd.DataFrame
    artifact_hash: str
    checkpoint_hash: str
    checkpoint_hash_source: str
    metadata: Mapping[str, Any]
    metadata_path: Path | None = None


@dataclass(frozen=True)
class TargetLabelBundle:
    """Fixed-horizon labels plus censoring bookkeeping for one target."""

    labels: pd.DataFrame
    raw_status: pd.DataFrame
    target_outcome: str
    horizon_days: int


@dataclass(frozen=True)
class FrozenProbe:
    """A selected pan-hematology standardized linear probe."""

    pipeline: Pipeline
    selected_c: float
    tuning_auroc: float


def _with_schema(frame: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    """Return a table with deterministic required columns, preserving extras."""
    result = frame.copy()
    for column in columns:
        if column not in result:
            result[column] = pd.Series(dtype="object")
    ordered = [*columns, *(column for column in result if column not in columns)]
    return result.loc[:, ordered]


def file_hash(path: str | Path) -> str:
    """Return the SHA256 of an immutable evaluation input artefact."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def patient_id_hash(subject_ids: Sequence[object]) -> str:
    """Hash a patient denominator in a stable, type-tolerant way."""
    values = sorted({str(value) for value in subject_ids})
    return hashlib.sha256("\n".join(values).encode("utf-8")).hexdigest()


def _subject_key(value: object) -> str:
    if pd.isna(value):
        raise OutcomeTransferEvaluationError("subject_id values must not be missing.")
    return str(value)


def _read_table(path: str | Path) -> pd.DataFrame:
    source = Path(path)
    suffix = source.suffix.lower()
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(source)
    if suffix in {".csv", ".txt"}:
        return pd.read_csv(source)
    raise OutcomeTransferEvaluationError(
        f"Unsupported table extension {source.suffix!r} for {source}; "
        "expected Parquet or CSV."
    )


def _read_artifact_frame(path: str | Path) -> pd.DataFrame:
    """Read an embedding artefact into the canonical wide table shape."""
    source = Path(path)
    if not source.exists():
        raise OutcomeTransferEvaluationError(f"Embedding artefact does not exist: {source}")
    if source.suffix.lower() == ".npz":
        with np.load(source, allow_pickle=False) as payload:
            if "subject_ids" not in payload or "embeddings" not in payload:
                raise OutcomeTransferEvaluationError(
                    f"{source} must contain NPZ arrays 'subject_ids' and 'embeddings'."
                )
            subject_ids = np.asarray(payload["subject_ids"])
            embeddings = np.asarray(payload["embeddings"], dtype=float)
        if embeddings.ndim != 2 or embeddings.shape[1] == 0:
            raise OutcomeTransferEvaluationError(
                f"{source} embeddings must be a non-empty two-dimensional array."
            )
        if len(subject_ids) != len(embeddings):
            raise OutcomeTransferEvaluationError(
                f"{source} has {len(subject_ids)} subject IDs but {len(embeddings)} embeddings."
            )
        frame = pd.DataFrame(
            embeddings,
            columns=[f"embedding_{index}" for index in range(embeddings.shape[1])],
        )
        frame.insert(0, "subject_id", subject_ids)
        return frame
    return _read_table(source)


def _read_metadata(path: str | Path | None) -> Mapping[str, Any]:
    if path is None:
        return {}
    source = Path(path)
    if not source.exists():
        raise OutcomeTransferEvaluationError(f"Checkpoint metadata does not exist: {source}")
    with source.open(encoding="utf-8") as handle:
        if source.suffix.lower() == ".json":
            payload = json.load(handle)
        else:
            payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, Mapping):
        raise OutcomeTransferEvaluationError(
            f"Checkpoint metadata {source} must contain a mapping."
        )
    # BONSAI's standard sidecars wrap the stage-specific fields in
    # ``checkpoint_metadata`` alongside architecture information.  Accept a
    # bare metadata mapping too, but never accidentally validate the wrapper
    # rather than the actual training provenance.
    wrapped = payload.get("checkpoint_metadata")
    if wrapped is not None:
        if not isinstance(wrapped, Mapping):
            raise OutcomeTransferEvaluationError(
                f"Checkpoint sidecar {source} has a non-mapping checkpoint_metadata field."
            )
        return dict(wrapped)
    return dict(payload)


def load_embedding_artifact(
    representation: str,
    seed: int,
    path: str | Path,
    *,
    metadata_path: str | Path | None = None,
) -> EmbeddingArtifact:
    """Load and validate one frozen embedding artefact.

    ``metadata_path`` is optional because historic DAPT/full-OPERA artefacts
    may pre-date transfer sidecars.  If it is absent, the immutable embedding
    artefact hash is recorded as the checkpoint identifier rather than claiming
    a checkpoint hash that cannot be verified.
    """
    if not representation:
        raise OutcomeTransferEvaluationError("Embedding representation must be non-empty.")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise OutcomeTransferEvaluationError("Embedding seed must be an integer.")
    source = Path(path)
    frame = _read_artifact_frame(source).copy()
    if "subject_id" not in frame:
        raise OutcomeTransferEvaluationError(f"Embedding artefact {source} is missing 'subject_id'.")
    frame["_subject_key"] = frame["subject_id"].map(_subject_key)
    if frame["_subject_key"].duplicated().any():
        examples = frame.loc[frame["_subject_key"].duplicated(), "subject_id"].head(10)
        raise OutcomeTransferEvaluationError(
            f"Embedding artefact {source} has duplicate subject IDs: {examples.tolist()}."
        )
    columns = embedding_columns(frame)
    finite = np.isfinite(frame[columns].to_numpy(dtype=float)).all(axis=1)
    if not finite.all():
        count = int((~finite).sum())
        raise OutcomeTransferEvaluationError(
            f"Embedding artefact {source} has {count} rows with non-finite embedding values. "
            "Drop or repair them before evaluation; the evaluator will not alter denominators."
        )
    # Transfer extraction writes a sibling ``*.metadata.json`` sidecar.  Use
    # it automatically when present, while allowing callers to explicitly
    # point at an existing BONSAI ``checkpoint_metadata.json`` sidecar.
    inferred_metadata = source.with_suffix(".metadata.json")
    selected_metadata = metadata_path
    if selected_metadata is None and inferred_metadata.exists():
        selected_metadata = inferred_metadata
    metadata = _read_metadata(selected_metadata)
    # DAPT is one frozen baseline, not three independently trained models.
    # The same DAPT artifact may intentionally be paired with each OPERA seed.
    if (
        representation != DAPT_REPRESENTATION
        and "seed" in metadata
        and int(metadata["seed"]) != seed
    ):
        raise OutcomeTransferEvaluationError(
            f"Checkpoint metadata seed={metadata['seed']!r} does not match requested seed={seed}."
        )
    if "condition" in metadata and representation != DAPT_REPRESENTATION:
        if str(metadata["condition"]) != representation:
            raise OutcomeTransferEvaluationError(
                "Checkpoint metadata condition does not match embedding representation: "
                f"{metadata['condition']!r} != {representation!r}."
            )
    artifact_hash = file_hash(source)
    supplied_checkpoint_hash = metadata.get("checkpoint_hash")
    checkpoint_hash = str(supplied_checkpoint_hash or artifact_hash)
    hash_source = "checkpoint_metadata" if supplied_checkpoint_hash else "embedding_artifact"
    return EmbeddingArtifact(
        representation=representation,
        seed=seed,
        path=source,
        frame=frame,
        artifact_hash=artifact_hash,
        checkpoint_hash=checkpoint_hash,
        checkpoint_hash_source=hash_source,
        metadata=metadata,
        metadata_path=Path(selected_metadata) if selected_metadata is not None else None,
    )


def _expand_path(value: str | Path) -> Path:
    return Path(os.path.expandvars(os.path.expanduser(str(value))))


def _resolve_repository_or_cwd_path(value: str | Path) -> Path:
    """Resolve a configured path without making the launch CWD part of its API.

    Transfer manifests deliberately use repository-relative paths for the
    registry and temporal split contract.  Evaluation commands are commonly
    launched from a results directory on the server, where interpreting those
    paths solely relative to the current working directory would fail.  Keep
    an existing caller-relative file usable (which is helpful for an explicit
    local override), then fall back to the immutable repository root used by
    the manifest resolver.
    """
    candidate = _expand_path(value)
    if candidate.is_absolute() or candidate.exists():
        return candidate
    return REPOSITORY_ROOT / candidate


def _path_from_raw(raw: str | Path, root: Path) -> Path:
    path = _expand_path(raw)
    return path if path.is_absolute() else root / path


def _outcome_metadata(registry: Mapping[str, Any], outcome: str) -> Mapping[str, Any]:
    for field in ("outcome_definitions", "outcome_configs", "outcome_metadata"):
        values = registry.get(field)
        if isinstance(values, Mapping) and isinstance(values.get(outcome), Mapping):
            return values[outcome]
    return {}


def _load_membership(registry: Mapping[str, Any]) -> pd.DataFrame:
    paths = registry.get("paths")
    columns = registry.get("cohort_columns")
    if not isinstance(paths, Mapping) or not paths.get("cohort_membership_file"):
        raise OutcomeTransferEvaluationError(
            "Registry paths.cohort_membership_file is required for transfer evaluation."
        )
    if not isinstance(columns, Mapping) or not columns.get("grouped"):
        raise OutcomeTransferEvaluationError(
            "Registry cohort_columns.grouped is required for transfer evaluation."
        )
    membership = _read_table(_expand_path(paths["cohort_membership_file"])).copy()
    grouped = str(columns["grouped"])
    required = {"subject_id", grouped}
    missing = required - set(membership.columns)
    if missing:
        raise OutcomeTransferEvaluationError(
            f"Cohort membership is missing required columns: {sorted(missing)}."
        )
    membership["_subject_key"] = membership["subject_id"].map(_subject_key)
    if membership["_subject_key"].duplicated().any():
        raise OutcomeTransferEvaluationError(
            "Cohort membership must contain one row per subject_id for frozen-probe "
            "denominator parity."
        )
    membership = membership.rename(columns={grouped: "cohort_grouped"})
    membership["cohort_grouped"] = membership["cohort_grouped"].astype(str)
    return membership[["subject_id", "_subject_key", "cohort_grouped"]].copy()


def _split_keys(split_contract: str | Path) -> dict[str, str]:
    contract_path = _resolve_repository_or_cwd_path(split_contract)
    if not contract_path.exists():
        raise OutcomeTransferEvaluationError(
            f"Transfer split contract does not exist: {contract_path}"
        )
    with contract_path.open(encoding="utf-8") as handle:
        contract = yaml.safe_load(handle) or {}
    keys = {
        "train": str(contract.get("train_key", "train")),
        "tuning": str(contract.get("val_key", contract.get("tuning_key", "tuning"))),
        "held_out": str(contract.get("test_key", "held_out")),
    }
    if len(set(keys.values())) != len(keys):
        raise OutcomeTransferEvaluationError(
            f"Transfer split contract has non-distinct split keys: {keys}."
        )
    return keys


def _resolve_outcome_inputs(
    registry: Mapping[str, Any], outcome: str
) -> tuple[Path, Path | None, Path | None, Any | None, int]:
    paths = registry.get("paths")
    if not isinstance(paths, Mapping) or not paths.get("outcomes_dir"):
        raise OutcomeTransferEvaluationError(
            "Registry paths.outcomes_dir is required for transfer evaluation."
        )
    outcome_root = _expand_path(paths["outcomes_dir"])
    metadata = _outcome_metadata(registry, outcome)
    outcome_path = _path_from_raw(metadata.get("outcome_file", f"{outcome}.parquet"), outcome_root)
    death = str(registry["death_outcome"])
    raw_competing = metadata.get("competing_outcome_path") or metadata.get(
        "competing_outcome_file"
    )
    if raw_competing in (None, "", "null") and outcome != death:
        raw_competing = f"{death}.parquet"
    competing_path = (
        None if raw_competing in (None, "", "null") else _path_from_raw(raw_competing, outcome_root)
    )
    raw_eligibility = metadata.get("eligibility_file")
    if raw_eligibility in (None, "", "null"):
        possible = registry.get("eligibility_files")
        if isinstance(possible, Mapping):
            raw_eligibility = possible.get(outcome)
    eligibility_path = (
        None
        if raw_eligibility in (None, "", "null")
        else _path_from_raw(raw_eligibility, outcome_root)
    )
    start_hours = metadata.get("n_hours_start_include", 1)
    if isinstance(start_hours, bool) or not isinstance(start_hours, int) or start_hours < 0:
        raise OutcomeTransferEvaluationError(
            f"Outcome {outcome!r} has invalid n_hours_start_include={start_hours!r}."
        )
    return outcome_path, competing_path, eligibility_path, metadata.get("registry_start_date"), start_hours


def _filtered_outcome_frame(
    registry: Mapping[str, Any],
    outcome: str,
    membership: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame | None, int]:
    """Load precisely the canonical ascertainment population for one endpoint."""
    outcome_path, competing_path, eligibility_path, registry_start_date, start_hours = (
        _resolve_outcome_inputs(registry, outcome)
    )
    if not outcome_path.exists():
        raise OutcomeTransferEvaluationError(
            f"Outcome file for {outcome!r} does not exist: {outcome_path}"
        )
    frame = _read_table(outcome_path).copy()
    required = {"subject_id", "split", "index_date", "censor_date"}
    missing = required - set(frame.columns)
    if missing:
        raise OutcomeTransferEvaluationError(
            f"Outcome {outcome!r} is missing required columns: {sorted(missing)}"
        )
    frame["_subject_key"] = frame["subject_id"].map(_subject_key)
    allowed = set(membership["_subject_key"])
    frame = frame.loc[frame["_subject_key"].isin(allowed)].copy()
    if frame.empty:
        raise OutcomeTransferEvaluationError(
            f"Outcome {outcome!r} has no rows in the pan-hematology membership."
        )
    if frame["_subject_key"].duplicated().any():
        raise OutcomeTransferEvaluationError(
            f"Outcome {outcome!r} has duplicate subject IDs after population filtering."
        )
    # The shared BONSAI eligibility functions intentionally operate on the
    # original subject_id column.  They are the same functions used by the
    # support preflight, preserving one definition of structural eligibility.
    if eligibility_path is not None:
        if not eligibility_path.exists():
            raise OutcomeTransferEvaluationError(
                f"Eligibility file for {outcome!r} does not exist: {eligibility_path}"
            )
        from opera.evaluation.cohort_flow import load_eligibility_frame

        frame = filter_outcome_eligibility(
            frame,
            load_eligibility_frame(eligibility_path),
            cohort=ALL_HEMATOLOGY,
            outcome_name=outcome,
            eligibility_scope="ascertainment",
        )
    frame = filter_registry_eligible_outcomes(
        frame,
        registry_start_date,
        cohort=ALL_HEMATOLOGY,
        outcome_name=outcome,
    )
    if frame.empty:
        raise OutcomeTransferEvaluationError(
            f"Outcome {outcome!r} has no structurally eligible patients."
        )
    competing: pd.DataFrame | None = None
    if competing_path is not None:
        if not competing_path.exists():
            raise OutcomeTransferEvaluationError(
                f"Competing-event file for {outcome!r} does not exist: {competing_path}"
            )
        competing = _read_table(competing_path).copy()
        missing = {"subject_id", "outcome_date"} - set(competing.columns)
        if missing:
            raise OutcomeTransferEvaluationError(
                f"Competing-event file for {outcome!r} is missing columns: {sorted(missing)}"
            )
        competing["_subject_key"] = competing["subject_id"].map(_subject_key)
        competing = competing.loc[competing["_subject_key"].isin(allowed)].copy()
    return frame, competing, start_hours


def build_target_labels(
    registry: Mapping[str, Any],
    *,
    target_outcome: str,
    horizon_days: int,
    membership: pd.DataFrame | None = None,
    split_keys: Mapping[str, str] | None = None,
) -> TargetLabelBundle:
    """Construct frozen-probe labels with the exact fixed-horizon semantics.

    Competing events remain explicit as ``event == 2`` and use the binary
    label generated by BONSAI (zero).  Patients censored before the horizon are
    never turned into controls: they are represented in ``raw_status`` but are
    excluded from ``labels`` exactly as in the support preflight.
    """
    if target_outcome not in registry["outcomes"]:
        raise OutcomeTransferEvaluationError(
            f"Unknown target outcome {target_outcome!r} in the canonical registry."
        )
    if isinstance(horizon_days, bool) or not isinstance(horizon_days, int) or horizon_days <= 0:
        raise OutcomeTransferEvaluationError("horizon_days must be a positive integer.")
    membership = membership.copy() if membership is not None else _load_membership(registry)
    split_keys = dict(split_keys or _split_keys(registry.get("split_contract", "")))
    # ``split_contract`` lives in the transfer manifest rather than the
    # registry in production, so callers normally supply split_keys.  The
    # explicit error below prevents accidental application of unknown splits.
    if set(split_keys) != {"train", "tuning", "held_out"}:
        raise OutcomeTransferEvaluationError(
            "split_keys must map exactly train, tuning, and held_out."
        )
    frame, competing, start_hours = _filtered_outcome_frame(registry, target_outcome, membership)
    frame = frame.merge(
        membership[["_subject_key", "cohort_grouped"]],
        on="_subject_key",
        how="left",
        validate="one_to_one",
    )
    if frame["cohort_grouped"].isna().any():
        raise OutcomeTransferEvaluationError(
            f"Outcome {target_outcome!r} lost group membership during label construction."
        )
    rows: list[dict[str, Any]] = []
    raw_rows: list[dict[str, Any]] = []
    for canonical_split, source_split in split_keys.items():
        split_frame = frame.loc[frame["split"].astype(str) == str(source_split)].copy()
        if split_frame.empty:
            continue
        raw = binarize_outcomes(
            split_frame,
            n_hours_start_include=start_hours,
            n_hours_end_include=horizon_days * 24,
            require_min_followup=False,
            competing_event_df=competing,
        )
        retained = binarize_outcomes(
            split_frame,
            n_hours_start_include=start_hours,
            n_hours_end_include=horizon_days * 24,
            require_min_followup=True,
            competing_event_df=competing,
        )
        metadata = split_frame.set_index("_subject_key")
        # binarize_outcomes uses integer IDs, while inputs may be pandas or
        # NumPy scalar types.  Map through the stable string key used by all
        # embedding parity checks.
        for subject_id, record in raw.items():
            key = _subject_key(subject_id)
            if key not in metadata.index:
                raise OutcomeTransferEvaluationError(
                    "Binarized subject IDs do not map back to the outcome frame; "
                    "subject IDs must be compatible with BONSAI outcome binarization."
                )
            row = metadata.loc[key]
            raw_rows.append(
                {
                    "subject_id": row["subject_id"],
                    "_subject_key": key,
                    "split": canonical_split,
                    "cohort_grouped": str(row["cohort_grouped"]),
                    "retained": key in {_subject_key(value) for value in retained},
                    "label": int(record["label"]),
                    "event": int(record.get("event", 0)),
                    "time_days": float(record.get("time_days", np.nan)),
                }
            )
        for subject_id, record in retained.items():
            key = _subject_key(subject_id)
            row = metadata.loc[key]
            rows.append(
                {
                    "subject_id": row["subject_id"],
                    "_subject_key": key,
                    "split": canonical_split,
                    "cohort_grouped": str(row["cohort_grouped"]),
                    "label": int(record["label"]),
                    "event": int(record.get("event", 0)),
                    "time_days": float(record.get("time_days", np.nan)),
                    "censor_abspos": record.get("censor_abspos"),
                }
            )
    labels = pd.DataFrame(rows)
    raw_status = pd.DataFrame(raw_rows)
    if labels.empty:
        raise OutcomeTransferEvaluationError(
            f"Target {target_outcome!r} has no fixed-horizon labels at {horizon_days} days."
        )
    if labels.duplicated(["_subject_key", "split"]).any():
        raise OutcomeTransferEvaluationError(
            f"Target {target_outcome!r} produced duplicate fixed-horizon labels."
        )
    return TargetLabelBundle(
        labels=labels.sort_values(["split", "_subject_key"]).reset_index(drop=True),
        raw_status=raw_status.sort_values(["split", "_subject_key"]).reset_index(drop=True),
        target_outcome=target_outcome,
        horizon_days=horizon_days,
    )


def _group_mask(frame: pd.DataFrame, evaluation_group: str) -> pd.Series:
    if evaluation_group == ALL_HEMATOLOGY:
        return pd.Series(True, index=frame.index)
    return frame["cohort_grouped"].astype(str) == str(evaluation_group)


def label_count_summary(bundle: TargetLabelBundle, evaluation_group: str) -> dict[str, int | float]:
    """Return required count/censoring fields for a result or status row."""
    result: dict[str, int | float] = {}
    aliases = {"train": "train", "tuning": "tuning", "held_out": "test"}
    for split, alias in aliases.items():
        labels = bundle.labels.loc[(bundle.labels["split"] == split) & _group_mask(bundle.labels, evaluation_group)]
        raw = bundle.raw_status.loc[
            (bundle.raw_status["split"] == split) & _group_mask(bundle.raw_status, evaluation_group)
        ]
        events = int((labels["label"] == 1).sum())
        non_events = int((labels["label"] == 0).sum())
        competing = int((labels["event"] == 2).sum())
        censored = int((~raw["retained"].astype(bool)).sum()) if not raw.empty else 0
        result[f"n_{alias}"] = int(len(labels))
        result[f"n_{alias}_events"] = events
        result[f"n_{alias}_non_events"] = non_events
        result[f"n_{alias}_competing_events"] = competing
        result[f"n_{alias}_censored_before_horizon"] = censored
    result["n_competing_events"] = int(result["n_test_competing_events"])
    result["n_censored_before_horizon"] = int(result["n_test_censored_before_horizon"])
    result["prevalence"] = (
        float(result["n_test_events"] / result["n_test"])
        if result["n_test"]
        else float("nan")
    )
    return result


def _features_for(artifact: EmbeddingArtifact, subject_keys: Sequence[str]) -> np.ndarray:
    columns = embedding_columns(artifact.frame)
    indexed = artifact.frame.set_index("_subject_key")
    if not indexed.index.is_unique:
        raise OutcomeTransferEvaluationError(
            f"Embedding artefact {artifact.path} has duplicate subject IDs."
        )
    wanted = list(subject_keys)
    missing = [key for key in wanted if key not in indexed.index]
    if missing:
        raise DenominatorParityError(
            f"Representation {artifact.representation!r}, seed={artifact.seed} is missing "
            f"{len(missing)} eligible labelled patients; examples={missing[:10]}."
        )
    return indexed.loc[wanted, columns].to_numpy(dtype=float)


def assert_embedding_denominator_parity(
    artifacts: Sequence[EmbeddingArtifact], bundle: TargetLabelBundle
) -> None:
    """Fail closed if any compared frozen representation lacks labelled patients."""
    expected_by_split = {
        split: bundle.labels.loc[bundle.labels["split"] == split, "_subject_key"].tolist()
        for split in ("train", "tuning", "held_out")
    }
    for artifact in artifacts:
        for split, keys in expected_by_split.items():
            if keys:
                _features_for(artifact, keys)


def _binary_metrics(labels: np.ndarray, probabilities: np.ndarray) -> dict[str, float]:
    """Metrics requested by the transfer protocol, preserving undefined cells."""
    values: dict[str, float] = {
        "auroc": float("nan"),
        "auprc": float("nan"),
        "brier_score": float("nan"),
        "calibration_intercept": float("nan"),
        "calibration_slope": float("nan"),
    }
    if len(labels) == 0:
        return values
    values["brier_score"] = float(brier_score_loss(labels, probabilities))
    calibration = calibration_intercept_slope(labels, probabilities)
    values["calibration_intercept"] = float(calibration["calibration_intercept"])
    values["calibration_slope"] = float(calibration["calibration_slope"])
    if len(np.unique(labels)) >= 2:
        values["auroc"] = float(roc_auc_score(labels, probabilities))
        values["auprc"] = float(average_precision_score(labels, probabilities))
    return values


def fit_standardized_linear_probe(
    artifact: EmbeddingArtifact,
    bundle: TargetLabelBundle,
    *,
    seed: int,
    c_grid: Sequence[float] = DEFAULT_C_GRID,
) -> FrozenProbe:
    """Fit on train, choose C only on tuning, and never inspect held-out labels."""
    if not c_grid or any(not np.isfinite(value) or value <= 0 for value in c_grid):
        raise OutcomeTransferEvaluationError("c_grid must contain positive finite values.")
    train = bundle.labels.loc[bundle.labels["split"] == "train"].copy()
    tuning = bundle.labels.loc[bundle.labels["split"] == "tuning"].copy()
    if len(np.unique(train["label"])) < 2:
        raise ProbeSupportError(
            f"{bundle.target_outcome}: pan-hematology train split has fewer than two classes."
        )
    if len(np.unique(tuning["label"])) < 2:
        raise ProbeSupportError(
            f"{bundle.target_outcome}: pan-hematology tuning split has fewer than two classes; "
            "C cannot be selected without inspecting held-out labels."
        )
    train_x = _features_for(artifact, train["_subject_key"].tolist())
    tune_x = _features_for(artifact, tuning["_subject_key"].tolist())
    train_y = train["label"].to_numpy(dtype=int)
    tune_y = tuning["label"].to_numpy(dtype=int)
    best: FrozenProbe | None = None
    for c_value in sorted({float(value) for value in c_grid}):
        pipeline = Pipeline(
            [
                ("scale", StandardScaler()),
                (
                    "probe",
                    LogisticRegression(
                        C=c_value,
                        solver="lbfgs",
                        max_iter=4000,
                        random_state=seed,
                    ),
                ),
            ]
        )
        pipeline.fit(train_x, train_y)
        tuning_probability = pipeline.predict_proba(tune_x)[:, 1]
        tuning_auroc = float(roc_auc_score(tune_y, tuning_probability))
        # Iterating C in ascending order and using strict comparison gives a
        # deterministic, more regularized tie-breaker.
        if best is None or tuning_auroc > best.tuning_auroc:
            best = FrozenProbe(pipeline=pipeline, selected_c=c_value, tuning_auroc=tuning_auroc)
    assert best is not None
    return best


def held_out_predictions(
    artifact: EmbeddingArtifact,
    probe: FrozenProbe,
    bundle: TargetLabelBundle,
) -> pd.DataFrame:
    """Score the fixed held-out labels once; cohort views must reuse this table."""
    test = bundle.labels.loc[bundle.labels["split"] == "held_out"].copy()
    if test.empty:
        raise ProbeSupportError(
            f"{bundle.target_outcome}: no eligible pan-hematology held-out patients."
        )
    x_test = _features_for(artifact, test["_subject_key"].tolist())
    result = test.copy()
    result["probability"] = probe.pipeline.predict_proba(x_test)[:, 1]
    result["representation"] = artifact.representation
    result["seed"] = artifact.seed
    return result


def assert_prediction_denominator_parity(
    predictions: Mapping[str, pd.DataFrame],
    *,
    context: str = "",
) -> None:
    """Ensure patient IDs, labels, time, and competing-event status match exactly."""
    required = {"_subject_key", "label", "event", "time_days"}
    canonical: pd.DataFrame | None = None
    canonical_name: str | None = None
    for name, frame in predictions.items():
        missing = required - set(frame.columns)
        if missing:
            raise DenominatorParityError(
                f"Prediction frame {name!r} lacks parity columns: {sorted(missing)}."
            )
        if frame["_subject_key"].duplicated().any():
            raise DenominatorParityError(
                f"Prediction frame {name!r} has duplicate held-out patient IDs."
            )
        current = (
            frame[["_subject_key", "label", "event", "time_days"]]
            .sort_values("_subject_key")
            .reset_index(drop=True)
        )
        if canonical is None:
            canonical = current
            canonical_name = name
            continue
        if len(current) != len(canonical) or not current["_subject_key"].equals(canonical["_subject_key"]):
            raise DenominatorParityError(
                f"Held-out patient denominator mismatch between {canonical_name!r} and {name!r}"
                + (f" for {context}." if context else ".")
            )
        for column in ("label", "event"):
            if not current[column].equals(canonical[column]):
                raise DenominatorParityError(
                    f"Held-out {column} mismatch between {canonical_name!r} and {name!r}"
                    + (f" for {context}." if context else ".")
                )
        if not np.allclose(
            current["time_days"].to_numpy(dtype=float),
            canonical["time_days"].to_numpy(dtype=float),
            equal_nan=True,
        ):
            raise DenominatorParityError(
                f"Held-out event-time mismatch between {canonical_name!r} and {name!r}"
                + (f" for {context}." if context else ".")
            )


def _family_lookup(plan: Mapping[str, Any]) -> dict[str, str]:
    raw = plan.get("outcome_families")
    if not isinstance(raw, Mapping):
        raise OutcomeTransferEvaluationError("Resolved transfer plan lacks outcome_families.")
    return {str(outcome): str(family) for outcome, family in raw.items()}


def _condition_metadata(
    plan: Mapping[str, Any],
    representation: str,
    target: str,
) -> dict[str, Any]:
    lookup = _family_lookup(plan)
    if target not in lookup:
        raise OutcomeTransferEvaluationError(f"Target {target!r} lacks a canonical family.")
    if representation == DAPT_REPRESENTATION:
        included: list[str] = []
        excluded: list[str] = []
        transfer_level = NO_OUTCOME_GUIDED_ADAPTATION
    else:
        condition = plan["conditions"][representation]
        included = list(condition["training_outcomes"])
        excluded = list(condition["training_excluded_outcomes"])
        transfer_level = str(condition["transfer_level"])
    target_family = lookup[target]
    lower_grade = target.replace("_g3plus", "_g2plus") if target.endswith("_g3plus") else None
    return {
        "target_family": target_family,
        "transfer_level": transfer_level,
        "direct_target_seen": bool(target in included),
        "same_family_seen": bool(
            any(item != target and lookup.get(item) == target_family for item in included)
        ),
        "matched_lower_grade_seen": bool(lower_grade and lower_grade in included),
        "included_outcome_count": len(included),
        "excluded_outcome_count": len(excluded),
    }


def _result_rows(
    *,
    plan: Mapping[str, Any],
    representation: str,
    artifact: EmbeddingArtifact,
    condition_name: str,
    target: str,
    horizon_days: int,
    evaluation_level: str,
    evaluation_group: str,
    prediction_frame: pd.DataFrame,
    bundle: TargetLabelBundle,
    selected_c: float,
    tuning_auroc: float,
) -> list[dict[str, Any]]:
    group = prediction_frame.loc[_group_mask(prediction_frame, evaluation_group)].copy()
    labels = group["label"].to_numpy(dtype=int)
    probability = group["probability"].to_numpy(dtype=float)
    metrics = _binary_metrics(labels, probability)
    counts = label_count_summary(bundle, evaluation_group)
    metadata = _condition_metadata(plan, representation, target)
    base = {
        "condition": representation,
        "comparison_condition": condition_name,
        "seed": int(artifact.seed),
        "target_outcome": target,
        "target_family": metadata["target_family"],
        "transfer_level": metadata["transfer_level"],
        "primary_horizon_days": int(horizon_days),
        "evaluation_level": evaluation_level,
        "evaluation_group": evaluation_group,
        "direct_target_seen": metadata["direct_target_seen"],
        "same_family_seen": metadata["same_family_seen"],
        "matched_lower_grade_seen": metadata["matched_lower_grade_seen"],
        "included_outcome_count": metadata["included_outcome_count"],
        "excluded_outcome_count": metadata["excluded_outcome_count"],
        "checkpoint_hash": artifact.checkpoint_hash,
        "checkpoint_hash_source": artifact.checkpoint_hash_source,
        "embedding_artifact_hash": artifact.artifact_hash,
        "registry_hash": plan["registry_hash"],
        "manifest_hash": plan["manifest_hash"],
        "test_denominator_hash": patient_id_hash(group["subject_id"].tolist()),
        "selected_probe_c": float(selected_c),
        "tuning_auroc_for_selection": float(tuning_auroc),
        "probe_type": "standardized_logistic_regression",
        "encoder_frozen": True,
        "n_test_metric_rows": int(len(group)),
        **counts,
    }
    return [{**base, "metric": metric, "value": value} for metric, value in metrics.items()]


def _needed_representation_keys(plan: Mapping[str, Any]) -> set[tuple[str, int]]:
    keys: set[tuple[str, int]] = set()
    for seed in plan["seeds"]:
        keys.add((DAPT_REPRESENTATION, int(seed)))
        keys.add(("opera_full", int(seed)))
        for name in plan["conditions"]:
            if name != "opera_full":
                keys.add((name, int(seed)))
    return keys


def _has_nonempty_provenance(value: Any) -> bool:
    """Return whether a required checkpoint-origin field is meaningful."""
    return (
        isinstance(value, str)
        and bool(value.strip())
        and value.strip().lower() not in {"none", "null"}
    )


def _validate_artifact_inventory(
    plan: Mapping[str, Any], artifacts: Mapping[tuple[str, int], EmbeddingArtifact]
) -> None:
    expected = _needed_representation_keys(plan)
    supplied = set(artifacts)
    missing = sorted(expected - supplied)
    if missing:
        raise OutcomeTransferEvaluationError(
            "Frozen-probe evaluation is missing required embedding artefacts: "
            f"{missing}. Provide one DAPT, full OPERA, and relevant ablation embedding "
            "artefact per canonical seed."
        )
    unexpected = sorted(supplied - expected)
    if unexpected:
        raise OutcomeTransferEvaluationError(
            f"Embedding artefacts name unsupported representation/seed cells: {unexpected}."
        )
    for (representation, seed), artifact in artifacts.items():
        if artifact.representation != representation or artifact.seed != seed:
            raise OutcomeTransferEvaluationError(
                "Embedding artifact mapping key does not match its declared representation/seed."
            )
        metadata = artifact.metadata
        if artifact.metadata_path is None:
            raise OutcomeTransferEvaluationError(
                f"{representation}/seed_{seed} has no embedding metadata sidecar. "
                "Run extract_outcome_transfer_embeddings first, or pass its verified "
                "metadata sidecar explicitly."
            )
        if "checkpoint_hash" not in metadata:
            raise OutcomeTransferEvaluationError(
                f"{representation}/seed_{seed} metadata lacks checkpoint_hash. "
                "Frozen representations must remain traceable to a checkpoint."
            )
        if representation == DAPT_REPRESENTATION:
            # DAPT is a single existing baseline that may be reused across
            # OPERA seeds, but it cannot be an arbitrary frozen encoder
            # relabelled as ``dapt``.  Extraction stamps this identity after
            # validating the source checkpoint itself.
            if metadata.get("condition") != DAPT_REPRESENTATION:
                raise OutcomeTransferEvaluationError(
                    f"dapt/seed_{seed} metadata condition must be 'dapt'."
                )
            if metadata.get("training_stage") != "hematology_domain_adaptation":
                raise OutcomeTransferEvaluationError(
                    f"dapt/seed_{seed} metadata must identify a "
                    "hematology_domain_adaptation checkpoint."
                )
            if not _has_nonempty_provenance(metadata.get("source_checkpoint")):
                raise OutcomeTransferEvaluationError(
                    f"dapt/seed_{seed} metadata lacks a non-empty source_checkpoint."
                )
            # DAPT is not retrained for this experiment, but its frozen
            # *embedding artifact* is target/split dependent because the
            # extractor censors at the verified common prediction origin.
            # Require the same immutable analysis context as the OPERA
            # artifacts before allowing one DAPT table to serve all seeds.
            for field in (
                "registry_hash",
                "manifest_hash",
                "base_contrastive_config_hash",
                "split_contract",
                "split_contract_hash",
            ):
                if metadata.get(field) != plan[field]:
                    raise OutcomeTransferEvaluationError(
                        f"dapt/seed_{seed} metadata {field} does not match "
                        "the resolved transfer plan."
                    )
            continue
        expected = plan["conditions"][representation]
        expected_values: Mapping[str, Any] = {
            "training_stage": "opera_contrastive_adaptation",
            "condition": representation,
            "transfer_level": expected["transfer_level"],
            "seed": seed,
            "registry_hash": plan["registry_hash"],
            "manifest_hash": plan["manifest_hash"],
            "base_contrastive_config_hash": plan[
                "base_contrastive_config_hash"
            ],
            "included_outcomes": list(expected["training_outcomes"]),
            "excluded_outcomes": list(expected["training_excluded_outcomes"]),
            "evaluation_outcomes": list(expected["evaluation_outcomes"]),
            "related_retained_outcomes": list(expected["related_retained_outcomes"]),
            "direct_dependencies_excluded": list(
                expected["direct_dependencies_excluded"]
            ),
            "selection_outcomes": list(expected["training_outcomes"]),
            "split_contract": plan["split_contract"],
            "split_contract_hash": plan["split_contract_hash"],
        }
        missing_metadata = [field for field in expected_values if field not in metadata]
        if missing_metadata:
            raise OutcomeTransferEvaluationError(
                f"{representation}/seed_{seed} metadata is incomplete; missing "
                f"{sorted(missing_metadata)}."
            )
        for field, expected_value in expected_values.items():
            if metadata[field] != expected_value:
                raise OutcomeTransferEvaluationError(
                    f"{representation}/seed_{seed} metadata {field} does not match "
                    "the resolved transfer plan."
                )
        if not _has_nonempty_provenance(metadata.get("source_dapt_checkpoint")):
            raise OutcomeTransferEvaluationError(
                f"{representation}/seed_{seed} metadata lacks a non-empty "
                "source_dapt_checkpoint."
            )
        outcome_set = metadata.get("outcome_set")
        expected_outcome_set = list(expected["training_outcomes"])
        if (
            not isinstance(outcome_set, (list, tuple))
            or set(outcome_set) != set(expected_outcome_set)
            or len(outcome_set) != len(expected_outcome_set)
        ):
            raise OutcomeTransferEvaluationError(
                f"{representation}/seed_{seed} metadata outcome_set does not match "
                "the resolved training outcome panel."
            )


def _prediction_rows(
    frame: pd.DataFrame,
    *,
    condition_name: str,
    target: str,
    horizon_days: int,
    plan: Mapping[str, Any],
    artifact: EmbeddingArtifact,
) -> pd.DataFrame:
    metadata = _condition_metadata(plan, artifact.representation, target)
    result = frame.copy()
    result["condition"] = artifact.representation
    result["comparison_condition"] = condition_name
    result["seed"] = artifact.seed
    result["target_outcome"] = target
    result["target_family"] = metadata["target_family"]
    result["transfer_level"] = metadata["transfer_level"]
    result["primary_horizon_days"] = horizon_days
    result["direct_target_seen"] = metadata["direct_target_seen"]
    result["same_family_seen"] = metadata["same_family_seen"]
    result["matched_lower_grade_seen"] = metadata["matched_lower_grade_seen"]
    result["included_outcome_count"] = metadata["included_outcome_count"]
    result["excluded_outcome_count"] = metadata["excluded_outcome_count"]
    result["registry_hash"] = plan["registry_hash"]
    result["manifest_hash"] = plan["manifest_hash"]
    result["checkpoint_hash"] = artifact.checkpoint_hash
    result["checkpoint_hash_source"] = artifact.checkpoint_hash_source
    result["embedding_artifact_hash"] = artifact.artifact_hash
    return result


def evaluate_frozen_transfer_probes(
    plan: Mapping[str, Any],
    *,
    registry: Mapping[str, Any],
    artifacts: Mapping[tuple[str, int], EmbeddingArtifact],
    c_grid: Sequence[float] = DEFAULT_C_GRID,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run all frozen pan-hematology transfer probes in a resolved plan.

    Returns ``(results, predictions, probe_status, failures)``.  Unsupported
    targets (for example a tuning split with only one class) are recorded as
    failures rather than silently dropped.  Denominator mismatches always
    raise :class:`DenominatorParityError` and stop the comparison.
    """
    _validate_artifact_inventory(plan, artifacts)
    membership = _load_membership(registry)
    split_keys = _split_keys(plan["split_contract"])
    results: list[dict[str, Any]] = []
    saved_predictions: list[pd.DataFrame] = []
    statuses: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    groups = sorted(membership["cohort_grouped"].unique().tolist())

    for condition_name, condition in plan["conditions"].items():
        if condition_name == "opera_full":
            continue
        if bool(condition.get("launch_blocked", False)):
            raise OutcomeTransferEvaluationError(
                f"{condition_name} is launch-blocked: {condition.get('launch_blocked_reason')}"
            )
        horizon = int(condition["primary_horizon_days"])
        for target in condition["evaluation_outcomes"]:
            bundle = build_target_labels(
                registry,
                target_outcome=str(target),
                horizon_days=horizon,
                membership=membership,
                split_keys=split_keys,
            )
            for seed in plan["seeds"]:
                seed = int(seed)
                representation_names = (DAPT_REPRESENTATION, condition_name, "opera_full")
                cell_artifacts = [artifacts[(name, seed)] for name in representation_names]
                assert_embedding_denominator_parity(cell_artifacts, bundle)
                prediction_by_representation: dict[str, pd.DataFrame] = {}
                for artifact in cell_artifacts:
                    try:
                        probe = fit_standardized_linear_probe(
                            artifact, bundle, seed=seed, c_grid=c_grid
                        )
                        predictions = held_out_predictions(artifact, probe, bundle)
                    except ProbeSupportError as exc:
                        statuses.append(
                            {
                                "condition": artifact.representation,
                                "comparison_condition": condition_name,
                                "seed": seed,
                                "target_outcome": target,
                                "primary_horizon_days": horizon,
                                "status": "unsupported_label_support",
                                "reason": str(exc),
                                **label_count_summary(bundle, ALL_HEMATOLOGY),
                            }
                        )
                        failures.append(
                            {
                                "stage": "frozen_probe",
                                "condition": artifact.representation,
                                "comparison_condition": condition_name,
                                "seed": seed,
                                "target_outcome": target,
                                "primary_horizon_days": horizon,
                                "failure_type": "unsupported_label_support",
                                "message": str(exc),
                            }
                        )
                        continue
                    prediction_by_representation[artifact.representation] = predictions
                    statuses.append(
                        {
                            "condition": artifact.representation,
                            "comparison_condition": condition_name,
                            "seed": seed,
                            "target_outcome": target,
                            "primary_horizon_days": horizon,
                            "status": "completed",
                            "reason": "",
                            "selected_probe_c": probe.selected_c,
                            "tuning_auroc_for_selection": probe.tuning_auroc,
                            "probe_type": "standardized_logistic_regression",
                            "encoder_frozen": True,
                            **label_count_summary(bundle, ALL_HEMATOLOGY),
                        }
                    )
                    for level, evaluation_groups in (
                        ("pan_hematology", [ALL_HEMATOLOGY]),
                        ("cohort_grouped", groups),
                    ):
                        for group in evaluation_groups:
                            results.extend(
                                _result_rows(
                                    plan=plan,
                                    representation=artifact.representation,
                                    artifact=artifact,
                                    condition_name=condition_name,
                                    target=str(target),
                                    horizon_days=horizon,
                                    evaluation_level=level,
                                    evaluation_group=group,
                                    prediction_frame=predictions,
                                    bundle=bundle,
                                    selected_c=probe.selected_c,
                                    tuning_auroc=probe.tuning_auroc,
                                )
                            )
                    saved_predictions.append(
                        _prediction_rows(
                            predictions,
                            condition_name=condition_name,
                            target=str(target),
                            horizon_days=horizon,
                            plan=plan,
                            artifact=artifact,
                        )
                    )
                if len(prediction_by_representation) == len(representation_names):
                    assert_prediction_denominator_parity(
                        prediction_by_representation,
                        context=f"{condition_name}/{target}/seed_{seed}",
                    )
    prediction_frame = (
        pd.concat(saved_predictions, ignore_index=True)
        if saved_predictions
        else pd.DataFrame()
    )
    return (
        _with_schema(pd.DataFrame(results), TRANSFER_RESULT_COLUMNS),
        _with_schema(prediction_frame, TRANSFER_PREDICTION_COLUMNS),
        _with_schema(pd.DataFrame(statuses), TRANSFER_PROBE_STATUS_COLUMNS),
        _with_schema(pd.DataFrame(failures), TRANSFER_FAILURE_COLUMNS),
    )


def write_frozen_probe_outputs(
    output_dir: str | Path,
    *,
    results: pd.DataFrame,
    predictions: pd.DataFrame,
    probe_status: pd.DataFrame,
    failures: pd.DataFrame,
) -> dict[str, Path]:
    """Write the reproducible evaluator artefacts expected by aggregation."""
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    paths = {
        "transfer_results": destination / "transfer_results.csv",
        "transfer_predictions": destination / "transfer_predictions.parquet",
        "transfer_probe_status": destination / "transfer_probe_status.csv",
        "transfer_failures": destination / "transfer_failures.csv",
    }
    _with_schema(results, TRANSFER_RESULT_COLUMNS).to_csv(
        paths["transfer_results"], index=False
    )
    _with_schema(predictions, TRANSFER_PREDICTION_COLUMNS).to_parquet(
        paths["transfer_predictions"], index=False
    )
    _with_schema(probe_status, TRANSFER_PROBE_STATUS_COLUMNS).to_csv(
        paths["transfer_probe_status"], index=False
    )
    _with_schema(failures, TRANSFER_FAILURE_COLUMNS).to_csv(
        paths["transfer_failures"], index=False
    )
    return paths


def resolve_registry(path: str | Path) -> dict[str, Any]:
    """Public tiny wrapper used by the transfer-only CLI and tests."""
    return load_registry(_resolve_repository_or_cwd_path(path))
