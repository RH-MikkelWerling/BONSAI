"""Validated inputs for hierarchical natural-rarity analyses.

This module deliberately contains no PyMC imports. It discovers standard
OPERA result/prediction artifacts, reconstructs the canonical fixed-horizon
population encoded by each prediction artifact, requires exact patient and
label parity between models, and produces paired patient-bootstrap deltas.
The resulting tables are therefore usable without the optional Bayesian
dependencies and are auditable before a model is fitted.
"""

from __future__ import annotations

import hashlib
import json
from itertools import combinations
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)

from opera.config_contracts import load_sweep_config
from opera.evaluation.cohort_flow import eligibility_file_path
from opera.evaluation.cohorts import population_subject_ids
from opera.evaluation.rarity import LOWER_IS_BETTER
from opera.evaluation.tasks import (
    competing_outcome_file_path,
    normalize_outcome_config,
    outcome_file_path,
)
from opera.functional.outcomes import resolve_registry_start_date
from opera.run.evaluate_predictions import outcome_window_size_metadata


PAIR_KEYS = (
    "cohort",
    "outcome",
    "outcome_window_hours",
    "split",
    "seed",
    "evaluation_subset",
)
REQUIRED_ARTIFACT_COLUMNS = {
    "model_family",
    "cohort",
    "outcome",
    "seed",
    "prediction_path",
}
SUPPORTED_METRICS = (
    "auroc",
    "auprc",
    "pr_skill",
    "brier_score",
    "brier_skill",
    "log_loss",
)


def _read_result_file(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".jsonl":
        rows = [json.loads(line) for line in path.read_text().splitlines() if line]
        return pd.DataFrame(rows)
    return pd.read_csv(path)


def discover_prediction_artifacts(
    results_root: str | Path,
    *,
    evaluation_subset: str = "full",
) -> pd.DataFrame:
    """Discover prediction artifacts through their standard result metadata.

    Result metadata is the source of truth; directory names are never parsed
    to infer cohort, outcome, model, or seed. Ambiguous duplicate keys fail.
    """
    root = Path(results_root)
    if not root.exists():
        raise FileNotFoundError(f"Results root does not exist: {root}")

    rows: list[dict] = []
    result_paths = sorted(root.rglob("result.csv"))
    if not result_paths:
        result_paths = sorted(root.rglob("result.jsonl"))
    for result_path in result_paths:
        metadata = _read_result_file(result_path)
        prediction_path = result_path.parent / "predictions.npz"
        if not prediction_path.exists():
            continue
        for record in metadata.to_dict("records"):
            if str(record.get("evaluation_subset", "full")) != evaluation_subset:
                continue
            if str(record.get("split", "held_out")) != "held_out":
                continue
            record["prediction_path"] = str(prediction_path.resolve())
            record["result_path"] = str(result_path.resolve())
            rows.append(record)

    frame = pd.DataFrame(rows)
    if frame.empty:
        raise FileNotFoundError(
            f"No standard prediction/result artifact pairs found under {root}."
        )
    missing = REQUIRED_ARTIFACT_COLUMNS.difference(frame.columns)
    if missing:
        raise ValueError(f"Result artifacts are missing columns: {sorted(missing)}")

    available_keys = [key for key in PAIR_KEYS if key in frame.columns]
    duplicate_keys = ["model_family", *available_keys]
    duplicates = frame.duplicated(duplicate_keys, keep=False)
    if duplicates.any():
        examples = frame.loc[duplicates, duplicate_keys].head(10).to_dict("records")
        raise ValueError(
            "Duplicate prediction artifacts exist for the same model/task key: "
            f"{examples}"
        )
    return frame.reset_index(drop=True)


def load_binary_prediction_artifact(path: str | Path) -> pd.DataFrame:
    """Load the canonical fixed-horizon rows from a prediction NPZ.

    Neural evaluation artifacts may contain the wider survival cohort and a
    ``binary_mask``. Precomputed fixed-horizon artifacts contain only eligible
    rows. Risk-score-only survival artifacts are rejected.
    """
    source = Path(path)
    with np.load(source, allow_pickle=False) as data:
        required = {"subject_ids", "labels", "probabilities"}
        missing = required.difference(data.files)
        if missing:
            raise ValueError(
                f"Binary prediction artifact {source} is missing {sorted(missing)}. "
                "Risk-score-only survival artifacts cannot enter this analysis."
            )
        subject_ids = np.asarray(data["subject_ids"])
        labels = np.asarray(data["labels"]).reshape(-1)
        probabilities = np.asarray(data["probabilities"]).reshape(-1)
        if "binary_mask" in data.files:
            mask = np.asarray(data["binary_mask"]).astype(bool).reshape(-1)
            if len(mask) != len(subject_ids):
                raise ValueError(f"binary_mask length mismatch in {source}.")
            subject_ids = subject_ids[mask]
            labels = labels[mask]
            probabilities = probabilities[mask]

    if not (len(subject_ids) == len(labels) == len(probabilities)):
        raise ValueError(f"Prediction arrays have inconsistent lengths in {source}.")
    frame = pd.DataFrame(
        {
            "subject_id": subject_ids,
            "label": labels.astype(int),
            "probability": probabilities.astype(float),
        }
    )
    if frame.empty:
        raise ValueError(
            f"Prediction artifact contains no binary-eligible rows: {source}"
        )
    if frame["subject_id"].duplicated().any():
        raise ValueError(
            f"Prediction artifact contains duplicate subject IDs: {source}"
        )
    if not set(frame["label"].unique()).issubset({0, 1}):
        raise ValueError(f"Binary labels outside {{0, 1}} in {source}.")
    if not np.isfinite(frame["probability"]).all():
        raise ValueError(f"Non-finite probabilities in {source}.")
    if not frame["probability"].between(0.0, 1.0).all():
        raise ValueError(f"Probabilities outside [0, 1] in {source}.")
    return frame.sort_values("subject_id").reset_index(drop=True)


def assert_paired_prediction_parity(
    model: pd.DataFrame,
    comparator: pd.DataFrame,
    *,
    task_label: str,
) -> pd.DataFrame:
    """Return an exact one-to-one paired frame or fail closed."""
    merged = model.merge(
        comparator,
        on="subject_id",
        suffixes=("_model", "_comparator"),
        validate="one_to_one",
        how="outer",
        indicator=True,
    )
    if not (merged["_merge"] == "both").all():
        counts = merged["_merge"].value_counts().to_dict()
        raise ValueError(f"Patient parity failed for {task_label}: {counts}")
    if not np.array_equal(merged["label_model"], merged["label_comparator"]):
        raise ValueError(f"Label parity failed for {task_label}.")
    return (
        merged.drop(columns="_merge").sort_values("subject_id").reset_index(drop=True)
    )


def _metric_values(
    labels: np.ndarray,
    probabilities: np.ndarray,
    metrics: Sequence[str],
) -> dict[str, float]:
    prevalence = float(labels.mean())
    result: dict[str, float] = {}
    if "auroc" in metrics:
        result["auroc"] = float(roc_auc_score(labels, probabilities))
    if "auprc" in metrics or "pr_skill" in metrics:
        auprc = float(average_precision_score(labels, probabilities))
        if "auprc" in metrics:
            result["auprc"] = auprc
        if "pr_skill" in metrics:
            result["pr_skill"] = (auprc - prevalence) / (1.0 - prevalence)
    if "brier_score" in metrics or "brier_skill" in metrics:
        brier = float(brier_score_loss(labels, probabilities))
        if "brier_score" in metrics:
            result["brier_score"] = brier
        if "brier_skill" in metrics:
            reference = float(
                brier_score_loss(labels, np.full(len(labels), prevalence))
            )
            result["brier_skill"] = 1.0 - brier / reference
    if "log_loss" in metrics:
        result["log_loss"] = float(log_loss(labels, probabilities, labels=[0, 1]))
    return result


def _metric_differences(
    frame: pd.DataFrame,
    metrics: Sequence[str],
) -> dict[str, float]:
    labels = frame["label_model"].to_numpy(dtype=int)
    model_values = _metric_values(
        labels,
        frame["probability_model"].to_numpy(dtype=float),
        metrics,
    )
    comparator_values = _metric_values(
        labels,
        frame["probability_comparator"].to_numpy(dtype=float),
        metrics,
    )
    return {
        metric: (
            comparator_values[metric] - model_values[metric]
            if metric in LOWER_IS_BETTER
            else model_values[metric] - comparator_values[metric]
        )
        for metric in metrics
    }


def paired_bootstrap_differences(
    paired: pd.DataFrame,
    *,
    n_bootstrap: int,
    seed: int,
    metrics: Sequence[str] = SUPPORTED_METRICS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute paired deltas and patient-bootstrap uncertainty."""
    unknown = sorted(set(metrics).difference(SUPPORTED_METRICS))
    if unknown:
        raise ValueError(f"Unsupported metrics: {unknown}")
    if paired["label_model"].nunique() < 2:
        raise ValueError("Paired test cohort must contain both binary classes.")

    observed = _metric_differences(paired, metrics)
    labels = paired["label_model"].to_numpy(dtype=int)
    rng = np.random.default_rng(seed)
    draws = {metric: [] for metric in metrics}
    n = len(paired)
    for _ in range(int(n_bootstrap)):
        indices = rng.integers(0, n, size=n)
        if np.unique(labels[indices]).size < 2:
            continue
        values = _metric_differences(paired.iloc[indices], metrics)
        for metric in metrics:
            value = values[metric]
            if np.isfinite(value):
                draws[metric].append(float(value))

    summary_rows: list[dict] = []
    draw_rows: list[dict] = []
    for metric in metrics:
        values = np.asarray(draws[metric], dtype=float)
        summary_rows.append(
            {
                "metric": metric,
                "difference": float(observed[metric]),
                "difference_se": (
                    float(values.std(ddof=1)) if values.size > 1 else np.nan
                ),
                "difference_ci_lower": (
                    float(np.quantile(values, 0.025)) if values.size else np.nan
                ),
                "difference_ci_upper": (
                    float(np.quantile(values, 0.975)) if values.size else np.nan
                ),
                "n_bootstrap_valid": int(values.size),
            }
        )
        draw_rows.extend(
            {
                "metric": metric,
                "bootstrap_index": index,
                "difference": value,
            }
            for index, value in enumerate(values)
        )
    return pd.DataFrame(summary_rows), pd.DataFrame(draw_rows)


def _stable_seed(base_seed: int, values: Iterable[object]) -> int:
    payload = "|".join(map(str, values)).encode("utf-8")
    digest = int(hashlib.sha256(payload).hexdigest()[:8], 16)
    return int((int(base_seed) + digest) % (2**32 - 1))


def _validate_pair_metadata(model_row: pd.Series, comparator_row: pd.Series) -> None:
    for column in (
        "n_train",
        "n_events_train",
        "n_test",
        "n_events_test",
        "evaluation_regime",
    ):
        if column not in model_row.index or column not in comparator_row.index:
            continue
        left, right = model_row.get(column), comparator_row.get(column)
        if pd.isna(left) or pd.isna(right):
            continue
        if str(left) != str(right):
            raise ValueError(
                f"Model/comparator metadata mismatch for {column}: {left!r} != {right!r}."
            )


def build_paired_delta_tables(
    artifacts: pd.DataFrame,
    *,
    model_family: str,
    comparator_family: str,
    n_bootstrap: int,
    seed: int,
    metrics: Sequence[str] = SUPPORTED_METRICS,
    min_test_positive: int = 2,
    min_test_negative: int = 2,
    primary_test_positive: int = 10,
    primary_test_negative: int = 10,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, np.ndarray]]:
    """Build paired cell/seed deltas, bootstrap draws, and cell memberships."""
    missing = REQUIRED_ARTIFACT_COLUMNS.difference(artifacts.columns)
    if missing:
        raise ValueError(f"Artifact table is missing columns: {sorted(missing)}")
    key_columns = [key for key in PAIR_KEYS if key in artifacts.columns]
    model_rows = artifacts[artifacts["model_family"] == model_family].copy()
    comparator_rows = artifacts[artifacts["model_family"] == comparator_family].copy()
    if model_rows.empty or comparator_rows.empty:
        raise ValueError(
            f"Both {model_family!r} and {comparator_family!r} must have predictions."
        )
    paired_rows = model_rows.merge(
        comparator_rows,
        on=key_columns,
        how="inner",
        suffixes=("_model", "_comparator"),
        validate="one_to_one",
    )
    if paired_rows.empty:
        raise ValueError("No matched model/comparator prediction cells were found.")

    summaries: list[pd.DataFrame] = []
    all_draws: list[pd.DataFrame] = []
    memberships: dict[str, np.ndarray] = {}
    for _, pair in paired_rows.iterrows():
        key_values = tuple(pair[column] for column in key_columns)
        model_row = pd.Series(
            {
                column: pair.get(f"{column}_model")
                for column in artifacts.columns
                if column not in key_columns
            }
        )
        comparator_row = pd.Series(
            {
                column: pair.get(f"{column}_comparator")
                for column in artifacts.columns
                if column not in key_columns
            }
        )
        _validate_pair_metadata(model_row, comparator_row)
        label = "/".join(map(str, key_values))
        model_predictions = load_binary_prediction_artifact(
            pair["prediction_path_model"]
        )
        comparator_predictions = load_binary_prediction_artifact(
            pair["prediction_path_comparator"]
        )
        paired = assert_paired_prediction_parity(
            model_predictions,
            comparator_predictions,
            task_label=label,
        )
        n_positive = int(paired["label_model"].sum())
        n_negative = int(len(paired) - n_positive)
        if n_positive < min_test_positive or n_negative < min_test_negative:
            continue

        cell_columns = [
            column
            for column in ("cohort", "outcome", "outcome_window_hours")
            if column in pair.index
        ]
        cell_id = "|".join(str(pair[column]) for column in cell_columns)
        members = paired["subject_id"].to_numpy()
        previous = memberships.get(cell_id)
        if previous is not None and not np.array_equal(
            np.sort(previous), np.sort(members)
        ):
            raise ValueError(
                f"Test patient membership changed across seeds for {cell_id}."
            )
        memberships[cell_id] = members

        cell_seed = _stable_seed(seed, key_values)
        summary, draws = paired_bootstrap_differences(
            paired,
            n_bootstrap=n_bootstrap,
            seed=cell_seed,
            metrics=metrics,
        )
        common = {column: pair[column] for column in key_columns}
        common.update(
            {
                "cell_id": cell_id,
                "model_family": model_family,
                "comparator_family": comparator_family,
                "n_test_patients": int(len(paired)),
                "n_test_positive": n_positive,
                "n_test_negative": n_negative,
                "analysis_tier": (
                    "primary"
                    if n_positive >= primary_test_positive
                    and n_negative >= primary_test_negative
                    else "partial_pool_only"
                ),
            }
        )
        for column in (
            "n_train",
            "n_events_train",
            "prevalence_train",
            "outcome_family",
            "cohort_group",
        ):
            if column in model_row.index:
                common[column] = model_row.get(column)
        summaries.append(summary.assign(**common))
        all_draws.append(draws.assign(**common))

    if not summaries:
        raise ValueError("No paired cells met the minimum held-out class counts.")
    return (
        pd.concat(summaries, ignore_index=True),
        pd.concat(all_draws, ignore_index=True),
        memberships,
    )


def build_task_size_metadata(sweep_config: str | Path) -> pd.DataFrame:
    """Derive fixed-horizon train/validation/test counts canonically."""
    config = load_sweep_config(sweep_config).to_mapping()
    outcomes = normalize_outcome_config(config["outcomes"])
    rows: list[dict] = []
    for cohort_name, cohort_cfg in config["cohorts"].items():
        data_dir = cohort_cfg["data_dir"]
        population_path = cohort_cfg.get(
            "population_file", str(Path(data_dir) / "population_full.csv")
        )
        allowed_ids = population_subject_ids(
            population_path,
            cohort_fine_col=cohort_cfg.get("cohort_fine_col"),
            cohort_fine_value=cohort_cfg.get("cohort_fine_value"),
        )
        for outcome_name, outcome_cfg in outcomes.items():
            outcome_path = outcome_file_path(data_dir, outcome_name, outcome_cfg)
            if not Path(outcome_path).exists():
                continue
            eligibility = eligibility_file_path(
                data_dir,
                cohort_name,
                outcome_name,
                outcome_cfg,
            )
            competing = competing_outcome_file_path(data_dir, outcome_cfg)
            sizes = outcome_window_size_metadata(
                outcome_path,
                n_hours_start_include=outcome_cfg.get("n_hours_start_include", 1),
                n_hours_end_include=outcome_cfg.get("n_hours_end_include"),
                evaluation_regime="fixed_horizon",
                competing_outcome_path=competing,
                eligibility_path=str(eligibility) if eligibility else None,
                registry_start_date=resolve_registry_start_date(
                    cohort_cfg,
                    outcome_cfg,
                ),
                cohort=cohort_name,
                outcome_name=outcome_name,
                allowed_subject_ids=allowed_ids,
            )
            rows.append(
                {
                    "cohort": cohort_name,
                    "cohort_group": cohort_cfg.get("training_cohort", cohort_name),
                    "outcome": outcome_name,
                    "outcome_window_hours": outcome_cfg.get("n_hours_end_include"),
                    **sizes,
                }
            )
    return pd.DataFrame(rows)


def attach_task_metadata(
    artifacts: pd.DataFrame,
    task_metadata: pd.DataFrame,
) -> pd.DataFrame:
    """Attach canonical task-size metadata without silent many-to-many joins."""
    keys = [
        key
        for key in ("cohort", "outcome", "outcome_window_hours")
        if key in artifacts.columns and key in task_metadata.columns
    ]
    if not {"cohort", "outcome"}.issubset(keys):
        raise ValueError("Task metadata requires cohort and outcome columns.")
    if task_metadata.duplicated(keys).any():
        raise ValueError(f"Task metadata contains duplicate keys: {keys}")
    metadata_columns = [
        column
        for column in (
            "n_train",
            "n_events_train",
            "prevalence_train",
            "n_val",
            "n_events_val",
            "n_test",
            "n_events_test",
            "outcome_family",
            "cohort_group",
        )
        if column in task_metadata.columns
    ]
    existing = artifacts.drop(columns=[c for c in metadata_columns if c in artifacts])
    merged = existing.merge(
        task_metadata[[*keys, *metadata_columns]],
        on=keys,
        how="left",
        validate="many_to_one",
    )
    if merged["n_events_train"].isna().any():
        examples = merged.loc[
            merged["n_events_train"].isna(), ["cohort", "outcome"]
        ].head(10)
        raise ValueError(
            "Training event counts are missing for discovered artifacts: "
            f"{examples.to_dict('records')}"
        )
    return merged


def apply_outcome_families(
    frame: pd.DataFrame,
    mapping: Mapping[str, str] | None,
    *,
    require_complete: bool = False,
) -> pd.DataFrame:
    result = frame.copy()
    mapping = dict(mapping or {})
    result["outcome_family"] = result["outcome"].map(mapping).fillna("Other")
    if require_complete:
        unmapped = sorted(
            result.loc[result["outcome_family"] == "Other", "outcome"]
            .astype(str)
            .unique()
        )
        if unmapped:
            raise ValueError(
                "Outcome-family mapping is incomplete; add explicit entries for: "
                f"{unmapped}"
            )
    return result


def summarize_patient_overlap(
    memberships: Mapping[str, np.ndarray],
) -> tuple[dict, pd.DataFrame]:
    """Summarize repeated test-patient participation without exporting IDs."""
    cell_sets = {name: set(values.tolist()) for name, values in memberships.items()}
    patient_counts: dict[object, int] = {}
    for values in cell_sets.values():
        for subject_id in values:
            patient_counts[subject_id] = patient_counts.get(subject_id, 0) + 1
    counts = np.asarray(list(patient_counts.values()), dtype=int)
    summary = {
        "n_cells": len(cell_sets),
        "n_unique_test_patients": len(patient_counts),
        "fraction_patients_in_multiple_cells": (
            float((counts > 1).mean()) if counts.size else 0.0
        ),
        "median_cells_per_patient": float(np.median(counts)) if counts.size else 0.0,
        "max_cells_per_patient": int(counts.max()) if counts.size else 0,
    }
    pair_rows = []
    for left, right in combinations(sorted(cell_sets), 2):
        overlap = len(cell_sets[left] & cell_sets[right])
        if overlap == 0:
            continue
        union = len(cell_sets[left] | cell_sets[right])
        pair_rows.append(
            {
                "left_cell": left,
                "right_cell": right,
                "n_overlap": overlap,
                "jaccard": float(overlap / union) if union else 0.0,
            }
        )
    return summary, pd.DataFrame(pair_rows)
