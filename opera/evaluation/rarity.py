"""Patient-level design and statistics for OPERA rarity experiments.

The functions in this module deliberately sit above the canonical outcome and
evaluation-cohort builders.  They do not redefine labels: they turn those
labels into auditable task tables, nested patient samples, paired metrics, and
macro summaries for the synthetic and natural rarity analyses.
"""

from __future__ import annotations

import hashlib
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

from bonsai.functional.outcomes import validate_split_integrity
from opera.evaluation.cohorts import build_evaluation_cohorts
from opera.functional.cohort_groups import fine_to_grouped


METRIC_NAMES = (
    "auroc",
    "auprc",
    "pr_skill",
    "brier_score",
    "brier_skill",
    "log_loss",
)
LOWER_IS_BETTER = frozenset({"brier_score", "log_loss"})


def read_table(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if path.suffix.lower() in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    return pd.read_csv(path)


def patient_id_hash(subject_ids: Iterable[object]) -> str:
    """Return a stable SHA256 over the sorted unique patient identifiers."""
    values = sorted({str(value) for value in subject_ids})
    return hashlib.sha256("\n".join(values).encode("utf-8")).hexdigest()


def validate_patient_splits(
    outcomes: pd.DataFrame,
    split_contract: Mapping[str, object],
) -> dict:
    """Validate date boundaries and guarantee one split per patient."""
    report = validate_split_integrity(outcomes, **dict(split_contract))
    split_counts = outcomes.groupby("subject_id")["split"].nunique(dropna=True)
    repeated = split_counts[split_counts > 1]
    if not repeated.empty:
        report = dict(report)
        report["ok"] = False
        report["patients_in_multiple_splits"] = int(len(repeated))
        report["multiple_split_examples"] = [str(v) for v in repeated.index[:10]]
    if not report["ok"]:
        raise ValueError(f"Temporal split contract failed: {report}")
    return report


def _unique_patient_dates(outcomes: pd.DataFrame) -> pd.DataFrame:
    columns = ["subject_id", "split", "index_date"]
    frame = outcomes[columns].copy()
    frame["index_date"] = pd.to_datetime(frame["index_date"], errors="coerce")
    if frame["index_date"].isna().any():
        raise ValueError("Outcome index_date must be non-null for rarity sampling.")
    inconsistent = frame.groupby("subject_id").agg(
        n_splits=("split", "nunique"), n_dates=("index_date", "nunique")
    )
    bad = inconsistent[(inconsistent["n_splits"] > 1) | (inconsistent["n_dates"] > 1)]
    if not bad.empty:
        raise ValueError(
            "Rarity tasks require one prediction origin and split per patient; "
            f"inconsistent IDs={list(map(str, bad.index[:10]))}."
        )
    return frame.drop_duplicates("subject_id", keep="first")


def build_task_label_table(
    *,
    outcomes: pd.DataFrame | str | Path,
    population: pd.DataFrame | str | Path,
    cohort_fine: str,
    cohort_fine_col: str = "cohort_fine",
    splits: Sequence[str] = ("train", "tuning", "held_out"),
    n_hours_start_include: int = 1,
    n_hours_end_include: int | None = None,
    competing_outcomes: pd.DataFrame | str | Path | None = None,
    eligibility: pd.DataFrame | str | Path | None = None,
    registry_start_date=None,
    outcome_name: str | None = None,
) -> pd.DataFrame:
    """Create one patient-level label/status row per split for a fine task.

    Positive and negative rows come from the fixed-horizon cohort. Patients in
    the ascertainment cohort but not the fixed-horizon cohort are explicitly
    marked indeterminate. This preserves the canonical censoring semantics.
    """
    outcome_frame = outcomes.copy() if isinstance(outcomes, pd.DataFrame) else read_table(outcomes)
    population_frame = (
        population.copy() if isinstance(population, pd.DataFrame) else read_table(population)
    )
    required_population = {"subject_id", cohort_fine_col}
    missing = required_population - set(population_frame.columns)
    if missing:
        raise ValueError(f"Population is missing columns: {sorted(missing)}")
    if population_frame["subject_id"].duplicated().any():
        raise ValueError("Population must contain one row per subject_id.")

    allowed = set(
        population_frame.loc[
            population_frame[cohort_fine_col].astype(str) == str(cohort_fine),
            "subject_id",
        ]
    )
    if not allowed:
        raise ValueError(f"No population patients found for cohort_fine={cohort_fine!r}.")

    dates = _unique_patient_dates(outcome_frame).set_index("subject_id")
    rows: list[dict] = []
    for split in splits:
        cohorts = build_evaluation_cohorts(
            outcome_frame,
            split=split,
            n_hours_start_include=n_hours_start_include,
            n_hours_end_include=n_hours_end_include,
            competing_outcomes=competing_outcomes,
            eligibility=eligibility,
            registry_start_date=registry_start_date,
            cohort=cohort_fine,
            outcome_name=outcome_name,
            allowed_subject_ids=allowed,
        )
        fixed = cohorts.fixed_horizon.records
        survival = cohorts.survival.records
        for subject_id, record in survival.items():
            if subject_id in fixed:
                label = int(fixed[subject_id]["label"])
                status = "positive" if label == 1 else "negative"
            else:
                label = pd.NA
                status = "indeterminate"
            index_date = dates.loc[subject_id, "index_date"]
            rows.append(
                {
                    "subject_id": subject_id,
                    "split": split,
                    "index_date": pd.Timestamp(index_date),
                    "index_year": int(pd.Timestamp(index_date).year),
                    "label": label,
                    "label_status": status,
                    "cohort_fine": cohort_fine,
                    "cohort_grouped": fine_to_grouped(cohort_fine),
                    "outcome": outcome_name,
                    "horizon_hours": n_hours_end_include,
                    "time_days": float(record.get("time_days", np.nan)),
                    "event": int(record.get("event", -1)),
                }
            )
    result = pd.DataFrame(rows)
    if result.empty:
        return result
    if result.duplicated(["subject_id", "split"]).any():
        raise ValueError("Task label table contains duplicate patient/split rows.")
    return result.sort_values(["split", "subject_id"]).reset_index(drop=True)


def task_count_record(labels: pd.DataFrame) -> dict:
    """Summarize determinate and indeterminate patient counts by split."""
    out: dict[str, int | float] = {}
    aliases = {"train": "train", "tuning": "tuning", "held_out": "test"}
    for split, alias in aliases.items():
        group = labels[labels["split"] == split]
        counts = group["label_status"].value_counts()
        pos = int(counts.get("positive", 0))
        neg = int(counts.get("negative", 0))
        ind = int(counts.get("indeterminate", 0))
        out[f"n_{alias}_positive"] = pos
        out[f"n_{alias}_negative"] = neg
        out[f"n_{alias}_indeterminate"] = ind
        out[f"n_{alias}_determinate"] = pos + neg
        out[f"n_{alias}_total"] = pos + neg + ind
        out[f"{alias}_prevalence"] = pos / (pos + neg) if pos + neg else np.nan
    return out


def classify_task_eligibility(
    counts: Mapping[str, int | float],
    *,
    min_train_patients: int,
    min_train_positive: int,
    min_train_negative: int,
    primary_test_positive: int,
    primary_test_negative: int,
    aggregate_test_positive: int,
    aggregate_test_negative: int,
    max_test_indeterminate_fraction: float,
) -> dict:
    """Annotate count-based training eligibility and reporting tiers."""
    train_reasons = []
    if int(counts["n_train_determinate"]) < min_train_patients:
        train_reasons.append("insufficient_train_patients")
    if int(counts["n_train_positive"]) < min_train_positive:
        train_reasons.append("insufficient_train_positives")
    if int(counts["n_train_negative"]) < min_train_negative:
        train_reasons.append("insufficient_train_negatives")

    test_total = int(counts["n_test_total"])
    ind_fraction = (
        int(counts["n_test_indeterminate"]) / test_total if test_total else 1.0
    )
    pos = int(counts["n_test_positive"])
    neg = int(counts["n_test_negative"])
    minority_class = min(pos, neg)
    if pos >= primary_test_positive and neg >= primary_test_negative:
        tier = "primary"
        test_reason = ""
    elif pos >= aggregate_test_positive and neg >= aggregate_test_negative:
        tier = "aggregate_only"
        test_reason = "below_primary_test_counts"
    elif minority_class > 0:
        tier = "partial_pool_only"
        test_reason = "below_aggregate_test_counts"
    else:
        tier = "non_evaluable"
        test_reason = "missing_test_class"
    synthetic_reasons = [*train_reasons]
    if tier == "non_evaluable":
        synthetic_reasons.append(test_reason)
    return {
        "synthetic_eligible": not synthetic_reasons,
        "synthetic_exclusion_reason": ";".join(synthetic_reasons),
        "natural_viability_tier": tier,
        "natural_exclusion_reason": test_reason if tier == "non_evaluable" else "",
        "natural_reporting_reason": test_reason,
        "minority_class": int(minority_class),
        "test_indeterminate_fraction": ind_fraction,
    }


def _nested_patient_order(frame: pd.DataFrame, seed: int) -> list[object]:
    """Produce a reproducible, approximately stratified patient ordering."""
    determinate = frame[frame["label_status"].isin(["positive", "negative"])].copy()
    if determinate["subject_id"].duplicated().any():
        raise ValueError("Sampling frame must contain unique patients.")
    rng = np.random.default_rng(seed)
    determinate["_stratum"] = (
        determinate["label"].astype(int).astype(str)
        + "::"
        + determinate["index_year"].astype(int).astype(str)
    )
    parts = []
    for _, group in determinate.groupby("_stratum", sort=True):
        group = group.copy()
        order = rng.permutation(len(group))
        group["_within_rank"] = np.arange(len(group))[order.argsort()]
        group["_priority"] = (group["_within_rank"] + rng.random(len(group))) / len(group)
        parts.append(group)
    ordered = pd.concat(parts).sort_values(["_priority", "_stratum", "subject_id"])
    return ordered["subject_id"].tolist()


def resolve_sample_sizes(
    n_available: int,
    absolute_sizes: Sequence[int],
    percentage_sizes: Sequence[float] = (),
) -> list[int]:
    sizes = {int(value) for value in absolute_sizes if 0 < int(value) <= n_available}
    for fraction in percentage_sizes:
        if not 0 < float(fraction) <= 1:
            raise ValueError("Percentage sample sizes must lie in (0, 1].")
        sizes.add(max(1, min(n_available, int(round(n_available * float(fraction))))))
    sizes.add(n_available)
    return sorted(sizes)


def build_nested_sample_manifest(
    labels: pd.DataFrame,
    *,
    sizes: Sequence[int],
    seed: int,
    min_positive: int,
    min_negative: int,
    downsample_tuning: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build nested train/tuning patient manifests shared by every model."""
    train = labels[labels["split"] == "train"]
    tuning = labels[labels["split"] == "tuning"]
    train_order = _nested_patient_order(train, seed)
    tune_order = _nested_patient_order(tuning, seed + 1_000_003)
    n_train_total = len(train_order)
    records: list[dict] = []
    skipped: list[dict] = []
    previous: set[object] = set()
    for size in sorted(set(map(int, sizes))):
        if size <= 0 or size > n_train_total:
            skipped.append({"sample_size": size, "seed": seed, "reason": "infeasible_size"})
            continue
        train_ids = set(train_order[:size])
        selected = train[train["subject_id"].isin(train_ids)]
        pos = int((selected["label"] == 1).sum())
        neg = int((selected["label"] == 0).sum())
        if pos < min_positive or neg < min_negative:
            skipped.append(
                {
                    "sample_size": size,
                    "seed": seed,
                    "reason": "insufficient_sample_classes",
                    "n_positive": pos,
                    "n_negative": neg,
                }
            )
            continue
        if not previous.issubset(train_ids):
            raise AssertionError("Nested sampling invariant failed.")
        previous = train_ids
        tune_n = (
            min(len(tune_order), max(1, int(round(len(tune_order) * size / n_train_total))))
            if downsample_tuning and tune_order
            else len(tune_order)
        )
        for split, ids in (("train", train_order[:size]), ("tuning", tune_order[:tune_n])):
            for subject_id in ids:
                row = labels[(labels["split"] == split) & (labels["subject_id"] == subject_id)].iloc[0]
                records.append(
                    {
                        "sample_size": size,
                        "seed": seed,
                        "split": split,
                        "subject_id": subject_id,
                        "label": int(row["label"]),
                        "index_year": int(row["index_year"]),
                    }
                )
    manifest = pd.DataFrame(records)
    if not manifest.empty:
        hashes = (
            manifest.groupby(["sample_size", "seed", "split"])["subject_id"]
            .apply(patient_id_hash)
            .rename("patient_id_hash")
            .reset_index()
        )
        manifest = manifest.merge(hashes, on=["sample_size", "seed", "split"], how="left")
    return manifest, pd.DataFrame(skipped)


def validate_nested_manifest(manifest: pd.DataFrame) -> None:
    if manifest.empty:
        return
    if manifest.duplicated(["sample_size", "seed", "split", "subject_id"]).any():
        raise ValueError("Sample manifest contains duplicate patient rows.")
    for (seed, split), group in manifest.groupby(["seed", "split"]):
        previous: set[object] = set()
        for _, level in group.groupby("sample_size", sort=True):
            current = set(level["subject_id"])
            if not previous.issubset(current):
                raise ValueError(f"Samples are not nested for seed={seed}, split={split}.")
            previous = current


def binary_metric_values(labels: np.ndarray, probabilities: np.ndarray) -> dict[str, float]:
    labels = np.asarray(labels, dtype=int)
    probabilities = np.clip(np.asarray(probabilities, dtype=float), 1e-7, 1 - 1e-7)
    if len(labels) == 0 or len(np.unique(labels)) < 2:
        raise ValueError("Binary metrics require non-empty positive and negative classes.")
    prevalence = float(labels.mean())
    auprc = float(average_precision_score(labels, probabilities))
    brier = float(brier_score_loss(labels, probabilities))
    prevalence_predictions = np.full(len(labels), prevalence)
    brier_reference = float(brier_score_loss(labels, prevalence_predictions))
    return {
        "auroc": float(roc_auc_score(labels, probabilities)),
        "auprc": auprc,
        "pr_skill": (auprc - prevalence) / (1.0 - prevalence),
        "brier_score": brier,
        "brier_skill": 1.0 - brier / brier_reference if brier_reference > 0 else np.nan,
        "log_loss": float(log_loss(labels, probabilities, labels=[0, 1])),
    }


def _cluster_resample(frame: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    subject_ids = frame["subject_id"].drop_duplicates().to_numpy()
    sampled = rng.choice(subject_ids, size=len(subject_ids), replace=True)
    parts = [frame[frame["subject_id"] == subject_id] for subject_id in sampled]
    return pd.concat(parts, ignore_index=True)


def bootstrap_metric_table(
    predictions: pd.DataFrame,
    *,
    n_bootstrap: int,
    seed: int,
    ci: float = 0.95,
) -> pd.DataFrame:
    """Compute requested metrics and patient-cluster bootstrap intervals."""
    required = {"subject_id", "label", "probability"}
    missing = required - set(predictions.columns)
    if missing:
        raise ValueError(f"Predictions are missing columns: {sorted(missing)}")
    observed = binary_metric_values(
        predictions["label"].to_numpy(), predictions["probability"].to_numpy()
    )
    rng = np.random.default_rng(seed)
    draws = {metric: [] for metric in METRIC_NAMES}
    for _ in range(int(n_bootstrap)):
        sample = _cluster_resample(predictions, rng)
        if sample["label"].nunique() < 2:
            continue
        values = binary_metric_values(sample["label"].to_numpy(), sample["probability"].to_numpy())
        for metric, value in values.items():
            if np.isfinite(value):
                draws[metric].append(value)
    alpha = (1.0 - ci) / 2.0
    rows = []
    for metric, estimate in observed.items():
        values = np.asarray(draws[metric], dtype=float)
        rows.append(
            {
                "metric": metric,
                "estimate": estimate,
                "ci_lower": float(np.quantile(values, alpha)) if values.size else np.nan,
                "ci_upper": float(np.quantile(values, 1 - alpha)) if values.size else np.nan,
                "bootstrap_se": float(values.std(ddof=1)) if values.size > 1 else np.nan,
                "n_bootstrap_valid": int(values.size),
                "n_test_patients": int(predictions["subject_id"].nunique()),
            }
        )
    return pd.DataFrame(rows)


def assert_prediction_parity(predictions_by_model: Mapping[str, pd.DataFrame]) -> None:
    """Require identical patient clusters and labels across compared models."""
    reference_name = None
    reference = None
    for name, frame in predictions_by_model.items():
        pairs = frame[["subject_id", "label"]].drop_duplicates().sort_values("subject_id")
        if reference is None:
            reference_name, reference = name, pairs.reset_index(drop=True)
            continue
        if not reference.equals(pairs.reset_index(drop=True)):
            raise ValueError(
                f"Fixed-test parity failed between {reference_name!r} and {name!r}."
            )


def paired_bootstrap_difference(
    model: pd.DataFrame,
    comparator: pd.DataFrame,
    *,
    model_name: str,
    comparator_name: str,
    n_bootstrap: int,
    seed: int,
    ci: float = 0.95,
) -> pd.DataFrame:
    """Paired patient-cluster bootstrap differences for all rarity metrics."""
    left = model.copy()
    right = comparator.copy()
    left["_row"] = left.groupby("subject_id").cumcount()
    right["_row"] = right.groupby("subject_id").cumcount()
    merged = left.merge(
        right,
        on=["subject_id", "_row"],
        suffixes=("_model", "_comparator"),
        validate="one_to_one",
    )
    if len(merged) != len(left) or len(merged) != len(right):
        raise ValueError("Paired model predictions do not contain identical test rows.")
    if not np.array_equal(merged["label_model"], merged["label_comparator"]):
        raise ValueError("Paired model predictions have different labels.")

    def differences(frame: pd.DataFrame) -> dict[str, float]:
        a = binary_metric_values(frame["label_model"], frame["probability_model"])
        b = binary_metric_values(frame["label_model"], frame["probability_comparator"])
        return {
            metric: (b[metric] - a[metric] if metric in LOWER_IS_BETTER else a[metric] - b[metric])
            for metric in METRIC_NAMES
        }

    observed = differences(merged)
    rng = np.random.default_rng(seed)
    draws = {metric: [] for metric in METRIC_NAMES}
    for _ in range(int(n_bootstrap)):
        sample = _cluster_resample(merged, rng)
        if sample["label_model"].nunique() < 2:
            continue
        values = differences(sample)
        for metric, value in values.items():
            if np.isfinite(value):
                draws[metric].append(value)
    alpha = (1.0 - ci) / 2.0
    rows = []
    for metric, estimate in observed.items():
        values = np.asarray(draws[metric], dtype=float)
        rows.append(
            {
                "metric": metric,
                "model": model_name,
                "comparator": comparator_name,
                "difference": estimate,
                "difference_ci_lower": float(np.quantile(values, alpha)) if values.size else np.nan,
                "difference_ci_upper": float(np.quantile(values, 1 - alpha)) if values.size else np.nan,
                "difference_se": float(values.std(ddof=1)) if values.size > 1 else np.nan,
                "n_bootstrap_valid": int(values.size),
                "n_test_patients": int(merged["subject_id"].nunique()),
            }
        )
    return pd.DataFrame(rows)


def macro_synthetic_summary(
    metrics: pd.DataFrame,
    *,
    n_bootstrap: int = 2000,
    seed: int = 42,
) -> pd.DataFrame:
    """Task-level macro learning curves with hierarchical task/seed bootstrap."""
    if metrics.empty:
        return pd.DataFrame()
    group_cols = ["model", "sample_size", "metric"]
    if "comparator" in metrics.columns:
        group_cols.append("comparator")
    rows = []
    rng = np.random.default_rng(seed)
    task_cols = ["cohort_fine", "outcome"]
    for keys, group in metrics.groupby(group_cols, dropna=False):
        task_seed = group.groupby([*task_cols, "seed"], as_index=False)["estimate"].mean()
        task_mean = task_seed.groupby(task_cols, as_index=False)["estimate"].mean()
        tasks = list(task_seed.groupby(task_cols, sort=False))
        draws = []
        for _ in range(int(n_bootstrap)):
            sampled_task_indices = rng.integers(0, len(tasks), size=len(tasks))
            values = []
            for index in sampled_task_indices:
                _, task = tasks[index]
                values.append(float(task.iloc[rng.integers(0, len(task))]["estimate"]))
            draws.append(float(np.mean(values)))
        rows.append(
            {
                **dict(zip(group_cols, keys if isinstance(keys, tuple) else (keys,))),
                "macro_mean": float(task_mean["estimate"].mean()),
                "macro_median": float(task_mean["estimate"].median()),
                "ci_lower": float(np.quantile(draws, 0.025)),
                "ci_upper": float(np.quantile(draws, 0.975)),
                "n_tasks": int(len(task_mean)),
                "n_task_seed_cells": int(len(task_seed)),
            }
        )
    return pd.DataFrame(rows)


def summarize_natural_differences(differences: pd.DataFrame) -> pd.DataFrame:
    """Macro, median, information-weighted, and patient-weighted natural effects."""
    if differences.empty:
        return pd.DataFrame()
    task_keys = ["cohort_fine", "outcome", "model", "comparator", "metric", "natural_viability_tier"]
    aggregate_spec = {
        "difference": ("difference", "mean"),
        "difference_se": ("difference_se", lambda value: float(np.sqrt(np.nanmean(np.square(value))))),
        "n_test_patients": ("n_test_patients", "max"),
    }
    differences = differences.groupby(task_keys, dropna=False, as_index=False).agg(**aggregate_spec)
    rows = []
    group_cols = ["model", "comparator", "metric", "natural_viability_tier"]
    for keys, group in differences.groupby(group_cols, dropna=False):
        values = pd.to_numeric(group["difference"], errors="coerce")
        valid = group.loc[values.notna()].copy()
        values = valid["difference"].astype(float)
        if valid.empty:
            continue
        variances = pd.to_numeric(valid.get("difference_se"), errors="coerce") ** 2
        info_weights = 1.0 / variances.replace(0, np.nan)
        patient_weights = pd.to_numeric(valid["n_test_patients"], errors="coerce")
        rows.append(
            {
                **dict(zip(group_cols, keys if isinstance(keys, tuple) else (keys,))),
                "macro_mean_difference": float(values.mean()),
                "median_difference": float(values.median()),
                "proportion_favouring_model": float((values > 0).mean()),
                "information_weighted_difference": (
                    float(np.average(values[info_weights.notna()], weights=info_weights.dropna()))
                    if info_weights.notna().any() else np.nan
                ),
                "patient_weighted_difference": float(np.average(values, weights=patient_weights)),
                "n_tasks": int(valid[["cohort_fine", "outcome"]].drop_duplicates().shape[0]),
            }
        )
    return pd.DataFrame(rows)


def estimate_label_savings(
    summary: pd.DataFrame,
    *,
    comparator: str,
    metric: str = "auroc",
    target: float | None = None,
) -> pd.DataFrame:
    """Interpolate labels needed for a target and comparator full-data score."""
    frame = summary[summary["metric"] == metric].copy()
    if frame.empty:
        return pd.DataFrame()
    full_comparator = frame[frame["model"] == comparator].sort_values("sample_size")
    comparator_target = (
        float(full_comparator.iloc[-1]["macro_mean"]) if not full_comparator.empty else np.nan
    )
    rows = []
    for model, group in frame.groupby("model"):
        group = group.sort_values("sample_size")
        x = group["sample_size"].to_numpy(dtype=float)
        y = group["macro_mean"].to_numpy(dtype=float)
        for name, threshold in (("prespecified", target), ("comparator_full", comparator_target)):
            if threshold is None or not np.isfinite(threshold):
                continue
            reached = np.where(y >= threshold)[0]
            labels_needed = np.nan
            if reached.size:
                idx = int(reached[0])
                labels_needed = x[idx]
                if idx > 0 and y[idx] != y[idx - 1]:
                    labels_needed = float(
                        np.exp(
                            np.interp(
                                threshold,
                                [y[idx - 1], y[idx]],
                                [np.log(x[idx - 1]), np.log(x[idx])],
                            )
                        )
                    )
            rows.append(
                {
                    "model": model,
                    "metric": metric,
                    "target_type": name,
                    "target_value": threshold,
                    "estimated_labels_needed": labels_needed,
                    "comparator": comparator,
                }
            )
    return pd.DataFrame(rows)
