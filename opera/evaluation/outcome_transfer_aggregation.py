"""Paired aggregation for the focused OPERA outcome-transfer experiment.

The input is the held-out patient-level prediction table emitted by
``outcome_transfer_evaluate``.  This module refuses to compare predictions
unless each representation has precisely the same patient IDs, labels, event
times, and competing-event indicators.  It deliberately does not know how to
fit a probe: grouped-cohort estimates are only filters of the already-saved
pan-hematology held-out predictions.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from opera.evaluation.outcome_transfer_evaluation import (
    ALL_HEMATOLOGY,
    DenominatorParityError,
    assert_prediction_denominator_parity,
    patient_id_hash,
)


PRIMARY_COMPARISON_METRICS = ("auroc", "auprc", "brier_score")
SEVERITY_CONDITION = "opera_no_g3"
SEVERITY_PRIMARY_SCOPE = "primary_matched_g2_g3"
SEVERITY_SECONDARY_SCOPE = "secondary_unmatched_g3"

TRANSFER_DELTA_COLUMNS = (
    "comparison_condition",
    "target_outcome",
    "target_family",
    "transfer_level",
    "primary_horizon_days",
    "evaluation_role",
    "matched_lower_grade_outcome",
    "seed",
    "evaluation_level",
    "evaluation_group",
    "contrast",
    "condition_a",
    "condition_b",
    "condition_a_checkpoint_hash",
    "condition_b_checkpoint_hash",
    "condition_a_checkpoint_hash_source",
    "condition_b_checkpoint_hash_source",
    "condition_a_embedding_artifact_hash",
    "condition_b_embedding_artifact_hash",
    "condition_a_direct_target_seen",
    "condition_b_direct_target_seen",
    "condition_a_same_family_seen",
    "condition_b_same_family_seen",
    "condition_a_matched_lower_grade_seen",
    "condition_b_matched_lower_grade_seen",
    "condition_a_included_outcome_count",
    "condition_b_included_outcome_count",
    "condition_a_excluded_outcome_count",
    "condition_b_excluded_outcome_count",
    "registry_hash",
    "manifest_hash",
    "test_denominator_hash",
    "n_test",
    "n_test_events",
    "n_test_non_events",
    "n_competing_events",
    "metric",
    "estimate",
    "ci_lower",
    "ci_upper",
    "p_value",
    "n_bootstrap_requested",
    "n_bootstrap_valid",
    "n_patients",
    "benefit_direction",
    "status",
)
TRANSFER_FAMILY_SUMMARY_COLUMNS = (
    "comparison_condition",
    "target_family",
    "transfer_level",
    "primary_horizon_days",
    "contrast",
    "metric",
    "n_outcomes",
    "n_outcome_seed_cells",
    "macro_estimate",
    "outcome_standard_error",
    "aggregation",
)
TRANSFER_COHORT_SUMMARY_COLUMNS = (
    "comparison_condition",
    "target_outcome",
    "target_family",
    "transfer_level",
    "primary_horizon_days",
    "seed",
    "contrast",
    "condition_a",
    "condition_b",
    "metric",
    "n_supported_grouped_cohorts",
    "n_patients_across_supported_cohorts",
    "macro_cohort_estimate",
    "cohort_standard_error",
    "aggregation",
)
TRANSFER_AGGREGATION_FAILURE_COLUMNS = (
    "stage",
    "comparison_condition",
    "target_outcome",
    "target_family",
    "transfer_level",
    "primary_horizon_days",
    "evaluation_role",
    "matched_lower_grade_outcome",
    "seed",
    "evaluation_level",
    "evaluation_group",
    "contrast",
    "condition_a",
    "condition_b",
    "metric",
    "failure_type",
    "message",
    "registry_hash",
    "manifest_hash",
)


class OutcomeTransferAggregationError(ValueError):
    """Raised when frozen-probe outputs cannot support an honest comparison."""


def _with_schema(frame: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    """Make empty aggregation outputs reviewable and machine-readable."""
    result = frame.copy()
    for column in columns:
        if column not in result:
            result[column] = pd.Series(dtype="object")
    ordered = [*columns, *(column for column in result if column not in columns)]
    return result.loc[:, ordered]


def resolve_severity_evaluation_targets(
    plan: Mapping[str, Any],
) -> tuple[list[dict[str, str]], list[str]]:
    """Return the resolved matched and secondary Grade 3+ target sets.

    The primary severity analysis is intentionally defined by the resolver's
    explicit ``matched_g2_g3_pairs``, not by an outcome-name suffix or by a
    hand-maintained plotting list.  Grade 3+ outcomes without a matching
    Grade 2+ endpoint remain valid *secondary* outputs, but cannot silently
    enter the primary severity aggregate or figure.
    """
    conditions = plan.get("conditions")
    if not isinstance(conditions, Mapping) or SEVERITY_CONDITION not in conditions:
        raise OutcomeTransferAggregationError(
            "The resolved transfer plan has no opera_no_g3 severity condition."
        )
    condition = conditions[SEVERITY_CONDITION]
    if not isinstance(condition, Mapping):
        raise OutcomeTransferAggregationError(
            "The opera_no_g3 severity condition must be a mapping."
        )
    if condition.get("transfer_level") != "severity_transfer":
        raise OutcomeTransferAggregationError(
            "opera_no_g3 must be declared as a severity_transfer condition."
        )

    raw_pairs = condition.get("matched_g2_g3_pairs")
    if not isinstance(raw_pairs, list) or not raw_pairs:
        raise OutcomeTransferAggregationError(
            "opera_no_g3 must contain non-empty resolved matched_g2_g3_pairs."
        )
    pairs: list[dict[str, str]] = []
    for pair in raw_pairs:
        if not isinstance(pair, Mapping):
            raise OutcomeTransferAggregationError(
                "Each matched_g2_g3_pairs entry must be a mapping."
            )
        lower = pair.get("lower_grade_outcome")
        target = pair.get("target_outcome")
        if not isinstance(lower, str) or not isinstance(target, str):
            raise OutcomeTransferAggregationError(
                "Matched G2/G3 pairs require string lower_grade_outcome and target_outcome."
            )
        if not lower.endswith("_g2plus") or not target.endswith("_g3plus"):
            raise OutcomeTransferAggregationError(
                "Matched G2/G3 pairs must join a *_g2plus outcome to a *_g3plus outcome."
            )
        expected_lower = f"{target[: -len('_g3plus')]}_g2plus"
        if lower != expected_lower:
            raise OutcomeTransferAggregationError(
                f"Invalid matched G2/G3 pair {lower!r} -> {target!r}."
            )
        pairs.append({"lower_grade_outcome": lower, "target_outcome": target})

    primary = [pair["target_outcome"] for pair in pairs]
    if len(primary) != len(set(primary)):
        raise OutcomeTransferAggregationError(
            "Resolved matched G2/G3 targets must be unique."
        )
    declared_primary = condition.get("primary_evaluation_outcomes")
    if declared_primary != primary:
        raise OutcomeTransferAggregationError(
            "opera_no_g3 primary_evaluation_outcomes must exactly equal the "
            "programmatically resolved matched G2/G3 target order."
        )
    secondary = condition.get("secondary_evaluation_outcomes", [])
    evaluation = condition.get("evaluation_outcomes")
    if (
        not isinstance(secondary, list)
        or not all(isinstance(target, str) for target in secondary)
        or set(primary) & set(secondary)
        or not isinstance(evaluation, list)
        or set(evaluation) != set([*primary, *secondary])
    ):
        raise OutcomeTransferAggregationError(
            "opera_no_g3 must keep matched primary and unmatched secondary "
            "Grade 3+ evaluation targets explicit and disjoint."
        )
    return pairs, list(secondary)


def _metric_value(labels: np.ndarray, probability: np.ndarray, metric: str) -> float:
    if metric == "auroc":
        return float(roc_auc_score(labels, probability))
    if metric == "auprc":
        return float(average_precision_score(labels, probability))
    if metric == "brier_score":
        return float(brier_score_loss(labels, probability))
    raise ValueError(f"Unsupported transfer comparison metric: {metric!r}")


def paired_bootstrap_metric_delta(
    labels: Sequence[int] | np.ndarray,
    probability_a: Sequence[float] | np.ndarray,
    probability_b: Sequence[float] | np.ndarray,
    *,
    metric: str,
    n_bootstrap: int = 2000,
    seed: int = 42,
) -> dict[str, Any]:
    """Calculate a patient-level paired bootstrap difference on a fixed test set.

    ``estimate`` is always the raw metric difference ``A - B``.  For Brier
    score this means a negative estimate favours A, recorded explicitly in
    ``benefit_direction`` rather than silently negating a calibration metric.
    Confidence intervals are percentile paired-bootstrap intervals; the
    two-sided p-value uses the centred-bootstrap convention used elsewhere in
    OPERA.
    """
    if n_bootstrap < 1:
        raise OutcomeTransferAggregationError("n_bootstrap must be at least one.")
    labels_arr = np.asarray(labels, dtype=int)
    a = np.asarray(probability_a, dtype=float)
    b = np.asarray(probability_b, dtype=float)
    if not (len(labels_arr) == len(a) == len(b)):
        raise OutcomeTransferAggregationError(
            "Paired bootstrap requires equally sized labels and prediction arrays."
        )
    if len(labels_arr) == 0:
        raise OutcomeTransferAggregationError(
            "Paired bootstrap has no held-out patients."
        )
    if len(np.unique(labels_arr)) < 2 and metric in {"auroc", "auprc"}:
        raise OutcomeTransferAggregationError(
            f"{metric} is undefined because this held-out subgroup has one class."
        )
    estimate = _metric_value(labels_arr, a, metric) - _metric_value(
        labels_arr, b, metric
    )
    rng = np.random.default_rng(seed)
    draws: list[float] = []
    for _ in range(n_bootstrap):
        sampled = rng.integers(0, len(labels_arr), size=len(labels_arr))
        sampled_y = labels_arr[sampled]
        if len(np.unique(sampled_y)) < 2 and metric in {"auroc", "auprc"}:
            continue
        draws.append(
            _metric_value(sampled_y, a[sampled], metric)
            - _metric_value(sampled_y, b[sampled], metric)
        )
    if not draws:
        raise OutcomeTransferAggregationError(
            f"No valid bootstrap draws were available for {metric}."
        )
    values = np.asarray(draws, dtype=float)
    centred = values - values.mean()
    p_value = max(
        float((np.abs(centred) >= abs(estimate)).mean()),
        1.0 / float(len(values)),
    )
    return {
        "metric": metric,
        "estimate": float(estimate),
        "ci_lower": float(np.quantile(values, 0.025)),
        "ci_upper": float(np.quantile(values, 0.975)),
        "p_value": p_value,
        "n_bootstrap_requested": int(n_bootstrap),
        "n_bootstrap_valid": int(len(values)),
        "n_patients": int(len(labels_arr)),
        "benefit_direction": "lower_is_better"
        if metric == "brier_score"
        else "higher_is_better",
    }


def _required_prediction_columns() -> set[str]:
    return {
        "condition",
        "comparison_condition",
        "seed",
        "target_outcome",
        "target_family",
        "transfer_level",
        "primary_horizon_days",
        "subject_id",
        "_subject_key",
        "cohort_grouped",
        "label",
        "event",
        "time_days",
        "probability",
        "checkpoint_hash",
        "checkpoint_hash_source",
        "embedding_artifact_hash",
        "direct_target_seen",
        "same_family_seen",
        "matched_lower_grade_seen",
        "included_outcome_count",
        "excluded_outcome_count",
        "registry_hash",
        "manifest_hash",
    }


def _validate_prediction_table(predictions: pd.DataFrame) -> None:
    missing = _required_prediction_columns() - set(predictions.columns)
    if missing:
        raise OutcomeTransferAggregationError(
            f"Transfer prediction table is missing columns: {sorted(missing)}."
        )
    if predictions.empty:
        raise OutcomeTransferAggregationError("Transfer prediction table is empty.")
    if (
        predictions["probability"].isna().any()
        or not np.isfinite(predictions["probability"].to_numpy(dtype=float)).all()
    ):
        raise OutcomeTransferAggregationError(
            "Transfer prediction probabilities must be finite."
        )
    if ((predictions["probability"] < 0) | (predictions["probability"] > 1)).any():
        raise OutcomeTransferAggregationError(
            "Transfer prediction probabilities must be in the [0, 1] range."
        )


def _comparison_specs(plan: Mapping[str, Any]) -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = []
    for condition_name, condition in plan["conditions"].items():
        if condition_name == "opera_full":
            continue
        matched_lower_grade: dict[str, str] = {}
        if condition_name == SEVERITY_CONDITION:
            pairs, secondary = resolve_severity_evaluation_targets(plan)
            matched_lower_grade = {
                pair["target_outcome"]: pair["lower_grade_outcome"] for pair in pairs
            }
            primary_targets = set(matched_lower_grade)
            secondary_targets = set(secondary)
        else:
            primary_targets = set(
                condition.get(
                    "primary_evaluation_outcomes", condition["evaluation_outcomes"]
                )
            )
            secondary_targets = set(condition.get("secondary_evaluation_outcomes", []))
        for target in condition["evaluation_outcomes"]:
            if target in primary_targets:
                evaluation_role = (
                    SEVERITY_PRIMARY_SCOPE
                    if condition_name == SEVERITY_CONDITION
                    else "primary_evaluation"
                )
            elif target in secondary_targets:
                evaluation_role = (
                    SEVERITY_SECONDARY_SCOPE
                    if condition_name == SEVERITY_CONDITION
                    else "secondary_evaluation"
                )
            else:
                raise OutcomeTransferAggregationError(
                    f"{condition_name}/{target} is absent from its resolved evaluation scope."
                )
            specs.append(
                {
                    "comparison_condition": condition_name,
                    "target_outcome": str(target),
                    "target_family": plan["outcome_families"][str(target)],
                    "transfer_level": condition["transfer_level"],
                    "primary_horizon_days": int(condition["primary_horizon_days"]),
                    "evaluation_role": evaluation_role,
                    "matched_lower_grade_outcome": matched_lower_grade.get(str(target)),
                }
            )
    return specs


def _assert_cohort_parity(
    predictions: Mapping[str, pd.DataFrame], *, context: str
) -> None:
    assert_prediction_denominator_parity(predictions, context=context)
    canonical: pd.Series | None = None
    name: str | None = None
    for representation, frame in predictions.items():
        cohorts = frame.set_index("_subject_key")["cohort_grouped"].sort_index()
        if canonical is None:
            canonical = cohorts
            name = representation
        elif not cohorts.equals(canonical):
            raise DenominatorParityError(
                f"Held-out cohort_grouped mismatch between {name!r} and {representation!r} "
                f"for {context}."
            )


def _constant_prediction_value(frame: pd.DataFrame, column: str) -> Any:
    """Read one representation-level provenance value from patient rows."""
    if column not in frame:
        raise OutcomeTransferAggregationError(
            f"Prediction frame is missing required provenance column {column!r}."
        )
    if frame.empty:
        # Empty cohort groups become explicit unsupported metric failures below;
        # retain a schema-valid blank provenance field rather than erroring
        # before that failure can be recorded.
        return ""
    values = frame[column].drop_duplicates()
    if len(values) != 1:
        raise OutcomeTransferAggregationError(
            f"Prediction provenance {column!r} is not constant within one "
            "representation/target/seed cell."
        )
    return values.iloc[0]


def _paired_rows(
    *,
    comparison_condition: str,
    target: str,
    target_family: str,
    transfer_level: str,
    horizon: int,
    evaluation_role: str,
    matched_lower_grade_outcome: str | None,
    seed: int,
    evaluation_level: str,
    evaluation_group: str,
    condition_a: str,
    condition_b: str,
    contrast: str,
    frame_a: pd.DataFrame,
    frame_b: pd.DataFrame,
    n_bootstrap: int,
    bootstrap_seed: int,
    registry_hash: str,
    manifest_hash: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Return paired deltas or one explicit unsupported-metric failure row."""
    merged = frame_a.merge(
        frame_b[["_subject_key", "probability"]],
        on="_subject_key",
        suffixes=("_a", "_b"),
        how="inner",
        validate="one_to_one",
    )
    if len(merged) != len(frame_a) or len(merged) != len(frame_b):
        raise DenominatorParityError(
            f"Merge lost held-out patients for {comparison_condition}/{target}/seed_{seed}."
        )
    exposure_fields = (
        "direct_target_seen",
        "same_family_seen",
        "matched_lower_grade_seen",
        "included_outcome_count",
        "excluded_outcome_count",
    )
    provenance_a = {
        field: _constant_prediction_value(frame_a, field) for field in exposure_fields
    }
    provenance_b = {
        field: _constant_prediction_value(frame_b, field) for field in exposure_fields
    }
    base = {
        "comparison_condition": comparison_condition,
        "target_outcome": target,
        "target_family": target_family,
        "transfer_level": transfer_level,
        "primary_horizon_days": horizon,
        "evaluation_role": evaluation_role,
        "matched_lower_grade_outcome": matched_lower_grade_outcome,
        "seed": seed,
        "evaluation_level": evaluation_level,
        "evaluation_group": evaluation_group,
        "contrast": contrast,
        "condition_a": condition_a,
        "condition_b": condition_b,
        "condition_a_checkpoint_hash": _constant_prediction_value(
            frame_a, "checkpoint_hash"
        ),
        "condition_b_checkpoint_hash": _constant_prediction_value(
            frame_b, "checkpoint_hash"
        ),
        "condition_a_checkpoint_hash_source": _constant_prediction_value(
            frame_a, "checkpoint_hash_source"
        ),
        "condition_b_checkpoint_hash_source": _constant_prediction_value(
            frame_b, "checkpoint_hash_source"
        ),
        "condition_a_embedding_artifact_hash": _constant_prediction_value(
            frame_a, "embedding_artifact_hash"
        ),
        "condition_b_embedding_artifact_hash": _constant_prediction_value(
            frame_b, "embedding_artifact_hash"
        ),
        **{f"condition_a_{field}": value for field, value in provenance_a.items()},
        **{f"condition_b_{field}": value for field, value in provenance_b.items()},
        "registry_hash": registry_hash,
        "manifest_hash": manifest_hash,
        "test_denominator_hash": patient_id_hash(merged["subject_id"].tolist()),
        "n_test": int(len(merged)),
        "n_test_events": int((merged["label"] == 1).sum()),
        "n_test_non_events": int((merged["label"] == 0).sum()),
        "n_competing_events": int((merged["event"] == 2).sum()),
    }
    output: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for metric in PRIMARY_COMPARISON_METRICS:
        try:
            calculated = paired_bootstrap_metric_delta(
                merged["label"].to_numpy(dtype=int),
                merged["probability_a"].to_numpy(dtype=float),
                merged["probability_b"].to_numpy(dtype=float),
                metric=metric,
                n_bootstrap=n_bootstrap,
                seed=bootstrap_seed,
            )
        except OutcomeTransferAggregationError as exc:
            failures.append(
                {
                    "stage": "aggregate",
                    **base,
                    "metric": metric,
                    "failure_type": "unsupported_metric_support",
                    "message": str(exc),
                }
            )
            continue
        output.append({**base, **calculated, "status": "completed"})
    return output, failures


def aggregate_transfer_predictions(
    plan: Mapping[str, Any],
    predictions: pd.DataFrame,
    *,
    n_bootstrap: int = 2000,
    bootstrap_seed: int = 20260717,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Create pan-hematology deltas, grouped deltas, macro summaries, failures.

    The return order is ``(transfer_deltas, transfer_cohort_results,
    transfer_family_summary, transfer_failures)``.  Grouped comparisons never
    fit a model; they only filter the stored pan-hematology held-out rows.
    """
    _validate_prediction_table(predictions)
    if n_bootstrap < 1:
        raise OutcomeTransferAggregationError("n_bootstrap must be at least one.")
    plan_registry = str(plan["registry_hash"])
    plan_manifest = str(plan["manifest_hash"])
    if set(predictions["registry_hash"].astype(str)) != {plan_registry}:
        raise OutcomeTransferAggregationError(
            "Prediction registry_hash does not match the resolved transfer plan."
        )
    if set(predictions["manifest_hash"].astype(str)) != {plan_manifest}:
        raise OutcomeTransferAggregationError(
            "Prediction manifest_hash does not match the resolved transfer plan."
        )

    main_rows: list[dict[str, Any]] = []
    cohort_rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for spec in _comparison_specs(plan):
        condition = spec["comparison_condition"]
        target = spec["target_outcome"]
        horizon = spec["primary_horizon_days"]
        task = predictions.loc[
            (predictions["comparison_condition"].astype(str) == condition)
            & (predictions["target_outcome"].astype(str) == target)
            & (predictions["primary_horizon_days"].astype(int) == horizon)
        ].copy()
        if task.empty:
            failures.append(
                {
                    "stage": "aggregate",
                    **spec,
                    "failure_type": "missing_target_predictions",
                    "message": "No frozen-probe prediction rows were found for this target.",
                }
            )
            continue
        for seed in plan["seeds"]:
            seed = int(seed)
            seed_task = task.loc[task["seed"].astype(int) == seed].copy()
            names = ("dapt", condition, "opera_full")
            frames: dict[str, pd.DataFrame] = {}
            missing = []
            for name in names:
                frame = seed_task.loc[seed_task["condition"].astype(str) == name].copy()
                if frame.empty:
                    missing.append(name)
                else:
                    frames[name] = frame
            if missing:
                failures.append(
                    {
                        "stage": "aggregate",
                        **spec,
                        "seed": seed,
                        "failure_type": "missing_representation_predictions",
                        "message": f"Missing held-out predictions for {missing}.",
                    }
                )
                continue
            context = f"{condition}/{target}/seed_{seed}"
            _assert_cohort_parity(frames, context=context)
            contrasts = (
                (condition, "dapt", "transfer_vs_dapt"),
                ("opera_full", condition, "full_vs_transfer"),
                ("opera_full", "dapt", "full_vs_dapt"),
            )
            groups = [ALL_HEMATOLOGY] + sorted(
                frames["dapt"]["cohort_grouped"].astype(str).unique().tolist()
            )
            for group in groups:
                group_frames = {
                    name: frame
                    if group == ALL_HEMATOLOGY
                    else frame.loc[frame["cohort_grouped"].astype(str) == group].copy()
                    for name, frame in frames.items()
                }
                level = (
                    "pan_hematology" if group == ALL_HEMATOLOGY else "cohort_grouped"
                )
                for a, b, contrast in contrasts:
                    rows, row_failures = _paired_rows(
                        comparison_condition=condition,
                        target=target,
                        target_family=spec["target_family"],
                        transfer_level=spec["transfer_level"],
                        horizon=horizon,
                        evaluation_role=spec["evaluation_role"],
                        matched_lower_grade_outcome=spec["matched_lower_grade_outcome"],
                        seed=seed,
                        evaluation_level=level,
                        evaluation_group=group,
                        condition_a=a,
                        condition_b=b,
                        contrast=contrast,
                        frame_a=group_frames[a],
                        frame_b=group_frames[b],
                        n_bootstrap=n_bootstrap,
                        bootstrap_seed=bootstrap_seed + seed,
                        registry_hash=plan_registry,
                        manifest_hash=plan_manifest,
                    )
                    failures.extend(row_failures)
                    if level == "pan_hematology":
                        main_rows.extend(rows)
                    else:
                        cohort_rows.extend(rows)
    deltas = _with_schema(pd.DataFrame(main_rows), TRANSFER_DELTA_COLUMNS)
    cohort_results = _with_schema(pd.DataFrame(cohort_rows), TRANSFER_DELTA_COLUMNS)
    failure_table = _with_schema(
        pd.DataFrame(failures), TRANSFER_AGGREGATION_FAILURE_COLUMNS
    )
    family_summary = summarize_family_deltas(deltas)
    return deltas, cohort_results, family_summary, failure_table


def summarize_family_deltas(deltas: pd.DataFrame) -> pd.DataFrame:
    """Macro-average outcome estimates without pooling patient rows across targets."""
    if deltas.empty:
        return pd.DataFrame(
            columns=[
                "comparison_condition",
                "target_family",
                "transfer_level",
                "primary_horizon_days",
                "contrast",
                "metric",
                "n_outcomes",
                "n_outcome_seed_cells",
                "macro_estimate",
                "outcome_standard_error",
                "aggregation",
            ]
        )
    # First average repeat seeds within each outcome, then average each outcome
    # once.  This is a macro-outcome summary, not a pseudo-patient pooled test.
    per_outcome = deltas.groupby(
        [
            "comparison_condition",
            "target_outcome",
            "target_family",
            "transfer_level",
            "primary_horizon_days",
            "contrast",
            "metric",
        ],
        dropna=False,
        as_index=False,
    ).agg(estimate=("estimate", "mean"), n_seeds=("seed", "nunique"))
    summary = per_outcome.groupby(
        [
            "comparison_condition",
            "target_family",
            "transfer_level",
            "primary_horizon_days",
            "contrast",
            "metric",
        ],
        dropna=False,
        as_index=False,
    ).agg(
        n_outcomes=("target_outcome", "nunique"),
        n_outcome_seed_cells=("n_seeds", "sum"),
        macro_estimate=("estimate", "mean"),
        outcome_standard_deviation=("estimate", "std"),
    )
    summary["outcome_standard_error"] = summary["outcome_standard_deviation"] / np.sqrt(
        summary["n_outcomes"].clip(lower=1)
    )
    summary["aggregation"] = "macro_outcome_after_seed_mean"
    return summary.drop(columns=["outcome_standard_deviation"])


def _empty_severity_summary(scope: str) -> pd.DataFrame:
    """Return a schema-stable empty severity aggregate."""
    return (
        pd.DataFrame(
            columns=[
                "severity_scope",
                "comparison_condition",
                "transfer_level",
                "primary_horizon_days",
                "contrast",
                "condition_a",
                "condition_b",
                "metric",
                "n_targets",
                "n_target_seed_cells",
                "target_outcomes",
                "lower_grade_outcomes",
                "macro_estimate",
                "target_standard_error",
                "aggregation",
            ]
        )
        .assign(severity_scope=scope)
        .iloc[0:0]
    )


def summarize_severity_deltas(
    plan: Mapping[str, Any],
    deltas: pd.DataFrame,
    *,
    scope: str = "primary",
) -> pd.DataFrame:
    """Macro-summarize paired No-G3 contrasts at a declared severity scope.

    ``scope='primary'`` uses only the targets in the resolver-produced
    ``matched_g2_g3_pairs``.  ``scope='secondary'`` retains unmatched Grade
    3+ endpoints as a clearly labelled supplemental analysis.  Both paths
    operate on the patient-level paired bootstrap deltas already calculated
    by :func:`aggregate_transfer_predictions`; neither pools raw patients nor
    falls back to an unpaired difference of independently fitted metrics.
    """
    if scope == "primary":
        severity_scope = SEVERITY_PRIMARY_SCOPE
    elif scope == "secondary":
        severity_scope = SEVERITY_SECONDARY_SCOPE
    else:
        raise OutcomeTransferAggregationError(
            "Severity scope must be either 'primary' or 'secondary'."
        )

    pairs, secondary_targets = resolve_severity_evaluation_targets(plan)
    target_to_lower = {
        pair["target_outcome"]: pair["lower_grade_outcome"] for pair in pairs
    }
    targets = list(target_to_lower) if scope == "primary" else list(secondary_targets)
    if not targets:
        return _empty_severity_summary(severity_scope)

    required = {
        "comparison_condition",
        "target_outcome",
        "transfer_level",
        "primary_horizon_days",
        "contrast",
        "condition_a",
        "condition_b",
        "metric",
        "seed",
        "estimate",
        "evaluation_level",
        "evaluation_group",
    }
    missing = required - set(deltas.columns)
    if missing:
        raise OutcomeTransferAggregationError(
            f"Transfer deltas are missing severity aggregation columns: {sorted(missing)}."
        )

    work = deltas.loc[
        (deltas["comparison_condition"].astype(str) == SEVERITY_CONDITION)
        & deltas["target_outcome"].astype(str).isin(targets)
        & (deltas["evaluation_level"].astype(str) == "pan_hematology")
        & (deltas["evaluation_group"].astype(str) == ALL_HEMATOLOGY)
    ].copy()
    if "status" in work.columns:
        work = work.loc[work["status"].astype(str) == "completed"].copy()
    if "evaluation_role" in work.columns and not work.empty:
        observed_roles = set(work["evaluation_role"].astype(str))
        if observed_roles != {severity_scope}:
            raise OutcomeTransferAggregationError(
                "Severity delta rows do not match their resolved primary/secondary scope."
            )
    if work.empty:
        return _empty_severity_summary(severity_scope)

    # Average repeat seeds within each held-out outcome first.  This retains
    # each target's paired estimate as the unit of the final macro summary.
    per_target = work.groupby(
        [
            "comparison_condition",
            "target_outcome",
            "transfer_level",
            "primary_horizon_days",
            "contrast",
            "condition_a",
            "condition_b",
            "metric",
        ],
        dropna=False,
        as_index=False,
    ).agg(estimate=("estimate", "mean"), n_seeds=("seed", "nunique"))
    summary = per_target.groupby(
        [
            "comparison_condition",
            "transfer_level",
            "primary_horizon_days",
            "contrast",
            "condition_a",
            "condition_b",
            "metric",
        ],
        dropna=False,
        as_index=False,
    ).agg(
        n_targets=("target_outcome", "nunique"),
        n_target_seed_cells=("n_seeds", "sum"),
        macro_estimate=("estimate", "mean"),
        target_standard_deviation=("estimate", "std"),
    )
    summary["target_standard_error"] = summary["target_standard_deviation"] / np.sqrt(
        summary["n_targets"].clip(lower=1)
    )
    summary = summary.drop(columns=["target_standard_deviation"])
    summary.insert(0, "severity_scope", severity_scope)
    summary["target_outcomes"] = json.dumps(targets)
    summary["lower_grade_outcomes"] = json.dumps(
        [target_to_lower[target] for target in targets if target in target_to_lower]
    )
    summary["aggregation"] = (
        "paired_patient_delta_macro_matched_g2_g3_after_seed_mean"
        if scope == "primary"
        else "paired_patient_delta_macro_unmatched_g3_after_seed_mean"
    )
    return summary[
        [
            "severity_scope",
            "comparison_condition",
            "transfer_level",
            "primary_horizon_days",
            "contrast",
            "condition_a",
            "condition_b",
            "metric",
            "n_targets",
            "n_target_seed_cells",
            "target_outcomes",
            "lower_grade_outcomes",
            "macro_estimate",
            "target_standard_error",
            "aggregation",
        ]
    ]


def summarize_grouped_cohort_deltas(cohort_results: pd.DataFrame) -> pd.DataFrame:
    """Macro-average valid grouped estimates without pooling patient rows.

    Each grouped cohort first contributes its own paired patient-level delta.
    This summary gives every supported group equal weight within a
    target/seed/contrast/metric cell, keeping it distinct from the primary
    patient-weighted pan-hematology estimate.
    """
    if cohort_results.empty:
        return pd.DataFrame(
            columns=[
                "comparison_condition",
                "target_outcome",
                "target_family",
                "transfer_level",
                "primary_horizon_days",
                "seed",
                "contrast",
                "condition_a",
                "condition_b",
                "metric",
                "n_supported_grouped_cohorts",
                "n_patients_across_supported_cohorts",
                "macro_cohort_estimate",
                "cohort_standard_error",
                "aggregation",
            ]
        )
    supported = cohort_results.copy()
    if "status" in supported:
        supported = supported.loc[supported["status"].astype(str) == "completed"].copy()
    if supported.empty:
        return summarize_grouped_cohort_deltas(pd.DataFrame())
    grouping = [
        "comparison_condition",
        "target_outcome",
        "target_family",
        "transfer_level",
        "primary_horizon_days",
        "seed",
        "contrast",
        "condition_a",
        "condition_b",
        "metric",
    ]
    summary = supported.groupby(grouping, dropna=False, as_index=False).agg(
        n_supported_grouped_cohorts=("evaluation_group", "nunique"),
        n_patients_across_supported_cohorts=("n_test", "sum"),
        macro_cohort_estimate=("estimate", "mean"),
        cohort_standard_deviation=("estimate", "std"),
    )
    summary["cohort_standard_error"] = summary["cohort_standard_deviation"] / np.sqrt(
        summary["n_supported_grouped_cohorts"].clip(lower=1)
    )
    summary["aggregation"] = "macro_grouped_cohort_within_target_seed"
    return summary.drop(columns=["cohort_standard_deviation"])


def write_transfer_aggregation_outputs(
    output_dir: str | Path,
    *,
    deltas: pd.DataFrame,
    cohort_results: pd.DataFrame,
    family_summary: pd.DataFrame,
    failures: pd.DataFrame,
    cohort_summary: pd.DataFrame | None = None,
    severity_primary_summary: pd.DataFrame | None = None,
    severity_secondary_summary: pd.DataFrame | None = None,
) -> dict[str, Path]:
    """Write the fixed output names required by the transfer protocol."""
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    paths = {
        "transfer_deltas": destination / "transfer_deltas.csv",
        "transfer_cohort_results": destination / "transfer_cohort_results.csv",
        "transfer_family_summary": destination / "transfer_family_summary.csv",
        "transfer_cohort_summary": destination / "transfer_cohort_summary.csv",
        "transfer_severity_primary_summary": destination
        / "transfer_severity_primary_summary.csv",
        "transfer_severity_secondary_summary": destination
        / "transfer_severity_secondary_summary.csv",
        "transfer_failures": destination / "transfer_failures.csv",
    }
    _with_schema(deltas, TRANSFER_DELTA_COLUMNS).to_csv(
        paths["transfer_deltas"], index=False
    )
    _with_schema(cohort_results, TRANSFER_DELTA_COLUMNS).to_csv(
        paths["transfer_cohort_results"], index=False
    )
    _with_schema(family_summary, TRANSFER_FAMILY_SUMMARY_COLUMNS).to_csv(
        paths["transfer_family_summary"], index=False
    )
    _with_schema(
        cohort_summary
        if cohort_summary is not None
        else summarize_grouped_cohort_deltas(cohort_results),
        TRANSFER_COHORT_SUMMARY_COLUMNS,
    ).to_csv(paths["transfer_cohort_summary"], index=False)
    (
        severity_primary_summary
        if severity_primary_summary is not None
        else _empty_severity_summary(SEVERITY_PRIMARY_SCOPE)
    ).to_csv(paths["transfer_severity_primary_summary"], index=False)
    (
        severity_secondary_summary
        if severity_secondary_summary is not None
        else _empty_severity_summary(SEVERITY_SECONDARY_SCOPE)
    ).to_csv(paths["transfer_severity_secondary_summary"], index=False)
    _with_schema(failures, TRANSFER_AGGREGATION_FAILURE_COLUMNS).to_csv(
        paths["transfer_failures"], index=False
    )
    return paths
