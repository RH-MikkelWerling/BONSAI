"""Evaluate external predictions on the canonical OPERA cohorts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from opera.evaluation.cohorts import (
    FIXED_HORIZON_REGIME,
    SURVIVAL_REGIME,
    EvaluationCohorts,
    assert_cohort_parity,
    build_evaluation_cohorts,
    cohort_summary,
    intersect_subject_restrictions,
    population_subject_ids,
    read_subject_ids,
)
from opera.evaluation.metrics import (
    bootstrap_survival_metrics,
    compute_survival_metrics,
    format_evaluation_summary,
    full_evaluation,
    validate_binary_evaluation_inputs,
)
from opera.evaluation.results_schema import (
    bootstrap_ci_rows,
    build_result_row,
    write_result_artifacts,
)
from opera.evaluation.subgroups import compute_subgroup_metrics, load_subgroup_table


def read_table(path: str) -> pd.DataFrame:
    source = Path(path)
    if source.suffix.lower() in {".parquet", ".pq"}:
        return pd.read_parquet(source)
    return pd.read_csv(source)


def _prediction_frame(
    predictions: pd.DataFrame,
    *,
    probability_col: str,
    risk_col: Optional[str],
) -> pd.DataFrame:
    if "subject_id" not in predictions.columns:
        raise ValueError("Prediction file must contain subject_id.")
    columns = ["subject_id"]
    if probability_col in predictions.columns:
        columns.append(probability_col)
    if risk_col and risk_col in predictions.columns and risk_col not in columns:
        columns.append(risk_col)
    frame = predictions[columns].copy()
    if probability_col in frame.columns:
        frame = frame.rename(columns={probability_col: "probability"})
    if risk_col == probability_col and "probability" in frame.columns:
        frame["risk_score"] = frame["probability"]
    elif risk_col and risk_col in frame.columns:
        frame = frame.rename(columns={risk_col: "risk_score"})
    if "risk_score" not in frame.columns and "probability" in frame.columns:
        frame["risk_score"] = frame["probability"]
    return frame


def build_eval_frames(
    predictions: pd.DataFrame,
    outcome_path: str,
    split: str,
    probability_col: str,
    n_hours_start_include: int,
    n_hours_end_include,
    evaluation_regime: str,
    risk_col: Optional[str] = None,
    competing_outcome_path: str = None,
    eligibility_path: str = None,
    registry_start_date=None,
    cohort: str = None,
    outcome_name: str = None,
    population_path: str = None,
    cohort_fine_col: str = None,
    cohort_fine_value: str = None,
    evaluation_subjects_path: str = None,
) -> tuple[EvaluationCohorts, dict[str, pd.DataFrame]]:
    """Join predictions to outcome-derived labels without letting files define cohorts."""
    if evaluation_regime not in {FIXED_HORIZON_REGIME, SURVIVAL_REGIME, "both"}:
        raise ValueError(f"Unknown evaluation_regime={evaluation_regime!r}.")
    pred = _prediction_frame(
        predictions,
        probability_col=probability_col,
        risk_col=risk_col,
    )
    allowed_ids = intersect_subject_restrictions(
        population_subject_ids(
            population_path,
            cohort_fine_col=cohort_fine_col,
            cohort_fine_value=cohort_fine_value,
        ),
        read_subject_ids(evaluation_subjects_path),
    )
    cohorts = build_evaluation_cohorts(
        outcome_path,
        split=split,
        n_hours_start_include=n_hours_start_include,
        n_hours_end_include=n_hours_end_include,
        competing_outcomes=competing_outcome_path,
        eligibility=eligibility_path,
        registry_start_date=registry_start_date,
        cohort=cohort,
        outcome_name=outcome_name,
        allowed_subject_ids=allowed_ids,
    )

    parity_regime = (
        FIXED_HORIZON_REGIME
        if evaluation_regime == FIXED_HORIZON_REGIME
        else SURVIVAL_REGIME
    )
    assert_cohort_parity(
        cohorts.for_regime(parity_regime),
        pred["subject_id"],
        model_name="prediction_file",
        outcome_name=outcome_name or "unknown",
    )

    frames: dict[str, pd.DataFrame] = {}
    if evaluation_regime in {FIXED_HORIZON_REGIME, "both"}:
        fixed = cohorts.fixed_horizon.to_frame().merge(
            pred,
            on="subject_id",
            how="left",
            validate="one_to_one",
        )
        if "probability" not in fixed.columns:
            raise ValueError(
                f"Fixed-horizon evaluation requires probability column {probability_col!r}."
            )
        if fixed["probability"].isna().any():
            raise ValueError("Fixed-horizon predictions contain missing probabilities.")
        fixed["binary_eligible"] = True
        frames[FIXED_HORIZON_REGIME] = fixed

    if evaluation_regime in {SURVIVAL_REGIME, "both"}:
        survival = cohorts.survival.to_frame().merge(
            pred,
            on="subject_id",
            how="left",
            validate="one_to_one",
        )
        if "risk_score" not in survival.columns:
            requested = risk_col or probability_col
            raise ValueError(
                f"Survival evaluation requires risk column {requested!r}."
            )
        if survival["risk_score"].isna().any():
            raise ValueError("Survival predictions contain missing risk scores.")
        frames[SURVIVAL_REGIME] = survival
    return cohorts, frames


def build_eval_frame(
    predictions: pd.DataFrame,
    outcome_path: str,
    split: str,
    probability_col: str,
    n_hours_start_include: int,
    n_hours_end_include,
    require_min_followup: bool,
    **kwargs,
) -> pd.DataFrame:
    """Backward-compatible single-regime wrapper used by callers and tests."""
    regime = FIXED_HORIZON_REGIME if require_min_followup else SURVIVAL_REGIME
    _, frames = build_eval_frames(
        predictions=predictions,
        outcome_path=outcome_path,
        split=split,
        probability_col=probability_col,
        n_hours_start_include=n_hours_start_include,
        n_hours_end_include=n_hours_end_include,
        evaluation_regime=regime,
        **kwargs,
    )
    return frames[regime]


def outcome_window_size_metadata(
    outcome_path: str,
    n_hours_start_include: int,
    n_hours_end_include,
    evaluation_regime: str = FIXED_HORIZON_REGIME,
    competing_outcome_path: str = None,
    eligibility_path: str = None,
    registry_start_date=None,
    cohort: str = None,
    outcome_name: str = None,
    allowed_subject_ids=None,
) -> dict:
    """Compute split sizes/events from the same canonical cohort builder."""
    metadata = {}
    regime = (
        FIXED_HORIZON_REGIME
        if evaluation_regime == "both"
        else evaluation_regime
    )
    for split_name, result_key in (
        ("train", "train"),
        ("tuning", "val"),
        ("held_out", "test"),
    ):
        cohorts = build_evaluation_cohorts(
            outcome_path,
            split=split_name,
            n_hours_start_include=n_hours_start_include,
            n_hours_end_include=n_hours_end_include,
            competing_outcomes=competing_outcome_path,
            eligibility=eligibility_path,
            registry_start_date=registry_start_date,
            cohort=cohort,
            outcome_name=outcome_name,
            allowed_subject_ids=allowed_subject_ids,
        )
        selected = cohorts.for_regime(regime)
        n_subjects = len(selected.subject_ids)
        n_events = selected.n_events
        metadata[f"n_{result_key}"] = int(n_subjects)
        metadata[f"n_events_{result_key}"] = int(n_events)
        metadata[f"prevalence_{result_key}"] = (
            float(n_events / n_subjects) if n_subjects else float("nan")
        )
    return metadata


def validate_binary_inputs(labels: np.ndarray, probabilities: np.ndarray) -> None:
    if len(labels) == 0:
        raise ValueError(
            "No prediction rows have sufficient follow-up for this binary horizon."
        )
    validate_binary_evaluation_inputs(labels, probabilities)


def format_survival_summary(report: dict) -> str:
    survival = report["survival"]
    ci = report.get("survival_bootstrap_ci", {})
    lines = [
        "=" * 70,
        "OPERA SURVIVAL EVALUATION REPORT",
        "=" * 70,
        (
            f"C-index: {survival['concordance_index']:.4f} "
            f"(n={survival['n_total']}, events={survival['n_events']})"
        ),
    ]
    c_ci = ci.get("concordance_index", {})
    if c_ci:
        lines.append(
            f"C-index 95% CI: [{c_ci.get('lower', float('nan')):.4f}, "
            f"{c_ci.get('upper', float('nan')):.4f}]"
        )
    for horizon, metrics in survival.get("per_horizon", {}).items():
        lines.append(
            f"{horizon}: IPCW-AUC={metrics['ipcw_auc']:.4f} "
            f"IPCW-Brier={metrics['ipcw_brier']:.4f} "
            f"cases={metrics['n_cases']} controls={metrics['n_controls']}"
        )
    lines.append("=" * 70)
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate prediction-file baseline")
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--outcome", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--probability_col", default="probability")
    parser.add_argument("--risk_col", default=None)
    parser.add_argument(
        "--evaluation_regime",
        default="both",
        choices=[FIXED_HORIZON_REGIME, SURVIVAL_REGIME, "both"],
    )
    parser.add_argument("--split", default="held_out")
    parser.add_argument("--cohort", default="unknown")
    parser.add_argument("--outcome_name", default="unknown")
    parser.add_argument("--model_family", default="tabular")
    parser.add_argument("--training_stage", default="prediction_file_evaluation")
    parser.add_argument("--n_hours_start_include", type=int, default=1)
    parser.add_argument("--n_hours_end_include", type=int, default=None)
    parser.add_argument("--n_bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument(
        "--rarity_mode", default="none", choices=["none", "synthetic", "real"]
    )
    parser.add_argument("--baseline_model", default=None)
    parser.add_argument("--ipi_coverage", type=float, default=None)
    parser.add_argument(
        "--evaluation_subset", default="full", choices=["full", "ipi_complete"]
    )
    parser.add_argument("--competing_outcome", default=None)
    parser.add_argument("--eligibility", default=None)
    parser.add_argument("--registry_start_date", default=None)
    parser.add_argument("--population", default=None)
    parser.add_argument("--cohort_fine_col", default=None)
    parser.add_argument("--cohort_fine_value", default=None)
    parser.add_argument("--evaluation_subjects", default=None)
    parser.add_argument("--subgroups", default=None)
    parser.add_argument("--subgroup_columns", default="")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions = read_table(args.predictions)
    cohorts, frames = build_eval_frames(
        predictions=predictions,
        outcome_path=args.outcome,
        split=args.split,
        probability_col=args.probability_col,
        risk_col=args.risk_col,
        n_hours_start_include=args.n_hours_start_include,
        n_hours_end_include=args.n_hours_end_include,
        evaluation_regime=args.evaluation_regime,
        competing_outcome_path=args.competing_outcome,
        eligibility_path=args.eligibility,
        registry_start_date=args.registry_start_date,
        cohort=args.cohort,
        outcome_name=args.outcome_name,
        population_path=args.population,
        cohort_fine_col=args.cohort_fine_col,
        cohort_fine_value=args.cohort_fine_value,
        evaluation_subjects_path=args.evaluation_subjects,
    )

    time_horizons = (
        [float(args.n_hours_end_include) / 24.0]
        if args.n_hours_end_include is not None
        else [365.0, 730.0]
    )
    binary_df = frames.get(FIXED_HORIZON_REGIME)
    survival_df = frames.get(SURVIVAL_REGIME)
    labels = probabilities = None

    if binary_df is not None:
        assert_cohort_parity(
            cohorts.fixed_horizon,
            binary_df["subject_id"],
            model_name=args.model_family,
            outcome_name=args.outcome_name,
        )
        labels = binary_df["label"].to_numpy(dtype=int)
        probabilities = binary_df["probability"].to_numpy(dtype=float)
        validate_binary_inputs(labels, probabilities)
        if survival_df is not None:
            assert_cohort_parity(
                cohorts.survival,
                survival_df["subject_id"],
                model_name=args.model_family,
                outcome_name=args.outcome_name,
            )
        report = full_evaluation(
            labels,
            probabilities,
            threshold=args.threshold,
            n_bootstrap=args.n_bootstrap,
            seed=args.seed,
            times=(
                survival_df["time_days"].to_numpy(dtype=float)
                if survival_df is not None
                else None
            ),
            events=(
                survival_df["event"].to_numpy(dtype=int)
                if survival_df is not None
                else None
            ),
            survival_probabilities=(
                survival_df["risk_score"].to_numpy(dtype=float)
                if survival_df is not None
                else None
            ),
            time_horizons=time_horizons,
        )
        summary = format_evaluation_summary(report)
    else:
        assert survival_df is not None
        assert_cohort_parity(
            cohorts.survival,
            survival_df["subject_id"],
            model_name=args.model_family,
            outcome_name=args.outcome_name,
        )
        times = survival_df["time_days"].to_numpy(dtype=float)
        events = survival_df["event"].to_numpy(dtype=int)
        risks = survival_df["risk_score"].to_numpy(dtype=float)
        report = {
            "survival": compute_survival_metrics(
                times,
                events,
                risks,
                time_horizons=time_horizons,
            ),
            "survival_bootstrap_ci": bootstrap_survival_metrics(
                times,
                events,
                risks,
                time_horizons=time_horizons,
                n_bootstrap=min(args.n_bootstrap, 500),
                seed=args.seed,
            ),
        }
        summary = format_survival_summary(report)

    print(cohort_summary(cohorts.fixed_horizon, args.outcome_name))
    print(cohort_summary(cohorts.survival, args.outcome_name))
    with open(output_dir / "evaluation_report.txt", "w", encoding="utf-8") as f:
        f.write(summary)
    if binary_df is not None:
        report["threshold_sweep"].to_csv(
            output_dir / "threshold_sweep.csv", index=False
        )
        report["decision_curve"].to_csv(output_dir / "decision_curve.csv", index=False)
        report["high_risk_enrichment"].to_csv(
            output_dir / "high_risk_enrichment.csv", index=False
        )
    bootstrap_ci_rows(report).to_csv(output_dir / "bootstrap_ci.csv", index=False)

    subgroup_columns = [
        item.strip() for item in args.subgroup_columns.split(",") if item.strip()
    ]
    if args.subgroups and subgroup_columns and binary_df is not None:
        subgroup_metrics = compute_subgroup_metrics(
            subject_ids=binary_df["subject_id"].to_numpy(),
            labels=labels,
            probabilities=probabilities,
            subgroup_df=load_subgroup_table(args.subgroups),
            columns=subgroup_columns,
            threshold=args.threshold,
        )
        subgroup_metrics.to_csv(output_dir / "subgroup_metrics.csv", index=False)

    if survival_df is not None:
        np.savez(
            output_dir / "predictions.npz",
            subject_ids=survival_df["subject_id"].to_numpy(),
            probabilities=survival_df.get(
                "probability", pd.Series(np.nan, index=survival_df.index)
            ).to_numpy(),
            risk_scores=survival_df["risk_score"].to_numpy(),
            times=survival_df["time_days"].to_numpy(),
            events=survival_df["event"].to_numpy(),
        )
    else:
        np.savez(
            output_dir / "predictions.npz",
            subject_ids=binary_df["subject_id"].to_numpy(),
            labels=labels,
            probabilities=probabilities,
        )

    allowed_ids = intersect_subject_restrictions(
        population_subject_ids(
            args.population,
            cohort_fine_col=args.cohort_fine_col,
            cohort_fine_value=args.cohort_fine_value,
        ),
        read_subject_ids(args.evaluation_subjects),
    )
    size_metadata = outcome_window_size_metadata(
        args.outcome,
        n_hours_start_include=args.n_hours_start_include,
        n_hours_end_include=args.n_hours_end_include,
        evaluation_regime=args.evaluation_regime,
        competing_outcome_path=args.competing_outcome,
        eligibility_path=args.eligibility,
        registry_start_date=args.registry_start_date,
        cohort=args.cohort,
        outcome_name=args.outcome_name,
        allowed_subject_ids=allowed_ids,
    )
    active = (
        cohorts.fixed_horizon
        if args.evaluation_regime in {FIXED_HORIZON_REGIME, "both"}
        else cohorts.survival
    )
    size_metadata.update(
        {
            "n_test": len(active.subject_ids),
            "n_events_test": active.n_events,
            "n_competing_events_test": int(
                (survival_df["event"] == 2).sum() if survival_df is not None else 0
            ),
            "prevalence_test": (
                float(active.n_events / len(active.subject_ids))
                if active.subject_ids
                else float("nan")
            ),
        }
    )
    cfg = {
        "model_family": args.model_family,
        "training_stage": args.training_stage,
        "cohort": args.cohort,
        "outcome": args.outcome_name,
        "ipi_coverage": args.ipi_coverage,
        "evaluation_subset": args.evaluation_subset,
        "evaluation_regime": args.evaluation_regime,
        "seed": args.seed,
        "labels": {"n_hours_end_include": args.n_hours_end_include},
        "registry_start_date": args.registry_start_date,
        "rarity": {
            "mode": args.rarity_mode,
            "baseline_model": args.baseline_model,
            "size_metadata": size_metadata,
        },
    }
    row = build_result_row(
        cfg,
        report,
        checkpoint_path=args.predictions,
        split=args.split,
        model_family=args.model_family,
        training_stage=args.training_stage,
    )
    write_result_artifacts(row, output_dir)
    json_safe = {
        key: {
            subkey: (value.tolist() if isinstance(value, np.ndarray) else value)
            for subkey, value in payload.items()
            if not isinstance(value, (pd.DataFrame, list))
        }
        for key, payload in report.items()
        if isinstance(payload, dict)
    }
    json_safe["result_metadata"] = row
    with open(output_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(json_safe, f, indent=2, default=str)
    print(summary.encode("ascii", errors="replace").decode("ascii"))
    print(f"Prediction evaluation complete: {output_dir}")


if __name__ == "__main__":
    main()
