"""
Evaluate a prediction file with the same OPERA metric suite.

This is intended for tabular baselines, IPI-family scores, and other non-neural
models. The input prediction file must contain `subject_id` plus a probability
column. If it also contains `label`, that label is used directly; otherwise
labels are derived from the outcome parquet and the configured horizon.

IPI rows are a special caveat: their "probabilities" are rank-normalized
ordinal clinical score categories, not calibrated event probabilities.
Calibration metrics are still computed for pipeline consistency, but should not
be interpreted as probability calibration in the same way as model outputs.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from opera.compat.bonsai import binarize_outcomes
from opera.functional.outcomes import (
    attach_prediction_censor_abspos,
    filter_outcome_eligibility,
    filter_registry_eligible_outcomes,
)
from opera.evaluation.metrics import (
    full_evaluation,
    format_evaluation_summary,
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


def build_eval_frame(
    predictions: pd.DataFrame,
    outcome_path: str,
    split: str,
    probability_col: str,
    n_hours_start_include: int,
    n_hours_end_include,
    require_min_followup: bool,
    competing_outcome_path: str = None,
    eligibility_path: str = None,
    registry_start_date=None,
    cohort: str = None,
    outcome_name: str = None,
) -> pd.DataFrame:
    if "subject_id" not in predictions.columns:
        raise ValueError("Prediction file must contain subject_id.")
    if probability_col not in predictions.columns:
        raise ValueError(
            f"Prediction file is missing probability column {probability_col!r}."
        )

    pred = predictions[
        ["subject_id", probability_col]
        + (["label"] if "label" in predictions.columns else [])
    ].copy()
    pred = pred.rename(columns={probability_col: "probability"})

    outcomes = pd.read_parquet(outcome_path)
    outcomes = outcomes[outcomes["split"] == split].copy()
    outcomes = filter_outcome_eligibility(
        outcomes,
        eligibility_path,
        cohort=cohort,
        outcome_name=outcome_name,
    )
    outcomes = attach_prediction_censor_abspos(outcomes)
    outcomes = filter_registry_eligible_outcomes(
        outcomes,
        registry_start_date,
        cohort=cohort,
        outcome_name=outcome_name,
    )

    competing_df = None
    if competing_outcome_path:
        competing_df = pd.read_parquet(competing_outcome_path)

    all_labels = binarize_outcomes(
        outcomes,
        n_hours_start_include=n_hours_start_include,
        n_hours_end_include=n_hours_end_include,
        require_min_followup=False,
        split_name=split,
        competing_event_df=competing_df,
    )
    if "label" in pred.columns:
        survival_df = pd.DataFrame.from_dict(all_labels, orient="index").reset_index(
            names="subject_id"
        )
        survival_df = survival_df.drop(columns=["label"], errors="ignore")
        frame = pred.dropna(subset=["probability", "label"]).merge(
            survival_df,
            on="subject_id",
            how="left",
        )
        frame["binary_eligible"] = True
        return frame

    full_fu_labels = binarize_outcomes(
        outcomes,
        n_hours_start_include=n_hours_start_include,
        n_hours_end_include=n_hours_end_include,
        require_min_followup=require_min_followup,
        split_name=split,
        competing_event_df=competing_df,
    )
    label_df = pd.DataFrame.from_dict(all_labels, orient="index").reset_index(
        names="subject_id"
    )
    frame = pred.merge(label_df, on="subject_id", how="inner").dropna(
        subset=["probability"]
    )
    frame["binary_eligible"] = frame["subject_id"].isin(full_fu_labels)
    return frame


def outcome_window_size_metadata(
    outcome_path: str,
    n_hours_start_include: int,
    n_hours_end_include,
    competing_outcome_path: str = None,
    eligibility_path: str = None,
    registry_start_date=None,
    cohort: str = None,
    outcome_name: str = None,
) -> dict:
    """Compute split sizes/events for the same horizon used in evaluation."""
    outcomes = pd.read_parquet(outcome_path)
    outcomes = filter_outcome_eligibility(
        outcomes,
        eligibility_path,
        cohort=cohort,
        outcome_name=outcome_name,
    )
    outcomes = attach_prediction_censor_abspos(outcomes)
    outcomes = filter_registry_eligible_outcomes(
        outcomes,
        registry_start_date,
        cohort=cohort,
        outcome_name=outcome_name,
    )
    competing_df = None
    if competing_outcome_path:
        competing_df = pd.read_parquet(competing_outcome_path)
    metadata = {}
    for split_name, result_key, require_followup in (
        ("train", "train", False),
        ("tuning", "val", n_hours_end_include is not None),
        ("held_out", "test", n_hours_end_include is not None),
    ):
        split_df = outcomes[outcomes["split"] == split_name].copy()
        labels = binarize_outcomes(
            split_df,
            n_hours_start_include=n_hours_start_include,
            n_hours_end_include=n_hours_end_include,
            require_min_followup=require_followup,
            split_name=split_name,
            competing_event_df=competing_df,
        )
        n_subjects = int(len(labels))
        n_events = int(sum(record["label"] for record in labels.values()))
        metadata[f"n_{result_key}"] = n_subjects
        metadata[f"n_events_{result_key}"] = n_events
        metadata[f"prevalence_{result_key}"] = (
            float(n_events / n_subjects) if n_subjects else float("nan")
        )
    return metadata


def validate_binary_inputs(labels: np.ndarray, probabilities: np.ndarray) -> None:
    if len(labels) == 0:
        raise ValueError(
            "No prediction rows have sufficient follow-up for this binary "
            "horizon. Check the outcome window or provide prediction labels."
        )
    validate_binary_evaluation_inputs(labels, probabilities)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate prediction-file baseline")
    parser.add_argument("--predictions", required=True)
    parser.add_argument(
        "--outcome", required=True, help="Outcome parquet used if labels are absent"
    )
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--probability_col", default="probability")
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
    parser.add_argument(
        "--competing_outcome",
        default=None,
        help="Optional path to competing-event (death) parquet for event=2 annotation",
    )
    parser.add_argument(
        "--eligibility",
        default=None,
        help="Optional patient-level outcome eligibility CSV/parquet",
    )
    parser.add_argument(
        "--registry_start_date",
        default=None,
        help="Optional first date with reliable registry outcome coverage",
    )
    parser.add_argument(
        "--subgroups",
        default=None,
        help="Optional CSV/parquet with subject_id plus subgroup columns",
    )
    parser.add_argument(
        "--subgroup_columns",
        default="",
        help="Comma-separated subgroup columns to evaluate",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions = read_table(args.predictions)
    eval_df = build_eval_frame(
        predictions=predictions,
        outcome_path=args.outcome,
        split=args.split,
        probability_col=args.probability_col,
        n_hours_start_include=args.n_hours_start_include,
        n_hours_end_include=args.n_hours_end_include,
        require_min_followup=args.n_hours_end_include is not None,
        competing_outcome_path=args.competing_outcome,
        eligibility_path=args.eligibility,
        registry_start_date=args.registry_start_date,
        cohort=args.cohort,
        outcome_name=args.outcome_name,
    )
    if eval_df.empty:
        raise ValueError("No evaluable prediction rows after joining labels/outcomes.")

    binary_df = eval_df[eval_df["binary_eligible"]].copy()
    labels = binary_df["label"].to_numpy()
    probabilities = binary_df["probability"].to_numpy()
    validate_binary_inputs(labels, probabilities)
    survival_probabilities = None
    if (
        "time_days" in eval_df.columns
        and eval_df[["time_days", "event"]].notna().all().all()
    ):
        times = eval_df["time_days"].to_numpy()
        events = eval_df["event"].to_numpy()
        survival_probabilities = eval_df["probability"].to_numpy()
    else:
        times = None
        events = None
    report = full_evaluation(
        labels,
        probabilities,
        threshold=args.threshold,
        n_bootstrap=args.n_bootstrap,
        seed=args.seed,
        times=times,
        events=events,
        survival_probabilities=survival_probabilities,
    )
    summary = format_evaluation_summary(report)
    with open(output_dir / "evaluation_report.txt", "w", encoding="utf-8") as f:
        f.write(summary)
    report["threshold_sweep"].to_csv(output_dir / "threshold_sweep.csv", index=False)
    report["decision_curve"].to_csv(output_dir / "decision_curve.csv", index=False)
    report["high_risk_enrichment"].to_csv(
        output_dir / "high_risk_enrichment.csv",
        index=False,
    )
    bootstrap_ci_rows(report).to_csv(output_dir / "bootstrap_ci.csv", index=False)
    subgroup_columns = [
        item.strip() for item in args.subgroup_columns.split(",") if item.strip()
    ]
    if args.subgroups and subgroup_columns:
        subgroup_df = load_subgroup_table(args.subgroups)
        subgroup_metrics = compute_subgroup_metrics(
            subject_ids=binary_df["subject_id"].to_numpy(),
            labels=labels,
            probabilities=probabilities,
            subgroup_df=subgroup_df,
            columns=subgroup_columns,
            threshold=args.threshold,
        )
        subgroup_metrics.to_csv(output_dir / "subgroup_metrics.csv", index=False)
    np.savez(
        output_dir / "predictions.npz",
        subject_ids=binary_df["subject_id"].to_numpy(),
        labels=labels,
        probabilities=probabilities,
    )

    size_metadata = outcome_window_size_metadata(
        args.outcome,
        n_hours_start_include=args.n_hours_start_include,
        n_hours_end_include=args.n_hours_end_include,
        competing_outcome_path=args.competing_outcome,
        eligibility_path=args.eligibility,
        registry_start_date=args.registry_start_date,
        cohort=args.cohort,
        outcome_name=args.outcome_name,
    )
    size_metadata.update(
        {
            "n_test": int(len(binary_df)),
            "n_events_test": int(labels.sum()),
            "n_competing_events_test": int(
                (eval_df.get("event", pd.Series(dtype=int)) == 2).sum()
            ),
            "prevalence_test": float(labels.mean()),
        }
    )
    cfg = {
        "model_family": args.model_family,
        "training_stage": args.training_stage,
        "cohort": args.cohort,
        "outcome": args.outcome_name,
        "ipi_coverage": args.ipi_coverage,
        "evaluation_subset": args.evaluation_subset,
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
