"""
Create joint-vs-per-cohort comparison tables.

Inputs can be a directory containing evaluation result.jsonl files or a CSV
created by opera.run.aggregate_results.
"""

import argparse
from pathlib import Path

import pandas as pd

from opera.evaluation.aggregation import (
    build_joint_vs_per_cohort_table,
    collect_result_rows,
)


def load_results(results_dir: str = None, results_csv: str = None) -> pd.DataFrame:
    if results_csv:
        return pd.read_csv(results_csv)
    if results_dir:
        return collect_result_rows(results_dir)
    raise ValueError("Provide --results_dir or --results_csv")


def main():
    parser = argparse.ArgumentParser(description="Joint vs per-cohort OPERA comparison")
    parser.add_argument("--results_dir", help="Directory containing result.jsonl files")
    parser.add_argument("--results_csv", help="Long all_results.csv from aggregate_results")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--joint_model", default="joint")
    parser.add_argument("--per_cohort_model", default="per_cohort")
    parser.add_argument("--metrics", default="auroc,auprc,brier_score")
    parser.add_argument(
        "--cohort_sizes",
        help="Optional CSV with cohort/outcome and size metadata columns.",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics = [metric.strip() for metric in args.metrics.split(",") if metric.strip()]
    results = load_results(args.results_dir, args.results_csv)
    cohort_sizes = pd.read_csv(args.cohort_sizes) if args.cohort_sizes else None

    table = build_joint_vs_per_cohort_table(
        results,
        joint_model=args.joint_model,
        per_cohort_model=args.per_cohort_model,
        metrics=metrics,
        cohort_sizes=cohort_sizes,
    )
    table.to_csv(output_dir / "joint_vs_per_cohort.csv", index=False)

    delta_cols = [
        col
        for col in table.columns
        if col.startswith(f"{args.joint_model}_minus_{args.per_cohort_model}__")
    ]
    summary_rows = []
    for col in delta_cols:
        metric = col.split("__", 1)[1]
        vals = table[col].dropna()
        summary_rows.append(
            {
                "metric": metric,
                "n_comparisons": int(len(vals)),
                "median_delta": float(vals.median()) if len(vals) else float("nan"),
                "mean_delta": float(vals.mean()) if len(vals) else float("nan"),
                "n_joint_better": int((vals > 0).sum()),
                "n_joint_worse": int((vals < 0).sum()),
            }
        )
    pd.DataFrame(summary_rows).to_csv(
        output_dir / "joint_vs_per_cohort_summary.csv",
        index=False,
    )
    print(f"Wrote joint-vs-per-cohort outputs to {output_dir}")


if __name__ == "__main__":
    main()
