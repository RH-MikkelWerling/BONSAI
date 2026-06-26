"""
Aggregate OPERA result artifacts into paper-ready tables.

Reads result.jsonl files emitted by evaluate.py/evaluate_joint.py and writes:
  - all_results.csv
  - results_wide.csv
  - model_summary.csv
  - optional delta table, e.g. joint_minus_per_cohort.csv
"""

import argparse
from pathlib import Path

from opera.evaluation.aggregation import (
    build_wide_metric_table,
    collect_result_rows,
    compute_model_delta_table,
    filter_results_for_paper_aggregates,
    pooled_transfer_effect,
    split_rarity_delta_tables,
    summarize_by_task_size,
    summarize_by_model,
    summarize_scale_ablation,
    validate_compatible_result_rows,
)
from opera.evaluation.subgroups import (
    build_subgroup_delta_table,
    collect_subgroup_metrics,
)


def _seed_stability(results, metrics):
    """Compute per-cell mean and SD across seeds for available metrics."""
    if results.empty or "seed" not in results.columns:
        return results.iloc[0:0].copy()
    metric_cols = [m for m in metrics if m in results.columns]
    if not metric_cols:
        return results.iloc[0:0].copy()
    group_cols = [
        col
        for col in (
            "cohort",
            "outcome",
            "outcome_window_hours",
            "model_family",
            "evaluation_subset",
        )
        if col in results.columns
    ]
    agg = {}
    for metric in metric_cols:
        agg[f"{metric}_mean"] = (metric, "mean")
        agg[f"{metric}_sd"] = (metric, "std")
    agg["n_seeds"] = ("seed", "nunique")
    return results.groupby(group_cols, dropna=False).agg(**agg).reset_index()


def main():
    parser = argparse.ArgumentParser(description="Aggregate OPERA result.jsonl files")
    parser.add_argument("--results_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--pattern", default="**/result.jsonl")
    parser.add_argument(
        "--metrics",
        default=("auroc,auprc,brier_score,ece,concordance_index,ipcw_auc,ipcw_brier"),
        help=(
            "Comma-separated metric names. Prefixes such as ipcw_auc also "
            "include horizon-specific columns like ipcw_auc_365d."
        ),
    )
    parser.add_argument("--baseline", help="Baseline model_family for delta table")
    parser.add_argument("--comparator", help="Comparator model_family for delta table")
    parser.add_argument(
        "--predictions_dir",
        default=None,
        help=(
            "Root of predictions.npz directory tree "
            "(cohort/outcome/model_variant/predictions.npz). "
            "When provided alongside --baseline and --comparator, also writes "
            "a patient-paired delta CSV with bootstrap/DeLong CIs and "
            "BH-corrected p-values per cell, requiring subject-ID alignment."
        ),
    )
    parser.add_argument(
        "--paired_n_bootstrap",
        type=int,
        default=2000,
        help="Bootstrap replicates for non-AUROC metrics in the paired delta table.",
    )
    parser.add_argument(
        "--subgroup_baseline",
        default="tabular_ehr",
        help="Baseline model_family for subgroup delta table.",
    )
    parser.add_argument(
        "--subgroup_comparator",
        default="opera",
        help="Comparator model_family for subgroup delta table.",
    )
    parser.add_argument(
        "--scale_col",
        default="pretraining_scale",
        help="Column used for pretraining-scale summaries when present.",
    )
    parser.add_argument("--rarity_plots", action="store_true")
    parser.add_argument(
        "--min_train_events",
        type=int,
        default=None,
        help="Mark real rare-cohort rows as supplement_only below this train-event count.",
    )
    parser.add_argument(
        "--min_test_events",
        type=int,
        default=None,
        help="Mark real rare-cohort rows as supplement_only below this test-event count.",
    )
    parser.add_argument(
        "--evaluation_subset",
        default="full",
        help="Patient subset used for aggregate paper tables.",
    )
    parser.add_argument(
        "--include_ipi",
        action="store_true",
        help="Include IPI rows in aggregate tables. Use only for dedicated IPI credibility outputs.",
    )
    parser.add_argument(
        "--allow_missing_evaluation_subset",
        action="store_true",
        help="Treat legacy rows without evaluation_subset as eligible for aggregate tables.",
    )
    parser.add_argument(
        "--strict_aggregation",
        action="store_true",
        help="Fail when aggregation validation reports error-severity issues.",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics = [item.strip() for item in args.metrics.split(",") if item.strip()]

    results = collect_result_rows(args.results_dir, pattern=args.pattern)
    if results.empty:
        print(f"No result.jsonl files found under {args.results_dir}")
        return

    results.to_csv(output_dir / "all_results.csv", index=False)

    validation = validate_compatible_result_rows(
        results,
        evaluation_subset=args.evaluation_subset,
        include_ipi=args.include_ipi,
    )
    if not validation.empty:
        validation.to_csv(output_dir / "aggregation_validation.csv", index=False)
        print(
            f"Aggregation validation warnings: {output_dir / 'aggregation_validation.csv'}"
        )
        if args.strict_aggregation and (validation["severity"] == "error").any():
            raise SystemExit(
                "Strict aggregation failed; see aggregation_validation.csv"
            )

    aggregate_results = filter_results_for_paper_aggregates(
        results,
        evaluation_subset=args.evaluation_subset,
        include_ipi=args.include_ipi,
        allow_missing_evaluation_subset=args.allow_missing_evaluation_subset,
    )
    aggregate_results.to_csv(output_dir / "paper_results.csv", index=False)
    if aggregate_results.empty:
        print(
            "No rows remain for aggregate paper tables after filtering. "
            "Raw rows are still available in all_results.csv."
        )
        return

    wide = build_wide_metric_table(aggregate_results, metrics=metrics)
    wide.to_csv(output_dir / "results_wide.csv", index=False)

    summary = summarize_by_model(aggregate_results, metrics=metrics)
    summary.to_csv(output_dir / "model_summary.csv", index=False)

    seed_stability = _seed_stability(aggregate_results, metrics)
    if not seed_stability.empty:
        seed_stability.to_csv(output_dir / "seed_stability.csv", index=False)

    task_size_summary = summarize_by_task_size(aggregate_results, metrics=metrics)
    if not task_size_summary.empty:
        task_size_summary.to_csv(output_dir / "task_size_summary.csv", index=False)

    subgroup_results = collect_subgroup_metrics(args.results_dir)
    if not subgroup_results.empty:
        subgroup_results = filter_results_for_paper_aggregates(
            subgroup_results,
            evaluation_subset=args.evaluation_subset,
            include_ipi=args.include_ipi,
            allow_missing_evaluation_subset=args.allow_missing_evaluation_subset,
        )
        subgroup_results.to_csv(output_dir / "subgroup_results.csv", index=False)
        subgroup_delta = build_subgroup_delta_table(
            subgroup_results,
            baseline_model=args.subgroup_baseline,
            comparator_model=args.subgroup_comparator,
            metrics=metrics,
        )
        if not subgroup_delta.empty:
            subgroup_delta.to_csv(output_dir / "subgroup_delta.csv", index=False)

    scale_summary = summarize_scale_ablation(
        aggregate_results,
        scale_col=args.scale_col,
        metrics=metrics,
    )
    if not scale_summary.empty:
        scale_summary.to_csv(output_dir / "pretraining_scale_summary.csv", index=False)

    if args.baseline and args.comparator:
        delta = compute_model_delta_table(
            wide,
            baseline=args.baseline,
            comparator=args.comparator,
            metrics=metrics,
        )
        name = f"{args.comparator}_minus_{args.baseline}.csv"
        delta.to_csv(output_dir / name, index=False)
        print(f"Delta table: {output_dir / name}")

        # Patient-paired delta: requires predictions.npz with subject-ID alignment.
        if args.predictions_dir:
            from opera.evaluation.significance import run_pairwise_comparisons

            paired = run_pairwise_comparisons(
                args.predictions_dir,
                contrasts=[
                    (
                        args.comparator,
                        args.baseline,
                        f"{args.comparator}_minus_{args.baseline}",
                    )
                ],
                metrics=["auroc"],
                alpha=0.05,
                n_bootstrap=args.paired_n_bootstrap,
                use_delong_for_auroc=True,
            )
            if not paired.empty:
                paired_name = (
                    f"paired_delta_{args.comparator}_minus_{args.baseline}.csv"
                )
                paired.to_csv(output_dir / paired_name, index=False)
                print(f"Paired delta table: {output_dir / paired_name}")
            else:
                print(
                    f"No paired predictions found for {args.comparator} vs "
                    f"{args.baseline} under {args.predictions_dir}"
                )

    if args.baseline:
        rarity_tables = split_rarity_delta_tables(
            aggregate_results,
            baseline=args.baseline,
            metrics=metrics,
            min_train_events=args.min_train_events,
            min_test_events=args.min_test_events,
        )
        for name, table in rarity_tables.items():
            if not table.empty:
                table.to_csv(output_dir / f"{name}.csv", index=False)
        if all(table.empty for table in rarity_tables.values()):
            print(
                "No rarity delta tables were written. Check that the baseline "
                f"model_family {args.baseline!r} exists for the same "
                "cohort/outcome/split/rarity_mode cells."
            )
        if args.rarity_plots:
            try:
                from opera.visualization.rarity_plots import write_rarity_plots

                write_rarity_plots(
                    rarity_tables["synthetic_rarity_pooled"],
                    rarity_tables["real_rarity_task_level"],
                    str(output_dir),
                )
            except Exception as exc:
                print(f"Rarity plotting failed: {exc}")

        # Pooled transfer effect: DerSimonian-Laird random-effects meta-analysis
        # over viable cells, stratified by cohort. Requires a rarity delta table.
        real_task = rarity_tables.get("real_rarity_task_level", None)
        synth_task_rows = rarity_tables.get("synthetic_rarity_task_level", None)
        for label, delta_tbl in [
            ("real", real_task),
            ("synthetic", synth_task_rows),
        ]:
            if delta_tbl is not None and not delta_tbl.empty:
                pool = pooled_transfer_effect(delta_tbl)
                if not pool.empty:
                    pool.to_csv(
                        output_dir / f"pooled_transfer_effect_{label}.csv",
                        index=False,
                    )

    print(
        f"Aggregated {len(aggregate_results)} compatible rows "
        f"({len(results)} raw rows) into {output_dir}"
    )


if __name__ == "__main__":
    main()
