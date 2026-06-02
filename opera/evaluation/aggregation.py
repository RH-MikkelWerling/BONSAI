import json
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

import pandas as pd


INDEX_COLUMNS = [
    "cohort",
    "outcome",
    "outcome_window_hours",
    "split",
    "seed",
    "training_fraction",
    "rarity_mode",
    "evaluation_subset",
]


def validate_compatible_result_rows(
    results: pd.DataFrame,
    evaluation_subset: str = "full",
    include_ipi: bool = False,
) -> pd.DataFrame:
    """Report rows that would make paper aggregates scientifically ambiguous.

    Inputs are raw result rows from one or more ``result.jsonl`` files. The
    output is a warning table with one row per detected issue. The scientific
    purpose is to keep full-cohort model comparisons separate from IPI-complete
    credibility analyses, because those rows represent different patient
    populations.
    """
    warnings: list[dict] = []
    if results.empty:
        return pd.DataFrame(warnings)

    if "evaluation_subset" not in results.columns:
        warnings.append(
            {
                "severity": "error",
                "category": "missing_column",
                "message": "evaluation_subset is missing; aggregate rows cannot be subset-filtered.",
                "n_rows": int(len(results)),
            }
        )
    else:
        missing_subset = results["evaluation_subset"].isna() | (
            results["evaluation_subset"].astype(str).str.len() == 0
        )
        if missing_subset.any():
            warnings.append(
                {
                    "severity": "error",
                    "category": "missing_evaluation_subset",
                    "message": "Rows without evaluation_subset are excluded from paper-safe aggregates.",
                    "n_rows": int(missing_subset.sum()),
                }
            )
        subsets = sorted(
            str(item)
            for item in results.loc[~missing_subset, "evaluation_subset"].dropna().unique()
        )
        if len(subsets) > 1:
            warnings.append(
                {
                    "severity": "warning",
                    "category": "mixed_evaluation_subsets",
                    "message": "Multiple evaluation subsets are present in raw results.",
                    "values": ",".join(subsets),
                    "n_rows": int(len(results)),
                }
            )
        off_subset = (
            results["evaluation_subset"].notna()
            & (results["evaluation_subset"].astype(str) != evaluation_subset)
        )
        if off_subset.any():
            warnings.append(
                {
                    "severity": "info",
                    "category": "excluded_subset",
                    "message": f"Rows outside evaluation_subset={evaluation_subset!r} are excluded from paper aggregates.",
                    "n_rows": int(off_subset.sum()),
                }
            )

    if not include_ipi and "model_family" in results.columns:
        ipi_rows = results["model_family"].astype(str).str.lower().eq("ipi")
        if ipi_rows.any():
            warnings.append(
                {
                    "severity": "info",
                    "category": "excluded_ipi",
                    "message": "IPI rows are excluded from aggregate foundation-model summaries.",
                    "n_rows": int(ipi_rows.sum()),
                }
            )

    duplicate_cols = [
        col
        for col in [*INDEX_COLUMNS, "model_family"]
        if col in results.columns
    ]
    if duplicate_cols:
        duplicates = results.duplicated(subset=duplicate_cols, keep=False)
        if duplicates.any():
            warnings.append(
                {
                    "severity": "warning",
                    "category": "duplicate_result_keys",
                    "message": "Duplicate model rows share the same aggregate key; pivot tables will use the first value.",
                    "n_rows": int(duplicates.sum()),
                    "key_columns": ",".join(duplicate_cols),
                }
            )
    return pd.DataFrame(warnings)


def filter_results_for_paper_aggregates(
    results: pd.DataFrame,
    evaluation_subset: str = "full",
    include_ipi: bool = False,
    allow_missing_evaluation_subset: bool = False,
) -> pd.DataFrame:
    """Return rows compatible with main paper aggregate comparisons.

    Inputs are raw result rows. The returned DataFrame keeps one patient subset
    only, excludes IPI by default, and leaves the raw data untouched. This
    prevents aggregate figures from mixing full-cohort rows with IPI-complete
    rows, which would otherwise compare models on different patients.
    """
    if results.empty:
        return results.copy()
    out = results.copy()
    if "evaluation_subset" not in out.columns:
        out["evaluation_subset"] = None
    if allow_missing_evaluation_subset:
        subset_mask = out["evaluation_subset"].isna() | (
            out["evaluation_subset"].astype(str) == evaluation_subset
        )
    else:
        subset_mask = out["evaluation_subset"].astype(str) == evaluation_subset
    out = out[subset_mask].copy()
    if not include_ipi and "model_family" in out.columns:
        out = out[~out["model_family"].astype(str).str.lower().eq("ipi")].copy()
    return out.reset_index(drop=True)


def expand_metric_columns(results: pd.DataFrame, metrics: Iterable[str]) -> list[str]:
    """Resolve requested metric names, including horizon-specific prefixes.

    For example, requesting ``ipcw_auc`` includes columns such as
    ``ipcw_auc_365d`` and ``ipcw_auc_730d`` when present.
    """
    if results.empty:
        return []
    columns = list(results.columns)
    expanded: list[str] = []
    for metric in metrics:
        if metric in results.columns:
            expanded.append(metric)
        expanded.extend(
            col
            for col in columns
            if col.startswith(f"{metric}_") and col not in expanded
        )
    return expanded


def read_result_jsonl(path: Path) -> List[dict]:
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                row = json.loads(line)
                row.setdefault("result_path", str(path))
                rows.append(row)
    return rows


def collect_result_rows(results_dir: str, pattern: str = "**/result.jsonl") -> pd.DataFrame:
    rows = []
    for path in Path(results_dir).glob(pattern):
        rows.extend(read_result_jsonl(path))
    return pd.DataFrame(rows)


def build_wide_metric_table(
    results: pd.DataFrame,
    metrics: Iterable[str] = ("auroc", "auprc", "brier_score"),
    model_col: str = "model_family",
) -> pd.DataFrame:
    if results.empty:
        return results

    available_metrics = expand_metric_columns(results, metrics)
    index_cols = [
        col
        for col in INDEX_COLUMNS
        if col in results.columns and not results[col].isna().all()
    ]
    table = results.pivot_table(
        index=index_cols,
        columns=model_col,
        values=available_metrics,
        aggfunc="first",
    )
    table.columns = [f"{model}__{metric}" for metric, model in table.columns]
    return table.reset_index()


def compute_model_delta_table(
    wide: pd.DataFrame,
    baseline: str,
    comparator: str,
    metrics: Iterable[str] = ("auroc", "auprc", "brier_score"),
) -> pd.DataFrame:
    rows = []
    if wide.empty:
        return pd.DataFrame(rows)

    id_cols = [col for col in INDEX_COLUMNS if col in wide.columns]
    expanded_metrics = []
    for metric in metrics:
        matches = [
            col.split("__", 1)[1]
            for col in wide.columns
            if "__" in col and col.split("__", 1)[1] == metric
        ]
        matches.extend(
            col.split("__", 1)[1]
            for col in wide.columns
            if "__" in col and col.split("__", 1)[1].startswith(f"{metric}_")
        )
        expanded_metrics.extend(matches or [metric])
    expanded_metrics = list(dict.fromkeys(expanded_metrics))

    for _, row in wide.iterrows():
        out = {col: row[col] for col in id_cols}
        for metric in expanded_metrics:
            base_col = f"{baseline}__{metric}"
            comp_col = f"{comparator}__{metric}"
            if base_col in wide.columns and comp_col in wide.columns:
                out[f"{comparator}_minus_{baseline}__{metric}"] = (
                    row[comp_col] - row[base_col]
                )
        rows.append(out)
    return pd.DataFrame(rows)


def build_delta_vs_baseline_table(
    results: pd.DataFrame,
    baseline: str,
    metrics: Iterable[str] = ("auroc",),
) -> pd.DataFrame:
    """
    Long per-task model-minus-baseline table.

    Rarity analyses use this as their primary table so synthetic label-scarcity
    rows and real rare-cohort rows can keep their own metadata.
    """
    if results.empty or "model_family" not in results.columns:
        return pd.DataFrame()

    metrics = expand_metric_columns(results, metrics)
    if not metrics:
        return pd.DataFrame()

    key_cols = [
        col
        for col in INDEX_COLUMNS
        if col in results.columns and col not in ("seed", "training_fraction")
    ]
    if "training_fraction" in results.columns:
        key_cols.append("training_fraction")

    baseline_cols = key_cols + metrics
    baseline_rows = results[results["model_family"] == baseline][baseline_cols].copy()
    if baseline_rows.empty:
        return pd.DataFrame()
    baseline_rows = baseline_rows.groupby(key_cols, dropna=False, as_index=False).agg(
        {metric: "median" for metric in metrics}
    )
    baseline_rows = baseline_rows.rename(
        columns={metric: f"baseline_{metric}" for metric in metrics}
    )

    merged = results.merge(baseline_rows, on=key_cols, how="inner")
    merged = merged[merged["model_family"] != baseline].copy()
    if merged.empty:
        return pd.DataFrame()
    merged["baseline_model"] = baseline
    for metric in metrics:
        merged[f"delta_{metric}_vs_baseline"] = (
            merged[metric] - merged[f"baseline_{metric}"]
        )
    return merged


def build_joint_vs_per_cohort_table(
    results: pd.DataFrame,
    joint_model: str = "joint",
    per_cohort_model: str = "per_cohort",
    metrics: Iterable[str] = ("auroc", "auprc", "brier_score"),
    cohort_sizes: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Build a direct joint-minus-per-cohort table.

    `results` should be the long result schema emitted by evaluation. Optional
    `cohort_sizes` may contain cohort/outcome plus columns such as n_subjects or
    n_train to make the comparison table self-describing.
    """
    wide = build_wide_metric_table(results, metrics=metrics)
    delta = compute_model_delta_table(
        wide,
        baseline=per_cohort_model,
        comparator=joint_model,
        metrics=metrics,
    )
    if cohort_sizes is not None and not cohort_sizes.empty:
        join_cols = [
            col
            for col in ("cohort", "outcome", "outcome_window_hours")
            if col in delta.columns and col in cohort_sizes.columns
        ]
        if join_cols:
            delta = delta.merge(cohort_sizes, on=join_cols, how="left")
    metric_cols = [col for col in delta.columns if col.endswith("__auroc")]
    if metric_cols:
        delta = delta.sort_values(metric_cols[0], ascending=False)
    return delta


def add_task_size_bins(
    results: pd.DataFrame,
    size_col: str = "n_total",
    bins: Optional[Sequence[float]] = None,
    labels: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """
    Add task-size bins for rarity/label-efficiency summaries.

    The default bins are deliberately broad and interpretable for paper tables.
    """
    if results.empty or size_col not in results.columns:
        return results.copy()

    if bins is None:
        bins = [0, 100, 500, 1000, 5000, float("inf")]
    if labels is None:
        labels = ["<100", "100-499", "500-999", "1k-4,999", ">=5k"]

    out = results.copy().reset_index(drop=True)
    out["task_size_bin"] = pd.cut(
        out[size_col],
        bins=bins,
        labels=labels,
        include_lowest=True,
        right=False,
    )
    return out


def add_rarity_size_bins(
    results: pd.DataFrame,
    size_col: str = "n_train",
) -> pd.DataFrame:
    """Add broad train-size bins for real rare-cohort summaries."""
    return add_task_size_bins(
        results,
        size_col=size_col,
        bins=[0, 25, 50, 100, 250, 500, float("inf")],
        labels=["<25", "25-49", "50-99", "100-249", "250-499", ">=500"],
    ).rename(columns={"task_size_bin": "rarity_size_bin"})


def mark_rare_cohort_stability(
    results: pd.DataFrame,
    min_train_events: Optional[int] = None,
    min_test_events: Optional[int] = None,
) -> pd.DataFrame:
    """
    Mark very small real-rare cells for supplement-only handling.

    Rows are never dropped here. The flags allow main-paper summaries and plots
    to filter unstable cells explicitly while preserving all raw results.
    """
    out = results.copy().reset_index(drop=True)
    out["supplement_only"] = False
    reasons = [[] for _ in range(len(out))]

    if min_train_events is not None and "n_events_train" in out.columns:
        mask = out["n_events_train"].fillna(0) < min_train_events
        out.loc[mask, "supplement_only"] = True
        for idx in list(out.index[mask]):
            reasons[idx].append("min_train_events")

    if min_test_events is not None and "n_events_test" in out.columns:
        mask = out["n_events_test"].fillna(0) < min_test_events
        out.loc[mask, "supplement_only"] = True
        for idx in list(out.index[mask]):
            reasons[idx].append("min_test_events")

    out["supplement_only_reason"] = [";".join(items) for items in reasons]
    return out


def split_rarity_delta_tables(
    results: pd.DataFrame,
    baseline: str,
    metrics: Iterable[str] = ("auroc",),
    min_train_events: Optional[int] = None,
    min_test_events: Optional[int] = None,
) -> dict:
    """
    Build rarity-specific delta tables without mixing rarity regimes.

    synthetic = common disease cells with reduced labels.
    real = genuinely small cohorts or small cohort-outcome cells.
    """
    delta = build_delta_vs_baseline_table(results, baseline=baseline, metrics=metrics)
    if delta.empty:
        return {
            "synthetic_rarity_task_level": pd.DataFrame(),
            "synthetic_rarity_pooled": pd.DataFrame(),
            "real_rarity_task_level": pd.DataFrame(),
            "real_rarity_pooled": pd.DataFrame(),
            "real_rarity_pooled_main": pd.DataFrame(),
        }

    if "rarity_mode" not in delta.columns:
        delta["rarity_mode"] = "none"
    synthetic = delta[delta["rarity_mode"] == "synthetic"].copy()
    real = delta[delta["rarity_mode"] == "real"].copy()

    if not real.empty and "n_train" in real.columns:
        real = add_rarity_size_bins(real, size_col="n_train")
    if not real.empty:
        real = mark_rare_cohort_stability(
            real,
            min_train_events=min_train_events,
            min_test_events=min_test_events,
        )
    real_main = (
        real[~real["supplement_only"]].copy()
        if "supplement_only" in real.columns
        else real.copy()
    )

    return {
        "synthetic_rarity_task_level": synthetic,
        "synthetic_rarity_pooled": summarize_rarity_deltas(
            synthetic,
            group_cols=["model_family", "training_fraction"],
            metric="auroc",
        ),
        "real_rarity_task_level": real,
        "real_rarity_pooled": summarize_rarity_deltas(
            real,
            group_cols=[
                "model_family",
                "rarity_size_bin" if "rarity_size_bin" in real.columns else "rarity_tier",
            ],
            metric="auroc",
        ),
        "real_rarity_pooled_main": summarize_rarity_deltas(
            real_main,
            group_cols=[
                "model_family",
                "rarity_size_bin"
                if "rarity_size_bin" in real_main.columns
                else "rarity_tier",
            ],
            metric="auroc",
        ),
    }


def summarize_rarity_deltas(
    delta_table: pd.DataFrame,
    group_cols: Iterable[str],
    metric: str = "auroc",
) -> pd.DataFrame:
    delta_col = f"delta_{metric}_vs_baseline"
    if delta_table.empty or delta_col not in delta_table.columns:
        return pd.DataFrame()
    group_cols = [col for col in group_cols if col in delta_table.columns]
    if not group_cols:
        return pd.DataFrame()
    return (
        delta_table.groupby(group_cols, dropna=False)
        .agg(
            median_delta_auroc=(delta_col, "median"),
            lower_delta_auroc=(delta_col, lambda x: float(x.quantile(0.025))),
            upper_delta_auroc=(delta_col, lambda x: float(x.quantile(0.975))),
            n_tasks=("outcome", "count"),
        )
        .reset_index()
    )


def summarize_by_task_size(
    results: pd.DataFrame,
    metrics: Iterable[str] = ("auroc", "auprc", "brier_score"),
    model_col: str = "model_family",
) -> pd.DataFrame:
    """Summarize performance by model and labelled task-size bin."""
    if results.empty:
        return results
    binned = add_task_size_bins(results)
    if "task_size_bin" not in binned.columns:
        return pd.DataFrame()

    available_metrics = expand_metric_columns(binned, metrics)
    if not available_metrics:
        return pd.DataFrame()

    agg_spec = {}
    for metric in available_metrics:
        agg_spec[f"{metric}_median"] = (metric, "median")
        agg_spec[f"{metric}_mean"] = (metric, "mean")
    agg_spec["n_results"] = (available_metrics[0], "count")
    group_cols = [model_col, "task_size_bin"]
    return binned.groupby(group_cols, dropna=False).agg(**agg_spec).reset_index()


def summarize_scale_ablation(
    results: pd.DataFrame,
    scale_col: str = "pretraining_scale",
    metrics: Iterable[str] = ("auroc", "auprc", "brier_score"),
) -> pd.DataFrame:
    """
    Summarize runs that differ by upstream pretraining scale/scope.

    The caller controls `scale_col`; common values are `pretraining_scale`,
    `source_checkpoint`, or `model_family`.
    """
    if results.empty or scale_col not in results.columns:
        return pd.DataFrame()
    available_metrics = expand_metric_columns(results, metrics)
    if not available_metrics:
        return pd.DataFrame()

    group_cols = [scale_col]
    for optional in ("training_stage", "training_fraction"):
        if optional in results.columns:
            group_cols.append(optional)

    agg_spec = {}
    for metric in available_metrics:
        agg_spec[f"{metric}_median"] = (metric, "median")
        agg_spec[f"{metric}_mean"] = (metric, "mean")
    agg_spec["n_results"] = (available_metrics[0], "count")
    return results.groupby(group_cols, dropna=False).agg(**agg_spec).reset_index()


def summarize_by_model(
    results: pd.DataFrame,
    metrics: Iterable[str] = ("auroc", "auprc", "brier_score"),
    group_cols: Optional[Iterable[str]] = None,
) -> pd.DataFrame:
    if results.empty:
        return results
    if group_cols is None:
        group_cols = ["model_family", "training_stage"]
    group_cols = [col for col in group_cols if col in results.columns]
    available_metrics = expand_metric_columns(results, metrics)
    if not group_cols or not available_metrics:
        return pd.DataFrame()

    agg_spec = {}
    for metric in available_metrics:
        agg_spec[f"{metric}_median"] = (metric, "median")
        agg_spec[f"{metric}_mean"] = (metric, "mean")
    agg_spec["n_results"] = (available_metrics[0], "count")
    return results.groupby(group_cols, dropna=False).agg(**agg_spec).reset_index()
