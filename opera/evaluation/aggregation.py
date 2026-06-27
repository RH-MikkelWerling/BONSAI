import json
import logging
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd


logger = logging.getLogger(__name__)


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
            for item in results.loc[~missing_subset, "evaluation_subset"]
            .dropna()
            .unique()
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
        off_subset = results["evaluation_subset"].notna() & (
            results["evaluation_subset"].astype(str) != evaluation_subset
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
        col for col in [*INDEX_COLUMNS, "model_family"] if col in results.columns
    ]
    if duplicate_cols:
        duplicates = results.duplicated(subset=duplicate_cols, keep=False)
        if duplicates.any():
            warnings.append(
                {
                    "severity": "error",
                    "category": "duplicate_result_keys",
                    "message": "Duplicate model rows share the same aggregate key; paper aggregation is ambiguous.",
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


def collect_result_rows(
    results_dir: str, pattern: str = "**/result.jsonl"
) -> pd.DataFrame:
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

    metric_cols = []
    for metric in expanded_metrics:
        base_col = f"{baseline}__{metric}"
        comp_col = f"{comparator}__{metric}"
        if base_col in wide.columns and comp_col in wide.columns:
            metric_cols.extend([base_col, comp_col])
    if metric_cols:
        before = len(wide)
        wide = wide.dropna(subset=metric_cols)
        dropped = before - len(wide)
        if dropped:
            logger.warning(
                "compute_model_delta_table dropped %d row(s) with NaN baseline "
                "or comparator metric values (baseline=%r, comparator=%r).",
                dropped,
                baseline,
                comparator,
            )

    for _, row in wide.iterrows():
        out = {col: row[col] for col in id_cols}
        for metric in expanded_metrics:
            base_col = f"{baseline}__{metric}"
            comp_col = f"{comparator}__{metric}"
            if base_col in wide.columns and comp_col in wide.columns:
                out[f"{comparator}_minus_{baseline}__{metric}"] = round(
                    float(row[comp_col] - row[base_col]),
                    12,
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
        ).round(12)
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
                "rarity_size_bin"
                if "rarity_size_bin" in real.columns
                else "rarity_tier",
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
    delta_table = delta_table.dropna(subset=[delta_col])
    if delta_table.empty:
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

    binned = binned.dropna(subset=available_metrics)
    if binned.empty:
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

    results = results.dropna(subset=available_metrics)
    if results.empty:
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

    results = results.dropna(subset=available_metrics)
    if results.empty:
        return pd.DataFrame()

    agg_spec = {}
    for metric in available_metrics:
        agg_spec[f"{metric}_median"] = (metric, "median")
        agg_spec[f"{metric}_mean"] = (metric, "mean")
    agg_spec["n_results"] = (available_metrics[0], "count")
    return results.groupby(group_cols, dropna=False).agg(**agg_spec).reset_index()


def compute_paired_denominators(
    result_rows: pd.DataFrame,
    *,
    tolerance: float = 0.05,
    raise_on_mismatch: bool = False,
) -> pd.DataFrame:
    """Check that n_total is consistent across variants for each (cohort, outcome, evaluation_subset) cell.

    For valid paired comparisons, the denominator (number of test patients)
    must match across all model variants evaluating the same cell with the same
    evaluation_subset. Mismatches larger than `tolerance` (fractional) are
    flagged. Designed for the 'full' and 'ipi_complete' subsets separately.

    Returns a DataFrame with columns:
        cohort, outcome, evaluation_subset, n_variants, n_total_min,
        n_total_max, n_total_cv, mismatch_flag
    where mismatch_flag is True when the coefficient of variation of n_total
    across variants exceeds tolerance.
    """
    summary_columns = [
        "cohort",
        "outcome",
        "evaluation_subset",
        "n_variants",
        "n_total_min",
        "n_total_max",
        "n_total_cv",
        "mismatch_flag",
    ]
    if result_rows.empty or "n_total" not in result_rows.columns:
        return pd.DataFrame(columns=summary_columns)

    group_cols = ["cohort", "outcome", "evaluation_subset"]
    available_group_cols = [col for col in group_cols if col in result_rows.columns]
    if not available_group_cols:
        return pd.DataFrame(columns=summary_columns)

    rows: list[dict] = []
    for keys, group in result_rows.groupby(available_group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        key_map = dict(zip(available_group_cols, keys))
        n_total = pd.to_numeric(group["n_total"], errors="coerce").dropna()
        if n_total.empty:
            continue
        mean = float(n_total.mean())
        std = float(n_total.std(ddof=0))
        cv = std / mean if mean > 0 else 0.0
        record = {
            "cohort": key_map.get("cohort"),
            "outcome": key_map.get("outcome"),
            "evaluation_subset": key_map.get("evaluation_subset"),
            "n_variants": int(len(n_total)),
            "n_total_min": float(n_total.min()),
            "n_total_max": float(n_total.max()),
            "n_total_cv": cv,
            "mismatch_flag": bool(cv > tolerance),
        }
        rows.append(record)

    summary = pd.DataFrame(rows, columns=summary_columns)

    if raise_on_mismatch and not summary.empty:
        flagged = summary[summary["mismatch_flag"]]
        if not flagged.empty:
            cells = ", ".join(
                f"(cohort={row.cohort!r}, outcome={row.outcome!r}, "
                f"evaluation_subset={row.evaluation_subset!r}, "
                f"n_total_min={row.n_total_min:g}, n_total_max={row.n_total_max:g}, "
                f"cv={row.n_total_cv:.4f})"
                for row in flagged.itertuples(index=False)
            )
            raise ValueError(
                "Paired denominator mismatch exceeds tolerance "
                f"({tolerance}) for cells: {cells}"
            )

    return summary


def collect_label_split_summaries(sweep_result_dir: Path) -> pd.DataFrame:
    """Collect all label_split_summary.csv files from a sweep result directory tree.

    Walks sweep_result_dir recursively, finds every label_split_summary.csv,
    reads each, and returns a concatenated DataFrame with an added column
    `cell_dir` (the parent directory path string) so rows can be traced back.

    Returns an empty DataFrame if no files are found.
    """
    sweep_result_dir = Path(sweep_result_dir)
    frames: list[pd.DataFrame] = []
    for path in sorted(sweep_result_dir.rglob("label_split_summary.csv")):
        frame = pd.read_csv(path)
        frame["cell_dir"] = str(path.parent)
        frames.append(frame)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def build_consort_table(label_summaries: pd.DataFrame) -> pd.DataFrame:
    """Build a CONSORT-like cohort-flow table from collected split summaries.

    Input is the output of collect_label_split_summaries().
    Output has one row per (cohort, outcome, split) and columns:
      cohort, outcome, split,
      n_subjects,           # total subjects in split
      n_labelled,           # subjects with a valid label (not excluded by eligibility)
      n_positive,           # events
      n_negative,           # non-events with full follow-up
      n_insufficient_followup,  # subjects censored before window close
      prevalence,           # n_positive / n_labelled
      n_model_variants,     # how many model variants have results for this cell

    When multiple model variants contributed rows for the same (cohort, outcome, split),
    the label columns are taken from the median (or the first, if all identical).
    n_model_variants counts distinct model_family values.

    Returns empty DataFrame if input is empty.
    """
    output_columns = [
        "cohort",
        "outcome",
        "split",
        "n_subjects",
        "n_labelled",
        "n_positive",
        "n_negative",
        "n_insufficient_followup",
        "prevalence",
        "n_model_variants",
    ]
    if label_summaries.empty:
        return pd.DataFrame(columns=output_columns)

    count_cols = [
        "n_subjects",
        "n_labelled",
        "n_positive",
        "n_negative",
        "n_insufficient_followup",
    ]
    group_cols = ["cohort", "outcome", "split"]
    available_group_cols = [col for col in group_cols if col in label_summaries.columns]
    if not available_group_cols:
        return pd.DataFrame(columns=output_columns)

    rows: list[dict] = []
    for keys, group in label_summaries.groupby(available_group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        key_map = dict(zip(available_group_cols, keys))
        record: dict = {col: key_map.get(col) for col in group_cols}

        for col in count_cols:
            if col in group.columns:
                values = pd.to_numeric(group[col], errors="coerce").dropna()
                record[col] = float(values.median()) if not values.empty else None
            else:
                record[col] = None

        n_labelled = record.get("n_labelled")
        n_positive = record.get("n_positive")
        if n_labelled is not None and n_positive is not None and float(n_labelled) > 0:
            record["prevalence"] = float(n_positive) / float(n_labelled)
        else:
            record["prevalence"] = None

        if "model_family" in group.columns:
            record["n_model_variants"] = int(group["model_family"].nunique(dropna=True))
        else:
            record["n_model_variants"] = int(len(group))

        rows.append(record)

    return pd.DataFrame(rows, columns=output_columns)


def check_competing_event_denominator_consistency(
    result_rows: pd.DataFrame,
    *,
    competing_event_col: str = "n_competing_events_test",
    tolerance: float = 0.01,
) -> list[str]:
    """Check that n_competing_events_test is consistent across variants per cell.

    Returns a list of warning strings. Empty list means consistent.
    A mismatch here means two variants removed different competing-event subjects
    from the test set -- pairing them would be incorrect.

    Only checks cells where competing_event_col is non-null for at least one variant.
    """
    if result_rows.empty or competing_event_col not in result_rows.columns:
        return []

    group_cols = ["cohort", "outcome"]
    available_group_cols = [col for col in group_cols if col in result_rows.columns]
    if not available_group_cols:
        return []

    messages: list[str] = []
    for keys, group in result_rows.groupby(available_group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        key_map = dict(zip(available_group_cols, keys))
        values = pd.to_numeric(group[competing_event_col], errors="coerce").dropna()
        if values.empty:
            continue
        mean = float(values.mean())
        std = float(values.std(ddof=0))
        cv = std / mean if mean > 0 else 0.0
        if cv > tolerance:
            messages.append(
                f"Inconsistent {competing_event_col} for "
                f"cohort={key_map.get('cohort')!r}, outcome={key_map.get('outcome')!r}: "
                f"ranges {float(values.min()):g}-{float(values.max()):g} across "
                f"{int(len(values))} variants (CV={cv:.4f} > {tolerance}). "
                "Variants removed different competing-event subjects; pairing is incorrect."
            )
    return messages


def _dersimonlain_re(thetas: np.ndarray, variances: np.ndarray) -> dict:
    """DerSimonian-Laird random-effects meta-analysis for one stratum.

    Returns a dict with pooled_effect, ci_lower, ci_upper, tau2, i2, Q, df,
    k_cells.  With a single cell the function returns the cell value with
    tau2=0 and I2=0.
    """
    k = len(thetas)
    if k == 0:
        return {}
    if k == 1:
        se = float(np.sqrt(variances[0]))
        return {
            "pooled_effect": float(thetas[0]),
            "ci_lower": float(thetas[0] - 1.96 * se),
            "ci_upper": float(thetas[0] + 1.96 * se),
            "tau2": 0.0,
            "i2": 0.0,
            "Q": 0.0,
            "df": 0,
            "k_cells": 1,
        }
    w = 1.0 / variances
    theta_fe = float(np.sum(w * thetas) / np.sum(w))
    Q = float(np.sum(w * (thetas - theta_fe) ** 2))
    df = k - 1
    c = float(np.sum(w) - np.sum(w**2) / np.sum(w))
    tau2 = max(0.0, (Q - df) / c) if c > 0 else 0.0
    w_re = 1.0 / (variances + tau2)
    theta_dl = float(np.sum(w_re * thetas) / np.sum(w_re))
    se_dl = float(np.sqrt(1.0 / np.sum(w_re)))
    i2 = max(0.0, (Q - df) / Q * 100.0) if Q > 0 else 0.0
    return {
        "pooled_effect": theta_dl,
        "ci_lower": theta_dl - 1.96 * se_dl,
        "ci_upper": theta_dl + 1.96 * se_dl,
        "tau2": tau2,
        "i2": i2,
        "Q": Q,
        "df": df,
        "k_cells": k,
    }


def pooled_transfer_effect(
    delta_table: pd.DataFrame,
    delta_col: str = "delta_auroc_vs_baseline",
    se_col: Optional[str] = None,
    ci_lower_col: Optional[str] = None,
    ci_upper_col: Optional[str] = None,
    stratify_col: Optional[str] = "cohort",
) -> pd.DataFrame:
    """DerSimonian-Laird random-effects pooled transfer-effect estimate.

    Produces a pooled estimate with tau^2 (between-cell heterogeneity), I^2,
    and a 95% CI.  Each row of *delta_table* is one study (cohort/outcome cell).

    **Scientific validity of tau^2 and I^2**

    The DL estimator decomposes total variance into within-cell sampling
    variance (how noisy each AUROC estimate is given the cell's n) and
    between-cell heterogeneity (how much true transfer effects vary across
    tasks).  This decomposition is only meaningful when within-cell SE is
    correctly calibrated.

    *Recommended path (``se_col`` or ``ci_lower_col``/``ci_upper_col``):*
    Supply the paired-bootstrap SE from ``run_pairwise_comparisons``.  Join
    the paired delta CSV on (cohort, outcome) and pass the derived SE column.
    The convenience params ``ci_lower_col`` / ``ci_upper_col`` let you pass
    the CI bounds directly; SE is derived as ``(upper - lower) / (2 × 1.96)``.
    With calibrated SE, tau^2 and I^2 reflect genuine heterogeneity.

    *Equal-weight fallback (no SE supplied):*
    When ``se_col``, ``ci_lower_col``, and ``ci_upper_col`` are all ``None``,
    all cells are given the same within-cell variance, set to the empirical
    cross-cell SD.  This forces Q = df, so ``tau2 = 0`` and ``I2 = 0`` by
    construction — a numerical artifact, not a scientific finding.  The pooled
    estimate reduces to the simple mean.  ``se_source="equal_weight_fallback"``
    is set in the output, and a ``UserWarning`` is emitted.  **Do not interpret
    tau^2 or I^2 from fallback rows as evidence of homogeneity.**

    *Within-cohort dependence:*
    Multiple outcomes within the same cohort share a patient pool and are
    therefore positively correlated.  DL treats cells as independent and will
    underestimate uncertainty when within-cohort correlation is high.  The
    ``stratum``-level rows pool within each cohort first, which partially
    mitigates this; the ``_all_`` row should be read with the caveat in mind.

    Parameters
    ----------
    delta_table:
        Per-cell delta table such as that produced by
        ``build_delta_vs_baseline_table``.  Must contain ``delta_col``.
    delta_col:
        Column holding the per-cell OPERA-minus-baseline delta.
    se_col:
        Column holding the per-cell standard error (e.g. derived from
        bootstrap CIs).  Takes precedence over ``ci_lower_col``/``ci_upper_col``.
    ci_lower_col, ci_upper_col:
        Columns holding 95% CI bounds (e.g. ``delta_lower`` / ``delta_upper``
        from the paired delta CSV).  SE is derived as
        ``(ci_upper - ci_lower) / (2 × 1.96)``.  Used only when ``se_col``
        is ``None``.
    stratify_col:
        Column used to stratify the pooling (``"cohort"`` by default).  One
        pooled row is produced per stratum, plus an overall row with
        ``stratum="_all_"``.  Pass ``None`` to skip per-stratum rows.

    Returns
    -------
    DataFrame with columns:
        stratum, pooled_effect, ci_lower, ci_upper, tau2, i2, Q, df, k_cells,
        se_source, model_family (when present in *delta_table*).
    """
    import warnings

    if delta_table.empty or delta_col not in delta_table.columns:
        return pd.DataFrame()

    tbl = delta_table.dropna(subset=[delta_col]).copy()
    if tbl.empty:
        return pd.DataFrame()

    # Resolve SE source priority: se_col > CI-derived > equal-weight fallback.
    _active_se_col: Optional[str] = None
    _se_source: str = "equal_weight_fallback"

    if se_col is not None and se_col in tbl.columns:
        tbl = tbl.dropna(subset=[se_col])
        tbl = tbl[tbl[se_col] > 0]
        if tbl.empty:
            return pd.DataFrame()
        _active_se_col = se_col
        _se_source = "provided_se_col"
    elif (
        ci_lower_col is not None
        and ci_upper_col is not None
        and ci_lower_col in tbl.columns
        and ci_upper_col in tbl.columns
    ):
        tbl = tbl.dropna(subset=[ci_lower_col, ci_upper_col])
        ci_se = (tbl[ci_upper_col] - tbl[ci_lower_col]) / (2.0 * 1.96)
        tbl = tbl[ci_se > 0].copy()
        if tbl.empty:
            return pd.DataFrame()
        tbl["_ci_derived_se"] = (
            (tbl[ci_upper_col] - tbl[ci_lower_col]) / (2.0 * 1.96)
        )
        _active_se_col = "_ci_derived_se"
        _se_source = "paired_bootstrap_ci"
    else:
        warnings.warn(
            "pooled_transfer_effect: no se_col or CI columns provided. "
            "Falling back to equal-weight pooling (empirical cross-cell SD as "
            "within-cell SE). tau2 and i2 will be 0 by construction and must "
            "NOT be interpreted as evidence of homogeneity. "
            "Pass ci_lower_col/ci_upper_col from the paired bootstrap delta "
            "CSV to obtain meaningful heterogeneity statistics.",
            UserWarning,
            stacklevel=2,
        )

    def _pool_group(group: pd.DataFrame) -> dict:
        thetas = group[delta_col].to_numpy(dtype=float)
        if _active_se_col is not None and _active_se_col in group.columns:
            variances = group[_active_se_col].to_numpy(dtype=float) ** 2
        else:
            # Equal-weight: empirical SD as common within-cell SE forces Q = df.
            empirical_sd = float(np.std(thetas, ddof=1)) if len(thetas) > 1 else 1e-4
            empirical_sd = max(empirical_sd, 1e-6)
            variances = np.full(len(thetas), empirical_sd**2)
        return _dersimonlain_re(thetas, variances)

    rows = []

    # Per-stratum rows.
    if stratify_col is not None and stratify_col in tbl.columns:
        for stratum, grp in tbl.groupby(stratify_col, dropna=False):
            result = _pool_group(grp)
            if result:
                row = {"stratum": str(stratum), "se_source": _se_source}
                if "model_family" in grp.columns:
                    families = grp["model_family"].dropna().unique()
                    row["model_family"] = families[0] if len(families) == 1 else "mixed"
                row.update(result)
                rows.append(row)

    # Overall pooled row.
    overall = _pool_group(tbl)
    if overall:
        row = {"stratum": "_all_", "se_source": _se_source}
        if "model_family" in tbl.columns:
            families = tbl["model_family"].dropna().unique()
            row["model_family"] = families[0] if len(families) == 1 else "mixed"
        row.update(overall)
        rows.append(row)

    return pd.DataFrame(rows)


def validate_result_rows_denominators(
    result_rows: pd.DataFrame,
    *,
    tolerance: float = 0.05,
) -> list[str]:
    """Return a list of human-readable mismatch warnings for paper tables.

    Returns empty list if all denominators are consistent.
    """
    summary = compute_paired_denominators(result_rows, tolerance=tolerance)
    if summary.empty:
        return []
    flagged = summary[summary["mismatch_flag"]]
    messages: list[str] = []
    for row in flagged.itertuples(index=False):
        messages.append(
            f"Inconsistent denominator for cohort={row.cohort!r}, "
            f"outcome={row.outcome!r}, evaluation_subset={row.evaluation_subset!r}: "
            f"n_total ranges {row.n_total_min:g}-{row.n_total_max:g} across "
            f"{row.n_variants} variants (CV={row.n_total_cv:.4f} > {tolerance})."
        )
    return messages
