import json
from pathlib import Path
from typing import Iterable, List, Optional

import numpy as np
import pandas as pd

from opera.evaluation.metrics import (
    compute_calibration_metrics,
    compute_discrimination_metrics,
)


def load_subgroup_table(path: str) -> pd.DataFrame:
    source = Path(path)
    if source.suffix.lower() == ".parquet":
        return pd.read_parquet(source)
    return pd.read_csv(source)


def compute_subgroup_metrics(
    subject_ids: np.ndarray,
    labels: np.ndarray,
    probabilities: np.ndarray,
    subgroup_df: pd.DataFrame,
    columns: Iterable[str],
    threshold: float = 0.5,
) -> pd.DataFrame:
    """
    Compute discrimination/calibration metrics by subgroup column/value.

    Groups with fewer than two classes are retained with counts but metric values
    left missing, making data sparsity visible rather than silently dropping it.
    """
    eval_df = pd.DataFrame(
        {
            "subject_id": subject_ids.astype(int),
            "label": labels.astype(int),
            "probability": probabilities.astype(float),
        }
    )
    merged = eval_df.merge(subgroup_df, on="subject_id", how="left")
    rows: List[dict] = []

    for column in columns:
        if column not in merged.columns:
            rows.append(
                {
                    "subgroup_column": column,
                    "subgroup_value": "__missing_column__",
                    "n_total": 0,
                }
            )
            continue
        for value, group in merged.groupby(column, dropna=False):
            y = group["label"].to_numpy()
            p = group["probability"].to_numpy()
            row = {
                "subgroup_column": column,
                "subgroup_value": value,
                "n_total": int(len(group)),
                "n_positive": int(y.sum()) if len(group) else 0,
                "prevalence": float(y.mean()) if len(group) else float("nan"),
            }
            if len(group) >= 2 and len(np.unique(y)) > 1:
                row.update(compute_discrimination_metrics(y, p, threshold=threshold))
                cal = compute_calibration_metrics(y, p)
                for key in (
                    "brier_score",
                    "ece",
                    "mce",
                    "calibration_intercept",
                    "calibration_slope",
                    "hl_statistic",
                    "hl_pvalue",
                ):
                    row[key] = cal.get(key)
            rows.append(row)

    return pd.DataFrame(rows)


def read_subgroup_metrics(path: Path) -> pd.DataFrame:
    """Read one subgroup metrics CSV and attach sibling result metadata.

    The input is a `subgroup_metrics.csv` file emitted by an evaluator. The
    output adds cohort, outcome, model, split, seed, and subset metadata from
    the colocated `result.jsonl`, which makes cross-model subgroup comparisons
    auditable after aggregation.
    """
    rows = pd.read_csv(path)
    result_path = path.parent / "result.jsonl"
    if not result_path.exists():
        rows["subgroup_path"] = str(path)
        return rows
    with open(result_path) as f:
        line = next((item.strip() for item in f if item.strip()), "")
    if not line:
        rows["subgroup_path"] = str(path)
        return rows
    metadata = json.loads(line)
    for key in (
        "cohort",
        "outcome",
        "outcome_window_hours",
        "split",
        "seed",
        "model_family",
        "training_stage",
        "evaluation_subset",
        "ipi_coverage",
    ):
        rows[key] = metadata.get(key)
    rows["subgroup_path"] = str(path)
    return rows


def collect_subgroup_metrics(
    results_dir: str,
    pattern: str = "**/subgroup_metrics.csv",
) -> pd.DataFrame:
    """Collect subgroup metrics from an evaluation result tree.

    Inputs are evaluator output directories. The returned long table can be
    saved directly or passed to `build_subgroup_delta_table` to quantify whether
    OPERA improvements over tabular baselines persist within clinical strata.
    """
    frames = [read_subgroup_metrics(path) for path in Path(results_dir).glob(pattern)]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def build_subgroup_delta_table(
    subgroup_results: pd.DataFrame,
    baseline_model: str = "tabular_ehr",
    comparator_model: str = "opera",
    metrics: Optional[Iterable[str]] = None,
) -> pd.DataFrame:
    """Compute comparator-minus-baseline subgroup metric deltas.

    The input is the long subgroup table from `collect_subgroup_metrics`. The
    output keeps one row per cohort/outcome/subgroup/seed cell and adds delta
    columns such as `delta_auroc_vs_tabular_ehr`. This directly supports the
    scientific question of whether OPERA's gain holds across clinically
    important subgroups.
    """
    if subgroup_results.empty:
        return pd.DataFrame()
    if metrics is None:
        metrics = (
            "auroc",
            "auprc",
            "brier_score",
            "ece",
            "calibration_intercept",
            "calibration_slope",
        )
    metrics = [metric for metric in metrics if metric in subgroup_results.columns]
    if not metrics:
        return pd.DataFrame()

    key_cols = [
        col
        for col in (
            "cohort",
            "outcome",
            "outcome_window_hours",
            "split",
            "seed",
            "evaluation_subset",
            "subgroup_column",
            "subgroup_value",
        )
        if col in subgroup_results.columns
    ]
    baseline = subgroup_results[subgroup_results["model_family"] == baseline_model]
    comparator = subgroup_results[subgroup_results["model_family"] == comparator_model]
    if baseline.empty or comparator.empty:
        return pd.DataFrame()

    base_cols = (
        key_cols
        + metrics
        + [
            col
            for col in ("n_total", "n_positive", "prevalence")
            if col in baseline.columns
        ]
    )
    comp_cols = (
        key_cols
        + metrics
        + [
            col
            for col in ("n_total", "n_positive", "prevalence")
            if col in comparator.columns
        ]
    )
    base = baseline[base_cols].rename(
        columns={
            **{metric: f"baseline_{metric}" for metric in metrics},
            "n_total": "baseline_n_total",
            "n_positive": "baseline_n_positive",
            "prevalence": "baseline_prevalence",
        }
    )
    comp = comparator[comp_cols].rename(
        columns={
            **{metric: f"comparator_{metric}" for metric in metrics},
            "n_total": "comparator_n_total",
            "n_positive": "comparator_n_positive",
            "prevalence": "comparator_prevalence",
        }
    )
    out = comp.merge(base, on=key_cols, how="inner")
    out["baseline_model"] = baseline_model
    out["comparator_model"] = comparator_model
    for metric in metrics:
        out[f"delta_{metric}_vs_{baseline_model}"] = (
            out[f"comparator_{metric}"] - out[f"baseline_{metric}"]
        )
    return out
