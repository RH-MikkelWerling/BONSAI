"""Test whether the rarity trend is invariant to cross-outcome weighting."""

from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from opera.evaluation.aggregation import (
    build_delta_vs_baseline_table,
    collect_result_rows,
    filter_results_for_paper_aggregates,
)
from opera.visualization.rarity_plots import plot_rarity_invariance_panel


TASK_COLUMNS = ["cohort", "outcome", "outcome_window_hours"]
EXPECTED_WEIGHTERS = ("kendall", "uniform", "famo")


def read_result_rows(path: Path) -> pd.DataFrame:
    """Read standard result rows from a directory, JSONL, CSV, or parquet."""
    if path.is_dir():
        return collect_result_rows(str(path))
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        rows = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append(json.loads(line))
        return pd.DataFrame(rows)
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    return pd.read_csv(path)


def _available_task_columns(frame: pd.DataFrame) -> list[str]:
    return [column for column in TASK_COLUMNS if column in frame.columns]


def _parse_weighter_models(values: Iterable[str]) -> dict[str, str]:
    mapping = {}
    for value in values:
        if "=" not in value:
            raise ValueError(
                "--weighter-model values must use WEIGHTER=MODEL_FAMILY syntax."
            )
        weighter, model_family = value.split("=", 1)
        weighter = weighter.strip().lower()
        model_family = model_family.strip()
        if not weighter or not model_family:
            raise ValueError("Weighter and model family names cannot be empty.")
        mapping[weighter] = model_family
    missing = sorted(set(EXPECTED_WEIGHTERS).difference(mapping))
    extra = sorted(set(mapping).difference(EXPECTED_WEIGHTERS))
    if missing or extra:
        raise ValueError(
            "Weighter mapping must define exactly kendall, uniform, and famo. "
            f"Missing={missing}, extra={extra}."
        )
    return mapping


def _collapse_and_validate_baseline(
    baseline_rows: pd.DataFrame,
    *,
    metric: str,
    class_balance_column: str,
) -> pd.DataFrame:
    if baseline_rows.empty:
        raise ValueError("The requested shared baseline has no result rows.")
    key_columns = [
        column
        for column in [
            *_available_task_columns(baseline_rows),
            "split",
            "seed",
            "evaluation_subset",
        ]
        if column in baseline_rows.columns
    ]
    comparison_columns = [
        column
        for column in [
            metric,
            "n_total",
            "checkpoint_path",
            class_balance_column,
        ]
        if column in baseline_rows.columns
    ]
    for keys, group in baseline_rows.groupby(key_columns, dropna=False):
        for column in comparison_columns:
            if group[column].dropna().astype(str).nunique() > 1:
                raise ValueError(
                    "Shared baseline parity failed for "
                    f"key={keys!r}, column={column!r}."
                )
    return baseline_rows.drop_duplicates(subset=key_columns, keep="first").copy()


def _validate_class_balance(
    model_rows: pd.DataFrame,
    baseline_rows: pd.DataFrame,
    *,
    class_balance_column: str,
) -> None:
    if class_balance_column not in model_rows.columns:
        return
    model_enabled = model_rows[class_balance_column].fillna(False).astype(bool).any()
    if not model_enabled:
        return
    if class_balance_column not in baseline_rows.columns:
        raise ValueError(
            "Class balancing is enabled for an OPERA run, but the shared baseline "
            f"does not report {class_balance_column!r}."
        )
    baseline_enabled = (
        baseline_rows[class_balance_column].fillna(False).astype(bool).all()
    )
    if not baseline_enabled:
        raise ValueError(
            "Class balancing is enabled for an OPERA run but not for every "
            "shared-baseline row, so the comparison is confounded."
        )


def _validate_model_baseline_denominators(
    model_rows: pd.DataFrame,
    baseline_rows: pd.DataFrame,
) -> None:
    if "n_total" not in model_rows.columns or "n_total" not in baseline_rows.columns:
        return
    task_columns = _available_task_columns(model_rows)
    baseline_counts = (
        baseline_rows.groupby(task_columns, dropna=False)["n_total"]
        .agg(
            lambda values: sorted(
                pd.to_numeric(values, errors="coerce").dropna().unique()
            )
        )
        .rename("baseline_n_total")
        .reset_index()
    )
    model_counts = (
        model_rows.groupby(task_columns, dropna=False)["n_total"]
        .agg(
            lambda values: sorted(
                pd.to_numeric(values, errors="coerce").dropna().unique()
            )
        )
        .rename("model_n_total")
        .reset_index()
    )
    paired = model_counts.merge(
        baseline_counts,
        on=task_columns,
        how="left",
        validate="one_to_one",
    )
    missing = paired["baseline_n_total"].isna()
    if missing.any():
        raise ValueError("Shared baseline is missing denominator rows for some tasks.")
    mismatched = paired.apply(
        lambda row: row["model_n_total"] != row["baseline_n_total"],
        axis=1,
    )
    if mismatched.any():
        records = paired.loc[
            mismatched,
            [*task_columns, "model_n_total", "baseline_n_total"],
        ].to_dict("records")
        raise ValueError(
            "Model and shared-baseline denominators differ for tasks: " + repr(records)
        )


def _task_key_set(frame: pd.DataFrame) -> set[tuple]:
    columns = _available_task_columns(frame)
    return set(frame[columns].drop_duplicates().itertuples(index=False, name=None))


def build_rarity_delta_table(
    results: pd.DataFrame,
    event_rates: pd.DataFrame,
    *,
    weighter_models: dict[str, str],
    baseline_model: str,
    evaluation_subset: str,
    metric: str = "auroc",
    class_balance_column: str = "class_balanced",
) -> pd.DataFrame:
    """Build paired per-task model-minus-baseline rows for three weighters."""
    if baseline_model == "ipi" and evaluation_subset != "ipi_complete":
        raise ValueError(
            "IPI comparisons must use evaluation_subset='ipi_complete' because "
            "IPI is not available for the full cohort."
        )
    filtered = filter_results_for_paper_aggregates(
        results,
        evaluation_subset=evaluation_subset,
        include_ipi=True,
    )
    if filtered.empty:
        raise ValueError(
            f"No result rows remain for evaluation_subset={evaluation_subset!r}."
        )
    if metric not in filtered.columns:
        raise ValueError(f"Result rows do not contain metric {metric!r}.")

    baseline_rows = _collapse_and_validate_baseline(
        filtered[filtered["model_family"] == baseline_model].copy(),
        metric=metric,
        class_balance_column=class_balance_column,
    )
    delta_frames = []
    task_sets = {}
    for weighter in EXPECTED_WEIGHTERS:
        model_family = weighter_models[weighter]
        model_rows = filtered[filtered["model_family"] == model_family].copy()
        if model_rows.empty:
            raise ValueError(
                f"No result rows found for {weighter} model_family={model_family!r}."
            )
        _validate_class_balance(
            model_rows,
            baseline_rows,
            class_balance_column=class_balance_column,
        )
        _validate_model_baseline_denominators(model_rows, baseline_rows)
        task_sets[weighter] = _task_key_set(model_rows)
        combined = pd.concat([baseline_rows, model_rows], ignore_index=True)
        delta = build_delta_vs_baseline_table(
            combined,
            baseline=baseline_model,
            metrics=(metric,),
        )
        delta = delta[delta["model_family"] == model_family].copy()
        delta["weighter"] = weighter
        delta_frames.append(delta)

    reference_tasks = task_sets["uniform"]
    for weighter, tasks in task_sets.items():
        if tasks != reference_tasks:
            missing = sorted(reference_tasks.difference(tasks))
            extra = sorted(tasks.difference(reference_tasks))
            raise ValueError(
                "Outcome task parity failed across weighters for "
                f"{weighter}: missing={missing}, extra={extra}."
            )

    delta_table = pd.concat(delta_frames, ignore_index=True)
    event_key_columns = [
        column
        for column in _available_task_columns(delta_table)
        if column in event_rates.columns
    ]
    if not {"cohort", "outcome"}.issubset(event_key_columns):
        raise ValueError("Event-rate table must contain cohort and outcome columns.")
    if "event_rate" not in event_rates.columns:
        raise ValueError("Event-rate table must contain event_rate.")
    event_table = event_rates[event_key_columns + ["event_rate"]].copy()
    if event_table.duplicated(event_key_columns).any():
        raise ValueError("Event-rate rows must be unique per outcome task.")
    delta_table = delta_table.merge(
        event_table,
        on=event_key_columns,
        how="left",
        validate="many_to_one",
    )
    if delta_table["event_rate"].isna().any():
        missing = delta_table.loc[
            delta_table["event_rate"].isna(),
            event_key_columns,
        ].drop_duplicates()
        raise ValueError(
            "Event rates are missing for outcome tasks: "
            + missing.to_dict("records").__repr__()
        )
    if ((delta_table["event_rate"] <= 0) | (delta_table["event_rate"] >= 1)).any():
        raise ValueError("event_rate values must be strictly between zero and one.")

    denominator_keys = [
        column
        for column in [
            *_available_task_columns(delta_table),
            "split",
            "seed",
            "evaluation_subset",
        ]
        if column in delta_table.columns
    ]
    if "n_total" in delta_table.columns:
        for keys, group in delta_table.groupby(denominator_keys, dropna=False):
            values = pd.to_numeric(group["n_total"], errors="coerce").dropna()
            if values.nunique() > 1:
                raise ValueError(
                    f"Paired denominator mismatch for outcome key={keys!r}."
                )
    return delta_table


def _slope_and_rank(
    event_rate: np.ndarray,
    delta: np.ndarray,
) -> tuple[float, float]:
    x = np.log10(np.asarray(event_rate, dtype=float))
    y = np.asarray(delta, dtype=float)
    if len(x) < 3 or np.unique(x).size < 2:
        return float("nan"), float("nan")
    slope = float(np.polyfit(x, y, deg=1)[0])
    rho = float(pd.Series(x).corr(pd.Series(y), method="spearman"))
    return slope, rho


def _percentile_interval(
    values: np.ndarray,
    confidence: float,
) -> tuple[float, float]:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return float("nan"), float("nan")
    alpha = (1.0 - confidence) / 2.0
    return (
        float(np.quantile(finite, alpha)),
        float(np.quantile(finite, 1.0 - alpha)),
    )


def summarize_rarity_trends(
    delta_table: pd.DataFrame,
    *,
    metric: str = "auroc",
    n_bootstrap: int = 2000,
    confidence: float = 0.95,
    seed: int = 17,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Bootstrap paired rarity slopes and test invariance across weighters."""
    if n_bootstrap < 1:
        raise ValueError("n_bootstrap must be positive.")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be between zero and one.")
    delta_column = f"delta_{metric}_vs_baseline"
    task_columns = _available_task_columns(delta_table)
    task_level = (
        delta_table.groupby([*task_columns, "weighter"], as_index=False, dropna=False)
        .agg(
            event_rate=("event_rate", "first"),
            delta=(delta_column, "median"),
            n_seeds=(delta_column, "count"),
        )
        .sort_values([*task_columns, "weighter"])
    )
    task_sets = {
        weighter: _task_key_set(group)
        for weighter, group in task_level.groupby("weighter")
    }
    reference = task_sets.get("uniform", set())
    if any(tasks != reference for tasks in task_sets.values()):
        raise ValueError(
            "Trend analysis requires identical outcome tasks per weighter."
        )
    if len(reference) < 3:
        raise ValueError("Rarity trend analysis requires at least three outcome tasks.")

    key_frame = (
        task_level[task_columns]
        .drop_duplicates()
        .sort_values(task_columns)
        .reset_index(drop=True)
    )
    ordered = {}
    for weighter in EXPECTED_WEIGHTERS:
        group = key_frame.merge(
            task_level[task_level["weighter"] == weighter],
            on=task_columns,
            how="left",
            validate="one_to_one",
        )
        ordered[weighter] = group

    rng = np.random.default_rng(seed)
    bootstrap_slopes = {weighter: [] for weighter in EXPECTED_WEIGHTERS}
    bootstrap_rhos = {weighter: [] for weighter in EXPECTED_WEIGHTERS}
    for _ in range(n_bootstrap):
        indices = rng.integers(0, len(key_frame), size=len(key_frame))
        for weighter, group in ordered.items():
            slope, rho = _slope_and_rank(
                group["event_rate"].to_numpy()[indices],
                group["delta"].to_numpy()[indices],
            )
            bootstrap_slopes[weighter].append(slope)
            bootstrap_rhos[weighter].append(rho)

    trend_rows = []
    for weighter, group in ordered.items():
        slope, rho = _slope_and_rank(
            group["event_rate"].to_numpy(),
            group["delta"].to_numpy(),
        )
        slope_lower, slope_upper = _percentile_interval(
            np.asarray(bootstrap_slopes[weighter]),
            confidence,
        )
        rho_lower, rho_upper = _percentile_interval(
            np.asarray(bootstrap_rhos[weighter]),
            confidence,
        )
        trend_rows.append(
            {
                "weighter": weighter,
                "n_tasks": len(group),
                "slope_vs_log10_event_rate": slope,
                "slope_ci_lower": slope_lower,
                "slope_ci_upper": slope_upper,
                "spearman_rho": rho,
                "spearman_ci_lower": rho_lower,
                "spearman_ci_upper": rho_upper,
            }
        )
    trends = pd.DataFrame(trend_rows)

    difference_rows = []
    for first, second in combinations(EXPECTED_WEIGHTERS, 2):
        differences = np.asarray(bootstrap_slopes[first]) - np.asarray(
            bootstrap_slopes[second]
        )
        lower, upper = _percentile_interval(differences, confidence)
        difference_rows.append(
            {
                "weighter_a": first,
                "weighter_b": second,
                "slope_difference": float(
                    trends.loc[
                        trends["weighter"] == first,
                        "slope_vs_log10_event_rate",
                    ].iloc[0]
                    - trends.loc[
                        trends["weighter"] == second,
                        "slope_vs_log10_event_rate",
                    ].iloc[0]
                ),
                "ci_lower": lower,
                "ci_upper": upper,
                "difference_detected": bool(lower > 0.0 or upper < 0.0),
            }
        )
    differences = pd.DataFrame(difference_rows)

    slopes = trends["slope_vs_log10_event_rate"].to_numpy()
    rhos = trends["spearman_rho"].to_numpy()
    slope_sign_preserved = bool(np.all(slopes > 0) or np.all(slopes < 0))
    rank_sign_preserved = bool(np.all(rhos > 0) or np.all(rhos < 0))
    slope_supported = bool(
        ((trends["slope_ci_lower"] > 0) | (trends["slope_ci_upper"] < 0)).all()
    )
    rank_supported = bool(
        ((trends["spearman_ci_lower"] > 0) | (trends["spearman_ci_upper"] < 0)).all()
    )
    no_slope_difference = bool((~differences["difference_detected"]).all())
    invariant = bool(
        slope_sign_preserved
        and rank_sign_preserved
        and slope_supported
        and rank_supported
        and no_slope_difference
    )
    details = {
        "gradient_invariant_to_weighter": invariant,
        "slope_sign_preserved": slope_sign_preserved,
        "monotonic_sign_preserved": rank_sign_preserved,
        "all_slope_intervals_exclude_zero": slope_supported,
        "all_rank_intervals_exclude_zero": rank_supported,
        "no_pairwise_slope_difference_detected": no_slope_difference,
    }
    return trends, differences, details


def run_rarity_invariance_panel(
    results: pd.DataFrame,
    event_rates: pd.DataFrame,
    output_dir: Path,
    *,
    weighter_models: dict[str, str],
    baseline_model: str,
    evaluation_subset: str,
    metric: str = "auroc",
    class_balance_column: str = "class_balanced",
    n_bootstrap: int = 2000,
    confidence: float = 0.95,
    seed: int = 17,
    pipeline_validation: bool = False,
) -> dict:
    """Write the paired rarity panel, trend intervals, and invariance verdict."""
    output_dir.mkdir(parents=True, exist_ok=True)
    delta_table = build_rarity_delta_table(
        results,
        event_rates,
        weighter_models=weighter_models,
        baseline_model=baseline_model,
        evaluation_subset=evaluation_subset,
        metric=metric,
        class_balance_column=class_balance_column,
    )
    trends, differences, details = summarize_rarity_trends(
        delta_table,
        metric=metric,
        n_bootstrap=n_bootstrap,
        confidence=confidence,
        seed=seed,
    )
    verdict = (
        "Rarity gradient is invariant to cross-outcome weighter."
        if details["gradient_invariant_to_weighter"]
        else "Rarity gradient is not invariant to cross-outcome weighter."
    )
    if pipeline_validation:
        verdict = "Pipeline validation only, not a scientific result. " + verdict
    summary = {
        "verdict": verdict,
        "pipeline_validation": pipeline_validation,
        "baseline_model": baseline_model,
        "evaluation_subset": evaluation_subset,
        "metric": metric,
        "weighter_models": weighter_models,
        "n_bootstrap": n_bootstrap,
        "confidence": confidence,
        **details,
    }
    delta_table.to_csv(output_dir / "rarity_invariance_task_deltas.csv", index=False)
    trends.to_csv(output_dir / "rarity_invariance_trends.csv", index=False)
    differences.to_csv(
        output_dir / "rarity_invariance_slope_differences.csv",
        index=False,
    )
    plot_rarity_invariance_panel(
        delta_table,
        baseline_model=baseline_model,
        metric=metric,
        save_path=str(output_dir / "rarity_invariance_panel.png"),
    )
    (output_dir / "rarity_invariance_verdict.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    (output_dir / "rarity_invariance_verdict.txt").write_text(
        verdict + "\n",
        encoding="utf-8",
    )
    return summary


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Test rarity-trend invariance across outcome weighters.",
    )
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--event-rates", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--weighter-model",
        action="append",
        required=True,
        help="Repeat as kendall=MODEL, uniform=MODEL, and famo=MODEL.",
    )
    parser.add_argument("--baseline-model", required=True)
    parser.add_argument(
        "--evaluation-subset",
        required=True,
        choices=["full", "ipi_complete"],
    )
    parser.add_argument("--metric", default="auroc")
    parser.add_argument("--class-balance-column", default="class_balanced")
    parser.add_argument("--n-bootstrap", type=int, default=2000)
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--pipeline-validation", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    results = read_result_rows(args.results)
    event_rates = read_result_rows(args.event_rates)
    summary = run_rarity_invariance_panel(
        results,
        event_rates,
        args.output_dir,
        weighter_models=_parse_weighter_models(args.weighter_model),
        baseline_model=args.baseline_model,
        evaluation_subset=args.evaluation_subset,
        metric=args.metric,
        class_balance_column=args.class_balance_column,
        n_bootstrap=args.n_bootstrap,
        confidence=args.confidence,
        seed=args.seed,
        pipeline_validation=args.pipeline_validation,
    )
    print(summary["verdict"])


if __name__ == "__main__":
    main()
