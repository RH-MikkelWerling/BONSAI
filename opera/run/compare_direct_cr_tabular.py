"""Paired held-out comparison of direct OPERA CIFs and tabular predictions."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from opera.evaluation.metrics import _km_admin_censoring_fn

_HORIZON_PATTERN = re.compile(r"_(\d+)d_predictions\.csv$")


def _tabular_artifacts(root: Path) -> dict[tuple[str, int], Path]:
    artifacts: dict[tuple[str, int], Path] = {}
    for metadata_path in root.glob("*_metadata.json"):
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        prediction_path = Path(metadata.get("prediction_file", ""))
        match = _HORIZON_PATTERN.search(prediction_path.name)
        outcome = str(metadata.get("outcome_name", ""))
        if not match or not outcome or not prediction_path.is_file():
            continue
        key = (outcome, int(match.group(1)))
        if key in artifacts:
            raise ValueError(f"Duplicate tabular artifact for {key}: {prediction_path}")
        artifacts[key] = prediction_path
    return artifacts


def _metrics(frame: pd.DataFrame, prediction: str, horizon: int) -> dict:
    """O(n log n) IPCW CIF metrics, equivalent to the shared evaluator."""
    times = frame["time_days"].to_numpy(dtype=float)
    events = frame["event"].to_numpy(dtype=int)
    risks = frame[prediction].to_numpy(dtype=float)
    horizon = float(horizon)
    censoring = _km_admin_censoring_fn(times, events)
    cases = (times <= horizon) & (events == 1)
    competing = (times <= horizon) & (events == 2)
    controls = competing | (times > horizon) | ((times >= horizon) & (events == 0))
    excluded = ~(cases | controls)
    case_risk = risks[cases]
    control_risk = risks[controls]
    case_weight = np.asarray(
        [1.0 / max(censoring(float(value)), 1e-6) for value in times[cases]]
    )
    control_times = times[controls]
    control_is_competing = competing[controls]
    control_weight = np.asarray(
        [
            1.0
            / max(censoring(float(value) if is_competing else horizon), 1e-6)
            for value, is_competing in zip(control_times, control_is_competing)
        ]
    )
    auc = float("nan")
    if len(case_risk) and len(control_risk):
        order = np.argsort(control_risk, kind="mergesort")
        sorted_risk = control_risk[order]
        sorted_weight = control_weight[order]
        cumulative = np.concatenate([[0.0], np.cumsum(sorted_weight)])
        left = np.searchsorted(sorted_risk, case_risk, side="left")
        right = np.searchsorted(sorted_risk, case_risk, side="right")
        concordant_weight = cumulative[left] + 0.5 * (
            cumulative[right] - cumulative[left]
        )
        denominator = case_weight.sum() * control_weight.sum()
        if denominator > 0:
            auc = float(np.dot(case_weight, concordant_weight) / denominator)
    brier = (
        np.dot(case_weight, (1.0 - case_risk) ** 2)
        + np.dot(control_weight, control_risk**2)
    ) / len(times)
    return {
        "cif_auc": auc,
        "cif_brier": float(brier),
        "n_cases": int(cases.sum()),
        "n_controls": int(controls.sum()),
        "n_excluded_admin_censoring": int(excluded.sum()),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--direct-predictions", type=Path, required=True)
    parser.add_argument("--tabular-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--n-bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.n_bootstrap < 1:
        raise ValueError("--n-bootstrap must be positive.")
    direct = pd.read_parquet(args.direct_predictions)
    required = {"subject_id", "outcome", "time_days", "event"}
    missing = required - set(direct)
    if missing:
        raise ValueError(f"Direct predictions are missing columns: {sorted(missing)}")
    artifacts = _tabular_artifacts(args.tabular_dir)
    if not artifacts:
        raise ValueError(
            f"No tabular prediction/metadata pairs found in {args.tabular_dir}"
        )

    rng = np.random.default_rng(args.seed)
    rows: list[dict] = []
    bootstrap_rows: list[dict] = []
    for (outcome, horizon), prediction_path in sorted(artifacts.items()):
        print(
            f"Comparing {outcome} at {horizon}d "
            f"({args.n_bootstrap} paired bootstraps)...",
            flush=True,
        )
        direct_column = f"cif_{horizon}d"
        if direct_column not in direct:
            continue
        outcome_direct = direct.loc[
            direct["outcome"].astype(str) == outcome,
            ["subject_id", "time_days", "event", direct_column],
        ]
        tabular = pd.read_csv(prediction_path)[["subject_id", "probability"]]
        paired = outcome_direct.merge(
            tabular, on="subject_id", how="inner", validate="one_to_one"
        ).rename(
            columns={
                direct_column: "opera_probability",
                "probability": "tabular_probability",
            }
        )
        if paired.empty:
            raise ValueError(f"No paired subjects for {outcome}/{horizon}d")
        opera = _metrics(paired, "opera_probability", horizon)
        tabular_metrics = _metrics(paired, "tabular_probability", horizon)
        rows.append(
            {
                "outcome": outcome,
                "horizon_days": horizon,
                "n_paired": len(paired),
                "opera_cif_auc": opera["cif_auc"],
                "tabular_cif_auc": tabular_metrics["cif_auc"],
                "delta_cif_auc": opera["cif_auc"] - tabular_metrics["cif_auc"],
                "opera_cif_brier": opera["cif_brier"],
                "tabular_cif_brier": tabular_metrics["cif_brier"],
                "delta_cif_brier": opera["cif_brier"] - tabular_metrics["cif_brier"],
                "n_cases": opera["n_cases"],
                "n_controls": opera["n_controls"],
            }
        )
        n = len(paired)
        for draw in range(args.n_bootstrap):
            sample = paired.iloc[rng.integers(0, n, size=n)]
            opera_draw = _metrics(sample, "opera_probability", horizon)
            tabular_draw = _metrics(sample, "tabular_probability", horizon)
            bootstrap_rows.append(
                {
                    "outcome": outcome,
                    "horizon_days": horizon,
                    "draw": draw,
                    "delta_cif_auc": opera_draw["cif_auc"] - tabular_draw["cif_auc"],
                    "delta_cif_brier": opera_draw["cif_brier"]
                    - tabular_draw["cif_brier"],
                }
            )
        print(f"  completed {outcome}/{horizon}d", flush=True)

    results = pd.DataFrame(rows)
    draws = pd.DataFrame(bootstrap_rows)
    if results.empty:
        raise ValueError("No matching outcome/horizon artifacts were comparable.")
    intervals = (
        draws.groupby(["outcome", "horizon_days"])[["delta_cif_auc", "delta_cif_brier"]]
        .quantile([0.025, 0.975])
        .unstack()
    )
    intervals.columns = [
        f"{metric}_{'lower' if quantile == 0.025 else 'upper'}"
        for metric, quantile in intervals.columns
    ]
    results = results.merge(
        intervals.reset_index(), on=["outcome", "horizon_days"], how="left"
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results.to_csv(args.output_dir / "paired_comparison.csv", index=False)
    draws.to_parquet(args.output_dir / "paired_bootstrap_draws.parquet", index=False)
    print(results.to_string(index=False))


if __name__ == "__main__":
    main()
