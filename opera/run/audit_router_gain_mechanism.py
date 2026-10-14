"""Select tuning outcomes for router mechanism checks using paired predictions.

This is exploratory, conditional on fixed selected checkpoints. It never reads
held-out rows and does not treat route weights as predictive contributions.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from opera.run.compare_approved_neural_tuning_predictions import (
    _families, _read_run, compare_predictions,
)


def select_outcomes(contrasts: pd.DataFrame, *, n_each: int = 3) -> pd.DataFrame:
    """Prespecify gain, neutral and loss probes without using route weights."""
    if n_each < 1 or contrasts.empty:
        raise ValueError("Need positive selection size and nonempty contrasts")
    ordered = contrasts.sort_values(["estimate", "outcome"], ascending=[False, True])
    selected: dict[str, str] = {}
    for reason, subset in (
        ("largest_gain", ordered.loc[ordered["estimate"] > 0].head(n_each)),
        ("smallest_absolute_gain", contrasts.assign(
            absolute=contrasts["estimate"].abs()).sort_values(
                ["absolute", "outcome"]).head(n_each)),
        ("largest_loss", ordered.loc[ordered["estimate"] < 0].tail(n_each).iloc[::-1]),
    ):
        for outcome in subset["outcome"]:
            selected.setdefault(str(outcome), reason)
    result = contrasts.loc[contrasts["outcome"].isin(selected)].copy()
    result["selection_reason"] = result["outcome"].map(selected)
    return result.sort_values(["selection_reason", "estimate", "outcome"])


def route_diagnostics(routes: pd.DataFrame, selected: pd.DataFrame) -> pd.DataFrame:
    """Patient-vs-cell gate variation on selected target outcomes."""
    required = {"subject_id", "cohort", "split", "outcome", "fine_index",
                "shared_weight", "disease_weights", "outcome_weights"}
    if required - set(routes):
        raise ValueError(f"Route export lacks columns: {sorted(required - set(routes))}")
    if routes.empty or not routes["split"].eq("tuning").all():
        raise ValueError("Route diagnostics require tuning-only rows")
    if routes.duplicated(["subject_id", "cohort", "outcome"]).any():
        raise ValueError("Duplicate patient-target routes")
    rows = []
    for (outcome, fine_index), group in routes.groupby(["outcome", "fine_index"]):
        if outcome not in set(selected["outcome"]) or len(group) < 20:
            continue
        disease_names = sorted(json.loads(group.iloc[0]["disease_weights"]))
        outcome_names = sorted(json.loads(group.iloc[0]["outcome_weights"]))
        matrices = []
        for column, names in (("disease_weights", disease_names),
                              ("outcome_weights", outcome_names)):
            parsed = [json.loads(value) for value in group[column]]
            if any(set(value) != set(names) for value in parsed):
                raise ValueError("Source inventory differs across route rows")
            matrices.append(np.asarray([[value[name] for name in names]
                                        for value in parsed], dtype=float))
        shared = group["shared_weight"].to_numpy(float)[:, None]
        weights = np.column_stack((shared, *matrices))
        if not np.isfinite(weights).all() or not np.allclose(weights.sum(axis=1), 1, atol=1e-3):
            raise ValueError("Invalid route weights")
        mean = weights.mean(axis=0)
        patient_tv = .5 * np.abs(weights - mean).sum(axis=1)
        rows.append({"outcome": outcome, "fine_index": int(fine_index),
                     "n_patients": len(group), "mean_shared_weight": float(mean[0]),
                     "mean_patient_tv_from_cell_mean": float(patient_tv.mean()),
                     "p90_patient_tv_from_cell_mean": float(np.quantile(patient_tv, .9)),
                     "max_patient_tv_from_cell_mean": float(patient_tv.max())})
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-predictions", type=Path, required=True)
    parser.add_argument("--router-predictions", type=Path, required=True)
    parser.add_argument("--family-map", type=Path, required=True)
    parser.add_argument("--routes", type=Path, action="append", default=[])
    parser.add_argument("--horizon", type=int, default=365)
    parser.add_argument("--bootstrap", type=int, default=400)
    parser.add_argument("--min-cases", type=int, default=20)
    parser.add_argument("--select-each", type=int, default=3)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    if args.horizon <= 0 or args.bootstrap < 50 or args.min_cases < 1:
        parser.error("Horizon must be positive, bootstrap >=50, min-cases >=1")
    all_router = pd.read_parquet(args.router_predictions, columns=["outcome", "split"])
    if all_router.empty or not all_router["split"].eq("tuning").all():
        raise ValueError("Router predictions must be tuning-only")
    outcomes = sorted(all_router["outcome"].unique().tolist())
    frames = {"baseline": _read_run(args.baseline_predictions, outcomes, args.horizon),
              "router": _read_run(args.router_predictions, outcomes, args.horizon)}
    _, contrasts, report = compare_predictions(
        frames, baseline="baseline", outcomes=outcomes, horizon=args.horizon,
        n_bootstrap=args.bootstrap, seed=42,
        family_by_outcome=_families(args.family_map, outcomes),
        min_cases=args.min_cases, skip_under_supported=True,
    )
    selected = select_outcomes(contrasts, n_each=args.select_each)
    diagnostics = pd.DataFrame()
    if args.routes:
        route_frames = [pd.read_parquet(path) for path in args.routes]
        routes = pd.concat(route_frames, ignore_index=True)
        prediction_keys = frames["router"][["subject_id", "outcome"]].drop_duplicates()
        route_keys = routes[["subject_id", "outcome"]].drop_duplicates()
        checked = route_keys.merge(prediction_keys, on=["subject_id", "outcome"],
                                   how="left", indicator=True, validate="one_to_one")
        if not checked["_merge"].eq("both").all():
            raise ValueError("Some route patient/target rows are absent from router predictions")
        diagnostics = route_diagnostics(routes, selected)
        if not diagnostics.empty:
            config = yaml.safe_load(args.family_map.read_text(encoding="utf-8"))
            fine_names = sorted(config["competing_risk"]["fine_names"])
            if not diagnostics["fine_index"].between(0, len(fine_names) - 1).all():
                raise ValueError("Route fine index is absent from saved router config")
            diagnostics.insert(1, "fine_disease",
                               diagnostics["fine_index"].map(dict(enumerate(fine_names))))
    args.output_dir.mkdir(parents=True)
    contrasts.sort_values("estimate", ascending=False).to_csv(
        args.output_dir / "all_outcome_gains.csv", index=False)
    selected.to_csv(args.output_dir / "selected_mechanism_outcomes.csv", index=False)
    diagnostics.to_csv(args.output_dir / "selected_route_patient_variation.csv", index=False)
    manifest = {**report, "selected_outcomes": selected["outcome"].tolist(),
                "route_outcomes_available": sorted(diagnostics["outcome"].unique().tolist())
                if not diagnostics.empty else [],
                "baseline_predictions": str(args.baseline_predictions.resolve()),
                "router_predictions": str(args.router_predictions.resolve()),
                "route_exports": [str(path.resolve()) for path in args.routes],
                "interpretation": "Selection is exploratory on tuning; gate variation is not causal contribution."}
    (args.output_dir / "audit_manifest_INTERNAL.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output_dir": str(args.output_dir.resolve()),
                      "selected_outcomes": manifest["selected_outcomes"],
                      "route_outcomes_available": manifest["route_outcomes_available"]}, indent=2))


if __name__ == "__main__":
    main()
