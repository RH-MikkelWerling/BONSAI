"""Plot real-data censoring diagnostics from patient coordinates and CIF exports."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from opera.visualization.censoring_aware_embeddings import plot_censoring_aware_embedding_panel
from opera.visualization.style import setup_style


def read_table(path):
    return pd.read_parquet(path) if Path(path).suffix == ".parquet" else pd.read_csv(path)


def align_map_data(coordinates, predictions, *, outcome, cohort=None, cohort_level=None,
                   membership=None, split="tuning", horizon=365):
    # Preserve older --cohort calls as grouped-cohort selections.
    cohort_level = cohort_level or ("cohort_grouped" if cohort else "all")
    if cohort_level not in {"all", "cohort_grouped", "cohort_fine"}:
        raise ValueError("Unknown cohort level.")
    if (cohort_level == "all" and cohort is not None) or (cohort_level != "all" and not cohort):
        raise ValueError("Specify --cohort for grouped/fine selection, and omit it for all.")
    required = {"subject_id", "outcome", "split", "time_days", "event", f"cif_{horizon}d"}
    if required - set(predictions):
        raise ValueError(f"Missing prediction columns: {sorted(required - set(predictions))}")
    if {"subject_id", "component_1", "component_2"} - set(coordinates):
        raise ValueError("Coordinates need subject_id, component_1 and component_2.")
    predictions = predictions.loc[(predictions["outcome"] == outcome) & (predictions["split"] == split)].copy()
    coordinates = coordinates.copy()
    if "split" in coordinates:
        coordinates = coordinates.loc[coordinates["split"] == split]
    if cohort_level != "all":
        column = cohort_level
        if column not in predictions:
            predictions[column] = np.nan
        # Older grouped exports used only 'cohort'. Never guess fine identities.
        if column == "cohort_grouped" and predictions[column].isna().all() and "cohort" in predictions:
            predictions[column] = predictions["cohort"]
        for source in (coordinates, membership):
            if source is None or column not in source:
                continue
            source = source.copy()
            if "split" in source:
                source = source.loc[source["split"] == split]
            identities = source[["subject_id", column]].dropna().drop_duplicates()
            if identities["subject_id"].duplicated().any():
                raise ValueError(f"Conflicting patient membership for {column}.")
            identities = identities.set_index("subject_id")[column]
            supplied = predictions["subject_id"].map(identities)
            conflict = predictions[column].notna() & supplied.notna() & (predictions[column] != supplied)
            if conflict.any():
                raise ValueError(f"Prediction and metadata membership disagree for {column}.")
            predictions[column] = predictions[column].fillna(supplied)
        if predictions[column].isna().any():
            raise ValueError(f"Missing {column} membership. Supply --membership or coordinates containing that column.")
        predictions = predictions.loc[predictions[column] == cohort]
        if column in coordinates:
            coordinates = coordinates.loc[coordinates[column].isna() | (coordinates[column] == cohort)]
    if predictions.empty:
        raise ValueError("No predictions match this outcome/cohort/split.")
    if predictions["subject_id"].duplicated().any() or coordinates["subject_id"].duplicated().any():
        raise ValueError("Duplicate patient IDs: select a single split/cohort or deduplicate the coordinate export.")
    joined = predictions.merge(coordinates[["subject_id", "component_1", "component_2"]],
                               on="subject_id", how="left", validate="one_to_one")
    if joined[["component_1", "component_2"]].isna().any().any():
        raise ValueError("Some prediction patients lack coordinates. Export a complete matching projection.")
    numeric = joined[["component_1", "component_2", "time_days", "event", f"cif_{horizon}d"]].to_numpy(float)
    if not np.isfinite(numeric).all() or (joined["time_days"] < 0).any() or not joined["event"].isin([0, 1, 2]).all():
        raise ValueError("Map inputs must be finite, with nonnegative times and event codes 0/1/2.")
    return joined


def main():
    setup_style()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--coordinates", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--outcome", required=True)
    parser.add_argument("--cohort-level", choices=["all", "cohort_grouped", "cohort_fine"], default=None,
                        help="Default: all, or cohort_grouped when --cohort is supplied.")
    parser.add_argument("--cohort", default=None, help="Exact cohort name at the selected level.")
    parser.add_argument("--membership", type=Path,
                        help="Optional patient membership CSV/Parquet with subject_id and cohort_fine/cohort_grouped.")
    parser.add_argument("--split", choices=["tuning", "held_out", "train"], default="tuning")
    parser.add_argument("--horizon", type=int, default=365)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--grid-size", type=int, default=100)
    parser.add_argument("--bandwidth", type=float)
    parser.add_argument("--bandwidth-scale", type=float, default=1.,
                        help="Multiply the kernel bandwidth for a controlled sensitivity comparison.")
    parser.add_argument("--residual-probability-limit", type=float, default=.2,
                        help="Common symmetric residual color range in probability units (default +/-0.2).")
    parser.add_argument("--min-effective-support", type=float, default=25)
    parser.add_argument("--min-density-fraction", type=float, default=.01)
    args = parser.parse_args()
    cohort_level = args.cohort_level or ("cohort_grouped" if args.cohort else "all")
    if (cohort_level == "all" and args.cohort) or (cohort_level != "all" and not args.cohort):
        parser.error("Grouped/fine selection requires --cohort; all selection must omit it.")
    if "ungated" in args.outcome.lower():
        parser.error("Ungated endpoints are excluded from this plotting workflow.")
    if args.split == "train":
        print("WARNING: training predictions show apparent calibration, not generalization.", flush=True)
    frame = align_map_data(read_table(args.coordinates), read_table(args.predictions),
                           outcome=args.outcome, cohort=args.cohort, cohort_level=cohort_level,
                           membership=read_table(args.membership) if args.membership else None,
                           split=args.split, horizon=args.horizon)
    print(f"Rendering {len(frame):,} patients; outcome={args.outcome}; cohort={args.cohort or 'pooled'}; split={args.split}", flush=True)
    fig, diagnostics = plot_censoring_aware_embedding_panel(
        frame[["component_1", "component_2"]].to_numpy(), frame["time_days"].to_numpy(), frame["event"].to_numpy(),
        horizon=args.horizon, predicted_cif=frame[f"cif_{args.horizon}d"].to_numpy(),
        bandwidth=args.bandwidth, grid_size=args.grid_size, min_effective_support=args.min_effective_support,
        bandwidth_scale=args.bandwidth_scale, residual_probability_limit=args.residual_probability_limit,
        min_density_fraction=args.min_density_fraction, residual_bootstrap=0,
        title=f"{args.outcome.replace('_', ' ')} · {args.cohort or 'All cohorts'} · {args.split}",
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for extension in ("png", "pdf"):
        fig.savefig(args.output_dir / f"censoring_aware_map.{extension}", dpi=250, bbox_inches="tight")
    plt.close(fig)
    diagnostics.update(outcome=args.outcome, cohort=args.cohort, cohort_level=cohort_level,
                       membership=str(args.membership.resolve()) if args.membership else None,
                       split=args.split, horizon=args.horizon,
                       coordinates=str(args.coordinates.resolve()), predictions=str(args.predictions.resolve()))
    (args.output_dir / "diagnostics.json").write_text(json.dumps(diagnostics, indent=2), encoding="utf-8")
    print(f"Wrote PNG, PDF and diagnostics to {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
