"""CLI for the censoring-aware five-panel embedding diagnostic."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from opera.compat.bonsai import binarize_outcomes
from opera.visualization.censoring_aware_embeddings import (
    plot_censoring_aware_embedding_panel,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--x-col", default="umap_1")
    parser.add_argument("--y-col", default="umap_2")
    parser.add_argument("--time-col")
    parser.add_argument("--event-col")
    parser.add_argument("--outcome", type=Path)
    parser.add_argument("--competing-outcome", type=Path)
    parser.add_argument("--subject-col", default="subject_id")
    parser.add_argument("--n-hours-start-include", type=int, default=1)
    parser.add_argument("--horizon-days", type=float, required=True)
    parser.add_argument(
        "--predicted-cif-col",
        help="Per-patient predicted cause-specific CIF at --horizon-days (panel D).",
    )
    parser.add_argument(
        "--embedding-cols",
        nargs="*",
        default=[],
        help="Full embedding columns for the optional full-embedding residual neighborhood.",
    )
    parser.add_argument(
        "--held-out-col",
        help="Boolean column marking patients the model never trained on.",
    )
    parser.add_argument("--n-neighbors", type=int, default=50)
    parser.add_argument("--residual-use-full-embedding", action="store_true")
    parser.add_argument(
        "--residual-units",
        choices=["z", "raw"],
        default="raw",
        help="Panel D scale: raw D(x) in CIF units (default), or exploratory local z.",
    )
    parser.add_argument("--residual-bootstrap", type=int, default=200)
    parser.add_argument(
        "--residual-z-limit",
        type=float,
        default=4.0,
        help="Symmetric colour limit for panel D in z units.",
    )
    parser.add_argument("--random-state", type=int, default=0)
    parser.add_argument("--bandwidth", type=float)
    parser.add_argument("--grid-size", type=int, default=80)
    parser.add_argument("--min-effective-support", type=float, default=25.0)
    parser.add_argument("--title", default="Outcome")
    parser.add_argument("--router-vectors", type=Path,
                        help="INTERNAL pre/post-router NPZ from evaluate_direct_competing_risk.")
    parser.add_argument("--target-outcome",
                        help="Filter --input predictions to this outcome for router movement.")
    parser.add_argument("--router-panel", choices=["movement", "time"], default="movement",
                        help="With --router-vectors, choose new movement B or original time B on the same PCA map.")
    parser.add_argument("--router-projection", choices=["pca", "umap"], default="pca",
                        help="Fit once to shared vectors; transform adapted vectors with the same map. UMAP arrows are exploratory.")
    parser.add_argument("--router-arrow-bins", type=int, default=7)
    parser.add_argument("--router-arrow-min-count", type=int, default=20)
    parser.add_argument("--router-arrow-scale", type=float,
                        help="Fixed display multiplier for comparable arrows; default adapts to each plot.")
    parser.add_argument("--router-arrow-style", choices=["standard", "compact"],
                        default="standard", help="Compact arrows are shorter and thinner for a denser grid.")
    parser.add_argument("--router-arrow-min-coherence", type=float, default=0.,
                        help="Hide local arrows whose mean direction cancels across patients (0 to 1).")
    parser.add_argument("--plot-data-output", type=Path,
                        help="Write aggregate-only INTERNAL grid surfaces and local arrows as NPZ; no patient vectors.")
    args = parser.parse_args()
    frame = (
        pd.read_parquet(args.input)
        if args.input.suffix == ".parquet"
        else pd.read_csv(args.input)
    )
    routed_coords = routed_norms = None
    if args.router_vectors is not None:
        if not args.target_outcome or args.outcome is not None:
            raise ValueError("Router panel needs --target-outcome and prediction --input, not --outcome")
        manifest = json.loads(args.router_vectors.with_suffix(".json").read_text(encoding="utf-8"))
        if manifest.get("target_outcome") != args.target_outcome or manifest.get("split") != "tuning":
            raise ValueError("Router vector outcome/split does not match tuning target")
        if "split" not in frame or not frame["split"].eq("tuning").all():
            raise ValueError("Router movement panel requires tuning-only predictions")
        frame = frame.loc[frame["outcome"].eq(args.target_outcome)].copy()
        if frame.empty or frame[args.subject_col].duplicated().any():
            raise ValueError("Target outcome needs unique patient predictions")
        with np.load(args.router_vectors) as artifact:
            ids = artifact["subject_ids"]
            shared = artifact["shared"]
            adapted = artifact["adapted"]
        if shared.ndim != 2 or adapted.shape != shared.shape or len(ids) != len(shared):
            raise ValueError("Routed vector export has inconsistent shapes")
        if len(np.unique(ids)) != len(ids) or not np.isfinite(shared).all() or not np.isfinite(adapted).all():
            raise ValueError("Routed vector export has duplicate IDs or nonfinite values")
        lookup = pd.Index(ids).get_indexer(frame[args.subject_col])
        if (lookup < 0).any():
            raise ValueError("Predictions include patients missing from router vector export")
        shared, adapted = shared[lookup], adapted[lookup]
        if args.router_projection == "pca":
            from sklearn.decomposition import PCA
            projector = PCA(n_components=2, random_state=args.random_state).fit(shared)
        else:
            try:
                from umap import UMAP
            except ImportError as exc:
                raise ImportError("UMAP projection requires umap-learn; use --router-projection pca if unavailable") from exc
            projector = UMAP(n_components=2, random_state=args.random_state,
                             transform_seed=args.random_state).fit(shared)
        before = projector.transform(shared)
        routed_coords = projector.transform(adapted)
        routed_norms = np.linalg.norm(adapted - shared, axis=1)
        frame["_router_projection_1"], frame["_router_projection_2"] = before.T
        args.x_col, args.y_col = "_router_projection_1", "_router_projection_2"
    if args.outcome is not None:
        outcome_frame = pd.read_parquet(args.outcome)
        competing_frame = (
            pd.read_parquet(args.competing_outcome)
            if args.competing_outcome is not None
            else None
        )
        records = binarize_outcomes(
            outcome_frame,
            n_hours_start_include=args.n_hours_start_include,
            n_hours_end_include=None,
            competing_event_df=competing_frame,
        )
        labels = pd.DataFrame(
            [
                {
                    args.subject_col: subject_id,
                    "_diagnostic_time": record["time_days"],
                    "_diagnostic_event": record["event"],
                }
                for subject_id, record in records.items()
            ]
        )
        frame = frame.merge(
            labels, on=args.subject_col, how="left", validate="one_to_one"
        )
        time_col, event_col = "_diagnostic_time", "_diagnostic_event"
    else:
        if args.time_col is None or args.event_col is None:
            raise ValueError("Provide --outcome, or both --time-col and --event-col.")
        time_col, event_col = args.time_col, args.event_col
    optional = [args.predicted_cif_col, args.held_out_col]
    required = [
        args.x_col,
        args.y_col,
        time_col,
        event_col,
        *args.embedding_cols,
        *[column for column in optional if column],
    ]
    missing = [column for column in required if column not in frame]
    if missing:
        raise ValueError(f"Input is missing required columns: {missing}")
    plot_data = {} if args.plot_data_output is not None else None
    fig, diagnostics = plot_censoring_aware_embedding_panel(
        frame[[args.x_col, args.y_col]].to_numpy(),
        frame[time_col].to_numpy(),
        frame[event_col].to_numpy(),
        horizon=args.horizon_days,
        predicted_cif=(
            frame[args.predicted_cif_col].to_numpy() if args.predicted_cif_col else None
        ),
        embedding=(
            frame[args.embedding_cols].to_numpy() if args.embedding_cols else None
        ),
        held_out=(
            frame[args.held_out_col].to_numpy().astype(bool)
            if args.held_out_col
            else None
        ),
        bandwidth=args.bandwidth,
        grid_size=args.grid_size,
        min_effective_support=args.min_effective_support,
        n_neighbors=args.n_neighbors,
        residual_standardize=args.residual_units == "z",
        residual_use_full_embedding=args.residual_use_full_embedding,
        residual_bootstrap=args.residual_bootstrap,
        residual_z_limit=args.residual_z_limit,
        random_state=args.random_state,
        title=args.title,
        routed_coords=routed_coords if args.router_panel == "movement" else None,
        routed_norms=routed_norms if args.router_panel == "movement" else None,
        router_arrow_bins=args.router_arrow_bins,
        router_arrow_min_count=args.router_arrow_min_count,
        router_arrow_scale=args.router_arrow_scale,
        router_arrow_style=args.router_arrow_style,
        router_arrow_min_coherence=args.router_arrow_min_coherence,
        plot_data=plot_data,
    )
    if args.router_vectors is not None:
        diagnostics["router_projection"] = args.router_projection
        diagnostics["router_projection_contract"] = "fit to shared tuning patient vectors; post-router vectors transformed using the same map"
        diagnostics["router_projection_arrows_interpretation"] = (
            "linear projected displacement" if args.router_projection == "pca"
            else "qualitative layout displacement; nonlinear distances and directions are not hidden-space effects"
        )
        if args.router_projection == "pca":
            diagnostics["router_pca_variance_ratio"] = projector.explained_variance_ratio_.tolist()
        diagnostics["router_target_outcome"] = args.target_outcome
        diagnostics["router_panel_variant"] = args.router_panel
    args.output.parent.mkdir(parents=True, exist_ok=True)
    from opera.visualization.style import save_fig

    save_fig(fig, args.output)
    plt.close(fig)
    args.output.with_suffix(".json").write_text(json.dumps(diagnostics, indent=2))
    if args.plot_data_output is not None:
        args.plot_data_output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(args.plot_data_output, **plot_data)


if __name__ == "__main__":
    main()
