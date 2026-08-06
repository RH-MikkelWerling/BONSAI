"""CLI for the censoring-aware five-panel embedding diagnostic."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
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
    parser.add_argument("--covariates", nargs="*", default=[])
    parser.add_argument("--bandwidth", type=float)
    parser.add_argument("--grid-size", type=int, default=80)
    parser.add_argument("--min-effective-support", type=float, default=25.0)
    parser.add_argument("--title", default="Outcome")
    args = parser.parse_args()
    frame = pd.read_parquet(args.input) if args.input.suffix == ".parquet" else pd.read_csv(args.input)
    if args.outcome is not None:
        outcome_frame = pd.read_parquet(args.outcome)
        competing_frame = (
            pd.read_parquet(args.competing_outcome)
            if args.competing_outcome is not None else None
        )
        records = binarize_outcomes(
            outcome_frame,
            n_hours_start_include=args.n_hours_start_include,
            n_hours_end_include=None,
            competing_event_df=competing_frame,
        )
        labels = pd.DataFrame([
            {
                args.subject_col: subject_id,
                "_diagnostic_time": record["time_days"],
                "_diagnostic_event": record["event"],
            }
            for subject_id, record in records.items()
        ])
        frame = frame.merge(labels, on=args.subject_col, how="left", validate="one_to_one")
        time_col, event_col = "_diagnostic_time", "_diagnostic_event"
    else:
        if args.time_col is None or args.event_col is None:
            raise ValueError("Provide --outcome, or both --time-col and --event-col.")
        time_col, event_col = args.time_col, args.event_col
    required = [args.x_col, args.y_col, time_col, event_col, *args.covariates]
    missing = [column for column in required if column not in frame]
    if missing:
        raise ValueError(f"Input is missing required columns: {missing}")
    fig, diagnostics = plot_censoring_aware_embedding_panel(
        frame[[args.x_col, args.y_col]].to_numpy(),
        frame[time_col].to_numpy(),
        frame[event_col].to_numpy(),
        horizon=args.horizon_days,
        covariates=frame[args.covariates] if args.covariates else None,
        bandwidth=args.bandwidth,
        grid_size=args.grid_size,
        min_effective_support=args.min_effective_support,
        title=args.title,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=200, bbox_inches="tight")
    plt.close(fig)
    args.output.with_suffix(".json").write_text(json.dumps(diagnostics, indent=2))


if __name__ == "__main__":
    main()
