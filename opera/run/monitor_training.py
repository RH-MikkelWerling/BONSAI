"""Print a compact live summary from a Lightning ``metrics.csv`` file."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


DEFAULT_METRICS = (
    "train/loss_epoch",
    "val/loss",
    "train/contrastive_loss_epoch",
    "val/contrastive_loss",
    "train/cr/loss_epoch",
    "val/cr/loss",
    "train/contrastive/excess_loss_epoch",
    "val/contrastive/excess_loss",
    "train/contrastive/target_entropy_epoch",
    "val/contrastive/target_entropy",
    "train/contrastive/relative_excess_epoch",
    "val/contrastive/relative_excess",
    "train/anchor_loss_epoch",
    "val/anchor_loss",
    "val/diagnostic_ever_event_projection_auroc_mean",
)


def resolve_metrics_csv(source: Path) -> Path:
    """Resolve a CSV directly or select the newest metrics CSV below a run."""
    if source.is_file():
        return source
    candidates = list(source.glob("**/metrics.csv"))
    if not candidates:
        raise FileNotFoundError(f"No metrics.csv found below {source}")
    return max(candidates, key=lambda path: path.stat().st_mtime)


def summarize_metrics(
    frame: pd.DataFrame,
    *,
    include_families: bool = False,
) -> pd.DataFrame:
    """Collapse Lightning's sparse metric rows into one row per epoch."""
    metrics = [name for name in DEFAULT_METRICS if name in frame.columns]
    lr_metrics = [name for name in frame.columns if "lr-" in name.lower()]
    metrics.extend(name for name in lr_metrics if name not in metrics)
    if include_families:
        family_metrics = [
            name
            for name in frame.columns
            if "/family_" in name or "/cr/family_" in name
        ]
        metrics.extend(name for name in family_metrics if name not in metrics)
    if "epoch" not in frame or not metrics:
        return frame[[name for name in ("epoch", "step", *metrics) if name in frame]].tail(20)

    rows = []
    for epoch, group in frame.groupby("epoch", sort=True, dropna=True):
        row = {"epoch": int(epoch)}
        if "step" in group and group["step"].notna().any():
            row["step"] = int(group["step"].dropna().max())
        for metric in metrics:
            if metric in lr_metrics and "step" in frame and "step" in row:
                # LearningRateMonitor rows commonly have a step but no epoch.
                # Attach the latest rate at or before this epoch's final step.
                values = frame.loc[
                    frame["step"].notna() & (frame["step"] <= row["step"]), metric
                ].dropna()
            else:
                values = group[metric].dropna()
            row[metric] = values.iloc[-1] if not values.empty else float("nan")
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "source",
        type=Path,
        help="metrics.csv or a run directory containing one",
    )
    parser.add_argument("--tail", type=int, default=20)
    parser.add_argument(
        "--families",
        action="store_true",
        help="Include family-level contrastive and competing-risk columns",
    )
    args = parser.parse_args()

    metrics_path = resolve_metrics_csv(args.source)
    # A concurrently written CSV can end with a partial line. The Python engine
    # tolerates a malformed trailing row without touching the training process.
    frame = pd.read_csv(metrics_path, engine="python", on_bad_lines="skip")
    summary = summarize_metrics(
        frame,
        include_families=args.families,
    ).tail(max(1, args.tail))
    print(f"metrics: {metrics_path}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
