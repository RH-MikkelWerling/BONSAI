"""Validation-only XGBoost outcome predictability prototype (not a model comparison)."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
import yaml

from opera.visualization.style import PALETTE, despine, save_fig


DEFAULT_DISPLAY_GROUPS = Path("opera/configs/outcome_display_groups_v1.yaml")


def prepare_data(
    csv_path: Path, registry_path: Path,
    display_groups_path: Path = DEFAULT_DISPLAY_GROUPS,
) -> pd.DataFrame:
    frame = pd.read_csv(csv_path)
    required = {"outcome", "cohort", "split", "evaluation_regime", "auroc", "n_positive"}
    missing = required - set(frame)
    if missing:
        raise ValueError(f"Missing validation aggregate columns: {sorted(missing)}")
    if not frame["split"].eq("tuning").all():
        raise ValueError("Prototype input must contain tuning rows only")
    frame = frame.loc[
        (frame["cohort"] == "ALL_HEMATOLOGY")
        & (frame["split"] == "tuning")
        & (frame["evaluation_regime"] == "fixed_horizon")
    ].copy()
    if frame.empty or frame["outcome"].duplicated().any():
        raise ValueError("Expected one tuning fixed-horizon row per outcome")
    registry = yaml.safe_load(registry_path.read_text(encoding="utf-8"))
    groups = yaml.safe_load(display_groups_path.read_text(encoding="utf-8"))
    members = [outcome for outcomes in groups.values() for outcome in outcomes]
    expected = set(registry["outcomes"])
    if len(members) != len(set(members)) or set(members) != expected:
        raise ValueError("Display groups must cover all registered outcomes exactly once")
    if max(map(len, groups.values())) > 8:
        raise ValueError("Display groups must not exceed eight outcomes")
    by_outcome = {
        outcome: group for group, outcomes in groups.items() for outcome in outcomes
    }
    frame["display_group"] = frame["outcome"].map(by_outcome)
    if frame["display_group"].isna().any():
        raise ValueError("Validation outcomes are absent from the display groups")
    for column in ("auroc", "n_positive"):
        frame[column] = pd.to_numeric(frame[column], errors="raise")
    if not frame["auroc"].between(0, 1).all() or not frame["n_positive"].gt(0).all():
        raise ValueError("AUROC and positive counts must be valid")
    return frame


def plot_predictability(frame: pd.DataFrame, output: Path, horizon_days: int) -> None:
    if horizon_days < 1:
        raise ValueError("horizon_days must be positive")
    fig, (ax, family_ax) = plt.subplots(
        1, 2, figsize=(12.0, 7.5), gridspec_kw={"width_ratios": [1.05, 1.1]}
    )
    supported = frame[frame["n_positive"] >= 20]
    sparse = frame[frame["n_positive"] < 20]
    ax.scatter(
        supported["n_positive"], supported["auroc"], s=25,
        color=PALETTE["tabular_ehr"], alpha=0.72, linewidths=0,
        label="≥20 validation events", zorder=3,
    )
    ax.scatter(
        sparse["n_positive"], sparse["auroc"], s=29,
        facecolors="white", edgecolors=PALETTE["tabular_ehr"], linewidths=1,
        label="<20; unstable estimate", zorder=3,
    )
    ax.set_xscale("log")
    ax.set_xlabel("Positive events in validation set (log scale)")
    ax.set_ylabel("Fixed-horizon AUROC")
    ax.set_title("A  Outcome-level predictability", loc="left")
    ax.set_ylim(max(0.45, float(frame["auroc"].min()) - 0.06), min(1, float(frame["auroc"].max()) + 0.04))
    despine(ax, "y")

    order = (
        frame.groupby("display_group")["auroc"].median().sort_values().index.tolist()
    )
    for y, group in enumerate(order):
        values = frame.loc[frame["display_group"] == group, "auroc"].to_numpy()
        family_ax.scatter(
            values, np.full(len(values), y), s=13, alpha=0.3,
            color=PALETTE["tabular_ehr"], linewidths=0, zorder=2,
        )
        if len(values) > 1:
            q25, q75 = np.quantile(values, [0.25, 0.75])
            family_ax.plot([q25, q75], [y, y], color=PALETTE["ink"], lw=2, zorder=3)
        family_ax.scatter(
            np.median(values), y, s=28, marker="D", color=PALETTE["ink"], zorder=4,
        )
        family_ax.text(
            1.01, y, f"n={len(values)}", transform=family_ax.get_yaxis_transform(),
            va="center", fontsize=7, color=PALETTE["ink_secondary"],
        )
    family_ax.set_yticks(range(len(order)), order)
    family_ax.tick_params(axis="y", labelsize=7.0)
    family_ax.set_xlabel("AUROC (points: endpoints; bar: IQR; diamond: median)")
    family_ax.set_title("B  Clinical endpoint groups", loc="left")
    family_ax.set_xlim(max(0.45, float(frame["auroc"].min()) - 0.06), min(1, float(frame["auroc"].max()) + 0.04))
    despine(family_ax, "x")
    support_handles = [
        Line2D([], [], linestyle="none", marker="o", markersize=5,
               color=PALETTE["tabular_ehr"], label="20+ validation events"),
        Line2D([], [], linestyle="none", marker="o", markersize=5,
               markerfacecolor="white", markeredgecolor=PALETTE["tabular_ehr"],
               label="<20 events; unstable"),
    ]
    ax.legend(
        handles=support_handles, title="Validation support",
        loc="lower right", frameon=True, facecolor="white",
        edgecolor=PALETTE["grid"], framealpha=0.97,
        fontsize=7.2, title_fontsize=7.4,
    )
    fig.suptitle(f"XGBoost validation predictability at {horizon_days} days", x=0.04, ha="left")
    fig.text(
        0.04, 0.015,
        "Descriptive binary AUROC (not neural CIF-AUC); event counts are validation support. "
        "Horizon is from run provenance, not a CSV field.",
        fontsize=7.3, color=PALETTE["ink_secondary"],
    )
    fig.subplots_adjust(left=0.07, right=0.92, bottom=0.13, top=0.91, wspace=0.91)
    save_fig(fig, str(output))
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--aggregates", type=Path, required=True)
    parser.add_argument("--registry", type=Path, default=Path("opera/configs/experiment_registry.yaml"))
    parser.add_argument("--display-groups", type=Path, default=DEFAULT_DISPLAY_GROUPS)
    parser.add_argument("--horizon-days", type=int, default=365)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    plot_predictability(
        prepare_data(args.aggregates, args.registry, args.display_groups),
        args.output, args.horizon_days,
    )
    print(f"Wrote validation-only predictability prototype: {args.output}")


if __name__ == "__main__":
    main()
