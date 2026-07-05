"""Publication figures for the Bayesian natural-rarity analysis."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from opera.visualization.style import CATEGORICAL, FIG_FULL, save_fig, setup_style


METRIC_LABELS = {
    "auroc": "AUROC",
    "auprc": "AUPRC",
    "pr_skill": "PR skill",
    "brier_score": "Brier score",
    "brier_skill": "Brier skill",
    "log_loss": "log loss",
}


def aggregate_scatter_cells(
    deltas: pd.DataFrame,
    *,
    metric: str,
    rarity_column: str = "n_events_train",
) -> pd.DataFrame:
    """Collapse seed-level deltas to one transparent raw point per task cell."""
    data = deltas[deltas["metric"] == metric].copy()
    required = {
        "cell_id",
        "cohort",
        "outcome",
        "difference",
        "difference_se",
        rarity_column,
    }
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"Scatter input is missing columns: {sorted(missing)}")
    data["outcome_family"] = data.get(
        "outcome_family", pd.Series("Other", index=data.index)
    ).fillna("Other")

    def summarize(group: pd.DataFrame) -> pd.Series:
        values = group["difference"].to_numpy(dtype=float)
        ses = group["difference_se"].to_numpy(dtype=float)
        n_seed = len(group)
        within = float(np.nansum(ses**2) / max(n_seed**2, 1))
        between = float(np.nanvar(values, ddof=1) / n_seed) if n_seed > 1 else 0.0
        return pd.Series(
            {
                "cohort": group["cohort"].iloc[0],
                "outcome": group["outcome"].iloc[0],
                "outcome_family": group["outcome_family"].iloc[0],
                rarity_column: group[rarity_column].iloc[0],
                "difference": float(np.nanmean(values)),
                "display_se": float(np.sqrt(within + between)),
                "n_seeds": n_seed,
                "n_test_positive": group.get(
                    "n_test_positive", pd.Series(np.nan, index=group.index)
                ).iloc[0],
                "n_test_negative": group.get(
                    "n_test_negative", pd.Series(np.nan, index=group.index)
                ).iloc[0],
                "analysis_tier": (
                    "primary"
                    if (group.get("analysis_tier", "primary") == "primary").all()
                    else "partial_pool_only"
                ),
            }
        )

    return (
        data.groupby("cell_id", as_index=False, sort=True)
        .apply(summarize, include_groups=False)
        .reset_index(drop=True)
    )


def _point_sizes(values: pd.Series) -> np.ndarray:
    numeric = pd.to_numeric(values, errors="coerce").fillna(1.0).clip(lower=1.0)
    transformed = np.sqrt(numeric.to_numpy(dtype=float))
    if np.ptp(transformed) <= 1e-12:
        return np.full(len(values), 36.0)
    return 22.0 + 58.0 * (transformed - transformed.min()) / np.ptp(transformed)


def _selected_labels(
    cells: pd.DataFrame,
    curve: pd.DataFrame,
    *,
    rarity_column: str,
    max_labels: int,
) -> pd.DataFrame:
    if cells.empty or max_labels <= 0:
        return cells.iloc[0:0]
    expected = np.interp(
        cells[rarity_column].to_numpy(dtype=float),
        curve["training_events"].to_numpy(dtype=float),
        curve["median"].to_numpy(dtype=float),
    )
    ranked = cells.assign(residual=np.abs(cells["difference"] - expected))
    candidates = pd.concat(
        [
            ranked.nsmallest(2, rarity_column),
            ranked.nlargest(2, rarity_column),
            ranked.nlargest(max(2, max_labels - 4), "residual"),
        ],
        ignore_index=True,
    ).drop_duplicates("cell_id")
    return candidates.head(max_labels)


def plot_hierarchical_rarity_curve(
    deltas: pd.DataFrame,
    curve: pd.DataFrame,
    *,
    metric: str = "auroc",
    rarity_column: str = "n_events_train",
    model_label: str = "OPERA",
    comparator_label: str = "XGBoost",
    title: str = "Model benefit across natural task information",
    show_predictive_interval: bool = True,
    max_labels: int = 10,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Draw raw task deltas, posterior curve, and distinct uncertainty bands."""
    setup_style()
    cells = aggregate_scatter_cells(
        deltas,
        metric=metric,
        rarity_column=rarity_column,
    )
    cells = cells.replace([np.inf, -np.inf], np.nan).dropna(
        subset=[rarity_column, "difference"]
    )
    cells = cells[cells[rarity_column] > 0].copy()
    if cells.empty:
        raise ValueError("No finite cells are available for the rarity figure.")
    required_curve = {
        "training_events",
        "median",
        "lower_50",
        "upper_50",
        "lower_95",
        "upper_95",
    }
    missing = required_curve.difference(curve.columns)
    if missing:
        raise ValueError(f"Posterior curve is missing columns: {sorted(missing)}")

    fig, ax = plt.subplots(figsize=FIG_FULL)
    x_min = min(cells[rarity_column].min(), curve["training_events"].min())
    x_max = max(cells[rarity_column].max(), curve["training_events"].max())
    y_values = np.concatenate(
        [cells["difference"].to_numpy(), curve["lower_95"], curve["upper_95"]]
    )
    y_pad = max(0.015, 0.08 * np.ptp(y_values))
    y_min, y_max = float(np.nanmin(y_values) - y_pad), float(np.nanmax(y_values) + y_pad)
    ax.axhspan(0.0, y_max, color="#EEF1FA", alpha=0.42, zorder=0)
    ax.axhspan(y_min, 0.0, color="#F4F4F4", alpha=0.58, zorder=0)

    x_curve = curve["training_events"].to_numpy(dtype=float)
    if show_predictive_interval and {
        "predictive_lower_95",
        "predictive_upper_95",
    }.issubset(curve.columns):
        ax.fill_between(
            x_curve,
            curve["predictive_lower_95"],
            curve["predictive_upper_95"],
            color="#AEB8D8",
            alpha=0.14,
            linewidth=0,
            label="95% prediction interval",
            zorder=1,
        )
    ax.fill_between(
        x_curve,
        curve["lower_95"],
        curve["upper_95"],
        color="#5264A8",
        alpha=0.18,
        linewidth=0,
        label="95% credible interval",
        zorder=2,
    )
    ax.fill_between(
        x_curve,
        curve["lower_50"],
        curve["upper_50"],
        color="#2D3A8C",
        alpha=0.25,
        linewidth=0,
        label="50% credible interval",
        zorder=3,
    )
    ax.plot(
        x_curve,
        curve["median"],
        color="#1A237E",
        linewidth=2.25,
        label="Posterior median",
        zorder=5,
    )

    families = sorted(cells["outcome_family"].astype(str).unique())
    colors = {
        family: CATEGORICAL[index % len(CATEGORICAL)]
        for index, family in enumerate(families)
    }
    cells["point_size"] = _point_sizes(cells["n_test_positive"])
    for family, group in cells.groupby("outcome_family", sort=True):
        primary = group[group["analysis_tier"] == "primary"]
        pooled = group[group["analysis_tier"] != "primary"]
        if not primary.empty:
            ax.scatter(
                primary[rarity_column],
                primary["difference"],
                s=primary["point_size"],
                c=colors[str(family)],
                edgecolors="white",
                linewidths=0.55,
                alpha=0.76,
                rasterized=True,
                zorder=4,
            )
        if not pooled.empty:
            ax.scatter(
                pooled[rarity_column],
                pooled["difference"],
                s=pooled["point_size"],
                facecolors="white",
                edgecolors=colors[str(family)],
                linewidths=0.9,
                alpha=0.60,
                rasterized=True,
                zorder=4,
            )

    labels = _selected_labels(
        cells,
        curve,
        rarity_column=rarity_column,
        max_labels=max_labels,
    )
    for _, row in labels.iterrows():
        y_offset = 6 if row["difference"] >= 0 else -9
        ax.annotate(
            f"{row['cohort']} · {str(row['outcome']).replace('_', ' ')}",
            (row[rarity_column], row["difference"]),
            xytext=(4, y_offset),
            textcoords="offset points",
            fontsize=6.5,
            color="#333333",
            ha="left",
            va="bottom" if y_offset > 0 else "top",
            zorder=6,
        )

    rug_y = y_min + 0.018 * (y_max - y_min)
    ax.scatter(
        cells[rarity_column],
        np.full(len(cells), rug_y),
        marker="|",
        s=18,
        color="#555555",
        alpha=0.22,
        linewidths=0.6,
        rasterized=True,
        zorder=2,
    )
    ax.axhline(0.0, color="#333333", linewidth=0.85, zorder=4)
    ax.set_xscale("log", base=2)
    ax.set_xlim(x_min / 1.12, x_max * 1.12)
    ax.set_ylim(y_min, y_max)
    ax.set_xlabel("Observed training events (log2 scale)")
    metric_name = METRIC_LABELS.get(metric, metric.replace("_", " ").upper())
    ax.set_ylabel(f"Δ{metric_name} ({model_label} − {comparator_label})")
    ax.set_title(title, loc="left", pad=9)
    ax.text(
        0.995,
        0.985,
        f"Positive values favour {model_label}",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=7.5,
        color="#46517E",
    )
    ax.grid(axis="x", which="minor", visible=False)

    family_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            markerfacecolor=colors[family],
            markeredgecolor="white",
            markersize=5.5,
            label=family,
        )
        for family in families
    ]
    interval_handles = [
        Line2D([0], [0], color="#1A237E", linewidth=2.2, label="Posterior median"),
        Patch(facecolor="#5264A8", alpha=0.18, label="95% credible interval"),
    ]
    if show_predictive_interval:
        interval_handles.append(
            Patch(facecolor="#AEB8D8", alpha=0.14, label="95% prediction interval")
        )
    first_legend = ax.legend(
        handles=family_handles,
        title="Outcome family",
        loc="upper left",
        bbox_to_anchor=(0.0, -0.17),
        ncol=min(4, max(1, len(families))),
        frameon=False,
        fontsize=7,
        title_fontsize=7.5,
    )
    ax.add_artist(first_legend)
    ax.legend(
        handles=interval_handles,
        loc="upper right",
        bbox_to_anchor=(1.0, -0.17),
        frameon=False,
        fontsize=7,
        ncol=1,
    )
    fig.tight_layout(rect=(0, 0.11, 1, 1))
    if save_path:
        save_fig(fig, save_path)
        target = Path(save_path)
        fig.savefig(target.with_suffix(".svg"), bbox_inches="tight", facecolor="white")
    return fig
