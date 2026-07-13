"""Vocabulary/code embedding atlases for BONSAI and OPERA checkpoints."""

from __future__ import annotations

from typing import Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from opera.evaluation.treatment_embeddings import embedding_columns
from opera.visualization.embedding_plots import _cohort_colors, reduce_embeddings
from opera.visualization.style import (
    ANNOT_SIZE,
    CATEGORICAL,
    LEGEND_SIZE,
    LEGEND_TITLE_SIZE,
    NOTE_SIZE,
    PALETTE,
    add_panel_label,
    clean_2d_axes,
    despine,
    save_fig,
)


def project_vocabulary_embeddings(
    frame: pd.DataFrame,
    *,
    method: str = "umap",
    seed: int = 42,
    **kwargs,
) -> pd.DataFrame:
    """Project token embedding columns and return reusable atlas coordinates."""
    cols = embedding_columns(frame)
    values = frame[cols].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Vocabulary embedding matrix contains non-finite values.")
    if method == "pca":
        coords = PCA(n_components=2, random_state=seed).fit_transform(values)
    elif method in {"umap", "tsne"}:
        kwargs.setdefault("random_state", seed)
        coords = reduce_embeddings(values, method=method, **kwargs)
    else:
        raise ValueError("Projection method must be one of: umap, tsne, pca.")

    output = frame.drop(columns=cols).copy()
    output.insert(0, "atlas_y", coords[:, 1])
    output.insert(0, "atlas_x", coords[:, 0])
    return output


def plot_vocabulary_embedding_atlas(
    coordinates: pd.DataFrame,
    *,
    color_col: str = "token_family",
    token_col: str = "token",
    highlight_tokens: Optional[Sequence[str]] = None,
    min_group_n: int = 5,
    max_group_labels: int = 20,
    max_highlight_labels: int = 40,
    title: str = "Vocabulary embedding atlas",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Render a token/code embedding atlas grouped by token family/source."""
    required = {"atlas_x", "atlas_y", color_col, token_col}
    missing = required - set(coordinates.columns)
    if missing:
        raise ValueError(f"Coordinate table is missing: {sorted(missing)}")

    frame = coordinates.copy()
    frame[color_col] = frame[color_col].fillna("unknown").astype(str)
    group_counts = frame[color_col].value_counts()
    retained_groups = group_counts[group_counts >= min_group_n].index.tolist()
    colors = _cohort_colors(retained_groups)

    fig, ax = plt.subplots(figsize=(7.0, 4.8))
    other = ~frame[color_col].isin(retained_groups)
    if other.any():
        ax.scatter(
            frame.loc[other, "atlas_x"],
            frame.loc[other, "atlas_y"],
            s=6,
            color=PALETTE["missing"],
            alpha=0.22,
            linewidths=0,
            rasterized=True,
            zorder=1,
        )
    for group in retained_groups:
        subset = frame[frame[color_col] == group]
        ax.scatter(
            subset["atlas_x"],
            subset["atlas_y"],
            s=8,
            color=colors[group],
            alpha=0.36,
            linewidths=0,
            rasterized=True,
            label=group,
            zorder=2,
        )
    for group in group_counts.loc[retained_groups].head(max_group_labels).index:
        subset = frame[frame[color_col] == group]
        ax.text(
            float(subset["atlas_x"].median()),
            float(subset["atlas_y"].median()),
            f"{group} (n={len(subset)})",
            fontsize=ANNOT_SIZE,
            fontweight="semibold",
            color=colors[group],
            ha="center",
            va="center",
            bbox={
                "boxstyle": "round,pad=0.14",
                "facecolor": "white",
                "edgecolor": colors[group],
                "linewidth": 0.45,
                "alpha": 0.82,
            },
            zorder=4,
        )

    highlights = set(highlight_tokens or [])
    if highlights:
        highlighted = frame[frame[token_col].astype(str).isin(highlights)].head(
            max_highlight_labels
        )
        ax.scatter(
            highlighted["atlas_x"],
            highlighted["atlas_y"],
            s=34,
            color=PALETTE["ink"],
            edgecolor="white",
            linewidth=0.6,
            zorder=5,
        )
        for _, row in highlighted.iterrows():
            ax.annotate(
                str(row[token_col]),
                (row["atlas_x"], row["atlas_y"]),
                xytext=(3, 3),
                textcoords="offset points",
                fontsize=ANNOT_SIZE,
                color=PALETTE["ink"],
                bbox={
                    "boxstyle": "round,pad=0.10",
                    "facecolor": "white",
                    "edgecolor": "none",
                    "alpha": 0.75,
                },
                zorder=6,
            )

    _clean_token_axis(ax)
    ax.set_title(title)
    ax.text(
        0.01,
        0.01,
        "One point = one vocabulary token/code",
        transform=ax.transAxes,
        fontsize=NOTE_SIZE,
        color=PALETTE["ink_muted"],
    )
    if len(retained_groups) <= 12:
        ax.legend(
            title=color_col,
            fontsize=LEGEND_SIZE,
            title_fontsize=LEGEND_TITLE_SIZE,
            loc="upper right",
            framealpha=0.88,
        )
    fig.tight_layout()
    save_fig(fig, save_path)
    return fig


def plot_token_movement(
    movement: pd.DataFrame,
    *,
    stage_col: str = "embedding_stage",
    token_col: str = "token",
    metric: str = "cosine_distance",
    top_n: int = 25,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Plot the most moved code tokens for each comparator stage."""
    required = {stage_col, token_col, metric}
    missing = required - set(movement.columns)
    if missing:
        raise ValueError(f"Movement table is missing: {sorted(missing)}")
    if top_n < 1:
        raise ValueError("top_n must be positive.")

    stages = sorted(movement[stage_col].dropna().astype(str).unique())
    if not stages:
        raise ValueError("Movement table contains no comparator stages.")
    fig_height = max(2.8, 0.22 * top_n * len(stages) + 0.9)
    fig, axes = plt.subplots(
        len(stages),
        1,
        figsize=(7.0, fig_height),
        squeeze=False,
    )
    for index, stage in enumerate(stages):
        ax = axes[index, 0]
        subset = (
            movement[movement[stage_col].astype(str) == stage]
            .sort_values(metric, ascending=False)
            .head(top_n)
            .iloc[::-1]
        )
        color = CATEGORICAL[index % len(CATEGORICAL)]
        y = np.arange(len(subset))
        ax.barh(y, subset[metric], color=color, alpha=0.82)
        ax.set_yticks(y, labels=subset[token_col].astype(str))
        ax.set_xlabel(metric.replace("_", " "))
        ax.set_title(f"Most shifted tokens: {stage}")
        despine(ax, grid_axis="x")
        add_panel_label(ax, chr(ord("A") + index), x=-0.09, y=1.03)
    fig.tight_layout()
    save_fig(fig, save_path)
    return fig


def _clean_token_axis(ax: plt.Axes) -> None:
    clean_2d_axes(ax)
    ax.set_xlabel("Atlas 1", fontsize=NOTE_SIZE, color=PALETTE["ink_muted"])
    ax.set_ylabel("Atlas 2", fontsize=NOTE_SIZE, color=PALETTE["ink_muted"])
