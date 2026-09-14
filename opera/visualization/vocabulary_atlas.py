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
    CATEGORICAL,
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
    if (
        frame.empty
        or not np.isfinite(frame[["atlas_x", "atlas_y"]].to_numpy(dtype=float)).all()
    ):
        raise ValueError("Atlas requires nonempty, finite coordinates.")
    frame[color_col] = frame[color_col].fillna("unknown").astype(str)
    group_counts = frame[color_col].value_counts()
    retained_groups = group_counts[group_counts >= min_group_n].index.tolist()
    colors = _cohort_colors(retained_groups)

    # A dedicated catalogue keeps the map free of overlapping family labels.
    fig = plt.figure(figsize=(8.4, max(5.0, 0.27 * len(retained_groups) + 1.7)))
    grid = fig.add_gridspec(
        1,
        2,
        width_ratios=[3.4, 1.45],
        left=0.025,
        right=0.97,
        bottom=0.045,
        top=0.87,
        wspace=0.12,
    )
    ax = fig.add_subplot(grid[0, 0])
    catalogue = fig.add_subplot(grid[0, 1])
    ax.set_label("vocabulary:map")
    catalogue.set_label("vocabulary:catalogue")
    ax.set_facecolor("#FAFBFD")
    from matplotlib.colors import to_rgba

    point_colors = [
        to_rgba(colors.get(group, PALETTE["missing"])) for group in frame[color_col]
    ]
    order = np.random.default_rng(0).permutation(len(frame))
    ax.scatter(
        frame["atlas_x"].to_numpy()[order],
        frame["atlas_y"].to_numpy()[order],
        c=np.asarray(point_colors)[order],
        s=max(3, min(16, 300 / np.sqrt(max(len(frame), 1)))),
        alpha=0.72,
        linewidths=0,
        rasterized=True,
    )
    ax.set_aspect("equal", adjustable="datalim")
    clean_2d_axes(ax)
    # Numbered medians link regions to the catalogue without placing names
    # over points. Omit nearby anchors rather than implying distinct clusters.
    positions = []
    span = max(np.ptp(frame["atlas_x"]), np.ptp(frame["atlas_y"]), 1e-9)
    for index, group in enumerate(retained_groups):
        subset = frame[frame[color_col] == group]
        position = subset[["atlas_x", "atlas_y"]].median().to_numpy()
        if index < max_group_labels and all(
            np.linalg.norm(position - prev) > 0.09 * span for prev in positions
        ):
            ax.annotate(
                str(index + 1),
                position,
                ha="center",
                va="center",
                fontsize=8,
                fontweight="bold",
                color=colors[group],
                bbox={
                    "boxstyle": "circle,pad=.28",
                    "fc": "white",
                    "ec": colors[group],
                    "lw": 0.8,
                },
            )
            positions.append(position)
    catalogue.set_axis_off()
    catalogue.text(
        0,
        1.04,
        "TOKEN FAMILIES",
        transform=catalogue.transAxes,
        fontsize=8,
        fontweight="bold",
        color=PALETTE["ink_secondary"],
    )
    other_n = int((~frame[color_col].isin(retained_groups)).sum())
    entries = [
        (group, int(group_counts[group]), colors[group]) for group in retained_groups
    ]
    if other_n:
        entries.append(("Other / below size threshold", other_n, PALETTE["missing"]))
    import textwrap

    row_height = min(0.115, 0.88 / max(len(entries), 1))
    for index, (group, count, color) in enumerate(entries):
        y = 0.93 - index * row_height
        catalogue.text(
            0,
            y,
            f"{index + 1:02d}",
            transform=catalogue.transAxes,
            fontsize=8,
            color=color,
            fontweight="bold",
            va="top",
        )
        catalogue.text(
            0.15,
            y,
            textwrap.fill(group.replace("_", " "), 23),
            transform=catalogue.transAxes,
            fontsize=8.5,
            va="top",
        )
        catalogue.text(
            1,
            y,
            f"{count:,}",
            transform=catalogue.transAxes,
            fontsize=8,
            ha="right",
            va="top",
            color=PALETTE["ink_secondary"],
        )
        catalogue.plot(
            [0.15, 0.15 + 0.70 * count / max(len(frame), 1)],
            [y - 0.06] * 2,
            transform=catalogue.transAxes,
            color=color,
            lw=2.8,
            solid_capstyle="round",
        )
    highlights = set(highlight_tokens or [])
    highlighted = frame[frame[token_col].astype(str).isin(highlights)].head(
        max_highlight_labels
    )
    # Explicit token callouts retain their true anchor, with labels in a gutter.
    for index, (_, row) in enumerate(highlighted.sort_values("atlas_y").iterrows()):
        ax.annotate(
            str(row[token_col]),
            (row["atlas_x"], row["atlas_y"]),
            xytext=(0.01, (index + 1) / (len(highlighted) + 1)),
            textcoords="axes fraction",
            fontsize=7,
            ha="left",
            va="center",
            arrowprops={"arrowstyle": "-", "color": PALETTE["connector"], "lw": 0.6},
            bbox={"fc": "white", "ec": "none", "alpha": 0.95, "pad": 2},
        )
    fig.text(0.04, 0.95, title, fontsize=13, fontweight="semibold", va="top")
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
