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


def plot_vocabulary_family_facets(
    coordinates: pd.DataFrame,
    *,
    color_col: str = "token_family",
    min_group_n: int = 5,
    title: str = "Vocabulary families in a shared atlas",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Highlight one family per panel on identical, greyed-out coordinates."""
    required = {"atlas_x", "atlas_y", color_col}
    missing = required - set(coordinates.columns)
    if missing:
        raise ValueError(f"Coordinate table is missing: {sorted(missing)}")
    frame = coordinates.copy()
    if frame.empty or not np.isfinite(frame[["atlas_x", "atlas_y"]].to_numpy(dtype=float)).all():
        raise ValueError("Atlas requires nonempty, finite coordinates.")
    frame[color_col] = frame[color_col].fillna("unknown").astype(str)
    counts = frame[color_col].value_counts()
    families = counts[counts >= min_group_n].index.tolist()
    if not families:
        raise ValueError("No family meets min_group_n.")
    colors = _cohort_colors(families)
    ncols = min(3, len(families))
    nrows = int(np.ceil(len(families) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.2 * ncols, 3.15 * nrows), squeeze=False)
    xy = frame[["atlas_x", "atlas_y"]].to_numpy()
    center = (xy.min(axis=0) + xy.max(axis=0)) / 2
    halfspan = max(np.ptp(xy[:, 0]), np.ptp(xy[:, 1])) * 0.55
    halfspan = max(halfspan, 1e-9)
    for ax, family in zip(axes.flat, families):
        selected = frame[color_col].eq(family).to_numpy()
        ax.set_facecolor("#FAFBFD")
        ax.scatter(xy[:, 0], xy[:, 1], s=2.5, c="#C8CDD2", alpha=0.28,
                   linewidths=0, rasterized=True)
        ax.scatter(xy[selected, 0], xy[selected, 1], s=4.5, c=colors[family],
                   alpha=0.85, linewidths=0, rasterized=True)
        ax.set_xlim(center[0] - halfspan, center[0] + halfspan)
        ax.set_ylim(center[1] - halfspan, center[1] + halfspan)
        ax.set_aspect("equal", adjustable="box")
        ax.set_title(f"{family.replace('_', ' ')}  ·  n={counts[family]:,}",
                     loc="left", fontsize=10, color=PALETTE["ink"])
        clean_2d_axes(ax)
    for ax in list(axes.flat)[len(families):]:
        ax.set_visible(False)
    fig.suptitle(title, x=0.04, ha="left", fontsize=14, fontweight="semibold")
    fig.subplots_adjust(left=0.04, right=0.98, bottom=0.035, top=0.92,
                        wspace=0.08, hspace=0.21)
    save_fig(fig, save_path)
    return fig


def plot_vocabulary_focus(
    coordinates: pd.DataFrame,
    neighbors: pd.DataFrame,
    *,
    query_token: str,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Show a token in the full atlas, a local view, and original-space neighbours."""
    required = {"token", "atlas_x", "atlas_y"}
    if not required.issubset(coordinates):
        raise ValueError(f"Atlas coordinates lack {sorted(required - set(coordinates))}")
    focal = coordinates.loc[coordinates["token"].eq(query_token)]
    if len(focal) != 1:
        raise ValueError(f"Expected one atlas point for {query_token!r}")
    ranked = neighbors.loc[neighbors["query_token"].eq(query_token)].sort_values("neighbor_rank")
    if ranked.empty:
        raise ValueError(f"No neighbours for {query_token!r}")
    marked = coordinates.loc[coordinates["token"].isin(ranked["neighbor_token"])].copy()
    if len(marked) != len(ranked):
        raise ValueError("Some cosine-nearest tokens are missing from atlas coordinates")
    points = pd.concat([focal, marked], ignore_index=True)
    xy = coordinates[["atlas_x", "atlas_y"]].to_numpy(dtype=float)
    fig = plt.figure(figsize=(12.2, 4.8))
    grid = fig.add_gridspec(1, 3, width_ratios=[1.05, 1.05, 1.25],
                           left=0.045, right=0.98, bottom=0.09, top=0.82, wspace=0.14)
    full, zoom, listing = [fig.add_subplot(grid[0, i]) for i in range(3)]
    for ax in (full, zoom):
        ax.set_facecolor("#FAFBFD")
        ax.scatter(xy[:, 0], xy[:, 1], s=2, c="#C8CDD2", alpha=0.32,
                   linewidths=0, rasterized=True)
        ax.scatter(marked["atlas_x"], marked["atlas_y"], s=21,
                   c=CATEGORICAL[3], edgecolors="white", linewidths=0.4, zorder=3)
        ax.scatter(focal["atlas_x"], focal["atlas_y"], s=48,
                   c=CATEGORICAL[0], edgecolors="white", linewidths=0.6, zorder=4)
        clean_2d_axes(ax)
        ax.set_aspect("equal", adjustable="box")
    full.set_title("Full vocabulary", loc="left", fontsize=10)
    zoom.set_title("Focal token + cosine neighbours", loc="left", fontsize=10)
    extent = points[["atlas_x", "atlas_y"]].to_numpy(dtype=float)
    mid = (extent.min(axis=0) + extent.max(axis=0)) / 2
    radius = max(np.ptp(extent[:, 0]), np.ptp(extent[:, 1]),
                 0.04 * max(np.ptp(xy[:, 0]), np.ptp(xy[:, 1]))) * 0.62
    zoom.set_xlim(mid[0] - radius, mid[0] + radius)
    zoom.set_ylim(mid[1] - radius, mid[1] + radius)
    for _, row in ranked.iterrows():
        point = marked.loc[marked["token"].eq(row["neighbor_token"])].iloc[0]
        zoom.annotate(str(int(row["neighbor_rank"])),
                      (point["atlas_x"], point["atlas_y"]),
                      xytext=(3, 3), textcoords="offset points", fontsize=7,
                      color=PALETTE["ink"])
    listing.set_axis_off()
    listing.text(0, 1.02, "NEAREST IN ORIGINAL EMBEDDING SPACE",
                 transform=listing.transAxes, fontsize=8, fontweight="semibold")
    for index, (_, row) in enumerate(ranked.iterrows()):
        y = 0.92 - index * 0.083
        listing.text(0, y, f"{int(row['neighbor_rank']):02d}",
                     transform=listing.transAxes, fontsize=8, color=CATEGORICAL[3], va="top")
        listing.text(0.11, y, str(row["neighbor_token"])[:43],
                     transform=listing.transAxes, fontsize=7.5, va="top")
        listing.text(1, y, f"{row['cosine_similarity']:.3f}",
                     transform=listing.transAxes, fontsize=7.5, ha="right", va="top",
                     color=PALETTE["ink_secondary"])
    fig.text(0.045, 0.95, query_token, fontsize=13, fontweight="semibold", va="top")
    fig.text(0.045, 0.88, "Ranked by cosine similarity in the learned token vectors; map is descriptive.",
             fontsize=8, color=PALETTE["ink_secondary"], va="top")
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
