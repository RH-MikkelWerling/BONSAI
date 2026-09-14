"""Disease geometry visualizations for OPERA embeddings."""

from __future__ import annotations

from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from opera.visualization.style import (
    ANNOT_SIZE,
    CATEGORICAL,
    FIG_FULL,
    LEGEND_SIZE,
    PALETTE,
    save_fig,
    sequential_cmap,
    setup_style,
)


def _layout(df: pd.DataFrame) -> np.ndarray:
    if {"x", "y"}.issubset(df.columns):
        return df[["x", "y"]].to_numpy(float)
    if {"pacmap_x", "pacmap_y"}.issubset(df.columns):
        return df[["pacmap_x", "pacmap_y"]].to_numpy(float)
    emb_cols = [c for c in df.columns if c.startswith("embedding_")]
    if len(emb_cols) >= 2:
        return df[emb_cols[:2]].to_numpy(float)
    raise ValueError(
        "embeddings_df must contain x/y, pacmap_x/pacmap_y, or embedding_* columns."
    )


def plot_disease_embedding(
    embeddings_df: pd.DataFrame,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Plot a PaCMAP disease embedding colored by disease.

    Input rows contain 2D coordinates and a `disease` column. The output figure
    shows disease clusters and annotates centroids, supporting the paper's
    claim that OPERA learns disease-aware representation geometry.
    """
    setup_style()
    coords = _layout(embeddings_df)
    df = embeddings_df.copy()
    fig, ax = plt.subplots(figsize=FIG_FULL)
    for i, (disease, group) in enumerate(df.groupby("disease", dropna=False)):
        idx = group.index.to_numpy()
        ax.scatter(
            coords[idx, 0],
            coords[idx, 1],
            s=8,
            color=CATEGORICAL[i % len(CATEGORICAL)],
            label=str(disease),
            rasterized=True,
            alpha=0.85,
        )
        centroid = coords[idx].mean(axis=0)
        ax.text(
            centroid[0],
            centroid[1],
            str(disease),
            fontsize=ANNOT_SIZE,
            ha="center",
            va="center",
        )
    ax.set_xlabel("PaCMAP 1")
    ax.set_ylabel("PaCMAP 2")
    ax.legend(fontsize=LEGEND_SIZE)
    fig.tight_layout()
    save_fig(fig, save_path)
    return fig


def plot_survival_gradient(
    embeddings_df: pd.DataFrame,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Plot embedding coordinates colored by survival quantile.

    Input rows contain 2D coordinates, `survival_quantile`, and disease labels.
    The figure visualizes whether survival risk forms smooth gradients across
    disease embedding space; disease boundaries are indicated by light contours.
    """
    setup_style()
    coords = _layout(embeddings_df)
    fig, ax = plt.subplots(figsize=FIG_FULL)
    sc = ax.scatter(
        coords[:, 0],
        coords[:, 1],
        c=embeddings_df["survival_quantile"],
        cmap=sequential_cmap("opera"),
        s=8,
        rasterized=True,
        alpha=0.9,
    )
    for _, group in embeddings_df.groupby("disease", dropna=False):
        idx = group.index.to_numpy()
        if len(idx) >= 3:
            try:
                ax.tricontour(
                    coords[idx, 0],
                    coords[idx, 1],
                    np.arange(len(idx)),
                    levels=[max(1, len(idx) * 0.3)],
                    colors=PALETTE["connector"],
                    linewidths=0.4,
                    alpha=0.3,
                )
            except Exception:
                pass
    fig.colorbar(sc, ax=ax, label="Survival quantile")
    ax.set_xlabel("PaCMAP 1")
    ax.set_ylabel("PaCMAP 2")
    fig.tight_layout()
    save_fig(fig, save_path)
    return fig


def plot_transfer_matrix(
    transfer_df: pd.DataFrame,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Plot transfer AUROC normalized by each disease diagonal.

    Input is either a matrix DataFrame or long rows with source/target/auroc.
    The output heatmap highlights cross-disease transfer above or below the
    within-disease reference, supporting adaptation and transfer claims.
    """
    setup_style()
    if {"source", "target", "auroc"}.issubset(transfer_df.columns):
        matrix = transfer_df.pivot(index="source", columns="target", values="auroc")
    else:
        matrix = transfer_df.copy()
    matrix = matrix.astype(float)
    diagonal = pd.Series(
        {
            item: matrix.loc[item, item]
            for item in matrix.index.intersection(matrix.columns)
        }
    )
    diagonal = diagonal.where(np.isfinite(diagonal) & (diagonal > 0))
    norm = matrix.div(diagonal.reindex(matrix.columns), axis="columns")
    fig, ax = plt.subplots(figsize=FIG_FULL)
    values = norm.to_numpy(float)
    finite = values[np.isfinite(values)]
    bound = max(0.25, float(np.max(np.abs(finite - 1))) if finite.size else 0.25)
    cmap = plt.colormaps["RdBu"].with_extremes(bad=PALETTE["missing"])
    im = ax.imshow(
        np.ma.masked_invalid(values), cmap=cmap, vmin=1 - bound, vmax=1 + bound
    )
    if max(values.shape) <= 12:
        for (row, col), value in np.ndenumerate(values):
            ax.text(
                col,
                row,
                f"{value:.2f}" if np.isfinite(value) else "—",
                ha="center",
                va="center",
                fontsize=8,
                color="white"
                if np.isfinite(value) and abs(value - 1) > 0.65 * bound
                else PALETTE["ink"],
            )
    ax.grid(False)
    ax.tick_params(length=0)
    ax.set_xlabel("Target disease")
    ax.set_ylabel("Source disease")
    ax.set_xticks(
        range(len(norm.columns)), labels=norm.columns, rotation=45, ha="right"
    )
    ax.set_yticks(range(len(norm.index)), labels=norm.index)
    fig.colorbar(im, ax=ax, label="Transfer AUROC / diagonal")
    fig.tight_layout()
    save_fig(fig, save_path)
    return fig
