"""Joint disease and first-line regimen maps for OPERA embeddings."""

from __future__ import annotations

from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from sklearn.decomposition import PCA

from opera.evaluation.treatment_embeddings import embedding_columns
from opera.visualization.embedding_plots import _cohort_colors, reduce_embeddings
from opera.visualization.style import CATEGORICAL, add_panel_label, save_fig


REGIMEN_MARKERS = ("o", "s", "^", "D", "P", "X", "v", "<", ">", "h", "p", "*")


def project_embedding_frame(
    frame: pd.DataFrame,
    *,
    method: str = "umap",
    seed: int = 42,
    **kwargs,
) -> np.ndarray:
    """Project canonical embedding columns to one shared two-dimensional map."""
    values = frame[embedding_columns(frame)].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Embedding matrix contains non-finite values.")
    if method == "pca":
        return PCA(n_components=2, random_state=seed).fit_transform(values)
    if method not in {"umap", "tsne"}:
        raise ValueError("Projection method must be one of: umap, tsne, pca.")
    kwargs.setdefault("random_state", seed)
    return reduce_embeddings(values, method=method, **kwargs)


def _clean_axis(ax: plt.Axes, xlabel: str = "Atlas 1", ylabel: str = "Atlas 2") -> None:
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel(xlabel, fontsize=7, color="#777777")
    ax.set_ylabel(ylabel, fontsize=7, color="#777777")
    for spine in ax.spines.values():
        spine.set_visible(False)


def _scatter_groups(
    ax: plt.Axes,
    coordinates: pd.DataFrame,
    *,
    group_col: str,
    colors: dict[str, str],
    min_n: int,
    max_labels: int,
) -> None:
    valid = coordinates.dropna(subset=[group_col]).copy()
    valid[group_col] = valid[group_col].astype(str)
    counts = valid[group_col].value_counts()
    retained = counts[counts >= min_n]
    other_mask = ~valid[group_col].isin(retained.index)
    if other_mask.any():
        ax.scatter(
            valid.loc[other_mask, "atlas_x"],
            valid.loc[other_mask, "atlas_y"],
            s=3,
            color="#D7D7D7",
            alpha=0.22,
            linewidths=0,
            rasterized=True,
            zorder=1,
        )
    for group in retained.index:
        subset = valid[valid[group_col] == group]
        ax.scatter(
            subset["atlas_x"],
            subset["atlas_y"],
            s=4,
            color=colors[group],
            alpha=0.30,
            linewidths=0,
            rasterized=True,
            zorder=2,
        )
    for group in retained.head(max_labels).index:
        subset = valid[valid[group_col] == group]
        x = float(subset["atlas_x"].mean())
        y = float(subset["atlas_y"].mean())
        ax.text(
            x,
            y,
            group,
            color=colors[group],
            fontsize=6.5,
            fontweight="semibold",
            ha="center",
            va="center",
            bbox={
                "boxstyle": "round,pad=0.16",
                "facecolor": "white",
                "edgecolor": colors[group],
                "linewidth": 0.45,
                "alpha": 0.88,
            },
            zorder=5,
        )


def plot_disease_treatment_atlas(
    coordinates: pd.DataFrame,
    centroids: pd.DataFrame,
    *,
    disease_col: str = "disease",
    treatment_col: str = "regimen_group",
    min_disease_n: int = 20,
    min_treatment_n: int = 20,
    min_joint_n: int = 10,
    max_disease_labels: int = 20,
    max_treatment_labels: int = 20,
    max_joint_labels: int = 28,
    title: str = "Hematology disease–treatment embedding atlas",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Render disease, regimen, and joint disease–regimen geography.

    All panels use identical patient coordinates.  In the joint panel, color
    identifies disease, marker identifies regimen, and line segments connect
    each disease centroid with its regimen-specific patient centroid.  Marker
    area and line width scale with the disease–regimen sample size.
    """
    required = {"atlas_x", "atlas_y", disease_col, treatment_col}
    missing = required - set(coordinates.columns)
    if missing:
        raise ValueError(f"Coordinate table is missing: {sorted(missing)}")

    disease_values = sorted(
        coordinates[disease_col].dropna().astype(str).unique().tolist()
    )
    treatment_values = sorted(
        coordinates[treatment_col].dropna().astype(str).unique().tolist()
    )
    disease_colors = _cohort_colors(disease_values)
    treatment_colors = _cohort_colors(treatment_values)
    treatment_markers = {
        value: REGIMEN_MARKERS[index % len(REGIMEN_MARKERS)]
        for index, value in enumerate(treatment_values)
    }

    fig, axes = plt.subplots(1, 3, figsize=(12.2, 4.35), sharex=True, sharey=True)
    fig.suptitle(title, fontsize=11, fontweight="semibold", y=1.01)

    _scatter_groups(
        axes[0],
        coordinates,
        group_col=disease_col,
        colors=disease_colors,
        min_n=min_disease_n,
        max_labels=max_disease_labels,
    )
    axes[0].set_title("Disease geography", fontsize=9)
    axes[0].text(
        0.01,
        0.01,
        "Color = disease",
        transform=axes[0].transAxes,
        fontsize=6.5,
        color="#555555",
    )

    _scatter_groups(
        axes[1],
        coordinates,
        group_col=treatment_col,
        colors=treatment_colors,
        min_n=min_treatment_n,
        max_labels=max_treatment_labels,
    )
    axes[1].set_title("First-line regimen geography", fontsize=9)
    axes[1].text(
        0.01,
        0.01,
        "Color = disease-specific regimen",
        transform=axes[1].transAxes,
        fontsize=6.5,
        color="#555555",
    )

    joint_ax = axes[2]
    joint_ax.scatter(
        coordinates["atlas_x"],
        coordinates["atlas_y"],
        s=3,
        color="#CFCFCF",
        alpha=0.10,
        linewidths=0,
        rasterized=True,
        zorder=1,
    )
    disease_centroids = centroids[centroids["kind"] == "disease"].copy()
    disease_centroids[disease_col] = disease_centroids[disease_col].astype(str)
    disease_centroids = disease_centroids.set_index(disease_col)
    joint = centroids[
        (centroids["kind"] == "disease_treatment") & (centroids["n"] >= min_joint_n)
    ].copy()
    joint[disease_col] = joint[disease_col].astype(str)
    joint[treatment_col] = joint[treatment_col].astype(str)

    maximum_n = max(float(joint["n"].max()), 1.0) if not joint.empty else 1.0
    for _, row in joint.iterrows():
        disease = str(row[disease_col])
        treatment = str(row[treatment_col])
        if disease not in disease_centroids.index:
            continue
        origin = disease_centroids.loc[disease]
        scale = np.sqrt(float(row["n"]) / maximum_n)
        joint_ax.plot(
            [origin["atlas_x"], row["atlas_x"]],
            [origin["atlas_y"], row["atlas_y"]],
            color=disease_colors[disease],
            alpha=0.18 + 0.36 * scale,
            linewidth=0.35 + 1.5 * scale,
            zorder=2,
        )
        joint_ax.scatter(
            row["atlas_x"],
            row["atlas_y"],
            s=18 + 100 * scale,
            marker=treatment_markers[treatment],
            color=disease_colors[disease],
            edgecolor="white",
            linewidth=0.55,
            alpha=0.88,
            zorder=4,
        )

    for disease, row in disease_centroids.iterrows():
        if int(row["n"]) < min_disease_n:
            continue
        joint_ax.scatter(
            row["atlas_x"],
            row["atlas_y"],
            s=65,
            marker="o",
            color=disease_colors[disease],
            edgecolor="#222222",
            linewidth=0.7,
            zorder=5,
        )
        joint_ax.annotate(
            disease,
            (row["atlas_x"], row["atlas_y"]),
            xytext=(0, 7),
            textcoords="offset points",
            fontsize=6.3,
            fontweight="bold",
            ha="center",
            va="bottom",
            color=disease_colors[disease],
            zorder=6,
        )

    for _, row in (
        joint.sort_values("n", ascending=False).head(max_joint_labels).iterrows()
    ):
        disease = str(row[disease_col])
        treatment = str(row[treatment_col])
        joint_ax.annotate(
            f"{disease} · {treatment}",
            (row["atlas_x"], row["atlas_y"]),
            xytext=(3, 3),
            textcoords="offset points",
            fontsize=5.5,
            color=disease_colors[disease],
            bbox={
                "boxstyle": "round,pad=0.10",
                "facecolor": "white",
                "edgecolor": "none",
                "alpha": 0.72,
            },
            zorder=7,
        )

    top_treatments = (
        coordinates[treatment_col].dropna().astype(str).value_counts().head(8).index
    )
    marker_handles = [
        Line2D(
            [0],
            [0],
            marker=treatment_markers[treatment],
            color="none",
            markerfacecolor="#777777",
            markeredgecolor="white",
            markersize=6,
            label=treatment,
        )
        for treatment in top_treatments
    ]
    if marker_handles:
        joint_ax.legend(
            handles=marker_handles,
            title="Regimen marker (most common)",
            fontsize=5.8,
            title_fontsize=6.0,
            loc="upper right",
            framealpha=0.88,
            ncol=1,
        )
    joint_ax.set_title("Joint disease–regimen atlas", fontsize=9)
    joint_ax.text(
        0.01,
        0.01,
        "Color = disease  ·  marker = regimen  ·  size = n",
        transform=joint_ax.transAxes,
        fontsize=6.5,
        color="#555555",
    )

    for label, ax in zip(("A", "B", "C"), axes):
        add_panel_label(ax, label, x=-0.04, y=1.03)
        _clean_axis(ax)
    fig.tight_layout()
    save_fig(fig, save_path)
    return fig


def plot_treatment_probe_performance(
    results: pd.DataFrame,
    *,
    disease_col: str = "disease",
    stage_col: str = "embedding_stage",
    metric: str = "balanced_accuracy",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Compare disease-conditioned regimen accessibility across embeddings."""
    required = {disease_col, stage_col, metric, "majority_balanced_accuracy"}
    missing = required - set(results.columns)
    if missing:
        raise ValueError(f"Treatment probe results are missing: {sorted(missing)}")
    diseases = sorted(results[disease_col].astype(str).unique())
    stages = sorted(results[stage_col].astype(str).unique())
    y_positions = {disease: index for index, disease in enumerate(diseases)}
    offsets = np.linspace(-0.22, 0.22, max(len(stages), 1))

    fig_height = max(2.8, 0.34 * len(diseases) + 1.2)
    fig, ax = plt.subplots(figsize=(7.0, fig_height))
    for stage_index, stage in enumerate(stages):
        subset = results[results[stage_col].astype(str) == stage]
        y = np.array([y_positions[str(value)] for value in subset[disease_col]])
        ax.scatter(
            subset[metric],
            y + offsets[stage_index],
            s=34,
            color=CATEGORICAL[stage_index % len(CATEGORICAL)],
            label=stage,
            edgecolor="white",
            linewidth=0.5,
            zorder=3,
        )
    baseline = results.groupby(disease_col)["majority_balanced_accuracy"].mean()
    for disease, value in baseline.items():
        ax.scatter(
            value,
            y_positions[str(disease)],
            marker="|",
            s=110,
            linewidth=1.2,
            color="#555555",
            zorder=2,
        )
    ax.set_yticks(range(len(diseases)), labels=diseases)
    ax.set_xlabel("Balanced accuracy for first-line regimen")
    ax.set_ylabel("Disease")
    ax.set_xlim(0, 1)
    ax.axvline(0.5, color="#BBBBBB", linewidth=0.7, linestyle="--", zorder=1)
    ax.legend(title="Frozen embedding", loc="lower right")
    ax.set_title("How much treatment-selection information is accessible?")
    ax.text(
        0.99,
        0.01,
        "Vertical ticks: majority-class baseline",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=6.5,
        color="#666666",
    )
    fig.tight_layout()
    save_fig(fig, save_path)
    return fig
