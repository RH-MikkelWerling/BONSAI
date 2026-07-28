"""Plot generic frozen patient embeddings with reusable 2D coordinates.

Rendering notes (why this differs from the first version):

* Categorical views draw every point in a single shuffled scatter call. Looping
  `scatter` per category makes draw order equal to category order, so a large
  level painted late silently buries every earlier level. That turns "level A is
  absent from this region" and "level A is underneath 29k points of level B" into
  the same picture.
* High-cardinality labels (more than `facet_threshold` levels) are drawn as small
  multiples: one panel per level, highlighted against a grey backdrop of the full
  cohort. No palette distinguishes 24 colours, let alone 40.
* Continuous views default to robust colour limits, because a skewed covariate
  (age concentrated in 60 to 85) rendered on full-range viridis looks flat even
  when it is structured.
* Point size and alpha scale with n instead of being fixed at s=8, alpha=0.55.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Ellipse
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

from bonsai.functional.outcomes import binarize_outcomes
from opera.run.extract_patient_embeddings import read_index_table
from opera.visualization.style import CATEGORICAL, clean_2d_axes, save_fig

GREY = "#c8c8c8"


# --------------------------------------------------------------------------- #
# loading
# --------------------------------------------------------------------------- #
def _left_join_on_subject(
    frame: pd.DataFrame,
    addition: pd.DataFrame,
    *,
    subject_col: str,
    source_label: str,
) -> pd.DataFrame:
    """Left-join one table onto ``frame`` via a string-cast subject id.

    Shared by patient metadata tables and outcome-derived columns so both get
    the same duplicate-id, column-overlap, and one-to-one guarantees.
    """
    if subject_col not in addition:
        raise ValueError(f"{source_label} is missing subject column {subject_col!r}.")
    if addition[subject_col].duplicated().any():
        raise ValueError(f"{source_label} contains duplicate patient identifiers.")
    addition = addition.copy()
    addition["_join_id"] = addition[subject_col].astype(str)
    addition = addition.drop(columns=[subject_col], errors="ignore")
    overlap = (set(frame) & set(addition)) - {"_join_id"}
    if overlap:
        raise ValueError(
            f"{source_label} repeats columns already loaded: {sorted(overlap)}."
        )
    return frame.merge(addition, on="_join_id", how="left", validate="one_to_one")


def load_embedding_frame(
    embedding_path: str | Path,
    metadata_paths: list[str],
    *,
    subject_col: str,
    outcome_frames: list[pd.DataFrame] = (),
) -> tuple[np.ndarray, pd.DataFrame]:
    """Load embeddings and left-join patient metadata and outcome-derived tables."""
    artifact = np.load(embedding_path)
    if not {"subject_ids", "embeddings"} <= set(artifact.files):
        raise ValueError("Embedding NPZ must contain subject_ids and embeddings.")
    subject_ids = artifact["subject_ids"]
    embeddings = np.asarray(artifact["embeddings"], dtype=np.float32)
    if embeddings.ndim != 2 or len(subject_ids) != len(embeddings):
        raise ValueError("Embedding NPZ has inconsistent subject and embedding shapes.")

    frame_values = {
        "subject_id": subject_ids,
        "_join_id": pd.Series(subject_ids).astype(str),
    }
    if "splits" in artifact.files:
        if len(artifact["splits"]) != len(subject_ids):
            raise ValueError("Embedding NPZ has an inconsistent splits array.")
        frame_values["split"] = artifact["splits"].astype(str)
    frame = pd.DataFrame(frame_values)

    # Any 1-D per-patient array in the NPZ becomes a colourable nuisance
    # covariate. Sequence length in particular is a prime suspect for the
    # satellite islands that t-SNE throws off the main mass.
    reserved = {"subject_ids", "embeddings", "splits"}
    for name in artifact.files:
        if name in reserved:
            continue
        values = np.asarray(artifact[name])
        if values.ndim == 1 and len(values) == len(subject_ids):
            frame_values[name] = values
    frame = pd.DataFrame(frame_values)

    # Norm is not metadata but it behaves like a covariate, and for mean-pooled
    # transformer states it often tracks record length closely enough to
    # masquerade as biology.
    frame["embedding_norm"] = np.linalg.norm(embeddings, axis=1)

    for raw_path in metadata_paths:
        metadata = read_index_table(raw_path)
        frame = _left_join_on_subject(
            frame, metadata, subject_col=subject_col, source_label=str(raw_path)
        )
    for index, outcome_frame in enumerate(outcome_frames):
        frame = _left_join_on_subject(
            frame,
            outcome_frame,
            subject_col="subject_id",
            source_label=f"outcome table #{index + 1}",
        )
    return embeddings, frame.drop(columns="_join_id")


def load_outcome_time_columns(
    outcome_path: str | Path,
    *,
    outcome_name: str | None = None,
    n_hours_start_include: int = 1,
    horizons_days: tuple[float, ...] = (),
) -> pd.DataFrame:
    """Derive time-to-event/event/binary-within-horizon columns from one outcome table.

    Reuses ``bonsai.functional.outcomes.binarize_outcomes`` so event-vs.-
    administrative-censoring semantics match training and evaluation exactly,
    rather than reimplementing that logic here. The outcome table must follow
    the standard ``outcomes/{name}.parquet`` contract (``subject_id``,
    ``index_date``, ``outcome_date``, ``censor_date``; see DATA_FORMAT.md
    section 4). Returns one row per patient with:

    - ``{name}_time_days``: time from index to the event or to censoring.
    - ``{name}_event``: whether the primary event was observed (open-ended,
      no horizon).
    - ``{name}_within_{horizon}d`` per entry in ``horizons_days``: whether the
      event was observed within that many days. Patients censored before the
      horizon without the event are left as missing (their status is
      genuinely unknown at that horizon), matching
      ``require_min_followup=True`` elsewhere in this repo.
    """
    outcome_frame = read_index_table(outcome_path)
    name = outcome_name or Path(outcome_path).stem
    records = binarize_outcomes(
        outcome_frame,
        n_hours_start_include=n_hours_start_include,
        n_hours_end_include=None,
    )
    if not records:
        raise ValueError(f"{outcome_path} produced no usable outcome records.")
    subject_ids = list(records.keys())
    derived = pd.DataFrame(
        {
            "subject_id": subject_ids,
            f"{name}_time_days": [
                records[subject_id]["time_days"] for subject_id in subject_ids
            ],
            f"{name}_event": [
                records[subject_id]["event"] == 1 for subject_id in subject_ids
            ],
        }
    )
    for horizon in horizons_days:
        if horizon <= 0:
            raise ValueError("--outcome-horizon-days must be positive.")
        horizon_records = binarize_outcomes(
            outcome_frame,
            n_hours_start_include=n_hours_start_include,
            n_hours_end_include=horizon * 24.0,
            require_min_followup=True,
        )
        label_by_subject = {
            subject_id: bool(record["label"])
            for subject_id, record in horizon_records.items()
        }
        column = f"{name}_within_{horizon:g}d"
        derived[column] = pd.array(
            [label_by_subject.get(subject_id) for subject_id in subject_ids],
            dtype="boolean",
        )
    return derived


# --------------------------------------------------------------------------- #
# projection
# --------------------------------------------------------------------------- #
def prepare_embeddings(
    embeddings: np.ndarray,
    *,
    scaling: str,
    pca_dim: int | None,
    seed: int,
) -> np.ndarray:
    """Scale and optionally pre-reduce before the 2D projection.

    Raw transformer states have per-dimension scales spanning orders of
    magnitude, so a euclidean neighbourhood graph is dominated by whichever few
    dimensions happen to be large. `l2` makes euclidean distance monotone in
    cosine distance; `standardize` equalises dimension influence.
    """
    matrix = np.asarray(embeddings, dtype=np.float32)
    if scaling == "l2":
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        matrix = matrix / np.clip(norms, 1e-8, None)
    elif scaling == "standardize":
        matrix = StandardScaler().fit_transform(matrix).astype(np.float32)
    elif scaling != "none":
        raise ValueError(f"Unknown --scaling {scaling!r}.")
    if pca_dim is not None and 0 < pca_dim < matrix.shape[1]:
        matrix = PCA(n_components=pca_dim, random_state=seed).fit_transform(matrix)
    return np.ascontiguousarray(matrix, dtype=np.float32)


def reduce_patient_embeddings(
    embeddings: np.ndarray,
    *,
    method: str,
    seed: int,
    n_neighbors: int,
    min_dist: float,
    perplexity: float = 30.0,
    tsne_init: str = "pca",
    metric: str = "euclidean",
) -> tuple[np.ndarray, str]:
    """Return deterministic PCA, t-SNE, or UMAP coordinates.

    `tsne_init` matters for interpretation. With init="pca" the macro layout is
    seeded from PC1/PC2 and t-SNE largely preserves it, so a clean global
    left-to-right gradient in the final plot is mostly a statement about the top
    principal components. Use init="random" as a cross-check before claiming the
    projection discovered the gradient.
    """
    if method == "pca":
        return PCA(n_components=2, random_state=seed).fit_transform(embeddings), "pca"
    if method == "tsne":
        if perplexity <= 0 or perplexity >= len(embeddings):
            raise ValueError(
                f"t-SNE perplexity must be between 0 and n_samples ({len(embeddings)})."
            )
        return (
            TSNE(
                n_components=2,
                perplexity=perplexity,
                learning_rate="auto",
                init=tsne_init,
                metric=metric,
                random_state=seed,
            ).fit_transform(embeddings),
            "tsne",
        )
    try:
        from umap import UMAP
    except ImportError as exc:
        raise ImportError(
            "UMAP is not installed. Load/install umap-learn or rerun with --method pca."
        ) from exc
    reducer = UMAP(
        n_components=2,
        n_neighbors=min(n_neighbors, max(2, len(embeddings) - 1)),
        min_dist=min_dist,
        metric="cosine" if metric == "euclidean" else metric,
        random_state=seed,
    )
    return reducer.fit_transform(embeddings), "umap"


# --------------------------------------------------------------------------- #
# rendering primitives
# --------------------------------------------------------------------------- #
def point_style(n: int) -> tuple[float, float]:
    """Marker area and alpha that keep density legible as n grows."""
    if n <= 2_000:
        return 14.0, 0.80
    if n <= 10_000:
        return 6.0, 0.45
    if n <= 40_000:
        return 3.0, 0.28
    return 1.6, 0.18


def palette(n_levels: int) -> list:
    """Distinct colours without silently wrapping the house palette."""
    if n_levels <= len(CATEGORICAL):
        return list(CATEGORICAL[:n_levels])
    cmap = plt.get_cmap("turbo")
    return [cmap(value) for value in np.linspace(0.02, 0.98, n_levels)]


def scatter_shuffled(
    ax,
    x: np.ndarray,
    y: np.ndarray,
    colors,
    *,
    size: float,
    alpha: float,
    seed: int = 0,
) -> None:
    """Draw all points in one call with randomised z-order.

    This is the fix for the buried-level problem. Occlusion still happens, but it
    is now unbiased with respect to label, so relative density in a region is
    readable rather than being an artefact of category sort order.
    """
    order = np.random.default_rng(seed).permutation(len(x))
    ax.scatter(
        np.asarray(x)[order],
        np.asarray(y)[order],
        c=np.asarray(colors, dtype=object)[order]
        if np.asarray(colors).dtype == object
        else np.asarray(colors)[order],
        s=size,
        alpha=alpha,
        linewidths=0,
        rasterized=True,
    )


def _finish_panel(ax, method: str, *, label_axes: bool = True) -> None:
    ax.set_aspect("equal", adjustable="datalim")
    if label_axes:
        ax.set_xlabel(f"{method.upper()} 1")
        ax.set_ylabel(f"{method.upper()} 2")
    clean_2d_axes(ax)


def _collapse_levels(
    values: np.ndarray, valid: np.ndarray, max_categories: int
) -> np.ndarray:
    counts = pd.Series(values[valid]).value_counts()
    if len(counts) <= max_categories:
        return values
    retained = set(counts.head(max_categories - 1).index)
    return np.asarray(
        [value if value in retained else "Other" for value in values], dtype=object
    )


def _ordered_levels(values: np.ndarray, valid: np.ndarray) -> list[str]:
    """Levels sorted by descending frequency, with Other pinned last."""
    counts = pd.Series(values[valid]).value_counts()
    levels = [str(level) for level in counts.index if str(level) != "Other"]
    if "Other" in {str(level) for level in counts.index}:
        levels.append("Other")
    return levels


# --------------------------------------------------------------------------- #
# categorical views
# --------------------------------------------------------------------------- #
def plot_categorical_points(
    frame: pd.DataFrame,
    values: np.ndarray,
    valid: np.ndarray,
    *,
    color_by: str,
    method: str,
    output_path: Path,
    seed: int,
) -> None:
    """Single-panel view for low-cardinality labels."""
    levels = _ordered_levels(values, valid)
    colors = dict(zip(levels, palette(len(levels))))
    size, alpha = point_style(int(valid.sum()))
    fig, ax = plt.subplots(figsize=(7.0, 5.4), layout="constrained")
    point_colors = [colors[str(value)] for value in values[valid]]
    scatter_shuffled(
        ax,
        frame.loc[valid, "component_1"].to_numpy(),
        frame.loc[valid, "component_2"].to_numpy(),
        point_colors,
        size=size,
        alpha=alpha,
        seed=seed,
    )
    handles = [
        plt.Line2D(
            [],
            [],
            marker="o",
            linestyle="",
            markersize=6,
            color=colors[level],
            label=f"{level} (n={int((values[valid] == level).sum()):,})",
        )
        for level in levels
    ]
    ax.legend(
        handles=handles,
        frameon=False,
        fontsize=8,
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
    )
    ax.set_title(f"Patient embeddings coloured by {color_by}")
    _finish_panel(ax, method)
    save_fig(fig, str(output_path))
    plt.close(fig)


def plot_categorical_facets(
    frame: pd.DataFrame,
    values: np.ndarray,
    valid: np.ndarray,
    *,
    color_by: str,
    method: str,
    output_path: Path,
    seed: int,
    max_facets: int = 30,
) -> None:
    """Small multiples: one highlighted level per panel over a grey backdrop.

    For 24 disease cohorts or 40 treatment arms this is the only honest view. It
    answers "does this level occupy a distinct region" per level, which is the
    question a mixed confetti plot cannot answer at all.
    """
    levels = _ordered_levels(values, valid)[:max_facets]
    colors = dict(zip(levels, palette(len(levels))))
    n_cols = min(6, max(3, math.ceil(math.sqrt(len(levels)))))
    n_rows = math.ceil(len(levels) / n_cols)
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(2.05 * n_cols, 2.15 * n_rows),
        sharex=True,
        sharey=True,
        layout="constrained",
    )
    axes = np.atleast_1d(axes).ravel()

    all_x = frame["component_1"].to_numpy()
    all_y = frame["component_2"].to_numpy()
    bg_size, bg_alpha = point_style(len(frame))
    for index, level in enumerate(levels):
        ax = axes[index]
        ax.scatter(
            all_x,
            all_y,
            s=bg_size * 0.7,
            alpha=min(0.35, bg_alpha),
            linewidths=0,
            color=GREY,
            rasterized=True,
        )
        mask = valid & (values == level)
        fg_size, fg_alpha = point_style(int(mask.sum()))
        ax.scatter(
            all_x[mask],
            all_y[mask],
            s=fg_size,
            alpha=min(0.85, fg_alpha + 0.25),
            linewidths=0,
            color=colors[level],
            rasterized=True,
        )
        ax.set_title(f"{level}\nn={int(mask.sum()):,}", fontsize=7.5)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect("equal", adjustable="box")
        for spine in ax.spines.values():
            spine.set_visible(False)
    for ax in axes[len(levels) :]:
        ax.set_visible(False)
    fig.suptitle(
        f"Patient embeddings by {color_by} ({method.upper()}, grey = full cohort)",
        fontsize=11,
    )
    save_fig(fig, str(output_path))
    plt.close(fig)


def plot_categorical_centroids(
    frame: pd.DataFrame,
    values: np.ndarray,
    valid: np.ndarray,
    *,
    color_by: str,
    method: str,
    output_path: Path,
    min_level_size: int = 25,
) -> None:
    """Centroid plus one-sigma covariance ellipse per level.

    Compresses the whole question to: do the level distributions have different
    locations at all? If every ellipse is concentric with every other, the label
    is not organising the projection.
    """
    levels = _ordered_levels(values, valid)
    colors = dict(zip(levels, palette(len(levels))))
    fig, ax = plt.subplots(figsize=(7.0, 5.4), layout="constrained")
    ax.scatter(
        frame["component_1"],
        frame["component_2"],
        s=1.2,
        alpha=0.12,
        linewidths=0,
        color=GREY,
        rasterized=True,
    )
    for level in levels:
        mask = valid & (values == level)
        if int(mask.sum()) < min_level_size:
            continue
        points = frame.loc[mask, ["component_1", "component_2"]].to_numpy()
        centre = points.mean(axis=0)
        covariance = np.cov(points, rowvar=False)
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        angle = math.degrees(math.atan2(*eigenvectors[:, 1][::-1]))
        width, height = 2.0 * np.sqrt(np.maximum(eigenvalues[::-1], 1e-12))
        ax.add_patch(
            Ellipse(
                centre,
                width,
                height,
                angle=angle,
                facecolor="none",
                edgecolor=colors[level],
                linewidth=1.3,
                alpha=0.9,
            )
        )
        ax.annotate(
            level,
            centre,
            fontsize=7,
            color=colors[level],
            ha="center",
            va="center",
            fontweight="bold",
        )
    ax.set_title(f"{color_by}: level centroids and one-sigma ellipses")
    _finish_panel(ax, method)
    save_fig(fig, str(output_path))
    plt.close(fig)


# --------------------------------------------------------------------------- #
# continuous views
# --------------------------------------------------------------------------- #
def plot_continuous(
    frame: pd.DataFrame,
    values: pd.Series,
    valid: np.ndarray,
    *,
    color_by: str,
    method: str,
    output_path: Path,
    seed: int,
    mode: str = "gradient",
    bins: int = 6,
    robust: bool = True,
) -> None:
    """Continuous view with robust limits, or binned into a discrete legend."""
    numeric = pd.to_numeric(values, errors="coerce")
    valid = valid & numeric.notna().to_numpy()
    finite = numeric[valid].to_numpy(dtype=float)
    size, alpha = point_style(int(valid.sum()))

    if mode in {"quantile", "equal"}:
        if mode == "quantile":
            edges = np.unique(np.quantile(finite, np.linspace(0, 1, bins + 1)))
        else:
            edges = np.linspace(finite.min(), finite.max(), bins + 1)
        codes = np.clip(np.digitize(finite, edges[1:-1]), 0, len(edges) - 2)
        cmap = plt.get_cmap("viridis")
        colors = [cmap(value) for value in np.linspace(0.05, 0.95, len(edges) - 1)]
        fig, ax = plt.subplots(figsize=(7.0, 5.4), layout="constrained")
        scatter_shuffled(
            ax,
            frame.loc[valid, "component_1"].to_numpy(),
            frame.loc[valid, "component_2"].to_numpy(),
            [colors[code] for code in codes],
            size=size,
            alpha=alpha,
            seed=seed,
        )
        handles = [
            plt.Line2D(
                [],
                [],
                marker="o",
                linestyle="",
                markersize=6,
                color=colors[index],
                label=(
                    f"{edges[index]:.4g} to {edges[index + 1]:.4g} "
                    f"(n={int((codes == index).sum()):,})"
                ),
            )
            for index in range(len(edges) - 1)
        ]
        ax.legend(
            handles=handles,
            frameon=False,
            fontsize=8,
            title=color_by,
            title_fontsize=8,
            bbox_to_anchor=(1.02, 1),
            loc="upper left",
        )
    else:
        low, high = (
            np.percentile(finite, [2, 98]) if robust else (finite.min(), finite.max())
        )
        if high <= low:
            low, high = finite.min(), finite.max() + 1e-9
        fig, ax = plt.subplots(figsize=(7.0, 5.4), layout="constrained")
        order = np.random.default_rng(seed).permutation(int(valid.sum()))
        points = ax.scatter(
            frame.loc[valid, "component_1"].to_numpy()[order],
            frame.loc[valid, "component_2"].to_numpy()[order],
            c=finite[order],
            cmap="viridis",
            vmin=low,
            vmax=high,
            s=size,
            alpha=alpha,
            linewidths=0,
            rasterized=True,
        )
        label = f"{color_by} (2nd to 98th pct)" if robust else color_by
        fig.colorbar(points, ax=ax, label=label)

    ax.set_title(f"Patient embeddings coloured by {color_by}")
    _finish_panel(ax, method)
    save_fig(fig, str(output_path))
    plt.close(fig)


# --------------------------------------------------------------------------- #
# dispatcher
# --------------------------------------------------------------------------- #
def plot_colored_coordinates(
    frame: pd.DataFrame,
    *,
    color_by: str,
    method: str,
    output_dir: Path,
    seed: int = 0,
    max_categories: int = 24,
    facet_threshold: int = 8,
    categorical_max_unique: int = 12,
    continuous_mode: str = "gradient",
    continuous_bins: int = 6,
    robust_clim: bool = True,
    also_centroids: bool = True,
) -> list[Path]:
    """Render every appropriate view of one metadata column.

    `categorical_max_unique` replaces the hard-coded 20 in the first version,
    which was independent of `max_categories` and would silently flip an integer
    year column between continuous and categorical depending on how many distinct
    years the cohort happened to span.
    """
    if color_by not in frame:
        raise ValueError(f"Unknown --color-by column {color_by!r}.")
    safe_name = "".join(
        character if character.isalnum() or character in "-_" else "_"
        for character in color_by
    )
    raw = frame[color_by]
    valid = raw.notna().to_numpy()
    if not valid.any():
        raise ValueError(f"Column {color_by!r} is entirely missing.")
    n_unique = raw[valid].nunique()
    is_categorical = (
        pd.api.types.is_bool_dtype(raw)
        or not pd.api.types.is_numeric_dtype(raw)
        or n_unique <= categorical_max_unique
    )
    written: list[Path] = []

    if not is_categorical:
        path = output_dir / f"patient_{method}_by_{safe_name}.png"
        plot_continuous(
            frame,
            raw,
            valid,
            color_by=color_by,
            method=method,
            output_path=path,
            seed=seed,
            mode=continuous_mode,
            bins=continuous_bins,
            robust=robust_clim,
        )
        return [path]

    values = _collapse_levels(raw.astype(str).to_numpy(), valid, max_categories)
    n_levels = len(_ordered_levels(values, valid))
    if n_levels <= facet_threshold:
        path = output_dir / f"patient_{method}_by_{safe_name}.png"
        plot_categorical_points(
            frame,
            values,
            valid,
            color_by=color_by,
            method=method,
            output_path=path,
            seed=seed,
        )
        written.append(path)
    else:
        path = output_dir / f"patient_{method}_by_{safe_name}_facets.png"
        plot_categorical_facets(
            frame,
            values,
            valid,
            color_by=color_by,
            method=method,
            output_path=path,
            seed=seed,
        )
        written.append(path)
    if also_centroids and n_levels > 2:
        path = output_dir / f"patient_{method}_by_{safe_name}_centroids.png"
        plot_categorical_centroids(
            frame,
            values,
            valid,
            color_by=color_by,
            method=method,
            output_path=path,
        )
        written.append(path)
    return written


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Project and plot an extract_patient_embeddings NPZ artifact."
    )
    parser.add_argument("--embeddings", required=True)
    parser.add_argument("--metadata", action="append", default=[])
    parser.add_argument("--subject-col", default="subject_id")
    parser.add_argument("--color-by", action="append", default=[])
    parser.add_argument("--split", default=None)
    parser.add_argument("--split-col", default="split")
    parser.add_argument("--method", choices=["umap", "tsne", "pca"], default="umap")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-neighbors", type=int, default=30)
    parser.add_argument("--min-dist", type=float, default=0.25)
    parser.add_argument("--perplexity", type=float, default=30.0)
    parser.add_argument(
        "--scaling",
        choices=["l2", "standardize", "none"],
        default="l2",
        help="Applied before projection. l2 makes euclidean distance track cosine.",
    )
    parser.add_argument(
        "--pca-dim",
        type=int,
        default=50,
        help="Pre-reduce to this many components before t-SNE/UMAP. 0 disables.",
    )
    parser.add_argument(
        "--tsne-init",
        choices=["pca", "random"],
        default="pca",
        help="init=pca seeds the macro layout from PC1/PC2; random is the control.",
    )
    parser.add_argument(
        "--max-categories",
        type=int,
        default=24,
        help="Keep the most frequent categorical levels and combine the rest.",
    )
    parser.add_argument(
        "--facet-threshold",
        type=int,
        default=8,
        help="Above this many levels, render small multiples instead of one panel.",
    )
    parser.add_argument(
        "--categorical-max-unique",
        type=int,
        default=12,
        help="Numeric columns with at most this many distinct values are categorical.",
    )
    parser.add_argument(
        "--continuous-mode",
        choices=["gradient", "quantile", "equal"],
        default="gradient",
        help="quantile/equal bin the covariate into a discrete, readable legend.",
    )
    parser.add_argument("--continuous-bins", type=int, default=6)
    parser.add_argument(
        "--no-robust-clim",
        action="store_true",
        help="Use full range instead of the 2nd to 98th percentile for colour limits.",
    )
    parser.add_argument("--no-centroids", action="store_true")
    parser.add_argument(
        "--max-points",
        type=int,
        default=None,
        help="Deterministically subsample before projection (useful for t-SNE).",
    )
    parser.add_argument(
        "--index-date-col",
        default=None,
        help="Derive treatment_year from this metadata date column.",
    )
    parser.add_argument(
        "--birth-date-col",
        default=None,
        help="Derive age_at_index using this column and --index-date-col.",
    )
    parser.add_argument(
        "--outcome",
        action="append",
        default=[],
        help=(
            "Path to an outcomes/{name}.parquet table (subject_id, index_date, "
            "outcome_date, censor_date). Adds {name}_time_days (continuous) and "
            "{name}_event columns available to --color-by; repeatable."
        ),
    )
    parser.add_argument(
        "--outcome-name",
        action="append",
        default=[],
        help=(
            "Column-name prefix for the matching --outcome entry, by position. "
            "Defaults to that file's stem when omitted; give either none or "
            "exactly one per --outcome."
        ),
    )
    parser.add_argument(
        "--outcome-horizon-days",
        type=float,
        action="append",
        default=[],
        help=(
            "Also derive a binary {name}_within_{horizon}d column (event "
            "observed within this many days) for every --outcome; repeatable, "
            "applied to all supplied outcomes."
        ),
    )
    parser.add_argument(
        "--outcome-n-hours-start-include",
        type=int,
        default=1,
        help=(
            "Minimum hours after index for an outcome event to count, matching "
            "the n_hours_start_include convention used elsewhere in this repo."
        ),
    )
    parser.add_argument(
        "--reuse-coordinates",
        default=None,
        help="Skip projection and replot from a previously written coordinates CSV.",
    )
    parser.add_argument("--output-dir", required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.reuse_coordinates is not None:
        frame = pd.read_csv(args.reuse_coordinates)
        if not {"component_1", "component_2"} <= set(frame):
            raise ValueError("Coordinates CSV must contain component_1/component_2.")
        resolved_method = args.method
    else:
        if args.outcome_name and len(args.outcome_name) != len(args.outcome):
            raise ValueError(
                "--outcome-name must be given once per --outcome, or omitted entirely."
            )
        outcome_names = args.outcome_name or [None] * len(args.outcome)
        outcome_frames = [
            load_outcome_time_columns(
                outcome_path,
                outcome_name=outcome_name,
                n_hours_start_include=args.outcome_n_hours_start_include,
                horizons_days=tuple(args.outcome_horizon_days),
            )
            for outcome_path, outcome_name in zip(args.outcome, outcome_names)
        ]
        embeddings, frame = load_embedding_frame(
            args.embeddings,
            args.metadata,
            subject_col=args.subject_col,
            outcome_frames=outcome_frames,
        )
        if args.index_date_col is not None:
            if args.index_date_col not in frame:
                raise ValueError(
                    f"Missing --index-date-col {args.index_date_col!r} in metadata."
                )
            index_dates = pd.to_datetime(frame[args.index_date_col], errors="coerce")
            frame["treatment_year"] = index_dates.dt.year.astype("Int64")
            if args.birth_date_col is not None:
                if args.birth_date_col not in frame:
                    raise ValueError(
                        f"Missing --birth-date-col {args.birth_date_col!r} in metadata."
                    )
                birth_dates = pd.to_datetime(
                    frame[args.birth_date_col], errors="coerce"
                )
                frame["age_at_index"] = (
                    index_dates - birth_dates
                ).dt.total_seconds() / (365.2425 * 24 * 60 * 60)
        if args.split is not None:
            if args.split_col not in frame:
                raise ValueError(
                    f"Cannot filter: missing split column {args.split_col!r}."
                )
            mask = frame[args.split_col].astype(str).to_numpy() == str(args.split)
            frame = frame.loc[mask].reset_index(drop=True)
            embeddings = embeddings[mask]
        if args.max_points is not None:
            if args.max_points < 3:
                raise ValueError("--max-points must be at least 3.")
            if len(frame) > args.max_points:
                selected = (
                    frame.sample(n=args.max_points, random_state=args.seed)
                    .index.sort_values()
                    .to_numpy()
                )
                frame = frame.loc[selected].reset_index(drop=True)
                embeddings = embeddings[selected]
        if len(frame) < 3:
            raise ValueError("At least three patients are required for a projection.")
        if args.max_categories < 2:
            raise ValueError("--max-categories must be at least 2.")

        prepared = prepare_embeddings(
            embeddings,
            scaling=args.scaling,
            pca_dim=args.pca_dim or None,
            seed=args.seed,
        )
        coords, resolved_method = reduce_patient_embeddings(
            prepared,
            method=args.method,
            seed=args.seed,
            n_neighbors=args.n_neighbors,
            min_dist=args.min_dist,
            perplexity=args.perplexity,
            tsne_init=args.tsne_init,
        )
        frame["component_1"] = coords[:, 0]
        frame["component_2"] = coords[:, 1]
        frame.to_csv(
            output_dir / f"patient_{resolved_method}_coordinates.csv", index=False
        )

    color_columns = args.color_by or (
        [args.split_col] if args.split_col in frame else []
    )
    written: list[Path] = []
    for column in color_columns:
        written.extend(
            plot_colored_coordinates(
                frame,
                color_by=column,
                method=resolved_method,
                output_dir=output_dir,
                seed=args.seed,
                max_categories=args.max_categories,
                facet_threshold=args.facet_threshold,
                categorical_max_unique=args.categorical_max_unique,
                continuous_mode=args.continuous_mode,
                continuous_bins=args.continuous_bins,
                robust_clim=not args.no_robust_clim,
                also_centroids=not args.no_centroids,
            )
        )
    print(
        f"Wrote {len(frame):,} {resolved_method.upper()} coordinates and "
        f"{len(written)} plot(s) to {output_dir}."
    )


if __name__ == "__main__":
    main()
