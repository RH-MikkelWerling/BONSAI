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
import fnmatch
import math
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Ellipse
from scipy.ndimage import gaussian_filter
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

from bonsai.functional.outcomes import binarize_outcomes
from opera.run.extract_patient_embeddings import read_index_table
from opera.visualization.style import (
    CATEGORICAL,
    PALETTE,
    clean_2d_axes,
    figure_title,
    save_fig,
    sequential_cmap,
    style_legend,
)

GREY = "#c8c8c8"


def _display_label(value: object, *, width: int = 30) -> str:
    """Human-readable, safely wrapped metadata label."""
    label = str(value).replace("_", " ").strip()
    return textwrap.fill(label, width=width, break_long_words=False)


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


def discover_outcome_paths(
    outcome_dir: str | Path,
    *,
    pattern: str = "*.parquet",
    exclude_patterns: tuple[str, ...] = ("*ungated*",),
) -> list[Path]:
    """Discover primary outcome tables, excluding audit/diagnostic variants."""
    root = Path(outcome_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"Outcome directory does not exist: {root}")
    paths = sorted(
        path
        for path in root.glob(pattern)
        if path.is_file()
        and not path.stem.casefold().endswith(("__audit", "_audit"))
        and not any(
            fnmatch.fnmatchcase(path.name.casefold(), item.casefold())
            for item in exclude_patterns
        )
    )
    if not paths:
        raise ValueError(
            f"No outcome tables matching {pattern!r} were found in {root}."
        )
    return paths


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
        return 6.0, 0.65
    if n <= 10_000:
        return 3.0, 0.40
    if n <= 40_000:
        return 1.5, 0.25
    return 0.8, 0.18


def palette(n_levels: int) -> list:
    """Distinct colours without silently wrapping the house palette."""
    if n_levels <= len(CATEGORICAL):
        return list(CATEGORICAL[:n_levels])
    cmap = plt.get_cmap("turbo")
    return [cmap(value) for value in np.linspace(0.02, 0.98, n_levels)]


def categorical_colors(levels: list[str]) -> dict[str, object]:
    """Assign stable semantic colours, then fill remaining categorical levels.

    Outcome-derived columns arrive as the strings ``True`` and ``False`` after
    missing follow-up has been excluded. Their colour must not depend on which
    class happens to be more prevalent in a cohort.
    """
    colors = dict(zip(levels, palette(len(levels))))
    for level in levels:
        normalized = str(level).strip().casefold()
        if normalized == "true":
            colors[level] = PALETTE["positive"]
        elif normalized == "false":
            colors[level] = PALETTE["negative"]
    return colors


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
    # Outcome booleans retain a stable legend and draw order across cohorts.
    if {level.casefold() for level in levels} <= {"true", "false"}:
        by_normalized = {level.casefold(): level for level in levels}
        levels = [
            by_normalized[name] for name in ("false", "true") if name in by_normalized
        ]
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
    colors = categorical_colors(levels)
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
            label=(
                _display_label(
                    f"{level} (n={int((values[valid] == level).sum()):,})", width=26
                )
            ),
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
    ax.set_title(
        f"Patient embeddings coloured by {_display_label(color_by, width=48)}",
        loc="left",
    )
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
    max_facets: int = 12,
) -> list[Path]:
    """Paginate all groups, keeping coordinates and category colors fixed."""
    if max_facets < 1:
        raise ValueError("max_facets must be positive.")
    levels = _ordered_levels(values, valid)
    colors = categorical_colors(levels)
    written = []
    for page, start in enumerate(range(0, len(levels), max_facets), 1):
        path = output_path if len(levels) <= max_facets else output_path.with_name(
            f"{output_path.stem}_page_{page:02d}{output_path.suffix}"
        )
        _plot_categorical_facet_page(
            frame, values, valid, color_by=color_by, method=method,
            output_path=path, seed=seed,
            levels=levels[start:start + max_facets], colors=colors,
        )
        written.append(path)
    return written


def _plot_categorical_facet_page(
    frame: pd.DataFrame,
    values: np.ndarray,
    valid: np.ndarray,
    *,
    color_by: str,
    method: str,
    output_path: Path,
    seed: int,
    levels: list[str],
    colors: dict[str, object],
) -> None:
    """Small multiples: one highlighted level per panel over a grey backdrop.

    For 24 disease cohorts or 40 treatment arms this is the only honest view. It
    answers "does this level occupy a distinct region" per level, which is the
    question a mixed confetti plot cannot answer at all.
    """
    # Four moderately wide columns remain legible in a paper PDF and leave
    # enough vertical room for two-line regimen names. Six narrow columns did
    # not: titles collided before the data could be read.
    n_cols = min(4, len(levels))
    n_rows = math.ceil(len(levels) / n_cols)
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(3.0 * n_cols, 2.65 * n_rows + 0.55),
        sharex=True,
        sharey=True,
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
        ax.set_title(
            f"{_display_label(level, width=28)}\n$n$ = {int(mask.sum()):,}",
            fontsize=8.2,
            linespacing=1.15,
            pad=7,
        )
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect("equal", adjustable="box")
        for spine in ax.spines.values():
            spine.set_visible(False)
    for ax in axes[len(levels) :]:
        ax.set_visible(False)
    figure_title(
        fig,
        f"Patient embeddings by {_display_label(color_by, width=52)}",
        f"{method.upper()} projection · coloured points show each group; grey points show the full cohort",
        y=0.995,
    )
    fig.subplots_adjust(
        top=0.90, bottom=0.035, left=0.025, right=0.985, hspace=0.36, wspace=0.08
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
    max_ellipses: int = 8,
) -> None:
    """Centroid plus one-sigma covariance ellipse per level.

    Compresses the whole question to: do the level distributions have different
    locations at all? If every ellipse is concentric with every other, the label
    is not organising the projection.
    """
    levels = _ordered_levels(values, valid)
    colors = categorical_colors(levels)
    fig, ax = plt.subplots(figsize=(9.2, 6.1))
    ax.scatter(
        frame["component_1"],
        frame["component_2"],
        s=1.2,
        alpha=0.12,
        linewidths=0,
        color=GREY,
        rasterized=True,
    )
    eligible = [
        level
        for level in levels
        if int((valid & (values == level)).sum()) >= min_level_size
    ]
    ellipse_levels = set(eligible[:max_ellipses])
    plotted: list[tuple[str, str, int]] = []
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
        if level in ellipse_levels:
            ax.add_patch(
                Ellipse(
                    centre,
                    width,
                    height,
                    angle=angle,
                    facecolor=colors[level],
                    edgecolor=colors[level],
                    linewidth=1.1,
                    alpha=0.08,
                )
            )
        # A compact numbered key is deliberate: long treatment names cannot
        # be placed inside a dense projection without hiding both one another
        # and the geometry. The key preserves identity while the map stays clean.
        number = len(plotted) + 1
        ax.scatter(
            centre[0],
            centre[1],
            s=34,
            color=colors[level],
            edgecolor="white",
            linewidth=0.8,
            zorder=4,
        )
        if len(eligible) <= 8:
            angle_offset = 2 * math.pi * (number - 1) / max(len(eligible), 1)
            offset = (18 * math.cos(angle_offset), 18 * math.sin(angle_offset))
            ax.annotate(
                str(number),
                centre,
                xytext=offset,
                textcoords="offset points",
                ha="center",
                va="center",
                fontsize=7.4,
                color=PALETTE["ink"],
                fontweight="semibold",
                zorder=5,
                bbox={
                    "boxstyle": "circle,pad=0.22",
                    "fc": "white",
                    "ec": colors[level],
                    "lw": 0.8,
                },
                arrowprops={"arrowstyle": "-", "color": colors[level], "lw": 0.7},
            )
        plotted.append((level, colors[level], int(mask.sum())))

    handles = [
        plt.Line2D(
            [],
            [],
            marker="o",
            linestyle="",
            markersize=5.5,
            markerfacecolor=color,
            markeredgecolor="white",
            label=f"{index}. {_display_label(level, width=38)}  (n={count:,})",
        )
        for index, (level, color, count) in enumerate(plotted, start=1)
    ]
    legend = ax.legend(
        handles=handles,
        title="Groups",
        loc="center left",
        bbox_to_anchor=(1.01, 0.5),
        frameon=False,
        fontsize=8,
        labelspacing=0.72,
        handletextpad=0.45,
    )
    style_legend(legend)
    ax.set_title(
        f"Group location and spread · {_display_label(color_by, width=52)}\n"
        f"Centroids for all groups; one-SD ellipses for the {len(ellipse_levels)} largest",
        loc="left",
        pad=12,
    )
    _finish_panel(ax, method)
    fig.subplots_adjust(left=0.05, right=0.70, top=0.90, bottom=0.08)
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
        cmap = sequential_cmap("opera")
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
            title=_display_label(color_by, width=32),
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
            cmap=sequential_cmap("opera"),
            vmin=low,
            vmax=high,
            s=size,
            alpha=alpha,
            linewidths=0,
            rasterized=True,
        )
        label = f"{color_by} (2nd to 98th pct)" if robust else color_by
        fig.colorbar(
            points,
            ax=ax,
            label=label,
            orientation="horizontal",
            shrink=0.65,
            pad=0.06,
            aspect=35,
        )

    ax.set_title(
        f"Patient embeddings coloured by {_display_label(color_by, width=48)}",
        loc="left",
    )
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
    max_categories: int = 60,
    facet_threshold: int = 6,
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
        written.extend(plot_categorical_facets(
            frame,
            values,
            valid,
            color_by=color_by,
            method=method,
            output_path=path,
            seed=seed,
        ))
    # Overlapping high-cardinality centroids and long keys add little beyond
    # the facets. Retain this diagnostic only for a legible number of groups.
    if also_centroids and 2 < n_levels <= 8:
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


def _kernel_probability_surface(
    x: np.ndarray,
    y: np.ndarray,
    labels: np.ndarray,
    *,
    bins: int = 450,
    bandwidth_bins: float = 1.25,
    min_support: float = 0.08,
    x_range: tuple[float, float] | None = None,
    y_range: tuple[float, float] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Estimate P(Y=1 | 2-D location) as a ratio of smoothed densities.

    The numerator and denominator use the same Gaussian kernel. ``min_support``
    is relative to the densest grid cell and masks extrapolation into empty
    parts of a projection. ``bandwidth_bins`` is measured on the original
    90-cell reference grid, so changing resolution does not change the
    smoothing length in projection coordinates.
    """
    if bins < 20 or bandwidth_bins <= 0 or not 0 <= min_support < 1:
        raise ValueError("Invalid KDE grid, bandwidth, or support threshold.")
    x_min, x_max = x_range if x_range is not None else (float(x.min()), float(x.max()))
    y_min, y_max = y_range if y_range is not None else (float(y.min()), float(y.max()))
    padding_x = max((x_max - x_min) * 0.025, 1e-6)
    padding_y = max((y_max - y_min) * 0.025, 1e-6)
    x_edges = np.linspace(x_min - padding_x, x_max + padding_x, bins + 1)
    y_edges = np.linspace(y_min - padding_y, y_max + padding_y, bins + 1)
    total, _, _ = np.histogram2d(x, y, bins=(x_edges, y_edges))
    positive, _, _ = np.histogram2d(
        x, y, bins=(x_edges, y_edges), weights=labels.astype(float)
    )
    sigma = bandwidth_bins * bins / 90.0
    total_density = gaussian_filter(total, sigma, mode="constant")
    positive_density = gaussian_filter(positive, sigma, mode="constant")
    probability = np.divide(
        positive_density,
        total_density,
        out=np.full_like(positive_density, np.nan),
        where=total_density > 0,
    )
    if total_density.max() > 0:
        probability[total_density < min_support * total_density.max()] = np.nan
    return x_edges, y_edges, probability


def plot_outcome_probability_facets(
    frame: pd.DataFrame,
    columns: list[str],
    *,
    method: str,
    output_dir: Path,
    horizon_days: float,
    min_class_size: int = 20,
    facets_per_page: int = 12,
    bins: int = 450,
    bandwidth_bins: float = 1.25,
    min_support: float = 0.08,
    colormap: str = "plasma",
) -> list[Path]:
    """Paginated KDE-smoothed outcome probability maps on fixed coordinates."""
    eligible: list[tuple[str, np.ndarray, np.ndarray]] = []
    for column in columns:
        numeric = pd.to_numeric(frame[column], errors="coerce")
        known = numeric.notna().to_numpy()
        # Horizon columns are pandas nullable booleans because early-censored
        # patients are genuinely unknown. Older pandas versions reject
        # ``BooleanArray.fillna(0)`` before ``to_numpy(dtype=float)`` gets a
        # chance to coerce it, so convert to nullable floating point first.
        labels = numeric.astype("Float64").fillna(0.0).to_numpy(dtype=float)
        positives = int((known & (labels == 1)).sum())
        negatives = int((known & (labels == 0)).sum())
        if positives >= min_class_size and negatives >= min_class_size:
            eligible.append((column, known, labels))
    if not eligible:
        return []

    output_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    x_all = frame["component_1"].to_numpy(dtype=float)
    y_all = frame["component_2"].to_numpy(dtype=float)
    x_range = (float(np.nanmin(x_all)), float(np.nanmax(x_all)))
    y_range = (float(np.nanmin(y_all)), float(np.nanmax(y_all)))
    for page, start in enumerate(range(0, len(eligible), facets_per_page), 1):
        page_items = eligible[start : start + facets_per_page]
        n_cols = min(4, len(page_items))
        n_rows = math.ceil(len(page_items) / n_cols)
        fig, axes = plt.subplots(
            n_rows,
            n_cols,
            figsize=(3.15 * n_cols, 3.05 * n_rows + 0.95),
            sharex=True,
            sharey=True,
            squeeze=False,
        )
        mesh = None
        for ax, (column, known, labels) in zip(axes.ravel(), page_items):
            x_edges, y_edges, probability = _kernel_probability_surface(
                x_all[known],
                y_all[known],
                labels[known],
                bins=bins,
                bandwidth_bins=bandwidth_bins,
                min_support=min_support,
                x_range=x_range,
                y_range=y_range,
            )
            mesh = ax.imshow(
                probability.T,
                origin="lower",
                extent=(x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]),
                cmap=plt.get_cmap(colormap).with_extremes(bad="white"),
                vmin=0,
                vmax=1,
                interpolation="nearest",
                rasterized=True,
            )
            n_positive = int((labels[known] == 1).sum())
            title = column.removesuffix(f"_within_{horizon_days:g}d")
            ax.set_title(
                f"{_display_label(title, width=27)}\n"
                f"{n_positive:,} / {int(known.sum()):,} events",
                fontsize=8.2,
                pad=6,
            )
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_aspect("equal", adjustable="box")
            for spine in ax.spines.values():
                spine.set_visible(False)
        for ax in axes.ravel()[len(page_items) :]:
            ax.set_visible(False)
        if mesh is not None:
            colorbar_ax = fig.add_axes([0.18, 0.045, 0.64, 0.018])
            cbar = fig.colorbar(
                mesh,
                cax=colorbar_ax,
                orientation="horizontal",
            )
            cbar.set_label(
                f"Smoothed observed event fraction by {horizon_days:g} days"
            )
        figure_title(
            fig,
            f"Outcome landscape at {horizon_days:g} days",
            "Known horizon status only; low-density regions masked. Descriptive, not calibrated risk.",
            y=0.995,
        )
        fig.subplots_adjust(
            top=0.85 if n_rows == 1 else 0.90,
            bottom=0.17 if n_rows == 1 else 0.115,
            left=0.035, right=0.985, hspace=0.38, wspace=0.12,
        )
        path = output_dir / f"patient_{method}_outcome_kde_{horizon_days:g}d_page_{page:02d}.png"
        save_fig(fig, str(path))
        plt.close(fig)
        written.append(path)
    return written


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def write_metadata_atlases(frame, columns, *, method, output_dir, max_categories=12):
    """Export coordinated overview pages from the same saved patient projection."""
    from opera.visualization.embedding_plots import plot_embedding_stage_metadata_grid

    selected = [
        column for column in columns if column in frame and frame[column].notna().any()
    ]
    written = []
    for page, start in enumerate(range(0, len(selected), 4), 1):
        metadata = {}
        kinds = {}
        for column in selected[start : start + 4]:
            series = frame[column]
            numeric = pd.to_numeric(series, errors="coerce")
            if numeric.notna().sum() == series.notna().sum() and numeric.nunique() > 12:
                metadata[column] = numeric.to_numpy(dtype=float, na_value=np.nan)
                kinds[column] = "continuous"
            else:
                values = series.astype(object).where(series.notna(), np.nan).to_numpy()
                valid = series.notna().to_numpy()
                metadata[column] = _collapse_levels(
                    values, valid, max_categories
                ).copy()
                metadata[column][~valid] = np.nan
                kinds[column] = "categorical"
        path = Path(output_dir) / f"patient_{method}_atlas_{page:02d}.png"
        fig = plot_embedding_stage_metadata_grid(
            {"patients": frame[["component_1", "component_2"]].to_numpy()},
            metadata,
            variable_kinds=kinds,
            stage_labels={"patients": "Patients"},
            variable_labels={
                column: _display_label(column, width=24) for column in metadata
            },
            axis_label=method.upper(),
            title="Patient representation atlas",
            save_path=str(path),
        )
        plt.close(fig)
        written.append(path)
    return written


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Project and plot an extract_patient_embeddings NPZ artifact."
    )
    parser.add_argument("--embeddings", required=True)
    parser.add_argument("--metadata", action="append", default=[])
    parser.add_argument("--subject-col", default="subject_id")
    parser.add_argument("--color-by", action="append", default=[])
    parser.add_argument(
        "--color-all-metadata",
        action="store_true",
        help=(
            "Plot every non-constant joined covariate, excluding identifiers, "
            "raw date columns, and projection coordinates."
        ),
    )
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
        default=60,
        help="Keep the most frequent categorical levels and combine the rest.",
    )
    parser.add_argument(
        "--facet-threshold",
        type=int,
        default=6,
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
        "--outcome-dir",
        default=None,
        help=(
            "Load every real outcome table in this directory. Audit files are "
            "excluded automatically; use --outcome-pattern to restrict discovery."
        ),
    )
    parser.add_argument("--outcome-pattern", default="*.parquet")
    parser.add_argument(
        "--outcome-exclude-pattern",
        action="append",
        default=None,
        help=(
            "Case-insensitive filename pattern excluded from --outcome-dir; "
            "repeatable. Defaults to '*ungated*'. Audit files are always excluded."
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
        "--outcome-kde-horizon-days",
        type=float,
        action="append",
        default=[],
        help=(
            "Write paginated, censoring-aware KDE probability maps for all "
            "loaded outcomes at this horizon; repeatable."
        ),
    )
    parser.add_argument("--outcome-kde-min-class-size", type=int, default=20)
    parser.add_argument("--outcome-kde-facets-per-page", type=int, default=12)
    parser.add_argument("--outcome-kde-grid-bins", type=int, default=450)
    parser.add_argument("--outcome-kde-colormap", choices=("plasma", "inferno", "viridis", "magma"), default="plasma")
    parser.add_argument(
        "--outcome-kde-bandwidth-bins", type=float, default=1.25,
        help="Gaussian smoothing width in cells of the original 90-cell reference grid; independent of rendering resolution.",
    )
    parser.add_argument("--outcome-kde-min-support", type=float, default=0.08)
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
    explicit_outcomes = list(args.outcome)
    if args.outcome_name and len(args.outcome_name) != len(explicit_outcomes):
        raise ValueError(
            "--outcome-name must be given once per --outcome, or omitted entirely."
        )
    discovered_outcomes = (
        discover_outcome_paths(
            args.outcome_dir,
            pattern=args.outcome_pattern,
            exclude_patterns=tuple(args.outcome_exclude_pattern or ["*ungated*"]),
        )
        if args.outcome_dir is not None
        else []
    )
    duplicate_paths = {
        str(Path(path).resolve()) for path in explicit_outcomes
    } & {str(path.resolve()) for path in discovered_outcomes}
    if duplicate_paths:
        raise ValueError(
            "Outcome tables were supplied both explicitly and via --outcome-dir: "
            f"{sorted(duplicate_paths)}"
        )
    outcome_paths: list[str | Path] = [*explicit_outcomes, *discovered_outcomes]
    outcome_names = [
        *(args.outcome_name or [None] * len(explicit_outcomes)),
        *([None] * len(discovered_outcomes)),
    ]
    requested_horizons = tuple(
        dict.fromkeys([*args.outcome_horizon_days, *args.outcome_kde_horizon_days])
    )
    outcome_frames = [
        load_outcome_time_columns(
            outcome_path,
            outcome_name=outcome_name,
            n_hours_start_include=args.outcome_n_hours_start_include,
            horizons_days=requested_horizons,
        )
        for outcome_path, outcome_name in zip(outcome_paths, outcome_names)
    ]

    if args.reuse_coordinates is not None:
        frame = pd.read_csv(args.reuse_coordinates)
        if not {"component_1", "component_2"} <= set(frame):
            raise ValueError("Coordinates CSV must contain component_1/component_2.")
        # Coordinate artifacts already contain metadata used during their first
        # rendering. Add only genuinely new columns so callers can enrich and
        # recolor an existing expensive UMAP without duplicate-column failures.
        frame["_join_id"] = frame["subject_id"].astype(str)
        additions = [read_index_table(path) for path in args.metadata] + outcome_frames
        for addition in additions:
            source_subject_col = (
                args.subject_col if args.subject_col in addition else "subject_id"
            )
            keep = [source_subject_col] + [
                column
                for column in addition.columns
                if column != source_subject_col and column not in frame.columns
            ]
            if len(keep) > 1:
                frame = _left_join_on_subject(
                    frame,
                    addition[keep],
                    subject_col=source_subject_col,
                    source_label="coordinate recoloring metadata",
                )
        frame = frame.drop(columns="_join_id", errors="ignore")
        resolved_method = args.method
    else:
        embeddings, frame = load_embedding_frame(
            args.embeddings,
            args.metadata,
            subject_col=args.subject_col,
            outcome_frames=outcome_frames,
        )
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
            birth_dates = pd.to_datetime(frame[args.birth_date_col], errors="coerce")
            frame["age_at_index"] = (index_dates - birth_dates).dt.total_seconds() / (
                365.2425 * 24 * 60 * 60
            )

    if args.color_all_metadata:
        excluded = {
            "subject_id",
            args.subject_col,
            "component_1",
            "component_2",
        }
        color_columns = [
            column
            for column in frame.columns
            if column not in excluded
            and "date" not in column.lower()
            and frame[column].notna().any()
            and frame[column].nunique(dropna=True) > 1
        ]
        # Explicit requests are retained even if their names resemble dates.
        color_columns = list(dict.fromkeys([*color_columns, *args.color_by]))
    else:
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
    written.extend(
        write_metadata_atlases(
            frame,
            color_columns,
            method=resolved_method,
            output_dir=output_dir,
            max_categories=args.max_categories,
        )
    )
    for horizon_days in args.outcome_kde_horizon_days:
        suffix = f"_within_{horizon_days:g}d"
        outcome_columns = [
            column for column in frame.columns if column.endswith(suffix)
        ]
        written.extend(
            plot_outcome_probability_facets(
                frame,
                outcome_columns,
                method=resolved_method,
                output_dir=output_dir,
                horizon_days=horizon_days,
                min_class_size=args.outcome_kde_min_class_size,
                facets_per_page=args.outcome_kde_facets_per_page,
                bins=args.outcome_kde_grid_bins,
                colormap=args.outcome_kde_colormap,
                bandwidth_bins=args.outcome_kde_bandwidth_bins,
                min_support=args.outcome_kde_min_support,
            )
        )
    print(
        f"Wrote {len(frame):,} {resolved_method.upper()} coordinates and "
        f"{len(written)} plot(s) to {output_dir}."
    )


if __name__ == "__main__":
    main()
