"""
OPERA Embedding & Training Visualizations.

Includes:
  1. UMAP / t-SNE embedding projections (with density contours)
  2. Survival time gradient coloring in embedding space
  3. Kendall sigma evolution and final bar
  4. Cosine similarity distributions (KDE)
  5. Training loss curves
  6. Source-vs-outcome diagnostic plots
"""

from __future__ import annotations
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.ticker as mticker
from matplotlib.lines import Line2D

from opera.visualization.style import (
    PALETTE, CATEGORICAL, save_fig, despine,
    FIG_FULL, FIG_SQUARE, FIG_HALF, FIG_TALL, add_panel_label,
    ANNOT_SIZE, LEGEND_SIZE,
)

SOURCE_PALETTE = {
    "LPR":    "#0072B2",
    "RKKP":   "#D55E00",
    "LAB":    "#009E73",
    "PATHX":  "#CC79A7",
    "FLOW":   "#E69F00",
    "multi_source": "#999999",
    "unknown": "#DDDDDD",
}


def _reduce_embeddings(
    embeddings: np.ndarray,
    method: str = "umap",
    **kwargs,
) -> np.ndarray:
    if method == "umap":
        from umap import UMAP
        defaults = {"n_neighbors": 30, "min_dist": 0.25,
                    "metric": "cosine", "random_state": 42}
        defaults.update(kwargs)
        return UMAP(n_components=2, **defaults).fit_transform(embeddings)
    else:
        from sklearn.manifold import TSNE
        defaults = {"perplexity": 30, "random_state": 42,
                    "learning_rate": "auto", "init": "pca"}
        defaults.update(kwargs)
        return TSNE(n_components=2, **defaults).fit_transform(embeddings)


def _density_contour(ax, x, y, color, levels=5, alpha=0.35):
    """Overlay KDE density contours for a point cloud."""
    try:
        from scipy.stats import gaussian_kde
        xy = np.vstack([x, y])
        kde = gaussian_kde(xy, bw_method=0.25)
        xg = np.linspace(x.min(), x.max(), 120)
        yg = np.linspace(y.min(), y.max(), 120)
        Xg, Yg = np.meshgrid(xg, yg)
        Z = kde(np.vstack([Xg.ravel(), Yg.ravel()])).reshape(Xg.shape)
        ax.contour(Xg, Yg, Z, levels=levels, colors=[color],
                   linewidths=0.7, alpha=alpha)
    except Exception:
        pass  # scipy not available or too few points


# ═════════════════════════════════════════════════════════════════════
# 1. UMAP colored by binary label
# ═════════════════════════════════════════════════════════════════════

def plot_embedding_projection(
    embeddings: np.ndarray,
    labels: np.ndarray,
    method: str = "umap",
    title: Optional[str] = None,
    label_names: Tuple[str, str] = ("Negative", "Positive"),
    density_contours: bool = True,
    save_path: Optional[str] = None,
    **reducer_kwargs,
) -> plt.Figure:
    """2D scatter of embeddings colored by binary label, with KDE contours."""
    coords = _reduce_embeddings(embeddings, method, **reducer_kwargs)

    fig, ax = plt.subplots(figsize=(4.5, 4.2))

    for label_val, color, name in [
        (0, PALETTE["negative"], label_names[0]),
        (1, PALETTE["positive"], label_names[1]),
    ]:
        mask = labels == label_val
        if mask.sum() == 0:
            continue
        ax.scatter(coords[mask, 0], coords[mask, 1],
                   c=color, s=5, alpha=0.35, label=f"{name} (n={mask.sum()})",
                   rasterized=True, linewidths=0, zorder=2)
        if density_contours and mask.sum() > 50:
            _density_contour(ax, coords[mask, 0], coords[mask, 1], color)

    ax.set_xlabel(f"{method.upper()} 1")
    ax.set_ylabel(f"{method.upper()} 2")
    ax.set_title(title or f"Embedding space  ({method.upper()})")
    ax.legend(markerscale=3, framealpha=0.9, fontsize=LEGEND_SIZE)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.grid(False)

    fig.tight_layout()
    save_fig(fig, save_path)
    return fig


# ═════════════════════════════════════════════════════════════════════
# 2. UMAP colored by continuous survival time
# ═════════════════════════════════════════════════════════════════════

def plot_embedding_survival_gradient(
    embeddings: np.ndarray,
    times: np.ndarray,
    events: np.ndarray,
    method: str = "umap",
    outcome_name: str = "outcome",
    title: Optional[str] = None,
    save_path: Optional[str] = None,
    **reducer_kwargs,
) -> plt.Figure:
    """
    2D scatter colored by time-to-event, with censored patients shown as
    open circles overlaid.

    This reveals where the model is certain (dense clusters of early/late
    events) vs where clinical ambiguity exists (mixed time regions).
    """
    valid = np.isfinite(times) & (events >= 0)
    coords = _reduce_embeddings(embeddings, method, **reducer_kwargs)

    fig, ax = plt.subplots(figsize=(5.2, 4.5))

    # Background: all patients colored by time
    t_plot = np.where(valid, times, np.nan)
    sc = ax.scatter(
        coords[:, 0], coords[:, 1],
        c=t_plot, cmap="plasma_r",
        s=5, alpha=0.55, rasterized=True,
        linewidths=0, zorder=2,
        vmin=np.nanpercentile(times[valid], 5) if valid.any() else 0,
        vmax=np.nanpercentile(times[valid], 95) if valid.any() else 1,
    )

    # Admin-censored patients: small hollow grey markers
    cens_mask = valid & (events == 0)
    if cens_mask.sum() > 0:
        ax.scatter(coords[cens_mask, 0], coords[cens_mask, 1],
                   s=7, facecolors="none", edgecolors="#888888",
                   linewidths=0.4, alpha=0.4, zorder=3,
                   label=f"Admin censored (n={cens_mask.sum()})", rasterized=True)

    # Competing-death patients: small hollow red markers
    comp_mask = valid & (events == 2)
    if comp_mask.sum() > 0:
        ax.scatter(coords[comp_mask, 0], coords[comp_mask, 1],
                   s=7, facecolors="none", edgecolors="#cc4444",
                   linewidths=0.4, alpha=0.4, zorder=3,
                   label=f"Competing death (n={comp_mask.sum()})", rasterized=True)

    cbar = fig.colorbar(sc, ax=ax, shrink=0.85, pad=0.02)
    cbar.set_label(f"Time to {outcome_name.replace('_', ' ')} (days)", fontsize=8)
    cbar.ax.tick_params(labelsize=7)

    ax.set_xlabel(f"{method.upper()} 1")
    ax.set_ylabel(f"{method.upper()} 2")
    ax.set_title(title or f"Embedding — {outcome_name.replace('_', ' ').title()}")
    if cens_mask.sum() > 0 or comp_mask.sum() > 0:
        ax.legend(markerscale=2, fontsize=LEGEND_SIZE, loc="lower right")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.grid(False)

    fig.tight_layout()
    save_fig(fig, save_path)
    return fig


# ═════════════════════════════════════════════════════════════════════
# 3. Multi-outcome embedding panel
# ═════════════════════════════════════════════════════════════════════

def plot_embedding_multi_outcome(
    embeddings: np.ndarray,
    outcome_labels: Dict[str, np.ndarray],
    method: str = "umap",
    save_path: Optional[str] = None,
    **reducer_kwargs,
) -> plt.Figure:
    """
    Grid of UMAPs — one per outcome — using a single shared projection.
    Shared coordinates make comparisons across panels interpretable.
    """
    coords = _reduce_embeddings(embeddings, method, **reducer_kwargs)

    names = sorted(outcome_labels.keys())
    n = len(names)
    cols = min(n, 3)
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(3.8 * cols, 3.5 * rows))
    if n == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    for i, name in enumerate(names):
        ax = axes[i]
        labs = outcome_labels[name]
        valid = labs >= 0

        ax.scatter(coords[~valid, 0], coords[~valid, 1],
                   c=PALETTE["missing"], s=3, alpha=0.12, rasterized=True, linewidths=0)

        for label_val, color in [(0, PALETTE["negative"]), (1, PALETTE["positive"])]:
            mask = valid & (labs == label_val)
            lbl = "Positive" if label_val else "Negative"
            ax.scatter(coords[mask, 0], coords[mask, 1],
                       c=color, s=5, alpha=0.4, label=lbl,
                       rasterized=True, linewidths=0, zorder=2)
            if mask.sum() > 50:
                _density_contour(ax, coords[mask, 0], coords[mask, 1], color, levels=4)

        ax.set_title(name.replace("_", " ").title(), fontsize=9, fontweight="semibold")
        ax.legend(markerscale=2, fontsize=6.5, loc="lower right")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.spines["left"].set_visible(False)
        ax.spines["bottom"].set_visible(False)
        ax.grid(False)

    for j in range(n, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(f"Embedding space by outcome  ({method.upper()})",
                 fontsize=10, fontweight="semibold", y=1.01)
    fig.tight_layout()
    save_fig(fig, save_path)
    return fig


# ═════════════════════════════════════════════════════════════════════
# 4. Sigma (learned uncertainty) plots
# ═════════════════════════════════════════════════════════════════════

def plot_sigma_barplot(
    sigma_values: Dict[str, float],
    title: str = "Learned outcome uncertainty  (σ)",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Horizontal lollipop plot of final σ values.

    Lower σ → outcome strongly structures the embedding (more important).
    Higher σ → outcome is harder to learn / noisier signal.
    """
    names = list(sigma_values.keys())
    vals  = [sigma_values[n] for n in names]
    display = [n.replace("_", " ").title() for n in names]

    order = np.argsort(vals)
    names_s = [display[i] for i in order]
    vals_s  = [vals[i] for i in order]
    colors  = [PALETTE["opera"] if v < 1.0 else PALETTE["positive"] for v in vals_s]

    fig, ax = plt.subplots(figsize=(max(4.5, 2.0 * len(names)), 3.0))

    y = np.arange(len(names_s))
    ax.hlines(y, 0, vals_s, color="#CCCCCC", linewidth=1.2, zorder=1)
    ax.scatter(vals_s, y, color=colors, s=60, zorder=3)
    ax.axvline(1.0, color=PALETTE["diagonal"], ls=":", lw=1,
               label="σ = 1  (initialisation)")

    for i, (v, name) in enumerate(zip(vals_s, names_s)):
        ax.text(v + 0.02, i, f"{v:.3f}", va="center", fontsize=ANNOT_SIZE)

    ax.set_yticks(y)
    ax.set_yticklabels(names_s)
    ax.set_xlabel("σ  (lower = outcome more strongly structures embedding)")
    ax.set_title(title)
    ax.legend(fontsize=LEGEND_SIZE)
    despine(ax, "none")
    ax.grid(axis="x", color=PALETTE["grid"], linewidth=0.5)
    ax.grid(axis="y", visible=False)

    fig.tight_layout()
    save_fig(fig, save_path)
    return fig


def plot_sigma_evolution(
    training_log: pd.DataFrame,
    outcome_names: List[str],
    title: str = "Outcome uncertainty  (σ) during training",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """σ and precision evolution per outcome during contrastive training."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=FIG_FULL)
    x_col = "epoch" if "epoch" in training_log.columns else "step"

    for name, color in zip(outcome_names, CATEGORICAL):
        display = name.replace("_", " ").title()
        col_s = f"train/sigma/{name}"
        col_p = f"train/precision/{name}"

        if col_s in training_log.columns:
            data = training_log[[x_col, col_s]].dropna()
            ax1.plot(data[x_col], data[col_s], color=color, lw=1.6, label=display)
        if col_p in training_log.columns:
            data = training_log[[x_col, col_p]].dropna()
            ax2.plot(data[x_col], data[col_p], color=color, lw=1.6, label=display)

    ax1.axhline(1.0, color=PALETTE["diagonal"], ls=":", lw=1)
    ax1.set_xlabel(x_col.title())
    ax1.set_ylabel("σ  (learned uncertainty)")
    ax1.set_title("Outcome uncertainty")
    ax1.legend(fontsize=LEGEND_SIZE)
    despine(ax1, "y")

    ax2.set_xlabel(x_col.title())
    ax2.set_ylabel("Precision  (1 / 2σ²)")
    ax2.set_title("Effective loss weight")
    ax2.legend(fontsize=LEGEND_SIZE)
    despine(ax2, "y")

    add_panel_label(ax1, "A")
    add_panel_label(ax2, "B")
    fig.suptitle(title, fontsize=10, fontweight="semibold", y=1.01)
    fig.tight_layout()
    save_fig(fig, save_path)
    return fig


# ═════════════════════════════════════════════════════════════════════
# 5. Cosine similarity distributions (KDE)
# ═════════════════════════════════════════════════════════════════════

def plot_similarity_distributions(
    embeddings: np.ndarray,
    labels: np.ndarray,
    n_pairs: int = 20000,
    seed: int = 42,
    title: str = "Cosine similarity distributions",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """KDE of cosine similarities for same-class vs different-class pairs."""
    try:
        from scipy.stats import gaussian_kde
    except ImportError:
        gaussian_kde = None

    rng = np.random.RandomState(seed)
    N = len(embeddings)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    emb_n = embeddings / (norms + 1e-12)

    pos_sims, neg_sims = [], []
    pairs = rng.randint(0, N, size=(n_pairs, 2))
    for i, j in pairs:
        if i == j:
            continue
        sim = float(emb_n[i] @ emb_n[j])
        if labels[i] == labels[j]:
            pos_sims.append(sim)
        else:
            neg_sims.append(sim)

    fig, ax = plt.subplots(figsize=(4.5, 3.2))

    x_grid = np.linspace(-1, 1, 300)
    for sims, color, name in [
        (neg_sims, PALETTE["negative"], "Different outcome"),
        (pos_sims, PALETTE["positive"], "Same outcome"),
    ]:
        if not sims:
            continue
        arr = np.array(sims)
        if gaussian_kde is not None and len(arr) > 10:
            kde = gaussian_kde(arr, bw_method=0.15)
            density = kde(x_grid)
            ax.plot(x_grid, density, color=color, lw=1.8, label=name)
            ax.fill_between(x_grid, density, alpha=0.15, color=color)
        else:
            ax.hist(arr, bins=60, density=True, alpha=0.5, color=color, label=name)

    ax.set_xlabel("Cosine similarity")
    ax.set_ylabel("Density")
    ax.set_title(title)
    ax.set_xlim(-1, 1)
    ax.legend(fontsize=LEGEND_SIZE)
    despine(ax, "y")

    fig.tight_layout()
    save_fig(fig, save_path)
    return fig


# ═════════════════════════════════════════════════════════════════════
# 6. Training loss curves
# ═════════════════════════════════════════════════════════════════════

def plot_training_curves(
    training_log: pd.DataFrame,
    metrics: Optional[List[str]] = None,
    smooth_window: int = 5,
    title: str = "Training curves",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Train / val loss and metric curves, with optional EMA smoothing."""
    if metrics is None:
        all_cols = [c for c in training_log.columns
                    if c.startswith("train/") or c.startswith("val/")]
        bases = sorted({c.replace("train/", "").replace("val/", "") for c in all_cols})
        # Keep only columns with both train and val
        metrics = [b for b in bases
                   if f"train/{b}" in training_log.columns
                   and f"val/{b}" in training_log.columns]

    if not metrics:
        return plt.figure()

    n = len(metrics)
    cols = min(n, 3)
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(3.8 * cols, 3.0 * rows))
    if n == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    x_col = "epoch" if "epoch" in training_log.columns else "step"

    def _smooth(series, w):
        if w <= 1:
            return series
        return series.rolling(w, min_periods=1, center=True).mean()

    for i, base in enumerate(metrics):
        ax = axes[i]
        for prefix, color, label in [
            ("train", PALETTE["opera"], "Train"),
            ("val",   PALETTE["positive"], "Val"),
        ]:
            col = f"{prefix}/{base}"
            if col in training_log.columns:
                data = training_log[[x_col, col]].dropna().reset_index(drop=True)
                raw  = data[col]
                smth = _smooth(raw, smooth_window)
                ax.plot(data[x_col], raw, color=color, alpha=0.3, lw=0.8)
                ax.plot(data[x_col], smth, color=color, lw=1.6, label=label)

        ax.set_xlabel(x_col.title())
        ax.set_ylabel(base.replace("_", " ").title())
        ax.set_title(base.replace("_", " ").title())
        ax.legend(fontsize=LEGEND_SIZE)
        despine(ax, "y")

    for j in range(n, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(title, fontsize=10, fontweight="semibold", y=1.01)
    fig.tight_layout()
    save_fig(fig, save_path)
    return fig


# ═════════════════════════════════════════════════════════════════════
# 7. Source-aware diagnostic plots
# ═════════════════════════════════════════════════════════════════════

def plot_embedding_by_source(
    embeddings: np.ndarray,
    source_labels: np.ndarray,
    method: str = "umap",
    title: Optional[str] = None,
    save_path: Optional[str] = None,
    **reducer_kwargs,
) -> plt.Figure:
    coords = _reduce_embeddings(embeddings, method, **reducer_kwargs)
    unique_sources = sorted(set(source_labels))

    fig, ax = plt.subplots(figsize=(4.8, 4.5))
    for source in unique_sources:
        mask = source_labels == source
        color = SOURCE_PALETTE.get(source, "#999999")
        ax.scatter(coords[mask, 0], coords[mask, 1],
                   c=color, s=5, alpha=0.35,
                   label=f"{source}  (n={mask.sum()})",
                   rasterized=True, linewidths=0)

    ax.set_title(title or f"Embedding by data source  ({method.upper()})")
    ax.legend(markerscale=3, fontsize=LEGEND_SIZE, loc="best")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.grid(False)

    fig.tight_layout()
    save_fig(fig, save_path)
    return fig


# ═════════════════════════════════════════════════════════════════════
# 8. Clinical embedding story — three-figure series
#
#  Call order in practice:
#     coords = reduce_embeddings(embeddings)   # compute UMAP once
#     plot_embedding_disease_map(coords, ...)
#     plot_clinical_variables_panel(coords, ...)
#     plot_embedding_boundary_patients(coords, ...)
# ═════════════════════════════════════════════════════════════════════

def reduce_embeddings(
    embeddings: np.ndarray,
    method: str = "umap",
    **kwargs,
) -> np.ndarray:
    """
    Public wrapper around _reduce_embeddings.

    Call this once and pass the resulting (n, 2) array to the clinical
    plotting functions below — UMAP is expensive and all three figures
    should share the same projection.
    """
    return _reduce_embeddings(embeddings, method, **kwargs)


def _cohort_colors(unique_groups) -> Dict:
    """
    Assign a stable, perceptually distinct color to each cohort/disease.

    For ≤8 groups: uses the jewel-tone CATEGORICAL palette (maximally
    distinct, warm/cool alternating).
    For 9–20 groups: interpolates through a wider HLS wheel anchored at
    the same jewel-tone hues, staying away from pure grey.
    For >20 groups: falls back to matplotlib's tab20, which is designed
    for this regime.

    In all cases the mapping is deterministic (sorted group order), so the
    same color is assigned to the same group across figures.
    """
    import matplotlib.colors as mcolors
    groups = sorted(unique_groups)
    n = len(groups)

    if n <= len(CATEGORICAL):
        palette = CATEGORICAL[:n]
    elif n <= 20:
        # Sample a wider HLS wheel: 8 anchor hues + intermediate steps
        import colorsys
        hues = np.linspace(0, 1, n, endpoint=False)
        # Shift to start near indigo (hue ≈ 0.65)
        hues = (hues + 0.65) % 1.0
        palette = [
            mcolors.to_hex(colorsys.hls_to_rgb(h, 0.38, 0.70))
            for h in hues
        ]
    else:
        cmap = plt.cm.get_cmap("tab20", n)
        palette = [mcolors.to_hex(cmap(i)) for i in range(n)]

    return {g: palette[i] for i, g in enumerate(groups)}


def remap_disease_groups(
    group_labels: np.ndarray,
    merge: Optional[Dict[str, List]] = None,
    drop: Optional[List] = None,
    min_n: int = 30,
    other_label: str = "Other",
    unknown_values: Optional[List] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Consolidate fine-grained disease labels into a plottable set.

    Parameters
    ----------
    group_labels : raw disease labels (strings or ints).
    merge : mapping of new_name → [list of original labels to merge].
        e.g. {"CLL/SLL": ["CLL", "SLL"], "T-cell": ["PTCL", "AITL", "ALCL", "TCL"]}
    drop : list of label values to remove entirely from the output.
        Rows with these labels are excluded (mask returned as second output).
    min_n : groups smaller than this after merging are folded into other_label.
    other_label : name for the catch-all group.
    unknown_values : label values treated as unknown/undefined — set to None
        so they can be shown as grey background by callers.

    Returns
    -------
    remapped : array of new group labels (same dtype=object).
        Groups that are unknown → None; dropped rows → excluded via mask.
    keep_mask : boolean mask of rows to retain (drop=False rows removed).

    Examples
    --------
    # Hematology coarse-to-medium remapping
    remapped, keep = remap_disease_groups(
        cohort_fine,
        merge={
            "CLL/SLL":  ["CLL", "SLL"],
            "T-cell":   ["PTCL", "AITL", "ALCL", "TCL"],
            "Plasma":   ["MM", "PCL", "MGUS"],
        },
        drop=["RT", "AMYLOIDOSIS", "MZL", "MBL"],
        unknown_values=["<NA>", "RT_DERIVED", "UNKNOWN"],
        min_n=50,
    )
    """
    labels = np.asarray(group_labels, dtype=object)
    n = len(labels)
    out = labels.copy()

    # Apply merges first
    if merge:
        for new_name, old_names in merge.items():
            for old in old_names:
                out[out == old] = new_name

    # Mark unknowns as None
    if unknown_values:
        for uv in unknown_values:
            out[out == uv] = None

    # Build drop mask
    drop_vals = set(drop or [])
    keep_mask = np.array([v not in drop_vals for v in out])

    # Apply min_n: groups below threshold → other_label
    out_kept = out[keep_mask]
    unique, counts = np.unique(
        [v for v in out_kept if v is not None], return_counts=True
    )
    small = set(unique[counts < min_n])
    if small:
        for i in range(len(out)):
            if out[i] in small:
                out[i] = other_label

    return out, keep_mask


def _local_density(coords: np.ndarray, k: int = 15) -> np.ndarray:
    """
    Estimate local density for each point as the inverse of mean distance
    to its k nearest neighbours.  Higher value = denser neighbourhood.
    """
    from sklearn.neighbors import NearestNeighbors
    n = len(coords)
    nn = NearestNeighbors(n_neighbors=min(k + 1, n), algorithm="ball_tree")
    nn.fit(coords)
    distances, _ = nn.kneighbors(coords)
    mean_dist = distances[:, 1:].mean(axis=1)          # skip self (col 0)
    mean_dist = np.where(mean_dist == 0, 1e-9, mean_dist)
    return 1.0 / mean_dist


def plot_embedding_disease_map(
    coords: np.ndarray,
    group_labels: np.ndarray,
    group_names: Optional[Dict] = None,
    density_contours: bool = True,
    isolation_percentile: Optional[float] = 10.0,
    prediction_errors: Optional[np.ndarray] = None,
    title: Optional[str] = None,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Main embedding figure: scatter colored by disease/cohort with KDE halos.

    Isolated patients (those in the lowest-density tail of the embedding) are
    overlaid as hollow markers — position in sparse regions is associated with
    higher prediction error (validated in plot_embedding_map_insights).

    Parameters
    ----------
    coords : (n, 2) pre-computed UMAP / t-SNE coordinates.
    group_labels : 1-D array of group identifiers (ints, strings, or codes).
    group_names : optional mapping from label value → display string.
    isolation_percentile : if set, patients below this density percentile
        are redrawn as hollow circles.  Set to None to disable.
    prediction_errors : optional |predicted - true| per patient.  When provided
        alongside isolation_percentile, a caption-ready stat is printed and
        added as a figure annotation.
    """
    from sklearn.neighbors import NearestNeighbors

    unique = sorted(set(group_labels))
    colors = _cohort_colors(unique)

    # Compute isolation mask once if needed
    isolated = np.zeros(len(coords), dtype=bool)
    isolation_stat = None
    if isolation_percentile is not None:
        density = _local_density(coords)
        threshold = np.percentile(density, isolation_percentile)
        isolated = density <= threshold

        if prediction_errors is not None and isolated.any() and (~isolated).any():
            err_iso  = prediction_errors[isolated].mean()
            err_rest = prediction_errors[~isolated].mean()
            isolation_stat = (
                f"Isolated patients: mean error {err_iso:.3f} "
                f"vs {err_rest:.3f} for remainder"
            )

    fig, ax = plt.subplots(figsize=(5.2, 4.8))

    for g in unique:
        mask = (group_labels == g) & ~isolated
        color = colors[g]
        name  = (group_names or {}).get(g, str(g))
        n_g   = (group_labels == g).sum()

        ax.scatter(coords[mask, 0], coords[mask, 1],
                   c=color, s=6, alpha=0.40,
                   label=f"{name}  (n={n_g:,})",
                   rasterized=True, linewidths=0, zorder=2)

        if density_contours and mask.sum() > 40:
            _density_contour(ax, coords[mask, 0], coords[mask, 1],
                             color, levels=4, alpha=0.45)

    # Isolated patients — hollow rings, colour still encodes disease
    if isolated.any():
        for g in unique:
            mask_iso = (group_labels == g) & isolated
            if not mask_iso.any():
                continue
            ax.scatter(coords[mask_iso, 0], coords[mask_iso, 1],
                       facecolors="none", edgecolors=colors[g],
                       s=18, linewidths=0.9, alpha=0.75,
                       rasterized=True, zorder=3)
        # Single legend entry for isolated class
        from matplotlib.lines import Line2D
        ax.add_artist(ax.legend(markerscale=2.5, fontsize=LEGEND_SIZE,
                                framealpha=0.92, loc="best", borderpad=0.6))
        iso_handle = Line2D([0], [0], marker="o", color="w",
                            markerfacecolor="none", markeredgecolor="#555555",
                            markeredgewidth=0.9, markersize=6,
                            label=f"Isolated  (n={isolated.sum():,})")
        ax.legend(handles=[iso_handle], fontsize=LEGEND_SIZE,
                  loc="lower right", framealpha=0.92)
    else:
        ax.legend(markerscale=2.5, fontsize=LEGEND_SIZE,
                  framealpha=0.92, loc="best", borderpad=0.6)

    if isolation_stat:
        ax.text(0.01, 0.01, isolation_stat,
                transform=ax.transAxes, fontsize=6.5, color="#555555",
                va="bottom", ha="left")

    ax.set_title(title or "Embedding space — population structure", fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ("left", "bottom", "top", "right"):
        ax.spines[spine].set_visible(False)
    ax.grid(False)
    ax.set_xlabel("UMAP 1", fontsize=8, color="#888888")
    ax.set_ylabel("UMAP 2", fontsize=8, color="#888888")

    fig.tight_layout()
    save_fig(fig, save_path)
    return fig


def plot_embedding_map_insights(
    coords: np.ndarray,
    group_labels: np.ndarray,
    predicted_probs: np.ndarray,
    labels: Optional[np.ndarray] = None,
    group_names: Optional[Dict] = None,
    k: int = 15,
    n_density_bins: int = 10,
    title: Optional[str] = None,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Three-panel figure answering "what does the map tell us?"

    Panel A — Uncertainty geography
        UMAP colored by prediction entropy H = -p log₂p - (1-p) log₂(1-p).
        High entropy (yellow/orange) = model is uncertain; correlates with
        sparse regions and disease boundaries.

    Panel B — Density vs. prediction error  (requires labels)
        Patients binned by local neighbourhood density (low → isolated,
        high → core cluster).  Mean absolute error per bin with 95% CI.
        Validates that position in the map encodes genuine predictive signal:
        isolated patients are harder to classify.

    Panel C — Inter-disease proximity heatmap
        For every ordered pair of disease groups, what fraction of the first
        group's k nearest neighbours belong to the second group?
        Diagonal = within-group coherence; off-diagonal = cross-group leakage.
        Reveals soft boundaries: which diseases look like each other, and in
        which direction (asymmetric — DLBCL pulls toward FL more than FL pulls
        toward DLBCL is a meaningful biological finding).

    Parameters
    ----------
    coords : (n, 2) pre-computed 2D coordinates.
    group_labels : disease / cohort assignment per patient.
    predicted_probs : model output probability ∈ [0, 1].
    labels : ground-truth binary labels (required for Panel B; Panel B is
        hidden if not provided).
    k : neighbourhood size for density and proximity calculations.
    """
    from sklearn.neighbors import NearestNeighbors

    n      = len(coords)
    unique = sorted(set(group_labels))
    colors = _cohort_colors(unique)
    n_groups = len(unique)
    group_idx = {g: i for i, g in enumerate(unique)}

    # ── Shared computations ──────────────────────────────────────────
    # Local density
    density = _local_density(coords, k=k)

    # Prediction entropy
    p  = np.clip(predicted_probs, 1e-7, 1 - 1e-7)
    entropy = -(p * np.log2(p) + (1 - p) * np.log2(1 - p))

    # k-NN indices (used for proximity matrix)
    nn = NearestNeighbors(n_neighbors=min(k + 1, n), algorithm="ball_tree")
    nn.fit(coords)
    _, indices = nn.kneighbors(coords)
    neighbor_idx = indices[:, 1:]    # drop self

    # Inter-disease proximity matrix  (row i → fraction of i's neighbours in group j)
    proximity = np.zeros((n_groups, n_groups))
    for pt_i in range(n):
        g_i = group_idx[group_labels[pt_i]]
        for nb in neighbor_idx[pt_i]:
            g_nb = group_idx[group_labels[nb]]
            proximity[g_i, g_nb] += 1
    # Normalise each row by total neighbours counted for that group
    row_sums = proximity.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    proximity /= row_sums

    # ── Layout ───────────────────────────────────────────────────────
    has_labels = labels is not None
    n_panels = 3 if has_labels else 2
    width_ratios = [1.1, 0.9, 1.0] if has_labels else [1.1, 1.0]
    fig, axes = plt.subplots(
        1, n_panels,
        figsize=(4.2 * n_panels + 0.4, 4.5),
        gridspec_kw={"width_ratios": width_ratios},
    )
    if n_panels == 2:
        ax_ent, ax_prox = axes
        ax_bin = None
    else:
        ax_ent, ax_bin, ax_prox = axes

    # ── Panel A: Entropy geography ───────────────────────────────────
    sc = ax_ent.scatter(
        coords[:, 0], coords[:, 1],
        c=entropy, cmap="YlOrRd",
        vmin=0, vmax=1,
        s=5, alpha=0.6, rasterized=True, linewidths=0,
    )
    cb = fig.colorbar(sc, ax=ax_ent, shrink=0.82, pad=0.02, aspect=20)
    cb.set_label("Prediction entropy  (bits)", fontsize=7.5)
    cb.ax.tick_params(labelsize=7)

    # Overlay disease centroids as labelled dots
    for g in unique:
        mask = group_labels == g
        cx, cy = coords[mask, 0].mean(), coords[mask, 1].mean()
        name = (group_names or {}).get(g, str(g))
        ax_ent.text(cx, cy, name, fontsize=7.5, fontweight="semibold",
                    ha="center", va="center",
                    bbox=dict(boxstyle="round,pad=0.2", fc="white",
                              ec=colors[g], lw=1.1, alpha=0.88),
                    zorder=5)

    ax_ent.set_title("Prediction uncertainty", fontsize=9, fontweight="semibold")
    ax_ent.set_xticks([])
    ax_ent.set_yticks([])
    for s in ("left", "bottom", "top", "right"):
        ax_ent.spines[s].set_visible(False)
    ax_ent.grid(False)
    ax_ent.set_xlabel("UMAP 1", fontsize=7.5, color="#888888")
    ax_ent.set_ylabel("UMAP 2", fontsize=7.5, color="#888888")

    # ── Panel B: Density vs. error  (only when labels available) ────
    if has_labels and ax_bin is not None:
        errors = np.abs(predicted_probs - labels.astype(float))
        # Bin by density percentile
        bin_edges = np.percentile(density, np.linspace(0, 100, n_density_bins + 1))
        bin_edges = np.unique(bin_edges)
        bin_ids = np.digitize(density, bin_edges[:-1]) - 1
        bin_ids = np.clip(bin_ids, 0, len(bin_edges) - 2)

        bin_centers, bin_means, bin_lo, bin_hi, bin_ns = [], [], [], [], []
        for b in range(len(bin_edges) - 1):
            mask = bin_ids == b
            if mask.sum() < 5:
                continue
            e = errors[mask]
            m = e.mean()
            se = e.std() / np.sqrt(len(e))
            bin_centers.append(density[mask].mean())
            bin_means.append(m)
            bin_lo.append(m - 1.96 * se)
            bin_hi.append(m + 1.96 * se)
            bin_ns.append(mask.sum())

        bin_centers = np.array(bin_centers)
        bin_means   = np.array(bin_means)
        bin_lo      = np.array(bin_lo)
        bin_hi      = np.array(bin_hi)

        ax_bin.plot(bin_centers, bin_means,
                    color=PALETTE["opera"], lw=1.8, marker="o",
                    markersize=4, zorder=3)
        ax_bin.fill_between(bin_centers, bin_lo, bin_hi,
                            color=PALETTE["opera"], alpha=0.15)

        ax_bin.set_xlabel("Local neighbourhood density\n(low = isolated)", fontsize=8)
        ax_bin.set_ylabel("Mean absolute error", fontsize=8)
        ax_bin.set_title("Isolation → prediction error", fontsize=9,
                         fontweight="semibold")
        # Annotate with correlation
        if len(bin_centers) > 2:
            rho = float(np.corrcoef(bin_centers, bin_means)[0, 1])
            ax_bin.text(0.97, 0.97, f"r = {rho:+.2f}",
                        transform=ax_bin.transAxes, ha="right", va="top",
                        fontsize=ANNOT_SIZE, color="#555555")
        despine(ax_bin, "y")

    # ── Panel C: Inter-disease proximity heatmap ─────────────────────
    group_display = [(group_names or {}).get(g, str(g)) for g in unique]

    im = ax_prox.imshow(proximity, cmap="Blues", vmin=0, vmax=proximity.max(),
                        aspect="auto")
    cb2 = fig.colorbar(im, ax=ax_prox, shrink=0.82, pad=0.03, aspect=20)
    cb2.set_label("Fraction of k-NN", fontsize=7.5)
    cb2.ax.tick_params(labelsize=7)

    ax_prox.set_xticks(range(n_groups))
    ax_prox.set_yticks(range(n_groups))
    ax_prox.set_xticklabels(group_display, rotation=40, ha="right", fontsize=7.5)
    ax_prox.set_yticklabels(group_display, fontsize=7.5)
    ax_prox.set_xlabel("Neighbour's disease", fontsize=8)
    ax_prox.set_ylabel("Patient's disease", fontsize=8)
    ax_prox.set_title("Inter-disease proximity", fontsize=9, fontweight="semibold")

    # Annotate cells
    for i in range(n_groups):
        for j in range(n_groups):
            v = proximity[i, j]
            text_color = "white" if v > 0.55 * proximity.max() else "#333333"
            ax_prox.text(j, i, f"{v:.2f}", ha="center", va="center",
                         fontsize=6.5, color=text_color)

    # Bold diagonal (within-group coherence)
    for i in range(n_groups):
        ax_prox.add_patch(plt.Rectangle(
            (i - 0.5, i - 0.5), 1, 1,
            fill=False, edgecolor=PALETTE["opera"], lw=1.5, zorder=3,
        ))

    panel_labels = ["A", "B", "C"] if has_labels else ["A", "B"]
    for ax, lbl in zip(axes, panel_labels):
        add_panel_label(ax, lbl)

    fig.suptitle(title or "Embedding space — clinical insights",
                 fontsize=10, fontweight="semibold", y=1.02)
    fig.tight_layout()
    save_fig(fig, save_path)
    return fig


def plot_clinical_variables_panel(
    coords: np.ndarray,
    clinical_vars: Dict[str, np.ndarray],
    cmaps: Optional[Dict[str, str]] = None,
    max_cols: int = 3,
    title: Optional[str] = None,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Small-multiples panel: same UMAP coords, each subplot colored by a
    different clinical variable.

    Continuous variables (float) → colorbar with diverging/sequential map.
    Categorical variables (int/str unique ≤ 12) → discrete legend.

    Parameters
    ----------
    coords : (n, 2) pre-computed 2D coordinates.
    clinical_vars : dict of variable_name → 1-D array (same length as coords).
        NaN / -1 / "" → shown as light grey "unknown" points.
    cmaps : optional override mapping variable_name → matplotlib cmap name.
        Defaults: age → "YlOrRd", ipi / score → "RdYlGn_r", else "viridis".

    Example
    -------
    plot_clinical_variables_panel(
        coords,
        {"Age at diagnosis": ages, "IPI score": ipi, "Stage": stage},
    )
    """
    names = list(clinical_vars.keys())
    n_vars = len(names)
    if n_vars == 0:
        return plt.figure()

    cols = min(n_vars, max_cols)
    rows = (n_vars + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(3.8 * cols, 3.6 * rows))
    if n_vars == 1:
        axes = np.array([axes])
    axes_flat = np.array(axes).flatten()

    _default_cmaps = {
        "age": "YlOrRd", "ipi": "RdYlGn_r", "score": "RdYlGn_r",
        "stage": "Blues", "default": "viridis",
    }

    def _pick_cmap(name_lower):
        if cmaps and name_lower in cmaps:
            return cmaps[name_lower]
        for key, cmap in _default_cmaps.items():
            if key in name_lower:
                return cmap
        return _default_cmaps["default"]

    for i, var_name in enumerate(names):
        ax   = axes_flat[i]
        vals = np.asarray(clinical_vars[var_name])

        # Determine if categorical
        non_nan = vals[~(vals == None)]  # noqa: E711
        try:
            numeric_vals = vals.astype(float)
            is_nan = ~np.isfinite(numeric_vals)
            unique_vals = np.unique(numeric_vals[~is_nan])
            is_categorical = (len(unique_vals) <= 10) and np.all(unique_vals == unique_vals.astype(int))
        except (ValueError, TypeError):
            is_categorical = True
            is_nan = np.array([v is None or str(v) == "" for v in vals])

        # Draw unknown/missing as background first
        if is_nan.any():
            ax.scatter(coords[is_nan, 0], coords[is_nan, 1],
                       c=PALETTE["missing"], s=4, alpha=0.20,
                       rasterized=True, linewidths=0, zorder=1)

        valid = ~is_nan

        if is_categorical:
            unique_groups = sorted(set(vals[valid]))
            colors = _cohort_colors(unique_groups)
            for g in unique_groups:
                mask = valid & (vals == g)
                ax.scatter(coords[mask, 0], coords[mask, 1],
                           c=colors[g], s=5, alpha=0.45,
                           label=str(g), rasterized=True, linewidths=0, zorder=2)
            ax.legend(markerscale=2, fontsize=6.5, borderpad=0.5,
                      title=var_name, title_fontsize=7)
        else:
            numeric_vals = vals.astype(float)
            cmap_name = _pick_cmap(var_name.lower())
            vmin = np.nanpercentile(numeric_vals[valid], 2)
            vmax = np.nanpercentile(numeric_vals[valid], 98)
            sc = ax.scatter(coords[valid, 0], coords[valid, 1],
                            c=numeric_vals[valid], cmap=cmap_name,
                            vmin=vmin, vmax=vmax,
                            s=5, alpha=0.55, rasterized=True, linewidths=0, zorder=2)
            cb = fig.colorbar(sc, ax=ax, shrink=0.75, pad=0.02, aspect=18)
            cb.ax.tick_params(labelsize=6.5)

        ax.set_title(var_name, fontsize=9, fontweight="semibold")
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ("left", "bottom", "top", "right"):
            ax.spines[spine].set_visible(False)
        ax.grid(False)

    for j in range(n_vars, len(axes_flat)):
        axes_flat[j].set_visible(False)

    fig.suptitle(title or "Clinical variables in embedding space",
                 fontsize=10, fontweight="semibold", y=1.01)
    fig.tight_layout()
    save_fig(fig, save_path)
    return fig


def plot_embedding_boundary_patients(
    coords: np.ndarray,
    group_labels: np.ndarray,
    group_names: Optional[Dict] = None,
    k: int = 15,
    boundary_threshold: float = 0.5,
    top_n_annotate: int = 0,
    subject_ids: Optional[np.ndarray] = None,
    title: Optional[str] = None,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Identify and visualise patients who sit at the boundary between groups.

    For each patient, compute the "coherence score" = fraction of k nearest
    neighbors who share the same group label.  Patients with coherence < threshold
    are "boundary patients" — they are positioned closer to a different group
    than their own, which can indicate:
      - Biologically ambiguous cases (e.g., DLBCL with FL-like gene expression)
      - Unusual clinical presentations
      - Rare subtypes bridging two disease categories

    Layout
    ------
    Panel A: UMAP scatter colored by coherence score (blue=coherent,
             orange/red=boundary), group centroids labelled.
    Panel B: Box plot of coherence scores per group — reveals which
             diseases are most clearly separated vs most ambiguous.

    Parameters
    ----------
    coords : (n, 2) pre-computed 2D coordinates.
    group_labels : 1-D array of group membership.
    k : number of nearest neighbors to use for coherence.
    boundary_threshold : patients with coherence < this are highlighted.
    top_n_annotate : if > 0 and subject_ids provided, annotate the N most
        anomalous patients with their subject ID (useful for clinical case review).
    """
    from sklearn.neighbors import NearestNeighbors

    n = len(coords)
    unique = sorted(set(group_labels))
    colors = _cohort_colors(unique)

    # k-NN in 2D coords (fast; could also use full embedding)
    nn = NearestNeighbors(n_neighbors=min(k + 1, n), algorithm="ball_tree")
    nn.fit(coords)
    _, indices = nn.kneighbors(coords)
    # indices[:,0] is the point itself — skip it
    neighbor_idx = indices[:, 1:]

    coherence = np.array([
        (group_labels[neighbor_idx[i]] == group_labels[i]).mean()
        for i in range(n)
    ])

    # ── Figure ───────────────────────────────────────────────────────
    fig, (ax_map, ax_box) = plt.subplots(
        1, 2, figsize=(9.0, 4.5),
        gridspec_kw={"width_ratios": [1.1, 0.9]},
    )

    # Panel A: scatter colored by coherence
    sc = ax_map.scatter(
        coords[:, 0], coords[:, 1],
        c=coherence, cmap="RdYlBu",
        vmin=0, vmax=1,
        s=6, alpha=0.65, rasterized=True, linewidths=0, zorder=2,
    )
    cb = fig.colorbar(sc, ax=ax_map, shrink=0.8, pad=0.02, aspect=20)
    cb.set_label("k-NN coherence  (1 = all neighbors same group)", fontsize=7.5)
    cb.ax.tick_params(labelsize=7)

    # Boundary patients: redraw as filled squares
    boundary = coherence < boundary_threshold
    ax_map.scatter(coords[boundary, 0], coords[boundary, 1],
                   c=coherence[boundary], cmap="RdYlBu",
                   vmin=0, vmax=1, s=14, alpha=0.9,
                   marker="s", linewidths=0.5, edgecolors="#333333",
                   rasterized=True, zorder=4,
                   label=f"Boundary patients (n={boundary.sum()})")

    # Group centroids
    for g in unique:
        mask = group_labels == g
        cx, cy = coords[mask, 0].mean(), coords[mask, 1].mean()
        name   = (group_names or {}).get(g, str(g))
        ax_map.text(cx, cy, name, fontsize=8, fontweight="semibold",
                    ha="center", va="center",
                    bbox=dict(boxstyle="round,pad=0.25", fc="white",
                              ec=colors[g], lw=1.2, alpha=0.85),
                    zorder=5)

    # Optionally annotate most anomalous
    if top_n_annotate > 0 and subject_ids is not None:
        worst_idx = np.argsort(coherence)[:top_n_annotate]
        for idx in worst_idx:
            ax_map.annotate(
                str(subject_ids[idx]),
                xy=(coords[idx, 0], coords[idx, 1]),
                xytext=(8, 8), textcoords="offset points",
                fontsize=6.5, color="#B5232A",
                arrowprops=dict(arrowstyle="-", lw=0.6, color="#AAAAAA"),
            )

    ax_map.set_title(title or "Boundary patients — k-NN coherence", fontsize=9)
    if boundary.sum() > 0:
        ax_map.legend(markerscale=1.5, fontsize=LEGEND_SIZE, loc="lower left")
    ax_map.set_xticks([])
    ax_map.set_yticks([])
    for spine in ("left", "bottom", "top", "right"):
        ax_map.spines[spine].set_visible(False)
    ax_map.grid(False)
    ax_map.set_xlabel("UMAP 1", fontsize=8, color="#888888")
    ax_map.set_ylabel("UMAP 2", fontsize=8, color="#888888")

    # Panel B: box plot of coherence per group, sorted by median
    group_coherence = {g: coherence[group_labels == g] for g in unique}
    sorted_groups = sorted(unique, key=lambda g: np.median(group_coherence[g]))
    y_labels = [(group_names or {}).get(g, str(g)) for g in sorted_groups]
    data_for_box = [group_coherence[g] for g in sorted_groups]

    bp = ax_box.boxplot(
        data_for_box,
        vert=False,
        patch_artist=True,
        widths=0.55,
        medianprops=dict(color="#333333", lw=1.5),
        whiskerprops=dict(lw=0.8),
        capprops=dict(lw=0.8),
        flierprops=dict(marker=".", ms=2.5, alpha=0.4),
    )
    for patch, g in zip(bp["boxes"], sorted_groups):
        patch.set_facecolor(colors[g])
        patch.set_alpha(0.65)
        patch.set_edgecolor("#444444")
        patch.set_linewidth(0.8)

    ax_box.axvline(boundary_threshold, color=PALETTE["positive"],
                   ls="--", lw=1.0, label=f"Threshold ({boundary_threshold})")
    ax_box.axvline(1.0, color=PALETTE["diagonal"], ls=":", lw=0.8)
    ax_box.set_yticklabels(y_labels, fontsize=8)
    ax_box.set_xlabel("k-NN coherence score")
    ax_box.set_title("Separation by group", fontsize=9)
    ax_box.set_xlim(0, 1.05)
    ax_box.legend(fontsize=LEGEND_SIZE)
    despine(ax_box, "none")
    ax_box.grid(axis="x", color=PALETTE["grid"], linewidth=0.5)
    ax_box.grid(axis="y", visible=False)

    add_panel_label(ax_map, "A")
    add_panel_label(ax_box, "B")

    fig.tight_layout()
    save_fig(fig, save_path)
    return fig


def plot_source_vs_outcome(
    embeddings: np.ndarray,
    source_labels: np.ndarray,
    outcome_labels: np.ndarray,
    method: str = "umap",
    outcome_name: str = "outcome",
    save_path: Optional[str] = None,
    **reducer_kwargs,
) -> plt.Figure:
    coords = _reduce_embeddings(embeddings, method, **reducer_kwargs)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.0, 4.0))

    for source in sorted(set(source_labels)):
        mask = source_labels == source
        color = SOURCE_PALETTE.get(source, "#999999")
        ax1.scatter(coords[mask, 0], coords[mask, 1],
                    c=color, s=5, alpha=0.35, label=source, rasterized=True, linewidths=0)
    ax1.set_title("By data source")
    ax1.legend(markerscale=3, fontsize=LEGEND_SIZE)

    valid = outcome_labels >= 0
    ax2.scatter(coords[~valid, 0], coords[~valid, 1],
                c=PALETTE["missing"], s=3, alpha=0.12, rasterized=True, linewidths=0)
    for label_val, color, name in [
        (0, PALETTE["negative"], "Negative"),
        (1, PALETTE["positive"], "Positive"),
    ]:
        mask = valid & (outcome_labels == label_val)
        ax2.scatter(coords[mask, 0], coords[mask, 1],
                    c=color, s=5, alpha=0.4, label=name, rasterized=True, linewidths=0)
    ax2.set_title(f"By {outcome_name.replace('_', ' ')}")
    ax2.legend(markerscale=3, fontsize=LEGEND_SIZE)

    for ax in (ax1, ax2):
        ax.set_xticks([])
        ax.set_yticks([])
        ax.spines["left"].set_visible(False)
        ax.spines["bottom"].set_visible(False)
        ax.grid(False)

    add_panel_label(ax1, "A")
    add_panel_label(ax2, "B")
    fig.tight_layout()
    save_fig(fig, save_path)
    return fig
