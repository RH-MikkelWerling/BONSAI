"""
OPERA Analysis Plots.

  1. Cross-outcome transfer matrix heatmap
  2. Sigma heatmap across cohorts
  3. Within-stratum embedding grids
  4. RKKP residual embedding
  5. Embedding confidence scatter
  6. Performance landscape  (smoothed accuracy over embedding space)
  7. Embedding atlas  (annotated clinical map with outcome axes)
  8. Landscape summary  (compact per-outcome error summary for main figures)
"""

from __future__ import annotations
from typing import Dict, List, Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from opera.visualization.style import (
    PALETTE,
    CATEGORICAL,
    save_fig,
    despine,
    add_panel_label,
    style_legend,
    sequential_cmap,
    ANNOT_SIZE,
    LABEL_SIZE,
    LEGEND_SIZE,
    LEGEND_TITLE_SIZE,
    NOTE_SIZE,
    SUBTITLE_SIZE,
    SUPTITLE_SIZE,
    TICK_SIZE,
    TITLE_SIZE,
)


# ═════════════════════════════════════════════════════════════════════
# 1. Cross-outcome transfer matrix
# ═════════════════════════════════════════════════════════════════════


def _outcome_display(name: str) -> str:
    return name.replace("_", "\n").title()


def plot_transfer_matrix(
    matrix: pd.DataFrame,
    metric: str = "AUROC",
    title: Optional[str] = None,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Annotated heatmap of the cross-outcome transfer matrix.

    Diagonal = cross-validated self-prediction (bold).
    Off-diagonal = zero-shot transfer (train on row, evaluate on column).

    Color: green = high discrimination, red = near-random.
    Diagonal cells shown in a distinct shade to distinguish from transfer.
    """
    n = len(matrix)
    values = matrix.values.astype(float)
    diag_vals = np.diag(values)

    fig, ax = plt.subplots(figsize=(max(5.0, 1.4 * n), max(4.5, 1.2 * n)))

    # Mask diagonal to plot separately
    off_diag = values.copy()
    np.fill_diagonal(off_diag, np.nan)

    # Off-diagonal heatmap (0.4 to 1.0 range)
    cmap_off = plt.cm.RdYlGn
    im = ax.imshow(off_diag, cmap=cmap_off, vmin=0.4, vmax=1.0, aspect="auto")

    # Diagonal overlay with a distinct colormap slice
    diag_cmap = plt.cm.Blues
    for i in range(n):
        v = diag_vals[i]
        bg = (
            diag_cmap(0.4 + 0.5 * (v - 0.4) / 0.6)
            if not np.isnan(v)
            else (0.9, 0.9, 0.9, 1)
        )
        rect = plt.Rectangle((i - 0.5, i - 0.5), 1, 1, color=bg, zorder=2)
        ax.add_patch(rect)

    # Annotations
    for i in range(n):
        for j in range(n):
            v = matrix.iloc[i, j]
            if np.isnan(v):
                text, fw = "—", "normal"
            else:
                text, fw = f"{v:.3f}", ("bold" if i == j else "normal")
            # White text on dark cells, dark ink on light
            bg_val = v if not np.isnan(v) else 0.7
            text_color = "white" if (bg_val < 0.52 or bg_val > 0.88) else PALETTE["ink"]
            ax.text(
                j,
                i,
                text,
                ha="center",
                va="center",
                fontsize=ANNOT_SIZE,
                fontweight=fw,
                color=text_color,
                zorder=3,
            )

    display_names = [_outcome_display(c) for c in matrix.columns]
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(display_names, fontsize=TICK_SIZE)
    ax.set_yticklabels(display_names, fontsize=TICK_SIZE)
    ax.set_xlabel("Evaluated on  (target outcome)", fontsize=LABEL_SIZE)
    ax.set_ylabel("Trained on  (source outcome)", fontsize=LABEL_SIZE)
    ax.set_title(
        title or f"Cross-outcome transfer  ({metric})",
        fontsize=TITLE_SIZE,
        fontweight="semibold",
        color=PALETTE["ink"],
    )

    cbar = fig.colorbar(im, ax=ax, shrink=0.75, pad=0.02)
    cbar.set_label(
        f"{metric}  (off-diagonal)", fontsize=NOTE_SIZE, color=PALETTE["ink_secondary"]
    )
    cbar.ax.tick_params(labelsize=NOTE_SIZE, colors=PALETTE["ink_secondary"])
    cbar.outline.set_edgecolor(PALETTE["panel_border"])

    # Legend for diagonal
    from matplotlib.patches import Patch

    legend_els = [
        Patch(facecolor=plt.cm.Blues(0.7), label="Diagonal: self-prediction (CV)"),
        Patch(facecolor=plt.cm.RdYlGn(0.8), label="Off-diagonal: transfer"),
    ]
    legend = ax.legend(
        handles=legend_els,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.36),
        fontsize=LEGEND_SIZE,
        frameon=True,
        ncol=2,
    )
    style_legend(legend)

    fig.tight_layout()
    fig.subplots_adjust(bottom=0.32)
    save_fig(fig, save_path)
    return fig


def plot_transfer_efficiency(
    efficiency: pd.DataFrame,
    title: str = "Transfer efficiency",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Efficiency = AUROC(A→B) / AUROC(B→B).
    Diagonal masked; off-diagonal colored 0.3 (no transfer) → 1.0 (full transfer).
    """
    n = len(efficiency)
    values = efficiency.values.astype(float)
    np.fill_diagonal(values, np.nan)

    fig, ax = plt.subplots(figsize=(max(5.0, 1.4 * n), max(4.5, 1.2 * n)))

    cmap = plt.cm.RdYlGn
    im = ax.imshow(values, cmap=cmap, vmin=0.3, vmax=1.0, aspect="auto")

    for i in range(n):
        for j in range(n):
            if i == j:
                ax.text(
                    j,
                    i,
                    "—",
                    ha="center",
                    va="center",
                    fontsize=ANNOT_SIZE,
                    color=PALETTE["ink_muted"],
                )
                continue
            v = efficiency.iloc[i, j]
            if np.isnan(v):
                continue
            text_color = "white" if (v < 0.45 or v > 0.88) else PALETTE["ink"]
            ax.text(
                j,
                i,
                f"{v:.2f}",
                ha="center",
                va="center",
                fontsize=ANNOT_SIZE,
                color=text_color,
            )

    display_names = [_outcome_display(c) for c in efficiency.columns]
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(display_names, fontsize=TICK_SIZE)
    ax.set_yticklabels(display_names, fontsize=TICK_SIZE)
    ax.set_xlabel("Target outcome", fontsize=LABEL_SIZE)
    ax.set_ylabel("Source outcome", fontsize=LABEL_SIZE)
    ax.set_title(
        title, fontsize=TITLE_SIZE, fontweight="semibold", color=PALETTE["ink"]
    )

    cbar = fig.colorbar(im, ax=ax, shrink=0.75, pad=0.02)
    cbar.set_label(
        "Transfer efficiency\n(fraction of self-prediction AUROC)",
        fontsize=NOTE_SIZE,
        color=PALETTE["ink_secondary"],
    )
    cbar.ax.tick_params(labelsize=NOTE_SIZE, colors=PALETTE["ink_secondary"])
    cbar.outline.set_edgecolor(PALETTE["panel_border"])

    fig.tight_layout()
    save_fig(fig, save_path)
    return fig


# ═════════════════════════════════════════════════════════════════════
# 2. Sigma heatmap across cohorts
# ═════════════════════════════════════════════════════════════════════


def plot_sigma_heatmap(
    sigma_data: pd.DataFrame,
    title: str = "Learned σ  (outcome × cohort)",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Heatmap of learned sigma values.

    Rows = outcomes, columns = cohorts.
    Blue = low sigma (outcome strongly structures embedding).
    Red  = high sigma (outcome contributes weak signal).
    """
    data = sigma_data.values.astype(float)
    vmin = np.nanpercentile(data, 5)
    vmax = np.nanpercentile(data, 95)

    # Reversed: blue = low (good) to red = high (weak)
    cmap = plt.cm.RdYlBu_r

    fig, ax = plt.subplots(
        figsize=(
            max(5.0, len(sigma_data.columns) * 1.3),
            max(3.5, len(sigma_data) * 0.8),
        )
    )

    im = ax.imshow(data, cmap=cmap, aspect="auto", vmin=vmin, vmax=vmax)

    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            v = data[i, j]
            if not np.isnan(v):
                mid = (vmin + vmax) / 2
                text_color = (
                    "white" if abs(v - mid) > 0.3 * (vmax - vmin) else PALETTE["ink"]
                )
                ax.text(
                    j,
                    i,
                    f"{v:.2f}",
                    ha="center",
                    va="center",
                    fontsize=ANNOT_SIZE,
                    color=text_color,
                )

    ax.set_xticks(range(len(sigma_data.columns)))
    ax.set_xticklabels(sigma_data.columns, rotation=30, ha="right", fontsize=TICK_SIZE)
    ax.set_yticks(range(len(sigma_data.index)))
    ax.set_yticklabels(
        [_outcome_display(s) for s in sigma_data.index], fontsize=TICK_SIZE
    )
    ax.set_title(
        title, fontsize=TITLE_SIZE, fontweight="semibold", color=PALETTE["ink"], pad=10
    )

    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label(
        "σ  (lower = stronger outcome structure)",
        fontsize=NOTE_SIZE,
        color=PALETTE["ink_secondary"],
    )
    cbar.ax.tick_params(labelsize=NOTE_SIZE, colors=PALETTE["ink_secondary"])
    cbar.outline.set_edgecolor(PALETTE["panel_border"])

    fig.tight_layout()
    save_fig(fig, save_path)
    return fig


# ═════════════════════════════════════════════════════════════════════
# 3. RKKP residual embedding
# ═════════════════════════════════════════════════════════════════════


def plot_residual_embedding(
    embeddings: np.ndarray,
    residuals: np.ndarray,
    method: str = "umap",
    title: str = "Embedding — residual from IPI prediction",
    save_path: Optional[str] = None,
    **reducer_kwargs,
) -> plt.Figure:
    """
    UMAP colored by residuals (actual − IPI predicted probability).

    Structure in the residual map → the model captures information
    the IPI score does not.
    """
    from opera.visualization.embedding_plots import _reduce_embeddings

    coords = _reduce_embeddings(embeddings, method, **reducer_kwargs)

    vmax = max(
        abs(np.nanpercentile(residuals, 2)), abs(np.nanpercentile(residuals, 98))
    )

    fig, ax = plt.subplots(figsize=(5.0, 4.5))
    sc = ax.scatter(
        coords[:, 0],
        coords[:, 1],
        c=residuals,
        cmap="RdBu_r",
        s=5,
        alpha=0.55,
        vmin=-vmax,
        vmax=vmax,
        rasterized=True,
        linewidths=0,
        zorder=2,
    )

    cbar = fig.colorbar(sc, ax=ax, shrink=0.85, pad=0.02)
    cbar.set_label(
        "Residual  (actual − IPI predicted)\n"
        "Red = worse than expected    Blue = better",
        fontsize=NOTE_SIZE,
        color=PALETTE["ink_secondary"],
    )
    cbar.ax.tick_params(labelsize=NOTE_SIZE, colors=PALETTE["ink_secondary"])
    cbar.outline.set_edgecolor(PALETTE["panel_border"])

    ax.set_title(
        title, fontsize=TITLE_SIZE, fontweight="semibold", color=PALETTE["ink"]
    )
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.grid(False)

    fig.tight_layout()
    save_fig(fig, save_path)
    return fig


# ═════════════════════════════════════════════════════════════════════
# 4. Within-stratum embedding grid
# ═════════════════════════════════════════════════════════════════════


def plot_within_stratum_grid(
    embeddings: np.ndarray,
    labels: np.ndarray,
    strata: np.ndarray,
    stratum_values: Optional[list] = None,
    method: str = "umap",
    stratum_name: str = "Risk stratum",
    outcome_name: str = "outcome",
    save_path: Optional[str] = None,
    min_n: int = 30,
    title: Optional[str] = None,
    subtitle: Optional[str] = None,
    stratum_labels: Optional[Dict[object, str]] = None,
    highlight_stratum: Optional[object] = None,
    emphasize_largest: bool = True,
    show_footer: bool = True,
) -> plt.Figure:
    """Show outcome geometry within clinical strata on one shared projection.

    Every panel contains the complete cohort in grey and overlays one stratum,
    split by observed outcome. Keeping the coordinates and axis limits fixed
    means that only the highlighted patients change between panels. The
    largest displayed stratum (or ``highlight_stratum``) receives a bordered
    frame, called out in the legend — the figure itself carries no
    interpretive text; that belongs in the surrounding narrative, not the
    plot.
    """
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    from opera.visualization.embedding_plots import _reduce_embeddings

    embeddings = np.asarray(embeddings)
    labels = np.asarray(labels)
    strata = np.asarray(strata)
    if embeddings.ndim != 2:
        raise ValueError("embeddings must be a two-dimensional array.")
    if labels.ndim != 1 or strata.ndim != 1:
        raise ValueError("labels and strata must be one-dimensional arrays.")
    if not (len(embeddings) == len(labels) == len(strata)):
        raise ValueError("embeddings, labels, and strata must have equal length.")
    if len(embeddings) == 0:
        raise ValueError("At least one embedding is required.")

    binary_label = np.isin(labels, [0, 1])
    if stratum_values is None:
        unique_s = list(pd.unique(strata))
        try:
            unique_s = sorted(unique_s)
        except TypeError:
            unique_s = sorted(unique_s, key=lambda value: str(value))
        stratum_values = [
            value
            for value in unique_s
            if (strata == value).sum() >= min_n
            and len(np.unique(labels[(strata == value) & binary_label])) >= 2
        ]
    else:
        stratum_values = list(stratum_values)

    n = len(stratum_values)
    if n == 0:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No valid strata", ha="center", va="center")
        ax.set_axis_off()
        return fig

    coords = np.asarray(_reduce_embeddings(embeddings, method))
    if coords.shape != (len(embeddings), 2):
        raise ValueError("The embedding reducer must return an (n_patients, 2) array.")
    if not np.isfinite(coords).all():
        raise ValueError("The shared embedding projection contains non-finite values.")

    key_stratum = highlight_stratum
    if key_stratum is None and emphasize_largest:
        key_stratum = max(
            stratum_values,
            key=lambda value: int((strata == value).sum()),
        )
    if key_stratum is not None and key_stratum not in stratum_values:
        raise ValueError("highlight_stratum must be one of the displayed strata.")

    cols = min(n, 3)
    rows = (n + cols - 1) // cols
    footer_height = 1.28 if show_footer else 0.28
    fig, axes = plt.subplots(
        rows,
        cols,
        figsize=(max(7.1, 3.25 * cols), 3.05 * rows + footer_height + 0.72),
        squeeze=False,
    )
    axes = axes.ravel()

    x_span = float(np.ptp(coords[:, 0])) or 1.0
    y_span = float(np.ptp(coords[:, 1])) or 1.0
    x_limits = (
        float(coords[:, 0].min() - 0.035 * x_span),
        float(coords[:, 0].max() + 0.035 * x_span),
    )
    y_limits = (
        float(coords[:, 1].min() - 0.035 * y_span),
        float(coords[:, 1].max() + 0.035 * y_span),
    )
    panel_aspect = 1.02
    x_width = x_limits[1] - x_limits[0]
    y_height = y_limits[1] - y_limits[0]
    x_midpoint = sum(x_limits) / 2
    y_midpoint = sum(y_limits) / 2
    if y_height < panel_aspect * x_width:
        y_height = panel_aspect * x_width
    else:
        x_width = y_height / panel_aspect
    x_limits = (x_midpoint - x_width / 2, x_midpoint + x_width / 2)
    y_limits = (y_midpoint - y_height / 2, y_midpoint + y_height / 2)
    background_color = "#DBDAD5"
    border_color = PALETTE["panel_border"]
    key_border_color = PALETTE["opera"]
    negative_color = PALETTE["negative"]
    positive_color = PALETTE["positive"]
    stratum_labels = stratum_labels or {}

    for i, s_val in enumerate(stratum_values):
        ax = axes[i]
        mask = strata == s_val
        ax.scatter(
            coords[:, 0],
            coords[:, 1],
            c=background_color,
            s=3.4,
            alpha=0.43,
            rasterized=True,
            linewidths=0,
            zorder=1,
        )

        for label_val, color in [(0, negative_color), (1, positive_color)]:
            selected = mask & (labels == label_val)
            if selected.any():
                ax.scatter(
                    coords[selected, 0],
                    coords[selected, 1],
                    c=color,
                    s=7.5,
                    alpha=0.88,
                    rasterized=True,
                    linewidths=0,
                    zorder=3,
                )

        n_s = int(mask.sum())
        valid_s = mask & binary_label
        prevalence = (
            f"{float(labels[valid_s].mean()):.0%}" if valid_s.any() else "not available"
        )
        display_stratum = stratum_labels.get(s_val, f"{stratum_name} {s_val}")
        # Two text() calls (bold name + regular detail) instead of a mathtext
        # "$\\bf{...}$" hack — that broke on any label containing an
        # underscore or other mathtext-special character.
        ax.text(
            0.0,
            1.065,
            str(display_stratum),
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=TITLE_SIZE - 2,
            fontweight="semibold",
            color=PALETTE["ink"],
        )
        ax.text(
            0.0,
            1.015,
            f"n = {n_s:,}    event prevalence = {prevalence}",
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=NOTE_SIZE,
            color=PALETTE["ink_secondary"],
        )
        ax.set_xlim(x_limits)
        ax.set_ylim(y_limits)
        ax.set_box_aspect(panel_aspect)
        ax.set_xticks([])
        ax.set_yticks([])
        is_key = s_val == key_stratum
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(1.4 if is_key else 0.8)
            spine.set_edgecolor(key_border_color if is_key else border_color)
        ax.grid(False)

    for j in range(n, len(axes)):
        axes[j].set_visible(False)

    default_title = f"{stratum_name} within-stratum separation in OPERA embedding space"
    if outcome_name != "outcome":
        default_title += f" — {outcome_name.replace('_', ' ')}"
    title = title or default_title
    subtitle = subtitle or (
        f"Shared {method.upper()} projection, shown separately within "
        f"clinician-defined {stratum_name} strata"
    )
    fig.suptitle(
        title,
        fontsize=SUPTITLE_SIZE,
        fontweight="semibold",
        color=PALETTE["ink"],
        y=0.985,
    )
    fig.text(
        0.5,
        0.947,
        subtitle,
        ha="center",
        va="top",
        fontsize=SUBTITLE_SIZE,
        color=PALETTE["ink_secondary"],
        style="italic",
    )

    if show_footer:
        legend_handles = [
            Line2D(
                [0],
                [0],
                marker="o",
                linestyle="none",
                markerfacecolor=background_color,
                markeredgecolor="none",
                markersize=7,
                label="All patients (background)",
            ),
            Line2D(
                [0],
                [0],
                marker="o",
                linestyle="none",
                markerfacecolor=negative_color,
                markeredgecolor="none",
                markersize=7,
                label="Within-stratum, no event",
            ),
            Line2D(
                [0],
                [0],
                marker="o",
                linestyle="none",
                markerfacecolor=positive_color,
                markeredgecolor="none",
                markersize=7,
                label="Within-stratum, event",
            ),
        ]
        if key_stratum is not None:
            legend_handles.append(
                Patch(
                    facecolor="none",
                    edgecolor=key_border_color,
                    linewidth=1.4,
                    label="Highlighted stratum",
                )
            )
        legend = fig.legend(
            handles=legend_handles,
            loc="lower center",
            bbox_to_anchor=(0.5, 0.01),
            ncol=len(legend_handles),
            fontsize=LEGEND_SIZE,
            handletextpad=0.5,
            columnspacing=1.5,
        )
        style_legend(legend)
        bottom_margin = 0.155
    else:
        bottom_margin = 0.05

    fig.subplots_adjust(
        left=0.035,
        right=0.985,
        top=0.80,
        bottom=bottom_margin,
        wspace=0.08,
        hspace=0.42,
    )
    save_fig(fig, save_path)
    return fig


# ═════════════════════════════════════════════════════════════════════
# 5. Model confidence scatter  (embedding locality)
# ═════════════════════════════════════════════════════════════════════


def plot_embedding_confidence(
    model_confidence: np.ndarray,
    outcome_labels: np.ndarray,
    subject_ids: Optional[np.ndarray] = None,
    highlight_ids: Optional[list] = None,
    outcome_name: str = "outcome",
    title: Optional[str] = None,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Scatter of per-patient model confidence (embedding locality score)
    vs outcome label, colored by label.

    Higher confidence = patient lies in a dense, consistent neighbourhood.
    Lower confidence  = patient is in a sparse or mixed-label region.

    This is NOT uncertainty calibration — it is a visualization of where
    the model's learned geometry is consistent vs ambiguous.
    """
    try:
        from scipy.stats import gaussian_kde
    except ImportError:
        gaussian_kde = None

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(7.0, 3.5), gridspec_kw={"width_ratios": [2, 1]}
    )

    valid = outcome_labels >= 0
    conf_v = model_confidence[valid]
    labs_v = outcome_labels[valid]

    # Left: scatter of confidence per patient, colored by label
    jitter = np.random.RandomState(42).uniform(-0.12, 0.12, size=valid.sum())
    for val, color, name in [
        (0, PALETTE["negative"], "Negative"),
        (1, PALETTE["positive"], "Positive"),
    ]:
        m = labs_v == val
        ax1.scatter(
            conf_v[m],
            jitter[m] + val,
            c=color,
            s=10,
            alpha=0.4,
            rasterized=True,
            linewidths=0,
            label=f"{name}  (n={m.sum()})",
        )

    ax1.set_xlabel("Embedding confidence  (locality score)", fontsize=LABEL_SIZE)
    ax1.set_yticks([0, 1])
    ax1.set_yticklabels(["Negative", "Positive"])
    ax1.set_title("Confidence by outcome")
    style_legend(ax1.legend(fontsize=LEGEND_SIZE))
    despine(ax1, "x")

    # Right: KDE of confidence distributions
    x_grid = np.linspace(conf_v.min(), conf_v.max(), 200)
    for val, color, name in [
        (0, PALETTE["negative"], "Negative"),
        (1, PALETTE["positive"], "Positive"),
    ]:
        m = labs_v == val
        arr = conf_v[m]
        if len(arr) < 5:
            continue
        if gaussian_kde is not None:
            kde = gaussian_kde(arr, bw_method=0.2)
            density = kde(x_grid)
            ax2.plot(x_grid, density, color=color, lw=1.8, label=name)
            ax2.fill_between(x_grid, density, alpha=0.15, color=color)
        else:
            ax2.hist(arr, bins=30, density=True, alpha=0.5, color=color, label=name)

    ax2.set_xlabel("Embedding confidence")
    ax2.set_ylabel("Density")
    ax2.set_title("Confidence distribution")
    style_legend(ax2.legend(fontsize=LEGEND_SIZE))
    despine(ax2, "y")

    add_panel_label(ax1, "A")
    add_panel_label(ax2, "B", x=-0.3)
    fig.suptitle(
        title or f"Embedding confidence — {outcome_name.replace('_', ' ')}",
        fontsize=SUPTITLE_SIZE,
        fontweight="semibold",
        color=PALETTE["ink"],
    )
    fig.tight_layout()
    fig.subplots_adjust(wspace=0.55)
    save_fig(fig, save_path)
    return fig


# ═════════════════════════════════════════════════════════════════════
# 6. Added-value comparison (embedding vs IPI vs combined)
# ═════════════════════════════════════════════════════════════════════


def plot_added_value_comparison(
    results_df: pd.DataFrame,
    title: str = "Added value of foundation model over IPI",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Dot chart: IPI-only vs embedding-only vs combined AUROC, per outcome.
    Delta annotations show the gain from adding embeddings to IPI.
    """
    outcomes = results_df["outcome"].tolist()
    n = len(outcomes)
    display = [o.replace("_", " ").title() for o in outcomes]

    fig, ax = plt.subplots(figsize=(5.5, max(3.0, 0.55 * n)))
    y = np.arange(n)

    for col, color, label, marker in [
        ("rkkp_only_auroc", PALETTE["ipi"], "IPI only", "D"),
        ("embedding_only_auroc", PALETTE["opera"], "Embedding only", "o"),
        ("combined_auroc", PALETTE["opera_joint"], "Combined", "s"),
    ]:
        if col not in results_df.columns:
            continue
        ax.scatter(
            results_df[col],
            y,
            color=color,
            s=55,
            marker=marker,
            label=label,
            zorder=3,
            linewidths=0,
        )

    # Delta annotations
    if (
        "combined_auroc" in results_df.columns
        and "rkkp_only_auroc" in results_df.columns
    ):
        for i, row in results_df.iterrows():
            delta = row["combined_auroc"] - row["rkkp_only_auroc"]
            color = PALETTE["opera_joint"] if delta >= 0 else PALETTE["positive"]
            ax.annotate(
                f"+{delta:.3f}" if delta >= 0 else f"{delta:.3f}",
                xy=(row["combined_auroc"], i),
                xytext=(8, 0),
                textcoords="offset points",
                va="center",
                ha="left",
                fontsize=ANNOT_SIZE,
                color=color,
                fontweight="bold",
            )

    ax.set_yticks(y)
    ax.set_yticklabels(display)
    ax.set_xlabel("AUROC")
    ax.set_title(title)
    style_legend(ax.legend(fontsize=LEGEND_SIZE, loc="center left"))
    ax.set_xlim(0.35, None)
    despine(ax, "none")
    ax.grid(axis="x", color=PALETTE["grid"], linewidth=0.5)
    ax.grid(axis="y", visible=False)

    fig.tight_layout()
    save_fig(fig, save_path)
    return fig


# ═════════════════════════════════════════════════════════════════════
# 7. Performance landscape  (smoothed accuracy over embedding space)
# ═════════════════════════════════════════════════════════════════════


def plot_performance_landscape(
    embeddings: np.ndarray,
    labels: np.ndarray,
    probabilities: np.ndarray,
    method: str = "umap",
    metric: str = "accuracy",
    grid_resolution: int = 80,
    smoothing_bandwidth: float = 0.08,
    min_patients_per_cell: int = 5,
    outcome_name: str = "outcome",
    title: Optional[str] = None,
    save_path: Optional[str] = None,
    coords: Optional[np.ndarray] = None,
    **reducer_kwargs,
) -> plt.Figure:
    """
    Smoothed performance landscape over the 2D embedding space.

    The embedding space is parsed into a regular grid. For each grid cell,
    a local performance metric is computed from the patients in that region.
    A Gaussian smoothing kernel is applied across the grid, producing a
    continuous surface that reveals where the model excels and where it
    struggles. Sparse cells (< min_patients_per_cell) are masked.

    This answers: "Do errors cluster in specific parts of the embedding space,
    or are they distributed randomly?" Spatial structure in errors is
    scientifically informative — it means identifiable patient subgroups
    that the model handles poorly.

    Parameters
    ----------
    embeddings : (N, D) — high-dimensional embeddings
    labels : (N,) — binary ground truth (0/1, or -1 for missing)
    probabilities : (N,) — predicted probabilities in [0, 1]
    method : "umap" or "tsne"
    metric : one of:
        "accuracy"     — fraction correctly classified (threshold = 0.5)
        "error"        — fraction incorrectly classified (= 1 − accuracy)
        "mean_prob"    — mean predicted probability (calibration geography)
        "brier"        — mean squared error (probability − label)²
        "signed_error" — mean (prob − label): pos = overconfident positive,
                         neg = overconfident negative
    grid_resolution : number of grid cells per axis (80 → 80×80 = 6400 cells)
    smoothing_bandwidth : Gaussian smoothing σ as a fraction of the axis range.
                          0.08 gives smooth surfaces while preserving structure.
    min_patients_per_cell : cells with fewer patients are masked (shown as grey).
    coords : pre-computed 2D coordinates (N, 2). Pass these to avoid re-running
             UMAP when layering multiple outcomes on the same projection.
    """
    from scipy.ndimage import gaussian_filter
    from opera.visualization.embedding_plots import _reduce_embeddings

    if coords is None:
        coords = _reduce_embeddings(embeddings, method, **reducer_kwargs)

    valid = labels >= 0
    coords_v = coords[valid]
    labels_v = labels[valid].astype(int)
    probs_v = probabilities[valid]

    # ── Build grid ───────────────────────────────────────────────────────
    pad_frac = 0.05
    x_min, x_max = coords_v[:, 0].min(), coords_v[:, 0].max()
    y_min, y_max = coords_v[:, 1].min(), coords_v[:, 1].max()
    x_edges = np.linspace(
        x_min - pad_frac * (x_max - x_min),
        x_max + pad_frac * (x_max - x_min),
        grid_resolution + 1,
    )
    y_edges = np.linspace(
        y_min - pad_frac * (y_max - y_min),
        y_max + pad_frac * (y_max - y_min),
        grid_resolution + 1,
    )

    x_idx = np.clip(np.digitize(coords_v[:, 0], x_edges) - 1, 0, grid_resolution - 1)
    y_idx = np.clip(np.digitize(coords_v[:, 1], y_edges) - 1, 0, grid_resolution - 1)

    # ── Compute metric per cell ────────────────────────────────────────────
    metric_grid = np.full((grid_resolution, grid_resolution), np.nan)
    count_grid = np.zeros((grid_resolution, grid_resolution), dtype=int)

    for xi in range(grid_resolution):
        for yi in range(grid_resolution):
            cell = (x_idx == xi) & (y_idx == yi)
            n_cell = cell.sum()
            count_grid[xi, yi] = n_cell
            if n_cell < min_patients_per_cell:
                continue
            l_c = labels_v[cell]
            p_c = probs_v[cell]
            if metric == "accuracy":
                metric_grid[xi, yi] = ((p_c >= 0.5).astype(int) == l_c).mean()
            elif metric == "error":
                metric_grid[xi, yi] = ((p_c >= 0.5).astype(int) != l_c).mean()
            elif metric == "mean_prob":
                metric_grid[xi, yi] = p_c.mean()
            elif metric == "brier":
                metric_grid[xi, yi] = ((p_c - l_c) ** 2).mean()
            elif metric == "signed_error":
                metric_grid[xi, yi] = (p_c - l_c).mean()
            else:
                raise ValueError(f"Unknown metric: {metric!r}")

    # ── Gaussian smoothing ────────────────────────────────────────────────
    sigma_px = smoothing_bandwidth * grid_resolution
    filled = metric_grid.copy()
    global_mean = float(np.nanmean(filled)) if not np.all(np.isnan(filled)) else 0.0
    filled[np.isnan(filled)] = global_mean
    smoothed = gaussian_filter(filled, sigma=sigma_px)
    smoothed[count_grid < min_patients_per_cell] = np.nan

    # ── Colormap config ────────────────────────────────────────────────────
    if metric == "accuracy":
        cmap, vmin, vmax = "RdYlGn", 0.3, 1.0
        cbar_label = "Local accuracy"
        ref_line = 0.5
    elif metric == "error":
        cmap, vmin, vmax = "RdYlGn_r", 0.0, 0.7
        cbar_label = "Local error rate"
        ref_line = 0.5
    elif metric == "mean_prob":
        cmap, vmin, vmax = "RdBu_r", 0.0, 1.0
        cbar_label = "Mean predicted probability"
        ref_line = None
    elif metric == "brier":
        cmap, vmin, vmax = "YlOrRd", 0.0, 0.5
        cbar_label = "Local Brier score"
        ref_line = None
    elif metric == "signed_error":
        obs = smoothed[~np.isnan(smoothed)]
        lim = max(0.2, float(np.nanpercentile(np.abs(obs), 95))) if len(obs) else 0.3
        cmap, vmin, vmax = "RdBu_r", -lim, lim
        cbar_label = "Signed error  (prob − label)\n+ overconfident positive   − overconfident negative"
        ref_line = 0.0
    else:
        cmap, vmin, vmax = sequential_cmap("opera"), None, None
        cbar_label = metric
        ref_line = None

    # ── Figure ────────────────────────────────────────────────────────────
    fig, (ax_map, ax_dist) = plt.subplots(
        1,
        2,
        figsize=(8.5, 4.5),
        gridspec_kw={"width_ratios": [3, 1]},
    )

    extent = [x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]]
    im = ax_map.imshow(
        smoothed.T,
        origin="lower",
        extent=extent,
        aspect="auto",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        interpolation="bilinear",
        alpha=0.88,
        zorder=1,
    )

    # Patient scatter — correct (white) vs incorrect (dark) dots
    preds_v = (probs_v >= 0.5).astype(int)
    correct = preds_v == labels_v
    ax_map.scatter(
        coords_v[correct, 0],
        coords_v[correct, 1],
        s=3,
        c="white",
        alpha=0.22,
        linewidths=0,
        rasterized=True,
        zorder=2,
    )
    ax_map.scatter(
        coords_v[~correct, 0],
        coords_v[~correct, 1],
        s=4,
        c=PALETTE["ink"],
        alpha=0.28,
        linewidths=0,
        rasterized=True,
        zorder=3,
    )

    from matplotlib.lines import Line2D

    style_legend(
        ax_map.legend(
            handles=[
                Line2D(
                    [0],
                    [0],
                    marker="o",
                    color="none",
                    markerfacecolor=PALETTE["ink_muted"],
                    markersize=5,
                    label=f"Correct  (n={correct.sum()})",
                ),
                Line2D(
                    [0],
                    [0],
                    marker="o",
                    color="none",
                    markerfacecolor=PALETTE["ink"],
                    markersize=5,
                    label=f"Incorrect  (n={(~correct).sum()})",
                ),
            ],
            loc="lower right",
            fontsize=NOTE_SIZE,
            framealpha=0.85,
        )
    )

    cbar = fig.colorbar(im, ax=ax_map, shrink=0.85, pad=0.02)
    cbar.set_label(cbar_label, fontsize=NOTE_SIZE, color=PALETTE["ink_secondary"])
    cbar.ax.tick_params(labelsize=NOTE_SIZE, colors=PALETTE["ink_secondary"])
    cbar.outline.set_edgecolor(PALETTE["panel_border"])
    if ref_line is not None and vmin is not None and vmin < ref_line < vmax:
        norm_pos = (ref_line - vmin) / (vmax - vmin)
        cbar.ax.axhline(norm_pos, color=PALETTE["zero_line"], lw=1.0, ls="--")

    ax_map.set_xlabel(f"{method.upper()} 1", fontsize=LABEL_SIZE)
    ax_map.set_ylabel(f"{method.upper()} 2", fontsize=LABEL_SIZE)
    ax_map.set_title(
        title or f"Performance landscape — {outcome_name.replace('_', ' ')}",
        fontsize=TITLE_SIZE,
        fontweight="semibold",
        color=PALETTE["ink"],
    )
    ax_map.set_xticks([])
    ax_map.set_yticks([])
    ax_map.spines["left"].set_visible(False)
    ax_map.spines["bottom"].set_visible(False)
    ax_map.grid(False)

    # ── Right panel: per-patient distribution by label ────────────────────
    if metric == "accuracy":
        per_pt = (preds_v == labels_v).astype(float)
        dist_label = "Correctly classified"
    elif metric == "error":
        per_pt = (preds_v != labels_v).astype(float)
        dist_label = "Error"
    elif metric == "mean_prob":
        per_pt = probs_v
        dist_label = "Predicted probability"
    elif metric == "brier":
        per_pt = (probs_v - labels_v) ** 2
        dist_label = "Squared error"
    elif metric == "signed_error":
        per_pt = probs_v - labels_v
        dist_label = "Signed error"
    else:
        per_pt = probs_v
        dist_label = metric

    try:
        from scipy.stats import gaussian_kde as _gkde

        x_d = np.linspace(per_pt.min() - 0.05, per_pt.max() + 0.05, 250)
        for val, color, name in [
            (0, PALETTE["negative"], "Negative"),
            (1, PALETTE["positive"], "Positive"),
        ]:
            m = labels_v == val
            if m.sum() < 5:
                continue
            arr = per_pt[m]
            bw = max(0.04, 1.06 * arr.std() * len(arr) ** (-0.2))
            kde_fn = _gkde(arr, bw_method=bw / arr.std() if arr.std() > 0 else 0.15)
            density = kde_fn(x_d)
            ax_dist.plot(density, x_d, color=color, lw=1.8, label=name)
            ax_dist.fill_betweenx(x_d, density, alpha=0.15, color=color)
    except ImportError:
        for val, color, name in [
            (0, PALETTE["negative"], "Negative"),
            (1, PALETTE["positive"], "Positive"),
        ]:
            m = labels_v == val
            ax_dist.hist(
                per_pt[m],
                bins=20,
                density=True,
                alpha=0.5,
                color=color,
                label=name,
                orientation="horizontal",
            )

    ax_dist.set_xlabel("Density", fontsize=LABEL_SIZE)
    ax_dist.set_ylabel(dist_label, fontsize=LABEL_SIZE)
    ax_dist.set_title(
        "By outcome label",
        fontsize=TITLE_SIZE,
        fontweight="semibold",
        color=PALETTE["ink"],
    )
    style_legend(ax_dist.legend(fontsize=LEGEND_SIZE))
    ax_dist.spines["top"].set_visible(False)
    ax_dist.spines["right"].set_visible(False)
    ax_dist.grid(False)

    add_panel_label(ax_map, "A")
    add_panel_label(ax_dist, "B")
    fig.tight_layout(w_pad=1.5)
    save_fig(fig, save_path)
    return fig


def plot_performance_landscape_multi_outcome(
    embeddings: np.ndarray,
    outcome_data: Dict[str, Dict],
    method: str = "umap",
    metric: str = "signed_error",
    grid_resolution: int = 70,
    smoothing_bandwidth: float = 0.09,
    min_patients_per_cell: int = 5,
    save_path: Optional[str] = None,
    **reducer_kwargs,
) -> plt.Figure:
    """
    Grid of performance landscape panels, one per outcome, sharing the same
    2D projection coordinates.

    Shared coordinates are critical — an error cluster appearing in the same
    spatial location across multiple outcomes strongly suggests a patient
    subgroup the model systematically misrepresents.

    Parameters
    ----------
    outcome_data : dict  outcome_name → {"labels": (N,), "probabilities": (N,)}
    metric : "signed_error" recommended — it shows direction of error, not
             just magnitude, which is more interpretable across outcomes.
    """
    from scipy.ndimage import gaussian_filter
    from opera.visualization.embedding_plots import _reduce_embeddings

    coords = _reduce_embeddings(embeddings, method, **reducer_kwargs)

    # Use the global embedding extent across all outcomes
    x_min, x_max = coords[:, 0].min(), coords[:, 0].max()
    y_min, y_max = coords[:, 1].min(), coords[:, 1].max()
    pad = 0.04
    x_edges = np.linspace(
        x_min - pad * (x_max - x_min),
        x_max + pad * (x_max - x_min),
        grid_resolution + 1,
    )
    y_edges = np.linspace(
        y_min - pad * (y_max - y_min),
        y_max + pad * (y_max - y_min),
        grid_resolution + 1,
    )

    names = sorted(outcome_data.keys())
    n = len(names)
    cols = min(n, 3)
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(4.0 * cols, 3.8 * rows))
    if n == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    for i, name in enumerate(names):
        ax = axes[i]
        d = outcome_data[name]
        labels_n = np.array(d["labels"])
        probs_n = np.array(d["probabilities"])
        valid = labels_n >= 0

        coords_v = coords[valid]
        labels_v = labels_n[valid].astype(int)
        probs_v = probs_n[valid]

        x_idx = np.clip(
            np.digitize(coords_v[:, 0], x_edges) - 1, 0, grid_resolution - 1
        )
        y_idx = np.clip(
            np.digitize(coords_v[:, 1], y_edges) - 1, 0, grid_resolution - 1
        )

        metric_grid = np.full((grid_resolution, grid_resolution), np.nan)
        count_grid = np.zeros((grid_resolution, grid_resolution), dtype=int)

        for xi in range(grid_resolution):
            for yi in range(grid_resolution):
                cell = (x_idx == xi) & (y_idx == yi)
                n_c = cell.sum()
                count_grid[xi, yi] = n_c
                if n_c < min_patients_per_cell:
                    continue
                l_c, p_c = labels_v[cell], probs_v[cell]
                if metric == "signed_error":
                    metric_grid[xi, yi] = (p_c - l_c).mean()
                elif metric == "accuracy":
                    metric_grid[xi, yi] = ((p_c >= 0.5) == l_c).mean()
                elif metric == "brier":
                    metric_grid[xi, yi] = ((p_c - l_c) ** 2).mean()

        sigma_px = smoothing_bandwidth * grid_resolution
        filled = metric_grid.copy()
        gm = float(np.nanmean(filled)) if not np.all(np.isnan(filled)) else 0.0
        filled[np.isnan(filled)] = gm
        smoothed = gaussian_filter(filled, sigma=sigma_px)
        smoothed[count_grid < min_patients_per_cell] = np.nan

        obs = smoothed[~np.isnan(smoothed)]
        if metric == "signed_error":
            lim = (
                max(0.2, float(np.nanpercentile(np.abs(obs), 95))) if len(obs) else 0.3
            )
            cmap, vmin, vmax = "RdBu_r", -lim, lim
        elif metric == "accuracy":
            cmap, vmin, vmax = "RdYlGn", 0.3, 1.0
        else:
            cmap, vmin, vmax = "YlOrRd", 0.0, 0.5

        extent = [x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]]
        im = ax.imshow(
            smoothed.T,
            origin="lower",
            extent=extent,
            aspect="auto",
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            interpolation="bilinear",
            alpha=0.88,
            zorder=1,
        )

        ax.scatter(
            coords_v[:, 0],
            coords_v[:, 1],
            s=2,
            c=PALETTE["ink_secondary"],
            alpha=0.10,
            linewidths=0,
            rasterized=True,
            zorder=2,
        )

        ax.set_title(
            name.replace("_", " ").title(),
            fontsize=TITLE_SIZE,
            fontweight="semibold",
            color=PALETTE["ink"],
        )
        ax.set_xticks([])
        ax.set_yticks([])
        ax.spines["left"].set_visible(False)
        ax.spines["bottom"].set_visible(False)
        ax.grid(False)

        cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
        cbar.ax.tick_params(labelsize=NOTE_SIZE, colors=PALETTE["ink_secondary"])
        cbar.outline.set_edgecolor(PALETTE["panel_border"])

    for j in range(n, len(axes)):
        axes[j].set_visible(False)

    metric_display = {
        "signed_error": "Signed error  (red = overconfident positive  /  blue = overconfident negative)",
        "accuracy": "Local accuracy  (green = high  /  red = low)",
        "brier": "Local Brier score  (yellow = low  /  red = high)",
    }.get(metric, metric)

    fig.suptitle(
        f"Performance landscape — {metric_display}",
        fontsize=SUPTITLE_SIZE,
        fontweight="semibold",
        color=PALETTE["ink"],
        y=1.01,
    )
    fig.tight_layout()
    save_fig(fig, save_path)
    return fig


# ═════════════════════════════════════════════════════════════════════
# 7. Embedding atlas  (the clinical map)
# ═════════════════════════════════════════════════════════════════════


def _compute_outcome_axes(
    coords_2d: np.ndarray,
    outcome_probabilities: Dict[str, np.ndarray],
    sigma_values: Optional[Dict[str, float]] = None,
    min_r2: float = 0.02,
) -> List[Dict]:
    """
    For each outcome, fit a linear regression from 2D UMAP coordinates to
    predicted probability.  The regression vector gives the direction in
    UMAP space along which that outcome's risk increases most rapidly.

    Scale each arrow by R² × (1/σ):
    - R²   measures how well spatial position predicts risk (structural fit)
    - 1/σ  measures how strongly this outcome organises the embedding space

    Outcomes where position barely predicts risk (low R²) or where the model
    learned little structure (high σ) produce negligibly short arrows and are
    excluded (filtered by min_r2).

    Returns list of dicts, one per outcome, sorted by arrow magnitude:
        {"name", "direction": (dx, dy) unit vector,
         "magnitude": R² × (1/σ),
         "r2": float, "sigma": float, "origin": (cx, cy) centroid}
    """
    results = []
    cx = float(coords_2d[:, 0].mean())
    cy = float(coords_2d[:, 1].mean())

    for name, probs in outcome_probabilities.items():
        valid = np.isfinite(probs)
        if valid.sum() < 20:
            continue

        x = coords_2d[valid, 0]
        y = coords_2d[valid, 1]
        p = probs[valid]

        # OLS: [x, y, 1] → p
        X = np.column_stack([x, y, np.ones(len(x))])
        try:
            coeffs, _, _, _ = np.linalg.lstsq(X, p, rcond=None)
        except np.linalg.LinAlgError:
            continue

        beta = coeffs[:2]  # [β_x, β_y]
        p_hat = X @ coeffs
        ss_res = ((p - p_hat) ** 2).sum()
        ss_tot = ((p - p.mean()) ** 2).sum()
        r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else 0.0

        if r2 < min_r2:
            continue

        sigma = float(sigma_values.get(name, 1.0)) if sigma_values else 1.0
        magnitude = r2 * (1.0 / max(sigma, 0.1))

        norm = float(np.linalg.norm(beta))
        if norm < 1e-8:
            continue
        direction = tuple(beta / norm)

        results.append(
            {
                "name": name,
                "direction": direction,
                "magnitude": magnitude,
                "r2": r2,
                "sigma": sigma,
                "origin": (cx, cy),
            }
        )

    results.sort(key=lambda d: d["magnitude"], reverse=True)
    return results


def plot_embedding_atlas(
    embeddings: np.ndarray,
    outcome_probabilities: Dict[str, np.ndarray],
    outcome_labels: Optional[Dict[str, np.ndarray]] = None,
    population_groups: Optional[Dict[str, np.ndarray]] = None,
    sigma_values: Optional[Dict[str, float]] = None,
    method: str = "umap",
    background_metric: str = "signed_error",
    background_outcome: Optional[str] = None,
    background_labels: Optional[np.ndarray] = None,
    max_axes: int = 6,
    min_r2: float = 0.03,
    grid_resolution: int = 70,
    smoothing_bandwidth: float = 0.10,
    min_patients_per_cell: int = 8,
    title: str = "Embedding Atlas",
    save_path: Optional[str] = None,
    coords: Optional[np.ndarray] = None,
    **reducer_kwargs,
) -> plt.Figure:
    """
    The clinical map of the embedding space.

    Layers (bottom to top):
    1. Performance landscape background — smoothed accuracy or signed error for
       the most structured outcome (lowest σ), giving the map a meaningful
       topography.
    2. Population density halos — if cohort/disease labels are provided, each
       group gets a soft density contour + centroid label.  This shows where
       different patient populations live on the map.
    3. Outcome risk axes — arrows showing the direction of increasing risk for
       each outcome, scaled by R² × (1/σ).  These are the "compass" of the map:
       moving along an axis raises that outcome's predicted risk.  Outcomes where
       spatial position strongly predicts risk (low σ, high R²) produce long,
       prominent arrows.

    Reading the atlas:
    - Red region + long mortality arrow pointing here → high-risk cluster
    - Blue region → where the model is confident and accurate
    - Disease label in red region → that population is harder to predict
    - A new patient placed here can be interpreted: "near the CLL cluster,
      in a region where mortality risk is high and model accuracy is moderate"

    Parameters
    ----------
    outcome_probabilities : dict  outcome_name → (N,) predicted probabilities
    outcome_labels : dict  outcome_name → (N,) binary labels (0/1/-1=missing)
    population_groups : dict  group_name → boolean (N,) mask of group membership
        e.g. {"DLBCL": dlbcl_mask, "CLL": cll_mask, "MDS": mds_mask}
    sigma_values : dict  outcome_name → σ from the trained contrastive model.
        If provided, axes are weighted by 1/σ.
    background_metric : "signed_error" or "accuracy" or "mean_prob"
    background_outcome : which outcome to use for the background landscape.
        If None, uses the outcome with lowest σ (most structured).
    max_axes : maximum number of outcome axes to draw (top-ranked by R²/σ).
    coords : pre-computed 2D UMAP coordinates — pass to avoid re-running UMAP.
    """
    from scipy.ndimage import gaussian_filter
    from opera.visualization.embedding_plots import _reduce_embeddings, _density_contour

    # ── Project to 2D ──────────────────────────────────────────────────────
    if coords is None:
        coords = _reduce_embeddings(embeddings, method, **reducer_kwargs)

    # ── Choose background outcome ──────────────────────────────────────────
    if background_outcome is None and sigma_values:
        # Pick the outcome with lowest σ that also has labels
        ranked = sorted(sigma_values.items(), key=lambda kv: kv[1])
        for name, _ in ranked:
            if outcome_labels and name in outcome_labels:
                background_outcome = name
                break
        if background_outcome is None:
            background_outcome = next(iter(outcome_probabilities))
    elif background_outcome is None:
        background_outcome = next(iter(outcome_probabilities))

    # ── Compute background landscape ────────────────────────────────────────
    bg_probs = outcome_probabilities.get(background_outcome)
    bg_labels = (
        (outcome_labels or {}).get(background_outcome)
        if background_labels is None
        else background_labels
    )

    smoothed, x_edges, y_edges, vmin, vmax, cmap = [None] * 6
    if bg_probs is not None and bg_labels is not None:
        valid_bg = bg_labels >= 0
        coords_bg = coords[valid_bg]
        labels_bg = bg_labels[valid_bg].astype(int)
        probs_bg = bg_probs[valid_bg]

        pad = 0.05
        x_min, x_max = coords[:, 0].min(), coords[:, 0].max()
        y_min, y_max = coords[:, 1].min(), coords[:, 1].max()
        x_edges = np.linspace(
            x_min - pad * (x_max - x_min),
            x_max + pad * (x_max - x_min),
            grid_resolution + 1,
        )
        y_edges = np.linspace(
            y_min - pad * (y_max - y_min),
            y_max + pad * (y_max - y_min),
            grid_resolution + 1,
        )

        x_idx = np.clip(
            np.digitize(coords_bg[:, 0], x_edges) - 1, 0, grid_resolution - 1
        )
        y_idx = np.clip(
            np.digitize(coords_bg[:, 1], y_edges) - 1, 0, grid_resolution - 1
        )

        mg = np.full((grid_resolution, grid_resolution), np.nan)
        cnt = np.zeros((grid_resolution, grid_resolution), dtype=int)
        for xi in range(grid_resolution):
            for yi in range(grid_resolution):
                cell = (x_idx == xi) & (y_idx == yi)
                cnt[xi, yi] = cell.sum()
                if cnt[xi, yi] < min_patients_per_cell:
                    continue
                l_c, p_c = labels_bg[cell], probs_bg[cell]
                if background_metric == "signed_error":
                    mg[xi, yi] = (p_c - l_c).mean()
                elif background_metric == "accuracy":
                    mg[xi, yi] = ((p_c >= 0.5) == l_c).mean()
                elif background_metric == "mean_prob":
                    mg[xi, yi] = p_c.mean()

        sigma_px = smoothing_bandwidth * grid_resolution
        filled = mg.copy()
        gm = float(np.nanmean(filled)) if not np.all(np.isnan(filled)) else 0.0
        filled[np.isnan(filled)] = gm
        smoothed = gaussian_filter(filled, sigma=sigma_px)
        smoothed[cnt < min_patients_per_cell] = np.nan

        if background_metric == "signed_error":
            obs = smoothed[~np.isnan(smoothed)]
            lim = (
                max(0.2, float(np.nanpercentile(np.abs(obs), 95))) if len(obs) else 0.3
            )
            cmap, vmin, vmax = "RdBu_r", -lim, lim
        elif background_metric == "accuracy":
            cmap, vmin, vmax = "RdYlGn", 0.3, 1.0
        elif background_metric == "mean_prob":
            cmap, vmin, vmax = "RdBu_r", 0.0, 1.0

    # ── Figure ─────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7.0, 6.2))

    # Background landscape
    if smoothed is not None:
        extent = [x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]]
        im = ax.imshow(
            smoothed.T,
            origin="lower",
            extent=extent,
            aspect="auto",
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            interpolation="bilinear",
            alpha=0.75,
            zorder=1,
        )
        cbar = fig.colorbar(im, ax=ax, shrink=0.6, pad=0.01, aspect=25)
        metric_label = {
            "signed_error": f"Signed error ({background_outcome.replace('_', ' ')})",
            "accuracy": f"Accuracy ({background_outcome.replace('_', ' ')})",
            "mean_prob": f"Mean risk ({background_outcome.replace('_', ' ')})",
        }.get(background_metric, background_metric)
        cbar.set_label(metric_label, fontsize=NOTE_SIZE, color=PALETTE["ink_secondary"])
        cbar.ax.tick_params(labelsize=NOTE_SIZE, colors=PALETTE["ink_secondary"])
        cbar.outline.set_edgecolor(PALETTE["panel_border"])
    else:
        # Neutral grey patient scatter as background
        ax.scatter(
            coords[:, 0],
            coords[:, 1],
            s=3,
            c=PALETTE["missing"],
            alpha=0.25,
            linewidths=0,
            rasterized=True,
            zorder=1,
        )

    # ── Population group halos ─────────────────────────────────────────────
    if population_groups:
        group_colors = CATEGORICAL[: len(population_groups)]
        for (group_name, mask), color in zip(population_groups.items(), group_colors):
            if mask.sum() < 10:
                continue
            gc = coords[mask]
            # Density contour
            _density_contour(ax, gc[:, 0], gc[:, 1], color, levels=4, alpha=0.55)
            # Centroid label with background box
            cx_g, cy_g = float(gc[:, 0].mean()), float(gc[:, 1].mean())
            ax.text(
                cx_g,
                cy_g,
                group_name,
                ha="center",
                va="center",
                fontsize=ANNOT_SIZE,
                fontweight="bold",
                color=color,
                zorder=6,
                bbox=dict(
                    boxstyle="round,pad=0.25",
                    facecolor="white",
                    edgecolor=color,
                    alpha=0.85,
                    linewidth=1.2,
                ),
            )

    # ── Outcome risk axes ──────────────────────────────────────────────────
    axes_data = _compute_outcome_axes(
        coords,
        outcome_probabilities,
        sigma_values=sigma_values,
        min_r2=min_r2,
    )
    axes_to_draw = axes_data[:max_axes]

    if axes_to_draw:
        # Compute a sensible arrow length: fraction of the plot extent
        x_range = coords[:, 0].max() - coords[:, 0].min()
        y_range = coords[:, 1].max() - coords[:, 1].min()
        base_len = 0.18 * max(x_range, y_range)

        # Normalize magnitudes to [0.5, 1.0] relative scale
        mags = np.array([d["magnitude"] for d in axes_to_draw])
        mag_min, mag_max = mags.min(), mags.max()
        if mag_max > mag_min:
            mags_norm = 0.5 + 0.5 * (mags - mag_min) / (mag_max - mag_min)
        else:
            mags_norm = np.ones_like(mags)

        # Arrow origin: center of all points (arrows radiate from the heart of the map)
        origin_x = float(coords[:, 0].mean())
        origin_y = float(coords[:, 1].mean())

        ax_colors = CATEGORICAL[: len(axes_to_draw)]
        for d, scale, color in zip(axes_to_draw, mags_norm, ax_colors):
            dx = d["direction"][0] * base_len * scale
            dy = d["direction"][1] * base_len * scale

            ax.annotate(
                "",
                xy=(origin_x + dx, origin_y + dy),
                xytext=(origin_x, origin_y),
                arrowprops=dict(
                    arrowstyle="-|>",
                    color=color,
                    lw=2.0,
                    mutation_scale=14,
                ),
                zorder=5,
            )

            # Label at arrow tip — slightly beyond the tip to avoid overlap
            label_x = origin_x + dx * 1.18
            label_y = origin_y + dy * 1.18
            display = d["name"].replace("_", " ").title()
            r2_str = f"R²={d['r2']:.2f}"
            if sigma_values:
                r2_str += f"  σ={d['sigma']:.2f}"
            ax.text(
                label_x,
                label_y,
                f"{display}\n{r2_str}",
                ha="center",
                va="center",
                fontsize=NOTE_SIZE,
                color=color,
                fontweight="semibold",
                zorder=7,
                bbox=dict(
                    boxstyle="round,pad=0.2",
                    facecolor="white",
                    edgecolor=color,
                    alpha=0.80,
                    linewidth=0.8,
                ),
            )

        # Origin marker
        ax.scatter(
            [origin_x], [origin_y], s=30, c=PALETTE["zero_line"], zorder=6, linewidths=0
        )

    ax.set_xlabel(f"{method.upper()} 1", fontsize=LABEL_SIZE)
    ax.set_ylabel(f"{method.upper()} 2", fontsize=LABEL_SIZE)
    ax.set_title(
        title, fontsize=TITLE_SIZE, fontweight="semibold", color=PALETTE["ink"]
    )
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.grid(False)

    # Legend for axes — placed below the map (not inside plot bounds) so it
    # never collides with population-group labels/contours, which are
    # positioned at data-dependent centroids and can land in any corner.
    if axes_to_draw:
        from matplotlib.lines import Line2D

        handles = [
            Line2D(
                [0],
                [0],
                color=color,
                lw=2.0,
                label=f"{d['name'].replace('_', ' ').title()}  "
                f"(R²={d['r2']:.2f}, σ={d['sigma']:.2f})",
            )
            for d, color in zip(axes_to_draw, ax_colors)
        ]
        style_legend(
            ax.legend(
                handles=handles,
                fontsize=LEGEND_SIZE,
                loc="upper center",
                bbox_to_anchor=(0.5, -0.05),
                ncol=2,
                framealpha=0.94,
                title="Outcome axes",
                title_fontsize=LEGEND_TITLE_SIZE,
            )
        )

    fig.tight_layout()
    if axes_to_draw:
        fig.subplots_adjust(bottom=0.24)
    save_fig(fig, save_path)
    return fig


# ═════════════════════════════════════════════════════════════════════
# 8. Landscape summary  (compact per-outcome bar for main figures)
# ═════════════════════════════════════════════════════════════════════


def plot_landscape_summary(
    outcome_data: Dict[str, Dict],
    coords: np.ndarray,
    sigma_values: Optional[Dict[str, float]] = None,
    grid_resolution: int = 60,
    smoothing_bandwidth: float = 0.09,
    min_patients_per_cell: int = 5,
    high_error_threshold: float = 0.35,
    title: str = "Embedding error structure by outcome",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Compact summary figure suitable for the main paper.

    For each outcome, computes the fraction of patients residing in
    "high-error" embedding regions (|signed_error| > threshold) and the
    mean absolute signed error.  Displayed as a horizontal bar chart sorted
    by σ (most structured outcomes first), with σ values annotated.

    This answers: "Which outcomes have the most spatially concentrated errors,
    and are those the same outcomes the model found easiest to learn (low σ)?"
    An outcome with low σ but high spatial error concentration is scientifically
    interesting — the model organized around this signal but still struggles in
    specific patient sub-populations.

    Parameters
    ----------
    outcome_data : dict  outcome_name → {"labels": (N,), "probabilities": (N,)}
    coords : (N, 2) pre-computed UMAP coordinates (shared across all outcomes)
    sigma_values : dict  outcome_name → σ
    high_error_threshold : |signed_error| above this is "high error"
    """
    from scipy.ndimage import gaussian_filter

    names = sorted(outcome_data.keys())
    if sigma_values:
        names = sorted(names, key=lambda n: sigma_values.get(n, 1.0))

    x_min, x_max = coords[:, 0].min(), coords[:, 0].max()
    y_min, y_max = coords[:, 1].min(), coords[:, 1].max()
    pad = 0.04
    x_edges = np.linspace(
        x_min - pad * (x_max - x_min),
        x_max + pad * (x_max - x_min),
        grid_resolution + 1,
    )
    y_edges = np.linspace(
        y_min - pad * (y_max - y_min),
        y_max + pad * (y_max - y_min),
        grid_resolution + 1,
    )

    rows_data = []
    for name in names:
        d = outcome_data[name]
        labels_n = np.array(d["labels"])
        probs_n = np.array(d["probabilities"])
        valid = labels_n >= 0
        if valid.sum() < 20:
            continue

        coords_v = coords[valid]
        labels_v = labels_n[valid].astype(int)
        probs_v = probs_n[valid]

        x_idx = np.clip(
            np.digitize(coords_v[:, 0], x_edges) - 1, 0, grid_resolution - 1
        )
        y_idx = np.clip(
            np.digitize(coords_v[:, 1], y_edges) - 1, 0, grid_resolution - 1
        )

        mg = np.full((grid_resolution, grid_resolution), np.nan)
        cnt = np.zeros((grid_resolution, grid_resolution), dtype=int)
        for xi in range(grid_resolution):
            for yi in range(grid_resolution):
                cell = (x_idx == xi) & (y_idx == yi)
                cnt[xi, yi] = cell.sum()
                if cnt[xi, yi] < min_patients_per_cell:
                    continue
                l_c, p_c = labels_v[cell], probs_v[cell]
                mg[xi, yi] = (p_c - l_c).mean()

        sigma_px = smoothing_bandwidth * grid_resolution
        filled = mg.copy()
        gm = float(np.nanmean(filled)) if not np.all(np.isnan(filled)) else 0.0
        filled[np.isnan(filled)] = gm
        smoothed = gaussian_filter(filled, sigma=sigma_px)
        smoothed[cnt < min_patients_per_cell] = np.nan

        obs = smoothed[~np.isnan(smoothed)]
        if len(obs) == 0:
            continue

        # Patient-level: look up each patient's smoothed cell value
        patient_smoothed = smoothed[x_idx, y_idx]
        has_data = cnt[x_idx, y_idx] >= min_patients_per_cell
        patient_smoothed = patient_smoothed[has_data]

        frac_high_error = float(
            (np.abs(patient_smoothed) > high_error_threshold).mean()
        )
        mean_abs_error = float(np.abs(obs).mean())
        sigma_v = (
            float(sigma_values.get(name, float("nan")))
            if sigma_values
            else float("nan")
        )

        rows_data.append(
            {
                "name": name,
                "frac_high_error": frac_high_error,
                "mean_abs_error": mean_abs_error,
                "sigma": sigma_v,
                "n_patients": int(valid.sum()),
            }
        )

    if not rows_data:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
        return fig

    n_out = len(rows_data)
    display_names = [r["name"].replace("_", " ").title() for r in rows_data]

    fig, (ax_frac, ax_err) = plt.subplots(
        1,
        2,
        figsize=(7.0, max(2.5, 0.45 * n_out)),
        sharey=True,
    )
    y = np.arange(n_out)

    # Left: fraction of patients in high-error regions
    fracs = [r["frac_high_error"] for r in rows_data]
    sigmas = [r["sigma"] for r in rows_data]

    # Color by sigma: low sigma = deep indigo (most structured), high = grey
    if any(np.isfinite(s) for s in sigmas):
        s_arr = np.array([s if np.isfinite(s) else 1.5 for s in sigmas])
        s_norm = (s_arr - s_arr.min()) / max(s_arr.max() - s_arr.min(), 1e-8)
        bar_colors = [plt.cm.Blues_r(0.3 + 0.5 * v) for v in s_norm]
    else:
        bar_colors = [PALETTE["opera"]] * n_out

    bars = ax_frac.barh(y, fracs, color=bar_colors, height=0.6, zorder=2)
    ax_frac.axvline(0, color=PALETTE["zero_line"], lw=0.8, zorder=1)

    for i, (bar, r) in enumerate(zip(bars, rows_data)):
        # Sigma annotation at right end of bar
        s_str = f"σ={r['sigma']:.2f}" if np.isfinite(r["sigma"]) else ""
        ax_frac.text(
            bar.get_width() + 0.01,
            i,
            s_str,
            va="center",
            fontsize=ANNOT_SIZE,
            color=PALETTE["ink_secondary"],
        )

    ax_frac.set_yticks(y)
    ax_frac.set_yticklabels(display_names)
    ax_frac.set_xlabel(
        f"Fraction in high-error region  (|err| > {high_error_threshold})"
    )
    ax_frac.set_title("Spatial error concentration")
    ax_frac.set_xlim(0, min(1.0, max(fracs) * 1.45) if fracs else 1.0)
    despine(ax_frac, "none")
    ax_frac.grid(axis="x", color=PALETTE["grid"], linewidth=0.5)
    ax_frac.grid(axis="y", visible=False)

    # Right: mean absolute error in embedding space
    errs = [r["mean_abs_error"] for r in rows_data]
    ax_err.barh(y, errs, color=bar_colors, height=0.6, zorder=2)
    ax_err.axvline(0, color=PALETTE["zero_line"], lw=0.8, zorder=1)
    ax_err.set_xlabel("Mean |signed error| in embedding space")
    ax_err.set_title("Mean error magnitude")
    ax_err.set_xlim(0, max(errs) * 1.3 if errs else 0.5)
    despine(ax_err, "none")
    ax_err.grid(axis="x", color=PALETTE["grid"], linewidth=0.5)
    ax_err.grid(axis="y", visible=False)

    # Colorbar legend for sigma
    if any(np.isfinite(s) for s in sigmas):
        import matplotlib.cm as cm

        sm = plt.cm.ScalarMappable(
            cmap=cm.Blues_r,
            norm=plt.Normalize(
                vmin=min(s for s in sigmas if np.isfinite(s)),
                vmax=max(s for s in sigmas if np.isfinite(s)),
            ),
        )
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=[ax_frac, ax_err], shrink=0.6, pad=0.01, aspect=20)
        cbar.set_label(
            "σ  (low = strongly structured)",
            fontsize=NOTE_SIZE,
            color=PALETTE["ink_secondary"],
        )
        cbar.ax.tick_params(labelsize=NOTE_SIZE, colors=PALETTE["ink_secondary"])
        cbar.outline.set_edgecolor(PALETTE["panel_border"])

    add_panel_label(ax_frac, "A")
    add_panel_label(ax_err, "B")
    fig.suptitle(
        title,
        fontsize=SUPTITLE_SIZE,
        fontweight="semibold",
        color=PALETTE["ink"],
        y=1.04,
    )
    # No fig.tight_layout() here: fig.colorbar(..., ax=[ax_frac, ax_err]) above
    # already shrinks both axes to make room for itself, and running
    # tight_layout afterward fights that placement (colorbar ends up drifting
    # over the right-hand panel instead of sitting outside it).
    fig.subplots_adjust(top=0.78, bottom=0.16, wspace=0.3)
    save_fig(fig, save_path)
    return fig
