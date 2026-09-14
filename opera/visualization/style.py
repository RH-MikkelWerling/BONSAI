"""
OPERA Visualization Style System.

Single source of truth for colors, typography, and layout.
Import this module in every visualization file.

Design philosophy
─────────────────
- Jewel-tone palette: high visual depth, print-safe, perceptually distinct.
  Inspired by Nature Medicine / Cell figure aesthetics — saturated but not
  garish, legible at small point sizes and at presentation scale alike.
- Journal two-column width: 7 inches full, 3.4 inches half
- Nature Medicine print safety: keep final CMYK conversion in mind; avoid
  low-contrast hue-only encodings and inspect converted proofs before submit.
- Minimal chrome: no top/right spines, hairline grid only where informative
- Rasterized scatter for large N (prevents multi-MB PDFs)
- Vector text always (not rasterized) so figures remain editable

Typography
──────────
DejaVu Sans ships with Matplotlib, providing consistent glyph metrics on
Windows, Linux and render hosts. PDF embeds TrueType fonts; SVG keeps text
editable (the receiving editor needs the font installed).

Colorblind safety
─────────────────
The palette is designed to remain distinguishable under deuteranopia
(the most common form): indigo/crimson/teal/amber are separable in both
hue and luminance. For figures with >4 overlaid traces, also vary the
marker shape or line style (solid / dashed / dotted / dash-dot) — never
rely on hue alone.
"""

from __future__ import annotations
from typing import Optional
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from pathlib import Path
import hashlib
from cycler import cycler
from matplotlib.collections import PathCollection


# ── Color palette ──────────────────────────────────────────────────────────────

PALETTE = {
    # Model tiers — luminance-ordered to encode hierarchy
    # Primary model is the most visually prominent (deepest, most saturated)
    "opera": "#2D3A8C",  # deep indigo          ← hero model
    "opera_joint": "#1A237E",  # darker navy
    "dapt": "#5C7AEA",  # periwinkle
    "pretrain": "#737D8C",  # slate; readable at small sizes
    "tabular_rkkp": "#1B7A4A",  # forest green
    "tabular_ehr": "#408579",  # muted teal
    "ipi": "#B5451B",  # burnt sienna
    # Binary class labels — warm/cool split, high contrast
    "positive": "#B65B63",  # muted cranberry
    "negative": "#71869E",  # desaturated blue
    "missing": "#D0D0D0",  # light grey
    # Text / ink roles — text always wears one of these, never a series hue
    "ink": "#17212B",  # primary text: titles, in-plot labels
    "ink_secondary": "#485463",  # subtitles, secondary annotations
    "ink_muted": "#657180",  # notes remain readable in print
    # Chrome — panel borders, connector lines, structural (non-data) ink
    "panel_border": "#CBD2DA",
    "connector": "#9B9B9B",
    # Neutral / decorative
    "diagonal": "#AAAAAA",
    "zero_line": "#2B2B2B",
    "grid": "#E8ECF1",
    "fill_alpha": 0.13,  # CI ribbon alpha
}

# Categorical sequence — 8 perceptually distinct jewel tones.
# Ordered so the first 4 are maximally separated (alternating warm/cool).
# Fixed order — assign by identity (disease, model tier, ...), never cycle
# or reassign when a filter changes which series are present.
CATEGORICAL = [
    "#2D3A8C",  # deep indigo
    "#C16A58",  # terracotta
    "#3E8A87",  # teal
    "#B18B4D",  # ochre
    "#7189BC",  # muted blue
    "#8974AC",  # lavender
    "#527B72",  # eucalyptus
    "#9E9E9E",  # grey (last resort / pretrain)
]

MODEL_DISPLAY = {
    "opera": "OPERA",
    "opera_joint": "OPERA (joint)",
    "dapt": "DAPT",
    "pretrain": "Pretrain",
    "tabular_rkkp": "Tabular + IPI",
    "tabular_ehr": "Tabular EHR",
    "ipi": "IPI score",
}

# ── Typography ─────────────────────────────────────────────────────────────────
# A single type scale, used everywhere instead of scattered magic numbers.
# Use the bundled font for reproducible layout on all render hosts.

FONT_FAMILY = ["DejaVu Sans", "Arial", "sans-serif"]

SUPTITLE_SIZE = 13  # figure-level title (multi-panel figures)
TITLE_SIZE = 11  # axes title / single-panel figure title
SUBTITLE_SIZE = 9.5  # descriptor line beneath a title
LABEL_SIZE = 9.5  # axis labels
TICK_SIZE = 8.5  # tick labels
LEGEND_TITLE_SIZE = 8.5
LEGEND_SIZE = 8.5
ANNOT_SIZE = 8.5  # in-plot data callouts / delta labels
NOTE_SIZE = 8  # small footnotes, source lines, "n=" captions
PANEL_LABEL_SIZE = 11  # bold A / B / C panel tags
BASE_SIZE = LABEL_SIZE  # kept for backward compatibility with older call sites

# ── Figure dimensions (inches) ─────────────────────────────────────────────────

FIG_NM_FULL = (7.09, 4.5)  # Nature Medicine two-column landscape
FIG_NM_HALF = (3.46, 2.8)  # Nature Medicine single-column landscape
FIG_FULL = FIG_NM_FULL  # two-column landscape
FIG_SQUARE = (3.4, 3.4)  # single-column square
FIG_HALF = FIG_NM_HALF  # single-column landscape
FIG_TALL = (3.4, 5.0)  # single-column tall
FIG_WIDE = (7.0, 3.0)  # two-column short

# ── Spacing constants ──────────────────────────────────────────────────────────
# Named instead of re-guessed per call site. Values are in the units the
# matplotlib API they feed into expects (points for pad=, figure-fraction
# for legend anchors).

TITLE_PAD = 10  # points, between title baseline and axes top
SUPTITLE_Y = 0.985  # figure-fraction, suptitle baseline
SUBTITLE_Y_OFFSET = 0.035  # figure-fraction, subtitle below suptitle
PANEL_LABEL_OFFSET = (-0.11, 1.05)  # axes-fraction, default (x, y) for A/B/C tags


def setup_style() -> None:
    """Apply the OPERA style sheet globally. Call once at module import."""
    mpl.rcParams.update(
        {
            # Font
            "font.family": "sans-serif",
            "font.sans-serif": FONT_FAMILY,
            "font.size": LABEL_SIZE,
            "axes.titlesize": TITLE_SIZE,
            "axes.titleweight": "semibold",
            "axes.titlecolor": PALETTE["ink"],
            "axes.titlelocation": "left",
            "axes.titlepad": TITLE_PAD,
            "axes.prop_cycle": cycler(color=CATEGORICAL),
            "axes.labelsize": LABEL_SIZE,
            "axes.labelcolor": PALETTE["ink"],
            "xtick.labelsize": TICK_SIZE,
            "ytick.labelsize": TICK_SIZE,
            "xtick.color": PALETTE["ink_secondary"],
            "ytick.color": PALETTE["ink_secondary"],
            "text.color": PALETTE["ink"],
            "legend.fontsize": LEGEND_SIZE,
            "legend.title_fontsize": LEGEND_TITLE_SIZE,
            "figure.titlesize": SUPTITLE_SIZE,
            "figure.titleweight": "semibold",
            "mathtext.fontset": "dejavusans",
            "mathtext.default": "regular",
            # Spines and ticks
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.edgecolor": PALETTE["panel_border"],
            "axes.linewidth": 0.7,
            "xtick.major.width": 0.7,
            "ytick.major.width": 0.7,
            "xtick.major.size": 3.5,
            "ytick.major.size": 3.5,
            "xtick.major.pad": 4,
            "ytick.major.pad": 4,
            # Grid — recessive hairline; individual plots turn it off via
            # despine(ax, grid_axis="none") where a grid isn't informative
            "axes.grid": True,
            "axes.grid.axis": "y",
            "grid.color": PALETTE["grid"],
            "grid.linewidth": 0.6,
            "grid.alpha": 1.0,
            # Lines
            "lines.linewidth": 1.8,
            "lines.markersize": 4.5,
            "errorbar.capsize": 2.5,
            "lines.solid_capstyle": "round",
            "lines.solid_joinstyle": "round",
            # Figure — generous, breathable layout by default
            "figure.dpi": 130,
            "figure.facecolor": "white",
            "figure.constrained_layout.use": False,  # opt in per-figure; mixed with manual axes placement elsewhere
            "figure.subplot.hspace": 0.32,
            "figure.subplot.wspace": 0.28,
            "axes.facecolor": "white",
            "axes.axisbelow": True,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.12,
            "savefig.facecolor": "white",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "svg.hashsalt": "opera-publication",
            # Legend — thin border, generous internal breathing room
            "legend.frameon": False,
            "legend.framealpha": 0.94,
            "legend.edgecolor": PALETTE["panel_border"],
            "legend.facecolor": "white",
            "legend.borderpad": 0.6,
            "legend.labelspacing": 0.55,
            "legend.handletextpad": 0.55,
            "legend.columnspacing": 1.3,
            "legend.borderaxespad": 0.6,
        }
    )


setup_style()


# ── Utility helpers ─────────────────────────────────────────────────────────────


def save_fig(fig: plt.Figure, path: Optional[str], dpi: int = 300) -> None:
    """Export publication artwork, keeping text vector and dense points raster.

    Raster requests also produce a PDF; PDF/SVG requests retain their format.
    Only dense scatter collections are rasterized, never curves or CI bands.
    """
    if path is not None:
        for ax in fig.axes:
            for collection in ax.collections:
                if (
                    isinstance(collection, PathCollection)
                    and len(collection.get_offsets()) >= 1000
                ):
                    collection.set_rasterized(True)
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(
            path, dpi=dpi, bbox_inches="tight", pad_inches=0.12, facecolor="white"
        )
        if path.suffix.lower() in {".png", ".jpg", ".jpeg"}:
            fig.savefig(
                path.with_suffix(".pdf"),
                dpi=dpi,
                bbox_inches="tight",
                pad_inches=0.12,
                facecolor="white",
            )


def model_color(name: str) -> str:
    """Return an identity-stable color, independent of Python's hash seed."""
    return PALETTE.get(name, CATEGORICAL[stable_index(name) % len(CATEGORICAL)])


def stable_index(name: str) -> int:
    """Stable identity for categorical encodings across processes and subsets."""
    return int.from_bytes(hashlib.sha256(str(name).encode("utf-8")).digest()[:8], "big")


def model_linestyle(name: str):
    """Redundant line encoding so comparisons do not depend on hue alone."""
    names = list(MODEL_DISPLAY)
    index = names.index(name) if name in names else stable_index(name)
    return ("-", "--", "-.", ":")[index % 4]


def model_label(name: str) -> str:
    return MODEL_DISPLAY.get(name, name.replace("_", " ").title())


def add_panel_label(
    ax: plt.Axes,
    label: str,
    x: Optional[float] = None,
    y: Optional[float] = None,
) -> None:
    """Add a bold panel label (A, B, C ...) in the top-left corner of an axis."""
    xo, yo = PANEL_LABEL_OFFSET
    ax.text(
        x if x is not None else xo,
        y if y is not None else yo,
        label,
        transform=ax.transAxes,
        fontsize=PANEL_LABEL_SIZE,
        fontweight="bold",
        color=PALETTE["ink"],
        va="top",
        ha="left",
    )


def despine(ax: plt.Axes, grid_axis: str = "y") -> None:
    """Remove top/right spines and set grid direction. Grid stays a hairline
    the same weight/color everywhere it's turned on, so charts read as one
    system rather than each picking its own grid style."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(PALETTE["panel_border"])
    ax.spines["bottom"].set_color(PALETTE["panel_border"])
    if grid_axis == "y":
        ax.yaxis.grid(True, color=PALETTE["grid"], linewidth=0.6, zorder=0)
        ax.xaxis.grid(False)
    elif grid_axis == "x":
        ax.xaxis.grid(True, color=PALETTE["grid"], linewidth=0.6, zorder=0)
        ax.yaxis.grid(False)
    elif grid_axis == "both":
        ax.grid(True, color=PALETTE["grid"], linewidth=0.6, zorder=0)
    elif grid_axis == "none":
        ax.grid(False)
    ax.set_axisbelow(True)


def clean_2d_axes(ax: plt.Axes) -> None:
    """Strip an embedding/projection axes down to just its labels — no
    ticks, no spines, no grid. Use for UMAP/t-SNE panels where the numeric
    coordinates themselves aren't meaningful."""
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.grid(False)


def figure_title(
    fig: plt.Figure,
    title: str,
    subtitle: Optional[str] = None,
    y: float = SUPTITLE_Y,
) -> None:
    """Render a clean figure title.

    ``subtitle`` remains accepted for call-site compatibility but is not drawn;
    descriptive prose belongs in the manuscript caption.
    """
    fig.suptitle(
        title,
        y=y,
        fontsize=SUPTITLE_SIZE,
        fontweight="semibold",
        color=PALETTE["ink"],
    )


def style_legend(legend) -> None:
    """Apply the shared legend chrome (thin panel-border frame, no heavy
    shadow) to a legend built with custom fontsize/handles, so hand-tuned
    legends still match the rcParams-driven default look."""
    frame = legend.get_frame()
    frame.set_edgecolor(PALETTE["panel_border"])
    frame.set_facecolor("white")
    frame.set_alpha(0.94)
    frame.set_linewidth(0.7)
    if legend.get_title() is not None:
        legend.get_title().set_fontweight("semibold")
        legend.get_title().set_color(PALETTE["ink"])


def sequential_cmap(hue: str = "opera"):
    """A one-hue light→dark colormap for continuous metadata (age, risk
    score, ...), matching the categorical palette instead of a generic
    matplotlib rainbow (viridis/plasma) that doesn't read as part of the
    same visual system as the rest of the figure."""
    import matplotlib.colors as mcolors

    dark = PALETTE.get(hue, hue)
    light = "#F3F5FC"
    return mcolors.LinearSegmentedColormap.from_list(f"opera_{hue}", [light, dark])


def ci_ribbon(
    ax: plt.Axes,
    x: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    color: str,
    alpha: float = 0.12,
) -> None:
    """Shade a confidence interval ribbon."""
    ax.fill_between(x, lower, upper, color=color, alpha=alpha, linewidth=0)
