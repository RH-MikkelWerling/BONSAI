"""
OPERA Visualization Style System.

Single source of truth for colors, typography, and layout.
Import this module in every visualization file.

Design philosophy
─────────────────
- Jewel-tone palette: high visual depth, print-safe, perceptually distinct.
  Inspired by Nature Medicine / Cell figure aesthetics — saturated but not
  garish, legible at 8pt in a two-column layout.
- Journal two-column width: 7 inches full, 3.4 inches half
- Nature Medicine print safety: keep final CMYK conversion in mind; avoid
  low-contrast hue-only encodings and inspect converted proofs before submit.
- Minimal chrome: no top/right spines, hairline grid only where informative
- Rasterized scatter for large N (prevents multi-MB PDFs)
- Vector text always (not rasterized) so figures remain editable

Colorblind safety
─────────────────
The palette is designed to remain distinguishable under deuteranopia
(the most common form): indigo/crimson/teal/amber are separable in both
hue and luminance. For figures with >4 overlaid traces, also vary the
line style (solid / dashed / dotted / dash-dot) — never rely on hue alone.
"""

from __future__ import annotations
from typing import Dict, List, Optional
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from pathlib import Path


# ── Color palette ──────────────────────────────────────────────────────────────

PALETTE = {
    # Model tiers — luminance-ordered to encode hierarchy
    # Primary model is the most visually prominent (deepest, most saturated)
    "opera":          "#2D3A8C",   # deep indigo          ← hero model
    "opera_joint":    "#1A237E",   # darker navy
    "dapt":           "#5C7AEA",   # periwinkle
    "pretrain":       "#9E9E9E",   # neutral grey
    "tabular_rkkp":   "#1B7A4A",   # forest green
    "tabular_ehr":    "#66BB99",   # sage green
    "ipi":            "#B5451B",   # burnt sienna

    # Binary class labels — warm/cool split, high contrast
    "positive":       "#B5232A",   # deep crimson
    "negative":       "#1565C0",   # royal blue
    "missing":        "#D0D0D0",   # light grey

    # Neutral / decorative
    "diagonal":       "#AAAAAA",
    "zero_line":      "#333333",
    "grid":           "#EEEEEE",
    "fill_alpha":     0.13,        # CI ribbon alpha
}

# Categorical sequence — 8 perceptually distinct jewel tones.
# Ordered so the first 4 are maximally separated (alternating warm/cool).
CATEGORICAL = [
    "#2D3A8C",   # deep indigo
    "#B5232A",   # deep crimson
    "#1B7A4A",   # forest green
    "#B07A10",   # dark amber
    "#5C7AEA",   # periwinkle
    "#7B3FA0",   # plum
    "#1A7A85",   # teal
    "#9E9E9E",   # grey (last resort / pretrain)
]

MODEL_DISPLAY = {
    "opera":          "OPERA",
    "opera_joint":    "OPERA (joint)",
    "dapt":           "DAPT",
    "pretrain":       "Pretrain",
    "tabular_rkkp":   "Tabular + IPI",
    "tabular_ehr":    "Tabular EHR",
    "ipi":            "IPI score",
}

# ── Typography ─────────────────────────────────────────────────────────────────

FONT_FAMILY = ["Helvetica Neue", "Arial", "DejaVu Sans", "sans-serif"]
BASE_SIZE    = 9       # pt — suits a 7" wide figure in a typical journal
TITLE_SIZE   = 10
LABEL_SIZE   = 9
TICK_SIZE    = 8
LEGEND_SIZE  = 8
ANNOT_SIZE   = 7.5

# ── Figure dimensions (inches) ─────────────────────────────────────────────────

FIG_NM_FULL = (7.09, 4.5)  # Nature Medicine two-column landscape
FIG_NM_HALF = (3.46, 2.8)  # Nature Medicine single-column landscape
FIG_FULL   = FIG_NM_FULL   # two-column landscape
FIG_SQUARE = (3.4, 3.4)    # single-column square
FIG_HALF   = FIG_NM_HALF   # single-column landscape
FIG_TALL   = (3.4, 5.0)    # single-column tall
FIG_WIDE   = (7.0, 3.0)    # two-column short


def setup_style() -> None:
    """Apply the OPERA style sheet globally. Call once at module import."""
    mpl.rcParams.update({
        # Font
        "font.family":          "sans-serif",
        "font.sans-serif":      FONT_FAMILY,
        "font.size":            BASE_SIZE,
        "axes.titlesize":       TITLE_SIZE,
        "axes.titleweight":     "semibold",
        "axes.labelsize":       LABEL_SIZE,
        "xtick.labelsize":      TICK_SIZE,
        "ytick.labelsize":      TICK_SIZE,
        "legend.fontsize":      LEGEND_SIZE,

        # Spines and ticks
        "axes.spines.top":      False,
        "axes.spines.right":    False,
        "axes.linewidth":       0.6,
        "xtick.major.width":    0.6,
        "ytick.major.width":    0.6,
        "xtick.major.size":     3,
        "ytick.major.size":     3,

        # Grid
        "axes.grid":            True,
        "grid.color":           PALETTE["grid"],
        "grid.linewidth":       0.5,
        "grid.alpha":           1.0,

        # Lines
        "lines.linewidth":      1.8,

        # Figure
        "figure.dpi":           130,
        "figure.facecolor":     "white",
        "axes.facecolor":       "white",
        "savefig.dpi":          300,
        "savefig.bbox":         "tight",
        "savefig.facecolor":    "white",

        # Legend
        "legend.frameon":       True,
        "legend.framealpha":    0.92,
        "legend.edgecolor":     "#DDDDDD",
        "legend.borderpad":     0.5,
    })


setup_style()


# ── Utility helpers ─────────────────────────────────────────────────────────────

def save_fig(fig: plt.Figure, path: Optional[str], dpi: int = 300) -> None:
    if path is not None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
        if path.suffix.lower() in {".png", ".jpg", ".jpeg"}:
            fig.savefig(
                path.with_suffix(".pdf"),
                bbox_inches="tight",
                facecolor="white",
            )


def model_color(name: str) -> str:
    """Return the canonical color for a known model name, else cycle through CATEGORICAL."""
    return PALETTE.get(name, CATEGORICAL[hash(name) % len(CATEGORICAL)])


def model_label(name: str) -> str:
    return MODEL_DISPLAY.get(name, name.replace("_", " ").title())


def add_panel_label(ax: plt.Axes, label: str, x: float = -0.12, y: float = 1.04) -> None:
    """Add a bold panel label (A, B, C ...) in the top-left corner of an axis."""
    ax.text(x, y, label, transform=ax.transAxes,
            fontsize=8, fontweight="bold", va="top", ha="left")


def despine(ax: plt.Axes, grid_axis: str = "y") -> None:
    """Remove top/right spines and set grid direction."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if grid_axis == "y":
        ax.yaxis.grid(True, color=PALETTE["grid"], linewidth=0.5)
        ax.xaxis.grid(False)
    elif grid_axis == "x":
        ax.xaxis.grid(True, color=PALETTE["grid"], linewidth=0.5)
        ax.yaxis.grid(False)
    elif grid_axis == "both":
        ax.grid(True, color=PALETTE["grid"], linewidth=0.5)
    elif grid_axis == "none":
        ax.grid(False)


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
