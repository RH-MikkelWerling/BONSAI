"""Render an editable, data-free social schematic of the OPERA router.

The scatter positions and risk curve are illustrative, never model outputs.
Run: python -m opera.run.render_router_linkedin
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Circle
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "paper" / "social"

BG = "#0B1230"
PANEL = "#111B3B"
PANEL_EDGE = "#344568"
WHITE = "#F7FAFF"
MUTED = "#ABB9D6"
TEAL = "#5EE0D0"
CORAL = "#FF8678"
VIOLET = "#AFA5FF"
GOLD = "#F5CF79"


def box(ax, x, y, w, h, *, face=PANEL, edge=PANEL_EDGE, radius=24, lw=1.2):
    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0,rounding_size={radius}",
        facecolor=face, edgecolor=edge, linewidth=lw,
    )
    ax.add_patch(patch)
    return patch


def label(ax, x, y, value, *, size=18, color=WHITE, weight="normal", ha="left", **kw):
    return ax.text(x, y, value, fontsize=size, color=color, fontweight=weight,
                   ha=ha, va="center", family="DejaVu Sans", **kw)


def arrow(ax, start, end, *, color, alpha=1, lw=2.0, rad=0.0, scale=15, z=3):
    patch = FancyArrowPatch(
        start, end, arrowstyle="-|>", mutation_scale=scale, linewidth=lw,
        color=color, alpha=alpha, connectionstyle=f"arc3,rad={rad}", zorder=z,
    )
    ax.add_patch(patch)
    return patch


def scatter_panel(ax, bounds, points, colors, *, alpha=.45):
    x0, y0, w, h = bounds
    ax.scatter(x0 + points[:, 0] * w, y0 + points[:, 1] * h,
               c=colors, s=21, alpha=alpha, linewidths=0, zorder=2)


def render(output: Path = OUT / "opera_router_linkedin") -> tuple[Path, Path]:
    OUT.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(10.8, 13.5), dpi=100, facecolor=BG)
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_xlim(0, 1080)
    ax.set_ylim(1350, 0)
    ax.axis("off")
    ax.set_facecolor(BG)

    # Header: one idea, one sentence; designed for a 4:5 social feed crop.
    box(ax, 60, 48, 300, 42, face="#202D5C", edge="#405694", radius=21)
    label(ax, 210, 69, "OPERA  /  THE METHOD", size=15, color=TEAL,
          weight="bold", ha="center")
    label(ax, 58, 155, "A different route for", size=48, weight="bold")
    label(ax, 58, 216, "every patient + outcome", size=48, weight="bold")
    label(ax, 60, 277, "Shared learning, selectively borrowed signal, personalized risk.",
          size=20, color=MUTED)

    # The two spaces are conceptual. The same nine highlighted patients are
    # connected across panels; all additional points are decorative context.
    left = (72, 375, 420, 348)
    right = (588, 375, 420, 348)
    box(ax, 54, 350, 455, 401, radius=29)
    box(ax, 571, 350, 455, 401, radius=29)
    label(ax, 76, 330, "01  SHARED PATIENT SPACE", size=17, color=TEAL,
          weight="bold")
    label(ax, 593, 330, "02  OUTCOME-SPECIFIC VIEW", size=17, color=TEAL,
          weight="bold")

    rng = np.random.default_rng(16)
    n = 225
    left_pts = rng.normal(loc=(.52, .51), scale=(.22, .20), size=(n, 2))
    left_pts = np.clip(left_pts, .065, .935)
    kind = rng.choice(3, size=n, p=(.38, .43, .19))
    dot_colors = np.array([TEAL, CORAL, VIOLET], dtype=object)[kind]
    scatter_panel(ax, left, left_pts, dot_colors, alpha=.39)

    # Separation is shown as an intuition for a target-specific representation,
    # not an assertion about an actual embedding or observed discrimination.
    centers = np.array([[.28, .68], [.76, .34], [.48, .18]])
    right_pts = centers[kind] + rng.normal(0, (.115, .115), size=(n, 2))
    right_pts = np.clip(right_pts, .065, .935)
    scatter_panel(ax, right, right_pts, dot_colors, alpha=.40)

    focus = np.array([12, 37, 63, 78, 104, 126, 158, 187, 208])
    for j, ix in enumerate(focus):
        color = dot_colors[ix]
        lx = left[0] + left_pts[ix, 0] * left[2]
        ly = left[1] + left_pts[ix, 1] * left[3]
        rx = right[0] + right_pts[ix, 0] * right[2]
        ry = right[1] + right_pts[ix, 1] * right[3]
        # The cross-panel path is deliberately light so it does not promise a
        # literal low-dimensional dynamical trajectory.
        if j in (0, 2, 4, 6, 8):
            arrow(ax, (lx + 7, ly), (rx - 8, ry), color=color,
                  alpha=.27, lw=1.3, rad=(-.08 + .02 * j), scale=12, z=3)
        for x, y in ((lx, ly), (rx, ry)):
            ax.add_patch(Circle((x, y), 8.4, facecolor=BG, edgecolor=color,
                                linewidth=2.1, zorder=5))
            ax.add_patch(Circle((x, y), 4.0, facecolor=color, edgecolor="none",
                                zorder=6))

    box(ax, 420, 484, 240, 103, face="#253268", edge="#7183D7", radius=22,
        lw=2.0)
    label(ax, 540, 516, "OUTCOME-AWARE", size=15, color=TEAL,
          weight="bold", ha="center", zorder=9)
    label(ax, 540, 552, "ROUTER", size=27, weight="bold", ha="center", zorder=9)
    label(ax, 78, 709, "same shared encoder", size=16, color=MUTED)
    label(ax, 996, 709, "different risk geometry", size=16,
          color=MUTED, ha="right")

    # The source banks visually feed the router, never the evaluation labels.
    label(ax, 58, 805, "03  BORROW FROM LEARNED SOURCES", size=17,
          color=TEAL, weight="bold")
    box(ax, 54, 840, 455, 167, radius=25)
    box(ax, 571, 840, 455, 167, radius=25)
    label(ax, 80, 873, "DISEASE ADAPTERS", size=16, color=TEAL, weight="bold")
    label(ax, 597, 873, "OUTCOME ADAPTERS", size=16, color=CORAL, weight="bold")
    disease = [(80, 904, 113, "CLL"), (203, 904, 118, "DLBCL"),
               (331, 904, 147, "Myeloma")]
    outcomes = [(597, 904, 117, "Infection"), (724, 904, 125, "Anaemia"),
                (859, 904, 141, "Survival")]
    for x, y, w, name in disease:
        box(ax, x, y, w, 49, face="#153F4A", edge="#3E8F98", radius=18)
        label(ax, x + w / 2, y + 25, name, size=15, ha="center")
    for x, y, w, name in outcomes:
        box(ax, x, y, w, 49, face="#513347", edge="#A96774", radius=18)
        label(ax, x + w / 2, y + 25, name, size=15, ha="center")
    label(ax, 80, 979, "Learned from eligible training examples", size=14,
          color=MUTED)
    label(ax, 597, 979, "Weighted for this patient + target", size=14,
          color=MUTED)
    arrow(ax, (354, 840), (484, 590), color=TEAL, alpha=.64, lw=2.1,
          rad=-.21, scale=15, z=1)
    arrow(ax, (728, 840), (596, 590), color=CORAL, alpha=.64, lw=2.1,
          rad=.21, scale=15, z=1)

    # A compact output module completes the causal reading direction.
    arrow(ax, (540, 1008), (540, 1052), color=GOLD, lw=2.0, scale=16)
    box(ax, 54, 1058, 972, 145, face="#18254B", edge="#576CA8", radius=25)
    label(ax, 80, 1094, "04  PREDICT THE TARGET OUTCOME", size=17,
          color=GOLD, weight="bold")
    label(ax, 80, 1145, "Patient-specific cumulative risk", size=25,
          weight="bold")
    days = [30, 60, 90, 180, 365, 730]
    xs = np.linspace(658, 963, len(days))
    ys = np.array([1164, 1159, 1151, 1135, 1120, 1109])
    ax.plot(xs, ys, color=GOLD, linewidth=3.3, zorder=6)
    ax.scatter(xs, ys, s=43, c=GOLD, edgecolors=BG, linewidths=1.2, zorder=7)
    label(ax, 658, 1181, "30", size=12, color=MUTED, ha="center")
    label(ax, 963, 1181, "730 days", size=12, color=MUTED, ha="center")

    label(ax, 58, 1255,
          "Every dot is illustrative. The router changes a representation, not a patient's outcome.",
          size=14, color=MUTED)
    label(ax, 58, 1295,
          "Conceptual schematic  •  no patient data or measured model performance shown",
          size=13, color="#8594B5")

    png = output.with_suffix(".png")
    svg = output.with_suffix(".svg")
    fig.savefig(png, dpi=200, facecolor=BG, bbox_inches=None, pad_inches=0)
    fig.savefig(svg, facecolor=BG, bbox_inches=None, pad_inches=0)
    plt.close(fig)
    return png, svg


if __name__ == "__main__":
    for path in render():
        print(path)
