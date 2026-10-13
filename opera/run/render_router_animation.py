"""Render a conceptual animated explanation of outcome-aware routing.

All dots and displacements are synthetic. No patient records or fitted-model
outputs are loaded. Run: python -m opera.run.render_router_animation
"""

from __future__ import annotations

from io import BytesIO
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "paper" / "social"
BG = "#FFFFFF"
PANEL = "#FBFCFE"
EDGE = "#CAD4E0"
WHITE = "#17212B"  # primary ink; retained name for existing drawing calls
MUTED = "#657180"
TEAL = "#277F7A"
CORAL = "#BD625C"
VIOLET = "#7D6AA6"
GOLD = "#A96D2D"
COLORS = np.array([TEAL, VIOLET, GOLD])  # disease marker outlines
NO_EVENT = "#A9C9D2"
NEUTRAL = "#B9C3D0"


def ease(value: float) -> float:
    value = float(np.clip(value, 0, 1))
    return value * value * (3 - 2 * value)


def opacity(t: float, start: float, end: float) -> float:
    return ease((t - start) / (end - start))


def add_box(ax, x, y, w, h, *, fill=PANEL, edge=EDGE, alpha=1,
            radius=18, zorder=1):
    patch = FancyBboxPatch(
        (x, y), w, h, boxstyle=f"round,pad=0,rounding_size={radius}",
        facecolor=fill, edgecolor=edge, linewidth=1.2, alpha=alpha,
        zorder=zorder,
    )
    ax.add_patch(patch)


def txt(ax, x, y, value, size, color=WHITE, *, weight="normal", alpha=1,
        ha="left"):
    ax.text(x, y, value, fontsize=size, color=color, fontweight=weight,
            alpha=alpha, ha=ha, va="center", family="DejaVu Sans",
            zorder=8)


def source_card(ax, x, y, title, entries, color, *, alpha):
    if alpha < .01:
        return
    add_box(ax, x, y, 328, 139, alpha=alpha)
    txt(ax, x + 22, y + 26, title, 13, color, weight="bold", alpha=alpha)
    for j, entry in enumerate(entries):
        width = 87 if j < 2 else 95
        xx = x + 22 + j * 99
        add_box(ax, xx, y + 54, width, 48,
                fill="#E9F5F3" if color == TEAL else "#FCF0ED",
                edge=color, alpha=alpha, radius=13)
        txt(ax, xx + width / 2, y + 78, entry, 11, WHITE,
            ha="center", alpha=alpha)
    txt(ax, x + 22, y + 119, "learned from training examples", 9,
        MUTED, alpha=alpha)


def make_positions():
    rng = np.random.default_rng(53)
    n = 165
    group = rng.integers(0, 3, n)
    seed = rng.normal(0, .035, size=(n, 2))
    spread = rng.normal(0, .17, size=(n, 2))
    shared = np.clip(np.array([.50, .51]) + spread, .08, .92)
    start = np.clip(np.array([.50, .51]) + seed, .08, .92)
    # Unequal shift within each disease group: the route also depends on the
    # patient state. This is a storytelling device, not a trained router.
    disease_directions = np.array([[-.09, -.07], [.10, -.05], [-.02, .10]])
    patient_factor = .45 + .55 * (shared[:, 0] + shared[:, 1]) / 2
    after_disease = shared + disease_directions[group] * patient_factor[:, None]
    target_score = .52 * shared[:, 0] - .38 * shared[:, 1] + .12 * (group == 1)
    target_score += rng.normal(0, .065, n)
    # Status is shown only as retrospective annotation. It never enters the
    # synthetic displacement calculation, which depends on target_score.
    status = target_score + rng.normal(0, .14, n) > np.quantile(target_score, .69)
    centered = target_score - np.median(target_score)
    after_outcome = after_disease + np.column_stack((.24 * centered, -.17 * centered))
    after_outcome = np.clip(after_outcome, .07, .93)
    return start, shared, after_disease, after_outcome, group, status


def draw_frame(ax, frame, data):
    start, shared, after_disease, after_outcome, group, status = data
    t = frame / 89
    spread = opacity(t, .02, .21)
    target = opacity(t, .23, .34)
    disease = opacity(t, .39, .59)
    outcome = opacity(t, .64, .84)
    result = opacity(t, .85, .96)
    position = start + spread * (shared - start)
    position += disease * (after_disease - shared)
    position += outcome * (after_outcome - after_disease)

    ax.clear()
    ax.set_xlim(0, 720)
    ax.set_ylim(900, 0)
    ax.axis("off")
    ax.set_facecolor(BG)
    ax.add_patch(FancyBboxPatch((0, 0), 720, 900, boxstyle="square,pad=0",
                                facecolor=BG, edgecolor="none"))

    txt(ax, 43, 42, "OPERA  /  METHOD SCHEMATIC", 11, TEAL, weight="bold")
    txt(ax, 43, 87, "Patient-specific borrowing", 29, WHITE, weight="bold")
    txt(ax, 43, 124, "How disease and outcome knowledge informs a prediction",
        14, MUTED)

    step = 0 if t < .23 else 1 if t < .39 else 2 if t < .64 else 3 if t < .85 else 4
    headings = [
        "First, learn a shared patient space",
        "Now ask one clinical question",
        "Visualize disease-source borrowing",
        "Visualize outcome-source borrowing",
        "Combine sources for this patient + target",
    ]
    txt(ax, 43, 188, headings[step], 17, WHITE, weight="bold")
    for j in range(5):
        add_box(ax, 43 + 130 * j, 217, 115, 5,
                fill=TEAL if j <= step else "#E6EBF1",
                edge="none", radius=2)

    add_box(ax, 43, 249, 634, 417, radius=26)
    txt(ax, 65, 276, "PATIENT REPRESENTATIONS", 12, MUTED,
        weight="bold")

    # Marker shape denotes disease and marker fill denotes retrospective status
    # after the target is introduced. Neither status nor 2D coordinates are
    # inputs to the actual router; all observations here are synthetic.
    px = 73 + position[:, 0] * 574
    py = 300 + position[:, 1] * 321
    markers = ("o", "s", "^")
    from matplotlib.colors import to_rgb
    neutral = np.asarray(to_rgb(NEUTRAL))
    event_color = np.asarray(to_rgb(CORAL))
    nonevent_color = np.asarray(to_rgb(NO_EVENT))
    for g, marker in enumerate(markers):
        for observed in (False, True):
            mask = (group == g) & (status == observed)
            destination = event_color if observed else nonevent_color
            fill = neutral * (1 - target) + destination * target
            ax.scatter(px[mask], py[mask], s=38, marker=marker,
                       facecolors=[fill], edgecolors=COLORS[g],
                       linewidths=1.0, alpha=.88, zorder=2)
    focus = [5, 14, 30, 52, 77, 96, 125, 153]
    if disease > 0 or outcome > 0:
        before = shared if outcome == 0 else after_disease
        strength = disease if outcome == 0 else outcome
        for ix in focus:
            sx = 73 + before[ix, 0] * 574
            sy = 300 + before[ix, 1] * 321
            ex, ey = px[ix], py[ix]
            if np.hypot(ex - sx, ey - sy) > 3:
                ax.add_patch(FancyArrowPatch(
                    (sx, sy), (ex, ey), arrowstyle="-|>",
                    mutation_scale=10, color=COLORS[group[ix]],
                    linewidth=1.25, alpha=.72 * strength, zorder=3,
                ))
            ax.scatter([ex], [ey], s=105, marker=markers[group[ix]],
                       facecolors=BG, edgecolors=COLORS[group[ix]], linewidths=1.7,
                       alpha=.35 + .65 * strength, zorder=4)

    if target > .01:
        add_box(ax, 373, 271, 278, 54, fill="#FFF8EC", edge=GOLD,
                alpha=target, radius=17, zorder=6)
        txt(ax, 512, 298, "TARGET  •  INFECTION AT 365 DAYS", 11,
            GOLD, weight="bold", alpha=target, ha="center")

    if target > .01:
        txt(ax, 66, 642, "Shape:  ○ CLL   □ DLBCL   △ Myeloma", 9,
            MUTED, alpha=target)
        txt(ax, 440, 642, "Fill:", 9, MUTED, alpha=target)
        ax.scatter([482, 559], [642, 642], s=45, c=[CORAL, NO_EVENT],
                   alpha=target, edgecolors="none", zorder=5)
        txt(ax, 493, 642, "event", 9, MUTED, alpha=target)
        txt(ax, 570, 642, "no event", 9, MUTED, alpha=target)

    disease_alpha = opacity(t, .36, .45)
    outcome_alpha = opacity(t, .61, .70)
    source_card(ax, 43, 693, "DISEASE SOURCES",
                ("CLL", "DLBCL", "Myeloma"), TEAL, alpha=disease_alpha)
    source_card(ax, 392, 693, "OUTCOME SOURCES",
                ("Infection", "Anaemia", "Survival"), CORAL,
                alpha=outcome_alpha)
    if disease_alpha > .03:
        ax.add_patch(FancyArrowPatch((204, 691), (269, 659), arrowstyle="-|>",
                                     mutation_scale=14, color=TEAL,
                                     linewidth=2, alpha=.65 * disease_alpha))
    if outcome_alpha > .03:
        ax.add_patch(FancyArrowPatch((556, 691), (472, 659), arrowstyle="-|>",
                                     mutation_scale=14, color=CORAL,
                                     linewidth=2, alpha=.65 * outcome_alpha))
    if result > .01:
        add_box(ax, 228, 572, 264, 62, fill="#FFF8EC", edge=GOLD,
                alpha=result, radius=17, zorder=6)
        txt(ax, 360, 591, "TARGET-SPECIFIC READOUT", 10, GOLD,
            weight="bold", ha="center", alpha=result)
        txt(ax, 360, 614, "30–730 day risk", 17, WHITE,
            weight="bold", ha="center", alpha=result)

    if step == 2:
        txt(ax, 43, 680,
            "Example: a CLL patient may draw on a DLBCL-trained adapter.",
            10, TEAL)
    elif step == 3:
        txt(ax, 43, 680,
            "Example: an infection target may draw on other outcome adapters.",
            10, CORAL)
    elif step == 4:
        txt(ax, 43, 680,
            "The source mixture is specific to this patient and target.",
            10, MUTED)

    txt(ax, 43, 856,
        "Conceptual motion only  •  synthetic patients  •  no measured model output",
        9, MUTED)
    txt(ax, 43, 875,
        "Sources shown in sequence for clarity; the router combines them jointly.",
        9, MUTED)


def render(output: Path = OUT / "opera_router_animation.gif") -> Path:
    OUT.mkdir(parents=True, exist_ok=True)
    data = make_positions()
    fig = plt.figure(figsize=(9, 11.25), dpi=80, facecolor=BG)
    ax = fig.add_axes((0, 0, 1, 1))
    frames = []
    for i in range(90):
        draw_frame(ax, i, data)
        buf = BytesIO()
        fig.savefig(buf, format="png", dpi=80, facecolor=BG,
                    bbox_inches=None, pad_inches=0)
        buf.seek(0)
        frames.append(Image.open(buf).convert("RGB"))
    plt.close(fig)
    # A small global palette prevents flickering color assignments between
    # independently quantized frames.
    palette = frames[-1].quantize(colors=192, method=Image.Quantize.MEDIANCUT)
    indexed = [frame.quantize(palette=palette) for frame in frames]
    indexed[0].save(output, save_all=True, append_images=indexed[1:],
                    duration=90, loop=0, optimize=False, disposal=2)
    return output


if __name__ == "__main__":
    print(render())
