"""Within-category splits for every clinical index (supp_index_splits, supp_index_splits_km_*):
each index category divided at adapted OPERA's median predicted risk within it, from the
aggregate Kaplan-Meier exports of the clinical step (km_split_summary.csv, km_split_curves.csv).

  supp_index_splits        one page in the visual language of Fig. 6A: the index category (sienna)
                           and OPERA's two halves of it (navy), joined by OPERA's spread;
  supp_index_splits_km_N   the Kaplan-Meier curves behind it, as an aligned grid: one row per
                           comparison, the index's lowest category on the left and its highest on
                           the right.

Usage:
  python -m scripts.render_clinical_splits scratch/R_EA_clinical_figures_20261117 \
      paper/figures/results_thesis_overlay/story
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, MultipleLocator

from opera.run.paper_story_figures import (CLINICAL_ROWS, INK, INK2, LABEL_SIZE, MUTED, OPERA, SIENNA, ZERO,
                                           _save, lighten, thesis_style, tidy)

LOWER, UPPER = "OPERA lower half", "OPERA upper half"
ENDPOINT = {"OS": "overall survival", "TF": "freedom from treatment failure"}
SHORT_END = {"OS": "OS", "TF": "TF"}
SHORT = {"Low-intermediate": "Low-int.", "High-intermediate": "High-int."}
# KM page: OPERA's halves coded by risk (teal lower, claret higher; claret rather than the index's
# sienna), the whole category grey and dashed.
HALF_LOW, HALF_HIGH, WHOLE = "#3E8A84", "#8E2C48", "#8A93A0"


def _comparisons(summary):
    """(score_id, index name, endpoint key, horizon in years, patients, categories low to high)."""
    out = []
    for sid, name, end in CLINICAL_ROWS:
        sm = summary[summary.score_id.eq(sid)]
        index = sm[sm.display.eq("index")]
        years = int(round(sm.horizon_days.iloc[0] / 365))
        out.append((sid, name, end.split(",")[0], years, int(index.n.sum()), list(dict.fromkeys(index.category))))
    return out


def _rows(summary, sid, cats):
    sm = summary[summary.score_id.eq(sid)]
    for cat in cats:
        part = sm[sm.category.eq(cat)]
        whole = part[part.display.eq("index")].iloc[0]
        halves = part[part.display.eq("within")].set_index("group")
        yield cat, whole, halves.loc[LOWER], halves.loc[UPPER]


def _label(name, end, years, n):
    """As Fig. 6A's row labels: index, endpoint and horizon, patients in brackets."""
    return f"{name}, {SHORT_END[end]} at {years} y (n={n:,})"


# -- S29: one page, Fig. 6A's dumbbells ----------------------------------------------------
def overview(summary, comps, out):
    """Two columns of comparisons. Each category is a row: the index category's survival (sienna
    diamond) sits on OPERA's spread, the bar between its lower- and higher-risk halves (navy
    diamonds; filled where the halves' 95% intervals do not overlap). Right: the gap in points."""
    columns = [comps[:6], comps[6:]]
    fig = plt.figure(figsize=(6.9, 8.3))
    top, bottom = 0.885, 0.1
    for k, col in enumerate(columns):
        ax = fig.add_axes([0.14 + 0.5 * k, bottom, 0.25, top - bottom])
        y, ticks, labels = 0.0, [], []
        for sid, name, end, years, n, cats in col:
            y += 0.5
            ax.annotate(_label(name, end, years, n), xy=(-0.5, y), xycoords=ax.get_yaxis_transform(), ha="left",
                        va="center", fontsize=7.8, fontweight="bold", color=INK, annotation_clip=False)
            y += 0.95
            for cat, whole, lo, hi in _rows(summary, sid, cats):
                sure = bool(lo.lower > hi.upper)
                ax.plot([hi.survival, lo.survival], [y, y], color=lighten(OPERA, 0.3), lw=2.2, zorder=1,
                        solid_capstyle="round")
                ax.scatter([whole.survival], [y], marker="D", s=24, color=SIENNA, edgecolor="white", linewidth=1,
                           zorder=3)
                for r in (lo, hi):
                    ax.scatter([r.survival], [y], marker="D", s=30, color=OPERA if sure else "white",
                               edgecolor="white" if sure else OPERA, linewidth=1 if sure else 1.1, zorder=4)
                gap = 100 * (lo.survival - hi.survival)
                ax.text(1.38, y - 0.1, f"{gap:.0f}", transform=ax.get_yaxis_transform(), ha="right", va="center",
                        fontsize=7.2, color=INK2 if sure else MUTED, fontweight="bold" if sure else "normal")
                ax.text(1.38, y + 0.33, f"({100 * lo.survival:.0f}% vs {100 * hi.survival:.0f}%)",
                        transform=ax.get_yaxis_transform(), ha="right", va="center", fontsize=5.6, color=MUTED)
                ticks.append(y)
                labels.append(f"{SHORT.get(cat, cat)} ({int(whole.n):,})")
                y += 1.0
        ax.set_ylim(y - 0.35, -0.2)
        ax.set_yticks(ticks, labels, fontsize=7.4)
        ax.set_xlim(0.3, 1.025)
        ax.set_xticks([0.4, 0.6, 0.8, 1.0], ["40%", "60%", "80%", "100%"])
        tidy(ax, grid="x")
        ax.spines["left"].set_visible(False)
        ax.set_xlabel("Event-free at the horizon", fontsize=LABEL_SIZE)
        ax.text(1.38, -0.15, "Gap, points", transform=ax.get_yaxis_transform(), ha="right", va="bottom",
                fontsize=6.6, color=MUTED, style="italic")
    fig.text(0.02, 0.975, "Within each index category, OPERA separates patients", fontsize=9.6,
             fontweight="bold", color=INK, va="top")
    fig.text(0.02, 0.952, "Each category split at OPERA's median predicted risk; same held-out patients",
             fontsize=7.8, color=MUTED, style="italic", va="top")
    fig.legend(handles=[Line2D([], [], marker="D", ls="none", color=SIENNA, markeredgecolor="white", markersize=5.2,
                               label="Index category"),
                        Line2D([0, 1], [0, 0], marker="D", ls="-", lw=2.2, color=lighten(OPERA, 0.3),
                               markerfacecolor=OPERA, markeredgecolor="white", markersize=5.6,
                               label="OPERA's lower- and higher-risk halves"),
                        Line2D([], [], marker="D", ls="none", markerfacecolor="white", markeredgecolor=OPERA,
                               markersize=5.2, label="Halves' intervals overlap")],
               loc="lower center", ncol=3, frameon=False, fontsize=7.4, bbox_to_anchor=(0.5, 0.0),
               handletextpad=0.4, columnspacing=1.6)
    _save(fig, out / "supp_index_splits")


# -- S30-S31: the curves, as an aligned grid --------------------------------------------------
def _spans(cats):
    """Column of each category on a four-column grid (lowest on the left, highest on the right),
    and the (start, span) of the columns the index has no category for."""
    if len(cats) == 4:
        return [0, 1, 2, 3], []
    if len(cats) == 3:
        return [0, 1, 3], [(2, 1)]
    if len(cats) == 2:
        return [0, 3], [(1, 2)]
    raise ValueError(cats)


def _not_available(fig, x0, y0, w, h, n_cats):
    """A quiet placeholder where the index has no category: pale fill and a short note."""
    ax = fig.add_axes([x0, y0, w, h])
    ax.set_facecolor("#F4F6F8")
    for side in ax.spines.values():
        side.set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.text(0.5, 0.5, f"The index has {n_cats} categories", transform=ax.transAxes, ha="center", va="center",
            fontsize=6.6, color=MUTED, style="italic")


def _step(ax, part, colour, lw, ls="-", band=True, z=2):
    ax.step(part.time_days / 365, part.survival, where="post", color=colour, lw=lw, ls=ls, zorder=z)
    if band:
        ax.fill_between(part.time_days / 365, part.lower, part.upper, step="post", color=colour, alpha=0.13, lw=0,
                        zorder=z - 1)


def _at(part, years):
    sel = part.loc[part.time_days <= years * 365]
    return int(part.at_risk.iloc[0]), int(sel.at_risk.iloc[-1]) if len(sel) else int(part.at_risk.iloc[0])


def _at_risk(part, t):
    sel = part.loc[part.time_days <= t * 365]
    return int(sel.at_risk.iloc[-1]) if len(sel) else int(part.at_risk.iloc[0])


def km_page(summary, curves, comps, out, stem, panel_h=0.74):
    """One row per comparison, one panel per category, with numbers at risk under every panel
    (start, half the horizon, the horizon) for each of OPERA's halves."""
    W, left, col_w, gap, head = 6.9, 0.98, 1.34, 0.13, 0.42
    title_h, ticks_h, risk_h, after = 0.17, 0.13, 0.13, 0.14
    row_h = title_h + panel_h + ticks_h + 2 * risk_h + after
    H = head + row_h * len(comps)
    fig = plt.figure(figsize=(W, H))
    for i, (sid, name, end, years, n, cats) in enumerate(comps):
        top = head + row_h * i + title_h  # top of the panels, inches from the top
        c = curves[curves.score_id.eq(sid)]
        y_mid = 1 - (top + panel_h / 2) / H
        # The index's name right against its first panel (right-aligned, clear of the y labels).
        lx = (left - 0.3) / W
        fig.text(lx, y_mid + 0.12 / H, name, fontsize=8.2, fontweight="bold", color=INK, va="center", ha="right")
        fig.text(lx, y_mid - 0.03 / H, f"{SHORT_END[end]} at {years} y", fontsize=6.6, color=INK2, va="center",
                 ha="right")
        fig.text(lx, y_mid - 0.15 / H, f"n={n:,}", fontsize=6.6, color=MUTED, va="center", ha="right")
        if years != 1:  # the one comparison on a longer horizon: say so where the eye lands
            fig.text(lx, y_mid - 0.3 / H, f"{years}-year axis", fontsize=6.4, color=INK2, style="italic",
                     fontweight="bold", va="center", ha="right")
        rows_y = [1 - (top + panel_h + ticks_h + (k + 0.55) * risk_h) / H for k in range(2)]
        fig.text((left - 0.17) / W, rows_y[0] + 0.9 * risk_h / H, "At risk", fontsize=5.6, color=MUTED, ha="right",
                 va="center", style="italic")
        for k, (label, colour) in enumerate((("Lower half", HALF_LOW), ("Higher half", HALF_HIGH))):
            fig.text((left - 0.17) / W, rows_y[k], label, fontsize=5.6, color=colour, ha="right", va="center")
        slots, empty = _spans(cats)
        for start, span in empty:
            _not_available(fig, (left + start * (col_w + gap)) / W, 1 - (top + panel_h) / H,
                           (span * col_w + (span - 1) * gap) / W, panel_h / H, len(cats))
        for start, (cat, whole, lo, hi) in zip(slots, _rows(summary, sid, cats)):
            x0 = left + start * (col_w + gap)
            ax = fig.add_axes([x0 / W, 1 - (top + panel_h) / H, col_w / W, panel_h / H])
            _step(ax, c[c.display.eq("index") & c.category.eq(cat)], WHOLE, lw=0.9, ls=(0, (3, 2)), band=False)
            parts = {g: c[c.display.eq("within") & c.category.eq(cat) & c.group.eq(g)] for g in (LOWER, UPPER)}
            _step(ax, parts[UPPER], HALF_HIGH, lw=1.2, z=3)
            _step(ax, parts[LOWER], HALF_LOW, lw=1.2, z=4)
            ax.axvline(years, color=ZERO, lw=0.5, ls=(0, (2, 2)), zorder=1)
            span_x = years * 1.12
            ax.set_xlim(0, span_x)
            ax.set_ylim(0.2, 1.03)
            ax.set_yticks([0.5, 1.0], ["50%", "100%"] if start == 0 else ["", ""])
            times = [0, years / 2, years]
            ax.set_xticks(times, ["0", f"{years / 2:g}", f"{years} y"])
            ax.tick_params(labelsize=5.9, pad=1)
            tidy(ax)
            ax.set_title(f"{cat} ({int(whole.n):,})", loc="left", fontsize=6.7, color=INK2, pad=1.5)
            for k, (r, colour) in enumerate(((lo, HALF_LOW), (hi, HALF_HIGH))):
                ax.text(0.04, 0.22 - 0.17 * k, f"{100 * r.survival:.0f}%", transform=ax.transAxes, fontsize=6.6,
                        color=colour, fontweight="bold", va="bottom")
            for k, (g, colour) in enumerate(((LOWER, HALF_LOW), (UPPER, HALF_HIGH))):
                for t in times:
                    fig.text((x0 + col_w * t / span_x) / W, rows_y[k], f"{_at_risk(parts[g], t):,}", fontsize=5.6,
                             color=colour, ha="center", va="center")
    for start, text, ha in ((0, "Lowest category", "left"), (3, "Highest category", "right")):
        x = left + start * (col_w + gap) + (col_w if ha == "right" else 0)
        fig.text(x / W, 1 - 0.3 / H, text, fontsize=7.0, color=MUTED, style="italic", ha=ha, va="center")
    fig.legend(handles=[Line2D([], [], color=HALF_LOW, lw=1.5, label="OPERA's lower-risk half"),
                        Line2D([], [], color=HALF_HIGH, lw=1.5, label="OPERA's higher-risk half"),
                        Line2D([], [], color=WHOLE, lw=1.0, ls=(0, (3, 2)), label="Whole index category")],
               loc="upper center", ncol=3, frameon=False, fontsize=7.2, bbox_to_anchor=(0.57, 1.0))
    _save(fig, out / stem)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("clinical", type=Path)
    p.add_argument("out", type=Path)
    a = p.parse_args()
    thesis_style()
    summary = pd.read_csv(a.clinical / "km_split_summary.csv")
    curves = pd.read_csv(a.clinical / "km_split_curves.csv")
    comps = _comparisons(summary)
    overview(summary, comps, a.out)
    km_page(summary, curves, comps[:6], a.out, "supp_index_splits_km_1")
    km_page(summary, curves, comps[6:], a.out, "supp_index_splits_km_2")


if __name__ == "__main__":
    main()
