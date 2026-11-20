"""Decision curves (supp_decision_curves) from the server's decision-curve export (20 Nov): for every
clinical-index comparison of Fig. 6, the net benefit per 100 patients of adapted OPERA, the clinical
index, treating all and treating none across threshold risks. No threshold is singled out: under each
curve a bar marks the thresholds at which adapted OPERA's net benefit exceeded the index's with a paired
bootstrap 95% interval excluding zero.

Usage:
  python -m scripts.render_decision_curves scratch/download_decision_curves_20261120 \
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

from opera.run.paper_story_figures import (CLINICAL_ROWS, INK, INK2, LABEL_SIZE, MUTED, OPERA, SIENNA, _save,
                                           lighten, thesis_style, tidy)

GREY = "#8A93A0"
STYLES = [("R_EA", "Adapted OPERA", OPERA, "-", 1.5), ("clinical_index", "Clinical index", SIENNA, "-", 1.1),
          ("treat_all", "Treat all", GREY, (0, (4, 2)), 0.9), ("treat_none", "Treat none", INK2, ":", 0.9)]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("dca", type=Path)
    p.add_argument("out", type=Path)
    a = p.parse_args()
    thesis_style()
    curves = pd.read_csv(a.dca / "decision_curves.csv")
    diffs = pd.read_csv(a.dca / "decision_curve_differences.csv")
    diffs = diffs[diffs.comparator.eq("clinical_index")]
    rows = [(sid, name, end) for sid, name, end in CLINICAL_ROWS if (curves.population == sid).any()]
    cols = 3
    nrow = int(np.ceil((len(rows) + 1) / cols))
    fig, axes = plt.subplots(nrow, cols, figsize=(6.9, 2.05 * nrow + 0.3))
    fig.subplots_adjust(left=0.08, right=0.985, top=0.95, bottom=0.07, hspace=0.62, wspace=0.28)
    for ax, (sid, name, end) in zip(axes.flat, rows):
        g = curves[curves.population.eq(sid)]
        tf = end.startswith("TF")
        xmax = 80 if tf else 50
        for strategy, label, colour, ls, lw in STYLES:
            part = g[g.strategy.eq(strategy)].sort_values("threshold")
            part = part[part.threshold * 100 <= xmax]
            ax.plot(part.threshold * 100, part.net_benefit * 100, color=colour, ls=ls, lw=lw, zorder=3)
        prev = g.prevalence.iloc[0] * 100
        lo = -0.25 * prev
        ax.set_ylim(lo, prev * 1.12)
        ax.set_xlim(0, xmax)
        # Thresholds where OPERA beat the index with an interval excluding zero.
        d = diffs[diffs.population.eq(sid) & diffs.status.eq("estimated")].sort_values("threshold")
        sig = d[(d.lower > 0) & (d.threshold * 100 <= xmax)].threshold.to_numpy() * 100
        step = 1 if not tf else 1
        for t in sig:
            ax.plot([t - step / 2, t + step / 2], [lo + 0.04 * (prev * 1.12 - lo)] * 2, color=OPERA, lw=3.2,
                    solid_capstyle="butt", zorder=4)
        tidy(ax)
        ax.tick_params(labelsize=6.6)
        horizon = "2 y" if "2y" in sid else "1 y"
        ax.set_title(f"{name}", loc="left", fontsize=8.2, fontweight="bold", color=INK, pad=11)
        ax.annotate(f"{'Treatment failure' if tf else 'Death'} within {horizon.replace(' y', ' year' if horizon == '1 y' else ' years')} (n={int(g.n.iloc[0]):,})",
                    xy=(0, 1), xycoords="axes fraction", xytext=(0, 2), textcoords="offset points",
                    fontsize=6.4, color=MUTED, style="italic", va="bottom")
    for ax in axes.flat[len(rows):]:
        ax.axis("off")
    leg = axes.flat[len(rows)]
    handles = [Line2D([], [], color=c, ls=ls, lw=lw, label=lab) for _, lab, c, ls, lw in STYLES]
    handles.append(Line2D([], [], color=OPERA, lw=3.2, solid_capstyle="butt",
                          label="OPERA above the index,\n95% interval excluding zero"))
    leg.legend(handles=handles, loc="center", frameon=False, fontsize=7.2, handlelength=2.0)
    fig.text(0.53, 0.012, "Threshold risk (%)", ha="center", fontsize=LABEL_SIZE, color=INK2)
    fig.text(0.012, 0.5, "Net benefit per 100 patients", va="center", rotation=90, fontsize=LABEL_SIZE,
             color=INK2)
    _save(fig, a.out / "supp_decision_curves")


if __name__ == "__main__":
    main()
