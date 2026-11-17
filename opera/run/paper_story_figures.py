"""Proposed main-figure set for the OPERA paper, restructured around its argument.

Draft for review (16 Nov 2026). Everything is read from existing held-out outputs;
nothing is refitted. Figures are written to ``paper/figures/results/story/`` so the
current figures are untouched.

  fig2_overview   Borrowing more, step by step: mean AUC on one common task panel and
                  the paired gain of each step.
  fig3_pooling    Whether to borrow (the current pooling figure, with the gain by
                  training support enlarged and shown as binned paired estimates).
  fig4_population What to borrow: two kinds of borrowing by training support, the
                  encoder steps for the joint model and OPERA, and the gain by disease.
  fig5_specialise How to borrow: specialisation and the gap to pooled XGBoost by encoder,
                  and where pooled XGBoost still leads.
  fig6_clinical   Against the clinical indices: discrimination and matched risk groups.

    python -m opera.run.paper_story_figures \
        --bundle scratch/download_bundle_20261112/tables \
        --cells scratch/registered_figures_20261114/plot_tables/paired_cells.csv \
        --clinical scratch/R_EA_clinical_review_20261115 \
        --out paper/figures/results/story
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import to_rgb  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.ticker import FuncFormatter, MaxNLocator, MultipleLocator  # noqa: E402

from opera.visualization import paper_figures as pf  # noqa: E402
from opera.visualization.style import LABEL_SIZE, PALETTE, TICK_SIZE, despine, setup_style  # noqa: E402

INK, INK2, MUTED = "#17212B", "#485463", "#657180"
GRID, ZERO = "#E8ECF1", "#737D8C"
S_COL, D_COL, O_COL, J_COL = pf.MODEL_STYLE["S"][0], pf.CONTRAST_STYLE["D-S"][0], pf.CONTRAST_STYLE["O-S"][0], pf.MODEL_STYLE["J"][0]
NAT, ADAPT, OPERA = pf.ORIGIN_STYLE["E"][0], pf.ORIGIN_STYLE["EA"][0], pf.MODEL_STYLE["OPERA"][0]
XP_COL, SIENNA = "#B18B4D", "#B5451B"
XGB_REF = "#8A93A0"  # per-task XGBoost as a grey reference line
BINS = [("(-1.001, 19.0]", "<20"), ("(19.0, 99.0]", "20–99"), ("(99.0, 499.0]", "100–499"), ("(499.0, inf]", "≥500")]


# -- helpers -------------------------------------------------------------------------
def thesis_style():
    """The thesis figures' look (thesis/figures/kappa/paper3_takeaways.py): TeX Gyre Pagella,
    which is also the manuscript's body face, muted ink and hairline axes."""
    import subprocess
    from matplotlib import font_manager
    for name in ("regular", "bold", "italic", "bolditalic"):
        try:
            path = subprocess.run(["kpsewhich", f"texgyrepagella-{name}.otf"], capture_output=True, text=True,
                                  env={"PATH": "/Library/TeX/texbin:/usr/bin"}).stdout.strip()
            if path:
                font_manager.fontManager.addfont(path)
        except OSError:
            pass
    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["TeX Gyre Pagella", "DejaVu Serif"], "mathtext.fontset": "custom",
        "font.size": 8.5, "text.color": INK, "axes.labelcolor": INK2, "axes.titlecolor": INK,
        "xtick.color": INK2, "ytick.color": INK2, "axes.edgecolor": "#CBD2DA", "axes.linewidth": 0.6,
        "xtick.labelsize": 8, "ytick.labelsize": 8, "axes.labelsize": 8.5, "legend.fontsize": 7.8,
        "pdf.fonttype": 42, "ps.fonttype": 42})


def lighten(colour, amount):
    return tuple(1 - amount * (1 - c) for c in to_rgb(colour))


def minus(text):
    return text.replace("-", "−")


def title(ax, letter, text, sub=None, x=0.0, keep_sub=False):
    """Thesis headings: bold letter and title; the italic line beneath only where asked
    (detail belongs in the caption, not on the figure)."""
    sub = sub if keep_sub else None
    ax.set_title(f"{letter}   {text}", loc="left", fontsize=9.6, fontweight="bold", pad=17 if sub else 7,
                 x=x)
    if sub:
        ax.annotate(sub, xy=(x, 1), xycoords="axes fraction", xytext=(14, 4), textcoords="offset points",
                    fontsize=7.8, color=MUTED, style="italic", va="bottom")


def tidy(ax, grid="y"):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(axis=grid, color=GRID, lw=0.6)
    ax.grid(axis="x" if grid == "y" else "y", visible=False)
    ax.tick_params(length=0)
    ax.set_axisbelow(True)


def estimate(ax, x, y, lo, hi, colour, marker="D", size=34, horizontal=False, filled=True, z=3):
    """Paper style: a tinted interval with slight whiskers and a white-edged marker."""
    err = [[y - lo], [hi - y]] if not horizontal else [[x - lo], [hi - x]]
    if horizontal:
        ax.errorbar([x], [y], xerr=err, fmt="none", ecolor=lighten(colour, 0.5), elinewidth=2.0,
                    capsize=2.4, capthick=1.0, zorder=z - 1)
    else:
        ax.errorbar([x], [y], yerr=err, fmt="none", ecolor=lighten(colour, 0.5), elinewidth=2.0,
                    capsize=2.4, capthick=1.0, zorder=z - 1)
    ax.scatter([x], [y], marker=marker, s=size, color=colour if filled else "white", edgecolor=colour if not filled
               else "white", linewidth=1.1, zorder=z)


class Bundle:
    def __init__(self, tables: Path):
        e = pd.read_csv(tables / "effect_summaries.csv.gz", low_memory=False)
        self.e = e[e.split.eq("held_out") & e.seed.eq(42) & e.metric.eq("auc") & e.horizon_days.eq(0)]
        self.tables = tables

    def row(self, contrast, stratum="all_cells", value="all"):
        r = self.e[(self.e.contrast == contrast) & (self.e.stratum == stratum) & (self.e.stratum_value == value)]
        assert len(r) == 1, (contrast, stratum, value, len(r))
        return r.iloc[0]

    def by(self, contrast, stratum):
        return self.e[(self.e.contrast == contrast) & (self.e.stratum == stratum)].set_index("stratum_value")

    def panel_means(self, models, n_boot=2000, seed=2026):
        """Equal-cell mean AUC over the cells where every model is evaluated, with
        intervals from resampling cells (descriptive)."""
        est = pd.read_csv(self.tables / "estimates.csv.gz", low_memory=False)
        est = est[est.metric.eq("auc") & est.seed.eq(42) & est.split.eq("held_out") & est.model.isin(models)]
        wide = est.pivot_table(index=["cohort", "outcome", "horizon_days"], columns="model", values="estimate").dropna()
        rng = np.random.default_rng(seed)
        idx = rng.integers(0, len(wide), (n_boot, len(wide)))
        out = {}
        for m in models:
            v = wide[m].to_numpy()
            boot = v[idx].mean(1)
            out[m] = (v.mean(), *np.percentile(boot, [2.5, 97.5]))
        return out, len(wide)


# -- Figure 2: the overview ------------------------------------------------------------
STEPS = [  # (model, label, colour)
    ("S_L", "Single\ntask", S_COL),
    ("D_L", "Across\ndiseases", D_COL),
    ("J_L", "Across\noutcomes", J_COL),
    ("J_E", "National\nencoder", NAT),
    ("J_EA", "Adapted\nencoder", ADAPT),
]
STEP_GAINS = [  # (contrast, label, colour)
    ("P1", "Pool across diseases", D_COL),
    ("P4", "Add the other outcomes", J_COL),
    ("E1", "Pretrain nationally", NAT),
    ("A1", "Adapt to the cohort", ADAPT),
    ("EXT_ARCH_EA", "Specialise (OPERA)", OPERA),
    ("EXT_R_EA_VS_XP", "OPERA vs pooled XGBoost", XP_COL),
]


def fig2_overview(b: Bundle, out: Path):
    means, n_cells = b.panel_means([m for m, *_ in STEPS] + ["R_EA", "XP"])
    fig = plt.figure(figsize=(6.4, 3.4))
    g = fig.add_gridspec(1, 2, width_ratios=[1.3, 1], wspace=0.28, left=0.085, right=0.975, top=0.9, bottom=0.17)
    ax = fig.add_subplot(g[0])
    for i, (m, label, col) in enumerate(STEPS):
        v, lo, hi = means[m]
        ax.bar(i, v - 0.5, bottom=0.5, width=0.62, color=col, zorder=2)
        ax.errorbar([i], [v], yerr=[[v - lo], [hi - v]], fmt="none", ecolor="#5B6676", elinewidth=1.6,
                    capsize=3.0, capthick=1.0, zorder=3)
        ax.text(i, hi + 0.006, f"{v:.3f}", ha="center", va="bottom", fontsize=8, color=INK, weight="bold")
    ax.set_xticks(range(len(STEPS)), [lab for _, lab, _ in STEPS], fontsize=6.9)
    ax.set_ylim(0.5, 0.8)
    ax.yaxis.set_major_locator(MultipleLocator(0.1))
    ax.set_ylabel("Mean AUC, held-out patients", fontsize=LABEL_SIZE)
    tidy(ax)
    title(ax, "A", "Borrowing more, step by step", f"The same {n_cells:,} disease–outcome–horizon tasks")

    bx = fig.add_subplot(g[1])
    est = pd.read_csv(b.tables / "estimates.csv.gz", low_memory=False)
    est = est[est.metric.eq("auc") & est.seed.eq(42) & est.split.eq("held_out")
              & est.model.isin([m for m, *_ in STEPS] + ["X"])]
    wide = est.pivot_table(index=["cohort", "outcome", "horizon_days"], columns="model", values="estimate").dropna()
    byh = wide.groupby("horizon_days").mean()
    print(f"fig2 B: {len(wide):,} tasks evaluable for all six models")
    xs = np.arange(len(byh))
    bx.plot(xs, byh["X"], color=XGB_REF, lw=1.3, ls=(0, (4, 2)), zorder=2)
    bx.scatter(xs, byh["X"], color=XGB_REF, marker="s", s=14, edgecolor="white", linewidth=0.7, zorder=3)
    for m, _, col in STEPS:
        bx.plot(xs, byh[m], color=col, lw=1.6, zorder=2)
        bx.scatter(xs, byh[m], color=col, s=16, edgecolor="white", linewidth=0.7, zorder=3)
    bx.set_xticks(xs, [str(h) for h in byh.index])
    bx.set_xlabel("Horizon (days)", fontsize=LABEL_SIZE)
    bx.set_ylim(0.5, 0.85)
    bx.yaxis.set_major_locator(MultipleLocator(0.1))
    tidy(bx)
    bx.legend(handles=[Line2D([], [], color=col, marker="o", markersize=4, lw=1.4, label=lab.replace("\n", " "))
                       for _, lab, col in STEPS]
                      + [Line2D([], [], color=XGB_REF, marker="s", markersize=3.6, lw=1.2, ls=(0, (4, 2)),
                                label="XGBoost per task")],
              loc="lower left", ncol=2, columnspacing=0.8, frameon=False,
              fontsize=6.8, handletextpad=0.4, labelspacing=0.3)
    title(bx, "B", "At every horizon")
    _save(fig, out / "fig2_overview")


# -- Figure 3: whether to borrow ---------------------------------------------------------
def support_panel(ax, b: Bundle, series, ylabel="AUC gain (95% CI)", annotate=True):
    xs = np.arange(len(BINS))
    for k, (contrast, label, col, offset) in enumerate(series):
        rows = b.by(contrast, "support_bin").reindex([key for key, _ in BINS])
        x = xs + offset
        ax.plot(x, rows.gain, color=lighten(col, 0.55), lw=1.3, zorder=2)
        for xi, r in zip(x, rows.itertuples()):
            estimate(ax, xi, r.gain, r.lower, r.upper, col, size=40)
    ax.axhline(0, color=ZERO, lw=0.8, zorder=1)
    ax.set_xticks(xs, [lab for _, lab in BINS])
    ax.set_xlim(-0.4, len(BINS) - 0.6)
    if annotate:
        ax.legend(handles=[Line2D([], [], marker="D", color=lighten(col, 0.55), markerfacecolor=col,
                                  markeredgecolor="white", markersize=6, lw=1.3, label=label)
                           for _, label, col, _ in series], loc="upper right", frameon=False, fontsize=7.5)
    ax.set_xlabel("Training events for the task", fontsize=LABEL_SIZE)
    ax.set_ylabel(ylabel, fontsize=LABEL_SIZE)
    ax.yaxis.set_major_locator(MultipleLocator(0.05))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: "0" if abs(v) < 1e-12 else minus(f"{v:+.2f}")))
    tidy(ax)


def fig3_pooling(b: Bundle, cells_path: Path, out: Path):
    cells = pd.read_csv(cells_path, low_memory=False)
    metric, horizon = "auc", pf.ALL_HORIZONS
    fig = plt.figure(figsize=(6.4, 8.6))
    top = fig.add_gridspec(1, 2, width_ratios=[1.55, 1], wspace=0.06, left=0.29, right=0.86, top=0.955, bottom=0.53)
    columns = pf._support_totals(cells, "endpoint_group", horizon).sort_values().index.tolist()
    ax = fig.add_subplot(top[0])
    image = pf.group_matrix(ax, cells, contrast="J-S", order=columns, metric=metric, horizon=horizon,
                            row_order="support",
                            vmax=pf.shared_gain_limit(cells.loc[cells.contrast == "J-S"], metric=metric, horizon=horizon))
    rows = pf._axis_order(cells, "cohort", "cohort_group", order="support", horizon=horizon)[0]
    pf._support_strip(ax, pf._support_totals(cells, "cohort", horizon).reindex(rows), axis="y", labels=rows)
    title(ax, "A", "Gain by disease and domain")
    bx = fig.add_subplot(top[1], sharey=ax)
    year = cells.loc[(cells.metric == metric) & pf._horizon_mask(cells, horizon)]
    for offset, contrast in zip((-0.22, 0, 0.22), ("D-S", "O-S", "J-S")):
        color, _ = pf.CONTRAST_STYLE[contrast]
        paired = year.loc[year.contrast == contrast, ["cohort", "disease_mean", "disease_lower", "disease_upper"]] \
            .drop_duplicates().set_index("cohort").reindex(rows)
        bx.hlines(np.arange(len(rows)) + offset, paired.disease_lower, paired.disease_upper, color=color, lw=0.9, alpha=0.5)
        bx.scatter(paired.disease_mean.to_numpy(float), np.arange(len(rows)) + offset, marker="D",
                   s=13 if contrast == "J-S" else 9, color=color, zorder=3)
    bx.axvline(0, color=PALETTE["zero_line"], lw=0.7)
    bx.tick_params(axis="y", labelleft=False, length=0)
    bx.xaxis.set_major_locator(MaxNLocator(3))
    bx.set_xlabel("AUC gain over specialist\n(mean, paired 95% CI)")
    bx.grid(axis="y", visible=False)
    title(bx, "B", "By pooling axis")
    despine(bx, "x")
    box = ax.get_position()
    cax = fig.add_axes([box.x0, 0.415, box.width, 0.010])
    cb = fig.colorbar(image, cax=cax, orientation="horizontal")
    cb.set_label("All-horizon AUC gain, joint vs specialist (median within domain)", fontsize=7.2)
    cb.ax.tick_params(labelsize=6.5)
    cb.outline.set_visible(False)

    bx.legend(handles=[Line2D([], [], marker="D", ls="", color=pf.CONTRAST_STYLE[c][0], markersize=4.5, label=n)
                       for c, n in (("D-S", "Across diseases"), ("O-S", "Across outcomes"), ("J-S", "Both (joint)"))],
              loc="upper left", bbox_to_anchor=(0.02, -0.17), frameon=False, fontsize=7.4, handletextpad=0.3)
    bottom = fig.add_gridspec(1, 2, width_ratios=[1.45, 1], wspace=0.55, left=0.10, right=0.98, top=0.325, bottom=0.06)
    cx = fig.add_subplot(bottom[0])
    support_panel(cx, b, [("P1", "Across diseases", pf.CONTRAST_STYLE["D-S"][0], -0.12),
                          ("P5", "Both (joint)", pf.CONTRAST_STYLE["J-S"][0], 0.0),
                          ("P2", "Across outcomes", pf.CONTRAST_STYLE["O-S"][0], 0.12)],
                  ylabel="AUC gain over specialist (95% CI)")
    avg = b.row("P1").gain
    cx.axhline(avg, color=lighten(pf.CONTRAST_STYLE["D-S"][0], 0.5), lw=0.9, ls=(0, (3, 2)), zorder=1)
    cx.annotate(minus(f"average {avg:+.3f}"), xy=(1, avg), xycoords=("axes fraction", "data"),
                xytext=(-3, -3), textcoords="offset points", fontsize=6.8, color=pf.CONTRAST_STYLE["D-S"][0],
                style="italic", ha="right", va="top")
    title(cx, "C", "The less a task has, the more it gains", "Paired gain over the specialists by training events")
    dx = fig.add_subplot(bottom[1])
    data = cells.loc[(cells.metric == metric) & pf._horizon_mask(cells, horizon) & cells.contrast.isin(["D-S", "O-S", "J-S"])]
    order = (data.loc[data.contrast == "J-S"].groupby("endpoint_group").gain.median().sort_values(ascending=False).index.tolist())
    summary = pf._summary_rows(data, ["endpoint_group", "contrast"])
    summary["endpoint_group"] = summary.endpoint_group.map(lambda g: pf.SHORT_GROUP.get(g, g))
    pf._diamond_forest(dx, summary, row_key="endpoint_group", row_order=[pf.SHORT_GROUP.get(g, g) for g in order],
                       series_key="contrast", series_style={k: pf.CONTRAST_STYLE[k] for k in ("D-S", "O-S", "J-S")},
                       xlabel="AUC gain (mean, 95% CI)", show_counts=False)
    dx.xaxis.set_major_locator(MaxNLocator(4))
    title(dx, "D", "By clinical domain")
    _save(fig, out / "fig3_pooling")


# -- Figure 4: what to borrow ------------------------------------------------------------
DISEASE_ORDER_KEY = "subtype_size"


def fig4_population(b: Bundle, cells_path: Path, out: Path):
    """Two kinds of borrowing by training events: pooling across diseases (sequence model and
    XGBoost) falls with support; pretraining on the population does not. In the style of
    Figures 3C and 5: diamonds with tinted intervals, the average as a dashed line."""
    panels = [("P1", "Sequence model", D_COL), ("T7", "XGBoost", XP_COL), ("E3", "Sequence model", ADAPT)]
    fig = plt.figure(figsize=(6.4, 2.85))
    g = fig.add_gridspec(1, 4, width_ratios=[1, 1, 0.12, 1], left=0.1, right=0.99, top=0.78, bottom=0.2,
                         wspace=0.14)
    first = fig.add_subplot(g[0])
    axes = [first, fig.add_subplot(g[1], sharey=first), fig.add_subplot(g[3], sharey=first)]
    xs = np.arange(len(BINS))
    for k, (ax, (c, name, col)) in enumerate(zip(axes, panels)):
        rows = b.by(c, "support_bin").reindex([key for key, _ in BINS])
        ax.axhline(b.row(c).gain, color=lighten(col, 0.5), lw=0.9, ls=(0, (3, 2)), zorder=1)
        ax.plot(xs, rows.gain, color=lighten(col, 0.55), lw=1.3, zorder=2)
        for xi, r in zip(xs, rows.itertuples()):
            estimate(ax, xi, r.gain, r.lower, r.upper, col, size=40)
        ax.axhline(0, color=ZERO, lw=0.8, zorder=1)
        ax.set_xticks(xs, [lab for _, lab in BINS])
        ax.set_xlim(-0.5, len(BINS) - 0.5)
        tidy(ax)
        if k:
            ax.spines["left"].set_visible(False)
            ax.tick_params(axis="y", labelleft=False)
        title(ax, "ABC"[k], name)
    axes[0].set_ylabel("AUC gain (95% CI)", fontsize=LABEL_SIZE)
    axes[0].yaxis.set_major_locator(MultipleLocator(0.05))
    axes[0].yaxis.set_major_formatter(FuncFormatter(lambda v, _: "0" if abs(v) < 1e-12 else minus(f"{v:+.2f}")))
    fig.text(0.545, 0.03, "Training events for the task", ha="center", fontsize=LABEL_SIZE, color=INK2)
    # one bracket over each kind of borrowing
    for (first, last), label in (((0, 1), "Pooling across diseases"), ((2, 2), "Pretraining on the population")):
        x0, x1 = axes[first].get_position().x0, axes[last].get_position().x1
        y = 0.905
        fig.add_artist(Line2D([x0, x1], [y - 0.025, y - 0.025], transform=fig.transFigure, color=INK2, lw=0.7))
        fig.text((x0 + x1) / 2, y, label, ha="center", va="bottom", fontsize=9.0, color=INK, style="italic")
    _save(fig, out / "fig4_population")


def supp_population_disease(b: Bundle, cells_path: Path, out: Path):
    """The adapted-versus-cohort encoder gain by disease (formerly Figure 5C)."""
    cells = pd.read_csv(cells_path, low_memory=False)
    size = pf._support_totals(cells, "cohort", pf.ALL_HORIZONS).sort_values()
    rows = b.by("E3", "cohort")
    alias = {"Amyloidosis": "AMYLOIDOSIS", "Solitary plasmacytoma": "SolM", "Transformed FL": "TRANSFORMED_FL"}
    present = [d for d in size.index if alias.get(d, d) in rows.index]
    fig, ax = plt.subplots(figsize=(4.2, 5.2))
    fig.subplots_adjust(left=0.36, right=0.96, top=0.95, bottom=0.1)
    for i, d in enumerate(present):
        r = rows.loc[alias.get(d, d)]
        estimate(ax, r.gain, i, r.lower, r.upper, ADAPT, horizontal=True, size=16)
    ax.axvline(0, color=ZERO, lw=0.8, zorder=1)
    ax.axvline(b.row("E3").gain, color=lighten(ADAPT, 0.5), lw=1.0, ls=(0, (3, 2)), zorder=1)
    ax.set_yticks(range(len(present)), present)
    ax.tick_params(axis="y", labelsize=7.4)
    ax.set_ylim(len(present) - 0.5, -0.6)
    ax.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(0.05))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: "0" if abs(v) < 1e-12 else minus(f"{v:+.2f}")))
    ax.set_xlabel("AUC gain, adapted vs cohort encoder", fontsize=LABEL_SIZE)
    tidy(ax, grid="x")
    ax.spines["left"].set_visible(False)
    _save(fig, out / "supp_population_disease")


# -- Figure 5: how to borrow ------------------------------------------------------------
def fig5_specialise(b: Bundle, out: Path):
    fig = plt.figure(figsize=(6.4, 6.4))
    g = fig.add_gridspec(2, 2, height_ratios=[1, 1.15], hspace=0.62, wspace=0.42, left=0.11, right=0.975,
                         top=0.92, bottom=0.08)
    panels = [(fig.add_subplot(g[0, 0]), "A", "OPERA's specialised parts", "OPERA minus the joint model",
               ["EXT_ARCH_L", "EXT_ARCH_E", "EXT_ARCH_EA"], OPERA, "o", 3, 0.005),
              (fig.add_subplot(g[0, 1]), "B", "The gap to pooled XGBoost", "OPERA minus pooled XGBoost",
               ["EXT_J_OA_OWN_L_VS_XP", "EXT_R_E_VS_XP", "EXT_R_EA_VS_XP"], SIENNA, "D", 2, 0.01)]
    for ax, letter, name, sub, contrasts, col, mk, digits, step in panels:
        rows = [b.row(c) for c in contrasts]
        x = [0, 1, 2]
        ax.plot(x, [r.gain for r in rows], color=lighten(col, 0.55), lw=1.2, zorder=2)
        for xi, r in zip(x, rows):
            estimate(ax, xi, r.gain, r.lower, r.upper, col, marker=mk, size=38)
            ax.annotate(minus(f"{r.gain:+.3f}"), (xi, r.gain), xytext=(8, 0), textcoords="offset points",
                        va="center", fontsize=7.2, color=INK2)
        ax.axhline(0, color=ZERO, lw=0.8, zorder=1)
        ax.set_xticks(x, ["Cohort", "National", "Adapted"])
        ax.set_xlim(-0.35, 2.6)
        ax.set_xlabel("Encoder", fontsize=LABEL_SIZE)
        ax.yaxis.set_major_locator(MultipleLocator(step))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _, k=digits: "0" if abs(v) < 1e-12 else minus(f"{v:+.{k}f}")))
        tidy(ax)
        title(ax, letter, name, sub)
    panels[0][0].set_ylabel("AUC difference (95% CI)", fontsize=LABEL_SIZE)
    panels[1][0].set_ylim(-0.045, 0.006)

    cx = fig.add_subplot(g[1, :])
    rows = b.by("EXT_R_EA_VS_XP", "endpoint_group").sort_values("gain")
    for i, (grp, r) in enumerate(rows.iterrows()):
        col = SIENNA if r.upper < 0 else (OPERA if r.lower > 0 else "#8A929C")
        estimate(cx, r.gain, i, r.lower, r.upper, col, horizontal=True, size=26)
    cx.axvline(0, color=ZERO, lw=0.8, zorder=1)
    cx.set_yticks(range(len(rows)), [pf.SHORT_GROUP.get(gname, gname) for gname in rows.index], fontsize=8)
    cx.set_ylim(len(rows) - 0.5, -0.6)
    cx.xaxis.set_major_formatter(FuncFormatter(lambda v, _: "0" if abs(v) < 1e-12 else minus(f"{v:+.2f}")))
    cx.set_xlabel("Adapted OPERA minus pooled XGBoost, AUC (95% CI)", fontsize=LABEL_SIZE)
    cx.annotate("\u2190 pooled XGBoost better", xy=(0, 1), xycoords=("data", "axes fraction"), xytext=(-6, 2),
                textcoords="offset points", ha="right", va="bottom", fontsize=7, color=SIENNA)
    cx.annotate("OPERA better \u2192", xy=(0, 1), xycoords=("data", "axes fraction"), xytext=(6, 2),
                textcoords="offset points", ha="left", va="bottom", fontsize=7, color=OPERA)
    tidy(cx, grid="x")
    cx.spines["left"].set_visible(False)
    cx.set_title("C   Where pooled XGBoost still leads", loc="left", fontsize=9.6, fontweight="bold", pad=16)
    _save(fig, out / "fig5_specialise")


# -- Figure 6: against the clinical indices ------------------------------------------------
CLINICAL_ROWS = [
    ("ipsswm_all_subtypes_native", "IPSS-WM", "OS"), ("malt_ipi_all_subtypes_native", "MALT-IPI", "OS"),
    ("cll_ipi_all_subtypes_native", "CLL-IPI", "OS"), ("mipi_all_subtypes_native", "MIPI", "OS"),
    ("r_ipi_all_subtypes_native", "R-IPI", "OS"), ("r_iss_all_subtypes_native", "R-ISS", "OS"),
    ("ips_all_subtypes_native", "IPS", "TF"), ("ipi_computed_native", "IPI (computed)", "OS"),
    ("nccn_ipi_native", "NCCN-IPI", "OS"), ("nccn_ipi_treatment_failure_2y_native", "NCCN-IPI", "TF, 2 y"),
    ("flipi2_all_subtypes_native", "FLIPI2", "TF"),
]


def endpoint_label(row, short=True):
    """Endpoint and horizon from the results themselves (FLIPI2 moved to overall survival)."""
    end = "OS" if row.outcome == "overall_survival" else "TF"
    years = int(round(row.horizon_days / 365))
    return end if (short and years == 1) else f"{end}, {years} y"


def _population_survival(figures: Path):
    """Survival at the horizon of all patients with the index: the size-weighted KM estimates
    of the index's categories (approximate; the categories partition the population)."""
    r = pd.read_csv(figures / "risk_group_point_estimates.csv")
    r = r[r.system != "model_group"]
    return (r.n * r.estimate).groupby(r.score_id).sum() / r.n.groupby(r.score_id).sum()


def degenerate(r, side, population):
    """The index's extreme category is no more extreme than all patients with the index, so any
    group, up to everyone, matches it: the comparison carries no information."""
    pop = population.get(r.name, np.nan)
    surv = r[f"{side}_index_survival"]
    return bool(np.isfinite(pop) and (surv <= pop if side == "low" else surv >= pop))


def fig6_clinical(clinical: Path, out: Path, variant="dots",
                  figures: Path = Path("scratch/R_EA_clinical_figures")):
    m = pd.read_csv(clinical / "matched_risk_and_auc.csv").set_index("score_id")
    population = _population_survival(figures)
    fig = plt.figure(figsize=(6.4, 6.1))
    g = fig.add_gridspec(2, 3, width_ratios=[1.45, 0.62, 0.62], height_ratios=[1, 0.3], wspace=0.12, hspace=0.62,
                         left=0.27, right=0.975, top=0.9, bottom=0.08)
    ax = fig.add_subplot(g[0, 0])
    n = len(CLINICAL_ROWS)
    labels = []
    for i, (sid, name, end) in enumerate(CLINICAL_ROWS):
        r = m.loc[sid]
        ax.plot([r.index_auc, r.model_auc], [i, i], color=lighten(OPERA, 0.3), lw=2.2, zorder=1, solid_capstyle="round")
        ax.scatter([r.index_auc], [i], marker="D", s=28, color=SIENNA, edgecolor="white", linewidth=1, zorder=3)
        ax.scatter([r.model_auc], [i], marker="D", s=34, color=OPERA, edgecolor="white", linewidth=1, zorder=3)
        ax.annotate(minus(f"{r.gain:+.2f}"), (max(r.index_auc, r.model_auc), i), xytext=(7, 0),
                    textcoords="offset points", va="center", fontsize=7, color=INK2 if r.lower > 0 else MUTED,
                    weight="bold" if r.lower > 0 else "normal")
        labels.append(f"{name}, {endpoint_label(r)} (n={int(r.n_patients):,})")
    top = m.loc[CLINICAL_ROWS[0][0]]
    ax.annotate("index", (top.index_auc, 0), xytext=(0, 7), textcoords="offset points", ha="center", va="bottom",
                fontsize=7, color=SIENNA, style="italic")
    ax.annotate("OPERA", (top.model_auc, 0), xytext=(0, 7), textcoords="offset points", ha="center", va="bottom",
                fontsize=7, color=OPERA, style="italic")
    ax.set_yticks(range(n), labels)
    ax.tick_params(axis="y", labelsize=7.6)
    ax.set_ylim(n - 0.5, -1.0)
    ax.set_xlim(0.40, 1.0)
    ax.set_xticks([0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    ax.axvline(0.5, color=ZERO, lw=0.7, ls=(0, (2, 2)), zorder=0)
    ax.set_xlabel("AUC", fontsize=LABEL_SIZE)
    tidy(ax, grid="x")
    ax.spines["left"].set_visible(False)
    title(ax, "A", "Discrimination", "Same held-out patients", keep_sub=True)

    for k, (side, col, letter, name, sub) in enumerate((("low", D_COL, "B", "Low risk", "as safe as the index"),
                                                       ("high", SIENNA, "C", "High risk", "as risky as the index"))):
        bx = fig.add_subplot(g[0, 1 + k], sharey=ax)
        if variant == "bars":
            _matched_bars(bx, m, side, letter, name, sub, first=k == 0)
            continue
        if variant == "dots":
            _matched_counts(bx, m, side, letter, name, sub, first=k == 0, population=population)
            continue
        for i, (sid, *_r) in enumerate(CLINICAL_ROWS):
            r = m.loc[sid]
            if r[f"{side}_status"] != "estimable":
                bx.text(45, i, "n.e.", ha="center", va="center", fontsize=6.6, color=MUTED, style="italic")
                continue
            if sid.startswith("flipi2") and side == "high":  # selects every patient: not informative
                bx.text(45, i, "all patients", ha="center", va="center", fontsize=6.6, color=MUTED, style="italic")
                continue
            p = 100 * r[f"{side}_additional_n"] / r.n_patients
            lo = 100 * r[f"{side}_additional_n_lower"] / r.n_patients
            hi = 100 * r[f"{side}_additional_n_upper"] / r.n_patients
            if np.isfinite(lo) and np.isfinite(hi):
                estimate(bx, p, i, lo, hi, col, horizontal=True, size=24, filled=bool(lo > 0))
            else:
                bx.scatter([p], [i], marker="D", s=24, color="white", edgecolor=col, linewidth=1.1, zorder=3)
        bx.axvline(0, color=ZERO, lw=0.8, zorder=1)
        bx.set_xlim(-25, 100)
        bx.set_xticks([0, 50, 100])
        bx.tick_params(axis="y", labelleft=False)
        tidy(bx, grid="x")
        bx.spines["left"].set_visible(False)
        title(bx, letter, name, sub, keep_sub=True)
        if k == 0:
            bx.set_xlabel("Additional patients at matched risk\n(% of patients, 95% CI)", fontsize=LABEL_SIZE,
                          x=1.1, ha="center")

    cx = fig.add_subplot(g[1, :])
    r = m.loc["r_iss_all_subtypes_native"]
    rows = [("R-ISS low risk", r.low_index_n, 1 - r.low_index_survival, SIENNA),
            ("Adapted OPERA, same risk", r.low_model_n, 1 - r.low_model_survival, OPERA)]
    for i, (lab, nn, risk, col) in enumerate(rows):
        cx.barh(i, nn, height=0.55, color=col, zorder=2)
        cx.text(nn + 12, i, f"{int(nn):,} patients; {100 * risk:.1f}% died within a year", va="center",
                fontsize=7.6, color=INK2)
    cx.set_yticks([0, 1], [lab for lab, *_ in rows])
    cx.tick_params(axis="y", labelsize=8)
    cx.set_ylim(1.6, -0.6)
    cx.set_xlim(0, 1029)
    cx.set_xlabel(f"Patients with R-ISS recorded (n={int(r.n_patients):,})", fontsize=LABEL_SIZE)
    tidy(cx, grid="x")
    cx.spines["left"].set_visible(False)
    title(cx, "D", "One example: who is at low risk?",
          "The largest OPERA group with one-year mortality no higher than R-ISS low risk")
    _save(fig, out / ("fig6_clinical" if variant == "dots" else "fig6_clinical_bars"))


def _matched_counts(bx, m, side, letter, name, sub, first, population):
    """How many times as many patients OPERA places at the index's observed risk (log scale), with
    its 95% interval; the counts (index -> OPERA) are written beside each point."""
    col = OPERA
    for i, (sid, *_r) in enumerate(CLINICAL_ROWS):
        r = m.loc[sid]
        if r[f"{side}_status"] != "estimable":
            bx.text(1.06, i, "n.e.", ha="left", va="center", fontsize=6.6, color=MUTED, style="italic")
            continue
        if degenerate(r, side, population):
            bx.text(1.06, i, "index not beyond average", ha="left", va="center", fontsize=6.3,
                    color=MUTED, style="italic")
            continue
        a, b_ = r[f"{side}_index_n"], r[f"{side}_model_n"]
        lo, hi = r[f"{side}_additional_n_lower"], r[f"{side}_additional_n_upper"]
        ratio = b_ / a
        sure = bool(np.isfinite(lo) and lo > 0)
        if np.isfinite(lo) and np.isfinite(hi):
            estimate(bx, ratio, i, max((a + lo) / a, 0.5), (a + hi) / a, col, horizontal=True, size=24,
                     filled=sure)
        else:
            bx.scatter([ratio], [i], marker="D", s=24, color="white", edgecolor=col, linewidth=1.1, zorder=3)
        bx.annotate(f"{int(a):,}\u2192{int(b_):,}", (1, i), xycoords=("axes fraction", "data"),
                    xytext=(-1, -6.2), textcoords="offset points", ha="right", va="center", fontsize=5.6,
                    color=MUTED)
    bx.axvline(1, color=ZERO, lw=0.8, zorder=1)
    bx.set_xscale("log")
    bx.set_xlim(0.8, 12)
    bx.set_xticks([1, 2, 5, 10], ["1\u00d7", "2\u00d7", "5\u00d7", "10\u00d7"])
    bx.minorticks_off()
    bx.tick_params(axis="y", labelleft=False)
    tidy(bx, grid="x")
    bx.spines["left"].set_visible(False)
    title(bx, letter, name, sub, keep_sub=True)
    if first:
        bx.set_xlabel("Patients identified at the same observed risk,\nOPERA relative to the index",
                      fontsize=LABEL_SIZE, x=1.1, ha="center")


def _matched_bars(bx, m, side, letter, name, sub, first):
    """Panel D's presentation for every index: the index's extreme category and the largest
    OPERA group at the same observed risk, as shares of the patients with the index."""
    h = 0.34
    for i, (sid, *_r) in enumerate(CLINICAL_ROWS):
        r = m.loc[sid]
        if r[f"{side}_status"] != "estimable" or (sid.startswith("flipi2") and side == "high"):
            text = "n.e." if r[f"{side}_status"] != "estimable" else "all patients"
            bx.text(4, i, text, ha="left", va="center", fontsize=6.6, color=MUTED, style="italic")
            continue
        n = r.n_patients
        idx, mod = 100 * r[f"{side}_index_n"] / n, 100 * r[f"{side}_model_n"] / n
        lo, hi = r[f"{side}_additional_n_lower"], r[f"{side}_additional_n_upper"]
        sure = np.isfinite(lo) and lo > 0
        bx.barh(i - h / 2, idx, height=h, color=SIENNA, zorder=2)
        bx.barh(i + h / 2, mod, height=h, color=OPERA if sure else lighten(OPERA, 0.45), zorder=2)
        if np.isfinite(lo) and np.isfinite(hi):
            a, b_ = 100 * (r[f"{side}_index_n"] + lo) / n, 100 * (r[f"{side}_index_n"] + hi) / n
            bx.errorbar([mod], [i + h / 2], xerr=[[mod - a], [b_ - mod]], fmt="none", ecolor="#5B6676",
                        elinewidth=0.9, capsize=1.8, capthick=0.8, zorder=3)
    bx.set_xlim(0, 106)
    bx.set_xticks([0, 50, 100])
    bx.tick_params(axis="y", labelleft=False)
    tidy(bx, grid="x")
    bx.spines["left"].set_visible(False)
    title(bx, letter, name, sub, keep_sub=True)
    if first:
        bx.set_xlabel("Patients in the group at the same observed risk\n(% of patients with the index)",
                      fontsize=LABEL_SIZE, x=1.1, ha="center")


# -- Clinical KM splits: what the continuous model adds within the index's categories ------
LOW_HALF, HIGH_HALF = "#3E8A84", SIENNA


def _km_step(ax, part, colour, lw=1.6, ls="-", band=True, z=2):
    ax.step(part.time_days / 365, part.survival, where="post", color=colour, lw=lw, ls=ls, zorder=z)
    if band:
        ax.fill_between(part.time_days / 365, part.lower, part.upper, step="post", color=colour,
                        alpha=0.12, lw=0, zorder=z - 1)


def fig_within_category(curves: pd.DataFrame, summary: pd.DataFrame, score_id: str, label: str,
                        endpoint: str, stem: Path):
    """One panel per index category: the category's KM curve, split at OPERA's median risk."""
    c = curves[curves.score_id.eq(score_id)]
    sm = summary[summary.score_id.eq(score_id)]
    cats = list(dict.fromkeys(sm.loc[sm.display.eq("index"), "category"]))
    fig, axes = plt.subplots(1, len(cats), figsize=(1.75 * len(cats) + 0.5, 2.45), sharey=True)
    axes = np.atleast_1d(axes)
    fig.subplots_adjust(left=0.09, right=0.99, top=0.8, bottom=0.3, wspace=0.08)
    horizon = int(sm.horizon_days.iloc[0])
    for ax, cat in zip(axes, cats):
        idx = c[c.display.eq("index") & c.category.eq(cat)]
        _km_step(ax, idx, "#9AA3AF", lw=1.1, ls=(0, (3, 2)), band=False, z=2)
        for grp, col in (("OPERA lower half", LOW_HALF), ("OPERA upper half", HIGH_HALF)):
            _km_step(ax, c[c.display.eq("within") & c.category.eq(cat) & c.group.eq(grp)], col, z=3)
        ax.axvline(horizon / 365, color=ZERO, lw=0.6, ls=(0, (2, 2)), zorder=1)
        rows = sm[sm.category.eq(cat)].set_index("group")
        txt = []
        for grp, col in (("OPERA lower half", LOW_HALF), ("OPERA upper half", HIGH_HALF)):
            if grp in rows.index:
                r = rows.loc[grp]
                txt.append((f"{100 * r.survival:.0f}%  (n={int(r.n)})", col))
        for k, (t, col) in enumerate(txt):
            ax.text(0.04, 0.06 + 0.11 * (len(txt) - 1 - k), t, transform=ax.transAxes, fontsize=6.8, color=col)
        ax.set_title(cat.replace("-", "-\n", 1) if len(cat) > 14 else cat, fontsize=8.2, loc="left", pad=4)
        ax.set_ylim(0, 1.02)
        ax.set_xlim(0, horizon * 1.25 / 365)
        tidy(ax)
        ax.set_xlabel("Years", fontsize=7.6)
    axes[0].set_ylabel(endpoint, fontsize=LABEL_SIZE)
    fig.suptitle(f"{label}: each category split by OPERA", x=0.09, ha="left", fontsize=9.4, fontweight="bold")
    fig.legend(handles=[Line2D([], [], color=LOW_HALF, lw=1.6, label="OPERA lower half"),
                        Line2D([], [], color=HIGH_HALF, lw=1.6, label="OPERA upper half"),
                        Line2D([], [], color="#9AA3AF", lw=1.1, ls=(0, (3, 2)), label="Whole category")],
               loc="lower center", ncol=3, frameon=False, fontsize=7, bbox_to_anchor=(0.54, 0.0))
    _save(fig, stem)


def fig_tails(curves: pd.DataFrame, summary: pd.DataFrame, score_id: str, label: str, endpoint: str,
              stem: Path):
    """OPERA's own risk groups (lowest 25% ... top 10%) next to the index's categories."""
    c = curves[curves.score_id.eq(score_id)]
    sm = summary[summary.score_id.eq(score_id)]
    horizon = int(sm.horizon_days.iloc[0])
    fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.6), sharey=True)
    fig.subplots_adjust(left=0.09, right=0.99, top=0.84, bottom=0.18, wspace=0.08)
    ramp = ["#3E8A84", "#B18B4D", "#C46A4F", "#7A2331"]
    cats = list(dict.fromkeys(sm.loc[sm.display.eq("index"), "category"]))
    for ax, display, groups, name in ((axes[0], "index", cats, f"{label} categories"),
                                     (axes[1], "tails", [g for g, *_ in __import__('opera.evaluation.clinical_km_splits', fromlist=['TAIL_GROUPS']).TAIL_GROUPS], "OPERA risk groups")):
        cols = ramp[-len(groups):] if display == "index" and len(groups) < 4 else ramp
        for grp, col in zip(groups, cols):
            key = "category" if display == "index" else "group"
            part = c[c.display.eq(display) & c[key].eq(grp)]
            if part.empty:
                continue
            r = sm[sm.display.eq(display) & sm[key].eq(grp)].iloc[0]
            _km_step(ax, part, col)
            ax.plot([], [], color=col, lw=1.6, label=f"{grp}: {100 * r.survival:.0f}% (n={int(r.n):,})")
        ax.legend(loc="lower left", frameon=False, fontsize=6.6, handlelength=1.2, labelspacing=0.3)
        ax.axvline(horizon / 365, color=ZERO, lw=0.6, ls=(0, (2, 2)), zorder=1)
        ax.set_ylim(0, 1.02)
        ax.set_xlim(0, horizon * 1.25 / 365)
        ax.set_xlabel("Years since first-line treatment", fontsize=7.6)
        tidy(ax)
        ax.set_title(name, loc="left", fontsize=8.6, fontweight="bold", pad=4)
    axes[0].set_ylabel(endpoint, fontsize=LABEL_SIZE)
    _save(fig, stem)


# -- Supplementary: tabular comparators, calibration and the atlas (adapted models) ----------
TAB_SERIES = [  # (contrast, label, colour)
    ("EXT_R_EA_VS_X", "Adapted OPERA vs XGBoost per task", OPERA),
    ("T7", "Pooled vs per-task XGBoost", XP_COL),
    ("EXT_R_EA_VS_XP", "Adapted OPERA vs pooled XGBoost", SIENNA),
]


def _group_order(b: Bundle):
    rows = b.by("EXT_R_EA_VS_XP", "endpoint_group").sort_values("gain")
    return rows.index.tolist()


def supp_tabular(b: Bundle, out: Path):
    fig = plt.figure(figsize=(6.4, 6.4))
    g = fig.add_gridspec(2, 1, height_ratios=[1.5, 1], hspace=0.62, left=0.2, right=0.975, top=0.93, bottom=0.08)
    ax = fig.add_subplot(g[0])
    order = _group_order(b)
    for k, (c, label, col) in enumerate(TAB_SERIES):
        rows = b.by(c, "endpoint_group").reindex(order)
        for i, r in enumerate(rows.itertuples()):
            estimate(ax, r.gain, i + (k - 1) * 0.24, r.lower, r.upper, col, horizontal=True, size=18)
    ax.axvline(0, color=ZERO, lw=0.8, zorder=1)
    ax.set_yticks(range(len(order)), [pf.SHORT_GROUP.get(gname, gname) for gname in order])
    ax.tick_params(axis="y", labelsize=7.8)
    ax.set_ylim(len(order) - 0.5, -0.6)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: "0" if abs(v) < 1e-12 else minus(f"{v:+.2f}")))
    ax.set_xlabel("AUC difference (95% CI)", fontsize=LABEL_SIZE)
    tidy(ax, grid="x")
    ax.spines["left"].set_visible(False)
    ax.legend(handles=[Line2D([], [], marker="D", ls="", color=col, markersize=5, label=lab) for _, lab, col in TAB_SERIES],
              loc="upper center", bbox_to_anchor=(0.42, -0.13), ncol=3, frameon=False, fontsize=7.0,
              handletextpad=0.2, columnspacing=1.0)
    title(ax, "A", "By outcome group", "Held-out, pooled over horizons")

    bx = fig.add_subplot(g[1])
    hz = [30, 60, 90, 180, 365, 730]
    allh = b.e  # horizon 0 only in the bundle view; reload horizon rows
    e = pd.read_csv(b.tables / "effect_summaries.csv.gz", low_memory=False)
    e = e[e.split.eq("held_out") & e.seed.eq(42) & e.metric.eq("auc") & e.stratum.eq("all_cells")]
    for k, (c, label, col) in enumerate(TAB_SERIES):
        rows = e[e.contrast == c].set_index("horizon_days").reindex(hz)
        x = np.arange(len(hz)) + (k - 1) * 0.12
        bx.plot(x, rows.gain, color=lighten(col, 0.55), lw=1.1, zorder=2)
        for xi, r in zip(x, rows.itertuples()):
            estimate(bx, xi, r.gain, r.lower, r.upper, col, size=22)
    bx.axhline(0, color=ZERO, lw=0.8, zorder=1)
    bx.set_xticks(range(len(hz)), [str(h) for h in hz])
    bx.set_xlabel("Horizon (days)", fontsize=LABEL_SIZE)
    bx.set_ylabel("AUC difference", fontsize=LABEL_SIZE)
    bx.yaxis.set_major_formatter(FuncFormatter(lambda v, _: "0" if abs(v) < 1e-12 else minus(f"{v:+.2f}")))
    tidy(bx)
    title(bx, "B", "By horizon", "Same contrasts at each horizon")
    _save(fig, out / "supp_tabular")


CAL_MODELS = [("R_EA", "Adapted OPERA", OPERA), ("J_EA", "Adapted joint", J_COL),
              ("XP", "Pooled XGBoost", XP_COL), ("X", "XGBoost per task", D_COL)]


def _cell_groups(tables: Path):
    c = pd.read_csv(tables / "contrasts.csv.gz", usecols=["cohort", "outcome", "endpoint_group"], low_memory=False)
    return c.drop_duplicates(["cohort", "outcome"]).set_index(["cohort", "outcome"]).endpoint_group


def _reliability(tables: Path, models, horizon=365, nbins=10):
    """Pooled reliability over all tasks: the saved per-task bins regrouped by predicted risk."""
    c = pd.read_csv(tables / "calibration.csv.gz", low_memory=False)
    c = c[c.split.eq("held_out") & c.seed.eq(42) & c.horizon_days.eq(horizon) & c.model.isin(models)
          & c.n.ge(20) & c.w.gt(0)].copy()
    out = []
    for m, part in c.groupby("model"):
        part = part.sort_values("predicted_weighted_mean")
        cum = part.w.cumsum() / part.w.sum()
        part["bin"] = np.minimum((cum * nbins).astype(int), nbins - 1)
        g = part.groupby("bin").apply(lambda f: pd.Series({
            "predicted": np.average(f.predicted_weighted_mean, weights=f.w),
            "observed": np.average(f.observed_fraction, weights=f.w), "n": f.n.sum()}))
        out.append(g.assign(model=m))
    return pd.concat(out)


def supp_calibration(b: Bundle, out: Path, horizon=365):
    cal = pd.read_csv(b.tables / "calibration_summary.csv.gz", low_memory=False)
    cal = cal[cal.split.eq("held_out") & cal.horizon_days.eq(horizon) & cal.status.eq("estimated")
              & cal.model.isin([m for m, *_ in CAL_MODELS])]
    groups = _cell_groups(b.tables)
    cal = cal.join(groups, on=["cohort", "outcome"])
    order = ["__all__"] + _group_order(b)
    labels = ["All tasks"] + [pf.SHORT_GROUP.get(gname, gname) for gname in order[1:]]
    fig = plt.figure(figsize=(6.4, 7.6))
    g = fig.add_gridspec(2, 2, height_ratios=[1.75, 1], wspace=0.30, hspace=0.30, left=0.17, right=0.975,
                         top=0.925, bottom=0.075)
    panels = [("calibration_slope", "A", "Calibration slope", (0.3, 1.75), False),
              ("observed_expected_ratio", "B", "Observed / expected events", (0.42, 2.4), True)]
    first = None
    for j_, (col_name, letter, name, lim, log) in enumerate(panels):
        ax = fig.add_subplot(g[0, j_], sharey=first)
        first = first or ax
        for i in range(len(order)):
            if i % 2 == 0:
                ax.axhspan(i - 0.5, i + 0.5, color="#F4F6F8", lw=0, zorder=0)
        for k, (m, label, col) in enumerate(CAL_MODELS):
            part = cal[cal.model == m]
            for i, grp in enumerate(order):
                v = (part if grp == "__all__" else part.loc[part.endpoint_group == grp])[col_name].dropna()
                if len(v) < 5:
                    continue
                q1, med, q3 = np.percentile(v, [25, 50, 75])
                y = i + (k - 1.5) * 0.2
                lo_, hi_ = max(q1, lim[0]), min(q3, lim[1])
                ax.hlines(y, lo_, hi_, color=lighten(col, 0.45), lw=1.7, zorder=2, capstyle="round")
                for edge, clipped, sign in ((lo_, q1 < lim[0], -1), (hi_, q3 > lim[1], 1)):
                    if clipped:  # interval runs off the axis
                        ax.scatter([edge], [y], marker="<" if sign < 0 else ">", s=9, color=lighten(col, 0.3),
                                   lw=0, zorder=2)
                big = grp == "__all__"
                ax.scatter([med], [y], marker="D", s=22 if big else 17, color=col, edgecolor="white",
                           linewidth=0.7, zorder=3)
        ax.axvline(1.0, color=ZERO, lw=0.8, zorder=1)
        ax.axhline(0.5, color="#CBD2DA", lw=0.6, zorder=1)
        if log:
            ax.set_xscale("log")
            ax.set_xticks([0.5, 0.7, 1, 1.4, 2], ["0.5", "0.7", "1", "1.4", "2"])
            ax.minorticks_off()
        ax.set_xlim(*lim)
        tidy(ax, grid="x")
        ax.spines["left"].set_visible(False)
        title(ax, letter, name)
        if j_:
            ax.tick_params(axis="y", labelleft=False)
    first.set_yticks(range(len(order)), labels)
    first.get_yticklabels()[0].set_fontweight("bold")
    first.tick_params(axis="y", labelsize=7.8)
    first.set_ylim(len(order) - 0.5, -0.5)

    rel = _reliability(b.tables, [m for m, *_ in CAL_MODELS], horizon)
    top = float(max(rel.predicted.max(), rel.observed.max())) * 1.3
    low = float(min(rel.predicted.min(), rel.observed.min())) / 1.3
    ticks = [t for t in (0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5) if low <= t <= top]

    def log_axis(axis, ticks_):
        axis.set_major_locator(matplotlib.ticker.FixedLocator(ticks_))
        axis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
        axis.set_minor_locator(matplotlib.ticker.NullLocator())

    cx = fig.add_subplot(g[1, 0])
    cx.plot([low, top], [low, top], color=ZERO, lw=0.8, ls=(0, (3, 2)), zorder=1)
    dx = fig.add_subplot(g[1, 1], sharex=cx)
    dx.axhline(1.0, color=ZERO, lw=0.8, ls=(0, (3, 2)), zorder=1)
    for m, label, col in CAL_MODELS:
        part = rel[rel.model == m]
        for ax, y in ((cx, part.observed), (dx, part.observed / part.predicted)):
            ax.plot(part.predicted, y, color=lighten(col, 0.55), lw=1.1, zorder=2)
            ax.scatter(part.predicted, y, color=col, s=15, edgecolor="white", linewidth=0.6, zorder=3)
    for ax in (cx, dx):
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(low, top)
        log_axis(ax.xaxis, ticks)
        ax.set_xlabel("Predicted one-year risk", fontsize=LABEL_SIZE)
        tidy(ax)
    cx.set_ylim(low, top)
    log_axis(cx.yaxis, ticks)
    cx.set_ylabel("Observed risk (IPCW)", fontsize=LABEL_SIZE)
    ratio = (rel.observed / rel.predicted)
    span = max(abs(np.log(ratio.min())), abs(np.log(ratio.max()))) * 1.15
    dx.set_ylim(np.exp(-span), np.exp(span))
    log_axis(dx.yaxis, [t for t in (0.5, 0.67, 0.8, 1, 1.25, 1.5, 2) if np.exp(-span) <= t <= np.exp(span)])
    dx.set_ylabel("Observed / predicted", fontsize=LABEL_SIZE)
    title(cx, "C", "Reliability")
    title(dx, "D", "Departure from ideal")
    fig.legend(handles=[Line2D([], [], marker="D", ls="", color=col, markersize=5, label=lab) for _, lab, col in CAL_MODELS],
               loc="upper center", ncol=4, frameon=False, fontsize=7.6, bbox_to_anchor=(0.57, 0.995),
               handletextpad=0.3, columnspacing=1.4)
    _save(fig, out / "supp_calibration")


def supp_atlas_matrix(b: Bundle, out: Path, model="R_EA"):
    est = pd.read_csv(b.tables / "estimates.csv.gz", low_memory=False)
    est = est[est.metric.eq("auc") & est.seed.eq(42) & est.split.eq("held_out") & est.model.eq(model)]
    groups = _cell_groups(b.tables)
    est = est.join(groups, on=["cohort", "outcome"])
    tab = est.groupby(["cohort", "endpoint_group"]).estimate.median().unstack()
    size = est.groupby("cohort").n.max().sort_values()
    tab = tab.reindex(size.index)
    cols = tab.median().sort_values(ascending=False).index
    tab = tab[cols]
    fig, ax = plt.subplots(figsize=(6.4, 6.0))
    fig.subplots_adjust(left=0.24, right=0.86, top=0.9, bottom=0.2)
    im = ax.imshow(tab.to_numpy(float), cmap=pf.ATLAS_CMAP, vmin=0.5, vmax=0.95, aspect="auto")
    ax.set_xticks(range(len(cols)), [pf.SHORT_GROUP.get(c, c) for c in cols], rotation=45, ha="right", fontsize=7.4)
    names = {"AMYLOIDOSIS": "Amyloidosis", "SolM": "Solitary plasmacytoma", "TRANSFORMED_FL": "Transformed FL"}
    ax.set_yticks(range(len(tab)), [names.get(d, d) for d in tab.index], fontsize=7.2)
    ax.tick_params(length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_xticks(np.arange(len(cols) + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(len(tab) + 1) - 0.5, minor=True)
    ax.grid(which="minor", color="white", lw=1.0)
    ax.grid(which="major", visible=False)
    cax = fig.add_axes([0.88, 0.2, 0.02, 0.7])
    cb = fig.colorbar(im, cax=cax)
    cb.outline.set_visible(False)
    cb.set_label("Median AUC over the group's tasks", fontsize=LABEL_SIZE)
    title(ax, "", "Adapted OPERA: held-out AUC by disease and outcome group",
          "Median over outcomes and horizons in each tile; diseases by held-out size, smallest at top")
    ax.set_title(ax.get_title(loc="left").strip(), loc="left", fontsize=9.6, fontweight="bold", pad=17)
    _save(fig, out / "supp_atlas_matrix")


GROUP_ORDER = ["Low", "Low-intermediate", "Intermediate", "High-intermediate", "High", "Very high"]
GROUP_SHORT = {"Low": "Low", "Low-intermediate": "Low-int.", "Intermediate": "Int.", "High-intermediate": "High-int.",
               "High": "High", "Very high": "Very high"}


def supp_clinical_groups(figures: Path, clinical: Path, out: Path):
    """Survival in each risk group at equal group sizes: index categories against OPERA's groups."""
    sm = pd.read_csv(figures / "size_matched_survival_comparisons.csv")
    m = pd.read_csv(clinical / "matched_risk_and_auc.csv").set_index("score_id")
    fig, axes = plt.subplots(4, 3, figsize=(6.4, 8.2), sharey=True)
    fig.subplots_adjust(left=0.1, right=0.98, top=0.9, bottom=0.07, hspace=0.75, wspace=0.12)
    for ax, (sid, name, end) in zip(axes.flat, CLINICAL_ROWS):
        part = sm[sm.score_id == sid].copy()
        part["k"] = part.group.map(GROUP_ORDER.index)
        part = part.sort_values("k")
        x = np.arange(len(part))
        ax.plot(x, part.index_estimate, color=lighten(SIENNA, 0.5), lw=1.1, zorder=1)
        ax.plot(x, part.model_estimate, color=lighten(OPERA, 0.5), lw=1.1, zorder=1)
        ax.scatter(x, part.index_estimate, marker="o", s=22, color=SIENNA, edgecolor="white", linewidth=0.8, zorder=3)
        ax.scatter(x, part.model_estimate, marker="D", s=24, color=OPERA, edgecolor="white", linewidth=0.8, zorder=3)
        for xi, r in zip(x, part.itertuples()):
            sig = r.lower > 0 or r.upper < 0
            ax.vlines(xi, min(r.model_estimate, r.index_estimate), max(r.model_estimate, r.index_estimate),
                      color=OPERA if sig else "#C5CAD3", lw=2.4 if sig else 1.2, zorder=2, alpha=0.35)
        ax.set_xticks(x, [GROUP_SHORT[gname] for gname in part.group], fontsize=6.8)
        ax.set_xlim(-0.4, len(x) - 0.6)
        ax.set_ylim(0.3, 1.04)
        ax.yaxis.set_major_locator(MultipleLocator(0.2))
        tidy(ax)
        r = m.loc[sid]
        horizon = "2 y" if "2y" in sid else "1 y"
        ax.set_title(name if end in ("OS", "TF") else f"{name}, {end}", loc="left", fontsize=8.4,
                     fontweight="bold", pad=12)
        ax.annotate(f"{end.split(',')[0]} at {horizon}; n={int(r.n_patients):,}", xy=(0, 1), xycoords="axes fraction",
                    xytext=(0, 2), textcoords="offset points", fontsize=6.6, color=MUTED, style="italic", va="bottom")
    for ax in axes[:, 0]:
        ax.set_ylabel("Survival", fontsize=8)
    leg = axes.flat[-1]
    leg.axis("off")
    leg.legend(handles=[Line2D([], [], marker="o", ls="-", color=lighten(SIENNA, 0.5), markerfacecolor=SIENNA,
                               markeredgecolor="white", markersize=6, label="Index category"),
                        Line2D([], [], marker="D", ls="-", color=lighten(OPERA, 0.5), markerfacecolor=OPERA,
                               markeredgecolor="white", markersize=6, label="Adapted OPERA group,\nsame size"),
                        Line2D([], [], color=OPERA, lw=2.4, alpha=0.35, label="Difference: interval\nexcludes zero")],
               loc="center", frameon=False, fontsize=7.4)
    fig.suptitle("Survival by risk group at equal group sizes", x=0.1, ha="left", fontsize=9.6, fontweight="bold")
    _save(fig, out / "supp_clinical_groups")


def supp_label_efficiency(fractions: Path, out: Path):
    """Adapted joint model with 10-100% of training patients, against the cohort-encoder model
    with all patients, at every horizon (one panel per horizon)."""
    d = pd.read_csv(fractions / "plotted_panel_estimates.csv")
    d = d[d.metric.eq("auc")]
    shares = [("J_EA_F10", 10), ("J_EA_F25", 25), ("J_EA_F50", 50), ("J_EA", 100)]
    horizons = sorted(d.horizon_days.unique())
    fig, axes = plt.subplots(2, 3, figsize=(6.4, 4.6), sharex=True)
    fig.subplots_adjust(left=0.1, right=0.98, top=0.86, bottom=0.12, hspace=0.55, wspace=0.32)
    for ax, h in zip(axes.flat, horizons):
        part = d[d.horizon_days == h].set_index("model")
        x = np.log10([sh for _, sh in shares])
        y = [part.loc[m, "estimate"] for m, _ in shares]
        lo = [part.loc[m, "lower"] for m, _ in shares]
        hi = [part.loc[m, "upper"] for m, _ in shares]
        ax.fill_between(x, lo, hi, color=lighten(ADAPT, 0.18), lw=0, zorder=1)
        ax.plot(x, y, color=ADAPT, lw=1.4, zorder=2)
        ax.scatter(x, y, color=ADAPT, s=18, edgecolor="white", linewidth=0.8, zorder=3)
        ref = part.loc["J_L"]
        ax.axhspan(ref.lower, ref.upper, color=lighten(S_COL, 0.18), lw=0, zorder=0)
        ax.axhline(ref.estimate, color=MUTED, lw=1.0, ls=(0, (3, 2)), zorder=1)
        ax.set_xticks(x, ["10%", "25%", "50%", "100%"])
        ax.yaxis.set_major_locator(MaxNLocator(3))
        tidy(ax)
        label = f"{h} days" if h < 365 else ("1 year" if h == 365 else "2 years")
        ax.set_title(label, loc="left", fontsize=8.6, fontweight="bold", pad=10)
        ax.annotate(f"{int(part.n_cells.iloc[0]):,} tasks", xy=(0, 1), xycoords="axes fraction", xytext=(0, 1),
                    textcoords="offset points", fontsize=6.6, color=MUTED, style="italic", va="bottom")
    for ax in axes[:, 0]:
        ax.set_ylabel("Mean AUC", fontsize=8)
    for ax in axes[-1]:
        ax.set_xlabel("Share of training patients", fontsize=8)
    fig.legend(handles=[Line2D([], [], color=ADAPT, marker="o", markersize=4, lw=1.4,
                               label="Adapted encoder, joint model (95% CI)"),
                        Line2D([], [], color=MUTED, ls=(0, (3, 2)), lw=1.0,
                               label="Cohort encoder, all training patients (95% CI)")],
               loc="upper center", ncol=2, frameon=False, fontsize=7.2, bbox_to_anchor=(0.54, 0.985))
    _save(fig, out / "supp_label_efficiency")


def _save(fig, stem: Path):
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(stem.with_suffix(".pdf"))
    fig.savefig(stem.with_suffix(".png"), dpi=200)
    plt.close(fig)
    print(f"wrote {stem}.pdf")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--bundle", type=Path, required=True)
    p.add_argument("--cells", type=Path, required=True)
    p.add_argument("--clinical", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--only", nargs="*")
    a = p.parse_args()
    setup_style()
    thesis_style()
    b = Bundle(a.bundle)
    jobs = {"fig2": lambda: fig2_overview(b, a.out), "fig3": lambda: fig3_pooling(b, a.cells, a.out),
            "fig4": lambda: fig4_population(b, a.cells, a.out), "fig5": lambda: fig5_specialise(b, a.out),
            "fig6": lambda: fig6_clinical(a.clinical, a.out),
            "fig6_bars": lambda: fig6_clinical(a.clinical, a.out, variant="bars"),
            "supp_tabular": lambda: supp_tabular(b, a.out), "supp_calibration": lambda: supp_calibration(b, a.out),
            "supp_atlas_matrix": lambda: supp_atlas_matrix(b, a.out),
            "supp_clinical_groups": lambda: supp_clinical_groups(Path("scratch/R_EA_clinical_figures"), a.clinical, a.out),
            "supp_label_efficiency": lambda: supp_label_efficiency(Path("scratch/fraction_figures_20261116_ready"), a.out),
            "supp_population_disease": lambda: supp_population_disease(b, a.cells, a.out)}
    for k, job in jobs.items():
        if not a.only or k in a.only:
            job()


if __name__ == "__main__":
    main()
