"""
OPERA Model Comparison Visualizations.

Key figures:
  1. Forest plot — AUROC/C-index with bootstrap CIs (the main results figure)
  2. Label efficiency curves — AUROC vs fraction of training labels
  3. Pairwise significance dots
  4. Multi-metric dot plot
"""

from __future__ import annotations
from typing import Dict, List, Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from opera.visualization.style import (
    PALETTE,
    save_fig,
    despine,
    ci_ribbon,
    FIG_FULL,
    FIG_TALL,
    model_color,
    model_label,
    ANNOT_SIZE,
    LEGEND_SIZE,
    setup_style,
)


# ═════════════════════════════════════════════════════════════════════
# 1. Forest plot  (the main paper figure)
# ═════════════════════════════════════════════════════════════════════


def plot_forest(
    model_results: Dict[str, Dict[str, Dict[str, float]]],
    metric: str = "auroc",
    title: Optional[str] = None,
    group_by_outcome: bool = False,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Publication-style forest plot with numeric annotation panel on the right.

    Parameters
    ----------
    model_results : dict
        name → {metric: {"mean", "lower", "upper"}}
        Keys should be model names; use the canonical names in style.py for
        automatic color + label assignment.
    metric : str
        Which metric to plot. Displayed in the x-axis label.
    group_by_outcome : bool
        If True, insert thin separators between groups when entries have a
        "group" field (expects model_results to be an OrderedDict with
        sentinel None values marking group boundaries).
    """
    names = list(model_results.keys())
    n = len(names)
    means = [model_results[nm][metric]["mean"] for nm in names]
    lowers = [model_results[nm][metric]["lower"] for nm in names]
    uppers = [model_results[nm][metric]["upper"] for nm in names]

    y_pos = np.arange(n)

    # Figure: main panel (70%) + numeric annotation panel (30%)
    fig, (ax_main, ax_num) = plt.subplots(
        1,
        2,
        figsize=(7.0, max(2.8, 0.42 * n)),
        gridspec_kw={"width_ratios": [3, 1]},
    )

    for i, nm in enumerate(names):
        color = model_color(nm)
        # CI line
        ax_main.plot(
            [lowers[i], uppers[i]],
            [i, i],
            color=color,
            lw=1.8,
            solid_capstyle="round",
            zorder=2,
        )
        # Point estimate — filled square (meta-analysis style)
        ax_main.scatter(
            means[i], i, color=color, s=50, marker="s", zorder=3, linewidths=0
        )

    # Reference line at 0.5 (random) or best model — depends on metric
    ref = 0.5 if metric in ("auroc", "auprc") else 0.0
    ax_main.axvline(ref, color=PALETTE["diagonal"], ls="--", lw=0.8, zorder=1)

    ax_main.set_yticks(y_pos)
    ax_main.set_yticklabels([model_label(nm) for nm in names])
    ax_main.invert_yaxis()
    ax_main.set_xlabel(metric.upper() + "  (95% CI)")
    ax_main.set_title(title or f"Forest plot — {metric.upper()}")
    despine(ax_main, "none")
    ax_main.grid(axis="x", color=PALETTE["grid"], linewidth=0.5)
    ax_main.grid(axis="y", visible=False)

    # Numeric annotation panel
    ax_num.set_ylim(ax_main.get_ylim())
    ax_num.set_xlim(0, 1)
    ax_num.axis("off")
    ax_num.text(
        0.5,
        1.01,
        f"{metric.upper()}  [95% CI]",
        transform=ax_num.transAxes,
        ha="center",
        va="bottom",
        fontsize=8,
        fontweight="semibold",
    )
    for i, (m, l, u) in enumerate(zip(means, lowers, uppers)):  # noqa: E741
        ax_num.text(
            0.5,
            i,
            f"{m:.3f}  [{l:.3f}–{u:.3f}]",
            ha="center",
            va="center",
            fontsize=ANNOT_SIZE,
            color=model_color(names[i]),
        )

    fig.tight_layout(w_pad=0)
    save_fig(fig, save_path)
    return fig


# ═════════════════════════════════════════════════════════════════════
# 2. Multi-metric dot plot  (alternative to grouped bar)
# ═════════════════════════════════════════════════════════════════════


def plot_metric_comparison(
    model_metrics: Dict[str, Dict[str, float]],
    metrics_to_plot: List[str] = ("auroc", "auprc", "f1", "sensitivity", "specificity"),
    title: str = "Model comparison",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Cleveland dot plot: each metric gets its own column, models are dots.
    Cleaner than grouped bars for comparing many models × many metrics.
    """
    names = list(model_metrics.keys())
    mnames = list(metrics_to_plot)
    n_m = len(names)
    n_met = len(mnames)

    fig, axes = plt.subplots(
        1, n_met, figsize=(2.0 * n_met, max(3.0, 0.45 * n_m)), sharey=True
    )
    if n_met == 1:
        axes = [axes]

    for j, met in enumerate(mnames):
        ax = axes[j]
        for i, nm in enumerate(names):
            val = model_metrics[nm].get(met, float("nan"))
            color = model_color(nm)
            ax.scatter(val, i, color=color, s=55, zorder=3, linewidths=0)
            ax.hlines(i, 0, val, color=color, lw=1.0, alpha=0.4, zorder=2)

        ax.set_xlabel(met.upper(), fontsize=8)
        ax.set_xlim(0, 1.02)
        ax.set_title(met.upper(), fontsize=8, fontweight="semibold")
        if j == 0:
            ax.set_yticks(range(n_m))
            ax.set_yticklabels([model_label(nm) for nm in names], fontsize=8)
        ax.invert_yaxis()
        despine(ax, "none")
        ax.grid(axis="x", color=PALETTE["grid"], linewidth=0.5)
        ax.grid(axis="y", visible=False)

    # Legend
    handles = [
        mpatches.Patch(color=model_color(nm), label=model_label(nm)) for nm in names
    ]
    axes[-1].legend(
        handles=handles,
        fontsize=LEGEND_SIZE,
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
        borderaxespad=0,
    )

    fig.suptitle(title, fontsize=10, fontweight="semibold")
    fig.tight_layout()
    save_fig(fig, save_path)
    return fig


# ═════════════════════════════════════════════════════════════════════
# 3. Pairwise significance plot
# ═════════════════════════════════════════════════════════════════════


def plot_pairwise_differences(
    difference_results: Dict[str, Dict[str, float]],
    title: str = "Pairwise AUROC differences  (Model vs reference)",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Forest-style plot of bootstrap AUROC differences with significance markers.

    difference_results: "A vs B" → {"diff_mean", "diff_lower", "diff_upper", "p_value"}
    """
    pairs = list(difference_results.keys())
    n = len(pairs)
    if n == 0:
        return plt.figure()

    fig, ax = plt.subplots(figsize=(5.5, max(2.5, 0.42 * n)))
    y = np.arange(n)

    for i, pair in enumerate(pairs):
        r = difference_results[pair]
        dm, dl, du, pv = r["diff_mean"], r["diff_lower"], r["diff_upper"], r["p_value"]

        if dl > 0:
            color = PALETTE["tabular_rkkp"]  # significantly positive
        elif du < 0:
            color = PALETTE["positive"]  # significantly negative
        else:
            color = PALETTE["missing"]  # not significant

        ax.plot([dl, du], [i, i], color=color, lw=2.0, solid_capstyle="round", zorder=2)
        ax.scatter(dm, i, color=color, s=45, marker="D", zorder=3, linewidths=0)

        stars = (
            "***" if pv < 0.001 else ("**" if pv < 0.01 else ("*" if pv < 0.05 else ""))
        )
        ax.text(
            max(du, 0) + 0.005,
            i,
            f"Δ={dm:+.3f}  {stars}",
            va="center",
            fontsize=ANNOT_SIZE,
        )

    ax.axvline(0, color=PALETTE["zero_line"], ls="--", lw=0.9, zorder=1)
    ax.set_yticks(y)
    ax.set_yticklabels(pairs, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("AUROC difference  (95% CI)")
    ax.set_title(title)
    despine(ax, "none")
    ax.grid(axis="x", color=PALETTE["grid"], linewidth=0.5)
    ax.grid(axis="y", visible=False)

    fig.tight_layout()
    save_fig(fig, save_path)
    return fig


# ═════════════════════════════════════════════════════════════════════
# 4. Label efficiency curves
# ═════════════════════════════════════════════════════════════════════


def plot_label_efficiency(
    efficiency_results: Dict[str, Dict[float, Dict[str, float]]],
    metric: str = "auroc",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Learning curves: AUROC vs fraction of labelled training data, with CI ribbons.

    efficiency_results : model_name → {fraction: {"mean", "lower", "upper"}}
    """
    fig, ax = plt.subplots(figsize=(4.8, 3.5))

    for name, fractions in efficiency_results.items():
        xs = sorted(fractions.keys())
        means = np.array([fractions[x]["mean"] for x in xs])
        lowers = np.array([fractions[x]["lower"] for x in xs])
        uppers = np.array([fractions[x]["upper"] for x in xs])
        pcts = np.array(xs) * 100
        color = model_color(name)

        ax.plot(
            pcts,
            means,
            color=color,
            lw=1.8,
            marker="o",
            markersize=4,
            label=model_label(name),
            zorder=3,
        )
        ci_ribbon(ax, pcts, lowers, uppers, color)

    ax.set_xlabel("Training labels used  (%)")
    ax.set_ylabel(f"{metric.upper()}  (95% CI)")
    ax.set_title("Label efficiency")
    ax.set_xlim(0, 105)
    ax.set_xticks([5, 10, 20, 40, 60, 80, 100])
    ax.legend(fontsize=LEGEND_SIZE, framealpha=0.92)
    despine(ax, "y")

    fig.tight_layout()
    save_fig(fig, save_path)
    return fig


# ═════════════════════════════════════════════════════════════════════
# 5. Delta plot  (joint vs per-cohort)
# ═════════════════════════════════════════════════════════════════════


def plot_joint_vs_percohort_delta(
    results_df: pd.DataFrame,
    per_cohort_col: str = "opera__auroc",
    joint_col: str = "opera_joint__auroc",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Lollipop chart of ΔAUROC (joint − per-cohort) per (cohort, outcome) cell.
    """
    df = results_df.dropna(subset=[per_cohort_col, joint_col]).copy()
    df["label"] = (
        df["cohort"].str.upper() + "  /  " + df["outcome"].str.replace("_", " ")
    )
    df["delta"] = df[joint_col] - df[per_cohort_col]
    df = df.sort_values("delta").reset_index(drop=True)

    n = len(df)
    fig, ax = plt.subplots(figsize=(5.5, max(3.0, 0.38 * n)))
    y = np.arange(n)

    pos_mask = df["delta"] >= 0
    ax.hlines(
        y[pos_mask],
        0,
        df["delta"].values[pos_mask],
        color=PALETTE["opera"],
        lw=1.4,
        zorder=2,
    )
    ax.scatter(
        df["delta"].values[pos_mask],
        y[pos_mask],
        color=PALETTE["opera"],
        s=40,
        zorder=3,
        linewidths=0,
    )

    ax.hlines(
        y[~pos_mask],
        df["delta"].values[~pos_mask],
        0,
        color=PALETTE["positive"],
        lw=1.4,
        zorder=2,
    )
    ax.scatter(
        df["delta"].values[~pos_mask],
        y[~pos_mask],
        color=PALETTE["positive"],
        s=40,
        zorder=3,
        linewidths=0,
    )

    ax.axvline(0, color=PALETTE["zero_line"], ls="--", lw=0.9, zorder=1)
    ax.set_yticks(y)
    ax.set_yticklabels(df["label"].tolist(), fontsize=7.5)
    ax.set_xlabel("ΔAUROC  (joint − per-cohort)")
    ax.set_title("Joint training adds over per-cohort")
    pct = 100 * pos_mask.mean()
    ax.text(
        0.98,
        0.02,
        f"{pct:.0f}% positive",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=ANNOT_SIZE,
        color="grey",
    )
    despine(ax, "none")
    ax.grid(axis="x", color=PALETTE["grid"], linewidth=0.5)
    ax.grid(axis="y", visible=False)

    fig.tight_layout()
    save_fig(fig, save_path)
    return fig


# ═════════════════════════════════════════════════════════════════════
# 6. Bootstrap difference test (utility)
# ═════════════════════════════════════════════════════════════════════


def bootstrap_difference_test(
    labels: np.ndarray,
    probs_a: np.ndarray,
    probs_b: np.ndarray,
    metric_fn,
    n_bootstrap: int = 2000,
    seed: int = 42,
) -> Dict[str, float]:
    rng = np.random.RandomState(seed)
    n = len(labels)
    diffs = []
    for _ in range(n_bootstrap):
        idx = rng.randint(0, n, size=n)
        y = labels[idx]
        if len(np.unique(y)) < 2:
            continue
        diffs.append(metric_fn(y, probs_a[idx]) - metric_fn(y, probs_b[idx]))
    diffs = np.array(diffs)
    p = (diffs <= 0).mean() if diffs.mean() > 0 else (diffs >= 0).mean()
    return {
        "diff_mean": float(diffs.mean()),
        "diff_lower": float(np.percentile(diffs, 2.5)),
        "diff_upper": float(np.percentile(diffs, 97.5)),
        "p_value": float(p),
    }


def plot_outcome_dot_matrix(
    results_df: pd.DataFrame,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot AUROC dot matrix across cohort-outcome cells and model families.

    Input is long result-schema rows; full-cohort rows are used by default so
    aggregate foundation-model comparisons do not mix IPI-complete subsets. The
    output highlights which model is best per task without including IPI.
    """
    setup_style()
    df = results_df.copy()
    if "evaluation_subset" in df.columns:
        df = df[df["evaluation_subset"].fillna("full") == "full"]
    models = ["tabular_rkkp", "tabular_ehr", "dapt", "mol", "opera", "opera_joint"]
    df = df[df["model_family"].isin(models)]
    df["task"] = df["cohort"].astype(str) + " / " + df["outcome"].astype(str)
    tasks = sorted(df["task"].unique())
    fig, ax = plt.subplots(figsize=FIG_TALL if len(tasks) > 14 else FIG_FULL)
    for y, task in enumerate(tasks):
        task_df = df[df["task"] == task]
        best = (
            task_df.sort_values("auroc", ascending=False)["model_family"].iloc[0]
            if not task_df.empty
            else None
        )
        for x, model in enumerate(models):
            row = task_df[task_df["model_family"] == model]
            if row.empty:
                continue
            val = float(row["auroc"].iloc[0])
            ax.scatter(
                x,
                y,
                s=40 + 160 * max(0, val - 0.5),
                color=model_color(model) if model == best else "#CCCCCC",
                edgecolor=model_color(model),
                linewidth=0.8,
            )
    ax.set_xticks(
        range(len(models)), [model_label(m) for m in models], rotation=45, ha="right"
    )
    ax.set_yticks(range(len(tasks)), tasks)
    ax.invert_yaxis()
    ax.set_ylabel("Cohort / outcome")
    ax.set_xlabel("Model")
    fig.tight_layout()
    save_fig(fig, save_path)
    return fig


def plot_ipi_credibility(
    results_df: pd.DataFrame,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot IPI, tabular-EHR, and OPERA on the IPI-complete subset.

    Input is long result-schema rows filtered to `evaluation_subset` equal to
    `ipi_complete` and IPI coverage at least 50%. This standalone credibility
    figure tests ML against the clinical score only on patients where IPI is
    actually computable.
    """
    setup_style()
    df = results_df.copy()
    if "evaluation_subset" in df.columns:
        df = df[df["evaluation_subset"] == "ipi_complete"]
    else:
        df = df.iloc[0:0]
    if "ipi_coverage" in df.columns:
        df = df[df["ipi_coverage"].fillna(0) >= 0.5]
    models = ["ipi", "tabular_ehr", "opera"]
    df = df[df["model_family"].isin(models)]
    tasks = sorted(
        (df["cohort"].astype(str) + " / " + df["outcome"].astype(str)).unique()
    )
    fig, ax = plt.subplots(figsize=FIG_FULL)
    width = 0.22
    for i, model in enumerate(models):
        vals = []
        for task in tasks:
            cohort, outcome = task.split(" / ", 1)
            row = df[
                (df["cohort"].astype(str) == cohort)
                & (df["outcome"].astype(str) == outcome)
                & (df["model_family"] == model)
            ]
            vals.append(float(row["auroc"].iloc[0]) if not row.empty else np.nan)
        x = np.arange(len(tasks)) + (i - 1) * width
        ax.bar(x, vals, width=width, color=model_color(model), label=model_label(model))
    for j, task in enumerate(tasks):
        cohort, outcome = task.split(" / ", 1)
        row = df[
            (df["cohort"].astype(str) == cohort)
            & (df["outcome"].astype(str) == outcome)
        ]
        cov = (
            row["ipi_coverage"].dropna().iloc[0]
            if "ipi_coverage" in row and row["ipi_coverage"].notna().any()
            else np.nan
        )
        if np.isfinite(cov):
            ax.text(j, 1.01, f"{100 * cov:.0f}%", ha="center", va="bottom", fontsize=7)
    ax.set_xticks(range(len(tasks)), tasks, rotation=45, ha="right")
    ax.set_ylabel("AUROC")
    ax.set_ylim(0, 1.08)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    save_fig(fig, save_path)
    return fig
