"""Survival-analysis figures for OPERA evaluation outputs."""

from __future__ import annotations

from typing import Optional

import matplotlib.pyplot as plt
import pandas as pd

from opera.visualization.style import (
    FIG_FULL,
    PALETTE,
    model_color,
    model_label,
    save_fig,
    setup_style,
)


def plot_timedep_auc_curves(
    results_df: pd.DataFrame,
    outcome: str,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Plot IPCW-AUC over time for model variants.

    Input rows contain `outcome`, `model_family`, `horizon_days`, `ipcw_auc`,
    and optional CI columns. The returned figure compares survival ranking
    performance across clinically meaningful horizons for one outcome.
    """
    setup_style()
    df = results_df[results_df["outcome"] == outcome].copy()
    fig, ax = plt.subplots(figsize=FIG_FULL)
    for model, group in df.groupby("model_family"):
        group = group.sort_values("horizon_days")
        color = model_color(str(model))
        ax.plot(
            group["horizon_days"],
            group["ipcw_auc"],
            marker="o",
            color=color,
            label=model_label(str(model)),
        )
        if {"ci_lower", "ci_upper"}.issubset(group.columns):
            ax.fill_between(
                group["horizon_days"],
                group["ci_lower"],
                group["ci_upper"],
                color=color,
                alpha=0.13,
                linewidth=0,
            )
    for x in (30, 90, 365, 730):
        ax.axvline(x, color=PALETTE["diagonal"], linewidth=0.7, linestyle="--")
    ax.set_xlabel("Horizon (days)")
    ax.set_ylabel("IPCW-AUC")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    save_fig(fig, save_path)
    return fig


def plot_calibration_comparison(
    results_df: pd.DataFrame,
    outcome: str,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Plot calibration curves by cohort for one outcome.

    Input rows contain calibration bin columns (`predicted`, `observed`) plus
    cohort, ECE, and HL p-value. The figure checks whether predicted risks are
    clinically interpretable across cohorts.
    """
    setup_style()
    df = results_df[results_df["outcome"] == outcome].copy()
    fig, ax = plt.subplots(figsize=FIG_FULL)
    for cohort, group in df.groupby("cohort"):
        label = str(cohort)
        if "ece" in group.columns and "hl_pvalue" in group.columns:
            label = f"{label} ECE={group['ece'].iloc[0]:.3f}, HL p={group['hl_pvalue'].iloc[0]:.3f}"
        ax.plot(group["predicted"], group["observed"], marker="o", label=label)
    ax.plot([0, 1], [0, 1], color=PALETTE["diagonal"], linestyle="--", linewidth=0.9)
    ax.set_xlabel("Predicted risk")
    ax.set_ylabel("Observed event rate")
    ax.legend(frameon=False, fontsize=7)
    fig.tight_layout()
    save_fig(fig, save_path)
    return fig


def plot_decision_curves(
    dca_df: pd.DataFrame,
    outcome: str,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Plot decision-curve net benefit for one outcome.

    Input rows contain threshold and net-benefit columns. The output compares
    model utility against treat-all and treat-none references across plausible
    clinical risk thresholds.
    """
    setup_style()
    df = (
        dca_df[dca_df["outcome"] == outcome].copy()
        if "outcome" in dca_df.columns
        else dca_df.copy()
    )
    fig, ax = plt.subplots(figsize=FIG_FULL)
    for model, group in df.groupby("model_family"):
        ax.plot(
            group["threshold"],
            group["net_benefit_model"],
            color=model_color(str(model)),
            label=model_label(str(model)),
        )
    if "net_benefit_treat_all" in df.columns:
        ref = df.sort_values("threshold")
        ax.plot(
            ref["threshold"],
            ref["net_benefit_treat_all"],
            color=PALETTE["diagonal"],
            linestyle="--",
            label="Treat all",
        )
    ax.axhline(0, color=PALETTE["zero_line"], linewidth=0.8, label="Treat none")
    ax.set_xlabel("Risk threshold")
    ax.set_ylabel("Net benefit")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    save_fig(fig, save_path)
    return fig
