"""
OPERA Classification Plots.

All figures are designed for journal publication (7" wide, 300 dpi).
Import style constants from opera.visualization.style — do not override
rcParams locally.
"""

from __future__ import annotations
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from sklearn.metrics import roc_curve, precision_recall_curve

from opera.visualization.style import (
    PALETTE, CATEGORICAL, save_fig, ci_ribbon, despine,
    FIG_FULL, FIG_SQUARE, FIG_HALF, FIG_TALL, add_panel_label,
)
from opera.evaluation.metrics import (
    decision_curve_analysis,
    compute_calibration_metrics,
    compute_discrimination_metrics,
    _plotting_horizons,
    _derive_time_horizons,
    compute_timedep_auc_curve,
)


# ═════════════════════════════════════════════════════════════════════
# 1. ROC Curve
# ═════════════════════════════════════════════════════════════════════

def plot_roc_curve(
    labels: np.ndarray,
    probabilities: np.ndarray,
    title: str = "ROC Curve",
    save_path: Optional[str] = None,
    bootstrap_ci: Optional[Dict] = None,
    ax: Optional[plt.Axes] = None,
    color: Optional[str] = None,
    label: Optional[str] = None,
) -> plt.Figure:
    """Single-model ROC curve with optional CI annotation."""
    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=FIG_SQUARE)
    else:
        fig = ax.figure

    color = color or PALETTE["opera"]
    fpr, tpr, _ = roc_curve(labels, probabilities)
    auroc = float(np.trapz(tpr, fpr))

    ci_str = ""
    if bootstrap_ci and "auroc" in bootstrap_ci:
        ci = bootstrap_ci["auroc"]
        ci_str = f" [{ci['lower']:.3f}–{ci['upper']:.3f}]"

    lbl = label or f"AUROC = {auroc:.3f}{ci_str}"

    ax.plot(fpr, tpr, color=color, lw=1.8, label=lbl, zorder=3)
    ax.fill_between(fpr, tpr, alpha=PALETTE["fill_alpha"], color=color, zorder=2)
    ax.plot([0, 1], [0, 1], "--", color=PALETTE["diagonal"], lw=1, zorder=1)

    ax.set_xlabel("False positive rate  (1 − specificity)")
    ax.set_ylabel("True positive rate  (sensitivity)")
    ax.set_xlim(-0.01, 1.01)
    ax.set_ylim(-0.01, 1.01)
    ax.set_title(title)
    ax.legend(loc="lower right")
    despine(ax, grid_axis="none")

    if standalone:
        fig.tight_layout()
        save_fig(fig, save_path)
    return fig


def plot_roc_comparison(
    results: Dict[str, Tuple[np.ndarray, np.ndarray]],
    bootstrap_cis: Optional[Dict[str, Dict]] = None,
    title: str = "ROC Comparison",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Multi-model ROC curves on a single axis."""
    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    bootstrap_cis = bootstrap_cis or {}

    for (name, (labels, probs)), color in zip(results.items(), CATEGORICAL):
        fpr, tpr, _ = roc_curve(labels, probs)
        auroc = float(np.trapz(tpr, fpr))
        ci = bootstrap_cis.get(name, {}).get("auroc", {})
        ci_str = f" [{ci['lower']:.3f}–{ci['upper']:.3f}]" if ci else ""
        display = f"{name.replace('_', ' ').title()}  {auroc:.3f}{ci_str}"
        ax.plot(fpr, tpr, color=color, lw=1.8, label=display, zorder=3)
        ax.fill_between(fpr, tpr, alpha=0.06, color=color)

    ax.plot([0, 1], [0, 1], "--", color=PALETTE["diagonal"], lw=1, zorder=1)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_xlim(-0.01, 1.01)
    ax.set_ylim(-0.01, 1.01)
    ax.set_title(title)
    ax.legend(loc="lower right", fontsize=7.5)
    despine(ax, "none")
    fig.tight_layout()
    save_fig(fig, save_path)
    return fig


# ═════════════════════════════════════════════════════════════════════
# 2. Precision-Recall Curve
# ═════════════════════════════════════════════════════════════════════

def plot_prc(
    labels: np.ndarray,
    probabilities: np.ndarray,
    title: str = "Precision-Recall Curve",
    save_path: Optional[str] = None,
    bootstrap_ci: Optional[Dict] = None,
    ax: Optional[plt.Axes] = None,
    color: Optional[str] = None,
    label: Optional[str] = None,
) -> plt.Figure:
    from sklearn.metrics import average_precision_score
    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=FIG_SQUARE)
    else:
        fig = ax.figure

    color = color or PALETTE["opera"]
    precision, recall, _ = precision_recall_curve(labels, probabilities)
    auprc = average_precision_score(labels, probabilities)
    prevalence = float(labels.mean())

    ci_str = ""
    if bootstrap_ci and "auprc" in bootstrap_ci:
        ci = bootstrap_ci["auprc"]
        ci_str = f" [{ci['lower']:.3f}–{ci['upper']:.3f}]"

    lbl = label or f"AUPRC = {auprc:.3f}{ci_str}"

    ax.plot(recall, precision, color=color, lw=1.8, label=lbl, zorder=3)
    ax.fill_between(recall, precision, prevalence, where=(precision >= prevalence),
                    alpha=PALETTE["fill_alpha"], color=color, zorder=2)
    ax.axhline(prevalence, color=PALETTE["diagonal"], ls="--", lw=1,
               label=f"Prevalence = {prevalence:.3f}", zorder=1)

    ax.set_xlabel("Recall  (sensitivity)")
    ax.set_ylabel("Precision  (PPV)")
    ax.set_xlim(-0.01, 1.01)
    ax.set_ylim(-0.01, 1.01)
    ax.set_title(title)
    ax.legend(loc="upper right")
    despine(ax, "none")

    if standalone:
        fig.tight_layout()
        save_fig(fig, save_path)
    return fig


# ═════════════════════════════════════════════════════════════════════
# 3. Calibration
# ═════════════════════════════════════════════════════════════════════

def plot_calibration(
    labels: np.ndarray,
    probabilities: np.ndarray,
    n_bins: int = 10,
    title: str = "Calibration",
    save_path: Optional[str] = None,
) -> plt.Figure:
    from sklearn.calibration import calibration_curve as sk_cal_curve
    frac_pos, mean_pred = sk_cal_curve(labels, probabilities, n_bins=n_bins,
                                        strategy="quantile")
    cal = compute_calibration_metrics(labels, probabilities, n_bins)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(3.6, 5.2),
                                    gridspec_kw={"height_ratios": [3, 1.2]},
                                    sharex=False)

    # ── Calibration curve ────────────────────────────────────────────
    ax1.plot([0, 1], [0, 1], "--", color=PALETTE["diagonal"], lw=1, zorder=1,
             label="Perfect calibration")
    ax1.plot(mean_pred, frac_pos, "o-", color=PALETTE["opera"], lw=1.8,
             markersize=5, zorder=3,
             label=f"Model  ECE={cal['ece']:.3f}")

    # Shaded region around perfect calibration
    ax1.fill_between([0, 1], [0, 0.05], [0.05, 0.1], alpha=0.05,
                     color=PALETTE["diagonal"], zorder=0)

    # HL annotation
    hl_p = cal.get("hl_pvalue", float("nan"))
    if not np.isnan(hl_p):
        color_hl = PALETTE["negative"] if hl_p >= 0.05 else PALETTE["positive"]
        ax1.text(0.97, 0.05,
                 f"HL  p = {hl_p:.3f}",
                 transform=ax1.transAxes, ha="right", va="bottom",
                 fontsize=7.5, color=color_hl)

    ax1.set_ylabel("Observed frequency")
    ax1.set_xlim(-0.01, 1.01)
    ax1.set_ylim(-0.01, 1.01)
    ax1.set_title(title)
    ax1.legend(loc="upper left", fontsize=7.5)
    despine(ax1, "none")

    # ── Prediction histogram ─────────────────────────────────────────
    bins = np.linspace(0, 1, 25)
    ax2.hist(probabilities[labels == 0], bins=bins, alpha=0.65,
             color=PALETTE["negative"], label="Negative", density=True)
    ax2.hist(probabilities[labels == 1], bins=bins, alpha=0.65,
             color=PALETTE["positive"], label="Positive", density=True)
    ax2.set_xlabel("Predicted probability")
    ax2.set_ylabel("Density")
    ax2.legend(fontsize=7.5)
    ax2.yaxis.set_major_locator(mticker.MaxNLocator(3))
    despine(ax2, "none")

    fig.tight_layout(h_pad=1.5)
    save_fig(fig, save_path)
    return fig


# ═════════════════════════════════════════════════════════════════════
# 4. Decision Curve Analysis
# ═════════════════════════════════════════════════════════════════════

def plot_decision_curve(
    labels: np.ndarray,
    probabilities: np.ndarray,
    title: str = "Decision Curve Analysis",
    save_path: Optional[str] = None,
    model_name: str = "Model",
) -> plt.Figure:
    dca = decision_curve_analysis(labels, probabilities)

    fig, ax = plt.subplots(figsize=(4.5, 3.2))

    ax.plot(dca["threshold"], dca["net_benefit_model"],
            color=PALETTE["opera"], lw=1.8, label=model_name, zorder=3)
    ax.plot(dca["threshold"], dca["net_benefit_treat_all"],
            color=PALETTE["tabular_rkkp"], lw=1.2, ls="--", label="Treat all", zorder=2)
    ax.axhline(0, color=PALETTE["diagonal"], ls=":", lw=1, label="Treat none", zorder=1)

    ax.set_xlabel("Decision threshold")
    ax.set_ylabel("Net benefit")
    ax.set_title(title)
    ax.legend(fontsize=7.5)
    ax.set_xlim(0, 1)

    y_lo = max(dca["net_benefit_treat_all"].min(), -0.15)
    ax.set_ylim(y_lo, None)
    despine(ax, "y")

    fig.tight_layout()
    save_fig(fig, save_path)
    return fig


# ═════════════════════════════════════════════════════════════════════
# 5. Time-dependent AUC (survival)
# ═════════════════════════════════════════════════════════════════════

def plot_timedep_auc(
    times: np.ndarray,
    events: np.ndarray,
    predicted_risk: np.ndarray,
    window_days: float,
    title: str = "Time-dependent AUC",
    save_path: Optional[str] = None,
    ax: Optional[plt.Axes] = None,
    color: Optional[str] = None,
    label: Optional[str] = None,
    n_bootstrap_ci: int = 200,
) -> plt.Figure:
    """
    IPCW-AUC vs time, with a shaded 95% CI ribbon.

    The curve shows how discriminative the model is at predicting
    the outcome up to each time point.
    """
    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(4.5, 3.2))
    else:
        fig = ax.figure

    color = color or PALETTE["opera"]
    plot_h = _plotting_horizons(window_days)
    ci_h   = _derive_time_horizons(window_days)

    curve = compute_timedep_auc_curve(
        times, events, predicted_risk,
        plot_horizons=plot_h,
        ci_horizons=ci_h,
        n_bootstrap_ci=n_bootstrap_ci,
    )

    if not curve["horizons"]:
        ax.text(0.5, 0.5, "Insufficient data", ha="center", va="center",
                transform=ax.transAxes)
        if standalone:
            save_fig(fig, save_path)
        return fig

    h   = np.array(curve["horizons"])
    auc = np.array(curve["auc"])
    lo  = np.array(curve["ci_lower"])
    hi  = np.array(curve["ci_upper"])

    lbl = label or (f"C = {np.nanmean(auc):.3f}"
                     f"  (n={curve['n_total']}, events={curve['n_events']})")

    ax.plot(h, auc, color=color, lw=1.8, label=lbl, zorder=3)
    if np.any(np.isfinite(lo)):
        ci_ribbon(ax, h, lo, hi, color)

    ax.axhline(0.5, color=PALETTE["diagonal"], ls="--", lw=1, zorder=1)
    ax.set_xlabel("Days from index date")
    ax.set_ylabel("IPCW-AUC")
    ax.set_ylim(None, None)
    ax.set_title(title)
    ax.legend(fontsize=7.5)
    ax.xaxis.set_major_locator(mticker.MaxNLocator(6, integer=True))
    despine(ax, "y")

    if standalone:
        fig.tight_layout()
        save_fig(fig, save_path)
    return fig


def plot_timedep_auc_comparison(
    curves: Dict[str, Dict],
    window_days: float,
    title: str = "Time-dependent AUC",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Overlay time-dependent AUC curves for multiple models.

    Parameters
    ----------
    curves : dict  model_name → output of compute_timedep_auc_curve()
    """
    fig, ax = plt.subplots(figsize=(4.8, 3.4))

    for (name, curve), color in zip(curves.items(), CATEGORICAL):
        if not curve["horizons"]:
            continue
        h   = np.array(curve["horizons"])
        auc = np.array(curve["auc"])
        lo  = np.array(curve["ci_lower"])
        hi  = np.array(curve["ci_upper"])

        mean_auc = np.nanmean(auc)
        lbl = f"{name.replace('_', ' ').title()}  ({mean_auc:.3f})"
        ax.plot(h, auc, color=color, lw=1.8, label=lbl, zorder=3)
        if np.any(np.isfinite(lo)):
            ci_ribbon(ax, h, lo, hi, color)

    ax.axhline(0.5, color=PALETTE["diagonal"], ls="--", lw=1, zorder=1)
    ax.set_xlabel("Days from index date")
    ax.set_ylabel("IPCW-AUC  (95% CI)")
    ax.set_title(title)
    ax.legend(fontsize=7.5, loc="lower right")
    despine(ax, "y")
    fig.tight_layout()
    save_fig(fig, save_path)
    return fig


# ═════════════════════════════════════════════════════════════════════
# 6. Threshold analysis
# ═════════════════════════════════════════════════════════════════════

def plot_threshold_analysis(
    labels: np.ndarray,
    probabilities: np.ndarray,
    title: str = "Threshold Analysis",
    save_path: Optional[str] = None,
) -> plt.Figure:
    thresholds = np.arange(0.02, 0.99, 0.01)
    curves: Dict[str, list] = {
        "Sensitivity": [], "Specificity": [], "PPV": [], "F1": []
    }
    colors = [PALETTE["positive"], PALETTE["negative"],
               PALETTE["tabular_rkkp"], PALETTE["opera"]]
    styles = ["-", "-", "--", ":"]

    for t in thresholds:
        m = compute_discrimination_metrics(labels, probabilities, t)
        curves["Sensitivity"].append(m["sensitivity"])
        curves["Specificity"].append(m["specificity"])
        curves["PPV"].append(m["ppv"])
        curves["F1"].append(m["f1"])

    fig, ax = plt.subplots(figsize=(4.5, 3.2))
    for (name, vals), color, ls in zip(curves.items(), colors, styles):
        ax.plot(thresholds, vals, color=color, lw=1.6, ls=ls, label=name)

    ax.set_xlabel("Decision threshold")
    ax.set_ylabel("Metric value")
    ax.set_title(title)
    ax.legend(fontsize=7.5)
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.01, 1.01)
    despine(ax, "y")
    fig.tight_layout()
    save_fig(fig, save_path)
    return fig


# ═════════════════════════════════════════════════════════════════════
# 7. Combined 2×2 evaluation panel
# ═════════════════════════════════════════════════════════════════════

def plot_evaluation_panel(
    labels: np.ndarray,
    probabilities: np.ndarray,
    times: Optional[np.ndarray] = None,
    events: Optional[np.ndarray] = None,
    survival_probabilities: Optional[np.ndarray] = None,
    window_days: Optional[float] = None,
    bootstrap_ci: Optional[Dict] = None,
    outcome_name: str = "",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    2×2 panel: ROC | PRC | Calibration | DCA or time-dependent AUC.

    The bottom-right panel shows the time-dependent AUC if survival data
    is provided, otherwise the threshold analysis.
    """
    fig, axes = plt.subplots(2, 2, figsize=(7.0, 6.0))
    (ax_roc, ax_prc), (ax_cal, ax_bot) = axes

    title_sfx = f" — {outcome_name.replace('_', ' ').title()}" if outcome_name else ""

    plot_roc_curve(labels, probabilities, ax=ax_roc,
                   title=f"ROC{title_sfx}", bootstrap_ci=bootstrap_ci)
    plot_prc(labels, probabilities, ax=ax_prc,
             title=f"Precision-Recall{title_sfx}", bootstrap_ci=bootstrap_ci)

    # Calibration (re-implement inline to fit on existing axis)
    from sklearn.calibration import calibration_curve as sk_cal_curve
    frac_pos, mean_pred = sk_cal_curve(labels, probabilities, n_bins=10,
                                        strategy="quantile")
    cal = compute_calibration_metrics(labels, probabilities)
    ax_cal.plot([0, 1], [0, 1], "--", color=PALETTE["diagonal"], lw=1)
    ax_cal.plot(mean_pred, frac_pos, "o-", color=PALETTE["opera"], lw=1.8,
                markersize=4, label=f"ECE={cal['ece']:.3f}")
    hl_p = cal.get("hl_pvalue", float("nan"))
    if not np.isnan(hl_p):
        color_hl = PALETTE["negative"] if hl_p >= 0.05 else PALETTE["positive"]
        ax_cal.text(0.97, 0.05, f"HL p={hl_p:.3f}",
                    transform=ax_cal.transAxes, ha="right", va="bottom",
                    fontsize=7, color=color_hl)
    ax_cal.set_xlabel("Predicted probability")
    ax_cal.set_ylabel("Observed frequency")
    ax_cal.set_title(f"Calibration{title_sfx}")
    ax_cal.legend(fontsize=7.5)
    ax_cal.set_xlim(-0.01, 1.01)
    ax_cal.set_ylim(-0.01, 1.01)
    despine(ax_cal, "none")

    if times is not None and events is not None and window_days is not None:
        risk_scores = survival_probabilities if survival_probabilities is not None else probabilities
        plot_timedep_auc(times, events, risk_scores,
                         window_days=window_days,
                         title=f"Time-dependent AUC{title_sfx}",
                         ax=ax_bot)
    else:
        plot_threshold_analysis(labels, probabilities,
                                title=f"Threshold analysis{title_sfx}",
                                save_path=None)
        # reuse ax_bot
        dca = decision_curve_analysis(labels, probabilities)
        ax_bot.plot(dca["threshold"], dca["net_benefit_model"],
                    color=PALETTE["opera"], lw=1.8, label="Model")
        ax_bot.plot(dca["threshold"], dca["net_benefit_treat_all"],
                    color=PALETTE["tabular_rkkp"], lw=1.2, ls="--", label="Treat all")
        ax_bot.axhline(0, color=PALETTE["diagonal"], ls=":", lw=1)
        ax_bot.set_xlabel("Decision threshold")
        ax_bot.set_ylabel("Net benefit")
        ax_bot.set_title(f"Decision Curve{title_sfx}")
        ax_bot.legend(fontsize=7.5)
        ax_bot.set_xlim(0, 1)
        despine(ax_bot, "y")

    for label, ax in zip("ABCD", axes.flat):
        add_panel_label(ax, label)

    fig.tight_layout(h_pad=2.0, w_pad=2.0)
    save_fig(fig, save_path)
    return fig


# ═════════════════════════════════════════════════════════════════════
# 8. Full evaluation figure set
# ═════════════════════════════════════════════════════════════════════

def plot_full_evaluation(
    labels: np.ndarray,
    probabilities: np.ndarray,
    output_dir: str,
    prefix: str = "",
    bootstrap_ci: Optional[Dict] = None,
    times: Optional[np.ndarray] = None,
    events: Optional[np.ndarray] = None,
    survival_probabilities: Optional[np.ndarray] = None,
    window_days: Optional[float] = None,
    outcome_name: str = "",
) -> Dict[str, plt.Figure]:
    """Generate the complete set of evaluation plots and save them."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    p = f"{prefix}_" if prefix else ""

    figs = {}

    figs["panel"] = plot_evaluation_panel(
        labels, probabilities,
        times=times, events=events, survival_probabilities=survival_probabilities,
        window_days=window_days,
        bootstrap_ci=bootstrap_ci, outcome_name=outcome_name,
        save_path=str(out / f"{p}evaluation_panel.png"),
    )
    figs["roc"] = plot_roc_curve(
        labels, probabilities, bootstrap_ci=bootstrap_ci,
        save_path=str(out / f"{p}roc_curve.png"),
    )
    figs["prc"] = plot_prc(
        labels, probabilities, bootstrap_ci=bootstrap_ci,
        save_path=str(out / f"{p}precision_recall.png"),
    )
    figs["calibration"] = plot_calibration(
        labels, probabilities,
        save_path=str(out / f"{p}calibration.png"),
    )
    figs["dca"] = plot_decision_curve(
        labels, probabilities,
        save_path=str(out / f"{p}decision_curve.png"),
    )
    figs["threshold"] = plot_threshold_analysis(
        labels, probabilities,
        save_path=str(out / f"{p}threshold_analysis.png"),
    )

    if times is not None and events is not None and window_days is not None:
        risk_scores = survival_probabilities if survival_probabilities is not None else probabilities
        figs["timedep_auc"] = plot_timedep_auc(
            times, events, risk_scores, window_days=window_days,
            save_path=str(out / f"{p}timedep_auc.png"),
        )

    return figs
