"""
OPERA Evaluation Suite — Comprehensive metrics for binary classification.

Covers:
  1. Discrimination: AUROC, AUPRC, sensitivity, specificity, F1, MCC
  2. Calibration: Brier score, expected calibration error, Hosmer-Lemeshow
  3. Confusion matrices at multiple thresholds
  4. Decision curve analysis (net benefit)
  5. Bootstrap confidence intervals for all scalar metrics
  6. Survival analysis metrics: C-index, IPCW-AUC, IPCW-Brier

All functions take numpy arrays of labels and probabilities.
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    classification_report,
    precision_recall_curve,
    roc_curve,
    f1_score,
    matthews_corrcoef,
    log_loss,
)
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression

try:
    from scipy.stats import chi2 as _chi2

    _SCIPY_AVAILABLE = True
except ImportError:
    _SCIPY_AVAILABLE = False


# ═════════════════════════════════════════════════════════════════════
# 1. Discrimination metrics
# ═════════════════════════════════════════════════════════════════════


def compute_discrimination_metrics(
    labels: np.ndarray,
    probabilities: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """Compute all discrimination metrics at a given threshold."""
    preds = (probabilities >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(labels, preds, labels=[0, 1]).ravel()

    metrics = {
        "auroc": roc_auc_score(labels, probabilities),
        "auprc": average_precision_score(labels, probabilities),
        "sensitivity": tp / (tp + fn) if (tp + fn) > 0 else 0.0,
        "specificity": tn / (tn + fp) if (tn + fp) > 0 else 0.0,
        "ppv": tp / (tp + fp) if (tp + fp) > 0 else 0.0,
        "npv": tn / (tn + fn) if (tn + fn) > 0 else 0.0,
        "f1": f1_score(labels, preds, zero_division=0),
        "mcc": matthews_corrcoef(labels, preds),
        "accuracy": (tp + tn) / (tp + tn + fp + fn),
        "log_loss": log_loss(labels, probabilities),
        "prevalence": labels.mean(),
        "n_positive": int(labels.sum()),
        "n_negative": int((1 - labels).sum()),
        "n_total": len(labels),
        "threshold": threshold,
    }
    return metrics


def validate_binary_evaluation_inputs(
    labels: np.ndarray,
    probabilities: np.ndarray,
) -> None:
    """Fail clearly when scalar binary metrics such as AUROC are undefined."""
    if len(labels) != len(probabilities):
        raise ValueError(
            "Binary labels and probabilities must have the same length; "
            f"got {len(labels)} labels and {len(probabilities)} probabilities."
        )
    if len(labels) == 0:
        raise ValueError("Binary evaluation has no eligible labelled patients.")
    if len(np.unique(labels)) < 2:
        raise ValueError(
            "Binary evaluation needs both positive and negative labels for "
            f"AUROC; got labels={sorted(set(labels.tolist()))}."
        )
    if np.any((probabilities < 0.0) | (probabilities > 1.0)):
        raise ValueError("Predicted probabilities must be in the [0, 1] range.")


def compute_metrics_at_thresholds(
    labels: np.ndarray,
    probabilities: np.ndarray,
    thresholds: Optional[List[float]] = None,
) -> pd.DataFrame:
    """Compute discrimination metrics across multiple thresholds."""
    if thresholds is None:
        thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    rows = []
    for t in thresholds:
        m = compute_discrimination_metrics(labels, probabilities, threshold=t)
        rows.append(m)
    return pd.DataFrame(rows)


def find_optimal_threshold(
    labels: np.ndarray,
    probabilities: np.ndarray,
    criterion: str = "youden",
) -> Tuple[float, Dict[str, float]]:
    """
    Find the optimal threshold using:
      - "youden": maximises sensitivity + specificity - 1
      - "f1": maximises F1 score
    """
    fpr, tpr, thresholds_roc = roc_curve(labels, probabilities)

    if criterion == "youden":
        j_scores = tpr - fpr
        best_idx = np.argmax(j_scores)
        best_threshold = thresholds_roc[best_idx]
    elif criterion == "f1":
        precision, recall, thresholds_pr = precision_recall_curve(labels, probabilities)
        f1_scores = 2 * (precision * recall) / (precision + recall + 1e-12)
        best_idx = np.argmax(f1_scores)
        best_threshold = thresholds_pr[min(best_idx, len(thresholds_pr) - 1)]
    else:
        raise ValueError(f"Unknown criterion: {criterion}")

    metrics = compute_discrimination_metrics(labels, probabilities, best_threshold)
    return best_threshold, metrics


# ═════════════════════════════════════════════════════════════════════
# 2. Calibration metrics
# ═════════════════════════════════════════════════════════════════════


def compute_calibration_metrics(
    labels: np.ndarray,
    probabilities: np.ndarray,
    n_bins: int = 10,
) -> Dict[str, float]:
    """Brier score, ECE, MCE, Hosmer-Lemeshow test, and calibration curve data."""
    brier = brier_score_loss(labels, probabilities)

    # ECE: weighted average of |accuracy - confidence| per bin
    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    bin_data = []

    for i in range(n_bins):
        mask = (probabilities >= bin_edges[i]) & (probabilities < bin_edges[i + 1])
        if mask.sum() == 0:
            continue
        bin_acc = labels[mask].mean()
        bin_conf = probabilities[mask].mean()
        bin_count = mask.sum()
        bin_data.append(
            {
                "bin_lower": bin_edges[i],
                "bin_upper": bin_edges[i + 1],
                "bin_accuracy": bin_acc,
                "bin_confidence": bin_conf,
                "bin_count": int(bin_count),
            }
        )
        ece += (bin_count / len(labels)) * abs(bin_acc - bin_conf)

    # Maximum calibration error
    if bin_data:
        mce = max(abs(b["bin_accuracy"] - b["bin_confidence"]) for b in bin_data)
    else:
        mce = 0.0

    result = {
        "brier_score": brier,
        "ece": ece,
        "mce": mce,
        "bin_data": bin_data,
    }
    result.update(hosmer_lemeshow_test(labels, probabilities))
    result.update(calibration_intercept_slope(labels, probabilities))
    return result


def calibration_intercept_slope(
    labels: np.ndarray,
    probabilities: np.ndarray,
) -> Dict[str, float]:
    """
    Estimate calibration intercept and slope.

    Fits logistic regression: y ~ logit(predicted_probability). Perfect
    calibration has intercept 0 and slope 1. Returns NaN when the fit is not
    identifiable, such as one-class labels.
    """
    if len(labels) < 3 or len(np.unique(labels)) < 2:
        return {
            "calibration_intercept": float("nan"),
            "calibration_slope": float("nan"),
        }

    eps = 1e-6
    probs = np.clip(probabilities.astype(float), eps, 1.0 - eps)
    logits = np.log(probs / (1.0 - probs)).reshape(-1, 1)
    try:
        clf = LogisticRegression(penalty=None, solver="lbfgs", max_iter=1000)
        clf.fit(logits, labels.astype(int))
    except TypeError:
        clf = LogisticRegression(penalty="none", solver="lbfgs", max_iter=1000)
        clf.fit(logits, labels.astype(int))
    except Exception:
        return {
            "calibration_intercept": float("nan"),
            "calibration_slope": float("nan"),
        }

    return {
        "calibration_intercept": float(clf.intercept_[0]),
        "calibration_slope": float(clf.coef_[0, 0]),
    }


def hosmer_lemeshow_test(
    labels: np.ndarray,
    probabilities: np.ndarray,
    n_groups: int = 10,
) -> Dict[str, float]:
    """
    Hosmer-Lemeshow goodness-of-fit test for calibration.

    Divides patients into deciles by predicted probability and compares
    observed vs expected event counts via chi-square.

    H0: model is well calibrated.
    p < 0.05 → reject H0 → poor calibration.

    Returns dict with keys: hl_statistic, hl_pvalue, hl_df.
    NaN values are returned if scipy is unavailable or n < n_groups.
    """
    if len(labels) < n_groups:
        return {
            "hl_statistic": float("nan"),
            "hl_pvalue": float("nan"),
            "hl_df": n_groups - 2,
        }

    order = np.argsort(probabilities)
    labels_s = labels[order]
    probs_s = probabilities[order]

    hl_stat = 0.0
    indices = np.array_split(np.arange(len(labels)), n_groups)
    for idx in indices:
        obs_pos = labels_s[idx].sum()
        exp_pos = probs_s[idx].sum()
        n_g = len(idx)
        obs_neg = n_g - obs_pos
        exp_neg = n_g - exp_pos
        hl_stat += (obs_pos - exp_pos) ** 2 / max(exp_pos, 1e-8)
        hl_stat += (obs_neg - exp_neg) ** 2 / max(exp_neg, 1e-8)

    df = n_groups - 2
    if _SCIPY_AVAILABLE:
        p_value = float(1.0 - _chi2.cdf(hl_stat, df))
    else:
        p_value = float("nan")

    return {"hl_statistic": float(hl_stat), "hl_pvalue": p_value, "hl_df": df}


def get_calibration_curve_data(
    labels: np.ndarray,
    probabilities: np.ndarray,
    n_bins: int = 10,
    strategy: str = "uniform",
) -> Tuple[np.ndarray, np.ndarray]:
    """Returns (fraction_of_positives, mean_predicted_value) for plotting."""
    return calibration_curve(labels, probabilities, n_bins=n_bins, strategy=strategy)


# ═════════════════════════════════════════════════════════════════════
# 3. Confusion matrices
# ═════════════════════════════════════════════════════════════════════


def get_confusion_matrix(
    labels: np.ndarray,
    probabilities: np.ndarray,
    threshold: float = 0.5,
) -> np.ndarray:
    """Returns 2x2 confusion matrix [[TN, FP], [FN, TP]]."""
    preds = (probabilities >= threshold).astype(int)
    return confusion_matrix(labels, preds, labels=[0, 1])


def get_classification_report(
    labels: np.ndarray,
    probabilities: np.ndarray,
    threshold: float = 0.5,
) -> str:
    preds = (probabilities >= threshold).astype(int)
    return classification_report(
        labels, preds, target_names=["Negative", "Positive"], digits=4
    )


# ═════════════════════════════════════════════════════════════════════
# 4. Decision curve analysis
# ═════════════════════════════════════════════════════════════════════


def decision_curve_analysis(
    labels: np.ndarray,
    probabilities: np.ndarray,
    thresholds: Optional[np.ndarray] = None,
) -> pd.DataFrame:
    """
    Compute net benefit at various threshold probabilities.

    Net benefit = TP/N - FP/N * (p_t / (1 - p_t))

    where p_t is the threshold probability.
    """
    if thresholds is None:
        thresholds = np.arange(0.01, 0.99, 0.01)

    N = len(labels)
    prevalence = labels.mean()
    rows = []

    for pt in thresholds:
        preds = (probabilities >= pt).astype(int)
        tp = ((preds == 1) & (labels == 1)).sum()
        fp = ((preds == 1) & (labels == 0)).sum()

        net_benefit_model = tp / N - fp / N * (pt / (1 - pt))
        net_benefit_treat_all = prevalence - (1 - prevalence) * (pt / (1 - pt))

        rows.append(
            {
                "threshold": pt,
                "net_benefit_model": net_benefit_model,
                "net_benefit_treat_all": net_benefit_treat_all,
                "net_benefit_treat_none": 0.0,
            }
        )

    return pd.DataFrame(rows)


def high_risk_enrichment(
    labels: np.ndarray,
    probabilities: np.ndarray,
    fractions: Optional[List[float]] = None,
) -> pd.DataFrame:
    """
    Summarize event enrichment among highest-risk patients.

    Rows report the observed event rate and lift versus population prevalence
    among the top predicted-risk fraction. This is useful for clinical
    interpretation of risk stratification.
    """
    if fractions is None:
        fractions = [0.01, 0.02, 0.05, 0.10, 0.20]

    labels = labels.astype(int)
    probabilities = probabilities.astype(float)
    prevalence = float(labels.mean()) if len(labels) else float("nan")
    order = np.argsort(-probabilities)
    rows = []
    for frac in fractions:
        n_top = max(1, int(np.ceil(len(labels) * frac)))
        idx = order[:n_top]
        top_labels = labels[idx]
        event_rate = float(top_labels.mean()) if n_top > 0 else float("nan")
        rows.append(
            {
                "top_fraction": frac,
                "n_top": int(n_top),
                "n_events": int(top_labels.sum()),
                "event_rate": event_rate,
                "population_prevalence": prevalence,
                "enrichment": (
                    event_rate / prevalence
                    if prevalence and np.isfinite(prevalence)
                    else float("nan")
                ),
                "min_probability": float(probabilities[idx].min())
                if n_top > 0
                else float("nan"),
                "max_probability": float(probabilities[idx].max())
                if n_top > 0
                else float("nan"),
            }
        )
    return pd.DataFrame(rows)


# ═════════════════════════════════════════════════════════════════════
# 5. Bootstrap confidence intervals
# ═════════════════════════════════════════════════════════════════════


def bootstrap_metrics(
    labels: np.ndarray,
    probabilities: np.ndarray,
    n_bootstrap: int = 1000,
    seed: int = 42,
    ci: float = 0.95,
    threshold: float = 0.5,
) -> Dict[str, Dict[str, float]]:
    """
    Bootstrap CIs for all scalar binary classification metrics.

    Returns dict: metric_name → {"mean", "lower", "upper", "std"}.
    """
    rng = np.random.RandomState(seed)
    n = len(labels)
    alpha = (1 - ci) / 2

    metric_names = [
        "auroc",
        "auprc",
        "brier_score",
        "sensitivity",
        "specificity",
        "ppv",
        "npv",
        "f1",
        "mcc",
        "accuracy",
        "log_loss",
    ]
    boot_results: Dict[str, list] = {m: [] for m in metric_names}

    for _ in range(n_bootstrap):
        idx = _stratified_bootstrap_indices(n, labels, rng)
        y_b = labels[idx]
        p_b = probabilities[idx]

        if len(np.unique(y_b)) < 2:
            continue

        preds_b = (p_b >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_b, preds_b, labels=[0, 1]).ravel()

        boot_results["auroc"].append(roc_auc_score(y_b, p_b))
        boot_results["auprc"].append(average_precision_score(y_b, p_b))
        boot_results["brier_score"].append(brier_score_loss(y_b, p_b))
        boot_results["sensitivity"].append(tp / (tp + fn) if (tp + fn) > 0 else 0.0)
        boot_results["specificity"].append(tn / (tn + fp) if (tn + fp) > 0 else 0.0)
        boot_results["ppv"].append(tp / (tp + fp) if (tp + fp) > 0 else 0.0)
        boot_results["npv"].append(tn / (tn + fn) if (tn + fn) > 0 else 0.0)
        boot_results["f1"].append(f1_score(y_b, preds_b, zero_division=0))
        boot_results["mcc"].append(matthews_corrcoef(y_b, preds_b))
        boot_results["accuracy"].append((tp + tn) / (tp + tn + fp + fn))
        boot_results["log_loss"].append(log_loss(y_b, p_b))

    ci_results = {}
    for m in metric_names:
        vals = np.array([v for v in boot_results[m] if np.isfinite(v)])
        if len(vals) == 0:
            ci_results[m] = {
                "mean": float("nan"),
                "std": float("nan"),
                "lower": float("nan"),
                "upper": float("nan"),
            }
        else:
            ci_results[m] = {
                "mean": float(vals.mean()),
                "std": float(vals.std()),
                "lower": float(np.quantile(vals, alpha)),
                "upper": float(np.quantile(vals, 1 - alpha)),
            }
    return ci_results


def _stratified_bootstrap_indices(n, events, rng):
    """Return event-stratified bootstrap indices preserving case/control counts."""
    events = np.asarray(events)
    pos_idx = np.where(events == 1)[0]
    neg_idx = np.where(events != 1)[0]
    boot_pos = (
        pos_idx[rng.randint(0, len(pos_idx), size=len(pos_idx))]
        if len(pos_idx)
        else np.array([], dtype=int)
    )
    boot_neg = (
        neg_idx[rng.randint(0, len(neg_idx), size=len(neg_idx))]
        if len(neg_idx)
        else np.array([], dtype=int)
    )
    return np.concatenate([boot_pos, boot_neg])


# ═════════════════════════════════════════════════════════════════════
# 6. Survival analysis metrics
# ═════════════════════════════════════════════════════════════════════


def _derive_time_horizons(window_days: float) -> List[float]:
    """
    Clinically meaningful IPCW reporting horizons (tables + JSON).

    Includes the window endpoint plus clinically interpretable intermediates:
        ≤ 45d  (e.g. 30d AKI):  [30]  — just the endpoint, no sub-window meaning
        ≤ 120d (e.g. 90d infx): [30, 60, 90]  — monthly
        ≤ 400d (e.g. 1y):       [91, 182, 274, 365]  — quarterly
        > 400d (e.g. 2y):       [182, 365, 548, 730]  — semi-annual
    """
    w = int(round(window_days))
    if window_days <= 45:
        return [float(w)]
    elif window_days <= 120:
        return [30.0, 60.0, float(w)]
    elif window_days <= 400:
        return [91.0, 182.0, 274.0, float(w)]
    else:
        return [182.0, 365.0, 548.0, float(w)]


def _plotting_horizons(window_days: float) -> List[float]:
    """
    Dense grid of time points for smooth time-dependent AUC curve plots.

    Finer than reporting horizons — meant for visualization only.
        ≤ 45d:  every 5 days
        ≤ 120d: every 14 days
        > 120d: every 28 days (monthly)
    """
    if window_days <= 45:
        step = 5
    elif window_days <= 120:
        step = 14
    else:
        step = 28
    w = int(round(window_days))
    points = list(range(step, w + 1, step))
    if not points or points[-1] < w:
        points.append(w)
    return [float(p) for p in points]


def compute_timedep_auc_curve(
    times: np.ndarray,
    events: np.ndarray,
    predicted_risk: np.ndarray,
    plot_horizons: List[float],
    ci_horizons: Optional[List[float]] = None,
    n_bootstrap_ci: int = 200,
    seed: int = 42,
) -> Dict:
    """
    Compute time-dependent IPCW-AUC at a dense grid of horizons for plotting,
    plus a bootstrap CI band at a sparse set of anchor points.

    Strategy
    --------
    - Point estimates: computed at every ``plot_horizons`` point (fast).
    - CI bands: bootstrapped only at ``ci_horizons`` (sparse), then linearly
      interpolated between anchors.  200 bootstrap samples is sufficient for
      a CI ribbon — smoothness matters more than precision here.

    Parameters
    ----------
    times, events, predicted_risk : (N,) arrays, all patients.
    plot_horizons : dense grid (from _plotting_horizons).
    ci_horizons   : sparse anchor points (from _derive_time_horizons).
                    Defaults to every 4th plot horizon.
    n_bootstrap_ci : bootstrap samples for CI (200 is sufficient for a ribbon).

    Returns
    -------
    {
        "horizons":  list of horizon values (days),
        "auc":       list of point-estimate IPCW-AUC values,
        "ci_lower":  list of lower-CI values (linearly interpolated),
        "ci_upper":  list of upper-CI values (linearly interpolated),
        "n_events":  total observed events,
        "n_total":   total patients,
    }
    """
    valid = np.isfinite(times) & np.isfinite(predicted_risk) & (events >= 0)
    times = times[valid].astype(float)
    events = events[valid].astype(int)
    predicted_risk = predicted_risk[valid].astype(float)

    if ci_horizons is None:
        step = max(1, len(plot_horizons) // 4)
        ci_horizons = [plot_horizons[i] for i in range(0, len(plot_horizons), step)]
        if plot_horizons[-1] not in ci_horizons:
            ci_horizons.append(plot_horizons[-1])

    max_obs = float(times.max())
    G_fn = _km_censoring_fn(times, events)

    # ── Point estimates at full plotting grid ─────────────────────────
    aucs = []
    valid_horizons = []
    for h in plot_horizons:
        if h > max_obs:
            continue
        res = compute_ipcw_metrics_at_horizon(times, events, predicted_risk, h, G_fn)
        auc = res.get("ipcw_auc", float("nan"))
        aucs.append(auc)
        valid_horizons.append(h)

    if not valid_horizons:
        return {
            "horizons": [],
            "auc": [],
            "ci_lower": [],
            "ci_upper": [],
            "n_events": int((events == 1).sum()),
            "n_total": len(times),
        }

    # ── Bootstrap CI at anchor points ─────────────────────────────────
    rng = np.random.RandomState(seed)
    n = len(times)
    ci_anchor_aucs: Dict[float, list] = {h: [] for h in ci_horizons if h <= max_obs}

    for _ in range(n_bootstrap_ci):
        idx = _stratified_bootstrap_indices(n, events, rng)
        t_b, e_b, r_b = times[idx], events[idx], predicted_risk[idx]
        if (e_b == 1).sum() == 0:  # no primary events in this bootstrap sample
            continue
        G_b = _km_censoring_fn(t_b, e_b)
        max_b = float(t_b.max())
        for h in ci_anchor_aucs:
            if h > max_b:
                continue
            res = compute_ipcw_metrics_at_horizon(t_b, e_b, r_b, h, G_b)
            v = res.get("ipcw_auc", float("nan"))
            if np.isfinite(v):
                ci_anchor_aucs[h].append(v)

    # Compute CI at anchors
    anchor_h, anchor_lo, anchor_hi = [], [], []
    for h in sorted(ci_anchor_aucs.keys()):
        vals = np.array(ci_anchor_aucs[h])
        if len(vals) < 10:
            continue
        anchor_h.append(h)
        anchor_lo.append(float(np.quantile(vals, 0.025)))
        anchor_hi.append(float(np.quantile(vals, 0.975)))

    # Interpolate CI to full grid
    h_arr = np.array(valid_horizons)
    if anchor_h:
        lo_interp = np.interp(h_arr, anchor_h, anchor_lo)
        hi_interp = np.interp(h_arr, anchor_h, anchor_hi)
    else:
        lo_interp = np.full_like(h_arr, float("nan"))
        hi_interp = np.full_like(h_arr, float("nan"))

    return {
        "horizons": valid_horizons,
        "auc": [float(v) for v in aucs],
        "ci_lower": lo_interp.tolist(),
        "ci_upper": hi_interp.tolist(),
        "n_events": int((events == 1).sum()),  # primary events only
        "n_total": len(times),
    }


def _km_censoring_fn(times: np.ndarray, events: np.ndarray):
    """
    Kaplan-Meier estimate of the censoring survival function G(t) = P(C > t).
    Returns a callable G(t) that gives the censoring probability just before t.

    Censoring indicator is 1 for any patient who did NOT experience the primary
    event (event != 1), which covers both admin-censored (event=0) and
    competing-death (event=2) patients.  This gives a valid IPCW weight because
    the censoring distribution is estimated from the non-primary-event process.
    """
    cens_event = (events != 1).astype(int)  # 1 for admin censored OR competing death
    unique_times = np.sort(np.unique(times))

    G = 1.0
    step_times = [0.0]
    step_vals = [1.0]

    for t in unique_times:
        n_at_risk = np.sum(times >= t)
        n_cens = np.sum((times == t) & (cens_event == 1))
        if n_at_risk > 0:
            G *= 1.0 - n_cens / n_at_risk
        step_times.append(float(t))
        step_vals.append(float(G))

    step_times = np.array(step_times)
    step_vals = np.array(step_vals)

    def G_fn(t: float) -> float:
        """G(t-): censoring survival just before time t."""
        idx = int(np.searchsorted(step_times, t, side="left")) - 1
        idx = max(0, min(idx, len(step_vals) - 1))
        return float(step_vals[idx])

    return G_fn


def compute_concordance_index(
    times: np.ndarray,
    events: np.ndarray,
    predicted_risk: np.ndarray,
) -> float:
    """
    Harrell's C-statistic (concordance index) for censored survival data.

    Counts concordant pairs among all comparable pairs:
        comparable: i had event and t_i < t_j
        concordant: risk_i > risk_j  (higher risk → earlier event)

    All patients are used (censored patients contribute as controls).
    """
    t_i = times[:, None]  # (n, 1)
    t_j = times[None, :]  # (1, n)
    e_i = events[:, None].astype(float)
    r_i = predicted_risk[:, None]
    r_j = predicted_risk[None, :]

    comparable = (e_i == 1) & (t_i < t_j)  # (n, n) bool
    concordant = comparable & (r_i > r_j)
    tied_risk = comparable & (r_i == r_j)

    n_comp = float(comparable.sum())
    if n_comp == 0:
        return float("nan")
    return float((concordant.sum() + 0.5 * tied_risk.sum()) / n_comp)


def compute_ipcw_metrics_at_horizon(
    times: np.ndarray,
    events: np.ndarray,
    predicted_risk: np.ndarray,
    horizon: float,
    G_fn,
) -> Dict[str, float]:
    """
    Compute cause-specific IPCW-weighted AUC and Brier score at a horizon.

    This treats competing events as non-primary-event censoring for the
    cause-specific hazard target. Fine-Gray subdistribution hazards are not
    implemented here.

    Parameters
    ----------
    times          : time-to-event or censoring in days for each patient.
    events         : 1 = observed primary event, 0 = censored, 2 = competing event.
    predicted_risk : P(event ≤ horizon) for each patient.
    horizon        : prediction horizon in days.
    G_fn           : censoring survival function from _km_censoring_fn().

    Returns
    -------
    dict with keys: ipcw_auc, ipcw_brier, n_cases, n_controls, n_excluded
    """
    n = len(times)

    # Cases: primary event observed before horizon
    case_mask = (times <= horizon) & (events == 1)
    # Controls: still at risk past horizon (any event status after horizon)
    ctrl_mask = times > horizon
    # Excluded: did not have primary event and time <= horizon
    # (admin censored OR competing death — both uninformative for the binary
    #  primary-event endpoint at this horizon)
    excl_mask = (times <= horizon) & (events != 1)

    n_cases = int(case_mask.sum())
    n_controls = int(ctrl_mask.sum())
    n_excluded = int(excl_mask.sum())

    # ── IPCW AUC ─────────────────────────────────────────────────────
    ipcw_auc = float("nan")
    if n_cases > 0 and n_controls > 0:
        case_times = times[case_mask]
        case_risks = predicted_risk[case_mask]
        ctrl_risks = predicted_risk[ctrl_mask]

        # IPCW weights: 1/G(t_i) for cases, 1/G(τ) for controls
        case_weights = np.array([1.0 / max(G_fn(t), 1e-6) for t in case_times])
        ctrl_weight = 1.0 / max(G_fn(horizon), 1e-6)

        # Vectorised concordance (case risk > control risk)
        r_case = case_risks[:, None]  # (n_cases, 1)
        r_ctrl = ctrl_risks[None, :]  # (1, n_controls)
        w_case = case_weights[:, None]  # (n_cases, 1)

        conc = (r_case > r_ctrl).astype(float) + 0.5 * (r_case == r_ctrl).astype(float)
        num = (conc * w_case * ctrl_weight).sum()
        den = (w_case * ctrl_weight).sum() * n_controls
        ipcw_auc = float(num / den) if den > 0 else float("nan")

    # ── IPCW Brier score ─────────────────────────────────────────────
    brier_sum = 0.0
    for i in range(n):
        p_i = float(predicted_risk[i])
        if case_mask[i]:
            w = 1.0 / max(G_fn(float(times[i])), 1e-6)
            brier_sum += w * (1.0 - p_i) ** 2
        elif ctrl_mask[i]:
            w = 1.0 / max(G_fn(horizon), 1e-6)
            brier_sum += w * (0.0 - p_i) ** 2
        # excluded patients contribute 0

    ipcw_brier = brier_sum / n if n > 0 else float("nan")

    return {
        "ipcw_auc": ipcw_auc,
        "ipcw_brier": ipcw_brier,
        "n_cases": n_cases,
        "n_controls": n_controls,
        "n_excluded": n_excluded,
    }


def compute_survival_metrics(
    times: np.ndarray,
    events: np.ndarray,
    predicted_risk: np.ndarray,
    time_horizons: Optional[List[float]] = None,
) -> Dict:
    """
    Compute survival analysis metrics that use ALL patients (including censored).

    Parameters
    ----------
    times          : (N,) time-to-event or censoring in days.
    events         : (N,) 1 = observed event, 0 = censored.
    predicted_risk : (N,) model output, interpreted as P(event ≤ horizon).
    time_horizons  : list of day horizons for time-dependent metrics.
                     Defaults to [30, 90, 365, 730].

    Returns
    -------
    dict with:
        "concordance_index"          : Harrell's C over all comparable pairs
        "n_total"                    : total patients used
        "n_events"                   : observed events
        "per_horizon"                : {horizon_days: {"ipcw_auc", "ipcw_brier", ...}}
    """
    if time_horizons is None:
        time_horizons = [30.0, 90.0, 365.0, 730.0]

    # Remove patients with missing survival data
    valid = np.isfinite(times) & np.isfinite(predicted_risk) & (events >= 0)
    times = times[valid].astype(float)
    events = events[valid].astype(int)
    predicted_risk = predicted_risk[valid].astype(float)

    n_total = int(len(times))
    n_events = int((events == 1).sum())  # primary events only

    if n_total < 2 or n_events == 0:
        return {
            "concordance_index": float("nan"),
            "n_total": n_total,
            "n_events": n_events,
            "per_horizon": {},
        }

    c_index = compute_concordance_index(times, events, predicted_risk)

    # Censoring distribution for IPCW weights
    G_fn = _km_censoring_fn(times, events)

    # Time-dependent metrics at each horizon
    max_obs_time = float(times.max())
    per_horizon = {}
    for h in time_horizons:
        if h > max_obs_time:
            continue  # no patients can be controls at this horizon
        label = f"{int(h)}d"
        per_horizon[label] = compute_ipcw_metrics_at_horizon(
            times, events, predicted_risk, h, G_fn
        )

    return {
        "concordance_index": c_index,
        "n_total": n_total,
        "n_events": n_events,
        "per_horizon": per_horizon,
    }


def bootstrap_survival_metrics(
    times: np.ndarray,
    events: np.ndarray,
    predicted_risk: np.ndarray,
    time_horizons: Optional[List[float]] = None,
    n_bootstrap: int = 500,
    seed: int = 42,
    ci: float = 0.95,
) -> Dict:
    """
    Bootstrap confidence intervals for the C-index and per-horizon IPCW AUC.
    Uses fewer bootstrap samples than binary metrics (survival metrics are slower).
    """
    rng = np.random.RandomState(seed)
    alpha = (1 - ci) / 2
    n = len(times)

    c_indices = []
    horizon_aucs: Dict[str, list] = {}

    for _ in range(n_bootstrap):
        idx = _stratified_bootstrap_indices(n, events, rng)
        t_b = times[idx]
        e_b = events[idx]
        r_b = predicted_risk[idx]

        if (e_b == 1).sum() == 0:  # no primary events in this bootstrap sample
            continue

        c_indices.append(compute_concordance_index(t_b, e_b, r_b))

        if time_horizons is not None:
            G_fn_b = _km_censoring_fn(t_b, e_b)
            max_t = float(t_b.max())
            for h in time_horizons:
                label = f"{int(h)}d"
                if h > max_t:
                    continue
                res = compute_ipcw_metrics_at_horizon(t_b, e_b, r_b, h, G_fn_b)
                horizon_aucs.setdefault(label, []).append(res["ipcw_auc"])

    def _ci(vals):
        vals = np.array([v for v in vals if np.isfinite(v)])
        if len(vals) == 0:
            return {"mean": float("nan"), "lower": float("nan"), "upper": float("nan")}
        return {
            "mean": float(vals.mean()),
            "lower": float(np.quantile(vals, alpha)),
            "upper": float(np.quantile(vals, 1 - alpha)),
        }

    result = {"concordance_index": _ci(c_indices)}
    for label, vals in horizon_aucs.items():
        result[f"ipcw_auc_{label}"] = _ci(vals)
    return result


# ═════════════════════════════════════════════════════════════════════
# 7. Aggregated evaluation report
# ═════════════════════════════════════════════════════════════════════


def full_evaluation(
    labels: np.ndarray,
    probabilities: np.ndarray,
    threshold: float = 0.5,
    n_bootstrap: int = 1000,
    seed: int = 42,
    times: Optional[np.ndarray] = None,
    events: Optional[np.ndarray] = None,
    survival_probabilities: Optional[np.ndarray] = None,
    time_horizons: Optional[List[float]] = None,
) -> Dict:
    """
    Run the complete evaluation suite and return all results.

    Binary metrics (AUROC, AUPRC, Brier, etc.) are computed on the
    ``labels`` / ``probabilities`` arrays as provided — these should
    already be filtered to patients with sufficient follow-up.

    Survival metrics (C-index, IPCW AUC/Brier) are computed from
    ``times`` and ``events`` if provided, using ALL patients including
    those censored before the end of the follow-up window.

    Returns a dict with keys:
        "discrimination"             : scalar binary metrics
        "calibration"                : Brier, ECE, MCE, bin_data
        "confusion_matrix"           : 2x2 ndarray
        "classification_report"      : str
        "optimal_threshold_youden"   : (threshold, metrics)
        "optimal_threshold_f1"       : (threshold, metrics)
        "decision_curve"             : DataFrame
        "high_risk_enrichment"       : DataFrame
        "bootstrap_ci"               : binary bootstrap CIs
        "threshold_sweep"            : DataFrame
        "survival"                   : survival metrics (if times/events given)
        "survival_bootstrap_ci"      : survival bootstrap CIs (if times/events given)
    """
    labels = np.asarray(labels)
    probabilities = np.asarray(probabilities, dtype=float)
    validate_binary_evaluation_inputs(labels, probabilities)

    report = {}

    report["discrimination"] = compute_discrimination_metrics(
        labels, probabilities, threshold
    )
    report["calibration"] = compute_calibration_metrics(labels, probabilities)
    report["confusion_matrix"] = get_confusion_matrix(labels, probabilities, threshold)
    report["classification_report"] = get_classification_report(
        labels, probabilities, threshold
    )
    report["optimal_threshold_youden"] = find_optimal_threshold(
        labels, probabilities, "youden"
    )
    report["optimal_threshold_f1"] = find_optimal_threshold(labels, probabilities, "f1")
    report["decision_curve"] = decision_curve_analysis(labels, probabilities)
    report["high_risk_enrichment"] = high_risk_enrichment(labels, probabilities)
    report["bootstrap_ci"] = bootstrap_metrics(
        labels, probabilities, n_bootstrap=n_bootstrap, seed=seed, threshold=threshold
    )
    report["threshold_sweep"] = compute_metrics_at_thresholds(labels, probabilities)

    if times is not None and events is not None:
        risk_scores = (
            np.asarray(survival_probabilities, dtype=float)
            if survival_probabilities is not None
            else probabilities
        )
        if not (len(times) == len(events) == len(risk_scores)):
            raise ValueError(
                "Survival evaluation requires times, events, and risk scores "
                "for the same patients; got "
                f"times={len(times)}, events={len(events)}, "
                f"risk_scores={len(risk_scores)}."
            )
        report["survival"] = compute_survival_metrics(
            times, events, risk_scores, time_horizons=time_horizons
        )
        report["survival_bootstrap_ci"] = bootstrap_survival_metrics(
            times,
            events,
            risk_scores,
            time_horizons=time_horizons,
            n_bootstrap=min(n_bootstrap, 500),  # survival CI is slower
            seed=seed,
        )

    return report


def format_evaluation_summary(report: Dict) -> str:
    """Pretty-print a summary of the evaluation report."""
    lines = []
    lines.append("=" * 70)
    lines.append("OPERA EVALUATION REPORT")
    lines.append("=" * 70)

    d = report["discrimination"]
    risk_score_only = report.get("evaluation_notes", {}).get("training_mode") == "cox"
    lines.append(f"\n── Discrimination (threshold={d['threshold']:.2f}) ──")
    lines.append(f"  AUROC:        {d['auroc']:.4f}")
    lines.append(f"  AUPRC:        {d['auprc']:.4f}")
    if not risk_score_only:
        lines.append(f"  Sensitivity:  {d['sensitivity']:.4f}")
        lines.append(f"  Specificity:  {d['specificity']:.4f}")
        lines.append(f"  PPV:          {d['ppv']:.4f}")
        lines.append(f"  NPV:          {d['npv']:.4f}")
        lines.append(f"  F1:           {d['f1']:.4f}")
        lines.append(f"  MCC:          {d['mcc']:.4f}")
        lines.append(f"  Log loss:     {d['log_loss']:.4f}")
    lines.append(
        f"  Prevalence:   {d['prevalence']:.4f} ({d['n_positive']}/{d['n_total']})"
    )

    c = report["calibration"]
    lines.append("\n── Calibration ──")
    lines.append(f"  Brier score:  {c['brier_score']:.4f}")
    lines.append(f"  ECE:          {c['ece']:.4f}")
    lines.append(f"  MCE:          {c['mce']:.4f}")
    if "calibration_intercept" in c:
        lines.append(f"  Intercept:    {c['calibration_intercept']:.4f}")
        lines.append(f"  Slope:        {c['calibration_slope']:.4f}")
    hl_p = c.get("hl_pvalue", float("nan"))
    hl_s = c.get("hl_statistic", float("nan"))
    if not (isinstance(hl_p, float) and np.isnan(hl_p)):
        calibrated = "good fit" if hl_p >= 0.05 else "POOR FIT"
        lines.append(f"  HL test:      χ²={hl_s:.2f}, p={hl_p:.4f} ({calibrated})")

    t_y, m_y = report["optimal_threshold_youden"]
    t_f, m_f = report["optimal_threshold_f1"]
    lines.append("\n── Optimal thresholds ──")
    lines.append(
        f"  Youden:  {t_y:.3f}  (Sens={m_y['sensitivity']:.3f}, Spec={m_y['specificity']:.3f})"
    )
    lines.append(f"  F1-max:  {t_f:.3f}  (F1={m_f['f1']:.3f})")

    if "high_risk_enrichment" in report and len(report["high_risk_enrichment"]) > 0:
        lines.append("\n-- High-risk enrichment --")
        for _, row in report["high_risk_enrichment"].head(3).iterrows():
            lines.append(
                f"  Top {100 * row['top_fraction']:.0f}%: "
                f"event rate={row['event_rate']:.4f}, "
                f"lift={row['enrichment']:.2f}x "
                f"({int(row['n_events'])}/{int(row['n_top'])})"
            )

    lines.append("\n── Bootstrap 95% CIs ──")
    display_order = [
        "auroc",
        "auprc",
        "sensitivity",
        "specificity",
        "ppv",
        "npv",
        "f1",
        "mcc",
        "accuracy",
        "brier_score",
        "log_loss",
    ]
    for metric in display_order:
        if metric not in report["bootstrap_ci"]:
            continue
        vals = report["bootstrap_ci"][metric]
        if np.isnan(vals["mean"]):
            continue
        lines.append(
            f"  {metric:15s}: {vals['mean']:.4f} [{vals['lower']:.4f}, {vals['upper']:.4f}]"
        )

    lines.append(f"\n── Confusion Matrix (threshold={d['threshold']:.2f}) ──")
    cm = report["confusion_matrix"]
    lines.append(f"  TN={cm[0, 0]:5d}  FP={cm[0, 1]:5d}")
    lines.append(f"  FN={cm[1, 0]:5d}  TP={cm[1, 1]:5d}")

    if "survival" in report:
        sv = report["survival"]
        sv_ci = report.get("survival_bootstrap_ci", {})
        lines.append(
            f"\n── Survival metrics (all patients, n={sv['n_total']}, events={sv['n_events']}) ──"
        )
        c_idx = sv["concordance_index"]
        c_ci = sv_ci.get("concordance_index", {})
        if c_ci:
            lines.append(
                f"  C-index:      {c_idx:.4f} [{c_ci.get('lower', float('nan')):.4f}, {c_ci.get('upper', float('nan')):.4f}]"
            )
        else:
            lines.append(f"  C-index:      {c_idx:.4f}")
        for label, hmet in sv.get("per_horizon", {}).items():
            auc_ci = sv_ci.get(f"ipcw_auc_{label}", {})
            ipcw_auc = hmet.get("ipcw_auc", float("nan"))
            ipcw_brier = hmet.get("ipcw_brier", float("nan"))
            n_cases = hmet.get("n_cases", 0)
            n_ctrl = hmet.get("n_controls", 0)
            n_excl = hmet.get("n_excluded", 0)
            if auc_ci:
                lines.append(
                    f"  IPCW-AUC@{label:>4s}: {ipcw_auc:.4f} [{auc_ci.get('lower', float('nan')):.4f}, "
                    f"{auc_ci.get('upper', float('nan')):.4f}]  "
                    f"(cases={n_cases}, controls={n_ctrl}, excluded={n_excl})"
                )
            else:
                lines.append(
                    f"  IPCW-AUC@{label:>4s}: {ipcw_auc:.4f}  IPCW-Brier: {ipcw_brier:.4f}  "
                    f"(cases={n_cases}, controls={n_ctrl}, excluded={n_excl})"
                )

    lines.append("\n" + "=" * 70)
    return "\n".join(lines)
