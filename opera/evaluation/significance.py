"""
Significance testing for OPERA model comparisons.

Two methods are provided:

1. Paired bootstrap permutation test
   ─────────────────────────────────
   Resamples patients with replacement, computes the metric delta on each
   resample, then tests H0: Δ = 0 by shifting the bootstrap distribution
   to be centred at zero before computing the tail probability.

   Valid for any metric (AUROC, AUPRC, Brier, etc.).
   Use for all metrics except AUROC when DeLong is available.

2. DeLong's test (AUROC only)
   ──────────────────────────
   Analytic paired test for comparing two AUROC values on the same test
   set.  More powerful than bootstrap and is the standard in clinical ML
   papers.

   Reference: DeLong et al. (1988) "Comparing the areas under two or
   more correlated receiver operating characteristic curves."

Both methods return a comparable result dict:
    {
        "metric":     str,
        "model_a":    str,
        "model_b":    str,
        "delta_mean": float,   # mean(A) - mean(B)
        "delta_lower": float,  # 95% CI lower
        "delta_upper": float,  # 95% CI upper
        "p_value":    float,   # two-sided
        "significant_95": bool,
        "significant_99": bool,
        "method":     str,
    }

Sweep-level comparison
───────────────────────
``run_pairwise_comparisons`` loads saved predictions.npz files from
sweep output directories, runs all specified contrasts, applies
Benjamini-Hochberg FDR correction, and returns a DataFrame suitable
for the paper's significance table.
"""

from typing import Dict, List, Optional, Tuple
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss


# ══════════════════════════════════════════════════════════════════════════════
# 1. Paired bootstrap permutation test
# ══════════════════════════════════════════════════════════════════════════════

def paired_bootstrap_test(
    labels: np.ndarray,
    probs_a: np.ndarray,
    probs_b: np.ndarray,
    metric: str = "auroc",
    n_bootstrap: int = 2000,
    seed: int = 42,
    model_a_name: str = "model_a",
    model_b_name: str = "model_b",
) -> Dict:
    """
    Paired bootstrap permutation test for H0: Δ(metric) = 0.

    The test statistic is delta = metric(A) - metric(B) on the observed data.
    We estimate its null distribution by:
      1. Drawing B bootstrap resamples of (labels, probs_a, probs_b) together
         (paired — same patients in each resample).
      2. Computing delta_b = metric_b(A) - metric_b(B) on each resample.
      3. Shifting the bootstrap distribution to be centred at zero:
         delta_b_shifted = delta_b - mean(delta_b)
      4. p-value = proportion of |delta_b_shifted| >= |observed delta|.

    This is the "shift" correction described in:
        Davison & Hinkley (1997), "Bootstrap Methods and their Application",
        §4.4.

    Parameters
    ----------
    metric : "auroc" | "auprc" | "brier"
    """
    metric_fns = {
        "auroc":  roc_auc_score,
        "auprc":  average_precision_score,
        "brier":  lambda y, p: -brier_score_loss(y, p),  # negate so higher=better
    }
    if metric not in metric_fns:
        raise ValueError(f"metric must be one of {list(metric_fns)}, got '{metric}'")

    fn = metric_fns[metric]
    rng = np.random.RandomState(seed)
    n = len(labels)

    # Observed delta
    observed_a = fn(labels, probs_a)
    observed_b = fn(labels, probs_b)
    observed_delta = observed_a - observed_b

    # Bootstrap resamples
    boot_deltas = []
    for _ in range(n_bootstrap):
        idx = rng.randint(0, n, size=n)
        y_b = labels[idx]
        if len(np.unique(y_b)) < 2:
            continue
        da = fn(y_b, probs_a[idx])
        db = fn(y_b, probs_b[idx])
        boot_deltas.append(da - db)

    boot_deltas = np.array(boot_deltas)

    # 95% CI from bootstrap
    ci_lower = np.percentile(boot_deltas, 2.5)
    ci_upper = np.percentile(boot_deltas, 97.5)

    # Shift-corrected p-value (two-sided)
    shifted = boot_deltas - boot_deltas.mean()
    p_value = float((np.abs(shifted) >= np.abs(observed_delta)).mean())
    p_value = max(p_value, 1.0 / len(boot_deltas))  # floor at 1/B

    return {
        "metric":         metric,
        "model_a":        model_a_name,
        "model_b":        model_b_name,
        "score_a":        float(observed_a),
        "score_b":        float(observed_b),
        "delta_mean":     float(observed_delta),
        "delta_lower":    float(ci_lower),
        "delta_upper":    float(ci_upper),
        "p_value":        p_value,
        "significant_95": ci_lower > 0 or ci_upper < 0,  # CI excludes zero
        "significant_99": float(np.percentile(boot_deltas, 0.5)) > 0 or
                          float(np.percentile(boot_deltas, 99.5)) < 0,
        "method":         "paired_bootstrap",
        "n_bootstrap":    len(boot_deltas),
        "n_patients":     n,
    }


# ══════════════════════════════════════════════════════════════════════════════
# 2. DeLong's test (AUROC only)
# ══════════════════════════════════════════════════════════════════════════════

def delong_test(
    labels: np.ndarray,
    probs_a: np.ndarray,
    probs_b: np.ndarray,
    model_a_name: str = "model_a",
    model_b_name: str = "model_b",
) -> Dict:
    """
    DeLong's method for comparing two correlated AUROCs.

    Computes the variance of (AUROC_A - AUROC_B) analytically using the
    structural components of the ROC curve, then derives a z-statistic
    and two-sided p-value.

    Reference: DeLong et al. (1988), Biometrics 44(3):837-845.
    """
    from scipy import stats

    def _structural_components(labels, probs):
        """Compute V10 and V01 (structural components for DeLong variance)."""
        pos_mask = labels == 1
        neg_mask = labels == 0
        pos = probs[pos_mask]
        neg = probs[neg_mask]
        n_pos = len(pos)
        n_neg = len(neg)

        # Psi: kernel function (Mann-Whitney U kernel)
        # Psi(x, y) = 1 if x > y, 0.5 if x == y, 0 if x < y
        def psi(x, y):
            return (x > y).astype(float) + 0.5 * (x == y).astype(float)

        # V10[i] = (1/n_neg) * sum_j psi(pos[i], neg[j])  for each pos[i]
        V10 = np.array([psi(p, neg).mean() for p in pos])
        # V01[j] = (1/n_pos) * sum_i psi(pos[i], neg[j])  for each neg[j]
        V01 = np.array([psi(pos, n_).mean() for n_ in neg])

        auroc = V10.mean()
        return auroc, V10, V01, n_pos, n_neg

    auroc_a, V10_a, V01_a, n_pos, n_neg = _structural_components(labels, probs_a)
    auroc_b, V10_b, V01_b, _,     _     = _structural_components(labels, probs_b)

    # Covariance matrix of (AUROC_A, AUROC_B)
    S10 = np.cov(np.vstack([V10_a, V10_b]))  # (2, 2)
    S01 = np.cov(np.vstack([V01_a, V01_b]))  # (2, 2)

    # Variance of (AUROC_A - AUROC_B)
    # Var(AUROC_A - AUROC_B) = (S10_AA - 2*S10_AB + S10_BB)/n_pos
    #                        + (S01_AA - 2*S01_AB + S01_BB)/n_neg
    var_diff = (
        (S10[0, 0] - 2 * S10[0, 1] + S10[1, 1]) / n_pos +
        (S01[0, 0] - 2 * S01[0, 1] + S01[1, 1]) / n_neg
    )

    delta = auroc_a - auroc_b

    if var_diff <= 0:
        # Degenerate case (identical predictions)
        return {
            "metric":         "auroc",
            "model_a":        model_a_name,
            "model_b":        model_b_name,
            "score_a":        float(auroc_a),
            "score_b":        float(auroc_b),
            "delta_mean":     float(delta),
            "delta_lower":    float(delta),
            "delta_upper":    float(delta),
            "p_value":        1.0,
            "significant_95": False,
            "significant_99": False,
            "method":         "delong",
            "n_patients":     len(labels),
            "note":           "degenerate: zero variance",
        }

    se = np.sqrt(var_diff)
    z  = delta / se
    p_value = float(2 * stats.norm.sf(np.abs(z)))  # two-sided

    # 95% CI from normal approximation
    ci_lower = float(delta - 1.96 * se)
    ci_upper = float(delta + 1.96 * se)

    return {
        "metric":         "auroc",
        "model_a":        model_a_name,
        "model_b":        model_b_name,
        "score_a":        float(auroc_a),
        "score_b":        float(auroc_b),
        "delta_mean":     float(delta),
        "delta_lower":    ci_lower,
        "delta_upper":    ci_upper,
        "p_value":        p_value,
        "z_statistic":    float(z),
        "significant_95": ci_lower > 0 or ci_upper < 0,
        "significant_99": p_value < 0.01,
        "method":         "delong",
        "n_patients":     len(labels),
    }


# ══════════════════════════════════════════════════════════════════════════════
# 3. Multiple comparison correction (Benjamini-Hochberg FDR)
# ══════════════════════════════════════════════════════════════════════════════

def benjamini_hochberg(p_values: np.ndarray, alpha: float = 0.05) -> Tuple[np.ndarray, np.ndarray]:
    """
    Benjamini-Hochberg FDR correction.

    Returns
    -------
    rejected : bool array — True if test is significant after correction
    p_adjusted : float array — adjusted p-values
    """
    n = len(p_values)
    order = np.argsort(p_values)
    ranks = np.empty_like(order)
    ranks[order] = np.arange(1, n + 1)

    p_adjusted = np.minimum(1.0, p_values * n / ranks)

    # Enforce monotonicity (BH requires cumulative min from right)
    p_adjusted_monotone = np.minimum.accumulate(p_adjusted[::-1])[::-1]
    # Re-index to original order
    p_adj_out = np.empty(n)
    p_adj_out[order] = np.minimum.accumulate(
        p_values[order] * n / np.arange(1, n + 1)
    )

    rejected = p_adj_out <= alpha
    return rejected, p_adj_out


# ══════════════════════════════════════════════════════════════════════════════
# 4. Sweep-level pairwise comparison runner
# ══════════════════════════════════════════════════════════════════════════════

# Contrasts of interest — (model_a, model_b, interpretation)
DEFAULT_CONTRASTS = [
    ("opera",         "dapt",         "Contrastive adds over DAPT"),
    ("dapt",          "base_pretrain","DAPT adds over base pretrain"),
    ("opera",         "tabular_ehr",  "OPERA vs tabular EHR-only"),
    ("opera",         "tabular_rkkp", "OPERA vs tabular ceiling"),
    ("opera_joint",   "opera",        "Joint training adds over per-cohort"),
    ("opera_joint",   "tabular_rkkp", "Joint OPERA vs tabular ceiling"),
]


def load_predictions(predictions_dir: Path) -> Optional[Dict[str, np.ndarray]]:
    """
    Load predictions.npz from an evaluate output directory.

    Returns all available arrays. Survival fields (times, events, binary_mask)
    are included when present — they are written by evaluate.py for prospective
    validation cohorts.
    """
    p = predictions_dir / "predictions.npz"
    if not p.exists():
        return None
    data = np.load(p)
    out = {
        "labels":        data["labels"],
        "probabilities": data["probabilities"],
        "subject_ids":   data["subject_ids"],
    }
    # Survival fields (may not exist in older prediction files)
    for field in ("times", "events", "binary_mask"):
        if field in data:
            out[field] = data[field]
    return out


# ══════════════════════════════════════════════════════════════════════════════
# 4b. Concordance-index bootstrap test (survival metric)
# ══════════════════════════════════════════════════════════════════════════════

def _concordance_index(times: np.ndarray, events: np.ndarray, risk: np.ndarray) -> float:
    """
    Harrell's concordance index (C-statistic) for survival data.

    Computed over all comparable pairs (i, j) where event[i] == 1 and
    times[i] < times[j].  Ties in risk score contribute 0.5.

    Falls back to lifelines.utils.concordance_index if available (faster
    for large N).
    """
    try:
        from lifelines.utils import concordance_index as _ci
        return float(_ci(times, -risk, events))  # lifelines: lower risk → longer survival
    except ImportError:
        pass

    # Pure-numpy fallback (O(n²) — fine for test sets up to ~5 k patients)
    n = len(times)
    concordant = 0.0
    tied       = 0.0
    comparable = 0.0
    for i in range(n):
        if events[i] != 1:
            continue
        for j in range(n):
            if i == j:
                continue
            if times[j] < times[i]:
                continue  # j must survive longer
            comparable += 1
            if risk[i] > risk[j]:
                concordant += 1
            elif risk[i] == risk[j]:
                tied += 0.5
    if comparable == 0:
        return float("nan")
    return (concordant + tied) / comparable


def concordance_bootstrap_test(
    times_a: np.ndarray,
    events_a: np.ndarray,
    risk_a: np.ndarray,
    risk_b: np.ndarray,
    n_bootstrap: int = 2000,
    seed: int = 42,
    model_a_name: str = "model_a",
    model_b_name: str = "model_b",
) -> Dict:
    """
    Paired bootstrap test for H0: C-index(A) = C-index(B).

    Both models must be evaluated on the same patients (same times/events).
    Uses the shift-corrected bootstrap (Davison & Hinkley §4.4).
    """
    rng = np.random.RandomState(seed)
    n   = len(times_a)

    observed_a = _concordance_index(times_a, events_a, risk_a)
    observed_b = _concordance_index(times_a, events_a, risk_b)
    observed_delta = observed_a - observed_b

    boot_deltas = []
    for _ in range(n_bootstrap):
        idx = rng.randint(0, n, size=n)
        t_b = times_a[idx]
        e_b = events_a[idx]
        if (e_b == 1).sum() < 2:  # need at least 2 primary events for concordance
            continue
        da = _concordance_index(t_b, e_b, risk_a[idx])
        db = _concordance_index(t_b, e_b, risk_b[idx])
        if np.isnan(da) or np.isnan(db):
            continue
        boot_deltas.append(da - db)

    boot_deltas = np.array(boot_deltas)
    if len(boot_deltas) < 10:
        return {
            "metric": "concordance_index", "model_a": model_a_name,
            "model_b": model_b_name, "score_a": float(observed_a),
            "score_b": float(observed_b), "delta_mean": float(observed_delta),
            "delta_lower": float("nan"), "delta_upper": float("nan"),
            "p_value": float("nan"), "significant_95": False,
            "significant_99": False, "method": "concordance_bootstrap",
            "n_bootstrap": len(boot_deltas), "n_patients": n,
            "note": "too few valid bootstrap resamples",
        }

    ci_lower = float(np.percentile(boot_deltas, 2.5))
    ci_upper = float(np.percentile(boot_deltas, 97.5))
    shifted  = boot_deltas - boot_deltas.mean()
    p_value  = float((np.abs(shifted) >= np.abs(observed_delta)).mean())
    p_value  = max(p_value, 1.0 / len(boot_deltas))

    return {
        "metric":         "concordance_index",
        "model_a":        model_a_name,
        "model_b":        model_b_name,
        "score_a":        float(observed_a),
        "score_b":        float(observed_b),
        "delta_mean":     float(observed_delta),
        "delta_lower":    ci_lower,
        "delta_upper":    ci_upper,
        "p_value":        p_value,
        "significant_95": ci_lower > 0 or ci_upper < 0,
        "significant_99": float(np.percentile(boot_deltas, 0.5)) > 0 or
                          float(np.percentile(boot_deltas, 99.5)) < 0,
        "method":         "concordance_bootstrap",
        "n_bootstrap":    len(boot_deltas),
        "n_patients":     n,
    }


def run_pairwise_comparisons(
    sweep_output_dir: str,
    contrasts: Optional[List[Tuple[str, str, str]]] = None,
    metrics: Optional[List[str]] = None,
    alpha: float = 0.05,
    n_bootstrap: int = 2000,
    use_delong_for_auroc: bool = True,
) -> pd.DataFrame:
    """
    Run all pairwise significance comparisons across the sweep output.

    Loads predictions.npz for each (cohort, outcome, variant) cell,
    runs the specified contrasts, applies BH FDR correction across all
    tests, and returns a DataFrame.

    Parameters
    ----------
    sweep_output_dir : str
        Root directory written by sweep.py (contains cohort/outcome/variant/ subdirs).
    contrasts : list of (model_a, model_b, label) tuples.
        Defaults to DEFAULT_CONTRASTS.
    metrics : list of metric names to test.
        Defaults to ["auroc"]. Add "auprc", "brier" for secondary metrics.
        Add "concordance_index" to test Harrell's C-statistic — requires
        times and events to be present in predictions.npz (written by
        evaluate.py for prospective validation cohorts).
    alpha : FDR threshold.
    use_delong_for_auroc : bool
        Use DeLong's test for AUROC (more powerful than bootstrap).

    Returns
    -------
    DataFrame with columns:
        cohort, outcome, model_a, model_b, contrast_label,
        metric, score_a, score_b, delta_mean, delta_lower, delta_upper,
        p_value, p_adjusted, significant, method
    """
    if contrasts is None:
        contrasts = DEFAULT_CONTRASTS
    if metrics is None:
        metrics = ["auroc"]

    sweep_root = Path(sweep_output_dir)
    rows = []

    # Discover all (cohort, outcome) cells
    cohort_dirs = [d for d in sweep_root.iterdir() if d.is_dir()]
    for cohort_dir in sorted(cohort_dirs):
        cohort = cohort_dir.name
        outcome_dirs = [d for d in cohort_dir.iterdir() if d.is_dir()]
        for outcome_dir in sorted(outcome_dirs):
            outcome = outcome_dir.name

            # Load predictions for all available variants in this cell
            variant_preds: Dict[str, Dict] = {}
            for variant_dir in sorted(outcome_dir.iterdir()):
                if not variant_dir.is_dir():
                    continue
                preds = load_predictions(variant_dir)
                if preds is not None:
                    variant_preds[variant_dir.name] = preds

            if len(variant_preds) < 2:
                continue

            # Run specified contrasts
            for model_a, model_b, contrast_label in contrasts:
                if model_a not in variant_preds or model_b not in variant_preds:
                    continue

                pa = variant_preds[model_a]
                pb = variant_preds[model_b]

                # Align on shared subject IDs (test sets must overlap)
                ids_a = set(pa["subject_ids"])
                ids_b = set(pb["subject_ids"])
                shared = np.array(sorted(ids_a & ids_b))
                if len(shared) < 20:
                    continue

                def _filter(p, ids):
                    id_mask = np.isin(p["subject_ids"], ids)
                    # Apply binary_mask (full-follow-up only) when available,
                    # so binary metrics (AUROC, AUPRC, Brier) exclude
                    # censored patients who lack full follow-up.
                    if "binary_mask" in p:
                        id_mask = id_mask & p["binary_mask"].astype(bool)
                    order = np.argsort(p["subject_ids"][id_mask])
                    return (p["labels"][id_mask][order],
                            p["probabilities"][id_mask][order])

                labels_a, probs_a = _filter(pa, shared)
                labels_b, probs_b = _filter(pb, shared)

                # Sanity: labels must be identical (same test set)
                if not np.array_equal(labels_a, labels_b):
                    import warnings
                    warnings.warn(
                        f"Labels differ for {cohort}/{outcome} "
                        f"{model_a} vs {model_b} — skipping"
                    )
                    continue

                # Survival fields — needed only for concordance_index
                def _filter_survival(p, ids):
                    mask  = np.isin(p["subject_ids"], ids)
                    order = np.argsort(p["subject_ids"][mask])
                    times = p["times"][mask][order]  if "times"  in p else None
                    events= p["events"][mask][order] if "events" in p else None
                    risk  = p["probabilities"][mask][order]
                    return times, events, risk

                for metric in metrics:
                    try:
                        if metric == "concordance_index":
                            t_a, e_a, r_a = _filter_survival(pa, shared)
                            _,   _,   r_b = _filter_survival(pb, shared)
                            if t_a is None or e_a is None:
                                import warnings
                                warnings.warn(
                                    f"No survival data for {cohort}/{outcome} "
                                    f"— skipping concordance_index"
                                )
                                continue
                            result = concordance_bootstrap_test(
                                t_a, e_a, r_a, r_b,
                                n_bootstrap=n_bootstrap,
                                model_a_name=model_a,
                                model_b_name=model_b,
                            )
                        elif metric == "auroc" and use_delong_for_auroc:
                            result = delong_test(
                                labels_a, probs_a, probs_b,
                                model_a_name=model_a,
                                model_b_name=model_b,
                            )
                        else:
                            result = paired_bootstrap_test(
                                labels_a, probs_a, probs_b,
                                metric=metric,
                                n_bootstrap=n_bootstrap,
                                model_a_name=model_a,
                                model_b_name=model_b,
                            )
                        rows.append({
                            "cohort":          cohort,
                            "outcome":         outcome,
                            "model_a":         model_a,
                            "model_b":         model_b,
                            "contrast_label":  contrast_label,
                            "metric":          metric,
                            "score_a":         result["score_a"],
                            "score_b":         result["score_b"],
                            "delta_mean":      result["delta_mean"],
                            "delta_lower":     result["delta_lower"],
                            "delta_upper":     result["delta_upper"],
                            "p_value":         result["p_value"],
                            "method":          result["method"],
                            "n_patients":      result.get("n_patients", len(labels_a)),
                        })
                    except Exception as e:
                        import warnings
                        warnings.warn(
                            f"Test failed for {cohort}/{outcome} "
                            f"{model_a} vs {model_b} [{metric}]: {e}"
                        )

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)

    # BH correction across all tests
    _, df["p_adjusted"] = benjamini_hochberg(df["p_value"].values, alpha=alpha)
    df["significant"] = df["p_adjusted"] <= alpha

    # Convenience: significance stars
    def _stars(p):
        if p < 0.001: return "***"
        if p < 0.01:  return "**"
        if p < 0.05:  return "*"
        return ""
    df["stars"]         = df["p_value"].apply(_stars)
    df["stars_adjusted"] = df["p_adjusted"].apply(_stars)

    return df.sort_values(["cohort", "outcome", "contrast_label", "metric"])


def format_significance_table(
    df: pd.DataFrame,
    metric: str = "auroc",
    contrast_filter: Optional[str] = None,
) -> str:
    """
    Pretty-print a significance summary suitable for supplementary tables.
    """
    sub = df[df["metric"] == metric].copy()
    if contrast_filter:
        sub = sub[sub["contrast_label"] == contrast_filter]

    lines = ["=" * 80, f"Significance table — {metric.upper()}", "=" * 80]
    lines.append(f"{'Cohort':<10} {'Outcome':<25} {'Contrast':<35} "
                 f"{'Δ':>8} {'95% CI':>18} {'p':>8} {'p_adj':>8} {'sig':>4}")
    lines.append("-" * 80)

    for _, r in sub.iterrows():
        ci = f"[{r['delta_lower']:+.3f}, {r['delta_upper']:+.3f}]"
        lines.append(
            f"{r['cohort']:<10} {r['outcome']:<25} {r['contrast_label']:<35} "
            f"{r['delta_mean']:+8.3f} {ci:>18} "
            f"{r['p_value']:8.4f} {r['p_adjusted']:8.4f} {r['stars_adjusted']:>4}"
        )

    lines.append("=" * 80)
    lines.append("* p<0.05  ** p<0.01  *** p<0.001  (Benjamini-Hochberg FDR corrected)")
    return "\n".join(lines)


def to_latex_significance_table(
    df: pd.DataFrame,
    metric: str = "auroc",
) -> str:
    """
    Generate a LaTeX significance table for the paper.
    One row per (cohort, outcome, contrast). Bold if significant after FDR.
    """
    sub = df[df["metric"] == metric].copy()

    lines = [
        "\\begin{table}[ht]",
        "\\centering",
        "\\caption{Pairwise significance tests — " + metric.upper() +
        " (DeLong's test; Benjamini-Hochberg FDR correction)}",
        "\\begin{tabular}{llllrrrl}",
        "\\toprule",
        "Cohort & Outcome & Model A & Model B & $\\Delta$ & 95\\% CI & $p$ & $p_{\\text{adj}}$ \\\\",
        "\\midrule",
    ]

    for _, r in sub.iterrows():
        ci = f"[{r['delta_lower']:+.3f}, {r['delta_upper']:+.3f}]"
        delta_str = f"{r['delta_mean']:+.3f}"
        if r["significant"]:
            delta_str = f"\\textbf{{{delta_str}}}"
        p_str     = f"{r['p_value']:.4f}"
        padj_str  = f"{r['p_adjusted']:.4f}{r['stars_adjusted']}"

        lines.append(
            f"{r['cohort'].upper()} & "
            f"{r['outcome'].replace('_', '\\_')} & "
            f"{r['model_a'].replace('_', '\\_')} & "
            f"{r['model_b'].replace('_', '\\_')} & "
            f"{delta_str} & {ci} & {p_str} & {padj_str} \\\\"
        )

    lines += [
        "\\bottomrule",
        "\\multicolumn{8}{l}{\\small $^{*}p<0.05$, $^{**}p<0.01$, "
        "$^{***}p<0.001$ after FDR correction. Bold = significant.}",
        "\\end{tabular}",
        "\\end{table}",
    ]
    return "\n".join(lines)
