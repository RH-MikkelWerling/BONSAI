"""
RKKP-Conditioned Embedding Analysis.

Answers: does the OPERA embedding contain information beyond what
clinicians recorded in the quality registry (RKKP)?

Three levels of analysis:

  1. Added-value test: Compare RKKP-only vs RKKP+embedding models.
     The AUROC/AUPRC improvement is the foundation model's unique
     contribution beyond clinician-recorded variables.

  2. Residual embedding structure: After regressing out the RKKP-predicted
     risk, does the embedding still show outcome-related structure?
     Visualised as UMAP colored by residuals.

  3. Within-stratum separation: Among patients that RKKP considers
     equivalent (same risk stratum), does the embedding separate outcomes?
     This is the most compelling result for a clinical audience:
     "Among patients clinicians would consider equivalent, the foundation
     model identifies a subgroup with meaningfully different outcomes."
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import logging


# ═════════════════════════════════════════════════════════════════════
# 1. Added-value analysis
# ═════════════════════════════════════════════════════════════════════

def added_value_analysis(
    embeddings: np.ndarray,
    rkkp_features: np.ndarray,
    labels: np.ndarray,
    n_folds: int = 5,
    seed: int = 42,
    embedding_pca_dims: Optional[int] = 32,
) -> Dict[str, Dict[str, float]]:
    """
    Compare three models via cross-validation:
      Model A: RKKP only  (logistic regression on quality registry features)
      Model B: Embedding only  (logistic regression on OPERA embeddings)
      Model C: RKKP + Embedding  (combined)

    The difference C - A is the added value of the foundation model.
    The difference C - B is the added value of clinician-recorded data.
    A > B means RKKP captures more than the embedding (expected baseline).
    B > A means the embedding captures more than RKKP (strong result).

    Parameters
    ----------
    embeddings : (N, D) array
    rkkp_features : (N, R) array
        Tabular RKKP features. NaN values are imputed with column mean.
    labels : (N,) binary int array
    n_folds : int
    seed : int
    embedding_pca_dims : int, optional
        If set, reduce embedding dimensionality before combining with RKKP
        to avoid the curse of dimensionality in the combined model.

    Returns
    -------
    dict with keys "rkkp_only", "embedding_only", "combined", each containing
    {"auroc": float, "auroc_std": float, "auprc": float, "auprc_std": float}.
    """
    # Impute NaN in RKKP
    rkkp = rkkp_features.copy()
    col_means = np.nanmean(rkkp, axis=0)
    for col in range(rkkp.shape[1]):
        mask = np.isnan(rkkp[:, col])
        rkkp[mask, col] = col_means[col]

    # Optionally reduce embedding dims
    if embedding_pca_dims and embeddings.shape[1] > embedding_pca_dims:
        pca = PCA(n_components=embedding_pca_dims, random_state=seed)
        emb_reduced = pca.fit_transform(embeddings)
    else:
        emb_reduced = embeddings

    # Combined features
    combined = np.hstack([rkkp, emb_reduced])

    results = {}
    for name, X in [("rkkp_only", rkkp),
                     ("embedding_only", emb_reduced),
                     ("combined", combined)]:
        aurocs, auprcs = [], []
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)

        for train_idx, test_idx in skf.split(X, labels):
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X[train_idx])
            X_test = scaler.transform(X[test_idx])

            clf = LogisticRegression(C=1.0, max_iter=1000, solver="lbfgs")
            clf.fit(X_train, labels[train_idx])

            probs = clf.predict_proba(X_test)[:, 1]
            y_test = labels[test_idx]

            if len(np.unique(y_test)) == 2:
                aurocs.append(roc_auc_score(y_test, probs))
                auprcs.append(average_precision_score(y_test, probs))

        results[name] = {
            "auroc": np.mean(aurocs),
            "auroc_std": np.std(aurocs),
            "auprc": np.mean(auprcs),
            "auprc_std": np.std(auprcs),
        }

    # Compute added values
    results["added_value_embedding"] = {
        "auroc_delta": results["combined"]["auroc"] - results["rkkp_only"]["auroc"],
        "auprc_delta": results["combined"]["auprc"] - results["rkkp_only"]["auprc"],
    }
    results["added_value_rkkp"] = {
        "auroc_delta": results["combined"]["auroc"] - results["embedding_only"]["auroc"],
        "auprc_delta": results["combined"]["auprc"] - results["embedding_only"]["auprc"],
    }

    return results


def added_value_multi_outcome(
    embeddings: np.ndarray,
    rkkp_features: np.ndarray,
    outcome_labels: Dict[str, np.ndarray],
    **kwargs,
) -> pd.DataFrame:
    """
    Run added-value analysis for each outcome separately.

    Returns a DataFrame with one row per outcome and columns for
    each model's AUROC/AUPRC plus the delta values.
    """
    rows = []
    for name, labels in sorted(outcome_labels.items()):
        valid = labels >= 0
        if valid.sum() < 20 or len(np.unique(labels[valid])) < 2:
            continue

        result = added_value_analysis(
            embeddings[valid], rkkp_features[valid], labels[valid], **kwargs
        )

        row = {"outcome": name}
        for model in ["rkkp_only", "embedding_only", "combined"]:
            for metric in ["auroc", "auprc"]:
                row[f"{model}_{metric}"] = result[model][metric]
        row["delta_auroc"] = result["added_value_embedding"]["auroc_delta"]
        row["delta_auprc"] = result["added_value_embedding"]["auprc_delta"]
        row["n_valid"] = int(valid.sum())
        row["prevalence"] = float(labels[valid].mean())
        rows.append(row)

    return pd.DataFrame(rows)


# ═════════════════════════════════════════════════════════════════════
# 2. Residual analysis
# ═════════════════════════════════════════════════════════════════════

def compute_rkkp_residuals(
    rkkp_features: np.ndarray,
    labels: np.ndarray,
    n_folds: int = 5,
    seed: int = 42,
) -> np.ndarray:
    """
    Cross-validated RKKP-predicted probabilities, then compute residuals.

    residual = actual_label - rkkp_predicted_probability

    Positive residual: patient did worse than RKKP predicted.
    Negative residual: patient did better than RKKP predicted.

    Returns (N,) float array of residuals.
    """
    rkkp = rkkp_features.copy()
    col_means = np.nanmean(rkkp, axis=0)
    for col in range(rkkp.shape[1]):
        mask = np.isnan(rkkp[:, col])
        rkkp[mask, col] = col_means[col]

    predicted = np.zeros(len(labels), dtype=float)
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)

    for train_idx, test_idx in skf.split(rkkp, labels):
        scaler = StandardScaler()
        X_train = scaler.fit_transform(rkkp[train_idx])
        X_test = scaler.transform(rkkp[test_idx])

        clf = LogisticRegression(C=1.0, max_iter=1000, solver="lbfgs")
        clf.fit(X_train, labels[train_idx])
        predicted[test_idx] = clf.predict_proba(X_test)[:, 1]

    residuals = labels.astype(float) - predicted
    return residuals


# ═════════════════════════════════════════════════════════════════════
# 3. Within-stratum analysis
# ═════════════════════════════════════════════════════════════════════

def within_stratum_analysis(
    embeddings: np.ndarray,
    labels: np.ndarray,
    strata: np.ndarray,
    min_stratum_size: int = 30,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Within each RKKP-defined risk stratum, evaluate whether the embedding
    still separates outcomes.

    Parameters
    ----------
    embeddings : (N, D)
    labels : (N,) binary
    strata : (N,) int or str — risk stratum assignment per patient
        e.g. IPI score buckets, or composite risk categories.
    min_stratum_size : int
        Minimum patients per stratum to include.

    Returns
    -------
    DataFrame with columns: stratum, n, prevalence, embedding_auroc, embedding_auprc
    """
    unique_strata = sorted(set(strata))
    rows = []

    for s in unique_strata:
        mask = strata == s
        if mask.sum() < min_stratum_size:
            continue

        emb_s = embeddings[mask]
        lab_s = labels[mask]

        if len(np.unique(lab_s)) < 2:
            continue

        # Cross-validated linear probe within this stratum
        aurocs, auprcs = [], []
        n_folds = min(5, min(np.sum(lab_s == 0), np.sum(lab_s == 1)))
        if n_folds < 2:
            continue

        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
        for train_idx, test_idx in skf.split(emb_s, lab_s):
            scaler = StandardScaler()
            X_train = scaler.fit_transform(emb_s[train_idx])
            X_test = scaler.transform(emb_s[test_idx])

            clf = LogisticRegression(C=1.0, max_iter=1000, solver="lbfgs")
            clf.fit(X_train, lab_s[train_idx])
            probs = clf.predict_proba(X_test)[:, 1]

            if len(np.unique(lab_s[test_idx])) == 2:
                aurocs.append(roc_auc_score(lab_s[test_idx], probs))
                auprcs.append(average_precision_score(lab_s[test_idx], probs))

        if aurocs:
            rows.append({
                "stratum": s,
                "n": int(mask.sum()),
                "prevalence": float(lab_s.mean()),
                "embedding_auroc": np.mean(aurocs),
                "embedding_auroc_std": np.std(aurocs),
                "embedding_auprc": np.mean(auprcs),
            })

    return pd.DataFrame(rows)


# ═════════════════════════════════════════════════════════════════════
# Reporting
# ═════════════════════════════════════════════════════════════════════

def format_added_value_report(
    multi_outcome_df: pd.DataFrame,
    within_stratum_df: Optional[pd.DataFrame] = None,
) -> str:
    """Pretty-print the RKKP-conditioned analysis."""
    lines = []
    lines.append("=" * 75)
    lines.append("RKKP-CONDITIONED EMBEDDING ANALYSIS")
    lines.append("Does the foundation model know something the clinician didn't write down?")
    lines.append("=" * 75)

    lines.append("\n── Added-value analysis (AUROC) ──")
    lines.append(f"{'Outcome':25s} {'RKKP':>8s} {'Embed':>8s} {'Combined':>8s} {'Delta':>8s}")
    lines.append("-" * 65)
    for _, row in multi_outcome_df.iterrows():
        delta_str = f"+{row['delta_auroc']:.3f}" if row["delta_auroc"] >= 0 else f"{row['delta_auroc']:.3f}"
        marker = " **" if row["delta_auroc"] > 0.02 else ""
        lines.append(
            f"{row['outcome']:25s} "
            f"{row['rkkp_only_auroc']:8.3f} "
            f"{row['embedding_only_auroc']:8.3f} "
            f"{row['combined_auroc']:8.3f} "
            f"{delta_str:>8s}{marker}"
        )

    if within_stratum_df is not None and len(within_stratum_df) > 0:
        lines.append(f"\n── Within-stratum embedding discrimination ──")
        lines.append("Among patients RKKP considers equivalent, can the embedding")
        lines.append("still separate outcomes?")
        lines.append(f"\n{'Stratum':20s} {'n':>6s} {'Prev':>7s} {'AUROC':>8s}")
        lines.append("-" * 50)
        for _, row in within_stratum_df.iterrows():
            lines.append(
                f"{str(row['stratum']):20s} "
                f"{row['n']:6d} "
                f"{row['prevalence']:7.3f} "
                f"{row['embedding_auroc']:8.3f}"
            )

        mean_auroc = within_stratum_df["embedding_auroc"].mean()
        if mean_auroc > 0.6:
            lines.append(
                f"\n  Mean within-stratum AUROC: {mean_auroc:.3f}"
                f"\n  → The embedding captures prognostic signal BEYOND"
                f"\n    what clinicians recorded at the point of care."
            )
        elif mean_auroc > 0.55:
            lines.append(
                f"\n  Mean within-stratum AUROC: {mean_auroc:.3f}"
                f"\n  → Modest residual signal. The embedding adds some"
                f"\n    information but RKKP captures most of the variance."
            )
        else:
            lines.append(
                f"\n  Mean within-stratum AUROC: {mean_auroc:.3f}"
                f"\n  → Minimal residual signal. RKKP captures nearly all"
                f"\n    the prognostic information the embedding encodes."
            )

    lines.append("\n" + "=" * 75)
    return "\n".join(lines)
