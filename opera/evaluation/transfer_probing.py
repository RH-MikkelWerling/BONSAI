"""
Cross-Outcome Transfer Probing.

After OPERA training, the embedding space is shaped by multiple outcomes
simultaneously.  This module answers: which outcomes share structure in
the learned representation?

Method
======
1. Freeze the OPERA encoder, extract embeddings for all patients.
2. For each outcome pair (A, B):
   - Train a linear probe for outcome A using patients labeled for A.
   - Evaluate on outcome B patients (zero-shot transfer).
   - Also train directly on B as a control.
3. The transfer matrix reveals latent outcome structure:
   - High transfer A→B means the embedding geometry for A contains B's signal.
   - Asymmetric transfer (A→B >> B→A) means A is more informative.

The transfer matrix is a novel scientific result: it reveals the latent
structure of hematological outcomes in EHR data, independent of any
clinical taxonomy.

This requires NO new model training — just linear probes on frozen embeddings.
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
import logging


def train_linear_probe(
    embeddings: np.ndarray,
    labels: np.ndarray,
    C: float = 1.0,
    max_iter: int = 1000,
) -> LogisticRegression:
    """Train a logistic regression probe on embeddings."""
    scaler = StandardScaler()
    X = scaler.fit_transform(embeddings)
    clf = LogisticRegression(C=C, max_iter=max_iter, solver="lbfgs")
    clf.fit(X, labels)
    # Attach scaler for inference
    clf._scaler = scaler
    return clf


def evaluate_probe(
    probe: LogisticRegression,
    embeddings: np.ndarray,
    labels: np.ndarray,
) -> Dict[str, float]:
    """Evaluate a trained probe on new data."""
    X = probe._scaler.transform(embeddings)
    probs = probe.predict_proba(X)[:, 1]

    if len(np.unique(labels)) < 2:
        return {"auroc": np.nan, "auprc": np.nan, "n": len(labels)}

    return {
        "auroc": roc_auc_score(labels, probs),
        "auprc": average_precision_score(labels, probs),
        "n": len(labels),
    }


def cross_outcome_transfer_matrix(
    embeddings: np.ndarray,
    outcome_labels: Dict[str, np.ndarray],
    n_folds: int = 5,
    seed: int = 42,
    C: float = 1.0,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
    """
    Compute the full cross-outcome transfer matrix.

    For each pair (source_outcome, target_outcome):
      - Train a probe on source_outcome's labeled patients.
      - Evaluate on target_outcome's labeled patients.

    Parameters
    ----------
    embeddings : (N, D) array
    outcome_labels : dict[outcome_name → (N,) int array]
        -1 for missing labels.
    n_folds : int
        Cross-validation folds for the diagonal (self-prediction).
    seed : int
    C : float
        Regularization for logistic regression.

    Returns
    -------
    auroc_matrix : DataFrame (source × target)
    auprc_matrix : DataFrame (source × target)
    details : dict with per-cell metadata
    """
    names = sorted(outcome_labels.keys())
    n = len(names)

    auroc_mat = np.full((n, n), np.nan)
    auprc_mat = np.full((n, n), np.nan)
    details = {}

    for i, source in enumerate(names):
        # Get source-labeled patients
        src_valid = outcome_labels[source] >= 0
        src_emb = embeddings[src_valid]
        src_lab = outcome_labels[source][src_valid]

        if len(np.unique(src_lab)) < 2:
            logging.warning(f"Skipping source {source}: < 2 classes")
            continue

        for j, target in enumerate(names):
            tgt_valid = outcome_labels[target] >= 0
            tgt_emb = embeddings[tgt_valid]
            tgt_lab = outcome_labels[target][tgt_valid]

            if len(np.unique(tgt_lab)) < 2:
                logging.warning(f"Skipping target {target}: < 2 classes")
                continue

            if i == j:
                # Diagonal: cross-validated self-prediction
                aurocs, auprcs = [], []
                skf = StratifiedKFold(n_splits=n_folds, shuffle=True,
                                       random_state=seed)
                for train_idx, test_idx in skf.split(src_emb, src_lab):
                    probe = train_linear_probe(
                        src_emb[train_idx], src_lab[train_idx], C=C
                    )
                    result = evaluate_probe(
                        probe, src_emb[test_idx], src_lab[test_idx]
                    )
                    aurocs.append(result["auroc"])
                    auprcs.append(result["auprc"])

                auroc_mat[i, j] = np.nanmean(aurocs)
                auprc_mat[i, j] = np.nanmean(auprcs)
                details[(source, target)] = {
                    "type": "self_cv",
                    "n_source": int(src_valid.sum()),
                    "auroc_std": np.nanstd(aurocs),
                }
            else:
                # Off-diagonal: train on source, evaluate on target.
                # Exclude patients who have labels for BOTH outcomes from
                # the source training set — they would leak target information.
                src_subject_ids = np.where(outcome_labels[source] >= 0)[0]
                tgt_subject_ids = np.where(outcome_labels[target] >= 0)[0]
                both = set(src_subject_ids) & set(tgt_subject_ids)

                # Source training: patients with source label but NOT target label
                source_only_mask = np.array([
                    (outcome_labels[source][i] >= 0 and i not in both)
                    for i in range(len(embeddings))
                ])
                if source_only_mask.sum() < 10 or len(np.unique(outcome_labels[source][source_only_mask])) < 2:
                    # Fall back to all source patients if too few remain
                    train_emb = src_emb
                    train_lab = src_lab
                else:
                    train_emb = embeddings[source_only_mask]
                    train_lab = outcome_labels[source][source_only_mask]

                probe = train_linear_probe(train_emb, train_lab, C=C)
                result = evaluate_probe(probe, tgt_emb, tgt_lab)

                auroc_mat[i, j] = result["auroc"]
                auprc_mat[i, j] = result["auprc"]
                details[(source, target)] = {
                    "type": "transfer",
                    "n_source": int(src_valid.sum()),
                    "n_target": int(tgt_valid.sum()),
                }

    auroc_df = pd.DataFrame(auroc_mat, index=names, columns=names)
    auprc_df = pd.DataFrame(auprc_mat, index=names, columns=names)

    return auroc_df, auprc_df, details


def compute_transfer_efficiency(
    auroc_matrix: pd.DataFrame,
) -> pd.DataFrame:
    """
    For each off-diagonal cell, compute transfer efficiency:
        efficiency(A→B) = AUROC(train on A, test on B) / AUROC(train on B, test on B)

    Values near 1.0 mean the source outcome's embedding structure almost
    fully captures the target outcome.  Values near 0.5 (random) mean
    no shared structure.

    Returns a DataFrame of the same shape.
    """
    diagonal = np.diag(auroc_matrix.values)
    # Broadcast: divide each column by its diagonal entry
    efficiency = auroc_matrix.values / diagonal[np.newaxis, :]
    np.fill_diagonal(efficiency, 1.0)
    return pd.DataFrame(efficiency, index=auroc_matrix.index,
                         columns=auroc_matrix.columns)


def format_transfer_report(
    auroc_matrix: pd.DataFrame,
    efficiency_matrix: pd.DataFrame,
) -> str:
    """Pretty-print the transfer analysis."""
    lines = []
    lines.append("=" * 70)
    lines.append("CROSS-OUTCOME TRANSFER ANALYSIS")
    lines.append("=" * 70)

    lines.append("\n── AUROC Transfer Matrix ──")
    lines.append("(rows = trained on, columns = evaluated on)")
    lines.append(auroc_matrix.round(3).to_string())

    lines.append("\n── Transfer Efficiency ──")
    lines.append("(fraction of self-prediction AUROC retained in transfer)")
    lines.append(efficiency_matrix.round(3).to_string())

    # Highlight strongest transfers
    lines.append("\n── Notable transfers ──")
    names = auroc_matrix.columns.tolist()
    for i, source in enumerate(names):
        for j, target in enumerate(names):
            if i == j:
                continue
            eff = efficiency_matrix.iloc[i, j]
            if eff > 0.9:
                lines.append(
                    f"  {source} → {target}: "
                    f"efficiency={eff:.3f} — strong shared structure"
                )
            elif eff < 0.6:
                lines.append(
                    f"  {source} → {target}: "
                    f"efficiency={eff:.3f} — weak transfer, distinct signals"
                )

    lines.append("\n" + "=" * 70)
    return "\n".join(lines)
