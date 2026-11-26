"""Descriptive test ROC sensitivity at common specificity; not a fixed cutoff.

Linear interpolation describes random allocation at a tied boundary. Recompute
the ROC within each paired patient-bootstrap draw, including its boundary.
"""
import numpy as np


def sensitivities(probability, label, weights, specificities=(.8, .9, .95), batch_size=32):
    probability, label = np.asarray(probability, float), np.asarray(label, int)
    weights = np.asarray(weights, float)
    if weights.ndim == 1:
        weights = weights[None, :]
    if probability.shape != label.shape or weights.shape[1] != len(label):
        raise ValueError('ROC inputs have different lengths')
    if not len(label) or not np.isfinite(probability).all() or not np.isin(label, [0,1]).all():
        raise ValueError('Need finite probabilities and binary observable labels')
    if not np.isfinite(weights).all() or (weights < 0).any():
        raise ValueError('Invalid ROC weights')
    if any(not 0 < s < 1 for s in specificities):
        raise ValueError('Specificity targets must lie between zero and one')
    order = np.argsort(-probability, kind='stable')
    scores, labels = probability[order], label[order]
    ends = np.r_[np.flatnonzero(scores[:-1] != scores[1:]), len(scores)-1]
    result = np.full((len(weights), len(specificities)), np.nan)
    for start in range(0, len(weights), batch_size):
        w = weights[start:start+batch_size, order]
        tp = np.c_[np.zeros(len(w)), np.cumsum(w*labels, axis=1)[:,ends]]
        fp = np.c_[np.zeros(len(w)), np.cumsum(w*(1-labels), axis=1)[:,ends]]
        for j, (cases, controls) in enumerate(zip(tp, fp)):
            if cases[-1] <= 0 or controls[-1] <= 0:
                continue
            tpr, fpr = cases/cases[-1], controls/controls[-1]
            for k, specificity in enumerate(specificities):
                target = 1-specificity
                hi = int(np.searchsorted(fpr, target, side='right'))
                lo = max(0, hi-1)
                # Keep vertical segments: deduplicating FPR by its maximum TPR
                # would shortcut intervening horizontal ROC segments.
                if hi == len(fpr) or fpr[lo] == target:
                    result[start+j,k] = tpr[lo]
                else:
                    fraction = (target-fpr[lo])/(fpr[hi]-fpr[lo])
                    result[start+j,k] = tpr[lo] + fraction*(tpr[hi]-tpr[lo])
    return result
