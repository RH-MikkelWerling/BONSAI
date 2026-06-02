"""IPCW utilities for OPERA horizon-specific finetuning."""

from __future__ import annotations

from typing import Dict

import numpy as np

from opera.evaluation.metrics import _km_censoring_fn


def compute_ipcw_train_weights(
    outcomes: Dict[int, dict],
    horizon_hours: float,
    eps: float = 1e-6,
) -> Dict[int, float]:
    """Compute normalized IPCW training weights for a fixed prediction horizon.

    Parameters
    ----------
    outcomes
        Output from ``bonsai.functional.outcomes.binarize_outcomes``.
        Competing events (``event == 2``) before the horizon are treated as
        censoring at their event time, not as confirmed controls.
    horizon_hours
        Prediction horizon in hours.
    eps
        Lower bound for censoring-survival probabilities.
    """
    if horizon_hours is None:
        raise ValueError("horizon_hours must be set for IPCW-BCE training.")
    if not outcomes:
        return {}

    subject_ids = list(outcomes.keys())
    times = np.array(
        [float(outcomes[sid].get("time_days", np.nan)) for sid in subject_ids],
        dtype=float,
    )
    events = np.array(
        [int(outcomes[sid].get("event", -1)) for sid in subject_ids],
        dtype=int,
    )
    valid = np.isfinite(times) & (events >= 0)
    weights = np.zeros(len(subject_ids), dtype=float)

    if valid.any():
        horizon_days = float(horizon_hours) / 24.0
        G_fn = _km_censoring_fn(times[valid], events[valid])

        for idx, is_valid in enumerate(valid):
            if not is_valid:
                continue
            time_days = float(times[idx])
            event = int(events[idx])
            if event == 1 and time_days <= horizon_days:
                weights[idx] = 1.0 / max(G_fn(time_days), eps)
            elif time_days > horizon_days:
                weights[idx] = 1.0 / max(G_fn(horizon_days), eps)
            elif event in {0, 2} and time_days <= horizon_days:
                weights[idx] = 0.0

    nonzero = weights > 0.0
    if nonzero.any():
        weights[nonzero] = weights[nonzero] / weights[nonzero].mean()

    return {
        subject_id: float(weight)
        for subject_id, weight in zip(subject_ids, weights)
    }


def attach_ipcw_weights(
    outcomes: Dict[int, dict],
    ipcw_weights: Dict[int, float],
) -> Dict[int, dict]:
    """Attach precomputed IPCW weights to outcome records in place."""
    for subject_id, outcome in outcomes.items():
        outcome["ipcw_weight"] = float(ipcw_weights.get(subject_id, 0.0))
    return outcomes
