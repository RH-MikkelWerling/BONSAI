"""IPCW utilities for OPERA horizon-specific finetuning."""

from __future__ import annotations

from typing import Dict

import numpy as np

from opera.evaluation.metrics import _km_admin_censoring_fn, _km_censoring_fn


def compute_ipcw_train_weights(
    outcomes: Dict[int, dict],
    horizon_hours: float,
    estimand: str = "net_risk",
    eps: float = 1e-6,
) -> Dict[int, float]:
    """Compute normalized IPCW training weights for a fixed prediction horizon.

    Parameters
    ----------
    outcomes
        Output from ``bonsai.functional.outcomes.binarize_outcomes``.
        Under ``net_risk``, competing events before the horizon are treated as
        censoring. Under ``cumulative_incidence``, they are observed controls.
    horizon_hours
        Prediction horizon in hours.
    eps
        Lower bound for censoring-survival probabilities.
    """
    if horizon_hours is None:
        raise ValueError("horizon_hours must be set for IPCW-BCE training.")
    if estimand not in {"net_risk", "cumulative_incidence"}:
        raise ValueError("estimand must be 'net_risk' or 'cumulative_incidence'.")
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
        G_fn = (
            _km_admin_censoring_fn(times[valid], events[valid])
            if estimand == "cumulative_incidence"
            else _km_censoring_fn(times[valid], events[valid])
        )

        for idx, is_valid in enumerate(valid):
            if not is_valid:
                continue
            time_days = float(times[idx])
            event = int(events[idx])
            if event == 1 and time_days <= horizon_days:
                weights[idx] = 1.0 / max(G_fn(time_days), eps)
            elif event == 2 and time_days <= horizon_days:
                if estimand == "cumulative_incidence":
                    weights[idx] = 1.0 / max(G_fn(time_days), eps)
            elif time_days >= horizon_days:
                weights[idx] = 1.0 / max(G_fn(horizon_days), eps)
            elif event in {0, 2} and time_days < horizon_days:
                weights[idx] = 0.0

    # A single global rescaling preserves the IPCW empirical-risk estimand and
    # gives weights mean one over the complete cohort. Do not self-normalize
    # only observed subjects: zero-weight censored subjects are part of the
    # empirical-risk denominator.
    mean_weight = weights.mean()
    if mean_weight > 0.0:
        weights = weights / mean_weight

    return {
        subject_id: float(weight) for subject_id, weight in zip(subject_ids, weights)
    }


def summarize_ipcw_weights(
    outcomes: Dict[int, dict],
    weights: Dict[int, float],
) -> Dict[str, float]:
    """Return support and stability diagnostics for one IPCW split."""
    subject_ids = list(outcomes)
    values = np.asarray([float(weights.get(sid, 0.0)) for sid in subject_ids])
    labels = np.asarray(
        [int(outcomes[sid].get("label", 0)) for sid in subject_ids],
        dtype=int,
    )
    observed = values > 0.0
    total_weight = float(values.sum())
    squared_weight = float(np.square(values).sum())
    effective_n = total_weight**2 / squared_weight if squared_weight > 0.0 else 0.0
    return {
        "n_total": int(len(values)),
        "n_nonzero": int(observed.sum()),
        "n_zero": int((~observed).sum()),
        "n_cases_nonzero": int(((labels == 1) & observed).sum()),
        "n_controls_nonzero": int(((labels == 0) & observed).sum()),
        "mean_weight": float(values.mean()) if len(values) else float("nan"),
        "max_weight": float(values.max()) if len(values) else float("nan"),
        "p99_weight": (
            float(np.quantile(values, 0.99)) if len(values) else float("nan")
        ),
        "effective_sample_size": float(effective_n),
        "effective_sample_fraction": (
            float(effective_n / len(values)) if len(values) else 0.0
        ),
    }


def attach_ipcw_weights(
    outcomes: Dict[int, dict],
    ipcw_weights: Dict[int, float],
) -> Dict[int, dict]:
    """Attach precomputed IPCW weights to outcome records in place."""
    for subject_id, outcome in outcomes.items():
        outcome["ipcw_weight"] = float(ipcw_weights.get(subject_id, 0.0))
    return outcomes
