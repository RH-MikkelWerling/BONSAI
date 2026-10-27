"""Training/native and paper metrics agree only for identical evaluation cells."""
import numpy as np
import pandas as pd
import pytest

from opera.evaluation.metrics import compute_competing_risk_metrics_at_horizon
from opera.evaluation.secondary_analysis import attach_ipcw_weights, metric_values


@pytest.mark.parametrize('horizon', [30, 60, 90, 180, 365, 730])
def test_native_and_analysis_ipcw_metrics_agree_with_censoring_competing_events_and_ties(horizon):
    rng = np.random.default_rng(19)
    times = np.r_[rng.integers(0, 850, 120), horizon, horizon, horizon]
    events = np.r_[rng.choice([0, 1, 2], 120), 0, 1, 2]
    risk = rng.choice([0., 0.1, 0.5, 0.9, 1.], len(times))
    frame = pd.DataFrame({'model': 'R_L', 'split': 'tuning', 'seed': 42,
                          'cohort': 'A', 'outcome': 'x', 'horizon_days': horizon,
                          'subject_id': np.arange(len(times)), 'time_days': times,
                          'event': events, 'probability': risk})
    weighted = attach_ipcw_weights(frame)
    actual = metric_values(weighted.label, weighted.probability, ipcw=weighted.ipcw_weight)
    expected = compute_competing_risk_metrics_at_horizon(times, events, risk, horizon)
    assert actual['auc'] == pytest.approx(expected['cif_auc'], abs=1e-12)
    assert actual['brier'] == pytest.approx(expected['cif_brier'], abs=1e-12)


def test_patient_pooled_auc_can_improve_without_any_within_disease_auc_gain():
    # Router distinguishes diseases with different baseline event rates, but
    # neither model distinguishes patients within either disease.
    events = np.array([1] * 9 + [0] + [1] + [0] * 9)
    times = np.where(events == 1, 10, 1000)
    joint = np.full(20, 0.5)
    routed = np.r_[np.full(10, 0.9), np.full(10, 0.1)]
    score = lambda indices, risk: compute_competing_risk_metrics_at_horizon(
        times[indices], events[indices], risk[indices], 365)['cif_auc']
    assert score(slice(None), routed) - score(slice(None), joint) == pytest.approx(0.4)
    for disease in [slice(0, 10), slice(10, 20)]:
        assert score(disease, routed) == score(disease, joint) == 0.5
