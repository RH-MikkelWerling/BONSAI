"""The faster restriction comparison preserves the original patient bootstrap."""
import numpy as np
import pandas as pd
import pytest

from opera.evaluation.metrics import compute_competing_risk_metrics_at_horizon
from opera.run import compare_input_restrictions as comparison


def reference_auc(frame, column, horizon):
    return compute_competing_risk_metrics_at_horizon(
        frame.time_days.to_numpy(float), frame.event.to_numpy(int),
        frame[column].to_numpy(float), horizon)["cif_auc"]


def predictions():
    rng = np.random.default_rng(42)
    n = 44
    parts = []
    for outcome in ("a", "b"):
        parts.append(pd.DataFrame(dict(
            subject_id=np.arange(n), cohort=np.where(np.arange(n) % 2, "D1", "D2"),
            outcome=outcome, time_days=rng.choice([0., 10., 30., 60., 100.], n),
            event=rng.choice([0, 1, 2], n), cif_60d=rng.choice([.1, .3, .7, .9], n))))
    return pd.concat(parts, ignore_index=True)


def test_all_restriction_estimates_and_intervals_match_original(monkeypatch):
    full = predictions()
    restricted = {"gap7": full.assign(cif_60d=full.cif_60d.to_numpy()[::-1]),
                  "gap28": full.assign(cif_60d=np.roll(full.cif_60d.to_numpy(), 7))}
    messages = []
    options = dict(horizon=60, draws=24, seed=9, domains={"a": "A", "b": "B"})
    actual = comparison.compare(full, restricted, progress=messages.append, **options)
    monkeypatch.setattr(comparison, "_auc", reference_auc)
    expected = comparison.compare(full, restricted, **options)
    for fast, slow in zip(actual, expected):
        pd.testing.assert_frame_equal(fast, slow, check_exact=False, rtol=1e-12, atol=1e-12)
    assert any("Restriction 2/2" in message for message in messages)
    assert any("outcome 2/2" in message for message in messages)
    assert messages[-1].endswith("24/24 draws complete")


@pytest.mark.parametrize("event", [0, 1, 2])
def test_auc_without_both_classes_remains_nan(event):
    frame = predictions().iloc[:4].assign(event=event, time_days=30.)
    assert np.isnan(comparison._auc(frame, "cif_60d", 60))


def test_auc_empty_resample_remains_nan():
    assert np.isnan(comparison._auc(predictions().iloc[:0], "cif_60d", 60))
