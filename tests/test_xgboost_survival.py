import numpy as np
import pytest

from opera.evaluation.comparison import breslow_baseline_cumulative_hazard
from opera.run.train_tabular_baselines import SURVIVAL_MODELS


def test_xgboost_cox_is_a_survival_model():
    assert "xgboost_cox" in SURVIVAL_MODELS


def test_breslow_baseline_hazard_handles_ties_and_censoring():
    event_times, cumulative = breslow_baseline_cumulative_hazard(
        times=np.array([1.0, 2.0, 2.0, 3.0]),
        events=np.array([1, 1, 0, 1]),
        log_risks=np.zeros(4),
    )
    np.testing.assert_array_equal(event_times, np.array([1.0, 2.0, 3.0]))
    # Risk-set sizes are 4, 3, and 1 respectively.
    np.testing.assert_allclose(cumulative, np.cumsum([1 / 4, 1 / 3, 1.0]))
    assert np.all(np.diff(cumulative) > 0)


def test_breslow_baseline_hazard_requires_aligned_inputs():
    with pytest.raises(ValueError, match="identical shapes"):
        breslow_baseline_cumulative_hazard(
            np.array([1.0, 2.0]), np.array([1]), np.array([0.0, 0.0])
        )
