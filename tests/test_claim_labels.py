"""Test-set claim labels and the paired bootstrap test (lock, 2026-10-21)."""

import numpy as np

from opera.evaluation.secondary_analysis import bootstrap_p_value, claim_label


def test_labels_separate_statistical_from_practical_significance():
    assert claim_label(0.008, 0.004, 0.012, 0.01) == "better"
    assert claim_label(0.015, 0.008, 0.022, 0.01) == "better, above margin"
    assert claim_label(-0.003, -0.006, -0.001, 0.01) == "worse"
    assert claim_label(0.001, -0.004, 0.006, 0.01) == "equivalent"
    assert claim_label(0.004, -0.008, 0.016, 0.01) == "inconclusive"
    assert claim_label(0.0, np.nan, np.nan, 0.01) == "not estimable"


def test_bootstrap_p_value_is_two_sided_and_bounded():
    rng = np.random.default_rng(0)
    assert bootstrap_p_value(rng.normal(0.02, 0.005, 2000)) < 0.01
    assert bootstrap_p_value(rng.normal(0.0, 0.01, 2000)) > 0.5
    assert bootstrap_p_value([np.nan, np.nan]) != bootstrap_p_value([np.nan, np.nan])
