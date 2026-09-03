import numpy as np

from opera.run.compare_abspos_diagnostics import bootstrap_mean_ci


def test_bootstrap_mean_ci_is_deterministic_and_contains_mean():
    values = np.arange(20, dtype=float)
    first = bootstrap_mean_ci(values, seed=7, n_bootstrap=200)
    second = bootstrap_mean_ci(values, seed=7, n_bootstrap=200)
    assert first == second
    assert first[0] < values.mean() < first[1]


def test_bootstrap_mean_ci_ignores_nonfinite_values():
    lower, upper = bootstrap_mean_ci(
        np.asarray([2.0, np.nan, np.inf]), seed=1, n_bootstrap=20
    )
    assert lower == upper == 2.0
