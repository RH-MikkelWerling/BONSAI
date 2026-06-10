import numpy as np

from opera.evaluation.significance import benjamini_hochberg


def test_benjamini_hochberg_returns_monotone_adjusted_values():
    p_values = np.array([0.04, 0.001, 0.03, 0.2])

    rejected, adjusted = benjamini_hochberg(p_values, alpha=0.05)

    assert np.allclose(adjusted, [0.0533333333, 0.004, 0.0533333333, 0.2])
    assert rejected.tolist() == [False, True, False, False]
