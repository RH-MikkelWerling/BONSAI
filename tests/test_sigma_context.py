import numpy as np
import pandas as pd

from opera.evaluation.sigma_context import (
    build_sigma_context_table,
    residualize_log_sigma,
)


def test_build_sigma_context_table_adds_effective_pair_descriptors():
    sigmas = {"mortality": 0.5, "aki": 1.5}
    metadata = pd.DataFrame(
        {
            "outcome": ["mortality", "aki"],
            "n_events": [100, 10],
            "n_total": [1000, 1000],
            "n_censored": [200, 500],
            "n_effective_pairs": [10000.0, 200.0],
        }
    )

    result = build_sigma_context_table(sigmas, metadata)

    assert "precision" in result
    assert "sigma_per_sqrt_effective_pair" in result
    assert "prevalence" in result
    assert result.loc[result["outcome"] == "mortality", "prevalence"].iloc[0] == 0.1


def test_residualize_log_sigma_returns_finite_residuals_when_identifiable():
    frame = pd.DataFrame(
        {
            "outcome": ["a", "b", "c", "d", "e"],
            "sigma": [0.5, 0.7, 1.2, 1.8, 2.0],
            "n_effective_pairs": [1000.0, 800.0, 400.0, 100.0, 50.0],
            "n_events": [80, 60, 30, 10, 5],
            "n_total": [1000, 1000, 1000, 1000, 1000],
        }
    )

    context = build_sigma_context_table(frame)
    result = residualize_log_sigma(context)

    assert "log_sigma_residual" in result
    assert np.isfinite(result["log_sigma_residual"]).all()
