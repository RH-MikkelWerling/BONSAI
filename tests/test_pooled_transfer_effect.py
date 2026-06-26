"""
Task 7 — Pooled transfer effect with heterogeneity.

Tests that ``pooled_transfer_effect`` produces DerSimonian-Laird random-effects
estimates with tau^2, I^2, and 95% CIs from a synthetic per-cell delta table.
"""

import numpy as np
import pandas as pd
import pytest

from opera.evaluation.aggregation import pooled_transfer_effect


def _make_delta_table(effects, cohorts, se=0.05):
    """Minimal per-cell delta table with known effects."""
    rows = [
        {
            "cohort": cohort,
            "outcome": f"outcome_{i}",
            "delta_auroc_vs_baseline": effect,
            "delta_auroc_se": se,
            "model_family": "opera",
        }
        for i, (effect, cohort) in enumerate(zip(effects, cohorts))
    ]
    return pd.DataFrame(rows)


# ──────────────────────────────────────────────────────────────────
# Homogeneous case: all cells equal → tau2 = 0, I2 = 0, pooled ≈ value
# ──────────────────────────────────────────────────────────────────

def test_homogeneous_cells_give_zero_heterogeneity():
    effects = [0.05] * 6
    cohorts = ["c1", "c1", "c2", "c2", "c3", "c3"]
    df = _make_delta_table(effects, cohorts)

    result = pooled_transfer_effect(df, se_col="delta_auroc_se")

    assert not result.empty
    overall = result[result["stratum"] == "_all_"].iloc[0]
    assert overall["pooled_effect"] == pytest.approx(0.05, abs=1e-6)
    assert overall["tau2"] == pytest.approx(0.0, abs=1e-6)
    assert overall["i2"] == pytest.approx(0.0, abs=1e-3)
    assert overall["k_cells"] == 6


def test_per_stratum_rows_produced_alongside_overall():
    effects = [0.05, 0.06, 0.07, 0.08]
    cohorts = ["c1", "c1", "c2", "c2"]
    df = _make_delta_table(effects, cohorts)

    result = pooled_transfer_effect(df, se_col="delta_auroc_se")

    strata = set(result["stratum"])
    assert "_all_" in strata
    assert "c1" in strata
    assert "c2" in strata

    c1_row = result[result["stratum"] == "c1"].iloc[0]
    assert c1_row["k_cells"] == 2
    assert c1_row["pooled_effect"] == pytest.approx(0.055, abs=1e-4)


# ──────────────────────────────────────────────────────────────────
# Heterogeneous case: spread effects → tau2 > 0, I2 > 0
# ──────────────────────────────────────────────────────────────────

def test_heterogeneous_cells_detect_nonzero_tau2_and_i2():
    # Tight SE so individual cell variance is small relative to spread.
    effects = [0.01, 0.05, 0.15, 0.02, 0.12, 0.07]
    cohorts = ["c1", "c1", "c2", "c2", "c3", "c3"]
    df = _make_delta_table(effects, cohorts, se=0.01)

    result = pooled_transfer_effect(df, se_col="delta_auroc_se")

    overall = result[result["stratum"] == "_all_"].iloc[0]
    assert overall["tau2"] > 0.0, "Expected nonzero between-study variance"
    assert overall["i2"] > 0.0, "Expected nonzero I²"
    # Pooled estimate lies inside the range of effects.
    assert 0.01 <= overall["pooled_effect"] <= 0.15


def test_ci_width_shrinks_with_more_cells():
    """More cells → narrower CI for a fixed common effect."""
    few = _make_delta_table([0.05] * 3, ["c1", "c2", "c3"])
    many = _make_delta_table([0.05] * 12, ["c1"] * 4 + ["c2"] * 4 + ["c3"] * 4)

    res_few = pooled_transfer_effect(few, se_col="delta_auroc_se")
    res_many = pooled_transfer_effect(many, se_col="delta_auroc_se")

    width_few = (
        res_few.loc[res_few["stratum"] == "_all_", "ci_upper"].iloc[0]
        - res_few.loc[res_few["stratum"] == "_all_", "ci_lower"].iloc[0]
    )
    width_many = (
        res_many.loc[res_many["stratum"] == "_all_", "ci_upper"].iloc[0]
        - res_many.loc[res_many["stratum"] == "_all_", "ci_lower"].iloc[0]
    )
    assert width_many < width_few


# ──────────────────────────────────────────────────────────────────
# Edge cases
# ──────────────────────────────────────────────────────────────────

def test_empty_table_returns_empty():
    result = pooled_transfer_effect(pd.DataFrame())
    assert result.empty


def test_missing_delta_col_returns_empty():
    df = pd.DataFrame({"cohort": ["c1"], "other_col": [0.05]})
    result = pooled_transfer_effect(df)
    assert result.empty


def test_single_cell_returns_cell_value_with_zero_heterogeneity():
    df = _make_delta_table([0.08], ["c1"])
    result = pooled_transfer_effect(df, se_col="delta_auroc_se")

    overall = result[result["stratum"] == "_all_"].iloc[0]
    assert overall["pooled_effect"] == pytest.approx(0.08, abs=1e-6)
    assert overall["tau2"] == pytest.approx(0.0)
    assert overall["i2"] == pytest.approx(0.0)
    assert overall["k_cells"] == 1


def test_equal_weight_fallback_when_no_se_col():
    """Without se_col, equal-weight pooling falls back to empirical SD."""
    effects = [0.04, 0.06, 0.08]
    cohorts = ["c1", "c2", "c3"]
    df = _make_delta_table(effects, cohorts)
    # Deliberately omit se_col
    result = pooled_transfer_effect(df)

    assert not result.empty
    overall = result[result["stratum"] == "_all_"].iloc[0]
    # Expected pooled effect ≈ mean of effects under equal weighting.
    assert overall["pooled_effect"] == pytest.approx(np.mean(effects), abs=0.02)


def test_no_stratification_produces_only_overall_row():
    effects = [0.05, 0.06, 0.07]
    cohorts = ["c1", "c2", "c3"]
    df = _make_delta_table(effects, cohorts)

    result = pooled_transfer_effect(df, stratify_col=None)

    assert list(result["stratum"]) == ["_all_"]
