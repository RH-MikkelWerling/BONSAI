"""
Task 7 — Pooled transfer effect with heterogeneity.

Tests that ``pooled_transfer_effect`` produces DerSimonian-Laird random-effects
estimates with tau^2, I^2, and 95% CIs from a synthetic per-cell delta table.

Scientific note (see docstring for full rationale):
- ``se_col`` or ``ci_lower_col``/``ci_upper_col`` from the paired bootstrap are
  required for meaningful tau^2 and I^2.
- The equal-weight fallback (no SE supplied) forces Q = df, so tau^2 = 0 and
  I^2 = 0 by construction.  It emits a UserWarning and sets
  ``se_source="equal_weight_fallback"`` to make this explicit.
"""

import numpy as np
import pandas as pd
import pytest

from opera.evaluation.aggregation import pooled_transfer_effect


def _make_delta_table(effects, cohorts, se=0.05, ci_half_width=None):
    """Minimal per-cell delta table with known effects and optional CI columns."""
    rows = []
    for i, (effect, cohort) in enumerate(zip(effects, cohorts)):
        row = {
            "cohort": cohort,
            "outcome": f"outcome_{i}",
            "delta_auroc_vs_baseline": effect,
            "delta_auroc_se": se,
            "model_family": "opera",
        }
        if ci_half_width is not None:
            row["ci_lower"] = effect - ci_half_width
            row["ci_upper"] = effect + ci_half_width
        rows.append(row)
    return pd.DataFrame(rows)


# ──────────────────────────────────────────────────────────────────
# se_col path: calibrated SE → meaningful tau2/I2
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
    assert overall["se_source"] == "provided_se_col"


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


def test_heterogeneous_cells_detect_nonzero_tau2_and_i2():
    """Tight SE relative to spread → tau2 > 0, I2 > 0 with calibrated SE."""
    effects = [0.01, 0.05, 0.15, 0.02, 0.12, 0.07]
    cohorts = ["c1", "c1", "c2", "c2", "c3", "c3"]
    df = _make_delta_table(effects, cohorts, se=0.01)

    result = pooled_transfer_effect(df, se_col="delta_auroc_se")

    overall = result[result["stratum"] == "_all_"].iloc[0]
    assert overall["tau2"] > 0.0, "Expected nonzero between-study variance"
    assert overall["i2"] > 0.0, "Expected nonzero I²"
    assert 0.01 <= overall["pooled_effect"] <= 0.15
    assert overall["se_source"] == "provided_se_col"


def test_ci_width_shrinks_with_more_cells():
    """More cells → narrower overall CI for a fixed common effect."""
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
# ci_lower_col / ci_upper_col path (derived from paired bootstrap)
# ──────────────────────────────────────────────────────────────────


def test_ci_derived_se_gives_same_result_as_se_col():
    """CI-derived SE (upper-lower)/2/1.96 must equal se_col when consistent."""
    se = 0.03
    ci_half_width = 1.96 * se  # exact correspondence
    effects = [0.01, 0.05, 0.12, 0.02, 0.10, 0.06]
    cohorts = ["c1", "c1", "c2", "c2", "c3", "c3"]

    df = _make_delta_table(effects, cohorts, se=se, ci_half_width=ci_half_width)

    res_se = pooled_transfer_effect(df, se_col="delta_auroc_se")
    res_ci = pooled_transfer_effect(
        df, ci_lower_col="ci_lower", ci_upper_col="ci_upper"
    )

    overall_se = res_se[res_se["stratum"] == "_all_"].iloc[0]
    overall_ci = res_ci[res_ci["stratum"] == "_all_"].iloc[0]

    assert overall_ci["pooled_effect"] == pytest.approx(
        overall_se["pooled_effect"], abs=1e-6
    )
    assert overall_ci["tau2"] == pytest.approx(overall_se["tau2"], abs=1e-6)
    assert overall_ci["i2"] == pytest.approx(overall_se["i2"], abs=1e-4)
    assert overall_ci["se_source"] == "paired_bootstrap_ci"


def test_ci_path_detects_heterogeneity_when_spread_exceeds_se():
    """CI-derived SE enables tau2 > 0 when genuine heterogeneity exceeds sampling noise."""
    se = 0.01
    ci_half_width = 1.96 * se
    effects = [0.01, 0.05, 0.15, 0.02, 0.12, 0.07]
    cohorts = ["c1", "c1", "c2", "c2", "c3", "c3"]
    df = _make_delta_table(effects, cohorts, se=se, ci_half_width=ci_half_width)

    result = pooled_transfer_effect(
        df, ci_lower_col="ci_lower", ci_upper_col="ci_upper"
    )
    overall = result[result["stratum"] == "_all_"].iloc[0]

    assert overall["tau2"] > 0.0
    assert overall["i2"] > 0.0
    assert overall["se_source"] == "paired_bootstrap_ci"


# ──────────────────────────────────────────────────────────────────
# Equal-weight fallback: warns and sets tau2=0 as an artifact
# ──────────────────────────────────────────────────────────────────


def test_equal_weight_fallback_warns_and_flags_se_source():
    """Fallback emits UserWarning and records se_source='equal_weight_fallback'."""
    effects = [0.04, 0.06, 0.08]
    cohorts = ["c1", "c2", "c3"]
    df = _make_delta_table(effects, cohorts)

    with pytest.warns(UserWarning, match="equal-weight"):
        result = pooled_transfer_effect(df)

    assert not result.empty
    overall = result[result["stratum"] == "_all_"].iloc[0]
    assert overall["se_source"] == "equal_weight_fallback"
    assert overall["pooled_effect"] == pytest.approx(np.mean(effects), abs=0.02)


def test_equal_weight_fallback_tau2_is_zero_by_construction():
    """With equal-weight fallback, Q=df exactly, so tau2=0 regardless of spread.
    This is a numerical artifact, not evidence of homogeneity."""
    # Use effects with genuine spread — tau2 should still come out zero.
    effects = [0.01, 0.10, 0.20]
    cohorts = ["c1", "c2", "c3"]
    df = _make_delta_table(effects, cohorts)

    with pytest.warns(UserWarning):
        result = pooled_transfer_effect(df)

    overall = result[result["stratum"] == "_all_"].iloc[0]
    assert overall["tau2"] == pytest.approx(0.0, abs=1e-9)
    assert overall["i2"] == pytest.approx(0.0, abs=1e-9)


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
    assert overall["se_source"] == "provided_se_col"


def test_no_stratification_produces_only_overall_row():
    effects = [0.05, 0.06, 0.07]
    cohorts = ["c1", "c2", "c3"]
    df = _make_delta_table(effects, cohorts)

    with pytest.warns(UserWarning):
        result = pooled_transfer_effect(df, stratify_col=None)

    assert list(result["stratum"]) == ["_all_"]
