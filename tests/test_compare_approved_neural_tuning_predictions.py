import numpy as np
import pandas as pd
import pytest

from opera.evaluation.metrics import compute_competing_risk_metrics_at_horizon
from opera.run.compare_approved_neural_tuning_predictions import (
    _registry_outcomes,
    compare_predictions,
    fast_cif_auc,
)


def test_approved_registry_list_and_mapping_rosters(tmp_path):
    path = tmp_path / "registry.yaml"
    path.write_text("outcomes:\n  - overall_survival\n  - sepsis\n", encoding="utf-8")
    assert _registry_outcomes(path) == ["overall_survival", "sepsis"]
    path.write_text("outcomes:\n  sepsis: {}\n  overall_survival: {}\n", encoding="utf-8")
    assert _registry_outcomes(path) == ["sepsis", "overall_survival"]


def test_fast_cif_auc_matches_reference_with_censoring_and_ties():
    rng = np.random.default_rng(12)
    times = rng.integers(5, 200, 120).astype(float)
    events = rng.choice([0, 1, 2], 120, p=[.25, .5, .25])
    risk = rng.choice([.1, .2, .5, .8], 120)
    fast, cases, controls = fast_cif_auc(times, events, risk, 90)
    slow = compute_competing_risk_metrics_at_horizon(times, events, risk, 90)
    assert fast == pytest.approx(slow["cif_auc"])
    assert cases == slow["n_cases"]
    assert controls == slow["n_controls"]


def _frame(risk):
    n = len(risk)
    return pd.DataFrame({
        "subject_id": np.arange(n), "cohort": "D1", "split": "tuning",
        "outcome": "x", "time_days": np.where(np.arange(n) % 2, 20., 100.),
        "event": np.where(np.arange(n) % 2, 1, 0), "cif_60d": risk,
    })


def test_paired_bootstrap_uses_same_patients_and_rejects_label_mismatch():
    n = 80
    baseline = _frame(np.linspace(.2, .8, n))
    better = _frame(np.where(np.arange(n) % 2, .9, .1))
    estimates, contrasts, report = compare_predictions(
        {"base": baseline, "new": better}, baseline="base", outcomes=["x"],
        horizon=60, n_bootstrap=60, seed=3, family_by_outcome={"x": "family"},
    )
    assert len(estimates) == 2
    assert len(contrasts) == 1
    assert contrasts.iloc[0]["estimate"] > 0
    assert report["inventory"]["x"]["n_valid_bootstrap"] == 60
    mismatched = better.copy()
    mismatched.loc[0, "event"] = 1
    with pytest.raises(ValueError, match="labels differ"):
        compare_predictions(
            {"base": baseline, "new": mismatched}, baseline="base", outcomes=["x"],
            horizon=60, n_bootstrap=60, seed=3, family_by_outcome={"x": "family"},
        )
    with pytest.raises(ValueError, match="Identical patient-level predictions"):
        compare_predictions(
            {"base": baseline, "duplicate": baseline.copy()}, baseline="base",
            outcomes=["x"], horizon=60, n_bootstrap=60, seed=3,
            family_by_outcome={"x": "family"},
        )


def test_all_outcome_mode_records_sparse_exclusions():
    n = 80
    baseline = _frame(np.linspace(.2, .8, n))
    better = _frame(np.where(np.arange(n) % 2, .9, .1))
    sparse_base = baseline.copy()
    sparse_new = better.copy()
    for frame in (sparse_base, sparse_new):
        frame["outcome"] = "sparse"
        frame["event"] = 0
        frame.loc[1, "event"] = 1
    estimates, contrasts, report = compare_predictions(
        {"base": pd.concat([baseline, sparse_base], ignore_index=True),
         "new": pd.concat([better, sparse_new], ignore_index=True)},
        baseline="base", outcomes=["x", "sparse"], horizon=60,
        n_bootstrap=60, seed=3,
        family_by_outcome={"x": "family", "sparse": "family"},
        skip_under_supported=True,
    )
    assert set(estimates["outcome"]) == {"x"}
    assert set(contrasts["outcome"]) == {"x"}
    assert report["n_outcomes_requested"] == 2
    assert report["n_outcomes_included"] == 1
    assert report["inventory"]["sparse"]["status"] == "excluded_low_support"
