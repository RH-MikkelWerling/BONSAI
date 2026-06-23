import json
import sys
import types

import numpy as np
import pandas as pd
import pytest

from opera.evaluation.comparison import (
    ComparisonRunner,
    comparison_results_to_frame,
    derive_horizon_labels,
    infer_tau_days,
    fit_cox_survival,
)
from opera.evaluation.run_all import flatten_grid_results


def _make_synthetic_inputs():
    rng = np.random.RandomState(7)
    patient_ids = [f"p{i:03d}" for i in range(40)]
    diseases = ["DLBCL"] * 20 + ["FL"] * 20

    outcomes_rows = []
    ehr_rows = []
    rkkp_rows = []
    embeddings_base = {}
    embeddings_dapt = {}
    embeddings_opera = {}

    for idx, (patient_id, disease) in enumerate(zip(patient_ids, diseases)):
        within_disease_idx = idx if idx < 20 else idx - 20
        if within_disease_idx < 10:
            split = "train"
        elif within_disease_idx < 16:
            split = "tuning"
        else:
            split = "held_out"

        event = 1 if idx % 3 == 0 else 0
        if idx % 5 == 0:
            time_to_event = 120.0
        elif idx % 4 == 0:
            time_to_event = 900.0
        else:
            time_to_event = 500.0

        outcomes_rows.append(
            {
                "patient_id": patient_id,
                "disease_subtype": disease,
                "outcome_name": "treatment_failure_12m",
                "event_indicator": event,
                "time_to_event": time_to_event,
                "index_date": "2021-01-01",
                "split": split,
            }
        )

        ehr_rows.append(
            {
                "patient_id": patient_id,
                "lab_mean": float(rng.normal(loc=0.2 if disease == "DLBCL" else -0.1)),
                "diag_count": float(rng.poisson(lam=3 + (idx % 4))),
                "med_count": float(rng.poisson(lam=2 + (idx % 3))),
            }
        )

        rkkp_rows.append(
            {
                "patient_id": patient_id,
                "age": 45 + idx,
                "sex": "M" if idx % 2 == 0 else "F",
                "ipi_score": float(idx % 5),
                "flipi_score": float((idx + 1) % 5),
            }
        )

        base = rng.normal(size=8)
        dapt = base + rng.normal(scale=0.1, size=8)
        opera = dapt + rng.normal(scale=0.1, size=8)
        embeddings_base[patient_id] = base
        embeddings_dapt[patient_id] = dapt
        embeddings_opera[patient_id] = opera

    return {
        "outcomes": pd.DataFrame(outcomes_rows),
        "ehr_features": pd.DataFrame(ehr_rows),
        "rkkp": pd.DataFrame(rkkp_rows),
        "embeddings_base": embeddings_base,
        "embeddings_dapt": embeddings_dapt,
        "embeddings_opera": embeddings_opera,
        "disease_cohorts": {
            "DLBCL": patient_ids[:20],
            "FL": patient_ids[20:],
        },
    }


def test_infer_tau_days_and_horizon_labels():
    assert infer_tau_days("OS_2y") == 2 * 365.25
    assert infer_tau_days("AKI") == 30.0
    assert infer_tau_days("treatment_failure_12m") == 365.0

    labels = derive_horizon_labels(
        times=np.array([100.0, 200.0, 600.0]),
        events=np.array([1, 0, 0]),
        tau_days=365.0,
    )
    assert labels["eligible"].tolist() == [True, False, True]
    assert labels["label"].tolist()[0] == 1.0
    assert np.isnan(labels["label"].tolist()[1])
    assert labels["label"].tolist()[2] == 0.0


def test_cox_fit_failure_is_raised_without_classifier_fallback(monkeypatch):
    class FailingCoxPHFitter:
        def __init__(self, penalizer):
            self.penalizer = penalizer

        def fit(self, *args, **kwargs):
            raise RuntimeError("forced fit failure")

    monkeypatch.setitem(
        sys.modules,
        "lifelines",
        types.SimpleNamespace(CoxPHFitter=FailingCoxPHFitter),
    )
    with pytest.raises(RuntimeError, match="no binary classifier fallback"):
        fit_cox_survival(
            features=pd.DataFrame({"age": [50, 60, 70, 80]}),
            times=np.array([10.0, 20.0, 30.0, 40.0]),
            events=np.array([1, 1, 0, 0]),
            tau_days=30.0,
            seed=42,
        )


def test_comparison_runner_returns_json_serializable_results():
    inputs = _make_synthetic_inputs()
    runner = ComparisonRunner(
        outcome_name="treatment_failure_12m",
        cohort_specification="DLBCL",
        mode="disease_specific",
        outcomes=inputs["outcomes"],
        rkkp=inputs["rkkp"],
        ehr_features=inputs["ehr_features"],
        embeddings_base=inputs["embeddings_base"],
        embeddings_dapt=inputs["embeddings_dapt"],
        embeddings_opera=inputs["embeddings_opera"],
        disease_cohorts=inputs["disease_cohorts"],
        n_splits=3,
        n_bootstrap=10,
        enabled_models=["ClinicalScore", "LinearProbe_base", "OPERA_mlp"],
        show_progress=False,
    )

    results = runner.run()

    assert results["cohort_specification"] == "DLBCL"
    assert results["evaluation_strategy"] == "prospective_holdout"
    assert set(results["models"].keys()) == {
        "ClinicalScore",
        "LinearProbe_base",
        "OPERA_mlp",
    }
    assert results["score_column"] == "clinical_score_selected"
    assert results["models"]["ClinicalScore"]["n_eval"] == 4
    assert "table_row" in results["models"]["ClinicalScore"]

    json.dumps(results)


def test_comparison_results_flatten_for_tables():
    inputs = _make_synthetic_inputs()
    runner = ComparisonRunner(
        outcome_name="treatment_failure_12m",
        cohort_specification="all",
        mode="pooled",
        outcomes=inputs["outcomes"],
        rkkp=inputs["rkkp"],
        ehr_features=inputs["ehr_features"],
        embeddings_base=inputs["embeddings_base"],
        embeddings_dapt=inputs["embeddings_dapt"],
        embeddings_opera=inputs["embeddings_opera"],
        disease_cohorts=inputs["disease_cohorts"],
        n_splits=3,
        n_bootstrap=5,
        enabled_models=["XGBoost_all", "OPERA"],
        show_progress=False,
    )
    pooled_results = runner.run()

    frame = comparison_results_to_frame(pooled_results)
    assert set(frame["model"]) == {"XGBoost_all", "OPERA"}
    assert set(frame["mode"]) == {"pooled"}
    assert set(frame["n_eval"]) == {8}

    grid = {
        "results": {
            "treatment_failure_12m": {
                "all_hematology": pooled_results,
            }
        }
    }
    flat = flatten_grid_results(grid)
    assert "grid_outcome_name" in flat.columns
    assert "grid_cohort_name" in flat.columns
