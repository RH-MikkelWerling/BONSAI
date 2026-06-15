import numpy as np
import pandas as pd
import pytest

from opera.analysis.rarity_invariance_panel import (
    EXPECTED_WEIGHTERS,
    build_rarity_delta_table,
    run_rarity_invariance_panel,
)
from opera.evaluation.metrics import compute_discrimination_metrics


WEIGHTER_MODELS = {
    "kendall": "opera_kendall",
    "uniform": "opera_uniform",
    "famo": "opera_famo",
}


def _synthetic_prediction_results() -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(44)
    event_rates = np.geomspace(0.03, 0.45, 9)
    result_rows = []
    rate_rows = []
    for outcome_index, event_rate in enumerate(event_rates):
        outcome = f"outcome_{outcome_index:02d}"
        n_subjects = 5000
        labels = rng.binomial(1, event_rate, size=n_subjects)
        noise = rng.normal(size=n_subjects)
        baseline_scores = noise + 0.45 * labels
        rarity_strength = -np.log10(event_rate)
        comparator_scores = noise + (0.45 + 0.38 * rarity_strength) * labels
        baseline_probability = 1.0 / (1.0 + np.exp(-baseline_scores))
        comparator_probability = 1.0 / (1.0 + np.exp(-comparator_scores))
        baseline_auroc = compute_discrimination_metrics(
            labels,
            baseline_probability,
        )["auroc"]
        comparator_auroc = compute_discrimination_metrics(
            labels,
            comparator_probability,
        )["auroc"]
        common = {
            "cohort": "synthetic",
            "outcome": outcome,
            "outcome_window_hours": 720,
            "split": "held_out",
            "seed": 42,
            "training_fraction": 1.0,
            "rarity_mode": "real",
            "evaluation_subset": "full",
            "n_total": n_subjects,
            "class_balanced": False,
        }
        result_rows.append(
            {
                **common,
                "model_family": "tabular_ehr",
                "checkpoint_path": "shared_baseline",
                "auroc": baseline_auroc,
            }
        )
        for weighter in EXPECTED_WEIGHTERS:
            result_rows.append(
                {
                    **common,
                    "model_family": WEIGHTER_MODELS[weighter],
                    "checkpoint_path": f"{weighter}_model",
                    "auroc": comparator_auroc,
                }
            )
        rate_rows.append(
            {
                "cohort": "synthetic",
                "outcome": outcome,
                "outcome_window_hours": 720,
                "event_rate": float(labels.mean()),
            }
        )
    return pd.DataFrame(result_rows), pd.DataFrame(rate_rows)


def test_rarity_invariance_pipeline_from_synthetic_predictions(tmp_path):
    results, event_rates = _synthetic_prediction_results()

    summary = run_rarity_invariance_panel(
        results,
        event_rates,
        tmp_path,
        weighter_models=WEIGHTER_MODELS,
        baseline_model="tabular_ehr",
        evaluation_subset="full",
        n_bootstrap=500,
        seed=9,
        pipeline_validation=True,
    )

    assert summary["pipeline_validation"]
    assert summary["gradient_invariant_to_weighter"]
    assert summary["verdict"].startswith("Pipeline validation only")
    assert (tmp_path / "rarity_invariance_task_deltas.csv").exists()
    assert (tmp_path / "rarity_invariance_trends.csv").exists()
    assert (tmp_path / "rarity_invariance_slope_differences.csv").exists()
    assert (tmp_path / "rarity_invariance_panel.png").exists()
    assert (tmp_path / "rarity_invariance_panel.pdf").exists()
    assert (tmp_path / "rarity_invariance_verdict.json").exists()


def test_rarity_invariance_rejects_ipi_on_full_subset():
    results, event_rates = _synthetic_prediction_results()
    results.loc[results["model_family"] == "tabular_ehr", "model_family"] = "ipi"

    with pytest.raises(ValueError, match="ipi_complete"):
        build_rarity_delta_table(
            results,
            event_rates,
            weighter_models=WEIGHTER_MODELS,
            baseline_model="ipi",
            evaluation_subset="full",
        )


def test_rarity_invariance_rejects_class_balance_confound():
    results, event_rates = _synthetic_prediction_results()
    results.loc[results["model_family"] != "tabular_ehr", "class_balanced"] = True

    with pytest.raises(ValueError, match="confounded"):
        build_rarity_delta_table(
            results,
            event_rates,
            weighter_models=WEIGHTER_MODELS,
            baseline_model="tabular_ehr",
            evaluation_subset="full",
        )


def test_rarity_invariance_rejects_baseline_denominator_mismatch():
    results, event_rates = _synthetic_prediction_results()
    first_outcome = results["outcome"].iloc[0]
    mask = (results["model_family"] == "tabular_ehr") & (
        results["outcome"] == first_outcome
    )
    results.loc[mask, "n_total"] = 4500

    with pytest.raises(ValueError, match="denominators differ"):
        build_rarity_delta_table(
            results,
            event_rates,
            weighter_models=WEIGHTER_MODELS,
            baseline_model="tabular_ehr",
            evaluation_subset="full",
        )
