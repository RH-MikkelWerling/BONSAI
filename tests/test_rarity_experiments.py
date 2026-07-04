import numpy as np
import pandas as pd
import pytest

from opera.evaluation.rarity import (
    assert_prediction_parity,
    bootstrap_metric_table,
    build_nested_sample_manifest,
    build_task_label_table,
    classify_task_eligibility,
    paired_bootstrap_difference,
    patient_id_hash,
    resolve_sample_sizes,
    summarize_natural_differences,
    task_count_record,
    validate_nested_manifest,
    validate_patient_splits,
)


SPLIT_CONTRACT = {
    "train_end": "2021-12-31",
    "val_start": "2022-01-01",
    "val_end": "2022-12-31",
    "test_start": "2023-01-01",
    "test_end": None,
    "date_col": "index_date",
    "train_key": "train",
    "val_key": "tuning",
    "test_key": "held_out",
}


def _labels(n_train=120, n_tuning=30):
    rows = []
    subject_id = 0
    for split, n, year in (("train", n_train, 2021), ("tuning", n_tuning, 2022)):
        for index in range(n):
            rows.append(
                {
                    "subject_id": subject_id,
                    "split": split,
                    "index_year": year,
                    "label": index % 4 == 0,
                    "label_status": "positive" if index % 4 == 0 else "negative",
                }
            )
            subject_id += 1
    return pd.DataFrame(rows)


def test_split_contract_rejects_patient_overlap():
    outcomes = pd.DataFrame(
        {
            "subject_id": [1, 1],
            "split": ["train", "held_out"],
            "index_date": pd.to_datetime(["2021-01-01", "2023-01-01"]),
        }
    )
    with pytest.raises(ValueError, match="split contract"):
        validate_patient_splits(outcomes, SPLIT_CONTRACT)


def test_task_labels_count_positive_negative_and_indeterminate():
    outcomes = pd.DataFrame(
        {
            "subject_id": [1, 2, 3, 4],
            "split": ["train", "train", "held_out", "held_out"],
            "index_date": pd.to_datetime(["2021-01-01", "2021-01-01", "2023-01-01", "2023-01-01"]),
            "outcome_date": pd.to_datetime(["2021-01-10", None, "2023-01-10", None]),
            "censor_date": pd.to_datetime(["2021-03-01", "2021-03-01", "2023-03-01", "2023-01-15"]),
        }
    )
    population = pd.DataFrame({"subject_id": [1, 2, 3, 4], "cohort_fine": ["DLBCL"] * 4})
    labels = build_task_label_table(
        outcomes=outcomes,
        population=population,
        cohort_fine="DLBCL",
        n_hours_end_include=24 * 30,
        outcome_name="event_30d",
    )
    counts = task_count_record(labels)
    assert counts["n_train_positive"] == 1
    assert counts["n_train_negative"] == 1
    assert counts["n_test_positive"] == 1
    assert counts["n_test_indeterminate"] == 1
    assert labels["cohort_grouped"].dropna().unique().tolist() == ["DLBCL_like"]


def test_nested_sampling_is_reproducible_unique_and_nested():
    labels = _labels()
    first, skipped = build_nested_sample_manifest(
        labels, sizes=[20, 50, 100], seed=17, min_positive=2, min_negative=2
    )
    second, _ = build_nested_sample_manifest(
        labels, sizes=[20, 50, 100], seed=17, min_positive=2, min_negative=2
    )
    assert skipped.empty
    pd.testing.assert_frame_equal(first, second)
    validate_nested_manifest(first)
    train = first[first["split"] == "train"]
    sets = [set(group["subject_id"]) for _, group in train.groupby("sample_size", sort=True)]
    assert sets[0] < sets[1] < sets[2]
    assert not first.duplicated(["sample_size", "seed", "split", "subject_id"]).any()
    assert patient_id_hash(sets[0]) == first[first["sample_size"] == 20]["patient_id_hash"].iloc[0]


def test_infeasible_and_one_class_levels_are_logged():
    labels = _labels(n_train=20, n_tuning=4)
    manifest, skipped = build_nested_sample_manifest(
        labels, sizes=[2, 10, 100], seed=4, min_positive=3, min_negative=3
    )
    assert 2 in set(skipped["sample_size"])
    assert 100 in set(skipped["sample_size"])
    assert set(manifest["sample_size"]) == {10}


def test_absolute_and_percentage_sample_sizes_are_deduplicated():
    assert resolve_sample_sizes(1000, [100, 250, 5000], [0.1, 0.5]) == [100, 250, 500, 1000]


def test_task_eligibility_uses_prespecified_counts_only():
    counts = {
        "n_train_determinate": 500,
        "n_train_positive": 50,
        "n_train_negative": 450,
        "n_test_total": 130,
        "n_test_positive": 50,
        "n_test_negative": 60,
        "n_test_indeterminate": 20,
    }
    result = classify_task_eligibility(
        counts,
        min_train_patients=500,
        min_train_positive=20,
        min_train_negative=20,
        primary_test_positive=50,
        primary_test_negative=50,
        aggregate_test_positive=10,
        aggregate_test_negative=10,
        max_test_indeterminate_fraction=0.5,
    )
    assert result["synthetic_eligible"] is True
    assert result["natural_viability_tier"] == "primary"


def _prediction_frame(offset=0.0):
    # Two rows per patient exercise cluster rather than row-level resampling.
    return pd.DataFrame(
        {
            "subject_id": np.repeat(np.arange(10), 2),
            "label": np.repeat([0, 1] * 5, 2),
            "probability": np.clip(np.repeat(np.linspace(0.05, 0.95, 10), 2) + offset, 0.001, 0.999),
        }
    )


def test_bootstrap_is_patient_clustered_and_reproducible():
    predictions = _prediction_frame()
    first = bootstrap_metric_table(predictions, n_bootstrap=40, seed=3)
    second = bootstrap_metric_table(predictions, n_bootstrap=40, seed=3)
    pd.testing.assert_frame_equal(first, second)
    assert set(first["n_test_patients"]) == {10}
    assert {"auroc", "auprc", "pr_skill", "brier_skill", "log_loss"}.issubset(first["metric"])


def test_paired_bootstrap_and_fixed_test_parity():
    baseline = _prediction_frame(-0.05)
    model = _prediction_frame(0.03)
    assert_prediction_parity({"model": model, "baseline": baseline})
    result = paired_bootstrap_difference(
        model,
        baseline,
        model_name="model",
        comparator_name="baseline",
        n_bootstrap=30,
        seed=5,
    )
    assert set(result["n_test_patients"]) == {10}
    bad = baseline[baseline["subject_id"] != 9]
    with pytest.raises(ValueError, match="parity"):
        assert_prediction_parity({"model": model, "baseline": bad})


def test_natural_summary_averages_seeds_before_counting_tasks():
    frame = pd.DataFrame(
        {
            "cohort_fine": ["A", "A", "B", "B"],
            "outcome": ["o"] * 4,
            "model": ["m"] * 4,
            "comparator": ["b"] * 4,
            "metric": ["auroc"] * 4,
            "natural_viability_tier": ["primary"] * 4,
            "seed": [1, 2, 1, 2],
            "difference": [0.1, 0.2, -0.1, 0.1],
            "difference_se": [0.05] * 4,
            "n_test_patients": [100, 100, 50, 50],
        }
    )
    summary = summarize_natural_differences(frame).iloc[0]
    assert summary["n_tasks"] == 2
    assert summary["macro_mean_difference"] == pytest.approx(0.075)


def test_new_rarity_plots_save_png_and_pdf(tmp_path):
    from opera.visualization.rarity_plots import (
        plot_combined_rarity_experiment,
        plot_natural_viability_heatmap,
    )

    eligibility = pd.DataFrame(
        {
            "cohort_fine": ["DLBCL"],
            "outcome": ["mortality_1y"],
            "n_test_positive": [55],
            "natural_viability_tier": ["primary"],
        }
    )
    synthetic = pd.DataFrame(
        {
            "model": ["joint_opera", "joint_opera"],
            "comparator": ["tabular", "tabular"],
            "sample_size": [100, 500],
            "metric": ["auroc", "auroc"],
            "macro_mean": [0.04, 0.02],
        }
    )
    natural = pd.DataFrame(
        {
            "outcome": ["mortality_1y"],
            "metric": ["auroc"],
            "n_train_determinate": [500],
            "difference": [0.03],
        }
    )
    heatmap = tmp_path / "heatmap.png"
    combined = tmp_path / "combined.png"
    plot_natural_viability_heatmap(eligibility, str(heatmap))
    plot_combined_rarity_experiment(synthetic, natural, save_path=str(combined))
    for path in (heatmap, combined):
        assert path.exists()
        assert path.with_suffix(".pdf").exists()
