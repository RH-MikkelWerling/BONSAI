import json

import pytest

from opera.evaluation.aggregation import (
    add_task_size_bins,
    build_joint_vs_per_cohort_table,
    build_delta_vs_baseline_table,
    build_wide_metric_table,
    collect_result_rows,
    compute_model_delta_table,
    filter_results_for_paper_aggregates,
    mark_rare_cohort_stability,
    split_rarity_delta_tables,
    summarize_by_model,
    summarize_by_task_size,
    summarize_scale_ablation,
    expand_metric_columns,
    validate_compatible_result_rows,
)


def test_collect_wide_and_delta_tables(tmp_path):
    rows = [
        {
            "cohort": "dlbcl",
            "outcome": "mortality",
            "outcome_window_hours": 8760,
            "split": "held_out",
            "seed": 42,
            "training_fraction": 1.0,
            "model_family": "per_cohort",
            "training_stage": "per_task_finetuning",
            "pretraining_scale": "small",
            "auroc": 0.70,
            "auprc": 0.30,
            "brier_score": 0.20,
            "n_total": 80,
            "rarity_mode": "real",
            "n_train": 60,
            "n_events_train": 2,
            "n_events_test": 4,
        },
        {
            "cohort": "dlbcl",
            "outcome": "mortality",
            "outcome_window_hours": 8760,
            "split": "held_out",
            "seed": None,
            "training_fraction": 0.1,
            "model_family": "per_cohort",
            "training_stage": "per_task_finetuning",
            "pretraining_scale": "small",
            "auroc": 0.60,
            "n_total": 80,
            "rarity_mode": "synthetic",
            "n_train": 8,
        },
        {
            "cohort": "dlbcl",
            "outcome": "mortality",
            "outcome_window_hours": 8760,
            "split": "held_out",
            "seed": 44,
            "training_fraction": 0.1,
            "model_family": "joint",
            "training_stage": "joint_finetuning",
            "pretraining_scale": "large",
            "auroc": 0.66,
            "n_total": 80,
            "rarity_mode": "synthetic",
            "n_train": 8,
        },
        {
            "cohort": "dlbcl",
            "outcome": "mortality",
            "outcome_window_hours": 8760,
            "split": "held_out",
            "seed": 42,
            "training_fraction": 1.0,
            "model_family": "joint",
            "training_stage": "joint_finetuning",
            "pretraining_scale": "large",
            "auroc": 0.75,
            "auprc": 0.35,
            "brier_score": 0.18,
            "n_total": 80,
            "rarity_mode": "real",
            "n_train": 60,
            "n_events_train": 2,
            "n_events_test": 4,
        },
    ]
    path = tmp_path / "nested" / "result.jsonl"
    path.parent.mkdir()
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")

    collected = collect_result_rows(str(tmp_path))
    wide = build_wide_metric_table(collected)
    delta = compute_model_delta_table(wide, baseline="per_cohort", comparator="joint")
    joint_table = build_joint_vs_per_cohort_table(
        collected,
        joint_model="joint",
        per_cohort_model="per_cohort",
    )
    summary = summarize_by_model(collected)
    binned = add_task_size_bins(collected)
    size_summary = summarize_by_task_size(collected)
    scale_summary = summarize_scale_ablation(collected)
    rarity_delta = build_delta_vs_baseline_table(collected, baseline="per_cohort")
    rarity_tables = split_rarity_delta_tables(
        collected,
        baseline="per_cohort",
        min_train_events=5,
        min_test_events=5,
    )

    assert len(collected) == 4
    full_fraction = wide[wide["training_fraction"] == 1.0].iloc[0]
    assert full_fraction["joint__auroc"] == 0.75
    full_delta = delta[delta["training_fraction"] == 1.0].iloc[0]
    assert full_delta["joint_minus_per_cohort__auroc"] == 0.05
    full_joint_table = joint_table[joint_table["training_fraction"] == 1.0].iloc[0]
    assert full_joint_table["joint_minus_per_cohort__auroc"] == 0.05
    assert set(summary["model_family"]) == {"joint", "per_cohort"}
    assert str(binned.loc[0, "task_size_bin"]) == "<100"
    assert len(size_summary) == 2
    assert set(scale_summary["pretraining_scale"]) == {"small", "large"}
    assert (
        rarity_delta[rarity_delta["training_fraction"] == 1.0].iloc[0][
            "delta_auroc_vs_baseline"
        ]
        == 0.05
    )
    assert not rarity_tables["real_rarity_task_level"].empty
    assert not rarity_tables["real_rarity_pooled"].empty
    assert not rarity_tables["synthetic_rarity_pooled"].empty
    assert set(rarity_tables["synthetic_rarity_task_level"]["rarity_mode"]) == {
        "synthetic"
    }
    assert set(rarity_tables["real_rarity_task_level"]["rarity_mode"]) == {"real"}
    assert rarity_tables["real_rarity_task_level"]["supplement_only"].all()
    assert rarity_tables["real_rarity_pooled_main"].empty


def test_wide_metric_table_ignores_empty_optional_index_columns():
    pd = pytest.importorskip("pandas")
    results = pd.DataFrame(
        [
            {
                "cohort": "dlbcl",
                "outcome": "mortality",
                "outcome_window_hours": 8760,
                "split": "held_out",
                "seed": None,
                "training_fraction": None,
                "rarity_mode": "none",
                "model_family": "baseline",
                "auroc": 0.7,
            },
            {
                "cohort": "dlbcl",
                "outcome": "mortality",
                "outcome_window_hours": 8760,
                "split": "held_out",
                "seed": None,
                "training_fraction": None,
                "rarity_mode": "none",
                "model_family": "opera",
                "auroc": 0.8,
            },
        ]
    )

    wide = build_wide_metric_table(results, metrics=("auroc",))

    assert len(wide) == 1
    assert wide["baseline__auroc"].iloc[0] == pytest.approx(0.7)
    assert wide["opera__auroc"].iloc[0] == pytest.approx(0.8)


def test_metric_prefix_expansion_includes_horizon_specific_ipcw_columns():
    pd = pytest.importorskip("pandas")
    results = pd.DataFrame(
        [
            {
                "cohort": "dlbcl",
                "outcome": "mortality",
                "split": "held_out",
                "model_family": "opera_ipcw_bce",
                "auroc": 0.7,
                "ipcw_auc_365d": 0.72,
                "ipcw_brier_365d": 0.14,
            },
            {
                "cohort": "dlbcl",
                "outcome": "mortality",
                "split": "held_out",
                "model_family": "baseline",
                "auroc": 0.6,
                "ipcw_auc_365d": 0.62,
                "ipcw_brier_365d": 0.18,
            },
        ]
    )

    assert expand_metric_columns(results, ["ipcw_auc"]) == ["ipcw_auc_365d"]
    wide = build_wide_metric_table(results, metrics=("ipcw_auc", "ipcw_brier"))

    assert "opera_ipcw_bce__ipcw_auc_365d" in wide.columns
    assert "baseline__ipcw_brier_365d" in wide.columns


def test_mark_rare_cohort_stability_keeps_rows_and_marks_reasons():
    pd = pytest.importorskip("pandas")
    rows = pd.DataFrame(
        [
            {"n_events_train": 1, "n_events_test": 10},
            {"n_events_train": 6, "n_events_test": 2},
        ]
    )

    marked = mark_rare_cohort_stability(
        rows,
        min_train_events=5,
        min_test_events=5,
    )

    assert len(marked) == 2
    assert marked["supplement_only"].tolist() == [True, True]
    assert marked["supplement_only_reason"].tolist() == [
        "min_train_events",
        "min_test_events",
    ]


def test_mark_rare_cohort_stability_handles_filtered_indices():
    pd = pytest.importorskip("pandas")
    rows = pd.DataFrame(
        [
            {"n_events_train": 10, "n_events_test": 10},
            {"n_events_train": 1, "n_events_test": 10},
            {"n_events_train": 6, "n_events_test": 2},
        ],
        index=[10, 20, 30],
    )

    marked = mark_rare_cohort_stability(
        rows.iloc[1:],
        min_train_events=5,
        min_test_events=5,
    )

    assert marked["supplement_only"].tolist() == [True, True]
    assert marked["supplement_only_reason"].tolist() == [
        "min_train_events",
        "min_test_events",
    ]


def test_rarity_pooled_main_keeps_stable_real_cells():
    pd = pytest.importorskip("pandas")
    rows = pd.DataFrame(
        [
            {
                "cohort": "rare",
                "outcome": "mortality",
                "split": "held_out",
                "training_fraction": 1.0,
                "model_family": "baseline",
                "auroc": 0.60,
                "rarity_mode": "real",
                "n_train": 80,
                "n_events_train": 10,
                "n_events_test": 8,
            },
            {
                "cohort": "rare",
                "outcome": "mortality",
                "split": "held_out",
                "training_fraction": 1.0,
                "model_family": "opera",
                "auroc": 0.70,
                "rarity_mode": "real",
                "n_train": 80,
                "n_events_train": 10,
                "n_events_test": 8,
            },
        ]
    )

    tables = split_rarity_delta_tables(
        rows,
        baseline="baseline",
        min_train_events=5,
        min_test_events=5,
    )

    assert not tables["real_rarity_pooled_main"].empty
    assert tables["real_rarity_task_level"]["supplement_only"].tolist() == [False]


def test_paper_aggregate_filter_excludes_ipi_and_ipi_complete_rows():
    pd = pytest.importorskip("pandas")
    rows = pd.DataFrame(
        [
            {"model_family": "tabular_ehr", "evaluation_subset": "full", "auroc": 0.7},
            {"model_family": "opera", "evaluation_subset": "full", "auroc": 0.8},
            {"model_family": "ipi", "evaluation_subset": "ipi_complete", "auroc": 0.6},
            {
                "model_family": "opera",
                "evaluation_subset": "ipi_complete",
                "auroc": 0.75,
            },
        ]
    )

    filtered = filter_results_for_paper_aggregates(rows)
    warnings = validate_compatible_result_rows(rows)

    assert filtered["model_family"].tolist() == ["tabular_ehr", "opera"]
    assert set(warnings["category"]) >= {"mixed_evaluation_subsets", "excluded_ipi"}


def test_paper_aggregate_filter_can_build_ipi_credibility_subset():
    pd = pytest.importorskip("pandas")
    rows = pd.DataFrame(
        [
            {"model_family": "tabular_ehr", "evaluation_subset": "full", "auroc": 0.7},
            {"model_family": "ipi", "evaluation_subset": "ipi_complete", "auroc": 0.6},
            {
                "model_family": "opera",
                "evaluation_subset": "ipi_complete",
                "auroc": 0.75,
            },
        ]
    )

    filtered = filter_results_for_paper_aggregates(
        rows,
        evaluation_subset="ipi_complete",
        include_ipi=True,
    )

    assert filtered["model_family"].tolist() == ["ipi", "opera"]
