import json

import pytest

from opera.evaluation.aggregation import (
    add_task_size_bins,
    build_consort_table,
    build_joint_vs_per_cohort_table,
    build_delta_vs_baseline_table,
    build_wide_metric_table,
    check_competing_event_denominator_consistency,
    collect_label_split_summaries,
    collect_result_rows,
    compute_model_delta_table,
    compute_paired_denominators,
    filter_results_for_paper_aggregates,
    validate_result_rows_denominators,
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
            "c_index_within_fine": 0.64,
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
    assert "c_index_within_fine" in collected.columns
    assert collected["c_index_within_fine"].dropna().tolist() == [0.64]
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


def test_duplicate_result_keys_are_strict_aggregation_errors():
    pd = pytest.importorskip("pandas")
    row = {
        "cohort": "dlbcl",
        "outcome": "mortality_1y",
        "split": "held_out",
        "seed": 42,
        "training_fraction": 1.0,
        "rarity_mode": "none",
        "evaluation_subset": "full",
        "model_family": "opera",
        "auroc": 0.8,
    }

    issues = validate_compatible_result_rows(pd.DataFrame([row, row]))
    duplicate = issues[issues["category"] == "duplicate_result_keys"].iloc[0]

    assert duplicate["severity"] == "error"


def _paired_denominator_rows(pd, n_totals):
    return pd.DataFrame(
        [
            {
                "cohort": "dlbcl",
                "outcome": "mortality",
                "evaluation_subset": "full",
                "model_family": family,
                "auroc": 0.7,
                "n_total": n_total,
            }
            for family, n_total in n_totals.items()
        ]
    )


def test_compute_paired_denominators_consistent():
    pd = pytest.importorskip("pandas")
    rows = _paired_denominator_rows(
        pd,
        {"baseline": 200, "opera": 200, "joint": 200},
    )

    summary = compute_paired_denominators(rows)

    assert len(summary) == 1
    cell = summary.iloc[0]
    assert cell["n_variants"] == 3
    assert cell["n_total_min"] == 200
    assert cell["n_total_max"] == 200
    assert cell["n_total_cv"] == 0.0
    assert bool(cell["mismatch_flag"]) is False


def test_compute_paired_denominators_mismatch():
    pd = pytest.importorskip("pandas")
    rows = _paired_denominator_rows(
        pd,
        {"baseline": 200, "opera": 260, "joint": 200},
    )

    summary = compute_paired_denominators(rows)

    assert len(summary) == 1
    cell = summary.iloc[0]
    assert cell["n_total_min"] == 200
    assert cell["n_total_max"] == 260
    assert cell["n_total_cv"] > 0.05
    assert bool(cell["mismatch_flag"]) is True


def test_compute_paired_denominators_raises():
    pd = pytest.importorskip("pandas")
    rows = _paired_denominator_rows(
        pd,
        {"baseline": 200, "opera": 260},
    )

    with pytest.raises(ValueError) as excinfo:
        compute_paired_denominators(rows, raise_on_mismatch=True)

    message = str(excinfo.value)
    assert "dlbcl" in message
    assert "mortality" in message
    assert "full" in message


def test_compute_paired_denominators_nan_n_total():
    pd = pytest.importorskip("pandas")
    rows = _paired_denominator_rows(
        pd,
        {"baseline": 200, "opera": None, "joint": 200},
    )

    summary = compute_paired_denominators(rows)

    assert len(summary) == 1
    cell = summary.iloc[0]
    assert cell["n_variants"] == 2
    assert cell["n_total_cv"] == 0.0
    assert bool(cell["mismatch_flag"]) is False


def test_validate_result_rows_denominators_returns_empty_on_clean_data():
    pd = pytest.importorskip("pandas")
    rows = _paired_denominator_rows(
        pd,
        {"baseline": 200, "opera": 200, "joint": 200},
    )

    assert validate_result_rows_denominators(rows) == []


def _write_label_split_summary(pd, path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def _label_split_row(
    split,
    *,
    cohort="dlbcl",
    outcome="mortality",
    n_subjects=100,
    n_labelled=90,
    n_positive=30,
    n_negative=55,
    n_insufficient_followup=5,
):
    return {
        "split": split,
        "n_subjects": n_subjects,
        "n_labelled": n_labelled,
        "n_positive": n_positive,
        "n_negative": n_negative,
        "n_insufficient_followup": n_insufficient_followup,
        "prevalence": n_positive / n_labelled,
        "outcome": outcome,
        "cohort": cohort,
    }


def test_collect_label_split_summaries_empty_dir(tmp_path):
    collected = collect_label_split_summaries(tmp_path)

    assert collected.empty


def test_collect_label_split_summaries_finds_nested_csvs(tmp_path):
    pd = pytest.importorskip("pandas")
    _write_label_split_summary(
        pd,
        tmp_path / "variant_a" / "label_split_summary.csv",
        [_label_split_row("train"), _label_split_row("held_out")],
    )
    _write_label_split_summary(
        pd,
        tmp_path / "variant_b" / "nested" / "label_split_summary.csv",
        [_label_split_row("train"), _label_split_row("held_out")],
    )

    collected = collect_label_split_summaries(tmp_path)

    assert len(collected) == 4
    assert "cell_dir" in collected.columns
    assert collected["cell_dir"].nunique() == 2


def test_build_consort_table_empty_input():
    pd = pytest.importorskip("pandas")

    table = build_consort_table(pd.DataFrame())

    assert table.empty


def test_build_consort_table_basic_structure():
    pd = pytest.importorskip("pandas")
    splits = ["train", "val", "held_out"]
    outcomes = ["mortality", "relapse"]
    variants = ["variant_a", "variant_b"]
    rows = []
    for variant in variants:
        for outcome in outcomes:
            for split in splits:
                row = _label_split_row(split, outcome=outcome)
                row["model_family"] = variant
                rows.append(row)
    label_summaries = pd.DataFrame(rows)

    table = build_consort_table(label_summaries)

    assert list(table.columns) == [
        "cohort",
        "outcome",
        "split",
        "n_subjects",
        "n_labelled",
        "n_positive",
        "n_negative",
        "n_insufficient_followup",
        "prevalence",
        "n_model_variants",
    ]
    assert len(table) == len(splits) * len(outcomes)
    assert (table["n_model_variants"] == 2).all()
    cell = table[
        (table["outcome"] == "mortality") & (table["split"] == "held_out")
    ].iloc[0]
    assert cell["n_subjects"] == 100
    assert cell["prevalence"] == pytest.approx(30 / 90)


def _competing_event_rows(pd, values):
    return pd.DataFrame(
        [
            {
                "cohort": "dlbcl",
                "outcome": "mortality",
                "model_family": family,
                "n_competing_events_test": value,
            }
            for family, value in values.items()
        ]
    )


def test_check_competing_event_denominator_consistency_clean():
    pd = pytest.importorskip("pandas")
    rows = _competing_event_rows(
        pd,
        {"baseline": 12, "opera": 12, "joint": 12},
    )

    assert check_competing_event_denominator_consistency(rows) == []


def test_check_competing_event_denominator_consistency_mismatch():
    pd = pytest.importorskip("pandas")
    rows = _competing_event_rows(
        pd,
        {"baseline": 12, "opera": 30, "joint": 12},
    )

    messages = check_competing_event_denominator_consistency(rows)

    assert messages
    assert any("dlbcl" in message and "mortality" in message for message in messages)
