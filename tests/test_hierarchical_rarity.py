import json
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from opera.analysis.bayesian_rarity import (
    fit_hierarchical_rarity_model,
    prepare_rarity_data,
)
from opera.evaluation.hierarchical_rarity import (
    apply_outcome_families,
    assert_paired_prediction_parity,
    attach_task_metadata,
    build_paired_delta_tables,
    discover_prediction_artifacts,
    load_binary_prediction_artifact,
    paired_bootstrap_differences,
    summarize_patient_overlap,
)
from opera.visualization.hierarchical_rarity_plots import (
    _selected_labels,
    aggregate_scatter_cells,
    plot_hierarchical_rarity_curve,
)


def _write_prediction(
    path: Path,
    probabilities: np.ndarray,
    *,
    with_survival_only_row: bool = False,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = len(probabilities)
    labels = np.tile([0, 1], n // 2)
    subject_ids = np.arange(n)
    if with_survival_only_row:
        subject_ids = np.append(subject_ids, 999)
        labels = np.append(labels, 0)
        probabilities = np.append(probabilities, 0.2)
        binary_mask = np.append(np.ones(n, dtype=np.uint8), 0)
        np.savez(
            path,
            subject_ids=subject_ids,
            labels=labels,
            probabilities=probabilities,
            binary_mask=binary_mask,
        )
    else:
        np.savez(
            path,
            subject_ids=subject_ids,
            labels=labels,
            probabilities=probabilities,
        )


def test_binary_loader_applies_fixed_horizon_mask(tmp_path):
    path = tmp_path / "predictions.npz"
    _write_prediction(
        path,
        np.linspace(0.05, 0.95, 20),
        with_survival_only_row=True,
    )
    frame = load_binary_prediction_artifact(path)
    assert len(frame) == 20
    assert 999 not in set(frame["subject_id"])


def test_binary_loader_rejects_survival_only_artifact(tmp_path):
    path = tmp_path / "predictions.npz"
    np.savez(
        path,
        subject_ids=np.arange(5),
        risk_scores=np.linspace(-1, 1, 5),
        times=np.arange(5),
        events=np.array([0, 1, 0, 1, 0]),
    )
    with pytest.raises(ValueError, match="Risk-score-only"):
        load_binary_prediction_artifact(path)


def test_prediction_parity_fails_on_different_patients():
    model = pd.DataFrame(
        {"subject_id": [1, 2], "label": [0, 1], "probability": [0.2, 0.8]}
    )
    comparator = pd.DataFrame(
        {"subject_id": [1, 3], "label": [0, 1], "probability": [0.3, 0.7]}
    )
    with pytest.raises(ValueError, match="Patient parity"):
        assert_paired_prediction_parity(model, comparator, task_label="test")


def test_discovery_uses_result_metadata_and_rejects_duplicates(tmp_path):
    cell = tmp_path / "cohort" / "outcome" / "model" / "seed_1"
    _write_prediction(cell / "predictions.npz", np.linspace(0.1, 0.9, 20))
    pd.DataFrame(
        [
            {
                "model_family": "opera",
                "cohort": "c",
                "outcome": "o",
                "seed": 1,
                "split": "held_out",
                "evaluation_subset": "full",
                "outcome_window_hours": 720,
            }
        ]
    ).to_csv(cell / "result.csv", index=False)
    discovered = discover_prediction_artifacts(tmp_path)
    assert discovered.loc[0, "prediction_path"].endswith("predictions.npz")
    assert discovered.loc[0, "cohort"] == "c"

    duplicate = tmp_path / "duplicate"
    duplicate.mkdir()
    _write_prediction(duplicate / "predictions.npz", np.linspace(0.1, 0.9, 20))
    pd.read_csv(cell / "result.csv").to_csv(duplicate / "result.csv", index=False)
    with pytest.raises(ValueError, match="Duplicate prediction artifacts"):
        discover_prediction_artifacts(tmp_path)


def _artifact_fixture(tmp_path: Path, n_cells: int = 8) -> pd.DataFrame:
    rows = []
    labels = np.tile([0, 1], 20)
    comparator = np.where(labels == 1, 0.62, 0.38)
    for cell_index in range(n_cells):
        model = np.where(
            labels == 1,
            0.64 + 0.01 * cell_index,
            0.36 - 0.01 * cell_index,
        )
        for family, probabilities in (
            ("opera", model),
            ("tabular_ehr_xgboost", comparator),
        ):
            path = tmp_path / f"cell_{cell_index}" / family / "predictions.npz"
            _write_prediction(path, probabilities)
            rows.append(
                {
                    "model_family": family,
                    "cohort": f"cohort_{cell_index % 2}",
                    "outcome": f"outcome_{cell_index}",
                    "outcome_window_hours": 720,
                    "split": "held_out",
                    "seed": 1,
                    "evaluation_subset": "full",
                    "prediction_path": str(path),
                    "n_events_train": 10 * (cell_index + 1),
                    "n_train": 200 + cell_index,
                    "prevalence_train": 0.1,
                    "outcome_family": "Mortality" if cell_index < 4 else "Toxicity",
                    "cohort_group": "group_a" if cell_index % 2 == 0 else "group_b",
                }
            )
    return pd.DataFrame(rows)


def test_paired_tables_overlap_and_model_preparation(tmp_path):
    artifacts = _artifact_fixture(tmp_path)
    deltas, draws, memberships = build_paired_delta_tables(
        artifacts,
        model_family="opera",
        comparator_family="tabular_ehr_xgboost",
        n_bootstrap=20,
        seed=7,
        metrics=("auroc", "brier_score"),
        min_test_positive=2,
        min_test_negative=2,
    )
    assert deltas["cell_id"].nunique() == 8
    assert set(deltas["metric"]) == {"auroc", "brier_score"}
    assert (deltas["difference"] >= 0).all()
    assert set(deltas["cohort_group"]) == {"group_a", "group_b"}
    auroc = deltas[deltas["metric"] == "auroc"]
    assert set(auroc["difference_se_method"]) == {"paired_delong"}
    assert set(auroc["minority_class"]) == {20}
    assert not draws.empty
    summary, pairs = summarize_patient_overlap(memberships)
    assert summary["fraction_patients_in_multiple_cells"] == 1.0
    assert not pairs.empty

    prepared = prepare_rarity_data(
        deltas,
        metric="auroc",
        spline_knots=4,
        standard_error_floor=0.001,
        grid_size=30,
    )
    assert len(prepared.cells) == 8
    assert prepared.spline_grid.shape[0] == 30
    assert np.isfinite(prepared.spline_cells).all()
    assert set(prepared.cells["cohort_group"]) == {"group_a", "group_b"}
    assert "minority_class" in prepared.cells
    assert not prepared.cells["n_events_train"].equals(
        prepared.cells["minority_class"]
    )
    assert len(prepared.group_of_cohort) == len(prepared.cohorts)
    assert len(prepared.family_of_outcome) == len(prepared.outcomes)
    assert np.all(prepared.group_of_cohort >= 0)
    assert np.all(prepared.group_of_cohort < len(prepared.groups))
    assert np.all(prepared.family_of_outcome >= 0)
    assert np.all(prepared.family_of_outcome < len(prepared.families))
    assert prepared.grid_event_counts.min() == pytest.approx(
        prepared.cells["n_events_train"].min()
    )

    scatter = aggregate_scatter_cells(deltas, metric="auroc")
    assert set(scatter["cohort_group"]) == {"group_a", "group_b"}


def test_small_sample_auroc_uses_bootstrap_se():
    labels = np.array([1] * 5 + [0] * 15)
    paired = pd.DataFrame(
        {
            "subject_id": np.arange(len(labels)),
            "label_model": labels,
            "label_comparator": labels,
            "probability_model": np.linspace(0.95, 0.05, len(labels)),
            "probability_comparator": np.linspace(0.85, 0.15, len(labels)),
        }
    )
    summary, draws = paired_bootstrap_differences(
        paired,
        n_bootstrap=50,
        seed=11,
        metrics=("auroc",),
        small_sample_minority_threshold=10,
    )
    row = summary.iloc[0]
    assert row["difference_se_method"] == "paired_bootstrap_small_sample"
    assert row["minority_class"] == 5
    assert np.isfinite(row["difference_se"])
    assert not draws.empty


def _nested_delta_fixture() -> pd.DataFrame:
    rows = []
    specs = [
        ("group_a", "cohort_a_rich", "outcome_a0", "family_a", 8, 60, 0.09, 0.01),
        ("group_a", "cohort_a_rich", "outcome_a1", "family_a", 10, 60, 0.08, 0.01),
        ("group_a", "cohort_a_rich", "outcome_a2", "family_a", 12, 60, 0.10, 0.01),
        ("group_a", "cohort_a_poor", "outcome_a3", "family_a", 14, 5, -0.01, 0.25),
        ("group_b", "cohort_b0", "outcome_b0", "family_b", 40, 60, -0.10, 0.01),
        ("group_b", "cohort_b0", "outcome_b1", "family_b", 50, 60, -0.09, 0.01),
        ("group_b", "cohort_b1", "outcome_b2", "family_b", 60, 60, -0.11, 0.01),
        ("group_b", "cohort_b1", "outcome_b3", "family_b", 70, 60, -0.08, 0.01),
    ]
    for index, (
        cohort_group,
        cohort,
        outcome,
        family,
        train_events,
        minority_class,
        difference,
        difference_se,
    ) in enumerate(specs):
        rows.append(
            {
                "cell_id": f"{cohort}|{outcome}|720",
                "cohort": cohort,
                "cohort_group": cohort_group,
                "outcome": outcome,
                "outcome_family": family,
                "metric": "auroc",
                "difference": difference,
                "difference_se": difference_se,
                "n_events_train": train_events,
                "minority_class": minority_class,
                "analysis_tier": (
                    "primary" if minority_class >= 25 else "partial_pool_only"
                ),
            }
        )
    return pd.DataFrame(rows)


def test_prepare_rarity_data_keeps_partial_pool_cells():
    deltas = _nested_delta_fixture()
    prepared = prepare_rarity_data(
        deltas,
        metric="auroc",
        spline_knots=4,
        standard_error_floor=0.001,
        fit_tiers=("primary",),
        grid_size=25,
    )
    assert len(prepared.cells) == len(deltas)
    poor = prepared.cells[prepared.cells["cohort"] == "cohort_a_poor"].iloc[0]
    assert poor["minority_class"] == 5
    assert "partial_pool_only" in set(prepared.observations["analysis_tier"])
    assert prepared.grid_event_counts.min() == pytest.approx(8.0)
    poor_cohort = prepared.cohorts.index("cohort_a_poor")
    rich_cohort = prepared.cohorts.index("cohort_a_rich")
    assert prepared.group_of_cohort[poor_cohort] == prepared.group_of_cohort[
        rich_cohort
    ]


def test_prepare_rarity_data_requires_reporting_counts():
    deltas = _nested_delta_fixture().drop(columns=["minority_class"])
    with pytest.raises(ValueError, match="minority_class"):
        prepare_rarity_data(
            deltas.drop(columns=["analysis_tier"]),
            metric="auroc",
            spline_knots=4,
        )


def test_hierarchical_model_samples_nested_hierarchy(tmp_path):
    pytest.importorskip("pymc")
    az = pytest.importorskip("arviz")

    output_root = Path.cwd() / ".pytest_bayes" / tmp_path.name
    prepared = prepare_rarity_data(
        _nested_delta_fixture(),
        metric="auroc",
        spline_knots=4,
        standard_error_floor=0.001,
        grid_size=8,
    )
    try:
        paths = fit_hierarchical_rarity_model(
            prepared,
            output_dir=output_root / "nested",
            draws=2,
            tune=2,
            chains=1,
            cores=1,
            target_accept=0.90,
            random_seed=17,
            prior_scale=0.15,
            random_effect_scale=0.10,
            require_convergence=False,
            nest_outcomes=True,
            min_report_minority=25,
            nuts_max_treedepth=1,
            progressbar=False,
        )
        idata = az.from_netcdf(paths["posterior"])
        assert "sigma_group" in idata.posterior
        assert "sigma_cohort_within" in idata.posterior
        assert "sigma_family" in idata.posterior
        assert "sigma_outcome_within" in idata.posterior
        diagnostics = json.loads(paths["diagnostics_json"].read_text())
        assert diagnostics["n_cohort_groups"] == 2
        assert diagnostics["nest_outcomes"] is True

        cells = pd.read_csv(paths["cells"])
        assert len(cells) == len(prepared.cells)
        poor = cells[cells["cohort"] == "cohort_a_poor"].iloc[0]
        assert bool(poor["reportable_standalone"]) is False
        assert np.isfinite(poor["posterior_median"])
    finally:
        shutil.rmtree(output_root, ignore_errors=True)


def test_hierarchical_model_samples_crossed_outcome_toggle(tmp_path):
    pytest.importorskip("pymc")
    az = pytest.importorskip("arviz")

    output_root = Path.cwd() / ".pytest_bayes" / tmp_path.name
    prepared = prepare_rarity_data(
        _nested_delta_fixture(),
        metric="auroc",
        spline_knots=4,
        standard_error_floor=0.001,
        grid_size=8,
    )
    try:
        paths = fit_hierarchical_rarity_model(
            prepared,
            output_dir=output_root / "crossed",
            draws=2,
            tune=2,
            chains=1,
            cores=1,
            target_accept=0.90,
            random_seed=23,
            prior_scale=0.15,
            random_effect_scale=0.10,
            require_convergence=False,
            nest_outcomes=False,
            nuts_max_treedepth=1,
            progressbar=False,
        )
        idata = az.from_netcdf(paths["posterior"])
        assert "sigma_outcome" in idata.posterior
        assert "sigma_family" not in idata.posterior
        assert "sigma_outcome_within" not in idata.posterior
    finally:
        shutil.rmtree(output_root, ignore_errors=True)


def test_attach_task_metadata_requires_complete_event_counts():
    artifacts = pd.DataFrame([{"cohort": "c", "outcome": "o", "model_family": "m"}])
    metadata = pd.DataFrame(
        [{"cohort": "c", "outcome": "o", "n_events_train": 12, "n_train": 50}]
    )
    attached = attach_task_metadata(artifacts, metadata)
    assert attached.loc[0, "n_events_train"] == 12
    with pytest.raises(ValueError, match="missing"):
        attach_task_metadata(artifacts, metadata.iloc[0:0])


def test_publication_plot_writes_png_pdf_svg(tmp_path):
    rows = []
    for index in range(10):
        rows.append(
            {
                "cell_id": f"c|o{index}|720",
                "cohort": f"c{index % 2}",
                "outcome": f"o{index}",
                "outcome_family": "Mortality" if index < 5 else "Toxicity",
                "cohort_group": f"group{index % 3}",
                "metric": "auroc",
                "difference": -0.01 + 0.004 * index,
                "difference_se": 0.01,
                "n_events_train": 10 * (index + 1),
                "n_test_positive": 10 + index,
                "n_test_negative": 20,
                "analysis_tier": "primary" if index > 1 else "partial_pool_only",
            }
        )
    deltas = pd.DataFrame(rows)
    x = np.geomspace(10, 100, 40)
    median = 0.015 - 0.008 * np.log2(x / 10) / np.log2(10)
    curve = pd.DataFrame(
        {
            "training_events": x,
            "median": median,
            "lower_50": median - 0.005,
            "upper_50": median + 0.005,
            "lower_95": median - 0.012,
            "upper_95": median + 0.012,
            "predictive_lower_95": median - 0.03,
            "predictive_upper_95": median + 0.03,
        }
    )
    target = tmp_path / "figure.png"
    figure = plot_hierarchical_rarity_curve(
        deltas,
        curve,
        max_labels=4,
        save_path=str(target),
    )
    plt.close(figure)
    assert target.exists()
    assert target.with_suffix(".pdf").exists()
    assert target.with_suffix(".svg").exists()
    scatter = aggregate_scatter_cells(deltas, metric="auroc")
    assert len(scatter) == 10


def test_selected_labels_prioritizes_highlight_outcomes():
    # cell_0 is the only instance of "target_outcome" and is also the
    # globally rarest cell. "Rarest evaluable" must not silently drop just
    # because "Key outcome" already claimed that row, and must not duplicate it.
    x = np.geomspace(10, 100, 10)
    curve = pd.DataFrame(
        {"training_events": x, "median": np.linspace(0.015, 0.007, 10)}
    )
    cells = pd.DataFrame(
        {
            "cell_id": [f"cell_{i}" for i in range(10)],
            "cohort": [f"c{i % 2}" for i in range(10)],
            "outcome": ["target_outcome"] + [f"o{i}" for i in range(1, 10)],
            "n_events_train": x,
            "difference": np.linspace(-0.01, 0.026, 10),
        }
    )
    labels = _selected_labels(
        cells,
        curve,
        rarity_column="n_events_train",
        max_labels=4,
        highlight_outcomes=["target_outcome", "outcome_not_present"],
    )
    assert labels["cell_id"].nunique() == len(labels)
    assert labels.loc[
        labels["label_category"] == "Key outcome", "cell_id"
    ].tolist() == ["cell_0"]
    assert "Rarest evaluable" in set(labels["label_category"])
    # cell_0 was already claimed by "Key outcome", so "Rarest evaluable" must
    # fall through to the next-rarest distinct cell, not vanish or duplicate.
    rarest_row = labels[labels["label_category"] == "Rarest evaluable"].iloc[0]
    assert rarest_row["cell_id"] != "cell_0"
    assert "Most data-rich" in set(labels["label_category"])


def test_selected_labels_prioritizes_exact_clinical_cells():
    x = np.geomspace(10, 100, 8)
    curve = pd.DataFrame({"training_events": x, "median": np.zeros(8)})
    cells = pd.DataFrame(
        {
            "cell_id": [f"cell_{i}" for i in range(8)],
            "cohort": ["DLBCL", "MM", "DLBCL", "CLL", "HL", "BL", "FL", "MCL"],
            "outcome": [
                "treatment_failure",
                "treatment_failure",
                *[f"o{i}" for i in range(6)],
            ],
            "n_events_train": x,
            "difference": np.linspace(-0.01, 0.02, 8),
        }
    )
    labels = _selected_labels(
        cells,
        curve,
        rarity_column="n_events_train",
        max_labels=3,
        highlight_cells=[
            {
                "cohort": "DLBCL",
                "outcome": "treatment_failure",
                "label": "DLBCL × treatment failure",
            }
        ],
    )
    first = labels.iloc[0]
    assert first["cell_id"] == "cell_0"
    assert first["annotation_label"] == "DLBCL × treatment failure"
    assert first["label_category"] == "Clinical anchor"


def test_outcome_family_mapping_is_deduplicated_and_manageable(monkeypatch, tmp_path):
    from opera.run.hierarchical_rarity import _load_config

    monkeypatch.setenv("BONSAI_RESULTS_ROOT", str(tmp_path / "results"))

    config_path = (
        Path(__file__).resolve().parents[1]
        / "opera"
        / "configs"
        / "hierarchical_rarity.yaml"
    )
    config = _load_config(config_path)
    mapping = config["outcome_families"]
    assert len(mapping) == len(set(mapping))
    families = set(mapping.values())
    # A handful of clinically coherent families, not one row per outcome and
    # not an undifferentiated "lab values" catch-all.
    assert 1 < len(families) <= 12
    assert "overall_survival" in mapping and "treatment_failure" in mapping
    assert "Organ & metabolic toxicity" not in families


def test_outcome_family_mapping_can_fail_closed():
    frame = pd.DataFrame({"outcome": ["mapped", "new_unmapped_outcome"]})
    with pytest.raises(ValueError, match="new_unmapped_outcome"):
        apply_outcome_families(
            frame,
            {"mapped": "Clinical family"},
            require_complete=True,
        )


def test_overlap_summary_does_not_export_patient_ids():
    summary, pairs = summarize_patient_overlap(
        {"a": np.array([1, 2, 3]), "b": np.array([2, 3, 4])}
    )
    assert summary["n_unique_test_patients"] == 4
    assert pairs.loc[0, "n_overlap"] == 2
    assert "subject_id" not in json.dumps(summary)
