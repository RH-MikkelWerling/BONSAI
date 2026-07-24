"""Tests for the typed sweep orchestration primitives.

Covers the pure argv builders in :mod:`opera.run.sweep_commands` and the typed
status records / tracker in :mod:`opera.run.sweep_types`.
"""

import json
import sys
from pathlib import Path

import pandas as pd

from opera.run.sweep import (
    _run_ipi_baseline_cell,
    _run_variant_cell,
    prepare_ipi_subset_predictions,
)
from opera.run.sweep_commands import (
    build_evaluate_cmd,
    build_finetune_cmd,
    build_prediction_evaluate_cmd,
)
from opera.run.sweep_types import (
    VALID_STAGES,
    VALID_STATUSES,
    StatusTracker,
    SweepCellRecord,
)


# ── build_finetune_cmd ──────────────────────────────────────────────────────


def test_build_finetune_cmd_minimal_shape():
    cmd = build_finetune_cmd(
        encoder_ckpt="/ckpt/best.ckpt",
        encoder_source="contrastive",
        cohort="dlbcl",
        cohort_data_dir="/data/dlbcl",
        outcome_name="mortality_1y",
        outcome_path="/data/dlbcl/outcomes/mortality_1y.parquet",
        output_dir=Path("/results/cell"),
        base_config="opera/configs/finetune.yaml",
    )
    assert isinstance(cmd, list)
    assert all(isinstance(part, str) for part in cmd)
    assert cmd[0] == sys.executable
    assert cmd[1] == "-m"


def test_build_finetune_cmd_default_module_and_config():
    cmd = build_finetune_cmd(
        encoder_ckpt="/ckpt/best.ckpt",
        encoder_source="contrastive",
        cohort="dlbcl",
        cohort_data_dir="/data/dlbcl",
        outcome_name="mortality_1y",
        outcome_path="/data/dlbcl/outcomes/mortality_1y.parquet",
        output_dir=Path("/results/cell"),
        base_config="opera/configs/finetune.yaml",
    )
    assert cmd[2] == "opera.run.finetune"
    assert cmd[3] == "--config-name=finetune"


def test_build_finetune_cmd_survival_module_and_config():
    for mode in ("cox", "ipcw_bce", "ipcw_cif_bce"):
        cmd = build_finetune_cmd(
            encoder_ckpt="/ckpt/best.ckpt",
            encoder_source="contrastive",
            cohort="dlbcl",
            cohort_data_dir="/data/dlbcl",
            outcome_name="mortality_1y",
            outcome_path="/data/dlbcl/outcomes/mortality_1y.parquet",
            output_dir=Path("/results/cell"),
            base_config="opera/configs/finetune.yaml",
            training_mode=mode,
        )
        assert cmd[2] == "opera.run.survival_finetune"
        assert cmd[3] == "--config-name=survival_finetune"
        assert f"training_mode={mode}" in cmd


def test_build_finetune_cmd_none_values_serialize_as_null():
    cmd = build_finetune_cmd(
        encoder_ckpt="null",
        encoder_source="random_init",
        cohort="dlbcl",
        cohort_data_dir="/data/dlbcl",
        outcome_name="mortality_1y",
        outcome_path="/data/dlbcl/outcomes/mortality_1y.parquet",
        output_dir=Path("/results/cell"),
        base_config="opera/configs/finetune.yaml",
        n_hours_end_include=None,
        registry_start_date=None,
    )
    assert "labels.n_hours_end_include=null" in cmd
    assert "labels.registry_start_date=null" in cmd
    # No literal Python "None" leaks into the argv.
    assert not any("None" in part for part in cmd)


def test_build_finetune_cmd_optional_paths_only_when_set():
    base_kwargs = dict(
        encoder_ckpt="/ckpt/best.ckpt",
        encoder_source="contrastive",
        cohort="dlbcl",
        cohort_data_dir="/data/dlbcl",
        outcome_name="mortality_1y",
        outcome_path="/data/dlbcl/outcomes/mortality_1y.parquet",
        output_dir=Path("/results/cell"),
        base_config="opera/configs/finetune.yaml",
    )
    without = build_finetune_cmd(**base_kwargs)
    assert not any(part.startswith("paths.competing_outcome=") for part in without)
    assert not any(part.startswith("paths.eligibility=") for part in without)

    with_paths = build_finetune_cmd(
        **base_kwargs,
        competing_outcome_path="/data/dlbcl/outcomes/competing.parquet",
        eligibility_path="/data/dlbcl/eligibility.parquet",
    )
    assert (
        "paths.competing_outcome=/data/dlbcl/outcomes/competing.parquet" in with_paths
    )
    assert "paths.eligibility=/data/dlbcl/eligibility.parquet" in with_paths


def test_build_finetune_cmd_extra_overrides_appended():
    cmd = build_finetune_cmd(
        encoder_ckpt="/ckpt/best.ckpt",
        encoder_source="contrastive",
        cohort="dlbcl",
        cohort_data_dir="/data/dlbcl",
        outcome_name="mortality_1y",
        outcome_path="/data/dlbcl/outcomes/mortality_1y.parquet",
        output_dir=Path("/results/cell"),
        base_config="opera/configs/finetune.yaml",
        extra_overrides=["seed=7", "trainer.max_epochs=3"],
    )
    assert "seed=7" in cmd
    assert "trainer.max_epochs=3" in cmd


# ── build_evaluate_cmd ──────────────────────────────────────────────────────


def test_build_evaluate_cmd_minimal_shape():
    cmd = build_evaluate_cmd(
        ckpt_path=Path("/results/cell/best.ckpt"),
        cohort="dlbcl",
        cohort_data_dir="/data/dlbcl",
        outcome_name="mortality_1y",
        outcome_path="/data/dlbcl/outcomes/mortality_1y.parquet",
        output_dir=Path("/results/cell"),
    )
    assert isinstance(cmd, list)
    assert all(isinstance(part, str) for part in cmd)
    assert cmd[0] == sys.executable
    assert cmd[1] == "-m"
    assert cmd[2] == "opera.run.evaluate"
    assert "run_dir=/results/cell" in cmd
    assert not any(part.startswith("ckpt_path=") for part in cmd)


def test_build_evaluate_cmd_joint_module_prepends_outcome_name():
    cmd = build_evaluate_cmd(
        ckpt_path=Path("/ckpt/joint.ckpt"),
        cohort="dlbcl",
        cohort_data_dir="/data/dlbcl",
        outcome_name="mortality_1y",
        outcome_path="/data/dlbcl/outcomes/mortality_1y.parquet",
        output_dir=Path("/results/cell"),
        encoder_source="joint",
    )
    assert cmd[2] == "opera.run.evaluate_joint"
    # outcome_name override is prepended ahead of the standard overrides.
    assert cmd[3] == "outcome_name=mortality_1y"
    assert "ckpt_path=/ckpt/joint.ckpt" in cmd


def test_build_evaluate_cmd_none_values_serialize_as_null():
    cmd = build_evaluate_cmd(
        ckpt_path=Path("/results/cell/best.ckpt"),
        cohort="dlbcl",
        cohort_data_dir="/data/dlbcl",
        outcome_name="mortality_1y",
        outcome_path="/data/dlbcl/outcomes/mortality_1y.parquet",
        output_dir=Path("/results/cell"),
        n_hours_end_include=None,
        registry_start_date=None,
    )
    assert "labels.n_hours_end_include=null" in cmd
    assert "labels.registry_start_date=null" in cmd
    assert not any("None" in part for part in cmd)


def test_build_evaluate_cmd_encoder_frozen_lowercased():
    cmd = build_evaluate_cmd(
        ckpt_path=Path("/results/cell/best.ckpt"),
        cohort="dlbcl",
        cohort_data_dir="/data/dlbcl",
        outcome_name="mortality_1y",
        outcome_path="/data/dlbcl/outcomes/mortality_1y.parquet",
        output_dir=Path("/results/cell"),
        encoder_frozen=True,
        head_type="linear_probe",
    )
    assert "encoder_frozen=true" in cmd
    assert "head_type=linear_probe" in cmd


def test_build_evaluate_cmd_cohort_fine_requires_both():
    only_col = build_evaluate_cmd(
        ckpt_path=Path("/results/cell/best.ckpt"),
        cohort="dlbcl",
        cohort_data_dir="/data/dlbcl",
        outcome_name="mortality_1y",
        outcome_path="/data/dlbcl/outcomes/mortality_1y.parquet",
        output_dir=Path("/results/cell"),
        cohort_fine_col="histology",
    )
    assert not any(part.startswith("cohort_fine_col=") for part in only_col)

    both = build_evaluate_cmd(
        ckpt_path=Path("/results/cell/best.ckpt"),
        cohort="dlbcl",
        cohort_data_dir="/data/dlbcl",
        outcome_name="mortality_1y",
        outcome_path="/data/dlbcl/outcomes/mortality_1y.parquet",
        output_dir=Path("/results/cell"),
        cohort_fine_col="histology",
        cohort_fine_value="gcb",
    )
    assert "cohort_fine_col=histology" in both
    assert "cohort_fine_value=gcb" in both


# ── build_prediction_evaluate_cmd ───────────────────────────────────────────


def test_build_prediction_evaluate_cmd_minimal_shape():
    cmd = build_prediction_evaluate_cmd(
        predictions_path="/results/preds.csv",
        cohort="dlbcl",
        outcome_name="mortality_1y",
        outcome_path="/data/dlbcl/outcomes/mortality_1y.parquet",
        model_family="tabular_ehr",
        output_dir=Path("/results/cell"),
        n_hours_start_include=1,
        n_hours_end_include=None,
    )
    assert isinstance(cmd, list)
    assert all(isinstance(part, str) for part in cmd)
    assert cmd[0] == sys.executable
    assert cmd[1] == "-m"
    assert cmd[2] == "opera.run.evaluate_predictions"
    assert "--evaluation_subset" in cmd


def test_build_prediction_evaluate_cmd_end_window_omitted_when_none():
    cmd = build_prediction_evaluate_cmd(
        predictions_path="/results/preds.csv",
        cohort="dlbcl",
        outcome_name="mortality_1y",
        outcome_path="/data/dlbcl/outcomes/mortality_1y.parquet",
        model_family="tabular_ehr",
        output_dir=Path("/results/cell"),
        n_hours_start_include=1,
        n_hours_end_include=None,
    )
    assert "--n_hours_end_include" not in cmd

    cmd_with = build_prediction_evaluate_cmd(
        predictions_path="/results/preds.csv",
        cohort="dlbcl",
        outcome_name="mortality_1y",
        outcome_path="/data/dlbcl/outcomes/mortality_1y.parquet",
        model_family="tabular_ehr",
        output_dir=Path("/results/cell"),
        n_hours_start_include=1,
        n_hours_end_include=8760,
    )
    assert "--n_hours_end_include" in cmd_with
    idx = cmd_with.index("--n_hours_end_include")
    assert cmd_with[idx + 1] == "8760"


def test_build_prediction_evaluate_cmd_optional_flags():
    cmd = build_prediction_evaluate_cmd(
        predictions_path="/results/preds.csv",
        cohort="dlbcl",
        outcome_name="mortality_1y",
        outcome_path="/data/dlbcl/outcomes/mortality_1y.parquet",
        model_family="ipi",
        output_dir=Path("/results/cell"),
        n_hours_start_include=1,
        n_hours_end_include=8760,
        competing_outcome_path="/data/dlbcl/outcomes/competing.parquet",
        eligibility_path="/data/dlbcl/eligibility.parquet",
        registry_start_date="2010-01-01",
        baseline_model="logistic",
        ipi_coverage=0.83,
        evaluation_subset="ipi_complete",
        seed=7,
        subgroup_path="/data/dlbcl/subgroups.parquet",
        subgroup_columns=["sex", "age_band"],
        population_path="/data/dlbcl/population_full.csv",
        cohort_fine_col="diagnosis",
        cohort_fine_value="AML",
        evaluation_subjects_path="/results/subject_ids.csv",
        evaluation_regime="survival",
        risk_col="risk_score",
    )
    assert "--competing_outcome" in cmd
    assert "--eligibility" in cmd
    assert "--registry_start_date" in cmd
    assert "--baseline_model" in cmd
    assert "--ipi_coverage" in cmd
    assert cmd[cmd.index("--ipi_coverage") + 1] == "0.83"
    assert cmd[cmd.index("--seed") + 1] == "7"
    assert cmd[cmd.index("--evaluation_subset") + 1] == "ipi_complete"
    assert "--subgroups" in cmd
    assert cmd[cmd.index("--subgroup_columns") + 1] == "sex,age_band"
    assert cmd[cmd.index("--population") + 1].endswith("population_full.csv")
    assert cmd[cmd.index("--cohort_fine_col") + 1] == "diagnosis"
    assert cmd[cmd.index("--cohort_fine_value") + 1] == "AML"
    assert cmd[cmd.index("--evaluation_subjects") + 1].endswith("subject_ids.csv")
    assert cmd[cmd.index("--evaluation_regime") + 1] == "survival"
    assert cmd[cmd.index("--risk_col") + 1] == "risk_score"


def test_build_prediction_evaluate_cmd_optional_flags_absent_when_unset():
    cmd = build_prediction_evaluate_cmd(
        predictions_path="/results/preds.csv",
        cohort="dlbcl",
        outcome_name="mortality_1y",
        outcome_path="/data/dlbcl/outcomes/mortality_1y.parquet",
        model_family="tabular_ehr",
        output_dir=Path("/results/cell"),
        n_hours_start_include=1,
        n_hours_end_include=None,
    )
    for flag in (
        "--competing_outcome",
        "--eligibility",
        "--registry_start_date",
        "--baseline_model",
        "--ipi_coverage",
        "--subgroups",
        "--subgroup_columns",
    ):
        assert flag not in cmd
    # subgroups require both path and columns
    only_path = build_prediction_evaluate_cmd(
        predictions_path="/results/preds.csv",
        cohort="dlbcl",
        outcome_name="mortality_1y",
        outcome_path="/data/dlbcl/outcomes/mortality_1y.parquet",
        model_family="tabular_ehr",
        output_dir=Path("/results/cell"),
        n_hours_start_include=1,
        n_hours_end_include=None,
        subgroup_path="/data/dlbcl/subgroups.parquet",
    )
    assert "--subgroups" not in only_path


def test_prepare_ipi_subset_predictions_respects_fine_cohort(tmp_path):
    population = pd.DataFrame(
        {
            "subject_id": [1, 2, 3],
            "cohort_fine": ["DLBCL", "FL", "DLBCL"],
            "nccn_ipi": [3, 5, 1],
        }
    )
    outcomes = pd.DataFrame(
        {
            "subject_id": [1, 2, 3],
            "split": ["held_out", "held_out", "held_out"],
            "index_date": pd.to_datetime(["2020-01-01"] * 3),
            "outcome_date": pd.to_datetime([None, None, None]),
            "censor_date": pd.to_datetime(["2021-01-01"] * 3),
        }
    )
    population_path = tmp_path / "population_full.csv"
    outcome_path = tmp_path / "outcome.parquet"
    population.to_csv(population_path, index=False)
    outcomes.to_parquet(outcome_path)

    prediction_path, coverage, subject_ids = prepare_ipi_subset_predictions(
        population_csv=str(population_path),
        outcome_parquet=str(outcome_path),
        ipi_score_col="nccn_ipi",
        output_dir=tmp_path / "ipi",
        cohort_fine_col="cohort_fine",
        cohort_fine_value="DLBCL",
    )

    assert prediction_path is not None
    assert coverage == 1.0
    assert subject_ids == {1, 3}
    written = pd.read_csv(prediction_path)
    assert set(written["subject_id"]) == {1, 3}


# ── SweepCellRecord ─────────────────────────────────────────────────────────


def test_sweep_cell_record_to_dict_full_schema():
    record = SweepCellRecord(
        cohort="dlbcl",
        outcome="mortality_1y",
        variant="opera",
        seed=42,
        stage="evaluate",
        status="success",
        output_dir="/results/cell",
    )
    d = record.to_dict()
    assert d["cohort"] == "dlbcl"
    assert d["timestamp"] == 0.0
    # Optional fields present even when unset.
    for key in ("artifact", "reason", "evaluation_subset", "training_fraction"):
        assert key in d
        assert d[key] is None


def test_sweep_cell_record_from_kwargs_ignores_unknown():
    record = SweepCellRecord.from_kwargs(
        cohort="dlbcl",
        outcome="mortality_1y",
        variant="opera",
        seed=42,
        stage="evaluate",
        status="success",
        output_dir="/results/cell",
        artifact="/results/cell/metrics.json",
        completely_unknown_field="ignored",
        another_extra=123,
    )
    assert record.artifact == "/results/cell/metrics.json"
    assert not hasattr(record, "completely_unknown_field")


def test_sweep_cell_record_is_frozen():
    record = SweepCellRecord(
        cohort="dlbcl",
        outcome="mortality_1y",
        variant="opera",
        seed=42,
        stage="evaluate",
        status="success",
        output_dir="/results/cell",
    )
    try:
        record.status = "failed"  # type: ignore[misc]
    except Exception as exc:
        assert "cannot assign" in str(exc).lower() or "frozen" in str(exc).lower()
    else:
        raise AssertionError("SweepCellRecord should be immutable")


def test_valid_stages_and_statuses_constants():
    assert "evaluate" in VALID_STAGES
    assert "evaluate_predictions" in VALID_STAGES
    assert isinstance(VALID_STAGES, frozenset)
    assert VALID_STATUSES == frozenset({"success", "failed", "skipped", "dry_run"})


# ── StatusTracker ───────────────────────────────────────────────────────────


def test_status_tracker_append_stamps_timestamp_and_returns_record():
    tracker = StatusTracker()
    record = tracker.append(
        cohort="dlbcl",
        outcome="mortality_1y",
        variant="opera",
        seed=42,
        stage="evaluate",
        status="success",
        output_dir="/results/cell",
    )
    assert isinstance(record, SweepCellRecord)
    assert record.timestamp > 0.0
    assert len(tracker) == 1


def test_status_tracker_counts():
    tracker = StatusTracker()
    common = dict(
        cohort="dlbcl",
        outcome="mortality_1y",
        variant="opera",
        seed=42,
        output_dir="/results/cell",
    )
    tracker.append(stage="evaluate", status="success", **common)
    tracker.append(stage="evaluate", status="success", **common)
    tracker.append(stage="evaluate", status="failed", **common)
    tracker.append(stage="configuration", status="skipped", **common)
    assert tracker.n_success() == 2
    assert tracker.n_failed() == 1
    assert tracker.n_skipped() == 1


def test_status_tracker_to_dataframe():
    tracker = StatusTracker()
    tracker.append(
        cohort="dlbcl",
        outcome="mortality_1y",
        variant="opera",
        seed=42,
        stage="evaluate",
        status="success",
        output_dir="/results/cell",
    )
    frame = tracker.to_dataframe()
    assert isinstance(frame, pd.DataFrame)
    assert len(frame) == 1
    assert "status" in frame.columns
    assert frame.iloc[0]["cohort"] == "dlbcl"


def test_status_tracker_write_creates_csv_and_jsonl(tmp_path):
    tracker = StatusTracker()
    tracker.append(
        cohort="dlbcl",
        outcome="mortality_1y",
        variant="opera",
        seed=42,
        stage="evaluate",
        status="success",
        output_dir="/results/cell",
        artifact="/results/cell/metrics.json",
    )
    tracker.append(
        cohort="dlbcl",
        outcome="mortality_1y",
        variant="ipi",
        seed=7,
        stage="evaluate_predictions",
        status="failed",
        output_dir="/results/ipi",
        reason="no metrics",
        evaluation_subset="ipi_complete",
    )
    tracker.write(tmp_path)

    csv_path = tmp_path / "sweep_cell_status.csv"
    jsonl_path = tmp_path / "sweep_cell_status.jsonl"
    assert csv_path.exists()
    assert jsonl_path.exists()

    df = pd.read_csv(csv_path)
    assert len(df) == 2
    assert set(["cohort", "outcome", "variant", "seed", "stage", "status"]).issubset(
        df.columns
    )

    lines = jsonl_path.read_text().strip().splitlines()
    assert len(lines) == 2
    first = json.loads(lines[0])
    assert first["variant"] == "opera"
    assert first["artifact"] == "/results/cell/metrics.json"
    second = json.loads(lines[1])
    assert second["reason"] == "no metrics"
    assert second["evaluation_subset"] == "ipi_complete"


def test_status_tracker_write_noop_when_empty(tmp_path):
    tracker = StatusTracker()
    tracker.write(tmp_path)
    assert not (tmp_path / "sweep_cell_status.csv").exists()
    assert not (tmp_path / "sweep_cell_status.jsonl").exists()


# ── _run_ipi_baseline_cell ──────────────────────────────────────────────────


def _ipi_cell_kwargs(tmp_path, **overrides):
    kwargs = dict(
        cohort_name="dlbcl",
        outcome_name="mortality_1y",
        outcome_cfg={"n_hours_start_include": 1, "n_hours_end_include": None},
        ipi_col=None,
        pop_file=str(tmp_path / "population_full.csv"),
        data_dir=str(tmp_path / "data"),
        output_dir=tmp_path / "out",
        seeds=[42],
        cfg={},
        dry_run=False,
        fail_fast=False,
        rarity_mode="none",
        baseline_model=None,
        subgroup_path=None,
        subgroup_columns=None,
    )
    kwargs.update(overrides)
    return kwargs


def test_run_ipi_baseline_cell_dry_run_appends_status(tmp_path):
    tracker = StatusTracker()
    results = _run_ipi_baseline_cell(
        tracker=tracker,
        **_ipi_cell_kwargs(tmp_path, ipi_col=None, dry_run=True),
    )
    assert results == []
    assert len(tracker) == 1
    assert tracker.records[0].status == "dry_run"
    assert tracker.records[0].variant == "ipi"


def test_run_ipi_baseline_cell_no_ipi_col_returns_empty(tmp_path):
    tracker = StatusTracker()
    results = _run_ipi_baseline_cell(
        tracker=tracker,
        **_ipi_cell_kwargs(tmp_path, ipi_col=None, dry_run=False),
    )
    assert results == []
    # No IPI column means no IPI baseline; nothing is recorded.
    assert len(tracker) == 0


# ── _run_variant_cell ───────────────────────────────────────────────────────


def _variant_cell_kwargs(tmp_path, variant_cfg, **overrides):
    kwargs = dict(
        cohort_name="dlbcl",
        outcome_name="mortality_1y",
        outcome_cfg={"n_hours_start_include": 1, "n_hours_end_include": None},
        variant_name="tabular_ehr",
        variant_cfg=variant_cfg,
        result_variant=variant_cfg.get("model_family", "tabular_ehr"),
        seed=42,
        cohort_fine_col=None,
        cohort_fine_value=None,
        data_dir=str(tmp_path / "data"),
        output_dir=tmp_path / "out",
        base_config="opera/configs/finetune.yaml",
        cfg={},
        dry_run=False,
        overwrite=False,
        fail_fast=False,
        rarity_mode="none",
        baseline_model=None,
        ipi_col=None,
        pop_file=str(tmp_path / "population_full.csv"),
        subgroup_path=None,
        subgroup_columns=None,
    )
    kwargs.update(overrides)
    return kwargs


def test_run_variant_cell_dry_run_results_file(tmp_path):
    tracker = StatusTracker()
    result = _run_variant_cell(
        tracker=tracker,
        **_variant_cell_kwargs(
            tmp_path,
            variant_cfg={"results_file": "/fake/path.json"},
            dry_run=True,
        ),
    )
    assert result is None
    assert len(tracker) == 1
    assert tracker.records[0].status == "dry_run"
    assert tracker.records[0].stage == "results_file"


def test_run_variant_cell_dry_run_predictions_file(tmp_path):
    tracker = StatusTracker()
    result = _run_variant_cell(
        tracker=tracker,
        **_variant_cell_kwargs(
            tmp_path,
            variant_cfg={"predictions_file": "/fake/pred.csv"},
            dry_run=True,
        ),
    )
    assert result is None
    assert len(tracker) == 1
    assert tracker.records[0].status == "dry_run"
    assert tracker.records[0].stage == "evaluate_predictions"


def test_run_variant_cell_missing_results_file_records_failure(tmp_path):
    tracker = StatusTracker()
    missing = str(tmp_path / "does_not_exist.json")
    result = _run_variant_cell(
        tracker=tracker,
        **_variant_cell_kwargs(
            tmp_path,
            variant_cfg={"results_file": missing},
            dry_run=False,
            fail_fast=False,
        ),
    )
    assert result is None
    assert len(tracker) == 1
    assert tracker.records[0].status == "failed"
    assert tracker.records[0].reason == "results_file not found"


def test_run_variant_cell_invalid_training_mode_records_failure(tmp_path):
    tracker = StatusTracker()
    result = _run_variant_cell(
        tracker=tracker,
        **_variant_cell_kwargs(
            tmp_path,
            variant_cfg={
                "encoder_ckpt": "/ckpt/best.ckpt",
                "training_mode": "invalid_mode",
            },
            variant_name="opera",
            result_variant="opera",
            dry_run=False,
            fail_fast=False,
        ),
    )
    assert result is None
    assert len(tracker) == 1
    assert tracker.records[0].status == "failed"
    assert tracker.records[0].stage == "configuration"
    assert "invalid_mode" in tracker.records[0].reason


def test_run_variant_cell_finetune_population_is_fine_cohort_only(
    tmp_path, monkeypatch
):
    """A fine-level sweep cell must finetune on the fine cohort alone.

    ``_run_variant_cell`` has no clinical-group/grouped-parent parameter at
    all (it was removed because it never affected population selection), so
    this test locks in the actual wiring: the population-selecting arguments
    reaching ``run_finetune`` are exactly ``cohort_name``/``cohort_fine_value``
    for this cell — never a grouped parent name.
    """
    import opera.run.sweep as sweep_module

    captured = {}

    def fake_run_finetune(**kwargs):
        captured.update(kwargs)
        return None

    monkeypatch.setattr(sweep_module, "run_finetune", fake_run_finetune)

    tracker = StatusTracker()
    result = _run_variant_cell(
        tracker=tracker,
        **_variant_cell_kwargs(
            tmp_path,
            variant_cfg={"encoder_ckpt": "/ckpt/best.ckpt", "training_mode": "cox"},
            variant_name="opera",
            result_variant="opera",
            cohort_name="RT",
            cohort_fine_col="cohort_fine",
            cohort_fine_value="RT",
            dry_run=False,
            fail_fast=False,
        ),
    )

    assert result is None  # fake run_finetune reports no checkpoint produced
    assert captured["cohort"] == "RT"
    assert captured["cohort_fine_value"] == "RT"
    assert "overwrite=false" in captured["extra_overrides"]
    assert "DLBCL_like" not in captured.values()
