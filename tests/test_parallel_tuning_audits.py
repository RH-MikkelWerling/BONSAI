import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import zipfile

import pytest
import yaml

spec = importlib.util.spec_from_file_location("parallel_audits",
    Path(__file__).resolve().parents[1] / "scripts/run_parallel_tuning_audits.py")
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def setup(tmp_path, split="tuning"):
    source = tmp_path / "analysis"
    (source / "analysis_inventory").mkdir(parents=True)
    (source / "analysis_inventory/analysis.yaml").write_text(yaml.safe_dump(dict(
        estimand="cif_ipcw", artifacts=[dict(model="R_L", split=split)])))
    (source / "clinical_scores").mkdir()
    for name in runner.CLINICAL_FILES:
        (source / "clinical_scores" / name).write_text("aggregate\n")
    (source / "clinical_scores/patient_metadata.parquet").write_text("never copy")
    (source / "configs").mkdir()
    (source / "configs/config_audit.json").write_text("{}")
    (source / "completion.json").write_text('{"status":"running"}')
    return source


@pytest.mark.parametrize("fail_reconciliation", [False, True])
def test_independent_jobs_continue_and_only_aggregates_are_packaged(tmp_path, monkeypatch, fail_reconciliation):
    source = setup(tmp_path)
    output = tmp_path / "parallel"
    before = {p.relative_to(source): p.read_bytes() for p in source.rglob("*") if p.is_file()}
    calls = []

    def execute(command, **kwargs):
        calls.append(command)
        module = command[3]
        if module.endswith("audit_tuning_router"):
            kwargs["stdout"].write('{"metric_reconciliation": []}\n')
            if fail_reconciliation:
                return SimpleNamespace(returncode=1)
        elif module.endswith("audit_router_vectors"):
            Path(command[-1]).write_text("{}")
        elif module.endswith("audit_frozen_router_mechanism"):
            target = output / "mechanism"
            target.mkdir()
            (target / "references_INTERNAL.pt").write_text("never package")
            with zipfile.ZipFile(target / "router_mechanism_review.zip", "w") as archive:
                archive.writestr("report.json", "{}")
        elif module.endswith("evaluate_whole_cohort_outcomes"):
            target = output / "whole_cohort"
            target.mkdir()
            for name in runner.WHOLE_COHORT_FILES:
                (target / name).write_text("aggregate\n")
        assert kwargs["env"]["OPERA_ANALYSIS_WORKERS"] == "2"
        assert kwargs["env"]["CUDA_VISIBLE_DEVICES"] == ""
        assert "--verify-artifacts" not in command
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(runner.subprocess, "run", execute)
    report = runner.run(source, output)
    assert len(calls) == 4
    assert report["status"] == ("finished_with_gaps" if fail_reconciliation else "complete")
    assert report["steps"]["whole_cohort"]["status"] == "complete"
    assert before == {p.relative_to(source): p.read_bytes() for p in source.rglob("*") if p.is_file()}
    with zipfile.ZipFile(output / "parallel_audits_review.zip") as archive:
        assert "clinical/score_inventory.csv" in archive.namelist()
        assert "mechanism/router_mechanism_review.zip" in archive.namelist()
        assert not any("INTERNAL" in name or "parquet" in name or "logs/" in name
                       for name in archive.namelist())
        assert json.loads(archive.read("parallel_audits.json"))["status"] == report["status"]


def test_refuses_heldout_before_any_outputs_or_subprocess(tmp_path, monkeypatch):
    source = setup(tmp_path, split="held_out")
    monkeypatch.setattr(runner.subprocess, "run", lambda *a, **kw: pytest.fail("No subprocess"))
    output = tmp_path / "parallel"
    with pytest.raises(ValueError, match="tuning-only"):
        runner.run(source, output)
    assert not output.exists()


def test_missing_clinical_reports_are_explicit(tmp_path):
    source = setup(tmp_path)
    (source / "clinical_scores/cutoff_diagnostics.csv").unlink()
    receipt = runner.snapshot_clinical(source, tmp_path / "snapshot")
    assert receipt["missing"] == ["cutoff_diagnostics.csv"]
    assert not (tmp_path / "snapshot/patient_metadata.parquet").exists()


def test_clinical_snapshot_io_failure_does_not_block_other_audits(tmp_path, monkeypatch):
    source = setup(tmp_path)
    def fail_snapshot(*args):
        raise OSError("Cannot copy one report")
    calls = []
    def execute(command, **kwargs):
        calls.append(command)
        return SimpleNamespace(returncode=1)
    monkeypatch.setattr(runner, "snapshot_clinical", fail_snapshot)
    monkeypatch.setattr(runner.subprocess, "run", execute)
    output = tmp_path / "parallel"
    report = runner.run(source, output)
    assert report["status"] == "finished_with_gaps"
    assert report["steps"]["clinical_snapshot"]["status"] == "failed"
    assert len(calls) == 4
    assert (output / "parallel_audits_review.zip").exists()
