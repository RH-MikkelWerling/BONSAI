import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml


spec = importlib.util.spec_from_file_location(
    "resume_full_rehearsal", Path(__file__).resolve().parents[1] / "scripts/resume_full_tuning_rehearsal.py")
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def setup_run(tmp_path):
    source = tmp_path / "source"
    root = tmp_path / "full_tuning_rehearsal_20261029_234123"
    for directory in (source / "analysis_inventory", root / "paired/analysis_inventory", root / "analysis"):
        directory.mkdir(parents=True)
    inventory = yaml.safe_dump(dict(estimand="cif_ipcw", artifacts=[
        dict(model="R_L", split="tuning", path="/never/read/patient/predictions")]))
    (source / "analysis_inventory/analysis.yaml").write_text(inventory)
    (root / "paired/analysis_inventory/analysis.yaml").write_text(inventory)
    config, checkpoint = tmp_path / "fit.yaml", tmp_path / "best.ckpt"
    config.write_text("frozen config")
    checkpoint.write_text("frozen checkpoint")
    fit = dict(config=str(config), checkpoint=str(checkpoint),
               config_sha256=runner.digest(config), checkpoint_sha256=runner.digest(checkpoint))
    (source / "completion.json").write_text(json.dumps(dict(inputs=dict(fit=fit))))
    (root / "analysis/completion.json").write_text('{"steps": {"borrowing_analysis_zero": {"status": "running"}}}')
    (root / "analysis/overnight.env").write_text(f"OPERA_PAPER_CELL_METADATA={tmp_path}/cells.csv\n")
    report = dict(status="running", source_analysis=str(source),
                  inventory_sha256=runner.digest(source / "analysis_inventory/analysis.yaml"),
                  prepared_inventory_sha256=runner.digest(root / "paired/analysis_inventory/analysis.yaml"),
                  paths={key: str(root / name) for key, name in
                         (("paired", "paired"), ("analysis", "analysis"), ("ig", "ig"), ("review", "review"))},
                  steps={name: dict(status=status) for name, status in
                         (("router_replay", "complete"), ("paired_analysis", "complete"), ("complete_figures", "running"))})
    (root / "full_rehearsal.json").write_text(json.dumps(report))
    return source, root, fit


@pytest.mark.parametrize("resume_ig", [False, True])
@pytest.mark.parametrize("downstream_complete", [False, True])
def test_resumes_nested_completion_and_finishes_without_repeating_completed_steps(
        tmp_path, monkeypatch, resume_ig, downstream_complete):
    source, root, fit = setup_run(tmp_path)
    if resume_ig:
        (root / "ig").mkdir()
        (root / "ig/refresh_manifest.json").write_text(json.dumps(dict(fit=fit)))
        (root / "ig/overnight.env").write_text("")
    # An interrupted fresh-only writer is preserved and retried.
    (root / "whole_cohort").mkdir()
    (root / "whole_cohort/partial.txt").write_text("keep for diagnosis")
    (root / "supplement_tables").mkdir()
    if downstream_complete:
        path = root / "full_rehearsal.json"
        report = json.loads(path.read_text())
        for name in ("plot_tables", "paper_numbers", "supplement_tables", "assemble", "package"):
            report["steps"][name] = dict(status="complete")
        path.write_text(json.dumps(report))
        (root / "review").mkdir()
        (root / "review/old.txt").write_text("stale review")
        (root / "paper_review.zip").write_bytes(b"stale review archive")
    calls = []

    def execute(command, **kwargs):
        calls.append(command)
        module = command[2] if command[1] == "-m" else "resume_ig"
        if module.endswith("complete_local_figures"):
            assert "--resume" in command
        elif module.endswith("evaluate_whole_cohort_outcomes"):
            (root / "whole_cohort").mkdir(exist_ok=False)
        elif module.endswith("prepare_attribution_refresh") or module == "resume_ig":
            (root / "ig/attributions/summary_365d").mkdir(parents=True)
            (root / "ig/attributions/summary_365d/new.json").write_text('{"fresh": true}')
            (root / "ig/overnight.env").write_text("")
        elif module.endswith("paper_experiments") and command[3] == "input-restrictions":
            (root / "ig/input_restrictions").mkdir()
            (root / "ig/input_restrictions/input_restriction_summary.csv").write_text("fresh,1\n")
        elif module.endswith("audit_tuning_router"):
            kwargs["stdout"].write('diagnostic progress\n{"passed": true}\n')
        elif module.endswith("render_supplement_tables"):
            (root / "supplement_tables").mkdir(exist_ok=False)
        elif module.endswith("assemble_figure_review"):
            (root / "review").mkdir(exist_ok=False)
            (root / "review/review_manifest.json").write_text('{"status": "review_ready"}')
        elif module.endswith("package_figure_review"):
            (root / "paper_review.zip").write_bytes(b"review bundle")
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(runner.subprocess, "run", execute)
    runner.recover(root)
    report = json.loads((root / "full_rehearsal.json").read_text())
    assert report["status"] == "complete"
    assert not any("verify_router_predictions" in " ".join(call) for call in calls)
    assert not any("analyse" in call for call in calls)
    assert all("held_out" not in call and "train" not in call for call in calls)
    assert (root / "analysis/attributions/summary_365d/new.json").is_file()
    assert json.loads((root / "cross_family_metrics.json").read_text()) == {"passed": True}
    archived, = root.glob("whole_cohort.interrupted_*/partial.txt")
    assert archived.read_text() == "keep for diagnosis"
    assert report["steps"]["complete_figures"]["previous_attempts"][0]["status"] == "running"
    calls.clear()
    runner.recover(root)
    assert calls == []


def test_recovery_rejects_changed_inventory_before_execution(tmp_path, monkeypatch):
    _, root, _ = setup_run(tmp_path)
    (root / "paired/analysis_inventory/analysis.yaml").write_text("artifacts: []")
    monkeypatch.setattr(runner.subprocess, "run", lambda *a, **kw: pytest.fail("No execution allowed"))
    with pytest.raises(ValueError, match="Frozen inventory changed"):
        runner.recover(root)


@pytest.mark.parametrize("module", ["complete_local_figures", "source_removal", "explain_direct_competing_risk"])
def test_recovery_refuses_live_original_with_relative_output_path(tmp_path, monkeypatch, module):
    root = tmp_path / "full_tuning_rehearsal_20261029_234123"
    root.mkdir()
    monkeypatch.setattr(runner.subprocess, "run", lambda *a, **kw: SimpleNamespace(
        stdout=f"42 S python -m opera.run.{module} --output-dir data/analysis/{root.name}/analysis\n"))
    with pytest.raises(RuntimeError, match="original run is still active"):
        with runner.recovery_lock(root):
            pytest.fail("Must not enter recovery")


def test_nested_failure_stops_before_new_expensive_stages(tmp_path, monkeypatch):
    _, root, _ = setup_run(tmp_path)
    calls = []

    def execute(command, **kwargs):
        calls.append(command)
        return SimpleNamespace(returncode=1)

    monkeypatch.setattr(runner.subprocess, "run", execute)
    with pytest.raises(RuntimeError, match="Figure completion failed"):
        runner.recover(root)
    assert len(calls) == 1
    assert json.loads((root / "full_rehearsal.json").read_text())["status"] == "finished_with_gaps"
