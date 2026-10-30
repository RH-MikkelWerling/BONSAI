"""One-endpoint repair: reproduce the export, preserve locks and isolate refits."""
import json
from pathlib import Path

import pandas as pd
import pytest
import yaml

from opera.run.audit_aki_correction import (
    AKI, compare, main as audit_main, read_endpoint, reconstruct,
)
from opera.run.build_aki_corrected_release import MANIFEST, REGISTRY, build, validate_release
from opera.run.decide_outcome_panel import sha256_file
from opera.run.paper_prediction_queue import bundle_runs
from opera.run.paper_tabular_queue import select_outcomes, tabular_jobs
from opera.run.prepare_aki_repair_bundle import prepare, scientific_hash
from opera.run.prepare_aki_tabular_overlay import prepare as prepare_overlay
from opera.run.verify_approved_outcome_training_v2 import verify


def endpoint():
    index = pd.to_datetime(["2021-01-01", "2022-01-01", "2023-01-01"])
    return pd.DataFrame({"subject_id": [1, 2, 3], "split": ["train", "tuning", "held_out"],
                         "index_date": index, "outcome_date": index + pd.Timedelta(days=30),
                         "censor_date": index + pd.Timedelta(days=730)})


def write_endpoint(path, frame):
    frame.to_parquet(path, index=False)
    return read_endpoint(path)


def test_reconstruction_uses_episode_starts_and_recomputes_censoring(tmp_path):
    frame = endpoint()
    old = write_endpoint(tmp_path / "old.parquet", frame)
    survival_frame = frame.copy()
    survival_frame["outcome_date"] = pd.NaT
    survival = write_endpoint(tmp_path / "survival.parquet", survival_frame)
    audit = frame[["subject_id", "split", "index_date"]].assign(
        eligible=True, last_measurement_date=frame.index_date + pd.Timedelta(days=40))
    audit_path = tmp_path / "coverage.parquet"
    audit.to_parquet(audit_path, index=False)
    episodes = pd.DataFrame({"patientid": [1, 1, 1, 2, 2, 3],
                             "aki_start_time": ["2021-01-01", "2021-01-31", "2021-03-01",
                                                "2022-01-01", "2025-01-01", "2023-02-10"],
                             "aki_episode": [1, 2, 3, 1, 2, 42]})
    table = tmp_path / "episodes.csv"
    episodes.to_csv(table, sep="\t", index=False)
    new = reconstruct(table, old, audit_path, survival, 180)
    assert new.loc["1", "outcome_date"] == pd.Timestamp("2021-01-31")
    assert new.loc["1", "censor_date"] == frame.censor_date.iloc[0]
    assert pd.isna(new.loc["2", "outcome_date"])
    assert new.loc["2", "censor_date"] == pd.Timestamp("2022-08-09")
    # Stage/episode number is not an inclusion criterion in OPERA's onset export.
    assert new.loc["3", "outcome_date"] == pd.Timestamp("2023-02-10")


def test_development_comparison_counts_date_and_observation_changes_not_heldout(tmp_path):
    frame = endpoint()
    old = write_endpoint(tmp_path / "old.parquet", frame)
    revised = frame.copy()
    revised.loc[0, "outcome_date"] = pd.NaT
    revised.loc[0, "censor_date"] = revised.loc[0, "index_date"] + pd.Timedelta(days=40)
    revised.loc[1, "outcome_date"] += pd.Timedelta(days=40)
    revised.loc[2, "outcome_date"] += pd.Timedelta(days=80)
    new = write_endpoint(tmp_path / "new.parquet", revised)
    mortality = frame.copy()
    mortality.outcome_date = pd.NaT
    survival = write_endpoint(tmp_path / "os.parquet", mortality)
    report, differences = compare(old, new, survival)
    assert report["event_dates_changed"] == 2
    assert report["censor_dates_changed"] == 1
    assert differences.subject_id.tolist() == ["1", "2"]
    at60 = next(row for row in report["horizon_comparison"]
                if row["split"] == "train" and row["horizon_days"] == 60)
    assert at60["events_removed"] == at60["observation_status_changed"] == 1
    assert {row["split"] for row in report["horizon_comparison"]} == {"train", "tuning"}


def test_competing_death_is_known_control(tmp_path):
    frame = endpoint()
    frame.outcome_date = pd.NaT
    frame.censor_date = frame.index_date + pd.Timedelta(days=10)
    old = write_endpoint(tmp_path / "old.parquet", frame)
    death = frame.copy()
    death.outcome_date = death.censor_date
    survival = write_endpoint(tmp_path / "os.parquet", death)
    report, _ = compare(old, old, survival)
    assert report["development_endpoint_identical"]
    assert all(row["old_known"] == 1 for row in report["horizon_comparison"])


def test_population_or_index_changes_are_not_silent_repairs(tmp_path):
    old = write_endpoint(tmp_path / "old.parquet", endpoint())
    with pytest.raises(ValueError, match="eligibility"):
        compare(old, old.iloc[:-1], old)
    changed = old.copy()
    changed.loc["1", "index_date"] -= pd.Timedelta(days=1)
    with pytest.raises(ValueError, match="index"):
        compare(old, changed, old)


def test_episode_audit_cli_reproduces_old_export_before_publishing(tmp_path, monkeypatch):
    frame = endpoint()
    old_path, os_path = tmp_path / "old.parquet", tmp_path / "os.parquet"
    frame.to_parquet(old_path, index=False)
    original_bytes = old_path.read_bytes()
    mortality = frame.copy()
    mortality.outcome_date = pd.NaT
    mortality.to_parquet(os_path, index=False)
    audit = frame[["subject_id", "split", "index_date"]].assign(
        eligible=True, last_measurement_date=frame.index_date + pd.Timedelta(days=40))
    coverage = tmp_path / "coverage.parquet"
    audit.to_parquet(coverage, index=False)
    episodes = pd.DataFrame({"patientid": frame.subject_id,
                             "aki_start_time": frame.outcome_date, "aki_episode": [1, 2, 1]})
    old_csv, new_csv = tmp_path / "old.csv", tmp_path / "new.csv"
    episodes.to_csv(old_csv, sep="\t", index=False)
    episodes.loc[0, "aki_start_time"] += pd.Timedelta(days=20)
    episodes.to_csv(new_csv, sep="\t", index=False)
    output = tmp_path / "audit"
    argv = ["audit", "--old-outcome", str(old_path), "--overall-survival", str(os_path),
            "--old-episodes", str(old_csv), "--corrected-episodes", str(new_csv),
            "--coverage-audit", str(coverage), "--output-dir", str(output)]
    monkeypatch.setattr("sys.argv", argv)
    audit_main()
    report = json.loads((output / "comparison_INTERNAL.json").read_text())
    assert report["old_endpoint_reproduced"] is True
    assert report["event_dates_changed"] == 1
    exported = pd.read_parquet(output / f"{AKI}.parquet")
    assert list(exported) == list(frame)
    assert exported.subject_id.dtype == frame.subject_id.dtype
    assert exported.loc[0, "outcome_date"] == pd.Timestamp("2021-02-20")
    assert old_path.read_bytes() == original_bytes
    assert not (output / "patient_differences_INTERNAL.parquet").exists()
    # A wrong original episode source must fail before publishing a candidate.
    argv[argv.index(str(old_csv))] = str(new_csv)
    failed = tmp_path / "mismatch"
    argv[-1] = str(failed)
    with pytest.raises(ValueError, match="not reproduced"):
        audit_main()
    assert not failed.exists()


@pytest.fixture
def correction(tmp_path):
    parent = tmp_path / "parent"
    (parent / "outcomes").mkdir(parents=True)
    names = ["overall_survival", AKI, *[f"endpoint_{i}" for i in range(67)]]
    registry = {"horizons_days": [30, 60, 90, 180, 365, 730], "death_outcome": "overall_survival",
                "outcomes_containing_death": ["overall_survival"],
                "cohort_groups": {"L": {"slug": "l", "fine": {"DLBCL": "dlbcl"}}},
                "outcomes": names, "outcome_families": {"all": names}, "model_variants": {}}
    (parent / REGISTRY).write_text(yaml.safe_dump(registry, sort_keys=False))
    frame = endpoint()
    entries = []
    for name in names:
        path = parent / "outcomes" / f"{name}.parquet"
        frame.to_parquet(path, index=False)
        entries.append({"outcome": name, "sha256": sha256_file(path)})
    manifest = {"version": 2, "status": "primary_release_candidate_pending_final_signoff",
                "outcomes": entries, "registry_sha256": sha256_file(parent / REGISTRY),
                "held_out_used_for_selection": False}
    (parent / MANIFEST).write_text(json.dumps(manifest))
    display = parent / "outcome_display_groups_primary_v2.yaml"
    display.write_text(yaml.safe_dump({"Clinical group": names}))
    (parent / "display_group_audit_INTERNAL.json").write_text(json.dumps({
        "registry_sha256": sha256_file(parent / REGISTRY), "output_sha256": sha256_file(display)}))
    templates = parent / "generated_configs"
    templates.mkdir()
    (templates / "base.yaml").write_text(yaml.safe_dump({
        "outcomes": {name: {"outcome_file": f"${{oc.env:BONSAI_OUTCOMES_DIR}}/{name}.parquet"} for name in names},
        "cross_outcome": {"aggregation": "endpoint_group_macro", "outcome_families": registry["outcome_families"],
                          "endpoint_groups": {"all": names}},
        "training": {"eval_monitor_metric": "val/cr_cif_auc_endpoint_group_macro",
                     "selection_min_cases": 20, "selection_min_controls": 20},
        "source_opera_ckpt": None}, sort_keys=False))
    revised = frame.copy()
    revised.loc[0, "outcome_date"] += pd.Timedelta(days=10)
    corrected = tmp_path / "corrected.parquet"
    revised.to_parquet(corrected, index=False)
    old_path = parent / "outcomes" / f"{AKI}.parquet"
    survival = parent / "outcomes" / "overall_survival.parquet"
    report, _ = compare(read_endpoint(old_path), read_endpoint(corrected), read_endpoint(survival))
    report["inputs"] = {"old_outcome": {"sha256": sha256_file(old_path)},
                        "overall_survival": {"sha256": sha256_file(survival)}}
    report["corrected_outcome"] = {"sha256": sha256_file(corrected)}
    comparison = tmp_path / "comparison.json"
    comparison.write_text(json.dumps(report))
    amended = tmp_path / "amended"
    build(parent, corrected, comparison, amended, "Historical reference window inequality correction only")
    return parent, amended, corrected, comparison


def test_release_keeps_all_68_other_files_and_registry_byte_identical(correction):
    parent, amended, _, _ = correction
    original, registry = validate_release(parent)
    updated, _ = validate_release(amended)
    assert updated["parent_manifest_sha256"] == sha256_file(parent / MANIFEST)
    assert sha256_file(parent / REGISTRY) == sha256_file(amended / REGISTRY)
    assert len(updated["unchanged_outcomes"]) == 68
    for name in registry["outcomes"]:
        same = sha256_file(parent / "outcomes" / f"{name}.parquet") == sha256_file(amended / "outcomes" / f"{name}.parquet")
        assert same == (name != AKI)
    assert json.loads((parent / MANIFEST).read_text()) == original
    assert (parent / "outcome_display_groups_primary_v2.yaml").read_bytes() == (
        amended / "outcome_display_groups_primary_v2.yaml").read_bytes()


def test_amended_release_passes_existing_v2_handoff_verifier(correction, monkeypatch):
    _, amended, _, _ = correction
    monkeypatch.setenv("BONSAI_OUTCOMES_DIR", str(amended / "outcomes"))
    result = verify(amended, amended / "generated_configs" / "base.yaml")
    assert result["outcomes"] == 69
    assert result["status"] == "structure_and_config_verified_pending_clinical_signoff"


def make_bundle(directory: Path, parent: Path, supervised_source=None, resume_checkpoint=None):
    (directory / "configs").mkdir(parents=True)
    rows = []
    for name, targets, regime in (("joint", [AKI, "endpoint_0"], "J"),
                                  ("aki_specialist", [AKI], "S"),
                                  ("other_specialist", ["endpoint_0"], "S")):
        config = {"outcomes": {key: {"outcome_file": str(parent / "outcomes" / f"{key}.parquet")}
                               for key in targets},
                  "cohorts": {"L": {"data_dir": str(parent), "population_file": "/secure/population.parquet"}},
                  "supervision_factorial": {"regime": regime, "fine_cohort": "DLBCL", "outcome": targets[0]},
                  "training": {"auto_resume": True}, "dapt_ckpt": "/secure/ssl.pt",
                  "paths": {"resume_ckpt": resume_checkpoint},
                  "source_opera_ckpt": supervised_source}
        path = directory / "configs" / f"{name}.yaml"
        path.write_text(yaml.safe_dump(config))
        digest = scientific_hash(config)
        rows.append({"name": name, "phase": "supervision", "origin": "L", "regime": regime, "seed": 42,
                     "condition": {}, "config_path": str(path.relative_to(directory)),
                     "config_sha256": sha256_file(path), "scientific_hash": digest,
                     "run_directory": f"L/seed42/{digest[:20]}"})
    (directory / "bundle.json").write_text(json.dumps({"runs": rows, "selected_router": None}))
    return rows


def test_bundle_interfaces_with_existing_queues_and_preserves_original_independent_fit(correction, tmp_path):
    parent, amended, _, _ = correction
    source = tmp_path / "source_bundle"
    original = make_bundle(source, parent)
    target = tmp_path / "repair_bundle"
    result = prepare([source], amended, target, ["J_L", "S_L"])
    assert result["dependency_counts"] == {"refit": 2, "carry_forward": 1}
    _, combined = bundle_runs(target)
    _, refits = bundle_runs(target / "refit_bundle")
    assert len(refits) == 2
    originals = {row["name"]: row for row in original}
    for row, path, config in combined:
        previous = originals[row["name"]]
        if row["name"] == "other_specialist":
            assert row["scientific_hash"] == previous["scientific_hash"]
            assert row["run_directory"] == previous["run_directory"]
            assert path.read_bytes() == (source / previous["config_path"]).read_bytes()
        else:
            assert row["scientific_hash"] != previous["scientific_hash"]
            assert row["run_directory"] != previous["run_directory"]
            assert config["outcomes"][AKI]["outcome_file"] == str(amended / "outcomes" / f"{AKI}.parquet")
        assert config["dapt_ckpt"] == "/secure/ssl.pt"
    # Existing tabular job derivation gets the new AKI path and original other path.
    jobs = tabular_jobs(target, [365])
    assert len(jobs) == 2
    for job in jobs:
        expected = amended if job["outcome"] == AKI else parent
        assert Path(job["outcome_path"]).parent == expected / "outcomes"
    assert select_outcomes(jobs, [AKI]) == [job for job in jobs if job["outcome"] == AKI]
    with pytest.raises(ValueError, match="not in"):
        select_outcomes(jobs, ["nonexistent"])
    overlay = tmp_path / "overlay"
    old_tabular, new_tabular = tmp_path / "old_tabular", tmp_path / "new_tabular"
    prepare_overlay(target, old_tabular, new_tabular, overlay)
    assert (overlay / "seed42/DLBCL/endpoint_0/365d").readlink() == old_tabular / "seed42/DLBCL/endpoint_0/365d"
    aki_link = overlay / "seed42/DLBCL/aki_first_documented/365d"
    assert aki_link.readlink() == new_tabular / "seed42/DLBCL/aki_first_documented/365d"
    assert not aki_link.exists()  # missing corrected fit never falls back to the old one


@pytest.mark.parametrize("kwargs", [{"supervised_source": "old_joint.ckpt"},
                                    {"resume_checkpoint": "old_last.ckpt"}])
def test_supervised_parent_dependency_is_not_silently_carried_forward(correction, tmp_path, kwargs):
    parent, amended, _, _ = correction
    source = tmp_path / "legacy"
    make_bundle(source, parent, **kwargs)
    with pytest.raises(ValueError, match="supervised weights"):
        prepare([source], amended, tmp_path / "bad", ["J_L", "S_L"])


def test_mutated_parent_or_candidate_cannot_be_published(correction, tmp_path):
    parent, _, corrected, comparison = correction
    changed = pd.read_parquet(corrected)
    changed.loc[0, "outcome_date"] += pd.Timedelta(days=1)
    changed.to_parquet(corrected, index=False)
    with pytest.raises(ValueError, match="exact old/new"):
        build(parent, corrected, comparison, tmp_path / "bad_release", "correction")
    assert not (tmp_path / "bad_release").exists()
