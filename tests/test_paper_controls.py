"""Pooled XGBoost and random-initialisation controls in the paper workflow."""

import json
from argparse import Namespace

from opera.run.paper_prediction_queue import bundle_runs, canonical_models
from opera.run.paper_tabular_queue import pooled_tabular_jobs, tabular_jobs, training_command
from test_paper_workflow import bundle


def test_pooled_xgboost_covers_every_disease_of_each_outcome(tmp_path):
    directory = bundle(tmp_path)
    cells = tabular_jobs(directory, (365,))
    pooled = pooled_tabular_jobs(directory, (365,))
    assert {job["outcome"] for job in pooled} == {job["outcome"] for job in cells}
    for job in pooled:
        diseases = sorted(c["cohort"] for c in cells if c["outcome"] == job["outcome"])
        assert job["cohorts"] == diseases and job["pooled"]
        command = training_command(job, Namespace(features="f.parquet", output_dir="out"))
        assert command[command.index("--cohort_fine_value") + 1] == ",".join(diseases)
        assert "--add_cohort_feature" not in command  # matches the neural pooled models
        assert command[command.index("--models") + 1] == "xgboost"


def test_random_initialisation_control_is_one_joint_fit(tmp_path):
    directory = bundle(tmp_path)
    manifest = json.loads((directory / "bundle.json").read_text())
    controls = [row for row in manifest["runs"] if row["phase"] == "pretraining_control"]
    assert len(controls) == 1 and controls[0]["origin"] == "RI"
    _, rows = bundle_runs(directory)
    for row, _, config in rows:
        if row["phase"] == "pretraining_control":
            assert config["encoder_initialization"] == "random"
            assert canonical_models(row, config, manifest) == ["J_RI"]


def test_population_filter_accepts_several_fine_cohorts(tmp_path):
    import pandas as pd

    from opera.evaluation.cohorts import population_subject_ids

    path = tmp_path / "membership.csv"
    pd.DataFrame({"subject_id": [1, 2, 3], "cohort_fine": ["DLBCL", "FL", "MM"]}).to_csv(
        path, index=False)
    assert population_subject_ids(path, "cohort_fine", "DLBCL,FL") == {1, 2}
    assert population_subject_ids(path, "cohort_fine", "MM") == {3}


def test_router_decomposition_controls_come_with_the_router_lock():
    from test_paper_run_bundle import base_config, reference

    from opera.run.prepare_paper_run_bundle import build_bundle

    lock = {"source_mode": "both", "rank": 8, "query_mode": "interacting",
            "query_width": 128}
    stage1 = build_bundle(base_config(), reference(), {"L": "/x/best.ckpt"})
    assert not [e for e in stage1 if e["phase"] == "router_controls"]  # needs the lock
    entries = build_bundle(base_config(), reference(), {"L": "/x/best.ckpt"},
                           selected_router=lock)
    controls = {e["name"].rsplit("control_", 1)[-1]: e["config"]["competing_risk"]
                for e in entries if e["phase"] == "router_controls"}
    assert set(controls) == {"JDE", "RDH", "ROH"}
    assert controls["JDE"]["adapter_source_mode"] == "none"
    assert controls["JDE"]["adapter_disease_embedding"] is True
    assert controls["RDH"]["adapter_borrowing"] is False
    assert controls["ROH"]["adapter_source_mode"] == "outcome_only"


def test_sequence_directory_override_reaches_vocabulary_and_cohorts(tmp_path):
    from opera.run.prepare_paper_run_bundle import load_resolved_config

    config = tmp_path / "base.yaml"
    config.write_text(
        "paths:\n  sequence_data_dir: /old/harmonized_v4\n"
        "  vocabulary: ${paths.sequence_data_dir}/vocabulary.pt\n"
        "cohorts:\n  DLBCL_like:\n    data_dir: ${paths.sequence_data_dir}\n")
    target = tmp_path / "with_sor_final_v1"
    resolved = load_resolved_config(config, target)
    assert resolved["paths"]["vocabulary"] == str(target.resolve() / "vocabulary.pt")
    assert resolved["cohorts"]["DLBCL_like"]["data_dir"] == str(target.resolve())
    assert load_resolved_config(config)["paths"]["sequence_data_dir"].endswith("harmonized_v4")


def test_national_arms_read_national_sequences_but_our_outcomes(tmp_path):
    from opera.run.prepare_paper_run_bundle import external_sequences

    config = {"paths": {"vocabulary": "/local/vocabulary.pt"},
              "cohorts": {"DLBCL": {"data_dir": "/local/dlbcl"}}}
    result = external_sequences(config, tmp_path)
    assert result["paths"]["vocabulary"] == str(tmp_path.resolve() / "vocabulary.pt")
    assert result["cohorts"]["DLBCL"]["data_dir"] == "/local/dlbcl"  # outcomes unchanged
    assert result["cohorts"]["DLBCL"]["subject_data_paths"] == [
        str(tmp_path.resolve() / "subject_data_train.pt"),
        str(tmp_path.resolve() / "subject_data_tuning.pt")]
    assert external_sequences({"paths": {}}, None) == {"paths": {}}


def test_national_meds_is_filtered_to_our_patients_in_their_local_split(tmp_path):
    import pandas as pd

    from opera.run.prepare_external_sequences import (
        filter_external_meds, local_split_assignment)

    local = tmp_path / "local" / "data"
    for split, ids in (("train", [1, 2]), ("tuning", [3])):
        (local / split).mkdir(parents=True)
        pd.DataFrame({"subject_id": ids, "time": pd.Timestamp("2020-01-01"),
                      "code": "X"}).to_parquet(local / split / "0.parquet")
    national = tmp_path / "national" / "data" / "train"
    national.mkdir(parents=True)
    pd.DataFrame({"subject_id": [1, 3, 3, 9], "code": ["A", "B", "C", "D"],
                  "time": pd.to_datetime(["2019-01-01", "2023-05-01", "2023-06-01",
                                          "2020-01-01"])}).to_parquet(national / "0.parquet")
    assignment = local_split_assignment(tmp_path / "local")
    assert assignment == {1: "train", 2: "train", 3: "tuning"}
    report = filter_external_meds(tmp_path / "national", assignment, tmp_path / "out")
    assert report["rows_by_split"] == {"train": 1, "tuning": 2}
    assert report["missing_from_national"] == 1  # patient 2
    tuning = pd.read_parquet(next((tmp_path / "out/meds/data/tuning").glob("*.parquet")))
    assert set(tuning.subject_id) == {3}
    assert report["latest_event"].startswith("2023-06-01")


def test_width_probe_changes_only_the_encoder(tmp_path):
    import yaml

    from opera.run.width_probe import JOINT, probe_config

    joint = {"dapt_ckpt": "/local/256/best.ckpt", "training": {"epochs": 30},
             "paths": {"vocabulary": "/v.pt"}}
    path = tmp_path / "configs" / "seed42" / "L" / f"{JOINT}.yaml"
    path.parent.mkdir(parents=True)
    path.write_text(yaml.safe_dump(joint))
    config = probe_config(tmp_path, tmp_path / "512" / "best.ckpt")
    assert config["dapt_ckpt"] == str((tmp_path / "512" / "best.ckpt").resolve())
    assert {k: v for k, v in config.items() if k != "dapt_ckpt"} == {
        k: v for k, v in joint.items() if k != "dapt_ckpt"}
