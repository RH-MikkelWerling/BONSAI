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
