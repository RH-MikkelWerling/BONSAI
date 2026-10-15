"""Paper environment, launch planning, artifact contracts and baseline parity."""

from argparse import Namespace
import json
from pathlib import Path

import pandas as pd
import pytest
from dotenv import dotenv_values

from opera.run.configure_paper_env import update_env
from opera.run.paper_experiments import build_commands, load_environment
from opera.run.paper_prediction_queue import bundle_runs, canonical_models, evaluate_row
from opera.run.prepare_paper_run_bundle import (
    build_bundle,
    load_resolved_config,
    write_bundle,
)
from opera.run.paper_tabular_queue import tabular_jobs, training_command
from opera.run.train_tabular_baselines import outcome_labels, tune_estimator_params
from test_supervision_factorial import base_config
from test_paper_run_bundle import reference


def bundle(tmp_path, lock=None):
    base = base_config()
    for config in base["cohorts"].values():
        config.update(
            data_dir=str(tmp_path), population_file=str(tmp_path / "population.parquet")
        )
    for name, target in base["outcomes"].items():
        target.setdefault("outcome_file", name + ".parquet")
    entries = build_bundle(
        base,
        reference(),
        {"L": "/future/local256/best.ckpt"},
        seeds=(42,),
        selected_router=lock,
    )
    output = tmp_path / "bundle"
    write_bundle(entries, output, provenance={}, selected_router=lock)
    return output


def complete_run(run, monitor, score=0.25, payload=b"checkpoint"):
    """Write a completed fit whose marker records the selected checkpoint."""
    import hashlib

    run.mkdir(parents=True, exist_ok=True)
    (run / "best.ckpt").write_bytes(payload)
    (run / "training_complete.json").write_text(
        json.dumps(
            {
                "status": "completed",
                "selection": {
                    "monitor": monitor,
                    "mode": "min",
                    "best_model_score": score,
                    "best_epoch": 3,
                    "best_global_step": 40,
                    "best_checkpoint": "best.ckpt",
                    "best_checkpoint_sha256": hashlib.sha256(payload).hexdigest(),
                },
            }
        )
    )


def test_env_update_preserves_secrets_comments_permissions_and_is_preview_first(
    tmp_path, capsys
):
    path = tmp_path / ".env"
    original = (
        b"# comment\r\nTOKEN=private-value\r\nexport OPERA_PAPER_ROOT = 'old'\r\n"
    )
    path.write_bytes(original)
    path.chmod(0o600)
    settings = {
        "OPERA_PAPER_ROOT": "/new folder/$(touch nope)",
        "OPERA_PAPER_LOCAL_CKPT": "/data/it's.ckpt",
    }
    report = update_env(path, settings)
    assert path.read_bytes() == original
    assert "private-value" not in json.dumps(report)
    applied = update_env(path, settings, apply=True)
    assert Path(applied["backup"]).read_bytes() == original
    assert b"TOKEN=private-value\r\n" in path.read_bytes()
    assert (
        dotenv_values(path, interpolate=False)["OPERA_PAPER_LOCAL_CKPT"]
        == "/data/it's.ckpt"
    )
    assert path.stat().st_mode & 0o777 == 0o600
    assert update_env(path, settings, apply=True)["status"] == "unchanged"
    assert not (tmp_path / "nope").exists()


def test_env_duplicate_or_unsupported_key_does_not_write(tmp_path):
    path = tmp_path / ".env"
    path.write_text("OPERA_PAPER_ROOT=a\nexport OPERA_PAPER_ROOT=b\n")
    for settings in ({"OPERA_PAPER_ROOT": "c"}, {"AWS_SECRET_ACCESS_KEY": "secret"}):
        with pytest.raises(ValueError):
            update_env(path, settings, apply=True)
    assert path.read_text().endswith("=b\n")


def test_scheduler_gpu_is_preserved_and_dotenv_is_not_executed(tmp_path, monkeypatch):
    path = tmp_path / ".env"
    path.write_text("CUDA_VISIBLE_DEVICES=0\nOPERA_PAPER_ROOT='$(false)'\n")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2,3")
    env = load_environment(path)
    assert env["CUDA_VISIBLE_DEVICES"] == "2,3"
    assert env["OPERA_PAPER_ROOT"] == "$(false)"


def test_resolved_configs_freeze_environment_paths(tmp_path, monkeypatch):
    path = tmp_path / "base.yaml"
    path.write_text(
        "paths:\n  vocabulary: ${oc.env:TEST_PAPER_SEQUENCE}/vocabulary.pt\n"
    )
    monkeypatch.setenv("TEST_PAPER_SEQUENCE", "/approved/one")
    frozen = load_resolved_config(path)
    monkeypatch.setenv("TEST_PAPER_SEQUENCE", "/approved/two")
    assert frozen["paths"]["vocabulary"] == "/approved/one/vocabulary.pt"


def args(action="train", **kwargs):
    return Namespace(
        action=action,
        phase="supervision",
        stage=1,
        regime=None,
        shard_count=1,
        shard_index=0,
        max_runs=None,
        execute=False,
        include_external=False,
        split="tuning",
        export_routes=False,
        prior_only=False,
        env_file=Path(".env"),
        **kwargs,
    )


def env(tmp_path):
    return {
        "OPERA_PAPER_ROOT": str(tmp_path / "paper"),
        "OPERA_RELEASE_DIR": str(tmp_path / "release"),
        "OPERA_PAPER_BASE_CONFIG": str(tmp_path / "base.yaml"),
        "OPERA_PAPER_ROUTER_REFERENCE": str(tmp_path / "reference.yaml"),
        "OPERA_PAPER_LOCAL_CKPT": str(tmp_path / "local.ckpt"),
        "OPERA_SEQUENCE_DATA_DIR": str(tmp_path / "sequences"),
        "OPERA_PAPER_ROUTER_LOCK": str(tmp_path / "router_lock.json"),
        "OPERA_PAPER_DECISIONS": str(tmp_path / "decisions.json"),
        "OPERA_PAPER_FEATURE_MATRIX": str(tmp_path / "features.parquet"),
    }


def test_training_stages_never_auto_select_router_or_launch(tmp_path):
    config = env(tmp_path)
    option = args()
    command = build_commands(option, config)[0]
    assert "--execute" not in command
    option.execute = True
    assert "--execute" in build_commands(option, config)[0]
    option.phase = "rank_screen"
    assert "--decisions" in build_commands(option, config)[0]
    option.phase = "external"
    assert str(tmp_path / "paper/bundles/stage2") in build_commands(option, config)[0]
    option = args("prepare")
    option.stage = 2
    option.include_external = True
    command = build_commands(option, config)[0]
    assert "--router-lock" in command and "--external-dapt-checkpoint" in command


def test_dapt_uses_same_direct_export_and_correct_local_dataset(tmp_path):
    command = build_commands(args("dapt"), env(tmp_path))[0]
    assert (
        f"paths.initialization_ckpt={tmp_path}/paper/external_direct/encoder.ckpt"
        in command
    )
    # Local adaptation reads our patients in the national encoder's own format.
    assert f"paths.dir={tmp_path}/paper/external_sequences/sequences" in command
    assert "training.batch_size=1" in command
    assert "training.accumulate_grad_batches=128" in command


def test_prediction_candidates_are_not_mislabeled_as_final_models(tmp_path):
    lock = {
        "source_mode": "disease_only",
        "rank": 8,
        "query_mode": "interacting",
        "query_width": 128,
    }
    directory = bundle(tmp_path, lock)
    payload, rows = bundle_runs(directory)
    assert len(rows) < len(payload["runs"])
    aliases = [
        canonical_models(row, config, payload)
        for row, _, config in rows
        if row["phase"] == "final_sources"
    ]
    assert ["RD_L", "R_L"] in aliases
    assert all(
        canonical_models(row, config, payload) == []
        for row, _, config in rows
        if row["phase"] in {"rank_screen", "query_screen"}
    )
    row, path, config = next(item for item in rows if item[0]["phase"] == "supervision")
    options = Namespace(
        runs_dir=tmp_path / "runs",
        output_dir=tmp_path / "predictions",
        split="tuning",
        execute=False,
    )
    assert evaluate_row(row, path, config, options)["status"] == "training_incomplete"
    assert not options.output_dir.exists()


def test_bundle_tampering_fails_before_evaluation(tmp_path):
    directory = bundle(tmp_path)
    payload = json.loads((directory / "bundle.json").read_text())
    path = directory / payload["runs"][0]["config_path"]
    path.write_text(path.read_text() + "seed: 123\n")
    with pytest.raises(ValueError, match="modified"):
        bundle_runs(directory)


def test_tabular_jobs_match_specialist_scopes_and_all_horizons(tmp_path):
    directory = bundle(tmp_path)
    jobs = tabular_jobs(directory, (30, 365))
    assert not any(
        j["cohort"] == "HCL" and j["outcome"] == "second_line_treatment" for j in jobs
    )
    assert {j["horizon_days"] for j in jobs} == {30, 365}
    option = Namespace(
        features=tmp_path / "features.parquet", output_dir=tmp_path / "tabular"
    )
    command = training_command(jobs[0], option)
    assert "--known_status_cif" in command
    assert command[command.index("--selection_metric") + 1] == "log_loss"
    assert command[command.index("--n_hours_end_include") + 1] == str(
        jobs[0]["horizon_days"] * 24
    )


def test_paper_labels_match_horizon_bce_status_not_censoring_as_negative(tmp_path):
    date = pd.Timestamp("2020-01-01")
    frame = pd.DataFrame(
        {
            "subject_id": [1, 2, 3, 4],
            "split": ["train"] * 4,
            "index_date": [date] * 4,
            "outcome_date": [date + pd.Timedelta(days=4), pd.NaT, pd.NaT, pd.NaT],
            "censor_date": [date + pd.Timedelta(days=d) for d in (40, 5, 40, 10)],
        }
    )
    competing = pd.DataFrame(
        {"subject_id": [4], "outcome_date": [date + pd.Timedelta(days=10)]}
    )
    path = tmp_path / "outcome.parquet"
    death = tmp_path / "death.parquet"
    frame.to_parquet(path, index=False)
    competing.to_parquet(death, index=False)
    labels = outcome_labels(
        str(path),
        split="train",
        n_hours_start_include=1,
        n_hours_end_include=30 * 24,
        require_min_followup=True,
        known_status_cif=True,
        competing_outcome_path=str(death),
    )
    assert labels.set_index("subject_id").label.to_dict() == {1: 1, 3: 0, 4: 0}


def test_log_loss_selection_works_on_single_class_tuning_without_changing_default():
    train = pd.DataFrame(
        {"x": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0], "label": [0, 0, 0, 1, 1, 1]}
    )
    tuning = pd.DataFrame({"x": [0.0, 1.0], "label": [0, 0]})
    assert tune_estimator_params("logistic", train, tuning, ["x"]) == {}
    selected = tune_estimator_params(
        "logistic", train, tuning, ["x"], selection_metric="log_loss"
    )
    assert "C" in selected


def test_prediction_reuse_checks_checkpoint_and_output_hashes(tmp_path, monkeypatch):
    import opera.run.paper_prediction_queue as queue

    directory = bundle(tmp_path)
    _, rows = bundle_runs(directory)
    row, path, config = next(item for item in rows if item[0]["phase"] == "supervision")
    run = tmp_path / "runs" / row["run_directory"]
    complete_run(run, config["training"]["eval_monitor_metric"])
    option = Namespace(
        runs_dir=tmp_path / "runs",
        output_dir=tmp_path / "predictions",
        split="tuning",
        execute=True,
        horizons="30,365",
        batch_size=1,
        num_workers=0,
        device="cpu",
        export_routes=False,
    )

    def fake_run(command, **kwargs):
        output = Path(command[command.index("--output-dir") + 1])
        (output / "direct_cr_predictions.parquet").write_bytes(b"predictions")

    monkeypatch.setattr(queue.subprocess, "run", fake_run)
    assert evaluate_row(row, path, config, option)["status"] == "evaluated"
    assert (
        evaluate_row(row, path, config, option)["status"]
        == "reused_verified_evaluation"
    )
    output = option.output_dir / row["run_directory"] / option.split
    (output / "direct_cr_predictions.parquet").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="changed"):
        evaluate_row(row, path, config, option)


def test_generated_bayes_recipes_keep_reference_simple_and_sensitivities_separate(
    tmp_path,
):
    import yaml
    from opera.run.prepare_paper_bayes_recipes import prepare
    from test_bayesian_contrasts import sparse_analysis

    sparse_analysis(tmp_path / "analysis")
    template = {
        "metric": "auc",
        "prior_scale": None,
        "practical_tolerance": None,
        "trend": "spline",
    }
    primary = prepare(
        template,
        tmp_path / "recipes",
        tmp_path / "analysis",
        tmp_path / "results",
        (42, 43),
    )
    assert primary
    for file in primary:
        config = yaml.safe_load(Path(file).read_text())
        # The agreed primary model is the regularised support spline.
        assert config["trend"] == "spline" and config["nest_outcomes"] is False
        assert config["prior_scale"] is None
    variants = {
        path.stem.removeprefix("pooling_auc_")
        for path in (tmp_path / "recipes").glob("pooling_auc_*.yaml")
    }
    assert {"exchangeable", "support_spline"} <= variants
    assert template["trend"] == "spline"


def test_native_cif_reader_selects_horizons_before_melting(tmp_path):
    from opera.run.analyse_locked_predictions import read_artifact

    path = tmp_path / "predictions.parquet"
    pd.DataFrame(
        {
            "subject_id": [1, 2],
            "cohort": ["CLL"] * 2,
            "outcome": ["event"] * 2,
            "split": ["tuning"] * 2,
            "time_days": [365.0, 30.0],
            "event": [0, 1],
            "cif_30d": [0.1, 0.2],
            "cif_365d": [0.2, 0.4],
        }
    ).to_parquet(path, index=False)
    spec = {
        "path": str(path),
        "format": "direct_cif",
        "seed": 42,
        "model": "J_L",
        "data_contract": "test",
        "include_horizons_days": [365],
    }
    frame, _ = read_artifact(spec)
    assert len(frame) == 2 and set(frame.horizon_days) == {365}
    spec["include_horizons_days"] = [730]
    with pytest.raises(ValueError, match="missing"):
        read_artifact(spec)


def test_real_tabular_queue_saves_both_models_and_single_class_tuning(tmp_path):
    from opera.run.paper_tabular_queue import run_job

    n = 30
    ids = list(range(n))
    date = pd.Timestamp("2020-01-01")
    splits = ["train"] * 20 + ["tuning"] * 4 + ["held_out"] * 6
    outcomes = pd.DataFrame(
        {
            "subject_id": ids,
            "split": splits,
            "index_date": [date] * n,
            "censor_date": [date + pd.Timedelta(days=100)] * n,
            "outcome_date": [
                date + pd.Timedelta(days=10)
                if i < 20 and i % 2 or i >= 24 and i % 2
                else pd.NaT
                for i in ids
            ],
        }
    )
    population = pd.DataFrame(
        {"subject_id": ids, "cohort_fine": ["CLL"] * n, "split": splits}
    )
    features = pd.DataFrame(
        {
            "subject_id": ids,
            "age": [50 + i for i in ids],
            "marker": [float(i % 2) for i in ids],
        }
    )
    for name, table in (
        ("outcome", outcomes),
        ("population", population),
        ("features", features),
    ):
        table.to_parquet(tmp_path / f"{name}.parquet", index=False)
    job = {
        "seed": 42,
        "cohort": "CLL",
        "outcome": "event",
        "horizon_days": 30,
        "population": str(tmp_path / "population.parquet"),
        "outcome_path": str(tmp_path / "outcome.parquet"),
        "competing_path": None,
        "eligibility": None,
        "registry_start_date": None,
        "n_hours_start_include": 1,
        "source_config_sha256": "test",
        "relative_dir": "seed42/CLL/event/30d",
    }
    options = Namespace(
        features=tmp_path / "features.parquet",
        output_dir=tmp_path / "tabular",
        execute=True,
    )
    assert run_job(job, options)["status"] == "fitted"
    assert run_job(job, options)["status"] == "reused_verified_fit"
    directory = options.output_dir / job["relative_dir"]
    for name in ("logistic", "xgboost"):
        metadata = json.loads((directory / f"{name}_metadata.json").read_text())
        assert metadata["selection_metric"] == "log_loss"
        assert metadata["label_estimand"] == "binary_known_status_cif"
        assert len(pd.read_csv(directory / f"{name}_tuning_predictions.csv")) == 4


def test_split_specific_inventory_is_compatible_with_coverage_analysis():
    from opera.evaluation.secondary_analysis import coverage_inventory

    roster = pd.DataFrame(
        [
            {
                "model": "J_L",
                "seed": 42,
                "cohort": "CLL",
                "outcome": "event",
                "horizon_days": 365,
                "split": split,
                "status": "prediction_available",
            }
            for split in ("tuning", "held_out")
        ]
    )
    frame = roster.drop(columns="status").assign(label=[0, 1], subject_id=[1, 2])
    summary = coverage_inventory(frame, roster)
    assert summary.set_index("split").n_cases.to_dict() == {"tuning": 0, "held_out": 1}


def test_inventory_collects_verified_models_and_records_missing_cells(tmp_path):
    import yaml
    from opera.run.paper_analysis_inventory import build_inventory
    from opera.run.run_paper_queue import file_hash

    directory = bundle(tmp_path)
    _, rows = bundle_runs(directory)
    row, _, config = next(
        item
        for item in rows
        if item[0]["phase"] == "supervision" and item[0]["regime"] == "J"
    )
    run = tmp_path / "runs" / row["run_directory"]
    run.mkdir(parents=True)
    (run / "training_complete.json").write_text("{}")
    (run / "best.ckpt").write_bytes(b"checkpoint")
    prediction_root = tmp_path / "predictions"
    for split in ("tuning", "held_out"):
        output = prediction_root / row["run_directory"] / split
        output.mkdir(parents=True)
        pd.DataFrame(
            {
                "subject_id": [1],
                "cohort": ["CLL_SLL"],
                "outcome": ["overall_survival"],
                "split": [split],
                "time_days": [365.0],
                "event": [0],
                "cif_365d": [0.2],
            }
        ).to_parquet(output / "direct_cr_predictions.parquet", index=False)
        (output / "evaluation_complete.json").write_text(
            json.dumps(
                {
                    "inputs": {
                        "config_sha256": row["config_sha256"],
                        "checkpoint_sha256": file_hash(run / "best.ckpt"),
                    },
                    "prediction_sha256": file_hash(
                        output / "direct_cr_predictions.parquet"
                    ),
                }
            )
        )
    patient = tmp_path / "patient.csv"
    cell = tmp_path / "cell.csv"
    template = tmp_path / "template.yaml"
    patient.write_text("subject_id,cohort_fine\n1,CLL\n")
    cell.write_text(
        "cohort,outcome,horizon_days,n_events_train\nCLL,overall_survival,365,10\n"
    )
    template.write_text("estimand: binary_known_status_unweighted\nartifacts: []\n")
    option = Namespace(
        output_dir=tmp_path / "inventory",
        bundle_dir=directory,
        runs_dir=tmp_path / "runs",
        prediction_dir=prediction_root,
        tabular_dir=tmp_path / "tabular",
        template=template,
        patient_metadata=patient,
        cell_metadata=cell,
        data_contract="test",
        tabular_horizons="30,365",
        analysis_horizons="365",
    )
    report = build_inventory(option)
    assert report["n_artifacts"] == 2
    recipe = yaml.safe_load((option.output_dir / "analysis.yaml").read_text())
    assert all(artifact["model"] == "J_L" for artifact in recipe["artifacts"])
    assert all(
        artifact["include_horizons_days"] == [365] for artifact in recipe["artifacts"]
    )
    coverage = pd.read_csv(option.output_dir / "coverage.csv")
    assert "tabular_missing" in set(coverage.status)
    assert set(coverage.horizon_days) == {365}
    assert not coverage.duplicated(
        ["model", "seed", "split", "cohort", "outcome", "horizon_days"]
    ).any()


def test_gpu_worker_reads_both_stages_and_live_decisions(tmp_path):
    config = env(tmp_path)
    command = build_commands(
        args("worker", worker_name="node3-gpu1", wait=True), config
    )[0]
    assert command[command.index("-m") + 1 :][:2] == ["opera.run.paper_worker", "next"]
    bundles = [command[i + 1] for i, part in enumerate(command) if part == "--bundle-dir"]
    assert bundles == [
        f"{tmp_path}/paper/bundles/stage1",
        f"{tmp_path}/paper/bundles/stage2",
    ]
    assert command[command.index("--decisions") + 1] == f"{tmp_path}/decisions.json"
    assert "--wait" in command and "node3-gpu1" in command
    status = build_commands(args("queue-status", worker_name=None, wait=False), config)[0]
    assert "status" in status and "--wait" not in status
