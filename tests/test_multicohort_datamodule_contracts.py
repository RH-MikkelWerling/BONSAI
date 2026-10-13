import importlib

import pytest
import pandas as pd

from opera.modules.datamodules.MultiCohortContrastiveDataModule import (
    MultiCohortContrastiveDataModule,
    compute_pooled_class_counts,
)
multicohort_module = importlib.import_module(
    "opera.modules.datamodules.MultiCohortContrastiveDataModule"
)


def test_multicohort_datamodule_fails_on_missing_required_outcome(tmp_path):
    data_dir = tmp_path / "dlbcl"
    (data_dir / "outcomes").mkdir(parents=True)
    module = MultiCohortContrastiveDataModule(
        cohort_configs={"dlbcl": {"data_dir": str(data_dir)}},
        outcome_configs={
            "aki_30d": {
                "outcome_file": "aki.parquet",
                "n_hours_start_include": 1,
                "n_hours_end_include": 720,
            }
        },
        predict_token_id=1,
        batch_size=2,
        num_workers=0,
        require_all_configured_cells=True,
    )

    with pytest.raises(FileNotFoundError, match="aki_30d"):
        module._load_outcomes_for_cohort(
            "dlbcl",
            {"data_dir": str(data_dir)},
            str(data_dir),
            "train",
        )


def test_multicohort_datamodule_can_explicitly_allow_missing_outcome(tmp_path):
    data_dir = tmp_path / "dlbcl"
    (data_dir / "outcomes").mkdir(parents=True)
    module = MultiCohortContrastiveDataModule(
        cohort_configs={"dlbcl": {"data_dir": str(data_dir)}},
        outcome_configs={
            "aki_30d": {
                "outcome_file": "aki.parquet",
                "n_hours_start_include": 1,
                "n_hours_end_include": 720,
            }
        },
        predict_token_id=1,
        batch_size=2,
        num_workers=0,
        require_all_configured_cells=False,
    )

    outcomes = module._load_outcomes_for_cohort(
        "dlbcl",
        {"data_dir": str(data_dir)},
        str(data_dir),
        "train",
    )

    assert outcomes == {"aki_30d": {}}


def test_validation_loader_uses_logical_batch_size_with_physical_microbatches():
    module = MultiCohortContrastiveDataModule(
        cohort_configs={},
        outcome_configs={},
        predict_token_id=1,
        batch_size=8,
        logical_batch_size=32,
        logical_val_batch_size=24,
        num_workers=0,
    )
    module.val_dataset = [{}] * 17

    loader = module.val_dataloader()

    assert loader.batch_size == 24


def test_small_specialist_keeps_its_only_partial_logical_training_batch():
    module = MultiCohortContrastiveDataModule(
        cohort_configs={}, outcome_configs={}, predict_token_id=1,
        batch_size=8, logical_batch_size=32, num_workers=0,
        batch_sampling={"type": "random"},
    )
    module.train_dataset = [{}] * 17
    module._setup_train_sampling()
    assert len(module.train_dataloader()) == 1
    module.train_dataset = [{}] * 65
    assert len(module.train_dataloader()) == 2  # pooled/full-batch behavior unchanged


def test_multicohort_indices_are_stable_and_sorted():
    module = MultiCohortContrastiveDataModule(
        cohort_configs={"DLBCL_like": {}, "CLL_SLL": {}},
        outcome_configs={},
        predict_token_id=1,
        batch_size=2,
        num_workers=0,
    )

    assert module.cohort_names == ["CLL_SLL", "DLBCL_like"]
    assert module.cohort_to_index == {"CLL_SLL": 0, "DLBCL_like": 1}


def test_fine_source_indices_are_separate_from_grouped_cohort_indices():
    module = MultiCohortContrastiveDataModule(
        cohort_configs={"DLBCL_like": {}, "CLL_SLL": {}},
        outcome_configs={}, predict_token_id=1, batch_size=2, num_workers=0,
        fine_names=["DLBCL", "PMBCL", "CLL"],
    )
    assert module.cohort_to_index == {"CLL_SLL": 0, "DLBCL_like": 1}
    assert module.fine_to_index == {"CLL": 0, "DLBCL": 1, "PMBCL": 2}


def test_compute_pooled_class_counts_uses_training_labels(tmp_path):
    data_dir = tmp_path / "dlbcl"
    outcomes_dir = data_dir / "outcomes"
    outcomes_dir.mkdir(parents=True)
    index_date = pd.Timestamp("2020-01-01")
    frame = pd.DataFrame(
        {
            "subject_id": [1, 2, 3, 4],
            "split": ["train", "train", "train", "tuning"],
            "index_date": [index_date] * 4,
            "censor_date": [index_date + pd.Timedelta(days=90)] * 4,
            "outcome_date": [
                index_date + pd.Timedelta(days=10),
                pd.NaT,
                index_date + pd.Timedelta(days=20),
                index_date + pd.Timedelta(days=5),
            ],
            "censor_abspos": [1.0, 1.0, 1.0, 1.0],
        }
    )
    frame.to_parquet(outcomes_dir / "aki.parquet")

    counts = compute_pooled_class_counts(
        cohort_configs={"dlbcl": {"data_dir": str(data_dir)}},
        outcome_configs={
            "aki_30d": {
                "outcome_file": "aki.parquet",
                "n_hours_start_include": 1,
                "n_hours_end_include": 720,
            }
        },
        split="train",
        require_min_followup=True,
        require_all_configured_cells=True,
    )

    assert counts == {"aki_30d": {"positive": 2, "negative": 1}}


def test_multi_outcome_label_cache_reuses_each_outcome_independently(tmp_path, monkeypatch):
    data_dir = tmp_path / "cohort"
    outcomes_dir = data_dir / "outcomes"
    outcomes_dir.mkdir(parents=True)
    for name in ("a", "b"):
        (outcomes_dir / f"{name}.parquet").write_bytes(name.encode())
    (data_dir / "population_full.csv").write_text("subject_id\n1\n")
    calls = []
    population_calls = []
    frame = pd.DataFrame({"split": ["train", "tuning"], "subject_id": [1, 1]})
    monkeypatch.setattr(multicohort_module.pd, "read_parquet", lambda path: (calls.append(str(path)), frame.copy())[1])
    monkeypatch.setattr(multicohort_module, "_read_population", lambda *_args: (
        population_calls.append(1), pd.DataFrame({"subject_id": [1]})
    )[1])
    monkeypatch.setattr(multicohort_module, "filter_outcome_eligibility", lambda df, *_args, **_kwargs: df)
    monkeypatch.setattr(multicohort_module, "attach_prediction_censor_abspos", lambda df: df)
    monkeypatch.setattr(multicohort_module, "filter_registry_eligible_outcomes", lambda df, *_args, **_kwargs: df)
    monkeypatch.setattr(multicohort_module, "binarize_outcomes", lambda df, **kwargs: {
        1: {"event": 1, "time_days": kwargs["n_hours_end_include"] / 24},
    })
    module = MultiCohortContrastiveDataModule(
        cohort_configs={"cohort": {"data_dir": str(data_dir)}},
        outcome_configs={
            "a": {"outcome_file": "a.parquet", "n_hours_start_include": 1, "n_hours_end_include": 720},
            "b": {"outcome_file": "b.parquet", "n_hours_start_include": 1, "n_hours_end_include": 1440},
        },
        predict_token_id=1, batch_size=2, num_workers=0,
        outcome_label_cache_dir=str(tmp_path / "cache"),
    )
    cfg = {"data_dir": str(data_dir)}
    first = module._load_outcomes_for_cohort("cohort", cfg, str(data_dir), "train")
    second = module._load_outcomes_for_cohort("cohort", cfg, str(data_dir), "train")
    assert first == second
    assert len(calls) == 2
    assert first["a"][1]["time_days"] == 30
    assert first["b"][1]["time_days"] == 60
    module._load_outcomes_for_cohort("cohort", cfg, str(data_dir), "tuning")
    assert len(calls) == 4
    assert len(population_calls) == 1
