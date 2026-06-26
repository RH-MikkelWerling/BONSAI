import pytest
import pandas as pd

from opera.modules.datamodules.MultiCohortContrastiveDataModule import (
    MultiCohortContrastiveDataModule,
    compute_pooled_class_counts,
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
