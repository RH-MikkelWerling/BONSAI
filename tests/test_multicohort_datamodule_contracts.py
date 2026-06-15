import pytest

from opera.modules.datamodules.MultiCohortContrastiveDataModule import (
    MultiCohortContrastiveDataModule,
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
