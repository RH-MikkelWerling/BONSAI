from pathlib import Path

from opera.evaluation.tasks import normalize_outcome_config, outcome_file_path


def test_normalize_outcome_config_supports_shared_event_file():
    outcomes = normalize_outcome_config(
        {
            "mortality_1y": {
                "outcome_file": "mortality.parquet",
                "n_hours_end_include": 8760,
            },
            "mortality_2y": {
                "outcome_file": "mortality.parquet",
                "n_hours_end_include": 17520,
            },
        }
    )

    assert outcomes["mortality_1y"]["outcome_file"] == "mortality.parquet"
    assert outcomes["mortality_2y"]["outcome_file"] == "mortality.parquet"
    assert Path(
        outcome_file_path("/data/dlbcl", "mortality_1y", outcomes["mortality_1y"])
    ) == Path("/data/dlbcl/outcomes/mortality.parquet")


def test_normalize_outcome_config_keeps_legacy_list_behavior():
    outcomes = normalize_outcome_config(["treatment_failure"])

    assert outcomes["treatment_failure"]["outcome_file"] == "treatment_failure.parquet"
    assert outcomes["treatment_failure"]["n_hours_end_include"] is None
