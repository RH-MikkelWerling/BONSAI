import pandas as pd
import pytest
import yaml

from opera.evaluation.cohort_flow import (
    eligibility_file_path,
    summarize_eligibility_frame,
    validate_eligibility_frame,
)
from opera.run.summarize_cohort_flow import build_cohort_flow


def _eligibility_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "subject_id": [1, 2, 3, 4],
            "split": ["train", "train", "tuning", "held_out"],
            "eligible": [True, False, True, False],
            "eligibility_reason": [
                "eligible",
                "registry_not_covered",
                "eligible",
                "insufficient_followup",
            ],
            "source_covered": [True, False, True, True],
            "followup_adequate": [True, True, True, False],
        }
    )


def test_validate_eligibility_frame_rejects_duplicates_and_missing_reason():
    frame = _eligibility_frame()
    frame.loc[1, "subject_id"] = 1
    frame.loc[1, "eligibility_reason"] = ""

    issues = validate_eligibility_frame(frame)

    assert any("duplicate subject_id" in issue for issue in issues)
    assert any("non-empty eligibility_reason" in issue for issue in issues)


def test_validate_eligibility_frame_rejects_incoherent_lab_coverage():
    frame = _eligibility_frame()
    frame["post_index_adequate"] = [True, True, True, True]
    frame["last_measurement_date"] = pd.to_datetime(
        ["2020-01-02", None, "2020-01-04", "2020-01-05"]
    )

    issues = validate_eligibility_frame(frame)

    assert any("last_measurement_date" in issue for issue in issues)


def test_summarize_eligibility_frame_reports_criteria_and_final_denominator():
    summary = summarize_eligibility_frame(
        _eligibility_frame(),
        cohort_name="dlbcl",
        outcome_name="aki_30d",
    )

    train = summary[summary["split"] == "train"]
    included = train[
        (train["stage"] == "final_eligibility") & (train["status"] == "included")
    ]
    source_fail = train[
        (train["stage"] == "source_covered") & (train["status"] == "fail")
    ]

    assert included["n"].item() == 1
    assert source_fail["n"].item() == 1
    assert "registry_not_covered" in set(train["reason"].dropna())


def test_eligibility_file_path_renders_cell_placeholders(tmp_path):
    path = eligibility_file_path(
        str(tmp_path),
        "dlbcl",
        "aki_30d",
        {"eligibility_file": "{cohort}_{outcome}.csv"},
    )

    assert path == tmp_path / "outcomes" / "dlbcl_aki_30d.csv"


def test_build_cohort_flow_requires_every_configured_cell(tmp_path):
    data_dir = tmp_path / "dlbcl"
    outcomes_dir = data_dir / "outcomes"
    outcomes_dir.mkdir(parents=True)
    _eligibility_frame().to_csv(outcomes_dir / "aki_eligibility.csv", index=False)
    config = {
        "cohorts": {"dlbcl": {"data_dir": str(data_dir)}},
        "outcomes": {
            "aki_30d": {"eligibility_file": "aki_eligibility.csv"},
            "mortality_1y": {},
        },
        "model_variants": {"random": {"encoder_source": "random_init"}},
    }
    config_path = tmp_path / "sweep.yaml"
    config_path.write_text(yaml.safe_dump(config))

    with pytest.raises(ValueError, match="dlbcl/mortality_1y"):
        build_cohort_flow(str(config_path))


def test_build_cohort_flow_reads_configured_sidecars(tmp_path):
    data_dir = tmp_path / "dlbcl"
    outcomes_dir = data_dir / "outcomes"
    outcomes_dir.mkdir(parents=True)
    _eligibility_frame().to_csv(outcomes_dir / "aki_eligibility.csv", index=False)
    config = {
        "cohorts": {"dlbcl": {"data_dir": str(data_dir)}},
        "outcomes": {
            "aki_30d": {"eligibility_file": "aki_eligibility.csv"},
        },
        "model_variants": {"random": {"encoder_source": "random_init"}},
    }
    config_path = tmp_path / "sweep.yaml"
    config_path.write_text(yaml.safe_dump(config))

    result = build_cohort_flow(str(config_path))

    assert set(result["cohort"]) == {"dlbcl"}
    assert set(result["outcome"]) == {"aki_30d"}
