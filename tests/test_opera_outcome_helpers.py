import pandas as pd
import pytest

from bonsai.functional.features import compute_abspos
from opera.functional.outcomes import (
    attach_prediction_censor_abspos,
    filter_outcome_eligibility,
)


def test_attach_prediction_censor_abspos_uses_index_date_not_followup_censor_date():
    outcomes = pd.DataFrame(
        {
            "subject_id": [1],
            "index_date": [pd.Timestamp("2020-01-01")],
            "censor_date": [pd.Timestamp("2021-01-01")],
        }
    )

    with_abspos = attach_prediction_censor_abspos(outcomes)

    assert (
        with_abspos.loc[0, "censor_abspos"]
        == compute_abspos(outcomes["index_date"]).iloc[0]
    )
    assert (
        with_abspos.loc[0, "censor_abspos"]
        != compute_abspos(outcomes["censor_date"]).iloc[0]
    )


def test_filter_outcome_eligibility_removes_only_ineligible_rows():
    outcomes = pd.DataFrame(
        {
            "subject_id": [1, 2],
            "split": ["train", "train"],
            "index_date": pd.to_datetime(["2020-01-01", "2020-01-02"]),
        }
    )
    eligibility = pd.DataFrame(
        {
            "subject_id": [1, 2],
            "split": ["train", "train"],
            "eligible": [True, False],
            "eligibility_reason": ["eligible", "registry_not_covered"],
        }
    )

    filtered = filter_outcome_eligibility(outcomes, eligibility)

    assert filtered["subject_id"].tolist() == [1]


def test_filter_outcome_eligibility_rejects_unrepresented_outcome_rows():
    outcomes = pd.DataFrame(
        {
            "subject_id": [1, 2],
            "split": ["train", "train"],
        }
    )
    eligibility = pd.DataFrame(
        {
            "subject_id": [1],
            "split": ["train"],
            "eligible": [True],
            "eligibility_reason": ["eligible"],
        }
    )

    with pytest.raises(ValueError, match="missing 1 outcome rows"):
        filter_outcome_eligibility(outcomes, eligibility)
