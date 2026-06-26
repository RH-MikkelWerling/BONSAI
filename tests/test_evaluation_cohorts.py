import pandas as pd
import pytest

from opera.evaluation.cohorts import (
    FIXED_HORIZON_REGIME,
    SURVIVAL_REGIME,
    assert_cohort_parity,
    build_evaluation_cohorts,
    population_subject_ids,
    population_subject_strata,
)


def _outcomes():
    return pd.DataFrame(
        {
            "subject_id": [1, 2, 3, 4],
            "split": ["held_out"] * 4,
            "index_date": pd.to_datetime(["2020-01-01"] * 4),
            "outcome_date": pd.to_datetime(["2020-01-10", None, None, None]),
            "censor_date": pd.to_datetime(
                ["2020-03-01", "2020-03-01", "2020-01-15", "2020-03-01"]
            ),
        }
    )


def test_canonical_cohorts_separate_early_censoring_and_eligibility():
    eligibility = pd.DataFrame(
        {
            "subject_id": [1, 2, 3, 4],
            "split": ["held_out"] * 4,
            "eligible": [True, True, True, False],
            "eligibility_reason": ["", "", "", "lab_not_ascertained"],
        }
    )

    cohorts = build_evaluation_cohorts(
        _outcomes(),
        split="held_out",
        n_hours_start_include=1,
        n_hours_end_include=24 * 30,
        eligibility=eligibility,
        cohort="aml",
        outcome_name="lab_outcome",
    )

    assert cohorts.fixed_horizon.regime == FIXED_HORIZON_REGIME
    assert cohorts.survival.regime == SURVIVAL_REGIME
    assert cohorts.fixed_horizon.subject_ids == {1, 2}
    assert cohorts.survival.subject_ids == {1, 2, 3}
    assert cohorts.fixed_horizon.n_events == 1
    assert cohorts.survival.n_events == 1


def test_canonical_cohorts_respect_allowed_subject_ids():
    cohorts = build_evaluation_cohorts(
        _outcomes(),
        split="held_out",
        n_hours_start_include=1,
        n_hours_end_include=24 * 30,
        allowed_subject_ids={1, 3},
    )

    assert cohorts.survival.subject_ids == {1, 3}
    assert cohorts.fixed_horizon.subject_ids == {1}


def test_cohort_parity_passes_and_reports_symmetric_difference():
    cohorts = build_evaluation_cohorts(
        _outcomes().iloc[:3],
        split="held_out",
        n_hours_start_include=1,
        n_hours_end_include=24 * 30,
    )

    assert_cohort_parity(
        cohorts.fixed_horizon,
        [1, 2],
        model_name="logistic",
        outcome_name="lab_outcome",
    )
    with pytest.raises(ValueError, match=r"missing_n=1 extra_n=1"):
        assert_cohort_parity(
            cohorts.fixed_horizon,
            [1, 99],
            model_name="logistic",
            outcome_name="lab_outcome",
        )


def test_population_subject_strata_preserves_evaluation_order(tmp_path):
    population = pd.DataFrame(
        {
            "subject_id": [1, 2, 3],
            "cohort_fine": ["DLBCL", "FL", "AML"],
        }
    )
    path = tmp_path / "population.csv"
    population.to_csv(path, index=False)

    strata = population_subject_strata(path, [3, 1], "cohort_fine")

    assert strata.tolist() == ["AML", "DLBCL"]


def test_population_subject_ids_filters_fine_cohort(tmp_path):
    population = pd.DataFrame(
        {
            "subject_id": [1, 2, 3],
            "cohort_fine": ["DLBCL", "FL", "DLBCL"],
        }
    )
    path = tmp_path / "population.csv"
    population.to_csv(path, index=False)

    subject_ids = population_subject_ids(
        path,
        cohort_fine_col="cohort_fine",
        cohort_fine_value="DLBCL",
    )

    assert subject_ids == {1, 3}
