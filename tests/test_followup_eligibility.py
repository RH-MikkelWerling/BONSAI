import pandas as pd

from bonsai.functional.outcomes import (
    binarize_outcomes,
    find,
    split_and_binarize_outcomes,
    summarize_binarized_split_outputs,
)


def test_split_specific_min_followup_exclusions():
    outcomes = pd.DataFrame(
        [
            {
                "subject_id": 1,
                "split": "train",
                "index_date": pd.Timestamp("2023-01-01"),
                "outcome_date": pd.NaT,
                "censor_date": pd.Timestamp("2023-01-05"),
                "censor_abspos": 0.0,
            },
            {
                "subject_id": 2,
                "split": "tuning",
                "index_date": pd.Timestamp("2023-01-01"),
                "outcome_date": pd.NaT,
                "censor_date": pd.Timestamp("2023-01-05"),
                "censor_abspos": 0.0,
            },
            {
                "subject_id": 3,
                "split": "held_out",
                "index_date": pd.Timestamp("2023-01-01"),
                "outcome_date": pd.NaT,
                "censor_date": pd.Timestamp("2023-01-05"),
                "censor_abspos": 0.0,
            },
            {
                "subject_id": 4,
                "split": "held_out",
                "index_date": pd.Timestamp("2023-01-01"),
                "outcome_date": pd.Timestamp("2023-01-03"),
                "censor_date": pd.Timestamp("2023-01-05"),
                "censor_abspos": 0.0,
            },
        ]
    )

    train, val, test = split_and_binarize_outcomes(
        outcomes,
        train_key="train",
        val_key="tuning",
        test_key="held_out",
        n_hours_start_include=1,
        n_hours_end_include=24 * 7,
        require_min_followup_train=False,
        require_min_followup_val=True,
        require_min_followup_test=True,
        outcome_name="example",
    )

    assert set(train) == {1}
    assert set(val) == set()
    assert set(test) == {4}
    assert test[4]["label"] == 1

    summary = summarize_binarized_split_outputs(
        outcomes,
        split_outputs={"train": train, "tuning": val, "held_out": test},
        require_min_followup_by_split={
            "train": False,
            "tuning": True,
            "held_out": True,
        },
        n_hours_end_include=24 * 7,
        outcome_name="example",
    )
    held_out = summary[summary["split"] == "held_out"].iloc[0]
    tuning = summary[summary["split"] == "tuning"].iloc[0]
    assert tuning["n_excluded_insufficient_followup"] == 1
    assert held_out["n_excluded_insufficient_followup"] == 1


def _make_outcome_row(subject_id, split, index_date, outcome_date, censor_date):
    return {
        "subject_id": subject_id,
        "split": split,
        "index_date": pd.Timestamp(index_date),
        "outcome_date": pd.NaT if outcome_date is None else pd.Timestamp(outcome_date),
        "censor_date": pd.Timestamp(censor_date),
        "censor_abspos": 0.0,
    }


def test_competing_event_codes_as_event_2():
    # Patient 1: primary event at 30 days → event=1
    # Patient 2: no primary event, died at 20 days before censor_date → event=2
    # Patient 3: no primary event, admin censored (no death) → event=0
    outcomes = pd.DataFrame([
        _make_outcome_row(1, "train", "2023-01-01", "2023-01-31", "2023-06-01"),
        _make_outcome_row(2, "train", "2023-01-01", None,         "2023-06-01"),
        _make_outcome_row(3, "train", "2023-01-01", None,         "2023-06-01"),
    ])
    death_df = pd.DataFrame([
        _make_outcome_row(2, "train", "2023-01-01", "2023-01-21", "2023-06-01"),
    ])

    result = binarize_outcomes(
        outcomes,
        n_hours_start_include=0,
        competing_event_df=death_df,
    )

    assert result[1]["event"] == 1
    assert result[2]["event"] == 2
    assert result[3]["event"] == 0


def test_competing_event_uses_death_date_as_followup_end():
    # Patient died at day 20; their time_days should be 20, not the censor date.
    outcomes = pd.DataFrame([
        _make_outcome_row(1, "train", "2023-01-01", None, "2023-06-01"),
    ])
    death_df = pd.DataFrame([
        _make_outcome_row(1, "train", "2023-01-01", "2023-01-21", "2023-06-01"),
    ])

    result = binarize_outcomes(
        outcomes,
        n_hours_start_include=0,
        competing_event_df=death_df,
    )

    assert result[1]["event"] == 2
    assert abs(result[1]["time_days"] - 20.0) < 0.1


def test_death_after_censor_date_stays_admin_censored():
    # If the death occurs after the admin censor date, the patient is still
    # event=0 (they were administratively censored before they died).
    outcomes = pd.DataFrame([
        _make_outcome_row(1, "train", "2023-01-01", None, "2023-02-01"),
    ])
    death_df = pd.DataFrame([
        _make_outcome_row(1, "train", "2023-01-01", "2023-04-01", "2023-06-01"),
    ])

    result = binarize_outcomes(
        outcomes,
        n_hours_start_include=0,
        competing_event_df=death_df,
    )

    assert result[1]["event"] == 0


def test_competing_event_satisfies_min_followup_requirement():
    # A competing-death patient should pass the min-followup filter
    # (they have a definitive outcome) even if their time < window end.
    outcomes = pd.DataFrame([
        _make_outcome_row(1, "train", "2023-01-01", None, "2023-06-01"),
    ])
    death_df = pd.DataFrame([
        _make_outcome_row(1, "train", "2023-01-01", "2023-01-10", "2023-06-01"),
    ])

    result = binarize_outcomes(
        outcomes,
        n_hours_start_include=0,
        n_hours_end_include=24 * 365,  # 1-year window
        require_min_followup=True,
        competing_event_df=death_df,
    )

    # Patient died at day 9 — before the 365-day window — but must be retained
    # because they have a confirmed competing outcome.
    assert 1 in result
    assert result[1]["event"] == 2


def test_find_does_not_mutate_input_dataframe():
    source = pd.DataFrame(
        {
            "subject_id": [1, 2],
            "code": ["A", "B"],
        }
    )

    _ = find(
        source,
        conditions=[{"col": "code", "vals": ["A"]}],
        dependence="independent",
    )

    assert "_prio" not in source.columns
