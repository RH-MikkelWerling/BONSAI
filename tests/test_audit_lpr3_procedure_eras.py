"""Checks for the read-only LPR2/LPR3 outcome comparison."""

import pandas as pd
import pytest

from opera.run.audit_lpr3_procedure_eras import compare_outcome, summarize


def _frame(events):
    return pd.DataFrame({
        "subject_id": [1, 2, 3],
        "index_date": pd.to_datetime(["2018-01-01", "2019-03-05", "2020-01-01"]),
        "censor_date": pd.to_datetime(["2019-02-01", "2020-04-01", "2020-02-01"]),
        "outcome_date": pd.to_datetime(events),
    }).set_index("subject_id")


def test_comparison_counts_new_and_earlier_events_by_era() -> None:
    old = _frame([None, "2019-03-20", None])
    new = _frame(["2018-01-06", "2019-03-10", None])
    compared = compare_outcome(old, new, "RBC_transfusion")
    rates, timing, changes = summarize(compared)
    pre = changes.loc[changes["index_era"].eq("pre-LPR3 index")].iloc[0]
    post = changes.loc[changes["index_era"].eq("post-LPR3 index")]
    assert pre["n_new_event"] == 1
    assert post["n_earlier_event"].sum() == 1
    assert rates.loc[(rates["index_year"].eq(2018)) &
                     (rates["version"].eq("new")) &
                     (rates["horizon_days"].eq(30)), "n_events"].iloc[0] == 1
    assert timing.loc[(timing["index_year"].eq(2018)) &
                      (timing["version"].eq("new")) &
                      (timing["time_bin"].eq("day 1–7")), "n_events"].iloc[0] == 1


def test_comparison_rejects_changed_population() -> None:
    old = _frame([None, None, None])
    new = old.drop(index=1)
    with pytest.raises(ValueError, match="population changed"):
        compare_outcome(old, new, "RBC_transfusion")
