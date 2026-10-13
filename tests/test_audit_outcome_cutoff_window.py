"""Aggregate administrative-date sensitivity checks."""

import pandas as pd

from opera.run.audit_outcome_cutoff_window import audit_cutoff_window


def test_index_and_event_cutoff_boundaries_follow_pipeline_conventions(tmp_path) -> None:
    pd.DataFrame({
        "subject_id": [1, 2, 3],
        "index_date": ["2020-01-01", "2026-03-04", "2026-03-19"],
        "outcome_date": ["2026-03-04", "2026-03-18", "2026-03-19"],
        "censor_date": ["2026-03-19"] * 3,
    }).to_parquet(tmp_path / "overall_survival.parquet", index=False)
    row = audit_cutoff_window(tmp_path, "2026-03-04", "2026-03-19").iloc[0]
    assert row["n_index_in_early_late_window"] == 1
    assert row["n_events_after_early_through_late"] == 2
    assert row["n_events_after_late"] == 0
