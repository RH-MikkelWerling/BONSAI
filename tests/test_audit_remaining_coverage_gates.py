from pathlib import Path

import pandas as pd

from opera.run.audit_remaining_coverage_gates import audit_one


def test_era_audit_uses_assessable_denominator_and_excludes_incomplete_years(monkeypatch):
    frame = pd.DataFrame({
        "subject_id": [1, 2, 3, 4],
        "index_date": ["2018-01-01", "2018-01-01", "2025-01-01", "2026-01-01"],
        "outcome_date": ["2018-01-10", None, "2025-01-15", None],
        "censor_date": ["2018-01-20", "2019-01-01", "2025-02-01", "2026-03-19"],
    })
    monkeypatch.setattr(pd, "read_parquet", lambda path, columns: frame[columns])
    rates, calendar = audit_one(Path("hospitalisation.parquet"), "hospitalisation")
    year_2018_day_30 = next(row for row in rates
                            if row["index_year"] == 2018 and row["horizon_days"] == 30)
    assert year_2018_day_30["n_assessable"] == 2
    assert year_2018_day_30["n_events"] == 1
    assert year_2018_day_30["event_fraction_assessable"] == 0.5
    assert not any(row["index_year"] == 2025 and row["horizon_days"] == 365
                   for row in rates)
    assert not any(row["index_year"] == 2026 for row in rates)
    assert any(row["event_year"] == 2025 and row["n_first_events"] == 1
               for row in calendar)
