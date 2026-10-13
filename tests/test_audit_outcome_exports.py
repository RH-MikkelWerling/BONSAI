import pandas as pd
from pathlib import Path

from opera.run.audit_outcome_exports import audit_outcome


def test_outcome_export_audit_flags_temporal_and_split_errors(monkeypatch):
    frame = pd.DataFrame({
        "subject_id": [1, 2, 2],
        "split": ["train", "train", "held_out"],
        "index_date": ["2021-01-01", "2022-02-01", "2023-01-01"],
        "outcome_date": ["2021-01-02", "2022-02-01", None],
        "censor_date": ["2021-01-03", "2022-02-03", "2022-12-01"],
    })
    monkeypatch.setattr(pd, "read_parquet", lambda path: frame)
    row = audit_outcome(Path("example.parquet"))
    assert row["duplicate_subjects"] == 1
    assert row["event_on_or_before_index"] == 1
    assert row["censor_before_index"] == 1
    assert row["split_index_mismatch"] == 1
