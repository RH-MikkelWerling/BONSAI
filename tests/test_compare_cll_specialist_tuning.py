from pathlib import Path

import pandas as pd
import pytest

import opera.run.compare_cll_specialist_tuning as audit


def _frame():
    return pd.DataFrame({
        "subject_id": list(range(25)), "cohort": ["CLL"] * 25,
        "outcome": ["sepsis"] * 25, "split": ["tuning"] * 25,
        "time_days": [100.] * 25, "event": [0] * 25,
        **{f"cif_{h}d": [.1] * 25 for h in audit.HORIZONS},
    })


def test_unsupported_horizons_are_reported_not_scored(monkeypatch):
    monkeypatch.setattr(audit, "_read_run", lambda path, outcomes, horizon: _frame())
    paths = {name: Path(name) for name in ("specialist", "clean", "loose")}
    estimates, contrasts = audit.compare_cell(paths, "sepsis")
    assert len(estimates) == len(audit.HORIZONS)
    assert estimates["status"].eq("unsupported").all()
    assert contrasts.empty


def test_mismatched_patient_sets_fail_even_when_under_supported(monkeypatch):
    def read(path, outcomes, horizon):
        frame = _frame()
        if str(path) == "loose":
            frame.loc[0, "subject_id"] = 999
        return frame
    monkeypatch.setattr(audit, "_read_run", read)
    paths = {name: Path(name) for name in ("specialist", "clean", "loose")}
    with pytest.raises(ValueError, match="Patient set"):
        audit.compare_cell(paths, "sepsis")
