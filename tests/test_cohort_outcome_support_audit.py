from __future__ import annotations

import pandas as pd

from opera.run.audit_cohort_outcome_support import _candidate_tier, audit_support


def _outcomes(subject_ids, split, event_ids):
    index = pd.Timestamp("2020-01-01")
    return pd.DataFrame(
        {
            "subject_id": subject_ids,
            "split": split,
            "index_date": index,
            "outcome_date": [
                index + pd.Timedelta(days=30) if sid in event_ids else pd.NaT
                for sid in subject_ids
            ],
            "censor_date": index + pd.Timedelta(days=120),
        }
    )


def test_candidate_tier_ignores_held_out_support():
    base = {
        "n_primary_events_train": 31,
        "n_primary_events_tuning": 11,
        "n_ipcw_controls_train": 50,
        "n_ipcw_controls_tuning": 20,
        "n_primary_events_held_out": 0,
    }
    assert _candidate_tier(base) == "scarce_confirmatory"
    base["n_primary_events_held_out"] = 100
    assert _candidate_tier(base) == "scarce_confirmatory"


def test_audit_support_applies_fine_cohort_membership(tmp_path, monkeypatch):
    subjects = list(range(24))
    splits = ["train"] * 12 + ["tuning"] * 6 + ["held_out"] * 6
    outcomes_dir = tmp_path / "outcomes"
    outcomes_dir.mkdir()
    _outcomes(subjects, splits, {0, 1, 12, 18}).to_parquet(
        outcomes_dir / "toxicity.parquet"
    )
    _outcomes(subjects, splits, set()).to_parquet(
        outcomes_dir / "overall_survival.parquet"
    )
    population = pd.DataFrame(
        {
            "subject_id": subjects,
            "cohort_fine": ["SMALL"] * 12 + ["OTHER"] * 12,
        }
    )
    population_path = tmp_path / "population.csv"
    population.to_csv(population_path, index=False)
    registry = {
        "paths": {
            "shared_data_dir": str(tmp_path),
            "cohort_membership_file": str(population_path),
            "outcomes_dir": str(outcomes_dir),
        },
        "cohort_columns": {"fine": "cohort_fine"},
        "cohort_groups": {
            "GROUP": {"fine": {"SMALL": 12, "OTHER": 12}}
        },
        "availability_rules": {},
        "death_outcome": "overall_survival",
        "outcomes_containing_death": ["overall_survival"],
    }
    table = audit_support(registry, outcomes=["toxicity"], horizon_days=90)
    small = table[table["cohort"] == "SMALL"].iloc[0]
    other = table[table["cohort"] == "OTHER"].iloc[0]
    assert small["n_survival_train"] == 12
    assert small["n_primary_events_train"] == 2
    assert small["primary_event_rate_train"] == 2 / 12
    assert other["n_survival_train"] == 0
