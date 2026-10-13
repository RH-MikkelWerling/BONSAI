from types import SimpleNamespace

import pandas as pd
import pytest

from opera.run import audit_cohort_outcome_support as audit


def test_train_tuning_only_never_builds_held_out(monkeypatch):
    registry = {
        "paths": {
            "cohort_membership_file": "population.parquet",
            "shared_data_dir": "shared",
            "outcomes_dir": "outcomes",
        },
        "death_outcome": "overall_survival",
        "cohort_columns": {"fine": "cohort_fine"},
        "cohort_groups": {"CLL_SLL": {"fine": {"CLL": 1, "SLL": 1}}},
    }
    visited = []
    parquet_filters = []

    def read_parquet(path, *, filters):
        parquet_filters.append(filters)
        return pd.DataFrame()

    monkeypatch.setattr(audit.pd, "read_parquet", read_parquet)
    monkeypatch.setattr(audit, "eligibility_file_path", lambda *a, **k: "eligibility")
    monkeypatch.setattr(audit, "population_subject_ids", lambda *a, **k: {1})

    def build(*args, split, **kwargs):
        visited.append((kwargs["cohort"], split))
        if split == "held_out":
            raise AssertionError("held-out cohort must remain closed")
        survival = SimpleNamespace(
            records={1: object()}, n_events=1,
            to_frame=lambda: pd.DataFrame({"event": [1]}),
        )
        return SimpleNamespace(survival=survival, fixed_horizon=SimpleNamespace(records={1: object()}))

    monkeypatch.setattr(audit, "build_evaluation_cohorts", build)
    monkeypatch.setattr(audit, "compute_ipcw_train_weights", lambda *a, **k: {})
    monkeypatch.setattr(audit, "summarize_ipcw_weights", lambda *a, **k: {
        "n_cases_nonzero": 1, "n_controls_nonzero": 0,
        "effective_sample_size": 1.0, "effective_sample_fraction": 1.0,
        "max_weight": 1.0,
    })
    table = audit.audit_support(
        registry, outcomes=["overall_survival"], horizon_days=365,
        fine_cohorts=["CLL"], include_held_out=False,
    )
    assert visited == [("CLL", "train"), ("CLL", "tuning")]
    assert parquet_filters == [
        [("split", "in", ["train", "tuning"])],
        [("split", "in", ["train", "tuning"])],
    ]
    assert len(table) == 1
    assert not any("held_out" in column for column in table)


def test_unknown_fine_cohort_rejected():
    registry = {
        "paths": {"cohort_membership_file": "population.parquet", "shared_data_dir": "shared", "outcomes_dir": "outcomes"},
        "death_outcome": "overall_survival",
        "cohort_groups": {"CLL_SLL": {"fine": {"CLL": 1}}},
    }
    with pytest.raises(ValueError, match="Unknown fine cohorts"):
        audit.audit_support(registry, outcomes=[], horizon_days=365, fine_cohorts=["MM"])


def test_approved_config_allows_new_outcome_and_uses_its_exclusions(monkeypatch):
    registry = {
        "paths": {"cohort_membership_file": "population.parquet", "shared_data_dir": "shared", "outcomes_dir": "outcomes"},
        "death_outcome": "overall_survival",
        "cohort_columns": {"fine": "cohort_fine"},
        "cohort_groups": {"CLL_SLL": {"fine": {"CLL": 1, "SLL": 1}}},
        "outcomes": ["overall_survival"],
    }
    approved = {
        "cohorts": {"CLL_SLL": {"exclude_outcomes": []}},
        "outcomes": {"new_approved_endpoint": {"competing_outcome_path": "death.parquet"}},
    }
    monkeypatch.setattr(audit.pd, "read_parquet", lambda path, **kwargs: pd.DataFrame())
    monkeypatch.setattr(audit, "eligibility_file_path", lambda *a, **k: None)
    monkeypatch.setattr(audit, "population_subject_ids", lambda *a, **k: {1})
    seen = []

    def build(*args, split, **kwargs):
        seen.append((kwargs["cohort"], split, kwargs["competing_outcomes"] is not None))
        survival = SimpleNamespace(records={}, n_events=0, to_frame=lambda: pd.DataFrame())
        return SimpleNamespace(survival=survival, fixed_horizon=SimpleNamespace(records={}))

    monkeypatch.setattr(audit, "build_evaluation_cohorts", build)
    monkeypatch.setattr(audit, "compute_ipcw_train_weights", lambda *a, **k: {})
    monkeypatch.setattr(audit, "summarize_ipcw_weights", lambda *a, **k: {
        "n_cases_nonzero": 0, "n_controls_nonzero": 0,
        "effective_sample_size": 0.0, "effective_sample_fraction": 0.0,
        "max_weight": 0.0,
    })
    table = audit.audit_support(
        registry, outcomes=["new_approved_endpoint"], horizon_days=365,
        fine_cohorts=["CLL"], include_held_out=False,
        training_config=approved,
    )
    assert len(table) == 1
    assert seen == [("CLL", "train", True), ("CLL", "tuning", True)]
    approved["cohorts"]["CLL_SLL"]["exclude_outcomes"] = ["new_approved_endpoint"]
    excluded = audit.audit_support(
        registry, outcomes=["new_approved_endpoint"], horizon_days=365,
        fine_cohorts=["CLL"], include_held_out=False,
        training_config=approved,
    )
    assert excluded.empty
