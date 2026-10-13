from pathlib import Path

import pandas as pd
import pytest
import yaml

from opera.run.decide_outcome_panel import (
    decide_outcome,
    load_rules,
    support_for_outcome,
    threshold_sensitivity,
)


REGISTRY = Path("opera/configs/experiment_registry.yaml")
RULES = Path("opera/configs/outcome_panel_decision_v1.yaml")


def test_rule_overrides_are_registered() -> None:
    names = yaml.safe_load(REGISTRY.read_text(encoding="utf-8"))["outcomes"]
    rules = load_rules(RULES, names)
    assert rules["primary_horizon_days"] == 365
    assert rules["clinical_actions"]["hypofibrinogenemia_g3plus"]["action"] == "retire"


def test_support_counts_events_without_treating_early_censor_as_negative() -> None:
    frame = pd.DataFrame({
        "subject_id": [1, 2, 3],
        "split": ["train", "train", "tuning"],
        "index_date": ["2021-01-01", "2021-01-01", "2022-01-01"],
        "outcome_date": ["2021-02-01", None, "2023-02-01"],
        "censor_date": ["2021-02-01", "2021-03-01", "2023-02-01"],
    })
    rows = pd.DataFrame(support_for_outcome(frame, "example", [30, 365, 730]))
    train_365 = rows.loc[rows["split"].eq("train") & rows["horizon_days"].eq(365)].iloc[0]
    tuning_365 = rows.loc[rows["split"].eq("tuning") & rows["horizon_days"].eq(365)].iloc[0]
    assert train_365["n_events"] == 1
    assert train_365["n_full_followup_or_event"] == 1
    assert tuning_365["n_events"] == 0
    assert tuning_365["n_ever_events"] == 1


def test_held_out_events_cannot_change_decision() -> None:
    names = yaml.safe_load(REGISTRY.read_text(encoding="utf-8"))["outcomes"]
    rules = load_rules(RULES, names)
    base = pd.DataFrame({
        "outcome": ["overall_survival"] * 3,
        "split": ["train", "tuning", "held_out"],
        "horizon_days": [365] * 3,
        "n_events": [100, 20, 0],
    })
    first = decide_outcome("overall_survival", base, rules)
    base.loc[base["split"].eq("held_out"), "n_events"] = 10000
    second = decide_outcome("overall_survival", base, rules)
    assert first["disposition"] == second["disposition"] == "primary_candidate"
    assert first["held_out_events_descriptive_only"] != second["held_out_events_descriptive_only"]


def test_clinical_override_beats_large_support() -> None:
    names = yaml.safe_load(REGISTRY.read_text(encoding="utf-8"))["outcomes"]
    rules = load_rules(RULES, names)
    support = pd.DataFrame({
        "outcome": ["ALP_increased_g3plus"] * 3,
        "split": ["train", "tuning", "held_out"],
        "horizon_days": [365] * 3,
        "n_events": [1000, 100, 500],
    })
    assert decide_outcome("ALP_increased_g3plus", support, rules)["disposition"] == "definition_hold"


def test_threshold_grid_excludes_clinical_holds_and_heldout() -> None:
    decisions = pd.DataFrame({
        "clinical_action": ["candidate", "candidate", "definition_hold", "retire"],
        "train_events": [150, 60, 1000, 1000],
        "tuning_events": [25, 15, 1000, 1000],
        "held_out_events_descriptive_only": [0, 10000, 10000, 10000],
    })
    grid = threshold_sensitivity(decisions)
    row = grid.loc[
        grid["min_train_events"].eq(100) & grid["min_tuning_events"].eq(20)
    ].iloc[0]
    assert row["n_primary_candidates"] == 1
    decisions["held_out_events_descriptive_only"] = 0
    pd.testing.assert_frame_equal(grid, threshold_sensitivity(decisions))


def test_duplicate_subjects_fail_before_counting() -> None:
    frame = pd.DataFrame({
        "subject_id": [1, 1], "split": ["train", "train"],
        "index_date": ["2021-01-01"] * 2,
        "outcome_date": [None] * 2,
        "censor_date": ["2022-01-01"] * 2,
    })
    with pytest.raises(ValueError, match="duplicate"):
        support_for_outcome(frame, "example", [365])
