import pandas as pd
import pytest

from opera.run.summarize_candidate_lab_overlap import aggregate, band


def test_bands_suppress_small_denominators() -> None:
    assert band(1, 9) == "withheld"
    assert band(19, 20) == ">=90%"
    assert band(4, 20) == "10-24%"
    with pytest.raises(ValueError, match="Invalid"):
        band(21, 20)


def test_aggregate_uses_train_and_tuning_not_heldout() -> None:
    frame = pd.DataFrame([{
        "outcome": "creatinine_increased_g2plus", "split": split,
        "horizon_days": 365, "old_eligible": 100,
        "candidate_eligible": 80, "shared_eligible": 80, "lost_eligibility": 20,
        "old_events_on_lost_eligibility": 5,
        "both_event_on_shared": 30 if split != "held_out" else 0,
        "old_only_event_on_shared": 10 if split != "held_out" else 80,
        "candidate_only_event_on_shared": 20 if split != "held_out" else 0,
    } for split in ("train", "tuning", "held_out")])
    result = aggregate(frame, "creatinine_increased_g2plus", 365)
    assert result["old_vs_candidate_event_overlap"] == "50-74%"
    assert result["candidate_events_not_in_old"] == "25-49%"
    assert result["eligibility_retained"] == "75-89%"


def test_aggregate_allows_extra_relative_rise_eligibility() -> None:
    frame = pd.DataFrame([{
        "outcome": "creatinine_rise_gt1p5x_preindex_baseline", "split": split,
        "horizon_days": 365, "old_eligible": 100,
        "candidate_eligible": 110, "shared_eligible": 90, "lost_eligibility": 10,
        "old_events_on_lost_eligibility": 1, "both_event_on_shared": 30,
        "old_only_event_on_shared": 10, "candidate_only_event_on_shared": 20,
    } for split in ("train", "tuning")])
    result = aggregate(frame, "creatinine_rise_gt1p5x_preindex_baseline", 365)
    assert result["eligibility_retained"] == ">=90%"
    assert result["candidate_extra_eligibility"] == "10-24%"
