import pandas as pd

from bonsai.functional.outcomes import (
    apply_prospective_split,
    summarize_outcome_splits,
    validate_split_integrity,
)


def test_apply_prospective_split_from_index_date_boundaries():
    outcomes = pd.DataFrame(
        {
            "subject_id": [1, 2, 3, 4],
            "index_date": pd.to_datetime(
                ["2023-06-01", "2023-10-01", "2024-01-01", "2025-01-01"]
            ),
            "outcome_date": [pd.NaT, pd.Timestamp("2023-11-01"), pd.NaT, pd.NaT],
        }
    )

    split = apply_prospective_split(
        outcomes,
        train_end="2023-06-30",
        val_start="2023-07-01",
        val_end="2023-12-31",
        test_start="2024-01-01",
        test_end="2024-12-31",
    )

    assert split.set_index("subject_id")["split"].to_dict() == {
        1: "train",
        2: "tuning",
        3: "held_out",
        4: "excluded",
    }


def test_summarize_outcome_splits_counts_subjects_and_events():
    outcomes = pd.DataFrame(
        {
            "subject_id": [1, 2, 3],
            "split": ["train", "held_out", "held_out"],
            "index_date": pd.to_datetime(["2023-01-01", "2024-01-01", "2024-02-01"]),
            "outcome_date": [pd.NaT, pd.Timestamp("2024-02-01"), pd.NaT],
            "label": [0, 1, 0],
        }
    )

    summary = summarize_outcome_splits(outcomes, outcome_name="mortality")
    held_out = summary[summary["split"] == "held_out"].iloc[0]

    assert held_out["outcome"] == "mortality"
    assert held_out["n_subjects"] == 2
    assert held_out["n_events_observed"] == 1
    assert held_out["label_prevalence"] == 0.5


def test_validate_split_integrity_detects_overlap_and_boundary_violations():
    outcomes = pd.DataFrame(
        {
            "subject_id": [1, 1, 2],
            "split": ["train", "held_out", "tuning"],
            "index_date": pd.to_datetime(["2023-01-01", "2023-06-01", "2025-01-01"]),
        }
    )

    report = validate_split_integrity(
        outcomes,
        train_end="2023-12-31",
        val_start="2023-07-01",
        val_end="2023-12-31",
        test_start="2024-01-01",
    )

    assert report["ok"] is False
    assert report["subject_overlap_counts"]["train__held_out"] == 1
    assert report["date_boundary_violations"]["tuning"] == 1
    assert report["date_boundary_violations"]["held_out"] == 1
