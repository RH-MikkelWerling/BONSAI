import pandas as pd
import pytest

from opera.run.plot_patient_embeddings import _left_join_on_subject


def _embeddings() -> pd.DataFrame:
    return pd.DataFrame({
        "subject_id": [1, 2], "_join_id": ["1", "2"],
        "split": ["tuning", "tuning"],
    })


def test_join_accepts_matching_split_and_keeps_one_column():
    metadata = pd.DataFrame({
        "subject_id": [1, 2], "split": ["tuning", "tuning"],
        "cohort_fine": ["A", "B"],
    })
    joined = _left_join_on_subject(
        _embeddings(), metadata, subject_col="subject_id", source_label="metadata"
    )
    assert list(joined["split"]) == ["tuning", "tuning"]
    assert "split_x" not in joined and "split_y" not in joined
    assert list(joined["cohort_fine"]) == ["A", "B"]


def test_join_rejects_conflicting_split():
    metadata = pd.DataFrame({
        "subject_id": [1, 2], "split": ["tuning", "held_out"],
    })
    with pytest.raises(ValueError, match="split labels conflicting"):
        _left_join_on_subject(
            _embeddings(), metadata, subject_col="subject_id", source_label="metadata"
        )


def test_join_still_rejects_unrelated_overlapping_fields():
    frame = _embeddings().assign(cohort_fine=["A", "B"])
    metadata = pd.DataFrame({
        "subject_id": [1, 2], "split": ["tuning", "tuning"],
        "cohort_fine": ["A", "B"],
    })
    with pytest.raises(ValueError, match="repeats columns"):
        _left_join_on_subject(
            frame, metadata, subject_col="subject_id", source_label="metadata"
        )
