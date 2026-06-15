"""Smoke tests that verify the tiny fixture produces correctly shaped data.

Marked @pytest.mark.smoke — these must run in under 1 second each.
"""

import pytest
import torch


@pytest.mark.smoke
def test_tiny_subjects_shape(tiny_subjects):
    assert len(tiny_subjects) == 40
    s = tiny_subjects[0]
    assert "subject_id" in s
    assert "input_ids" in s
    assert s["input_ids"].dtype == torch.long


@pytest.mark.smoke
def test_tiny_vocab_has_special_tokens(tiny_vocab):
    for tok in ("[PAD]", "[MASK]", "[CLS]"):
        assert tok in tiny_vocab, f"Missing {tok}"


@pytest.mark.smoke
def test_tiny_outcome_frame_splits(tiny_subjects):
    from tests.fixtures.tiny_data import make_outcome_frame

    df = make_outcome_frame(tiny_subjects)
    assert set(df["split"].unique()) == {"train", "tuning", "held_out"}
    assert "subject_id" in df.columns
    assert "event" in df.columns


@pytest.mark.smoke
def test_tiny_fixture_dir_has_expected_files(tiny_fixture_dir):
    assert (tiny_fixture_dir / "subject_data_held_out.pt").exists()
    assert (tiny_fixture_dir / "vocabulary.pt").exists()
    assert (tiny_fixture_dir / "outcomes" / "mortality_1y.parquet").exists()
    assert (tiny_fixture_dir / "population_full.csv").exists()


@pytest.mark.smoke
def test_binarize_outcomes_on_tiny_fixture(tiny_subjects):
    from tests.fixtures.tiny_data import make_outcome_frame
    from opera.compat.bonsai import binarize_outcomes

    df = make_outcome_frame(tiny_subjects)
    result = binarize_outcomes(
        df,
        n_hours_start_include=1,
        n_hours_end_include=8760,
        require_min_followup=False,
    )
    assert isinstance(result, dict)
    assert len(result) > 0


@pytest.mark.smoke
def test_subject_data_pt_round_trips(tiny_subjects, tmp_path):
    import torch
    from tests.fixtures.tiny_data import make_subject_data_pt

    path = make_subject_data_pt(tiny_subjects, tmp_path / "subjects.pt")
    loaded = torch.load(path, weights_only=False)
    assert len(loaded) == len(tiny_subjects)
    assert loaded[0]["subject_id"] == tiny_subjects[0]["subject_id"]
