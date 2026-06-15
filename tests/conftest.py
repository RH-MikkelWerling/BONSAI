"""Shared pytest fixtures for BONSAI/OPERA tests."""

import sys
from pathlib import Path

import pandas as pd
import pytest

# Add tests/ to sys.path so `from fixtures.tiny_data import ...` resolves.
sys.path.insert(0, str(Path(__file__).parent))

from fixtures.tiny_data import (  # noqa: E402
    make_outcome_parquet,
    make_subject_data_pt,
    make_subjects,
    make_vocabulary,
    make_vocabulary_pt,
)


@pytest.fixture(scope="session")
def tiny_subjects():
    """40 deterministic subjects in BONSAI format."""
    return make_subjects()


@pytest.fixture(scope="session")
def tiny_vocab():
    """Tiny deterministic vocabulary dict."""
    return make_vocabulary()


@pytest.fixture
def tiny_fixture_dir(tmp_path, tiny_subjects, tiny_vocab):
    """A tmp directory with the files evaluate.py and finetune.py expect."""
    d = tmp_path / "cohort"
    d.mkdir()
    (d / "outcomes").mkdir()
    make_subject_data_pt(tiny_subjects, d / "subject_data_held_out.pt")
    make_subject_data_pt(tiny_subjects, d / "subject_data_train.pt")
    make_subject_data_pt(tiny_subjects, d / "subject_data_tuning.pt")
    make_vocabulary_pt(d / "vocabulary.pt")
    make_outcome_parquet(tiny_subjects, d / "outcomes" / "mortality_1y.parquet")
    pop = pd.DataFrame({"subject_id": [s["subject_id"] for s in tiny_subjects]})
    pop.to_csv(d / "population_full.csv", index=False)
    return d
