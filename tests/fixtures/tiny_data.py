"""Deterministic tiny data generators for fast smoke tests.

All generators are seeded and produce identical output on repeated calls.
Use these in tests that need real data shapes without training.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import torch

N_SUBJECTS = 40  # small enough to run in <100ms
N_TOKENS = 12  # vocabulary size (includes special tokens)
SEQ_LEN = 8  # tokens per synthetic subject sequence
SEED = 0

SPECIAL_TOKENS = ("[PAD]", "[MASK]", "[CLS]", "[SEP]")

# Splits assigned deterministically: 60% train, 20% tuning, 20% held_out.
_SPLIT_ORDER = ("train", "tuning", "held_out")
_SPLIT_FRACTIONS = (0.6, 0.2, 0.2)

# Anchor date for synthetic index dates; chosen so that one-year windows fall
# comfortably inside the registry coverage used elsewhere in the test suite.
_INDEX_ANCHOR = datetime(2020, 1, 1)


def make_subjects(n: int = N_SUBJECTS, seed: int = SEED) -> list[dict]:
    """Return a list of subject dicts in BONSAI FinetuneDataset format.

    Each subject has: subject_id (int), input_ids (LongTensor, len=8),
    attention_mask (LongTensor), segment (LongTensor), abspos (FloatTensor),
    age (FloatTensor).

    For interoperability with BONSAI datasets (which read ``code`` rather than
    ``input_ids``), ``code`` is provided as an alias of ``input_ids``.
    """
    rng = np.random.default_rng(seed)
    n_special = len(SPECIAL_TOKENS)
    cls_id = SPECIAL_TOKENS.index("[CLS]")
    sep_id = SPECIAL_TOKENS.index("[SEP]")

    subjects: list[dict] = []
    for offset in range(n):
        subject_id = offset + 1

        # Build a [CLS] ... [SEP] sequence using only non-special code ids in
        # the interior so the sequence looks like a real tokenized record.
        interior = rng.integers(n_special, N_TOKENS, size=SEQ_LEN - 2)
        input_ids = np.concatenate(([cls_id], interior, [sep_id])).astype(np.int64)

        # Two background tokens (segment 0) followed by event tokens (segment 1).
        segment = np.array([0, 0] + [1] * (SEQ_LEN - 2), dtype=np.int64)

        # Monotonic absolute positions in hours; deterministic per subject.
        base = float(offset * 24)
        abspos = base + np.arange(SEQ_LEN, dtype=np.float64) * 6.0

        # Ages increase slightly across the sequence; baseline varies per subject.
        baseline_age = 40.0 + float(rng.integers(0, 40))
        age = baseline_age + np.arange(SEQ_LEN, dtype=np.float64) * 0.1

        subject = {
            "subject_id": int(subject_id),
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "code": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.ones(SEQ_LEN, dtype=torch.long),
            "segment": torch.tensor(segment, dtype=torch.long),
            "abspos": torch.tensor(abspos, dtype=torch.float),
            "age": torch.tensor(age, dtype=torch.float),
        }
        subjects.append(subject)

    return subjects


def make_vocabulary(n_tokens: int = N_TOKENS) -> dict[str, int]:
    """Return a vocab dict with [PAD], [MASK], [CLS], [SEP] plus n_tokens-4 codes."""
    if n_tokens < len(SPECIAL_TOKENS):
        raise ValueError(
            f"n_tokens must be at least {len(SPECIAL_TOKENS)} to hold the "
            f"special tokens, got {n_tokens}."
        )

    vocabulary: dict[str, int] = {
        token: idx for idx, token in enumerate(SPECIAL_TOKENS)
    }
    for idx in range(len(SPECIAL_TOKENS), n_tokens):
        vocabulary[f"D_CODE_{idx:03d}"] = idx
    return vocabulary


def _assign_splits(n: int) -> list[str]:
    """Deterministically assign split labels to ``n`` ordered subjects."""
    n_train = int(round(n * _SPLIT_FRACTIONS[0]))
    n_tuning = int(round(n * _SPLIT_FRACTIONS[1]))
    n_held_out = n - n_train - n_tuning
    counts = (n_train, n_tuning, n_held_out)
    splits: list[str] = []
    for split_name, count in zip(_SPLIT_ORDER, counts):
        splits.extend([split_name] * count)
    return splits


def make_outcome_frame(
    subjects: list[dict],
    *,
    split_col: str = "split",
    event_rate: float = 0.3,
    seed: int = SEED,
) -> pd.DataFrame:
    """Return a minimal outcome DataFrame compatible with binarize_outcomes.

    Columns: subject_id, split (train/tuning/held_out), index_date,
    censor_date, event, time_to_event_days.
    Splits: 60% train, 20% tuning, 20% held_out.
    """
    rng = np.random.default_rng(seed)
    n = len(subjects)
    splits = _assign_splits(n)

    rows = []
    for position, subject in enumerate(subjects):
        subject_id = int(subject["subject_id"])
        index_date = _INDEX_ANCHOR + timedelta(days=position)
        # One year of potential follow-up for every subject.
        censor_date = index_date + timedelta(days=365)

        has_event = bool(rng.random() < event_rate)
        if has_event:
            # Event lands inside the one-year follow-up window.
            time_to_event_days = float(rng.integers(1, 365))
            outcome_date = index_date + timedelta(days=time_to_event_days)
            event = 1
        else:
            time_to_event_days = 365.0
            outcome_date = pd.NaT
            event = 0

        rows.append(
            {
                "subject_id": subject_id,
                split_col: splits[position],
                "index_date": pd.Timestamp(index_date),
                "censor_date": pd.Timestamp(censor_date),
                "outcome_date": outcome_date,
                "event": event,
                "time_to_event_days": time_to_event_days,
            }
        )

    frame = pd.DataFrame(rows)
    frame["index_date"] = pd.to_datetime(frame["index_date"])
    frame["censor_date"] = pd.to_datetime(frame["censor_date"])
    frame["outcome_date"] = pd.to_datetime(frame["outcome_date"])
    return frame


def make_outcome_parquet(
    subjects: list[dict],
    path: Path,
    *,
    seed: int = SEED,
) -> Path:
    """Write make_outcome_frame() output to a parquet at path. Returns path."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = make_outcome_frame(subjects, seed=seed)
    frame.to_parquet(path, index=False)
    return path


def make_vocabulary_pt(path: Path, n_tokens: int = N_TOKENS) -> Path:
    """Save vocabulary dict as a .pt file using torch.save. Returns path."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(make_vocabulary(n_tokens), path)
    return path


def make_subject_data_pt(
    subjects: list[dict],
    path: Path,
) -> Path:
    """Save subject list as a .pt file using torch.save. Returns path."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(subjects, path)
    return path
