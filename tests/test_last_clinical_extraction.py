"""Tests for the last-clinical-token extraction (narrow cls_last alternative).

Covers: the [CLS]-is-last-token verification (and its failure-reporting
path), the clamp behavior for patients with no clinical token before [CLS],
end-to-end ordering/schema against a synthetic encoder and subject pool, the
required subject-id/order match against a reference NPZ, and the spectral
summary helper.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

from bonsai.modules.networks.bonsai_nets import BonsaiBase
from opera.run.extract_last_clinical_embeddings import (
    assert_cls_is_last_token,
    assert_matches_reference_npz,
    extract_last_clinical_embeddings,
    spectral_summary,
)
from opera.run.extract_outcome_transfer_embeddings import (
    OutcomeTransferExtractionError,
)


def _small_config(num_layers: int = 2):
    return {
        "vocab_size": 8,
        "hidden_size": 8,
        "num_layers": num_layers,
        "num_attention_heads": 2,
        "max_seqlen": 32,
        "bias": False,
        "dropout": 0.0,
        "attention_dropout": 0.0,
        "causal": True,
        "attn_type": "sdpa",
    }


# ---------------------------------------------------------------------------
# assert_cls_is_last_token
# ---------------------------------------------------------------------------


def test_assert_cls_is_last_token_passes_when_cls_is_last():
    batch = {
        "code": torch.tensor([[5, 6, 1], [5, 1, 0]]),
        "attention_mask": torch.tensor(
            [[True, True, True], [True, True, False]]
        ),
        "subject_id": torch.tensor([10, 20]),
    }
    assert_cls_is_last_token(batch, cls_token_id=1)


def test_assert_cls_is_last_token_raises_and_reports_offending_subject_ids():
    batch = {
        # Row 0: last real token is code 6, not [CLS]=1 -- malformed.
        "code": torch.tensor([[5, 1, 6], [5, 6, 1]]),
        "attention_mask": torch.tensor([[True, True, True], [True, True, True]]),
        "subject_id": torch.tensor([111, 222]),
    }
    with pytest.raises(OutcomeTransferExtractionError, match=r"subject_ids=\[111\]"):
        assert_cls_is_last_token(batch, cls_token_id=1)


# ---------------------------------------------------------------------------
# spectral_summary
# ---------------------------------------------------------------------------


def test_spectral_summary_matches_hand_computed_formula(tmp_path):
    rng = np.random.RandomState(0)
    embeddings = rng.normal(size=(50, 6)).astype(np.float32)
    path = tmp_path / "synthetic.npz"
    np.savez_compressed(
        path,
        subject_ids=np.arange(50),
        embeddings=embeddings,
        splits=np.full(50, "train"),
    )

    participation_ratio, effective_rank, n90 = spectral_summary(path)

    centered = embeddings.astype(np.float64) - embeddings.astype(np.float64).mean(0)
    s = np.linalg.svd(centered, compute_uv=False) ** 2
    q = s / s.sum()
    expected_pr = s.sum() ** 2 / (s**2).sum()
    expected_er = np.exp(-(q * np.log(q + 1e-12)).sum())
    expected_n90 = int(np.searchsorted(np.cumsum(q), 0.90) + 1)

    assert participation_ratio == pytest.approx(expected_pr)
    assert effective_rank == pytest.approx(expected_er)
    assert n90 == expected_n90


# ---------------------------------------------------------------------------
# assert_matches_reference_npz
# ---------------------------------------------------------------------------


def test_assert_matches_reference_npz_passes_on_identical_order(tmp_path):
    path = tmp_path / "ref.npz"
    np.savez_compressed(
        path,
        subject_ids=np.array([3, 1, 2]),
        embeddings=np.zeros((3, 4)),
        splits=np.array(["train", "train", "tuning"]),
    )
    assert_matches_reference_npz(np.array([3, 1, 2]), path)


def test_assert_matches_reference_npz_fails_loudly_on_reordering(tmp_path):
    path = tmp_path / "ref.npz"
    np.savez_compressed(
        path,
        subject_ids=np.array([3, 1, 2]),
        embeddings=np.zeros((3, 4)),
        splits=np.array(["train", "train", "tuning"]),
    )
    with pytest.raises(OutcomeTransferExtractionError, match="do not exactly match"):
        assert_matches_reference_npz(np.array([1, 2, 3]), path)


# ---------------------------------------------------------------------------
# end-to-end extract_last_clinical_embeddings
# ---------------------------------------------------------------------------


def _write_subject(subject_id, codes, ages, absposes, segments):
    return {
        "subject_id": subject_id,
        "code": torch.tensor(codes, dtype=torch.long),
        "age": torch.tensor(ages, dtype=torch.float32),
        "abspos": torch.tensor(absposes, dtype=torch.float32),
        "segment": torch.tensor(segments, dtype=torch.int64),
    }


def _build_fixture(tmp_path):
    subjects = {
        # Subject 3 has NO token at all (not even background) before
        # censoring -- the appended [CLS] becomes the sole token in its
        # sequence, so lengths-1 goes negative and the clamp fires. This is
        # the literal condition _pool_last_clinical's clamp detects; per
        # DATA_FORMAT.md every real patient always carries background
        # tokens, so on real data this clamp is expected to fire on 0
        # patients -- it is a background-token-absent guard, not a
        # "no clinical event" detector (a patient with background tokens but
        # zero clinical events lands on their last background token, index
        # 0, which is >= 0 and therefore does NOT hit the clamp).
        1: _write_subject(1, [7, 5, 6], [0.0, 40.0, 41.0], [0.0, 1.0, 2.0], [0, 1, 2]),
        2: _write_subject(2, [7, 5, 6, 5], [0.0, 30.0, 31.0, 32.0], [0.0, 1.0, 2.0, 3.0], [0, 1, 2, 3]),
        3: _write_subject(3, [], [], [], []),
        4: _write_subject(4, [7, 5, 5, 6], [0.0, 20.0, 21.0, 22.0], [0.0, 1.0, 2.0, 3.0], [0, 1, 2, 3]),
        5: _write_subject(5, [7, 6, 5], [0.0, 60.0, 61.0], [0.0, 1.0, 2.0], [0, 1, 2]),
    }
    physical_a = tmp_path / "subject_data_shard_a.pt"
    physical_b = tmp_path / "subject_data_shard_b.pt"
    torch.save([subjects[1], subjects[2], subjects[3]], physical_a)
    torch.save([subjects[4], subjects[5]], physical_b)

    reference = pd.DataFrame(
        {
            "subject_id": [1, 2, 3, 4, 5],
            "split": ["train", "train", "train", "tuning", "held_out"],
            "censor_abspos": [2.0, 3.0, 0.0, 3.0, 2.0],
        }
    )
    vocabulary = {
        "[PAD]": 0,
        "[CLS]": 1,
        "[SEP]": 2,
        "[UNK]": 3,
        "[MASK]": 4,
        "CODE_A": 5,
        "CODE_B": 6,
        "BACKGROUND//sex_male": 7,
    }
    return {
        "paths": {"shard_a": physical_a, "shard_b": physical_b},
        "reference": reference,
        "vocabulary": vocabulary,
    }


def test_extract_last_clinical_embeddings_orders_matches_and_flags_clamp(tmp_path):
    torch.manual_seed(5)
    encoder = BonsaiBase(**_small_config()).eval()
    fixture = _build_fixture(tmp_path)

    subject_ids, splits, embeddings, clamped_subject_ids, split_counts = (
        extract_last_clinical_embeddings(
            encoder,
            reference=fixture["reference"],
            subject_split_paths=fixture["paths"],
            vocabulary=fixture["vocabulary"],
            split_keys={"train": "train", "tuning": "tuning", "held_out": "held_out"},
            max_len=32,
            batch_size=2,
            num_workers=0,
            device=torch.device("cpu"),
        )
    )

    assert subject_ids.tolist() == [1, 2, 3, 4, 5]
    assert splits.tolist() == ["train", "train", "train", "tuning", "held_out"]
    assert embeddings.shape == (5, 8)
    assert split_counts == {"train": 3, "tuning": 1, "held_out": 1}
    # Subject 3 has no token at all before [CLS] -> hits the clamp.
    assert clamped_subject_ids == [3]


def test_last_clinical_embedding_equals_hidden_state_one_before_cls(tmp_path):
    """Directly check the pooled vector against manual indexing for one patient."""
    torch.manual_seed(6)
    config = _small_config()
    encoder = BonsaiBase(**config).eval()

    from bonsai.functional.censoring import censor_subject
    from bonsai.functional.collate import dynamic_padding
    from opera.compat.bonsai import encoder_hidden_state

    subject = _write_subject(
        1, [7, 5, 6, 5], [0.0, 40.0, 41.0, 42.0], [0.0, 1.0, 2.0, 3.0], [0, 1, 2, 3]
    )
    censored = censor_subject(subject, censor_date_abspos=3.0, predict_token_id=1)
    censored["attention_mask"] = torch.ones(len(censored["code"]), dtype=torch.bool)
    batch = dynamic_padding([censored])

    with torch.no_grad():
        hidden = encoder_hidden_state(encoder(batch))
    # code is [7, 5, 6, 5, 1] -- [CLS] appended at the end; "one before CLS"
    # is index 3 (code value 5, the last real clinical event).
    assert batch["code"].tolist() == [[7, 5, 6, 5, 1]]
    expected = hidden[0, 3]

    # extract_last_clinical_embeddings requires all three prospective splits
    # to be non-empty (matching production's extract_shared_split_embeddings),
    # so add one throwaway patient each for tuning/held_out alongside the
    # patient under test.
    other_subjects = {
        2: _write_subject(2, [7, 6], [0.0, 50.0], [0.0, 1.0], [0, 1]),
        3: _write_subject(3, [7, 5], [0.0, 20.0], [0.0, 1.0], [0, 1]),
    }
    shard_path = tmp_path / "shard.pt"
    torch.save([subject, other_subjects[2], other_subjects[3]], shard_path)

    _, _, embeddings, clamped_subject_ids, _ = extract_last_clinical_embeddings(
        encoder,
        reference=pd.DataFrame(
            {
                "subject_id": [1, 2, 3],
                "split": ["train", "tuning", "held_out"],
                "censor_abspos": [3.0, 1.0, 1.0],
            }
        ),
        subject_split_paths={"shard": shard_path},
        vocabulary={
            "[PAD]": 0,
            "[CLS]": 1,
            "[SEP]": 2,
            "[UNK]": 3,
            "[MASK]": 4,
            "CODE_A": 5,
            "CODE_B": 6,
            "BACKGROUND//sex_male": 7,
        },
        split_keys={"train": "train", "tuning": "tuning", "held_out": "held_out"},
        max_len=32,
        batch_size=1,
        num_workers=0,
        device=torch.device("cpu"),
    )

    assert clamped_subject_ids == []
    subject_1_embedding = embeddings[0]  # train split is emitted first
    torch.testing.assert_close(
        torch.from_numpy(subject_1_embedding), expected, atol=1e-6, rtol=1e-6
    )
