import torch

from opera.evaluation.input_restriction import (
    hidden_code_ids, kept_code_ids, restrict_sample, restricting_collate)

VOCAB = {"[PAD]": 0, "[CLS]": 1, "BG_SEX//F": 2, "LABTEST//NPU1//BIN_3": 3,
         "MEDICATION//J01": 4, "DIAGNOSIS//C83": 5}


def sample():
    return {"code": torch.tensor([2, 5, 3, 4, 3, 1]),
            "abspos": torch.tensor([0.0, 10.0, 24 * 100, 24 * 380, 24 * 395, 24 * 400]),
            "attention_mask": torch.ones(6, dtype=torch.bool),
            "outcome_x": torch.tensor([1.0, 0.0, 1.0, 0.0, 1.0, 0.0]),
            "subject_id": 7}


def test_hiding_a_source_removes_its_tokens_from_every_per_token_field():
    hidden = hidden_code_ids(VOCAB, ["LABTEST"])
    out = restrict_sample(sample(), hidden_ids=hidden, protected_ids=kept_code_ids(VOCAB))
    assert out["code"].tolist() == [2, 5, 4, 1]
    assert len(out["abspos"]) == len(out["attention_mask"]) == 4
    assert out["outcome_x"].tolist() == sample()["outcome_x"].tolist()  # labels untouched
    assert out["subject_id"] == 7


def test_time_window_keeps_recent_events_and_background():
    out = restrict_sample(sample(), protected_ids=kept_code_ids(VOCAB), keep_last_days=30)
    # Background sex token and the prediction token survive; only the last 30 days remain.
    assert out["code"].tolist() == [2, 4, 3, 1]


def test_special_tokens_are_never_hidden_and_collate_wraps():
    assert hidden_code_ids(VOCAB, ["[", "BG_"]) == set()
    collate = restricting_collate(lambda batch: [s["code"].tolist() for s in batch],
                                  hidden_ids={4}, protected_ids=kept_code_ids(VOCAB))
    assert collate([sample()]) == [[2, 5, 3, 3, 1]]


def test_gap_uses_prediction_time_preserves_boundary_and_aligned_fields():
    original = sample()
    # Last clinical measurement is five days before index; the gap is counted
    # from the prediction token, so the measurement exactly on the edge stays.
    out = restrict_sample(original, protected_ids=kept_code_ids(VOCAB), drop_last_days=5)
    assert out["code"].tolist() == original["code"].tolist()
    out = restrict_sample(original, protected_ids=kept_code_ids(VOCAB), drop_last_days=6)
    assert out["code"].tolist() == [2, 5, 3, 4, 1]
    assert len(out["abspos"]) == len(out["attention_mask"]) == 5
    assert torch.equal(out["outcome_x"], original["outcome_x"])
    assert len(original["code"]) == 6
    zero = restrict_sample(original, protected_ids=kept_code_ids(VOCAB), drop_last_days=0)
    assert torch.equal(zero["code"], original["code"])
