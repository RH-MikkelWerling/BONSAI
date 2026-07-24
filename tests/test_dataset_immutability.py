import copy

import pytest
import torch

from bonsai.functional.censoring import censor_subject
from bonsai.functional.truncation import truncate_subject
from bonsai.modules.datasets.FinetuneDataset import FinetuneDataset
from bonsai.modules.datasets.PretrainDataset import (
    ARPretrainDataset,
    MLMPretrainDataset,
    PretrainDataset,
)
from opera.modules.datasets.ContrastiveDataset import ContrastiveDataset


def _subject():
    return {
        "subject_id": 11,
        "code": torch.tensor([5, 6, 7, 8, 9], dtype=torch.long),
        "abspos": torch.tensor([0.0, 1.0, 2.0, 3.0, 4.0]),
        "segment": torch.tensor([0, 0, 1, 1, 1], dtype=torch.long),
        "age": torch.tensor([40.0, 40.1, 40.2, 40.3, 40.4]),
    }


def _assert_subject_unchanged(actual, expected):
    assert actual.keys() == expected.keys()
    for key, value in expected.items():
        if isinstance(value, torch.Tensor):
            assert torch.equal(actual[key], value), key
        else:
            assert actual[key] == value


def _assert_samples_equal(left, right):
    assert left.keys() == right.keys()
    for key, value in left.items():
        if isinstance(value, torch.Tensor):
            assert torch.equal(value, right[key]), key
        else:
            assert value == right[key]


def test_finetune_dataset_indexing_does_not_mutate_subject():
    subjects = [_subject()]
    original = copy.deepcopy(subjects[0])
    outcomes = {
        11: {
            "label": 1,
            "censor_abspos": 2.5,
            "time_days": 1.0,
            "event": 1,
        }
    }
    dataset = FinetuneDataset(
        subjects,
        outcomes=outcomes,
        predict_token_id=1,
        background_length=2,
        max_len=4,
    )

    first = dataset[0]
    second = dataset[0]

    _assert_samples_equal(first, second)
    _assert_subject_unchanged(subjects[0], original)
    assert "target" not in subjects[0]
    assert "attention_mask" not in subjects[0]


def test_contrastive_dataset_indexing_does_not_mutate_subject():
    subjects = [_subject()]
    original = copy.deepcopy(subjects[0])
    outcomes = {
        "mortality": {
            11: {
                "label": 0,
                "censor_abspos": 3.5,
                "time_days": 30.0,
                "event": 0,
            }
        }
    }
    dataset = ContrastiveDataset(
        subjects,
        outcome_dicts=outcomes,
        predict_token_id=1,
        background_length=2,
        max_len=4,
    )

    first = dataset[0]
    second = dataset[0]

    _assert_samples_equal(first, second)
    _assert_subject_unchanged(subjects[0], original)
    assert "outcome_mortality" not in subjects[0]


def test_contrastive_dataset_passes_through_competing_death_event():
    subjects = [_subject()]
    outcomes = {
        "tx_failure": {
            11: {
                "label": 0,
                "censor_abspos": 3.5,
                "time_days": 30.0,
                "event": 2,  # competing death
            }
        }
    }
    dataset = ContrastiveDataset(
        subjects,
        outcome_dicts=outcomes,
        predict_token_id=1,
        background_length=2,
        max_len=4,
    )

    sample = dataset[0]
    assert sample["event_tx_failure"].item() == 2


def test_contrastive_dataset_rejects_inconsistent_prediction_origins():
    outcomes = {
        "aki_30d": {
            11: {
                "label": 0,
                "censor_abspos": 3.5,
                "time_days": 30.0,
                "event": 0,
            }
        },
        "mortality_1y": {
            11: {
                "label": 0,
                "censor_abspos": 4.5,
                "time_days": 365.0,
                "event": 0,
            }
        },
    }

    with pytest.raises(ValueError, match="inconsistent prediction origins"):
        ContrastiveDataset(
            [_subject()],
            outcome_dicts=outcomes,
            predict_token_id=1,
            background_length=2,
            max_len=4,
        )


def test_pretrain_dataset_indexing_and_helpers_do_not_mutate_subject():
    subjects = [_subject()]
    original = copy.deepcopy(subjects[0])
    dataset = PretrainDataset(subjects, max_len=3, background_length=1)

    first = dataset[0]
    second = dataset[0]

    _assert_samples_equal(first, second)
    _assert_subject_unchanged(subjects[0], original)

    censored = censor_subject(subjects[0], censor_date_abspos=2.5, predict_token_id=1)
    truncated = truncate_subject(subjects[0], max_len=3, background_length=1)
    _assert_subject_unchanged(subjects[0], original)
    assert len(censored["code"]) != len(subjects[0]["code"])
    assert len(truncated["code"]) != len(subjects[0]["code"])


def test_random_window_truncation_preserves_background_tokens():
    subject = {
        "subject_id": 22,
        "code": torch.arange(10, 22, dtype=torch.long),
        "abspos": torch.arange(12, dtype=torch.float),
        "segment": torch.tensor([0, 0] + [1] * 10, dtype=torch.long),
        "age": torch.arange(40, 52, dtype=torch.float),
    }
    generator = torch.Generator().manual_seed(123)

    truncated = truncate_subject(
        subject,
        max_len=6,
        background_length=2,
        strategy="random_window",
        generator=generator,
    )

    assert len(truncated["code"]) == 6
    assert torch.equal(truncated["code"][:2], subject["code"][:2])
    assert torch.equal(truncated["segment"][:2], subject["segment"][:2])


def test_ar_pretraining_masks_artificial_background_to_window_boundary():
    subject = {
        "subject_id": 33,
        "code": torch.arange(10, 22, dtype=torch.long),
        "abspos": torch.arange(12, dtype=torch.float),
        "segment": torch.tensor([0, 0] + [1] * 10, dtype=torch.long),
        "age": torch.arange(40, 52, dtype=torch.float),
    }
    dataset = ARPretrainDataset(
        [subject],
        max_len=5,
        background_length=2,
        truncation_strategy="tail",
    )

    sample = dataset[0]

    assert torch.equal(sample["code"], torch.tensor([10, 11, 18, 19, 20]))
    assert sample["target"][1].item() == -100
    assert torch.equal(sample["target"][[0, 2, 3, 4]], torch.tensor([11, 19, 20, 21]))


def test_ar_pretraining_builds_next_token_value_targets():
    subject = {
        "subject_id": 44,
        "code": torch.tensor([5, 6, 7, 8, 9], dtype=torch.long),
        "abspos": torch.arange(5, dtype=torch.float),
        "segment": torch.zeros(5, dtype=torch.long),
        "age": torch.arange(40, 45, dtype=torch.float),
        "value_bin": torch.tensor([0, 2, 3, 0, 4], dtype=torch.long),
        "value_normalized": torch.tensor([0.0, 0.2, 0.3, 0.0, 0.4]),
        "value_present": torch.tensor([False, True, True, False, True]),
    }
    dataset = ARPretrainDataset([subject], max_len=4, background_length=0)

    sample = dataset[0]

    assert torch.equal(sample["code"], torch.tensor([5, 6, 7, 8]))
    assert torch.equal(sample["target"], torch.tensor([6, 7, 8, 9]))
    assert torch.equal(
        sample["target_value_mask"], torch.tensor([True, True, False, True])
    )
    assert torch.equal(sample["target_value_bin"], torch.tensor([2, 3, -100, 4]))
    torch.testing.assert_close(
        sample["target_value_normalized"],
        torch.tensor([0.2, 0.3, 0.0, 0.4]),
    )


def test_mlm_pretraining_masks_value_inputs_for_selected_value_tokens():
    subject = {
        "subject_id": 55,
        "code": torch.tensor([5, 6, 7], dtype=torch.long),
        "abspos": torch.arange(3, dtype=torch.float),
        "segment": torch.zeros(3, dtype=torch.long),
        "age": torch.arange(40, 43, dtype=torch.float),
        "value_bin": torch.tensor([0, 2, 3], dtype=torch.long),
        "value_normalized": torch.tensor([0.0, 0.2, 0.3]),
        "value_present": torch.tensor([False, True, True]),
    }
    dataset = MLMPretrainDataset(
        [subject],
        max_len=3,
        background_length=0,
        vocabulary={
            "[PAD]": 0,
            "[CLS]": 1,
            "[SEP]": 2,
            "[UNK]": 3,
            "[MASK]": 4,
            "A": 5,
            "B": 6,
            "C": 7,
        },
        masking_select_ratio=1.0,
        masking_mask_ratio=0.0,
        masking_random_ratio=0.0,
    )

    sample = dataset[0]

    assert torch.equal(sample["target"], torch.tensor([5, 6, 7]))
    assert torch.equal(sample["target_value_mask"], torch.tensor([False, True, True]))
    assert torch.equal(sample["target_value_bin"], torch.tensor([-100, 2, 3]))
    torch.testing.assert_close(
        sample["target_value_normalized"],
        torch.tensor([0.0, 0.2, 0.3]),
    )
    assert torch.equal(sample["value_bin"], torch.tensor([0, 0, 0]))
    torch.testing.assert_close(
        sample["value_normalized"],
        torch.tensor([0.0, 0.0, 0.0]),
    )
    assert torch.equal(sample["value_present"], torch.tensor([False, False, False]))


def test_ar_combined_binning_predicts_value_from_preceding_event():
    subject = {
        "subject_id": 55,
        "code": torch.tensor([5, 6, 7, 6, 7]),
        "abspos": torch.arange(5, dtype=torch.float),
        "segment": torch.zeros(5, dtype=torch.long),
        "age": torch.arange(40, 45, dtype=torch.float),
        "value_bin": torch.tensor([0, 0, 3, 0, 4]),
        "value_normalized": torch.tensor([0.0, 0.0, 0.3, 0.0, 0.4]),
        "value_present": torch.tensor([False, False, True, False, True]),
    }
    dataset = ARPretrainDataset(
        [subject],
        max_len=4,
        background_length=0,
        vocabulary={"[VAL]": 7},
        value_embedding_mode="combined_binning",
    )

    sample = dataset[0]

    # The state at code 6 predicts the following bin representative.
    assert torch.equal(
        sample["target_value_mask"], torch.tensor([False, True, False, True])
    )
    torch.testing.assert_close(
        sample["target_value_normalized"], torch.tensor([0.0, 0.3, 0.0, 0.4])
    )
    # [VAL] is a numeric regression target, not a categorical CE target.
    assert torch.equal(sample["target"], torch.tensor([6, -100, 6, -100]))
