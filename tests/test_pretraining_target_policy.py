import torch

from bonsai.modules.datasets.PretrainDataset import ARPretrainDataset


def _subject():
    return {
        "subject_id": 1,
        "code": torch.tensor([1, 5, 2, 6, 0]),
        "age": torch.arange(5, dtype=torch.float),
        "abspos": torch.tensor([10.0, 20.0, 20.0, 30.0, 0.0]),
        "segment": torch.tensor([0, 1, 1, 2, 0]),
    }


def test_sep_is_retained_as_input_but_can_be_excluded_as_target():
    dataset = ARPretrainDataset(
        [_subject()], max_len=4, background_length=1,
        vocabulary={"[PAD]": 0, "[CLS]": 1, "[SEP]": 2, "A": 5, "B": 6},
        ignore_target_tokens=["[SEP]"],
    )
    item = dataset[0]
    assert 2 in item["code"]
    assert 2 not in item["target"]


def test_same_timestamp_transition_can_be_excluded():
    dataset = ARPretrainDataset(
        [_subject()], max_len=4, background_length=1,
        vocabulary={"[PAD]": 0, "[CLS]": 1, "[SEP]": 2, "A": 5, "B": 6},
        ignore_same_time_targets=True,
    )
    item = dataset[0]
    # A at time 20 predicting SEP at the same timestamp is ignored.
    assert item["target"][1].item() == -100


def test_unknown_ignored_target_token_fails_fast():
    try:
        ARPretrainDataset(
            [_subject()], max_len=4, background_length=1,
            vocabulary={"[PAD]": 0}, ignore_target_tokens=["[SEP]"],
        )
    except ValueError as exc:
        assert "absent from vocabulary" in str(exc)
    else:
        raise AssertionError("Expected an invalid target policy to fail.")
