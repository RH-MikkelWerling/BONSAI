import torch

from bonsai.modules.datasets.PretrainDataset import ARPretrainDataset
from bonsai.modules.lightningmodules.PretrainModule import compute_pretrain_loss


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
        [_subject()],
        max_len=4,
        background_length=1,
        vocabulary={"[PAD]": 0, "[CLS]": 1, "[SEP]": 2, "A": 5, "B": 6},
        ignore_target_tokens=["[SEP]"],
    )
    item = dataset[0]
    assert 2 in item["code"]
    assert 2 not in item["target"]


def test_same_timestamp_transition_can_be_excluded():
    dataset = ARPretrainDataset(
        [_subject()],
        max_len=4,
        background_length=1,
        vocabulary={"[PAD]": 0, "[CLS]": 1, "[SEP]": 2, "A": 5, "B": 6},
        ignore_same_time_targets=True,
    )
    item = dataset[0]
    # A at time 20 predicting SEP at the same timestamp is ignored.
    assert item["target"][1].item() == -100


def test_same_timestamp_transition_is_retained_by_default():
    dataset = ARPretrainDataset(
        [_subject()],
        max_len=4,
        background_length=1,
        vocabulary={"[PAD]": 0, "[CLS]": 1, "[SEP]": 2, "A": 5, "B": 6},
    )
    item = dataset[0]
    # A at time 20 predicts SEP at time 20 under the ordinary ordered-stream
    # objective. The optional policy above remains available as an ablation.
    assert item["target"][1].item() == 2


def test_unknown_ignored_target_token_fails_fast():
    try:
        ARPretrainDataset(
            [_subject()],
            max_len=4,
            background_length=1,
            vocabulary={"[PAD]": 0},
            ignore_target_tokens=["[SEP]"],
        )
    except ValueError as exc:
        assert "absent from vocabulary" in str(exc)
    else:
        raise AssertionError("Expected an invalid target policy to fail.")


def test_prefixed_metadata_is_input_only():
    subject = _subject()
    subject["code"] = torch.tensor([1, 7, 5, 6, 0])
    dataset = ARPretrainDataset(
        [subject],
        max_len=4,
        background_length=1,
        vocabulary={"[PAD]": 0, "[CLS]": 1, "A": 5, "B": 6, "META//X": 7},
        input_only_target_prefixes=["META//"],
    )
    item = dataset[0]
    assert 7 in item["code"]
    assert 7 not in item["target"]


def test_event_normalized_weights_sum_to_one_per_eligible_timestamp():
    dataset = ARPretrainDataset(
        [_subject()],
        max_len=4,
        background_length=1,
        vocabulary={"[PAD]": 0, "[CLS]": 1, "[SEP]": 2, "A": 5, "B": 6},
        ignore_target_tokens=["[SEP]"],
        event_normalized_code_loss=True,
    )
    item = dataset[0]
    weights = item["code_loss_weight"]
    eligible_times = item["abspos"][item["target"] != -100]
    for timestamp in torch.unique(eligible_times):
        mask = (item["abspos"] == timestamp) & (item["target"] != -100)
        assert torch.isclose(weights[mask].sum(), torch.tensor(1.0))
    assert torch.all(weights[item["target"] == -100] == 0)


def test_event_normalized_cross_entropy_uses_supplied_mass():
    logits = torch.tensor([[3.0, 0.0], [0.0, 3.0], [0.0, 3.0]])
    labels = torch.tensor([0, 0, 1])
    weights = torch.tensor([1.0, 0.5, 0.5])
    loss, _, _, _ = compute_pretrain_loss(
        {"logits": logits, "labels": labels, "code_loss_weight": weights},
        torch.nn.CrossEntropyLoss(),
        torch.nn.CrossEntropyLoss(),
        torch.nn.MSELoss(),
    )
    expected = (
        torch.nn.functional.cross_entropy(logits, labels, reduction="none") * weights
    ).sum() / weights.sum()
    assert torch.allclose(loss, expected)
