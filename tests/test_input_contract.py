import torch

from bonsai.functional.input_contract import (
    checkpoint_numeric_value_control,
    resolve_numeric_value_control,
)
from bonsai.modules.datasets.FinetuneDataset import FinetuneDataset


def _subject():
    return {
        "subject_id": 1,
        "code": torch.tensor([1, 2, 3]),
        "segment": torch.tensor([0, 1, 1]),
        "age": torch.tensor([1.0, 2.0, 3.0]),
        "abspos": torch.tensor([1.0, 2.0, 3.0]),
        "numeric_value": torch.tensor([float("nan"), 0.25, -0.5]),
    }


def _dataset(control):
    return FinetuneDataset(
        [_subject()],
        outcomes={1: {"label": 1, "censor_abspos": None}},
        predict_token_id=9,
        background_length=1,
        max_len=8,
        numeric_value_control=control,
    )


def test_masked_finetune_contract_removes_values_without_mutating_source():
    dataset = _dataset("masked")
    item = dataset[0]

    assert torch.isnan(item["numeric_value"]).all()
    assert torch.isfinite(dataset.subjects[0]["numeric_value"][1:]).all()


def test_observed_finetune_contract_preserves_values():
    item = _dataset("observed")[0]
    assert item["numeric_value"][1:].tolist() == [0.25, -0.5]


def test_checkpoint_contract_is_inherited():
    hparams = {
        "checkpoint_metadata": {
            "input_contract": {"numeric_value_control": "masked"}
        }
    }
    assert resolve_numeric_value_control("inherit", hparams) == "masked"
    assert resolve_numeric_value_control("observed", hparams) == "observed"


def test_legacy_no_values_checkpoint_is_recognized():
    hparams = {
        "checkpoint_metadata": {
            "training_stage": "daly_care_only_pretraining_no_values"
        }
    }
    assert checkpoint_numeric_value_control(hparams) == "masked"


def test_legacy_ordinary_checkpoint_defaults_to_observed():
    hparams = {
        "checkpoint_metadata": {"training_stage": "daly_care_only_pretraining"}
    }
    assert checkpoint_numeric_value_control(hparams) == "observed"
