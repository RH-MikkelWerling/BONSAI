import polars as pl
import pytest
import torch

from bonsai.functional.collate import dynamic_padding
from bonsai.functional.censoring import append_predict_token
from bonsai.functional.checkpointing import load_pretrained_encoder_checked
from bonsai.functional.create_data import prepare_continuous_numeric_values
from bonsai.functional.model_config import normalize_bonsai_model_config
from bonsai.modules.datasets.PretrainDataset import (
    ARPretrainDataset,
    MLMPretrainDataset,
)
from bonsai.modules.lightningmodules.PretrainModule import compute_pretrain_loss
from bonsai.modules.networks.bonsai_nets import BonsaiFinetune, BonsaiPretrain
from bonsai.modules.networks.components.embeddings import ContinuousValueEmbedding


def _model_config():
    return {
        "vocab_size": 12,
        "max_seqlen": 8,
        "hidden_size": 8,
        "num_layers": 1,
        "num_attention_heads": 2,
        "bias": False,
        "dropout": 0.0,
        "attention_dropout": 0.0,
        "causal": True,
        "attn_type": "sdpa",
        "value_embedding_mode": "film",
        "abspos_encoding": "legacy",
    }


def _subject():
    return {
        "subject_id": 1,
        "code": torch.tensor([5, 6, 7, 8]),
        "abspos": torch.arange(4, dtype=torch.float),
        "segment": torch.zeros(4, dtype=torch.long),
        "age": torch.arange(40, 44, dtype=torch.float),
        "numeric_value": torch.tensor([float("nan"), 0.2, float("nan"), 0.8]),
    }


def test_continuous_ingestion_uses_ehr2meds_normalized_value():
    frame = pl.DataFrame(
        {
            "code": ["LAB", "DX"],
            "numeric_value": [120.0, None],
            "numeric_value_normalized": [0.75, None],
        }
    )
    result = prepare_continuous_numeric_values(frame)
    assert result["numeric_value"].to_list() == [0.75, None]


def test_continuous_ingestion_rejects_non_normalized_values():
    frame = pl.DataFrame({"numeric_value_normalized": [1.2]})
    with pytest.raises(ValueError, match=r"in \[0, 1\]"):
        prepare_continuous_numeric_values(frame)


def test_dynamic_padding_uses_nan_for_continuous_values_and_targets():
    batch = [
        {
            "subject_id": 1,
            "code": torch.tensor([5, 6]),
            "numeric_value": torch.tensor([0.2, 0.3]),
            "numeric_target": torch.tensor([0.3, float("nan")]),
        },
        {
            "subject_id": 2,
            "code": torch.tensor([7]),
            "numeric_value": torch.tensor([0.4]),
            "numeric_target": torch.tensor([float("nan")]),
        },
    ]
    padded = dynamic_padding(batch)
    assert torch.isnan(padded["numeric_value"][1, 1])
    assert torch.isnan(padded["numeric_target"][1, 1])


def test_prediction_token_has_missing_continuous_value():
    subject = _subject()
    result = append_predict_token(subject, censor_date_abspos=4.0, predict_token_id=1)
    assert torch.isnan(result["numeric_value"][-1])


def test_ar_continuous_targets_are_shifted_without_leaking_future_value():
    sample = ARPretrainDataset([_subject()], max_len=3, background_length=0)[0]
    torch.testing.assert_close(
        sample["numeric_value"],
        torch.tensor([float("nan"), 0.2, float("nan")]),
        equal_nan=True,
    )
    torch.testing.assert_close(
        sample["numeric_target"],
        torch.tensor([0.2, float("nan"), 0.8]),
        equal_nan=True,
    )


def test_mlm_continuous_target_is_hidden_at_selected_positions():
    dataset = MLMPretrainDataset(
        [_subject()],
        max_len=4,
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
            "D": 8,
        },
        masking_select_ratio=1.0,
        masking_mask_ratio=0.8,
        masking_random_ratio=0.0,
    )
    sample = dataset[0]
    assert torch.isnan(sample["numeric_value"]).all()
    torch.testing.assert_close(
        sample["numeric_target"], _subject()["numeric_value"], equal_nan=True
    )


def test_film_preserves_concepts_when_value_is_missing():
    torch.manual_seed(7)
    layer = ContinuousValueEmbedding(8)
    concepts = torch.randn(2, 3, 8)
    values = torch.tensor([[float("nan"), 0.2, float("nan")], [0.8, float("nan"), 0.5]])
    output = layer(values, concepts)
    missing = ~torch.isfinite(values)
    assert torch.equal(output[missing], concepts[missing])
    assert torch.isfinite(output).all()


def test_film_pretraining_uses_regression_without_bin_classification():
    model = BonsaiPretrain(**_model_config()).eval()
    batch = {
        "code": torch.tensor([[5, 6, 7]]),
        "age": torch.tensor([[40.0, 41.0, 42.0]]),
        "abspos": torch.tensor([[1.0, 2.0, 3.0]]),
        "segment": torch.tensor([[0, 0, 0]]),
        "attention_mask": torch.tensor([[True, True, True]]),
        "numeric_value": torch.tensor([[float("nan"), 0.2, float("nan")]]),
        "target": torch.tensor([[6, 7, -100]]),
        "numeric_target": torch.tensor([[0.2, float("nan"), float("nan")]]),
    }
    output = model(batch)
    _, _, _, losses = compute_pretrain_loss(
        output,
        torch.nn.CrossEntropyLoss(),
        torch.nn.CrossEntropyLoss(),
        torch.nn.MSELoss(),
    )
    assert set(losses) == {"code", "value_regression", "total"}
    assert output["value_prediction"].shape == (1,)


def test_film_config_is_checkpoint_reconstructable_and_rejects_bins():
    normalized = normalize_bonsai_model_config(_model_config())
    assert normalized["value_embedding_mode"] == "film"
    assert normalized["value_bin_vocab_size"] == 0
    with pytest.raises(ValueError, match="requires value_bin_vocab_size=0"):
        normalize_bonsai_model_config({**_model_config(), "value_bin_vocab_size": 5})


def test_film_pretraining_handles_batch_without_numeric_targets():
    model = BonsaiPretrain(**_model_config()).eval()
    batch = {
        "code": torch.tensor([[5, 6, 7]]),
        "age": torch.tensor([[40.0, 41.0, 42.0]]),
        "abspos": torch.tensor([[1.0, 2.0, 3.0]]),
        "segment": torch.tensor([[0, 0, 0]]),
        "attention_mask": torch.tensor([[True, True, True]]),
        "numeric_value": torch.full((1, 3), float("nan")),
        "target": torch.tensor([[6, 7, -100]]),
        "numeric_target": torch.full((1, 3), float("nan")),
    }
    output = model(batch)
    loss, _, _, losses = compute_pretrain_loss(
        output,
        torch.nn.CrossEntropyLoss(),
        torch.nn.CrossEntropyLoss(),
        torch.nn.MSELoss(),
    )
    assert set(losses) == {"code", "total"}
    assert torch.isfinite(loss)


def test_film_encoder_transfers_strictly_from_pretrain_to_finetune():
    pretrain = BonsaiPretrain(**_model_config())
    finetune = BonsaiFinetune(**_model_config(), predict_token_id=1)
    lightning_state = {
        f"model.{key}": value for key, value in pretrain.state_dict().items()
    }
    load_pretrained_encoder_checked(finetune, lightning_state)
    torch.testing.assert_close(
        finetune.embeddings.continuous_value_embedding.gamma.weight,
        pretrain.embeddings.continuous_value_embedding.gamma.weight,
    )


def test_film_pretraining_encoder_transfers_without_scalar_head():
    pretrain = BonsaiPretrain(**_model_config())
    finetune = BonsaiFinetune(**_model_config(), predict_token_id=1)
    state = {f"model.{key}": value for key, value in pretrain.state_dict().items()}
    load_pretrained_encoder_checked(finetune, state)
    assert torch.equal(
        finetune.embeddings.continuous_value_embedding.gamma.weight,
        pretrain.embeddings.continuous_value_embedding.gamma.weight,
    )
