import pytest
import torch
from transformers import ModernBertConfig

from bonsai.functional.checkpointing import (
    MODEL_CONFIG_KEY,
    load_finetune_model_from_checkpoint,
)
from bonsai.modules.networks.bonsai_nets import BonsaiFinetune


def _small_config():
    return ModernBertConfig(
        vocab_size=17,
        hidden_size=8,
        num_hidden_layers=1,
        num_attention_heads=1,
        intermediate_size=16,
        type_vocab_size=4,
        max_position_embeddings=32,
        pad_token_id=0,
        cls_token_id=1,
        sep_token_id=2,
        embedding_dropout=0.0,
        is_causal=False,
    )


def test_finetune_checkpoint_reconstructs_exact_config_and_shapes(tmp_path):
    model = BonsaiFinetune(_small_config())
    ckpt_path = tmp_path / "model.ckpt"
    torch.save(
        {
            "hyper_parameters": {
                MODEL_CONFIG_KEY: model.config.to_dict(),
                "model_class": model.__class__.__name__,
            },
            "state_dict": {
                f"model.{key}": value
                for key, value in model.state_dict().items()
            },
        },
        ckpt_path,
    )

    loaded = load_finetune_model_from_checkpoint(str(ckpt_path), strict=True)

    assert loaded.config.to_dict() == model.config.to_dict()
    assert {
        key: tuple(value.shape)
        for key, value in loaded.state_dict().items()
    } == {
        key: tuple(value.shape)
        for key, value in model.state_dict().items()
    }


def test_finetune_checkpoint_without_full_config_fails_clearly(tmp_path):
    ckpt_path = tmp_path / "old.ckpt"
    torch.save({"hyper_parameters": {"hidden_size": 8}, "state_dict": {}}, ckpt_path)

    with pytest.raises(ValueError, match="missing 'model_config'"):
        load_finetune_model_from_checkpoint(str(ckpt_path), strict=True)
