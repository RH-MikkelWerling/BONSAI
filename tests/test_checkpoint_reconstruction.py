import pytest
import torch

from bonsai.functional.checkpointing import (
    MODEL_CONFIG_KEY,
    MODEL_INIT_CONFIG_KEY,
    clean_lightning_state_dict,
    load_finetune_model_from_checkpoint,
    load_joint_model_from_checkpoint,
    load_pretrained_encoder_checked,
)
from bonsai.modules.networks.bonsai_nets import BonsaiFinetune, BonsaiPretrain
from bonsai.functional.model_config import LegacyCheckpointError
from opera.compat.bonsai import BonsaiEncoder
from opera.modules.networks.joint_finetune_net import JointFinetuneModel


def _small_config():
    return {
        "vocab_size": 17,
        "hidden_size": 8,
        "num_layers": 1,
        "num_attention_heads": 1,
        "max_seqlen": 32,
        "bias": False,
        "dropout": 0.0,
        "attention_dropout": 0.0,
        "causal": False,
        "attn_type": "sdpa",
    }


def test_finetune_checkpoint_reconstructs_exact_config_and_shapes(tmp_path):
    model = BonsaiFinetune(**_small_config(), predict_token_id=1)
    ckpt_path = tmp_path / "model.ckpt"
    torch.save(
        {
            "hyper_parameters": {
                MODEL_CONFIG_KEY: dict(model.hparams),
                "model_class": model.__class__.__name__,
            },
            "state_dict": {
                **{f"model.{key}": value for key, value in model.state_dict().items()},
                "train_loss.pos_weight": torch.tensor([2.0]),
                "val_loss.pos_weight": torch.tensor([2.0]),
            },
        },
        ckpt_path,
    )

    loaded = load_finetune_model_from_checkpoint(str(ckpt_path), strict=True)

    assert loaded.hparams == model.hparams
    assert {key: tuple(value.shape) for key, value in loaded.state_dict().items()} == {
        key: tuple(value.shape) for key, value in model.state_dict().items()
    }


def test_clean_lightning_state_dict_preserves_bare_state_dict():
    state = {"encoder.weight": torch.ones(2, 2)}

    cleaned = clean_lightning_state_dict(state)

    assert set(cleaned) == {"encoder.weight"}


def test_joint_checkpoint_reconstructs_saved_model_settings(tmp_path):
    config = _small_config()
    model = JointFinetuneModel(
        encoder=BonsaiEncoder(**config),
        outcome_names=["aki_30d", "mortality_1y"],
        hidden_size=config["hidden_size"],
        pooling="cls_last",
        freeze_encoder=True,
        dropout=0.25,
    )
    ckpt_path = tmp_path / "joint.ckpt"
    torch.save(
        {
            "hyper_parameters": {
                MODEL_CONFIG_KEY: dict(model.encoder.hparams),
                MODEL_INIT_CONFIG_KEY: {
                    "hidden_size": config["hidden_size"],
                    "pooling": "cls_last",
                    "freeze_encoder": True,
                    "dropout": 0.25,
                },
                "model_class": model.__class__.__name__,
                "outcome_names": model.outcome_names,
            },
            "state_dict": {
                **{f"model.{key}": value for key, value in model.state_dict().items()},
                "val_auroc.aki_30d._update_count": torch.tensor(1),
            },
        },
        ckpt_path,
    )

    loaded = load_joint_model_from_checkpoint(str(ckpt_path), strict=True)

    assert loaded.outcome_names == model.outcome_names
    assert loaded.pooling == "cls_last"
    assert loaded.freeze_encoder is True
    assert loaded.dropout.p == pytest.approx(0.25)
    assert {key: tuple(value.shape) for key, value in loaded.state_dict().items()} == {
        key: tuple(value.shape) for key, value in model.state_dict().items()
    }


def test_finetune_checkpoint_without_full_config_fails_clearly(tmp_path):
    ckpt_path = tmp_path / "old.ckpt"
    torch.save({"hyper_parameters": {"hidden_size": 8}, "state_dict": {}}, ckpt_path)

    with pytest.raises(ValueError, match="missing 'model_config'"):
        load_finetune_model_from_checkpoint(str(ckpt_path), strict=True)


def test_pretrained_encoder_load_allows_only_new_finetune_head():
    config = _small_config()
    source = BonsaiPretrain(**config)
    target = BonsaiFinetune(**config, predict_token_id=1)
    state_dict = {f"model.{key}": value for key, value in source.state_dict().items()}

    load_pretrained_encoder_checked(target, state_dict)


def test_pretrained_encoder_load_rejects_missing_backbone_key():
    config = _small_config()
    source = BonsaiPretrain(**config)
    target = BonsaiFinetune(**config, predict_token_id=1)
    state_dict = {f"model.{key}": value for key, value in source.state_dict().items()}
    backbone_key = next(
        key for key in state_dict if not key.startswith("model.pretrain_head.")
    )
    del state_dict[backbone_key]

    with pytest.raises(RuntimeError, match="Missing encoder keys"):
        load_pretrained_encoder_checked(target, state_dict)


def test_legacy_modernbert_checkpoint_fails_with_migration_message(tmp_path):
    ckpt_path = tmp_path / "legacy.ckpt"
    torch.save(
        {
            "hyper_parameters": {
                MODEL_CONFIG_KEY: {
                    "vocab_size": 17,
                    "hidden_size": 8,
                    "num_hidden_layers": 1,
                    "num_attention_heads": 1,
                    "max_position_embeddings": 32,
                }
            },
            "state_dict": {},
        },
        ckpt_path,
    )

    with pytest.raises(LegacyCheckpointError, match="legacy ModernBERT"):
        load_finetune_model_from_checkpoint(str(ckpt_path))
