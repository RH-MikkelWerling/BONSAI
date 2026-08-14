from pathlib import Path

import pytest
import torch

from bonsai.functional.model_config import normalize_bonsai_model_config
from bonsai.functional.model_config import validate_pretraining_attention
from bonsai.modules.datasets.PretrainDataset import (
    ARPretrainDataset,
    MLMPretrainDataset,
)
from bonsai.modules.networks.bonsai_nets import (
    BonsaiBase,
    BonsaiFinetune,
    BonsaiPretrain,
    pack_valid_tokens,
    unpack_valid_tokens,
)
from bonsai.modules.networks.components.embeddings import Time2Vec
from opera.modules.networks.opera_nets import OperaContrastiveModel
from opera.run.evaluate import extract_patient_embeddings


def _small_model_config():
    return {
        "vocab_size": 12,
        "max_seqlen": 8,
        "hidden_size": 8,
        "num_layers": 1,
        "num_attention_heads": 2,
        "bias": False,
        "dropout": 0.0,
        "attention_dropout": 0.0,
        "causal": False,
        "attn_type": "sdpa",
    }


def test_time2vec_stays_finite_for_epoch_hours_under_fp16_autocast():
    layer = Time2Vec(output_dim=8, clip_range=100)
    epoch_hours = torch.tensor([[500_000.0]])

    with torch.autocast(device_type="cpu", dtype=torch.float16):
        output = layer(epoch_hours)

    assert output.dtype == torch.float32
    assert torch.isfinite(output).all()


def test_legacy_yaml_names_translate_to_native_flash_config():
    config = normalize_bonsai_model_config(
        {
            "vocab_size": 12,
            "max_position_embeddings": 8,
            "hidden_size": 8,
            "num_hidden_layers": 1,
            "num_attention_heads": 2,
            "embedding_dropout": 0.2,
            "is_causal": True,
        }
    )

    assert config["max_seqlen"] == 8
    assert config["num_layers"] == 1
    assert config["causal"] is True
    assert config["dropout"] == pytest.approx(0.2)
    assert config["attn_type"] == "flash"


def test_flash_varlen_pack_round_trip_ignores_padding():
    states = torch.arange(3 * 5 * 2, dtype=torch.float32).reshape(3, 5, 2)
    mask = torch.tensor(
        [
            [1, 1, 1, 1, 1],
            [1, 1, 0, 0, 0],
            [1, 1, 1, 0, 0],
        ],
        dtype=torch.bool,
    )

    packed, cu_seqlens = pack_valid_tokens(states, mask)
    restored = unpack_valid_tokens(packed, mask, hidden_size=2)

    assert cu_seqlens.tolist() == [0, 5, 7, 10]
    assert torch.equal(restored[mask], states[mask])
    assert torch.count_nonzero(restored[~mask]) == 0


def test_sdpa_valid_representations_are_invariant_to_extra_right_padding():
    torch.manual_seed(4)
    model = BonsaiFinetune(**_small_model_config(), predict_token_id=1).eval()
    short = {
        "code": torch.tensor([[2, 3, 1]]),
        "age": torch.tensor([[40.0, 41.0, 41.0]]),
        "abspos": torch.tensor([[1.0, 2.0, 2.0]]),
        "segment": torch.tensor([[0, 1, 1]]),
        "attention_mask": torch.tensor([[True, True, True]]),
    }
    padded = {
        "code": torch.tensor([[2, 3, 1, 0, 0]]),
        "age": torch.tensor([[40.0, 41.0, 41.0, 0.0, 0.0]]),
        "abspos": torch.tensor([[1.0, 2.0, 2.0, 0.0, 0.0]]),
        "segment": torch.tensor([[0, 1, 1, 0, 0]]),
        "attention_mask": torch.tensor([[True, True, True, False, False]]),
    }

    with torch.no_grad():
        short_rep = model.get_pooled_representation(short)
        padded_rep = model.get_pooled_representation(padded)

    assert torch.allclose(short_rep, padded_rep, atol=1e-6)


def test_causal_sdpa_prevents_future_tokens_from_changing_prefix_states():
    """Regression test for upstream #349's causal-SDPA contract."""
    torch.manual_seed(9)
    config = _small_model_config()
    config["causal"] = True
    model = BonsaiBase(**config).eval()
    common = {
        "age": torch.tensor([[40.0, 41.0, 42.0, 43.0, 44.0]]),
        "abspos": torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0]]),
        "segment": torch.tensor([[0, 1, 1, 1, 1]]),
        "attention_mask": torch.ones((1, 5), dtype=torch.bool),
    }
    first = {**common, "code": torch.tensor([[2, 3, 4, 5, 6]])}
    changed_future = {**common, "code": torch.tensor([[2, 3, 4, 10, 11]])}

    with torch.no_grad():
        first_states = model(first)
        changed_states = model(changed_future)

    assert torch.allclose(first_states[:, :3], changed_states[:, :3], atol=1e-6)
    assert not torch.allclose(first_states[:, 3:], changed_states[:, 3:])


def test_evaluation_extracts_native_prediction_token_representation():
    model = BonsaiFinetune(**_small_model_config(), predict_token_id=1).eval()
    batch = {
        "code": torch.tensor([[2, 3, 1]]),
        "age": torch.tensor([[40.0, 41.0, 41.0]]),
        "abspos": torch.tensor([[1.0, 2.0, 2.0]]),
        "segment": torch.tensor([[0, 1, 1]]),
        "attention_mask": torch.tensor([[True, True, True]]),
    }

    with torch.no_grad():
        extracted = extract_patient_embeddings(model, batch)
        expected = model.get_pooled_representation(batch)

    assert torch.allclose(extracted, expected)


def test_native_encoder_accepts_differentiable_token_embedding_override():
    model = BonsaiFinetune(**_small_model_config(), predict_token_id=1).eval()
    token_embeddings = torch.randn(1, 3, 8, requires_grad=True)
    batch = {
        "code": torch.tensor([[2, 3, 1]]),
        "attention_mask": torch.tensor([[True, True, True]]),
        "token_embeddings": token_embeddings,
    }

    model(batch).sum().backward()

    assert token_embeddings.grad is not None
    assert torch.isfinite(token_embeddings.grad).all()


def test_native_pretrain_model_predicts_combined_binned_values():
    model = BonsaiPretrain(**_small_model_config(), value_bin_vocab_size=5).eval()
    batch = {
        "code": torch.tensor([[2, 3, 4, 5]]),
        "age": torch.tensor([[40.0, 41.0, 42.0, 43.0]]),
        "abspos": torch.tensor([[1.0, 2.0, 3.0, 4.0]]),
        "segment": torch.tensor([[0, 0, 1, 1]]),
        "attention_mask": torch.tensor([[True, True, True, True]]),
        "value_bin": torch.tensor([[0, 2, 0, 3]]),
        "value_normalized": torch.tensor([[0.0, 0.5, 0.0, 0.8]]),
        "value_present": torch.tensor([[False, True, False, True]]),
        "target": torch.tensor([[3, -100, 5, -100]]),
        "target_value_mask": torch.tensor([[False, True, False, True]]),
        "target_value_bin": torch.tensor([[-100, 2, -100, 3]]),
        "target_value_normalized": torch.tensor([[0.0, 0.5, 0.0, 0.8]]),
    }

    with torch.no_grad():
        output = model(batch)

    assert output["logits"].shape == (2, 12)
    assert torch.equal(output["labels"], torch.tensor([3, 5]))
    assert output["value_bin_logits"].shape == (2, 5)
    assert torch.equal(output["target_value_bin"], torch.tensor([2, 3]))
    assert output["value_prediction"].shape == (2,)
    torch.testing.assert_close(
        output["target_value_normalized"], torch.tensor([0.5, 0.8])
    )


def test_combined_binning_uses_ce_plus_scalar_mse_without_bin_ce():
    from bonsai.modules.lightningmodules.PretrainModule import compute_pretrain_loss

    model = BonsaiPretrain(
        **_small_model_config(),
        value_bin_vocab_size=1,
        value_embedding_mode="combined_binning",
    ).eval()
    batch = {
        "code": torch.tensor([[2, 3, 4]]),
        "age": torch.tensor([[40.0, 41.0, 42.0]]),
        "abspos": torch.tensor([[1.0, 2.0, 3.0]]),
        "segment": torch.tensor([[0, 0, 0]]),
        "attention_mask": torch.tensor([[True, True, True]]),
        "value_bin": torch.tensor([[0, 0, 3]]),
        "value_normalized": torch.tensor([[0.0, 0.0, 0.3]]),
        "value_present": torch.tensor([[False, False, True]]),
        "target": torch.tensor([[3, -100, 5]]),
        "target_value_mask": torch.tensor([[False, True, False]]),
        "target_value_bin": torch.tensor([[-100, 3, -100]]),
        "target_value_normalized": torch.tensor([[0.0, 0.3, 0.0]]),
    }

    output = model(batch)
    _, _, _, losses = compute_pretrain_loss(
        output,
        torch.nn.CrossEntropyLoss(),
        torch.nn.CrossEntropyLoss(),
        torch.nn.MSELoss(),
    )

    assert set(losses) == {"code", "value_regression", "total"}
    assert torch.isfinite(losses["total"])


def test_primary_training_configs_default_to_flash_attention():
    root = Path(__file__).parents[1]
    paths = [
        root / "configs" / "pretrain.yaml",
        root / "configs" / "finetune.yaml",
        root / "opera" / "configs" / "finetune.yaml",
    ]
    for path in paths:
        assert "attn_type: flash" in path.read_text(encoding="utf-8")

    # The local DALY-CARE V100 cannot run FlashAttention; SDPA is the
    # architecture-equivalent operational fallback for this one config.
    daly = root / "opera" / "configs" / "daly_care_pretrain.yaml"
    assert "attn_type: sdpa" in daly.read_text(encoding="utf-8")


def test_autoregressive_pretraining_rejects_noncausal_attention():
    with pytest.raises(ValueError, match="requires causal=True"):
        validate_pretraining_attention(ARPretrainDataset, causal=False)
    validate_pretraining_attention(ARPretrainDataset, causal=True)
    validate_pretraining_attention(MLMPretrainDataset, causal=False)


def test_opera_embedding_model_accepts_native_bonsai_encoder_output():
    config = _small_model_config()
    encoder = BonsaiBase(**config)
    model = OperaContrastiveModel(
        encoder=encoder,
        outcome_names=["mortality"],
        hidden_size=config["hidden_size"],
        projection_hidden_dim=8,
        projection_dim=4,
        pooling="cls_last",
    ).eval()
    batch = {
        "code": torch.tensor([[2, 3, 1], [4, 1, 0]]),
        "age": torch.tensor([[40.0, 41.0, 41.0], [50.0, 50.0, 0.0]]),
        "abspos": torch.tensor([[1.0, 2.0, 2.0], [4.0, 4.0, 0.0]]),
        "segment": torch.tensor([[0, 1, 1], [0, 1, 0]]),
        "attention_mask": torch.tensor([[True, True, True], [True, True, False]]),
    }

    with torch.no_grad():
        embeddings = model.get_embeddings(batch)

    assert embeddings.shape == (2, 4)
    assert torch.allclose(embeddings.norm(dim=1), torch.ones(2), atol=1e-6)
