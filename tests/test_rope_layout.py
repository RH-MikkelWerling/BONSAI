"""RoPE layout must follow the trained weights, not the runtime backend.

flash_attn's RotaryEmbedding defaults to ``interleaved=False`` (rotate
``(i, i + d/2)``) while the torchtune SDPA path rotates adjacent pairs. This
fork's flash path used the flash default; upstream FGA-DIKU switched to
``interleaved=True`` on 2026-09-29 (a3fa6ac). Loading weights under the wrong
layout is silent, so the layout is recorded in model metadata and honoured by
SDPA. Unrecorded external checkpoints follow current upstream conventions.
"""

from pathlib import Path

import pytest
import torch

from bonsai.functional.checkpointing import (
    MODEL_CONFIG_KEY,
    load_finetune_model_from_checkpoint,
)
from bonsai.functional.model_config import normalize_bonsai_model_config
from bonsai.functional.pretrain_transfer import (
    prepare_pretrain_transfer,
    source_pretrain_model_config,
)
from bonsai.modules.networks.bonsai_nets import (
    BonsaiBase,
    BonsaiFinetune,
    BonsaiPretrain,
)
from bonsai.modules.networks.components.layer import attn_types
from bonsai.modules.networks.components.rope import RotaryPositionalEmbeddings
from bonsai.run.export_external_encoder import export_external_encoder
from opera.compat.bonsai import build_bonsai_encoder

HIDDEN, HEADS = 16, 2
HEAD_DIM = HIDDEN // HEADS


def _config(**overrides):
    config = dict(
        vocab_size=12,
        max_seqlen=32,
        hidden_size=HIDDEN,
        num_layers=2,
        num_attention_heads=HEADS,
        bias=False,
        dropout=0.0,
        attention_dropout=0.0,
        causal=True,
        attn_type="sdpa",
        abspos_encoding="scaled_time2vec",
    )
    config.update(overrides)
    return config


def _batch(batch_size=2, seqlen=9):
    generator = torch.Generator().manual_seed(3)
    attention_mask = torch.ones(batch_size, seqlen, dtype=torch.bool)
    attention_mask[1, 6:] = False
    return {
        "code": torch.randint(5, 12, (batch_size, seqlen), generator=generator),
        "age": torch.rand(batch_size, seqlen, generator=generator) * 80,
        "abspos": torch.rand(batch_size, seqlen, generator=generator) * 4e5,
        "segment": torch.arange(seqlen).repeat(batch_size, 1),
        "attention_mask": attention_mask,
    }


def _flash_reference(x, theta):
    """flash_attn apply_rotary_emb_torch with interleaved=False."""
    positions = torch.arange(x.shape[1], dtype=torch.float32)
    angles = torch.outer(positions, theta)
    cos = torch.cat((angles.cos(), angles.cos()), dim=-1)[None, :, None, :]
    sin = torch.cat((angles.sin(), angles.sin()), dim=-1)[None, :, None, :]
    x1, x2 = x.chunk(2, dim=-1)
    rotated = torch.cat((-x2, x1), dim=-1)
    return x * cos + rotated * sin


def _interleave_permutation(dim):
    """Index map taking a non-interleaved head vector to interleaved order."""
    half = dim // 2
    order = torch.empty(dim, dtype=torch.long)
    order[0::2] = torch.arange(half)
    order[1::2] = torch.arange(half, dim)
    return order


def test_non_interleaved_sdpa_rope_matches_flash_attn_formula():
    rope = RotaryPositionalEmbeddings(dim=HEAD_DIM, max_seq_len=32, interleaved=False)
    x = torch.randn(2, 11, HEADS, HEAD_DIM)
    torch.testing.assert_close(rope(x), _flash_reference(x, rope.theta))


def test_interleaved_and_non_interleaved_are_the_same_rotation_up_to_permutation():
    order = _interleave_permutation(HEAD_DIM)
    x = torch.randn(2, 7, HEADS, HEAD_DIM)
    half = RotaryPositionalEmbeddings(dim=HEAD_DIM, interleaved=False)(x)
    pairs = RotaryPositionalEmbeddings(dim=HEAD_DIM, interleaved=True)(x[..., order])
    torch.testing.assert_close(pairs, half[..., order])


def test_permuting_qk_weights_converts_a_model_between_layouts():
    torch.manual_seed(0)
    half_model = BonsaiBase(**_config(rope_interleaved=False)).eval()
    pair_model = BonsaiBase(**_config(rope_interleaved=True)).eval()
    state = half_model.state_dict()
    order = _interleave_permutation(HEAD_DIM)
    head_order = torch.cat([head * HEAD_DIM + order for head in range(HEADS)])
    for key in [k for k in state if k.endswith("mha.Wqkv.weight")]:
        weight = state[key].clone()
        weight[:HIDDEN] = weight[:HIDDEN][head_order]
        weight[HIDDEN : 2 * HIDDEN] = weight[HIDDEN : 2 * HIDDEN][head_order]
        state[key] = weight
    pair_model.load_state_dict(state)
    batch = _batch()
    mask = batch["attention_mask"]
    with torch.no_grad():
        torch.testing.assert_close(
            pair_model.encode(batch)[mask], half_model.encode(batch)[mask]
        )
        # Without the permutation the same weights behave differently.
        pair_model.load_state_dict(half_model.state_dict())
        assert not torch.allclose(
            pair_model.encode(batch)[mask], half_model.encode(batch)[mask], atol=1e-4
        )


@pytest.mark.parametrize(
    ("saved", "expected"),
    [
        ({"attn_type": "flash"}, False),
        ({"attn_type": "sdpa"}, True),
        ({"attn_type": "flash", "rope_interleaved": True}, True),
        ({"attn_type": "sdpa", "rope_interleaved": False}, False),
    ],
)
def test_layout_resolves_from_saved_backend_before_runtime_override(saved, expected):
    config = normalize_bonsai_model_config(_config(**saved), attn_type="sdpa")
    assert config["attn_type"] == "sdpa"
    assert config["rope_interleaved"] is expected
    encoder = build_bonsai_encoder(_config(**saved), attn_type="sdpa")
    assert encoder.hparams["rope_interleaved"] is expected


def test_constructed_models_record_their_layout():
    assert BonsaiBase(**_config()).hparams["rope_interleaved"] is True
    assert (
        BonsaiPretrain(**_config(rope_interleaved=False)).hparams["rope_interleaved"]
        is False
    )


def test_local_flash_checkpoint_reloaded_under_sdpa_keeps_non_interleaved(tmp_path):
    torch.manual_seed(1)
    reference = BonsaiFinetune(
        **_config(rope_interleaved=False), predict_token_id=1
    ).eval()
    saved = {k: v for k, v in reference.hparams.items() if k != "rope_interleaved"}
    saved["attn_type"] = "flash"  # pre-layout local flash checkpoint
    path = tmp_path / "flash.ckpt"
    torch.save(
        {
            "hyper_parameters": {
                MODEL_CONFIG_KEY: saved,
                "model_class": "BonsaiFinetune",
            },
            "state_dict": {f"model.{k}": v for k, v in reference.state_dict().items()},
        },
        path,
    )
    loaded = load_finetune_model_from_checkpoint(str(path), attn_type="sdpa").eval()
    assert loaded.hparams["rope_interleaved"] is False
    batch = _batch()
    with torch.no_grad():
        torch.testing.assert_close(loaded.encode(batch), reference.encode(batch))


def _external_bundle(tmp_path: Path, *, saved_extra, interleaved, upstream=False):
    vocab = {
        "[PAD]": 0,
        "[CLS]": 1,
        "[SEP]": 2,
        "[UNK]": 3,
        "[MASK]": 4,
        **{f"DIAGNOSIS//{i}": 5 + i for i in range(7)},
    }
    torch.manual_seed(2)
    source = BonsaiPretrain(
        **_config(vocab_size=len(vocab), rope_interleaved=interleaved)
    ).eval()
    saved = {k: v for k, v in source.hparams.items() if k != "rope_interleaved"}
    if upstream:
        # Current FGA-DIKU checkpoints: flat hparams, no architecture tag,
        # no abspos_encoding (Time2Vec over kilo-hours) and no value mode.
        for key in (
            "architecture_version",
            "abspos_encoding",
            "attention_dropout",
            "value_bin_vocab_size",
        ):
            saved.pop(key, None)
        saved.update(value_embedding_mode=None, learning_rate=1e-4)
    saved.update(saved_extra)
    torch.save(vocab, tmp_path / "source_vocab.pt")
    torch.save(vocab, tmp_path / "target_vocab.pt")
    hyper_parameters = saved if upstream else {MODEL_CONFIG_KEY: saved}
    torch.save(
        {
            "hyper_parameters": hyper_parameters,
            "state_dict": {f"model.{k}": v for k, v in source.state_dict().items()},
        },
        tmp_path / "source.ckpt",
    )
    return (
        source,
        tmp_path / "source.ckpt",
        tmp_path / "source_vocab.pt",
        tmp_path / "target_vocab.pt",
    )


def test_upstream_checkpoints_follow_current_upstream_conventions(tmp_path):
    _, ckpt, _, _ = _external_bundle(
        tmp_path, saved_extra={"attn_type": "flash"}, interleaved=True, upstream=True
    )
    config = source_pretrain_model_config(ckpt)
    assert config["rope_interleaved"] is True
    assert config["abspos_encoding"] == "scaled_time2vec"
    assert config["value_embedding_mode"] == "legacy"
    # A checkpoint from before upstream a3fa6ac can be declared explicitly.
    assert (
        source_pretrain_model_config(ckpt, "non_interleaved")["rope_interleaved"]
        is False
    )


def test_local_flash_checkpoint_without_layout_is_non_interleaved(tmp_path):
    _, ckpt, _, _ = _external_bundle(
        tmp_path, saved_extra={"attn_type": "flash"}, interleaved=False
    )
    assert source_pretrain_model_config(ckpt)["rope_interleaved"] is False
    with pytest.raises(ValueError, match="contradicts"):
        source_pretrain_model_config(ckpt, "interleaved")


def test_declared_layout_cannot_contradict_recorded_or_sdpa_layout(tmp_path):
    _, ckpt, _, _ = _external_bundle(
        tmp_path,
        saved_extra={"attn_type": "flash", "rope_interleaved": True},
        interleaved=True,
    )
    with pytest.raises(ValueError, match="contradicts"):
        source_pretrain_model_config(ckpt, "non_interleaved")
    sdpa_dir = tmp_path / "sdpa"
    sdpa_dir.mkdir()
    _, sdpa_ckpt, _, _ = _external_bundle(
        sdpa_dir, saved_extra={"attn_type": "sdpa"}, interleaved=True, upstream=True
    )
    assert source_pretrain_model_config(sdpa_ckpt)["rope_interleaved"] is True
    with pytest.raises(ValueError, match="contradicts"):
        source_pretrain_model_config(sdpa_ckpt, "non_interleaved")


def test_transfer_rejects_a_target_built_with_the_other_layout(tmp_path):
    _, ckpt, source_vocab, target_vocab = _external_bundle(
        tmp_path, saved_extra={"attn_type": "flash"}, interleaved=False
    )
    target = BonsaiPretrain(**_config(rope_interleaved=True))
    with pytest.raises(ValueError, match="rope_interleaved"):
        prepare_pretrain_transfer(
            target,
            ckpt,
            source_vocab,
            target_vocab,
            load_weights=True,
        )


@pytest.mark.parametrize(
    ("layout", "interleaved"), [("non_interleaved", False), ("interleaved", True)]
)
def test_external_export_preserves_declared_flash_layout_under_sdpa(
    tmp_path, layout, interleaved
):
    source, ckpt, source_vocab, target_vocab = _external_bundle(
        tmp_path,
        saved_extra={"attn_type": "flash", "abspos_encoding": "scaled_time2vec"},
        interleaved=interleaved,
        upstream=True,
    )
    result = export_external_encoder(
        ckpt,
        source_vocab,
        target_vocab,
        tmp_path / "export",
        source_rope_layout=layout,
    )
    payload = torch.load(result["checkpoint"], weights_only=False)
    config = payload["hyper_parameters"]["model_config"]
    assert config["attn_type"] == "sdpa"
    assert config["rope_interleaved"] is interleaved
    exported = build_bonsai_encoder(config).eval()
    exported.load_state_dict(
        {k.removeprefix("model."): v for k, v in payload["state_dict"].items()}
    )
    batch = _batch()
    with torch.no_grad():
        torch.testing.assert_close(exported.encode(batch), source.encode(batch))


@pytest.mark.skipif(
    "flash" not in attn_types or not torch.cuda.is_available(),
    reason="numerical flash/SDPA parity requires flash-attn on a CUDA GPU",
)
@pytest.mark.parametrize("interleaved", [False, True])
def test_flash_and_sdpa_backends_agree_for_the_same_layout(interleaved):
    torch.manual_seed(4)
    flash = BonsaiBase(**_config(attn_type="flash", rope_interleaved=interleaved))
    sdpa = BonsaiBase(**_config(attn_type="sdpa", rope_interleaved=interleaved))
    sdpa.load_state_dict(flash.state_dict())
    flash, sdpa = flash.cuda().half().eval(), sdpa.cuda().half().eval()
    batch = {k: v.cuda() for k, v in _batch().items()}
    mask = batch["attention_mask"]
    with torch.no_grad():
        torch.testing.assert_close(
            flash.encode(batch)[mask].float(),
            sdpa.encode(batch)[mask].float(),
            atol=2e-2,
            rtol=2e-2,
        )
