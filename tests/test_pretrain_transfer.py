from pathlib import Path

import pytest
import torch

from bonsai.functional.pretrain_transfer import prepare_pretrain_transfer
from bonsai.run.audit_dapt_to_opera import audit_dapt_to_opera
from bonsai.functional.checkpointing import (
    extract_encoder_state_dict,
    get_saved_encoder_config,
)
from bonsai.modules.networks.bonsai_nets import BonsaiPretrain
from opera.compat.bonsai import build_bonsai_encoder


def _model(vocab_size: int, *, hidden_size: int = 8) -> BonsaiPretrain:
    return BonsaiPretrain(
        vocab_size=vocab_size, max_seqlen=16, hidden_size=hidden_size,
        num_layers=1, num_attention_heads=2, bias=False, dropout=0.0,
        attention_dropout=0.0, causal=True, attn_type="sdpa",
        abspos_encoding="fourier",
    )


def _bundle(tmp_path: Path):
    source_vocab = {
        "[PAD]": 0, "[CLS]": 1, "[SEP]": 2, "[UNK]": 3,
        "[MASK]": 4, "DIAGNOSIS//A": 5,
    }
    target_vocab = {
        "[PAD]": 0, "[SEP]": 1, "[CLS]": 2, "[UNK]": 3,
        "[MASK]": 4, "DIAGNOSIS//A": 5, "MEDICATION//B": 6,
    }
    source_vocab_path = tmp_path / "source_vocab.pt"
    target_vocab_path = tmp_path / "target_vocab.pt"
    checkpoint_path = tmp_path / "source.ckpt"
    torch.save(source_vocab, source_vocab_path)
    torch.save(target_vocab, target_vocab_path)
    source_model = _model(len(source_vocab))
    torch.save({
        "hyper_parameters": {"model_config": source_model.hparams},
        "state_dict": {f"model.{key}": value for key, value in source_model.state_dict().items()},
    }, checkpoint_path)
    return source_model, source_vocab_path, target_vocab_path, checkpoint_path


def test_pretrain_transfer_remaps_tokens_and_preserves_new_local_rows(tmp_path):
    source, source_vocab, target_vocab, checkpoint = _bundle(tmp_path)
    target = _model(7)
    new_token_before = target.embeddings.code_embedding.weight[6].detach().clone()
    report = prepare_pretrain_transfer(
        target, checkpoint, source_vocab, target_vocab, load_weights=True,
    )
    assert report["status"] == "loaded"
    assert report["matched_tokens"] == 6
    assert report["new_local_tokens"] == 1
    assert torch.equal(
        target.embeddings.code_embedding.weight[2],
        source.embeddings.code_embedding.weight[1],
    )
    assert torch.equal(target.embeddings.code_embedding.weight[6], new_token_before)
    layer_key = next(iter(source.layers[0].state_dict()))
    assert torch.equal(
        target.layers[0].state_dict()[layer_key],
        source.layers[0].state_dict()[layer_key],
    )
    # Domain-adapted pretraining uses the ordinary BONSAI checkpoint shape;
    # this is the exact strict encoder load used by OPERA fine-tuning.
    adapted_checkpoint = {
        "hyper_parameters": {"model_config": target.hparams},
        "state_dict": {f"model.{key}": value for key, value in target.state_dict().items()},
    }
    encoder_config = get_saved_encoder_config(adapted_checkpoint["hyper_parameters"])
    encoder = build_bonsai_encoder(encoder_config, vocab_size=7)
    encoder.load_state_dict(
        extract_encoder_state_dict(adapted_checkpoint["state_dict"]), strict=True,
    )
    with torch.no_grad():
        states = encoder.encode({
            "code": torch.tensor([[2, 1]]),
            "age": torch.zeros((1, 2)),
            "abspos": torch.zeros((1, 2)),
            "segment": torch.zeros((1, 2), dtype=torch.long),
            "attention_mask": torch.ones((1, 2), dtype=torch.bool),
        })
    assert states.shape == (1, 2, 8)
    assert torch.isfinite(states).all()


def test_pretrain_transfer_rejects_incompatible_encoder(tmp_path):
    _, source_vocab, target_vocab, checkpoint = _bundle(tmp_path)
    with pytest.raises(ValueError, match="architectures differ"):
        prepare_pretrain_transfer(
            _model(7, hidden_size=16), checkpoint, source_vocab, target_vocab,
        )


def test_pretrain_transfer_rejects_low_token_overlap(tmp_path):
    _, source_vocab, target_vocab, checkpoint = _bundle(tmp_path)
    with pytest.raises(ValueError, match="local non-special tokens"):
        prepare_pretrain_transfer(
            _model(7), checkpoint, source_vocab, target_vocab,
            min_token_coverage=1.0,
        )


def test_domain_adapted_checkpoint_audits_for_opera(tmp_path):
    _, source_vocab, target_vocab, source_checkpoint = _bundle(tmp_path)
    target = _model(7)
    report = prepare_pretrain_transfer(
        target, source_checkpoint, source_vocab, target_vocab, load_weights=True,
    )
    dapt_checkpoint = tmp_path / "dapt.ckpt"
    torch.save({
        "hyper_parameters": {
            "model_config": target.hparams,
            "checkpoint_metadata": {
                "training_stage": "domain_adaptive_pretraining",
                "source_pretraining_transfer": report,
            },
        },
        "state_dict": {f"model.{key}": value for key, value in target.state_dict().items()},
    }, dapt_checkpoint)
    result = audit_dapt_to_opera(dapt_checkpoint, target_vocab)
    assert result["status"] == "opera_strict_load_and_forward_passed"
    with pytest.raises(ValueError, match="vocabulary differs"):
        audit_dapt_to_opera(dapt_checkpoint, source_vocab)


def test_original_upstream_fixed_head_count_is_recovered():
    from bonsai.functional.pretrain_transfer import resolve_source_encoder_config
    raw = dict(vocab_size=8, max_seqlen=32, hidden_size=256, num_layers=2,
               bias=False, dropout=0., causal=True, attn_type='sdpa',
               value_embedding_mode=None)
    result = resolve_source_encoder_config(raw)
    assert result['num_attention_heads'] == 4
    assert result['abspos_encoding'] == 'scaled_time2vec'
    assert result['value_embedding_mode'] == 'legacy'
    assert resolve_source_encoder_config({**raw, 'num_attention_heads': 8})['num_attention_heads'] == 8
    with pytest.raises(ValueError, match='fixed 64-dimensional heads'):
        resolve_source_encoder_config({**raw, 'hidden_size': 255})
