"""Audited native-BONSAI checkpoint transfer into a local MEDS vocabulary."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import torch

from bonsai.functional.checkpointing import (
    extract_encoder_state_dict,
    get_saved_encoder_config,
)
from bonsai.functional.model_config import normalize_bonsai_model_config


_CORE_CONFIG = (
    "max_seqlen", "hidden_size", "num_layers", "num_attention_heads",
    "bias", "causal", "value_bin_vocab_size", "value_embedding_mode",
    "abspos_encoding",
)
_SPECIAL_TOKENS = ("[PAD]", "[CLS]", "[SEP]", "[UNK]", "[MASK]")
_CODE_WEIGHT = "embeddings.code_embedding.weight"


def source_pretrain_model_config(checkpoint_path: str | Path) -> dict:
    """Read the native encoder specification without constructing a model."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict) or "hyper_parameters" not in checkpoint:
        raise ValueError("Source must be a BONSAI Lightning pretraining checkpoint")
    return get_saved_encoder_config(checkpoint["hyper_parameters"])


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _vocabulary(path: Path) -> dict[str, int]:
    value = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(value, dict) or not value:
        raise ValueError(f"Vocabulary must be a nonempty token-to-ID mapping: {path}")
    vocabulary = {}
    for token, index in value.items():
        if not isinstance(token, str) or not isinstance(index, int):
            raise ValueError(f"Vocabulary must map string tokens to integer IDs: {path}")
        vocabulary[token] = index
    if sorted(vocabulary.values()) != list(range(len(vocabulary))):
        raise ValueError(f"Vocabulary IDs must be contiguous from zero: {path}")
    missing = set(_SPECIAL_TOKENS) - set(vocabulary)
    if missing:
        raise ValueError(f"Vocabulary lacks required special tokens: {sorted(missing)}")
    return vocabulary


def prepare_pretrain_transfer(
    model: Any,
    checkpoint_path: str | Path,
    source_vocab_path: str | Path,
    target_vocab_path: str | Path,
    *,
    min_token_coverage: float = 0.0,
    load_weights: bool = False,
) -> dict:
    """Check architecture/token identity and optionally initialize a local model.

    A different vocabulary is allowed only through exact token-string mapping.
    Non-vocabulary encoder tensors must match strictly. Missing local tokens
    retain the target model's random initialization for domain adaptation.
    """
    checkpoint_path = Path(checkpoint_path)
    source_vocab_path = Path(source_vocab_path)
    target_vocab_path = Path(target_vocab_path)
    if not 0 <= min_token_coverage <= 1:
        raise ValueError("min_token_coverage must lie between zero and one")
    for path in (checkpoint_path, source_vocab_path, target_vocab_path):
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(f"Missing or empty transfer input: {path}")
    source_vocab = _vocabulary(source_vocab_path)
    target_vocab = _vocabulary(target_vocab_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict) or not {
        "state_dict", "hyper_parameters"
    }.issubset(checkpoint):
        raise ValueError("Source must be a BONSAI Lightning pretraining checkpoint")
    source_config = get_saved_encoder_config(checkpoint["hyper_parameters"])
    target_config = normalize_bonsai_model_config(model.hparams)
    mismatch = {
        key: (source_config[key], target_config[key])
        for key in _CORE_CONFIG if source_config[key] != target_config[key]
    }
    if mismatch:
        raise ValueError(f"Source and target encoder architectures differ: {mismatch}")
    if source_config["vocab_size"] != len(source_vocab):
        raise ValueError("Source checkpoint vocabulary size differs from source vocabulary")
    if target_config["vocab_size"] != len(target_vocab):
        raise ValueError("Target model vocabulary size differs from local vocabulary")
    source_state = extract_encoder_state_dict(checkpoint["state_dict"])
    target_state = model.state_dict()
    target_encoder_keys = {
        key for key in target_state
        if not key.startswith(("pretrain_head.", "value_head.", "value_bin_head."))
    }
    if set(source_state) != target_encoder_keys:
        raise ValueError(
            "Encoder tensor names differ; "
            f"missing={sorted(target_encoder_keys - set(source_state))[:10]}, "
            f"unexpected={sorted(set(source_state) - target_encoder_keys)[:10]}"
        )
    for key in target_encoder_keys - {_CODE_WEIGHT}:
        if source_state[key].shape != target_state[key].shape:
            raise ValueError(f"Encoder tensor shape differs for {key}")
    if source_state[_CODE_WEIGHT].shape != (
        len(source_vocab), target_config["hidden_size"]
    ):
        raise ValueError("Source code embedding shape disagrees with checkpoint config")

    overlap = sorted(set(source_vocab) & set(target_vocab))
    non_special = set(target_vocab) - set(_SPECIAL_TOKENS)
    covered_non_special = non_special & set(source_vocab)
    coverage = len(covered_non_special) / max(len(non_special), 1)
    if coverage < min_token_coverage:
        raise ValueError(
            f"Only {coverage:.1%} of local non-special tokens match source vocabulary; "
            f"required {min_token_coverage:.1%}"
        )
    if load_weights:
        remapped = dict(source_state)
        local_code = target_state[_CODE_WEIGHT].clone()
        source_code = source_state[_CODE_WEIGHT]
        for token in overlap:
            local_code[target_vocab[token]] = source_code[source_vocab[token]]
        remapped[_CODE_WEIGHT] = local_code
        missing, unexpected = model.load_state_dict(remapped, strict=False)
        allowed_missing = {
            key for key in target_state if key.startswith(
                ("pretrain_head.", "value_head.", "value_bin_head.")
            )
        }
        if set(missing) - allowed_missing or unexpected:
            raise RuntimeError(
                f"Pretraining transfer left missing={missing}, unexpected={unexpected}"
            )
    return {
        "status": "loaded" if load_weights else "compatible_not_loaded",
        "checkpoint": str(checkpoint_path.resolve()),
        "checkpoint_sha256": _sha256(checkpoint_path),
        "source_vocabulary": str(source_vocab_path.resolve()),
        "source_vocabulary_sha256": _sha256(source_vocab_path),
        "target_vocabulary": str(target_vocab_path.resolve()),
        "target_vocabulary_sha256": _sha256(target_vocab_path),
        "source_vocabulary_size": len(source_vocab),
        "target_vocabulary_size": len(target_vocab),
        "matched_tokens": len(overlap),
        "local_non_special_token_coverage": coverage,
        "new_local_tokens": len(target_vocab) - len(overlap),
        "source_abspos_encoding": source_config["abspos_encoding"],
        "target_abspos_encoding": target_config["abspos_encoding"],
        "operational_model_differences": {
            key: (source_config[key], target_config[key])
            for key in ("attn_type", "dropout", "attention_dropout")
            if source_config[key] != target_config[key]
        },
    }
