"""Audited native-BONSAI checkpoint transfer into a local MEDS vocabulary."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import torch

from bonsai.functional.checkpointing import (
    MODEL_CONFIG_KEY,
    extract_encoder_state_dict,
)
from bonsai.functional.model_config import (
    NATIVE_ARCHITECTURE_VERSION,
    config_to_dict,
    has_explicit_rope_layout,
    legacy_rope_interleaved,
    normalize_bonsai_model_config,
    require_native_checkpoint_config,
)


_CORE_CONFIG = (
    "max_seqlen", "hidden_size", "num_layers", "num_attention_heads",
    "bias", "causal", "value_bin_vocab_size", "value_embedding_mode",
    "abspos_encoding", "rope_interleaved",
)
ROPE_LAYOUTS = {"interleaved": True, "non_interleaved": False}
_SPECIAL_TOKENS = ("[PAD]", "[CLS]", "[SEP]", "[UNK]", "[MASK]")
_CODE_WEIGHT = "embeddings.code_embedding.weight"


def _raw_source_config(hparams: dict) -> dict:
    return config_to_dict(
        hparams[MODEL_CONFIG_KEY] if MODEL_CONFIG_KEY in hparams else hparams
    )


def _is_local_native(raw: dict) -> bool:
    return raw.get("architecture_version") == NATIVE_ARCHITECTURE_VERSION


def resolve_source_rope_interleaved(
    hparams: dict, source_rope_layout: str | None = None
) -> bool:
    """Return the RoPE layout the source weights were trained with.

    Precedence: a recorded ``rope_interleaved``; then this fork's native
    checkpoints, whose layout follows the saved backend (SDPA interleaved,
    local flash non-interleaved); otherwise upstream FGA-DIKU BONSAI, which is
    interleaved for both backends since 2026-09-29 (commit a3fa6ac). An older
    upstream flash checkpoint must be declared ``non_interleaved``.
    """
    if source_rope_layout is not None and source_rope_layout not in ROPE_LAYOUTS:
        raise ValueError(f"source_rope_layout must be one of {sorted(ROPE_LAYOUTS)}")
    raw = _raw_source_config(hparams)
    declared = None if source_rope_layout is None else ROPE_LAYOUTS[source_rope_layout]
    if has_explicit_rope_layout(raw):
        inferred, basis = bool(raw["rope_interleaved"]), "recorded"
    elif raw.get("attn_type") == "sdpa":
        inferred, basis = True, "sdpa"
    elif _is_local_native(raw):
        inferred, basis = legacy_rope_interleaved(str(raw.get("attn_type"))), "local"
    else:
        # Upstream flash: current convention unless the caller declares the
        # pre-a3fa6ac layout of an older training commit.
        return True if declared is None else declared
    if declared is not None and declared != inferred:
        raise ValueError(
            f"Declared source RoPE layout {source_rope_layout!r} contradicts the "
            f"layout implied by the checkpoint ({basis}): rope_interleaved={inferred}"
        )
    return inferred


def resolve_source_encoder_config(
    hparams: dict, source_rope_layout: str | None = None
) -> dict:
    """Normalized source encoder config, including upstream conventions.

    Current upstream BONSAI checkpoints store flat hyperparameters without
    ``abspos_encoding`` and with ``value_embedding_mode=None`` for code-only
    models. Upstream feeds absolute time in thousands of hours to Time2Vec,
    which is this fork's ``scaled_time2vec`` applied to hours.
    """
    raw = _raw_source_config(hparams)
    if not _is_local_native(raw):
        raw = dict(raw)
        if raw.get("value_embedding_mode") is None:
            raw["value_embedding_mode"] = "legacy"
        raw.setdefault("abspos_encoding", "scaled_time2vec")
        if "num_attention_heads" not in raw:
            # Original BONSAI MHA used a fixed head_dim=64 and did not save
            # the derived head count. Do not apply this inference to local
            # native checkpoints or overwrite an explicitly recorded count.
            width = int(raw.get("hidden_size", 0))
            if width <= 0 or width % 64:
                raise ValueError("Legacy upstream checkpoint lacks num_attention_heads "
                                 "and hidden_size does not support fixed 64-dimensional heads")
            raw["num_attention_heads"] = width // 64
    config = require_native_checkpoint_config(raw)
    config["rope_interleaved"] = resolve_source_rope_interleaved(
        hparams, source_rope_layout
    )
    return config


def source_pretrain_model_config(
    checkpoint_path: str | Path, source_rope_layout: str | None = None
) -> dict:
    """Read the native encoder specification without constructing a model."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict) or "hyper_parameters" not in checkpoint:
        raise ValueError("Source must be a BONSAI Lightning pretraining checkpoint")
    return resolve_source_encoder_config(
        checkpoint["hyper_parameters"], source_rope_layout
    )


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
    source_rope_layout: str | None = None,
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
    source_config = resolve_source_encoder_config(
        checkpoint["hyper_parameters"], source_rope_layout
    )
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
        "rope_interleaved": source_config["rope_interleaved"],
        "source_rope_layout_recorded": has_explicit_rope_layout(
            _raw_source_config(checkpoint["hyper_parameters"])
        ),
        "source_convention": (
            "local_native"
            if _is_local_native(_raw_source_config(checkpoint["hyper_parameters"]))
            else "upstream_bonsai"
        ),
        "target_abspos_encoding": target_config["abspos_encoding"],
        "operational_model_differences": {
            key: (source_config[key], target_config[key])
            for key in ("attn_type", "dropout", "attention_dropout")
            if source_config[key] != target_config[key]
        },
    }
