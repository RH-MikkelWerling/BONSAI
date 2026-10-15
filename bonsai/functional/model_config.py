"""Configuration helpers for the native BONSAI transformer architecture."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


NATIVE_ARCHITECTURE_VERSION = "bonsai-native-rope-v1"

MODEL_CONSTRUCTOR_KEYS = {
    "vocab_size",
    "max_seqlen",
    "hidden_size",
    "num_layers",
    "num_attention_heads",
    "bias",
    "dropout",
    "attention_dropout",
    "causal",
    "attn_type",
    "value_bin_vocab_size",
    "value_embedding_mode",
    "abspos_encoding",
    "rope_interleaved",
}
REQUIRED_MODEL_CONSTRUCTOR_KEYS = MODEL_CONSTRUCTOR_KEYS - {
    "value_bin_vocab_size",
    "value_embedding_mode",
    "abspos_encoding",
    "rope_interleaved",
}

_ALIASES = {
    "max_position_embeddings": "max_seqlen",
    "num_hidden_layers": "num_layers",
    "is_causal": "causal",
}


def legacy_rope_interleaved(attn_type: str) -> bool:
    """RoPE layout used by this fork before the layout was recorded.

    The SDPA path (torchtune RoPE) always rotated adjacent pairs. The local
    FlashAttention path used flash_attn's default ``interleaved=False``. This
    describes LOCAL checkpoints only: upstream FGA-DIKU switched its flash
    path to ``interleaved=True`` on 2026-09-29 (commit a3fa6ac), so a foreign
    flash checkpoint without an explicit layout is ambiguous.
    """
    if attn_type not in {"flash", "sdpa"}:
        raise ValueError("attn_type must be either 'flash' or 'sdpa'.")
    return attn_type == "sdpa"


def has_explicit_rope_layout(config: Any) -> bool:
    """Whether saved model metadata records its RoPE layout."""
    value = config_to_dict(config).get("rope_interleaved")
    return value is not None


class LegacyCheckpointError(ValueError):
    """Raised when ModernBERT weights are offered to the native architecture."""


def config_to_dict(config: Any) -> dict:
    """Convert OmegaConf, mappings, and legacy config objects to a plain dict."""
    if config is None:
        return {}
    if isinstance(config, Mapping):
        return dict(config)
    if hasattr(config, "to_dict"):
        return dict(config.to_dict())
    try:
        return dict(config)
    except (TypeError, ValueError) as exc:
        raise TypeError(
            f"Unsupported model config type: {type(config).__name__}"
        ) from exc


def normalize_bonsai_model_config(
    config: Any,
    *,
    vocab_size: int | None = None,
    default_attn_type: str = "flash",
    **overrides: Any,
) -> dict:
    """Return native constructor kwargs from current or legacy config names.

    This translates old experiment YAML keys, not old checkpoint weights. The
    native RoPE transformer has a different parameterization and state dict.
    """
    source = config_to_dict(config)
    for old_name, new_name in _ALIASES.items():
        if (
            old_name in source
            and new_name in source
            and source[old_name] != source[new_name]
        ):
            raise ValueError(
                f"Conflicting model config values for {old_name!r} and {new_name!r}."
            )
        if new_name not in source and old_name in source:
            source[new_name] = source[old_name]

    # The RoPE layout is a property of the trained weights, not of the runtime
    # backend: resolve it from the saved backend before any attn_type override.
    if source.get("rope_interleaved") is None and source.get("attn_type") is not None:
        source["rope_interleaved"] = legacy_rope_interleaved(source["attn_type"])

    if vocab_size is not None:
        source["vocab_size"] = int(vocab_size)
    source.update({key: value for key, value in overrides.items() if value is not None})

    source.setdefault("bias", False)
    source.setdefault("dropout", source.get("embedding_dropout", 0.1))
    source.setdefault("attention_dropout", 0.0)
    source.setdefault("causal", False)
    source.setdefault("attn_type", default_attn_type)

    source.setdefault("value_bin_vocab_size", 0)
    source.setdefault("value_embedding_mode", "legacy")
    source.setdefault("abspos_encoding", "legacy")
    if source.get("rope_interleaved") is None:
        source["rope_interleaved"] = legacy_rope_interleaved(source["attn_type"])
    source["rope_interleaved"] = bool(source["rope_interleaved"])

    result = {key: source[key] for key in MODEL_CONSTRUCTOR_KEYS if key in source}
    missing = REQUIRED_MODEL_CONSTRUCTOR_KEYS - set(result)
    if missing:
        raise ValueError(f"Model config is missing required keys: {sorted(missing)}")
    if result["hidden_size"] % result["num_attention_heads"] != 0:
        raise ValueError("hidden_size must be divisible by num_attention_heads.")
    if result["attn_type"] not in {"flash", "sdpa"}:
        raise ValueError("attn_type must be either 'flash' or 'sdpa'.")
    if result["abspos_encoding"] not in {
        "legacy",
        "scaled_time2vec",
        "fourier",
        "sequence_relative_fourier",
        "sequence_relative_fourier_with_gaps",
        "none",
    }:
        raise ValueError(
            "abspos_encoding must be one of 'legacy', 'scaled_time2vec', "
            "'fourier', 'sequence_relative_fourier', "
            "'sequence_relative_fourier_with_gaps', or 'none'."
        )
    if result["value_embedding_mode"] not in {"legacy", "combined_binning", "film"}:
        raise ValueError(
            "value_embedding_mode must be 'legacy', 'combined_binning', or 'film'."
        )
    if result["value_embedding_mode"] == "film" and result["value_bin_vocab_size"] != 0:
        raise ValueError("film value embedding requires value_bin_vocab_size=0.")
    return result


def is_native_model_config(config: Any) -> bool:
    """Whether checkpoint metadata describes the native BONSAI architecture."""
    values = config_to_dict(config)
    return values.get("architecture_version") == NATIVE_ARCHITECTURE_VERSION or {
        "max_seqlen",
        "num_layers",
        "attn_type",
    }.issubset(values)


def require_native_checkpoint_config(config: Any) -> dict:
    """Validate and normalize architecture metadata loaded from a checkpoint."""
    values = config_to_dict(config)
    if not is_native_model_config(values):
        raise LegacyCheckpointError(
            "This checkpoint uses the legacy ModernBERT BONSAI architecture. "
            "Its weights are not compatible with the native RoPE/FlashAttention "
            "architecture; use a legacy environment or retrain the checkpoint."
        )
    return normalize_bonsai_model_config(values)


def validate_pretraining_attention(dataset_class: type, *, causal: bool) -> None:
    """Reject autoregressive targets wired to bidirectional attention."""
    from bonsai.modules.datasets.PretrainDataset import ARPretrainDataset

    if issubclass(dataset_class, ARPretrainDataset) and not causal:
        raise ValueError(
            "ARPretrainDataset requires causal=True; bidirectional attention "
            "would expose future target tokens and leak the pretraining label."
        )
