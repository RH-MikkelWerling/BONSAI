"""Stable OPERA-facing imports for BONSAI internals.

OPERA intentionally builds on BONSAI, but collaborators may update BONSAI
module paths or constructor details independently.  Import shared BONSAI
objects through this module when touching OPERA code; if BONSAI moves a symbol,
the repair should usually happen here rather than across every experiment
script.
"""

from __future__ import annotations

from importlib import import_module

import torch

from bonsai.functional.model_config import normalize_bonsai_model_config


_SYMBOLS = {
    "BiGRU": ("opera.modules.networks.pooling", "BiGRU"),
    "BonsaiEncoder": ("bonsai.modules.networks.bonsai_nets", "BonsaiBase"),
    "BonsaiFinetune": ("bonsai.modules.networks.bonsai_nets", "BonsaiFinetune"),
    "BonsaiPretrain": ("bonsai.modules.networks.bonsai_nets", "BonsaiPretrain"),
    "FinetuneDataset": ("bonsai.modules.datasets.FinetuneDataset", "FinetuneDataset"),
    "binarize_outcomes": ("bonsai.functional.outcomes", "binarize_outcomes"),
    "compute_abspos": ("bonsai.functional.features", "compute_abspos"),
    "dynamic_padding": ("bonsai.functional.collate", "dynamic_padding"),
    "filter_subject_data": ("bonsai.functional.subject_data", "filter_subject_data"),
    "split_and_binarize_outcomes": (
        "bonsai.functional.outcomes",
        "split_and_binarize_outcomes",
    ),
}


def build_bonsai_encoder(config=None, **overrides):
    """Construct the native BONSAI encoder from current or legacy YAML keys."""
    from bonsai.modules.networks.bonsai_nets import BonsaiBase

    return BonsaiBase(**normalize_bonsai_model_config(config, **overrides))


def build_bonsai_pretrain(config=None, **overrides):
    """Construct a native BONSAI pretraining model."""
    from bonsai.modules.networks.bonsai_nets import BonsaiPretrain

    return BonsaiPretrain(**normalize_bonsai_model_config(config, **overrides))


def build_bonsai_finetune(config=None, *, predict_token_id: int, **overrides):
    """Construct a native BONSAI binary finetuning model."""
    from bonsai.modules.networks.bonsai_nets import BonsaiFinetune

    kwargs = normalize_bonsai_model_config(config, **overrides)
    return BonsaiFinetune(**kwargs, predict_token_id=predict_token_id)


def encoder_hidden_state(output) -> torch.Tensor:
    """Return sequence states across native and legacy encoder output formats."""
    if torch.is_tensor(output):
        return output
    if hasattr(output, "last_hidden_state"):
        return output.last_hidden_state
    if isinstance(output, (tuple, list)) and output and torch.is_tensor(output[0]):
        return output[0]
    raise TypeError(f"Unsupported encoder output type: {type(output).__name__}")


def encoder_hparams(encoder) -> dict:
    """Return native architecture metadata from an encoder."""
    if hasattr(encoder, "hparams"):
        return dict(encoder.hparams)
    if hasattr(encoder, "config"):
        return normalize_bonsai_model_config(encoder.config)
    raise TypeError(f"Encoder {type(encoder).__name__} exposes no model config.")


def __getattr__(name: str):
    try:
        module_name, symbol_name = _SYMBOLS[name]
    except KeyError as exc:
        raise AttributeError(name) from exc

    module = import_module(module_name)
    symbol = getattr(module, symbol_name)
    globals()[name] = symbol
    return symbol


__all__ = list(_SYMBOLS) + [
    "build_bonsai_encoder",
    "build_bonsai_finetune",
    "build_bonsai_pretrain",
    "encoder_hidden_state",
    "encoder_hparams",
]
