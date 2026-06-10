"""Checkpoint helpers for BONSAI and OPERA training runs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

import torch

MODEL_CONFIG_KEY = "model_config"
ENCODER_CONFIG_KEY = "encoder_config"


def attach_model_config(module: Any, model: Any) -> None:
    """Store model architecture config and class name in module.hparams.

    Supports models that expose .config directly (BonsaiFinetune, BonsaiPretrain)
    or models whose encoder exposes .config (JointFinetuneModel, OperaContrastiveModel).
    """
    if hasattr(model, "config"):
        config_dict = model.config.to_dict()
    elif hasattr(model, "encoder") and hasattr(model.encoder, "config"):
        config_dict = model.encoder.config.to_dict()
    else:
        module.hparams["model_class"] = model.__class__.__name__
        return
    module.hparams[MODEL_CONFIG_KEY] = config_dict
    module.hparams["model_class"] = model.__class__.__name__


def attach_checkpoint_metadata(
    module: Any,
    checkpoint_metadata: Optional[dict],
) -> None:
    """Store training-stage metadata in module.hparams."""
    if checkpoint_metadata is not None:
        module.hparams["checkpoint_metadata"] = checkpoint_metadata


def save_checkpoint_metadata_sidecar(
    output_dir: str,
    module: Any,
    extra: Optional[dict] = None,
) -> Path:
    """Save a JSON sidecar alongside a Lightning checkpoint.

    The sidecar carries model class, architecture config, training-stage
    metadata, and any caller-supplied extras, making checkpoints
    self-describing without loading the full .ckpt file.
    """
    hparams = module.hparams
    payload: dict = {}
    for key in (
        "model_class",
        MODEL_CONFIG_KEY,
        ENCODER_CONFIG_KEY,
        "checkpoint_metadata",
    ):
        if key in hparams:
            payload[key] = hparams[key]
    if extra:
        payload["extra"] = extra

    path = Path(output_dir) / "checkpoint_metadata.json"
    path.write_text(json.dumps(payload, indent=2, default=str))
    return path


def get_saved_encoder_config(hparams: dict) -> dict:
    """Extract model/encoder architecture config from a checkpoint's hparams.

    Handles both structured checkpoints (which embed config under
    MODEL_CONFIG_KEY) and flat config dicts (random_init case where
    the caller passes model config directly as hparams).
    """
    if MODEL_CONFIG_KEY in hparams:
        return dict(hparams[MODEL_CONFIG_KEY])
    return dict(hparams)


def clean_lightning_state_dict(state_dict: dict) -> dict:
    """Strip the Lightning 'model.' prefix from all state-dict keys."""
    result = {}
    for key, value in state_dict.items():
        if key.startswith("model."):
            result[key[len("model.") :]] = value
        else:
            result[key] = value
    return result


def load_state_dict_checked(
    model: Any,
    state_dict: dict,
    strict: bool = True,
) -> None:
    """Load a state dict, raising a clear RuntimeError on mismatch."""
    try:
        missing, unexpected = model.load_state_dict(state_dict, strict=strict)
    except RuntimeError as exc:
        raise RuntimeError(
            f"State dict mismatch loading {type(model).__name__}: {exc}"
        ) from exc
    if strict and (missing or unexpected):
        raise RuntimeError(
            f"State dict mismatch loading {type(model).__name__}. "
            f"Missing keys ({len(missing)}): {missing[:5]}... "
            f"Unexpected keys ({len(unexpected)}): {unexpected[:5]}..."
        )


def load_pretrained_encoder_checked(model: Any, state_dict: dict) -> None:
    """Load BONSAI encoder weights while allowing a newly initialized task head."""
    encoder_state = {}
    for key, value in state_dict.items():
        if not key.startswith("model."):
            continue
        clean_key = key[len("model.") :]
        if clean_key.startswith(("head.", "decoder.", "cls.")):
            continue
        encoder_state[clean_key] = value

    if not encoder_state:
        raise RuntimeError("Checkpoint contains no BONSAI encoder weights.")

    try:
        missing, unexpected = model.load_state_dict(encoder_state, strict=False)
    except RuntimeError as exc:
        raise RuntimeError(
            f"Encoder state mismatch loading {type(model).__name__}: {exc}"
        ) from exc

    meaningful_missing = [key for key in missing if not key.startswith("cls.")]
    if meaningful_missing or unexpected:
        raise RuntimeError(
            f"Encoder state mismatch loading {type(model).__name__}. "
            f"Missing encoder keys: {meaningful_missing[:10]}; "
            f"unexpected keys: {unexpected[:10]}."
        )


def load_finetune_model_from_checkpoint(
    ckpt_path: str,
    strict: bool = True,
    map_location: str = "cpu",
):
    """Reconstruct a BonsaiFinetune model from a Lightning checkpoint."""
    from transformers import ModernBertConfig
    from bonsai.modules.networks.bonsai_nets import BonsaiFinetune

    ckpt = torch.load(ckpt_path, map_location=map_location, weights_only=False)
    hparams = ckpt["hyper_parameters"]
    if MODEL_CONFIG_KEY not in hparams:
        raise ValueError(
            f"Checkpoint {ckpt_path!r} is missing '{MODEL_CONFIG_KEY}'. "
            "Ensure the checkpoint was saved with attach_model_config()."
        )
    model_config = hparams[MODEL_CONFIG_KEY]
    model = BonsaiFinetune(ModernBertConfig(**model_config))
    clean_state = clean_lightning_state_dict(ckpt["state_dict"])
    load_state_dict_checked(model, clean_state, strict=strict)
    return model


def load_joint_model_from_checkpoint(
    ckpt_path: str,
    strict: bool = True,
    map_location: str = "cpu",
):
    """Reconstruct a JointFinetuneModel from a Lightning checkpoint."""
    from transformers import ModernBertConfig
    from opera.modules.networks.joint_finetune_net import JointFinetuneModel
    from opera.compat.bonsai import BonsaiEncoder

    ckpt = torch.load(ckpt_path, map_location=map_location, weights_only=False)
    hparams = ckpt["hyper_parameters"]
    if MODEL_CONFIG_KEY not in hparams:
        raise ValueError(
            f"Checkpoint {ckpt_path!r} is missing '{MODEL_CONFIG_KEY}'. "
            "Ensure the checkpoint was saved with attach_model_config()."
        )
    model_config = hparams[MODEL_CONFIG_KEY]
    outcome_names = list(hparams.get("outcome_names", []))
    encoder = BonsaiEncoder(ModernBertConfig(**model_config))
    model = JointFinetuneModel(
        encoder=encoder,
        outcome_names=outcome_names,
        hidden_size=model_config.get("hidden_size", 768),
    )
    clean_state = clean_lightning_state_dict(ckpt["state_dict"])
    load_state_dict_checked(model, clean_state, strict=strict)
    return model
