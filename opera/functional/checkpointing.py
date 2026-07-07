"""OPERA-side checkpoint loading helpers."""

from __future__ import annotations

import torch

from bonsai.functional.checkpointing import (
    MODEL_CONFIG_KEY,
    clean_lightning_state_dict,
    load_finetune_model_from_checkpoint,
    load_state_dict_checked,
)
from bonsai.functional.model_config import require_native_checkpoint_config
from opera.modules.networks.linear_probe_net import BonsaiLinearProbe


def load_opera_finetune_model_from_checkpoint(
    ckpt_path: str,
    strict: bool = True,
    map_location: str = "cpu",
):
    """Load an OPERA/BONSAI finetune-style checkpoint for evaluation.

    Inputs are a Lightning checkpoint path and strictness flags. The returned
    model is either BONSAI's ordinary finetune model or OPERA's strict linear
    probe model, depending on the saved `model_class`. The scientific purpose
    is to keep frozen linear-probe readouts evaluable through the same pipeline
    as full outcome finetuning.
    """
    ckpt = torch.load(ckpt_path, map_location=map_location, weights_only=False)
    hparams = ckpt["hyper_parameters"]
    model_class = hparams.get("model_class")
    if model_class != "BonsaiLinearProbe":
        return load_finetune_model_from_checkpoint(
            ckpt_path,
            strict=strict,
            map_location=map_location,
        )
    model_config = hparams.get(MODEL_CONFIG_KEY)
    if model_config is None:
        raise ValueError(
            f"Checkpoint {ckpt_path!r} is missing '{MODEL_CONFIG_KEY}'. "
            "Linear-probe checkpoints require exact model config metadata."
        )
    model = BonsaiLinearProbe(require_native_checkpoint_config(model_config))
    clean_state = clean_lightning_state_dict(ckpt["state_dict"])
    load_state_dict_checked(model, clean_state, strict=strict)
    return model
