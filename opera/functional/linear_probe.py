"""Utilities for frozen-encoder linear probing."""

from __future__ import annotations

from typing import Sequence


def freeze_encoder_for_linear_probe(
    model,
    trainable_prefixes: Sequence[str] = ("finetune_head.",),
) -> dict:
    """Freeze an encoder model while leaving readout-head parameters trainable.

    Parameters
    ----------
    model
        PyTorch module with named parameters. Native BONSAI finetune models use
        the `finetune_head.` prefix for the outcome head. Strict linear-probe
        wrappers can pass `classifier.` explicitly.
    trainable_prefixes
        Parameter-name prefixes that remain trainable.

    Returns
    -------
    dict
        Counts and trainable parameter names for run metadata.

    Scientific purpose
    ------------------
    Linear probing tests whether a pretrained, domain-adapted, or OPERA-adapted
    representation already makes the outcome linearly accessible without
    changing the encoder.
    """
    trainable_names: list[str] = []
    frozen_names: list[str] = []
    trainable_prefixes = tuple(trainable_prefixes)
    for name, parameter in model.named_parameters():
        keep_trainable = name.startswith(trainable_prefixes)
        parameter.requires_grad = keep_trainable
        if keep_trainable:
            trainable_names.append(name)
        else:
            frozen_names.append(name)
    if not trainable_names:
        raise ValueError(
            "Linear probe freeze found no trainable readout parameters. "
            f"Checked prefixes={trainable_prefixes}."
        )
    return {
        "encoder_frozen": True,
        "trainable_prefixes": list(trainable_prefixes),
        "n_trainable_parameters": int(len(trainable_names)),
        "n_frozen_parameters": int(len(frozen_names)),
        "trainable_parameter_names": trainable_names,
    }
