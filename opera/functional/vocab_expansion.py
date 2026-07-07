"""
Vocabulary expansion utilities for OPERA DAPT.

When domain-specific data sources (e.g. RKKP quality registry) introduce
tokens not present in the base pretrained vocabulary, the embedding and
decoder layers need to be resized.  New token embeddings are randomly
initialised while pretrained embeddings are preserved.

This module provides:

  1. merge_vocabularies()  — Merge base + domain vocab with source prefixes
  2. expand_model_vocab()  — Resize embedding + decoder, preserving weights
  3. get_vocab_aware_param_groups() — Separate param groups for differential LR

Design rationale
================
New embeddings start from random init and need higher LR to catch up with
the pretrained embeddings.  But you don't want to blast the pretrained
embeddings with that higher LR.  The solution is the same differential-LR
pattern used throughout OPERA (encoder vs projection head, etc.), but
applied at the embedding matrix level.

We achieve this WITHOUT modifying BONSAI code by:
  - Creating the model with the expanded vocab_size from the start
  - Loading the pretrained weights into the first V_base rows
  - Letting the remaining rows keep their random init
  - Using param groups in the Lightning module to set different LRs

Source prefixes
===============
Tokens from different registries should use namespace prefixes:
  LPR//D123, RKKP//ann_arbor_stage_III, LAB//hemoglobin, etc.

This is consistent with the BONSAI convention (BACKGROUND//sex_male, etc.)
and lets you analyse attention patterns by source downstream.
"""

from typing import Dict, Tuple, Optional
import logging
import torch
import torch.nn as nn


def merge_vocabularies(
    base_vocab: Dict[str, int],
    domain_vocab: Dict[str, int],
    domain_prefix: Optional[str] = None,
) -> Tuple[Dict[str, int], int, int]:
    """
    Merge a base vocabulary with domain-specific tokens.

    Domain tokens that already exist in the base vocab (after prefixing)
    are skipped — no duplicates.  New tokens get IDs starting after the
    last base token.

    Parameters
    ----------
    base_vocab : dict
        Token → ID mapping from base pretraining.
    domain_vocab : dict
        Token → ID mapping from domain data.  These IDs are ignored;
        new IDs are assigned sequentially after the base vocab.
    domain_prefix : str, optional
        If provided, all domain tokens are prefixed: "RKKP//token".
        Tokens that already start with this prefix are not double-prefixed.

    Returns
    -------
    merged_vocab : dict
        Combined token → ID mapping.
    n_base : int
        Size of original base vocabulary (= first N token IDs).
    n_new : int
        Number of new tokens added.
    """
    merged = dict(base_vocab)  # copy
    next_id = max(base_vocab.values()) + 1
    n_new = 0

    for token in sorted(domain_vocab.keys()):
        # Apply prefix if specified
        if domain_prefix and not token.startswith(domain_prefix):
            prefixed = f"{domain_prefix}//{token}"
        else:
            prefixed = token

        # Skip if already in base
        if prefixed in merged:
            continue

        merged[prefixed] = next_id
        next_id += 1
        n_new += 1

    n_base = len(base_vocab)
    logging.info(f"Vocabulary merge: {n_base} base + {n_new} new = {len(merged)} total")
    return merged, n_base, n_new


def merge_multiple_vocabularies(
    base_vocab: Dict[str, int],
    domain_vocabs: Dict[str, Dict[str, int]],
) -> Tuple[Dict[str, int], int, int]:
    """
    Merge base vocab with multiple domain-specific vocabularies.

    Parameters
    ----------
    base_vocab : dict
    domain_vocabs : dict
        Mapping from source prefix → domain vocabulary.
        Example: {"RKKP": rkkp_vocab, "FLOW": flow_vocab}

    Returns
    -------
    merged_vocab, n_base, n_new
    """
    merged = dict(base_vocab)
    next_id = max(base_vocab.values()) + 1
    n_new = 0

    for prefix, domain_vocab in sorted(domain_vocabs.items()):
        for token in sorted(domain_vocab.keys()):
            if not token.startswith(prefix):
                prefixed = f"{prefix}//{token}"
            else:
                prefixed = token

            if prefixed in merged:
                continue

            merged[prefixed] = next_id
            next_id += 1
            n_new += 1

    n_base = len(base_vocab)
    logging.info(
        f"Multi-source vocabulary merge: {n_base} base + {n_new} new "
        f"({len(domain_vocabs)} sources) = {len(merged)} total"
    )
    return merged, n_base, n_new


def expand_model_vocab(
    model: nn.Module,
    old_vocab_size: int,
    new_vocab_size: int,
    init_std: float = 0.02,
) -> nn.Module:
    """
    Resize the embedding and decoder layers of a BonsaiPretrain model,
    preserving pretrained weights for the first ``old_vocab_size`` tokens.

    New token embeddings are initialised from N(0, init_std).
    New decoder weights are initialised from N(0, init_std) with zero bias.

    Parameters
    ----------
    model : BonsaiPretrain
        Model with the OLD vocab size already loaded.
    old_vocab_size : int
        Number of tokens in the original vocabulary.
    new_vocab_size : int
        Number of tokens in the expanded vocabulary.
    init_std : float
        Std for random init of new embeddings.

    Returns
    -------
    model : BonsaiPretrain
        Same model object, mutated in-place with resized layers.
    """
    if new_vocab_size <= old_vocab_size:
        logging.info("No vocabulary expansion needed.")
        return model

    n_new = new_vocab_size - old_vocab_size
    hidden_size = model.hparams["hidden_size"]
    logging.info(
        f"Expanding vocabulary: {old_vocab_size} → {new_vocab_size} (+{n_new} tokens)"
    )

    # ── Expand code_embedding ────────────────────────────────────────
    old_embed = model.embeddings.code_embedding
    pad_idx = old_embed.padding_idx

    new_embed = nn.Embedding(new_vocab_size, hidden_size, padding_idx=pad_idx)
    # Copy pretrained weights
    with torch.no_grad():
        new_embed.weight[:old_vocab_size] = old_embed.weight
        # Initialise new tokens
        nn.init.normal_(new_embed.weight[old_vocab_size:], mean=0.0, std=init_std)
    model.embeddings.code_embedding = new_embed

    # ── Expand tied pretraining head (Linear: hidden_size → vocab_size) ─
    if hasattr(model, "pretrain_head"):
        old_head = model.pretrain_head
        new_head = nn.Linear(
            hidden_size, new_vocab_size, bias=old_head.bias is not None
        )
        with torch.no_grad():
            if old_head.bias is not None:
                new_head.bias[:old_vocab_size] = old_head.bias
                new_head.bias[old_vocab_size:] = 0.0
        new_head.weight = model.embeddings.code_embedding.weight
        model.pretrain_head = new_head

    # ── Update config ────────────────────────────────────────────────
    model.hparams["vocab_size"] = new_vocab_size

    return model


def get_vocab_aware_param_groups(
    model: nn.Module,
    base_lr: float,
    new_embed_lr_multiplier: float = 5.0,
    old_vocab_size: int = 0,
) -> list:
    """
    Create param groups that give new (expanded) embedding rows a higher LR.

    The key insight: nn.Embedding.weight is a single parameter tensor.
    We can't assign different LRs to different rows of the same tensor.

    Instead, we separate the model into:
      - "new_embed_params": code_embedding and tied pretraining-head parameters
        (these contain BOTH old and new rows, but we set a moderate LR
         that's a compromise — not as low as the pretrained LR, not as
         high as pure random-init LR)
      - "pretrained_params": everything else

    For finer control, see the _freeze_pretrained_embeddings() helper
    which freezes the first V_base rows entirely, letting you set a high
    LR on the embedding layer that only effectively trains the new rows.

    Parameters
    ----------
    model : nn.Module
    base_lr : float
        LR for pretrained parameters.
    new_embed_lr_multiplier : float
        Multiplier for embedding/decoder layers when vocab is expanded.
    old_vocab_size : int
        If > 0, the embedding and decoder layers get the higher LR.
        If 0 (no expansion), all params get base_lr.
    """
    if old_vocab_size == 0:
        # No expansion — single param group
        return [
            {
                "params": [p for p in model.parameters() if p.requires_grad],
                "lr": base_lr,
            }
        ]

    embed_decoder_params = []
    other_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "code_embedding" in name or name.startswith("pretrain_head"):
            embed_decoder_params.append(param)
        else:
            other_params.append(param)

    groups = [
        {"params": other_params, "lr": base_lr},
    ]
    if embed_decoder_params:
        groups.append(
            {
                "params": embed_decoder_params,
                "lr": base_lr * new_embed_lr_multiplier,
            }
        )

    return groups


def freeze_pretrained_embeddings(
    model: nn.Module,
    old_vocab_size: int,
) -> None:
    """
    Register a backward hook that zeros gradients for the first V_base rows
    of the code_embedding and decoder.  This lets you set a high LR for the
    embedding layer that only effectively trains the new rows.

    This is the most precise approach but adds hook complexity.  The simpler
    alternative (separate param group with moderate LR) is usually sufficient.

    Call AFTER model construction and weight loading.
    """

    def _zero_old_grad(grad, n_old):
        grad_clone = grad.clone()
        grad_clone[:n_old] = 0.0
        return grad_clone

    embed = model.embeddings.code_embedding.weight
    embed.register_hook(lambda g: _zero_old_grad(g, old_vocab_size))

    if hasattr(model, "pretrain_head") and model.pretrain_head.bias is not None:
        model.pretrain_head.bias.register_hook(
            lambda g: _zero_old_grad(g, old_vocab_size)
        )

    logging.info(
        f"Registered gradient hooks: freezing first {old_vocab_size} rows "
        f"of code_embedding and the pretraining head"
    )
