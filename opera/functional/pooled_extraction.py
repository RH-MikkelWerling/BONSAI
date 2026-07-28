"""Mask-aware pooling variants for the CLS-vs-pooling ablation (AUDIT_FINDINGS.md Part B).

Given hidden states captured from a *single* forward pass (one tensor per
target transformer layer, all aligned to the same padded batch), this module
computes several patient-level pooling variants. Every variant except ``cls``
deliberately excludes the appended prediction/CLS token from its pooling
window: that token is appended only by ``FinetuneDataset``/extraction code
and never occurs in any pretraining sequence (see AUDIT_FINDINGS.md Part A1),
so folding it into a mean/last/max over "real" tokens would mix one
out-of-distribution position into a summary that is supposed to describe
actual clinical history.

``attn_cls`` (attention-weighted pooling using the final layer's attention
into the CLS position) is intentionally not implemented here. This
architecture's attention (``bonsai/modules/networks/components/mha.py``)
calls ``torch.nn.functional.scaled_dot_product_attention``, a fused kernel
that does not return attention probabilities; retrieving them would require
reimplementing the attention math by hand rather than reading an existing
value, so per the task instructions this variant is skipped rather than
approximated with a new learned pooling head.
"""

from __future__ import annotations

from typing import Mapping

import torch

POOLING_NAMES = ("cls", "mean", "last", "mean_last_128", "max")
DEPTH_LABELS = ("final", "d75", "d50", "d25")


def resolve_target_layers(num_layers: int) -> dict[str, int]:
    """Map depth labels to concrete 1-indexed transformer-layer numbers.

    ``final`` is always ``num_layers``. The other three resolve to the
    nearest layer to 3/4, 1/2, and 1/4 depth. For shallow stacks
    (``num_layers <= 3``) two labels may resolve to the same layer index;
    callers should dedupe by index before running redundant work.
    """
    if num_layers < 1:
        raise ValueError("num_layers must be positive.")
    fractions = {"final": 1.0, "d75": 0.75, "d50": 0.50, "d25": 0.25}
    layers: dict[str, int] = {}
    for label, fraction in fractions.items():
        layer = num_layers if label == "final" else max(1, round(fraction * num_layers))
        layers[label] = min(layer, num_layers)
    return layers


def predict_token_mask(code: torch.Tensor, predict_token_id: int) -> torch.Tensor:
    """Boolean mask marking the single appended prediction/CLS position per row."""
    mask = code == predict_token_id
    counts = mask.sum(dim=1)
    if not torch.all(counts == 1):
        raise ValueError(
            "Every sequence must contain exactly one predict/CLS token to pool."
        )
    return mask


def pool_variants(
    hidden_by_layer_label: Mapping[str, torch.Tensor],
    *,
    attention_mask: torch.Tensor,
    predict_mask: torch.Tensor,
    last_k: int = 128,
) -> dict[tuple[str, str], torch.Tensor]:
    """Compute every (layer_label, pooling_name) variant in one pass.

    Parameters
    ----------
    hidden_by_layer_label:
        Mapping from layer label (e.g. ``"final"``, ``"d75"``) to that
        layer's ``(batch, seq, hidden)`` states. All tensors must come from
        the same forward pass over the same padded batch.
    attention_mask, predict_mask:
        ``(batch, seq)`` boolean tensors aligned to that same batch.
    last_k:
        Window size for ``mean_last_128`` (default matches the pooling name).

    Returns
    -------
    dict keyed by ``(layer_label, pooling_name)`` -> ``(batch, hidden)`` tensor.
    """
    attention_mask = attention_mask.bool()
    predict_mask = predict_mask.bool()
    content_mask = attention_mask & ~predict_mask
    lengths_content = content_mask.sum(dim=1)
    if torch.any(lengths_content == 0):
        raise ValueError(
            "Found a sequence with no content tokens outside the predict "
            "token; mean/last/max/mean_last_128 pooling is undefined for it."
        )
    batch_size = attention_mask.shape[0]
    device = attention_mask.device
    cls_lengths = attention_mask.sum(dim=1) - 1
    batch_arange = torch.arange(batch_size, device=device)

    out: dict[tuple[str, str], torch.Tensor] = {}
    for layer_label, hidden in hidden_by_layer_label.items():
        if hidden.shape[:2] != attention_mask.shape:
            raise ValueError(
                f"Hidden states for layer {layer_label!r} do not match the "
                "batch/sequence shape of attention_mask."
            )
        hidden_size = hidden.shape[-1]

        out[(layer_label, "cls")] = hidden[batch_arange, cls_lengths]

        content_f = content_mask.unsqueeze(-1).to(hidden.dtype)
        out[(layer_label, "mean")] = (hidden * content_f).sum(dim=1) / (
            lengths_content.clamp(min=1).to(hidden.dtype).unsqueeze(-1)
        )

        neg_inf = torch.finfo(hidden.dtype).min
        masked_for_max = hidden.masked_fill(~content_mask.unsqueeze(-1), neg_inf)
        out[(layer_label, "max")], _ = masked_for_max.max(dim=1)

        last_vec = torch.empty(
            batch_size, hidden_size, dtype=hidden.dtype, device=device
        )
        mean_last_k_vec = torch.empty_like(last_vec)
        for row in range(batch_size):
            idx = content_mask[row].nonzero(as_tuple=True)[0]
            last_vec[row] = hidden[row, idx[-1]]
            tail_idx = idx[-last_k:]
            mean_last_k_vec[row] = hidden[row, tail_idx].mean(dim=0)
        out[(layer_label, "last")] = last_vec
        out[(layer_label, "mean_last_128")] = mean_last_k_vec

    return out
