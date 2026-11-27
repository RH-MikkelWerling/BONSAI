"""Within-patient, corresponding-head comparisons of outcome attention weights."""
import torch


def reading_mask(attention_mask):
    """Match OutcomeAttentionReadout: exclude padding and the final CLS token."""
    valid = attention_mask.bool().clone()
    lengths = valid.sum(1).long() - 1
    if (lengths < 0).any():
        raise ValueError('Empty input sequence')
    # The production readout uses right-padded sequences.
    expected = torch.arange(valid.shape[1], device=valid.device)[None] <= lengths[:, None]
    if not torch.equal(valid, expected):
        raise ValueError('Attention export requires the production right-padding convention')
    rows = torch.arange(len(valid), device=valid.device)
    valid[rows, lengths] = False
    empty = ~valid.any(1)
    valid[rows[empty], lengths[empty]] = True
    return valid


def pattern_statistics(weights, valid):
    """Weights are B,H,O,S. Average head-wise cosine matrices within each patient.

    Also compare deviations from uniform attention on valid tokens. Uniform
    distributions have undefined centred cosine and are omitted, never set to zero.
    No comparison is made between token positions belonging to different patients.
    """
    if weights.ndim != 4 or valid.shape != (weights.shape[0], weights.shape[-1]):
        raise ValueError('Expected B,H,O,S weights and B,S mask')
    x = weights.double()
    mask = valid[:, None, None, :]
    if not torch.isfinite(x).all() or (x < 0).any():
        raise ValueError('Invalid attention probabilities')
    if (x.masked_select(~mask.expand_as(x)).abs() > 1e-7).any():
        raise ValueError('Masked tokens received attention')
    if not torch.allclose(x.sum(-1), torch.ones_like(x.sum(-1)), atol=1e-5, rtol=1e-5):
        raise ValueError('Attention must sum to one per outcome and head')
    def cosine(z, centred=False):
        norm = z.norm(dim=-1)
        unit = z / norm.clamp_min(1e-12)[..., None]
        similarity = unit @ unit.transpose(-1, -2)
        if centred:
            good = norm > 1e-7
            similarity = similarity.masked_fill(~(good[..., :, None] & good[..., None, :]), float('nan'))
        return torch.nanmean(similarity, dim=1).clamp(-1, 1)
    uniform = mask.to(x.dtype) / valid.sum(1)[:, None, None, None]
    centred = cosine(x - uniform, True)
    entropy = -(x * x.clamp_min(1e-300).log()).sum(-1).mean(1)
    denominator = valid.sum(1).double().log()
    entropy = entropy / denominator[:, None].clamp_min(1e-12)
    entropy[denominator == 0] = float('nan')
    return {'attention_cosine':cosine(x), 'centred_attention_cosine':centred,
            'normalised_entropy':entropy}


def extract_statistics(model, batch):
    """Read weights directly from the trained MHA; do not patch its forward path."""
    readout = model.outcome_attention_readout
    if model.training or readout.training:
        raise ValueError('Attention export requires eval mode')
    valid = reading_mask(batch['attention_mask'])
    hidden = model._encode_hidden(batch)
    query = readout.outcome_queries()[None].expand(len(hidden), -1, -1)
    _, weights = readout.attention(query, hidden, hidden, key_padding_mask=~valid,
                                   need_weights=True, average_attn_weights=False)
    return pattern_statistics(weights, valid)
