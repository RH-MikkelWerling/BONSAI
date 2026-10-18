"""Input restriction at inference: what a trained model gets from each source.

A trained model is evaluated again with some of each patient's events hidden:
whole sources (by code prefix, e.g. ``LABTEST`` or ``MEDICATION``) or history
older than a time window. Hidden events are removed from the sequence before
batching, so positions, pooling and the prediction token stay valid, and the
model is not retrained. The drop in discrimination shows how much the model
relies on what was hidden. It describes the model, not causal effects.
"""

from __future__ import annotations

import torch

from opera.interpretability.code_meaning import DEMOGRAPHIC_PREFIXES

# Special tokens and demographics (sex, age: timed at birth, so a time window
# would otherwise remove them).
ALWAYS_KEPT = ("[", *DEMOGRAPHIC_PREFIXES)
HOURS_PER_DAY = 24.0


def hidden_code_ids(vocabulary, prefixes):
    """Vocabulary ids whose code starts with one of ``prefixes`` (never special tokens)."""
    prefixes = tuple(p.strip() for p in prefixes if p and p.strip())
    return {int(i) for name, i in vocabulary.items()
            if prefixes and str(name).startswith(prefixes)
            and not str(name).startswith(ALWAYS_KEPT)}


def kept_code_ids(vocabulary):
    """Ids that are never removed, whatever the restriction."""
    return {int(i) for name, i in vocabulary.items() if str(name).startswith(ALWAYS_KEPT)}


def restrict_sample(sample, *, hidden_ids=frozenset(), protected_ids=frozenset(),
                    keep_last_days=None, time_key="abspos"):
    """Copy of one patient's sample with the restricted events removed.

    Every per-token tensor (same length as ``code``) is filtered with the same
    mask. Outcome, time and event fields and scalars are left untouched.
    ``keep_last_days`` keeps only events within that many days of the most
    recent event (``abspos`` in hours), plus special and background tokens.
    """
    code = sample["code"]
    protected = torch.isin(code, torch.tensor(sorted(protected_ids), dtype=code.dtype))
    keep = ~torch.isin(code, torch.tensor(sorted(hidden_ids), dtype=code.dtype)) | protected
    if keep_last_days is not None and time_key in sample:
        times = sample[time_key].to(torch.float64)
        recent = times >= times.max() - float(keep_last_days) * HOURS_PER_DAY
        keep &= recent | protected
    return keep_tokens(sample, keep)


def keep_tokens(sample, keep):
    """Copy of one patient's sample with only the tokens where ``keep`` is True."""
    length = len(sample["code"])
    restricted = {}
    for key, value in sample.items():
        per_token = (isinstance(value, torch.Tensor) and value.ndim >= 1
                     and value.shape[0] == length
                     and not key.startswith(("outcome_", "time_", "event_")))
        restricted[key] = value[keep] if per_token else value
    return restricted


def restricting_collate(collate, **restriction):
    """Wrap a collate function so every sample is restricted before padding."""
    def wrapped(batch):
        return collate([restrict_sample(sample, **restriction) for sample in batch])
    return wrapped
