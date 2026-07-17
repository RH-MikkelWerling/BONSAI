"""Immutable truncation strategies for longitudinal subject records."""

from __future__ import annotations

from typing import Dict, Literal, Optional, Tuple

import torch

from bonsai.functional.subject_data import clone_subject

TruncationStrategy = Literal["tail", "random_window", "mixed_window"]
SEQUENCE_FIELDS = ("code", "abspos", "segment", "age")


def sequence_tensor_fields(subject: Dict[str, torch.Tensor]) -> tuple[str, ...]:
    """Return per-token tensor fields aligned to ``subject["code"]``."""
    sequence_length = len(subject["code"])
    return tuple(
        key
        for key, value in subject.items()
        if isinstance(value, torch.Tensor)
        and value.ndim > 0
        and len(value) == sequence_length
    )


def _clinical_window_start(
    clinical_length: int,
    kept_clinical_tokens: int,
    strategy: TruncationStrategy,
    tail_window_probability: float,
    generator: Optional[torch.Generator],
) -> int:
    last_start = clinical_length - kept_clinical_tokens
    if last_start <= 0 or strategy == "tail":
        return max(last_start, 0)
    if strategy == "mixed_window":
        if not 0.0 <= tail_window_probability <= 1.0:
            raise ValueError("tail_window_probability must be in [0, 1].")
        use_tail = torch.rand((), generator=generator).item() < tail_window_probability
        if use_tail:
            return last_start
    if strategy not in {"random_window", "mixed_window"}:
        raise ValueError(f"Unknown truncation strategy: {strategy!r}")
    return int(
        torch.randint(
            low=0,
            high=last_start + 1,
            size=(1,),
            generator=generator,
        ).item()
    )


def truncate_subject(
    subject: Dict[str, torch.Tensor],
    max_len: int,
    background_length: int,
    strategy: TruncationStrategy = "tail",
    tail_window_probability: float = 0.5,
    generator: Optional[torch.Generator] = None,
    *,
    return_metadata: bool = False,
) -> dict | Tuple[dict, dict]:
    """Return an immutable background-plus-clinical-window view."""
    result = clone_subject(subject)
    sequence_length = len(result["code"])
    background_length = int(background_length)
    if max_len <= 0:
        raise ValueError("max_len must be positive.")
    if background_length < 0 or background_length > sequence_length:
        raise ValueError("background_length is outside the subject sequence.")
    if max_len < background_length:
        raise ValueError("max_len cannot be smaller than background_length.")

    metadata = {
        "truncated": sequence_length > max_len,
        "clinical_window_started_mid_history": False,
    }
    if sequence_length > max_len:
        kept_clinical_tokens = max_len - background_length
        clinical_length = sequence_length - background_length
        start_offset = _clinical_window_start(
            clinical_length,
            kept_clinical_tokens,
            strategy,
            tail_window_probability,
            generator,
        )
        clinical_start = background_length + start_offset
        clinical_stop = clinical_start + kept_clinical_tokens
        background_indices = torch.arange(background_length, dtype=torch.long)
        clinical_indices = torch.arange(clinical_start, clinical_stop, dtype=torch.long)
        indices = torch.cat((background_indices, clinical_indices))
        for field in sequence_tensor_fields(result):
            result[field] = result[field][indices]
        metadata["clinical_window_started_mid_history"] = start_offset > 0

    if return_metadata:
        return result, metadata
    return result
