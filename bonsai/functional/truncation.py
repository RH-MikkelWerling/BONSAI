"""Immutable truncation strategies for longitudinal subject records."""

from __future__ import annotations

from typing import Dict, Literal, Optional, Tuple

import torch

from bonsai.functional.subject_data import clone_subject

TruncationStrategy = Literal["tail", "random_window", "mixed_window", "all_windows"]
SEQUENCE_FIELDS = ("code", "abspos", "segment", "age")


def infer_background_length(subject: Dict[str, torch.Tensor]) -> int:
    """Return the length of the leading background event group.

    Processed MEDS segment identifiers are one-based before dataset-time
    normalization.  Inferring the background from ``segment == 0`` therefore
    silently returned zero.  Background is instead the complete leading
    segment, irrespective of the numeric segment identifier.
    """
    segments = subject.get("segment")
    if not isinstance(segments, torch.Tensor) or segments.ndim != 1:
        raise ValueError("subject['segment'] must be a one-dimensional tensor.")
    if len(segments) == 0:
        return 0
    first_segment = segments[0]
    different = torch.nonzero(segments != first_segment, as_tuple=False)
    return int(different[0].item()) if len(different) else len(segments)


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


def _event_group_boundaries(clinical_segments: torch.Tensor) -> torch.Tensor:
    """Offsets delimiting complete contiguous clinical event groups."""
    clinical_length = len(clinical_segments)
    if clinical_length == 0:
        return torch.tensor([0], dtype=torch.long)
    return torch.cat(
        (
            torch.tensor([0], dtype=torch.long),
            torch.nonzero(
                clinical_segments[1:] != clinical_segments[:-1],
                as_tuple=False,
            ).flatten()
            + 1,
            torch.tensor([clinical_length], dtype=torch.long),
        )
    )


def exhaustive_clinical_windows(
    subject: Dict[str, torch.Tensor],
    max_len: int,
    background_length: int,
) -> list[tuple[int, int]]:
    """Partition clinical history into chronological, event-complete windows.

    Returned offsets are relative to the clinical portion of the record. The
    background block is not included because callers prepend it to every view.
    Clinical tokens occur in exactly one returned window.
    """
    sequence_length = len(subject["code"])
    background_length = int(background_length)
    if max_len <= 0 or not 0 <= background_length <= sequence_length:
        raise ValueError("Invalid max_len or background_length.")
    if max_len < background_length:
        raise ValueError("max_len cannot be smaller than background_length.")
    clinical_budget = max_len - background_length
    if clinical_budget <= 0 and sequence_length > background_length:
        raise ValueError("max_len leaves no room for clinical event groups.")
    clinical_segments = subject["segment"][background_length:]
    if len(clinical_segments) == 0:
        return [(0, 0)]
    boundaries = _event_group_boundaries(clinical_segments)
    group_lengths = boundaries[1:] - boundaries[:-1]
    if bool((group_lengths > clinical_budget).any()):
        raise ValueError(
            "A clinical event group cannot fit without being split: "
            f"largest_group={int(group_lengths.max().item())}, "
            f"available_tokens={clinical_budget}."
        )
    windows: list[tuple[int, int]] = []
    start_group = 0
    used = 0
    for group_index, group_length in enumerate(group_lengths.tolist()):
        if used and used + int(group_length) > clinical_budget:
            windows.append(
                (
                    int(boundaries[start_group].item()),
                    int(boundaries[group_index].item()),
                )
            )
            start_group = group_index
            used = 0
        used += int(group_length)
    windows.append(
        (int(boundaries[start_group].item()), int(boundaries[-1].item()))
    )
    return windows


def subject_window_count(
    subject: Dict[str, torch.Tensor], max_len: int, background_length: int
) -> int:
    """Number of exhaustive examples generated for one subject."""
    return len(exhaustive_clinical_windows(subject, max_len, background_length))


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
    window_index: Optional[int] = None,
    *,
    return_metadata: bool = False,
) -> dict | Tuple[dict, dict]:
    """Return an immutable background-plus-clinical-window view.

    Clinical windows start and stop only at segment boundaries.  This keeps
    all tokens belonging to one MEDS event group together.  A single event
    group larger than the available clinical-token budget is rejected rather
    than silently split.
    """
    result = clone_subject(subject)
    sequence_length = len(result["code"])
    background_length = int(background_length)
    # Backward-compatible guard for callers that derived this value with
    # ``(segment == 0).sum()``. Raw ehr2meds sequences are one-based, so zero
    # there means "not inferred", rather than "no background event".
    if (
        background_length == 0
        and sequence_length
        and int(result["segment"][0].item()) != 0
    ):
        background_length = infer_background_length(result)
    if max_len <= 0:
        raise ValueError("max_len must be positive.")
    if background_length < 0 or background_length > sequence_length:
        raise ValueError("background_length is outside the subject sequence.")
    if max_len < background_length:
        raise ValueError("max_len cannot be smaller than background_length.")

    metadata = {
        "truncated": sequence_length > max_len,
        "clinical_window_started_mid_history": False,
        "background_length": background_length,
        "window_index": 0,
        "n_windows": 1,
    }
    if strategy == "all_windows":
        windows = exhaustive_clinical_windows(result, max_len, background_length)
        if window_index is None:
            raise ValueError("all_windows requires an explicit window_index.")
        if not 0 <= int(window_index) < len(windows):
            raise IndexError(
                f"window_index={window_index} is outside [0, {len(windows)})."
            )
        start_offset, stop_offset = windows[int(window_index)]
        metadata["window_index"] = int(window_index)
        metadata["n_windows"] = len(windows)
        if sequence_length > max_len:
            clinical_start = background_length + start_offset
            clinical_stop = background_length + stop_offset
            indices = torch.cat(
                (
                    torch.arange(background_length, dtype=torch.long),
                    torch.arange(clinical_start, clinical_stop, dtype=torch.long),
                )
            )
            for field in sequence_tensor_fields(result):
                result[field] = result[field][indices]
            metadata["clinical_window_started_mid_history"] = start_offset > 0
    elif sequence_length > max_len:
        kept_clinical_tokens = max_len - background_length
        clinical_length = sequence_length - background_length
        clinical_segments = result["segment"][background_length:]
        if kept_clinical_tokens <= 0:
            raise ValueError("max_len leaves no room for clinical event groups.")

        # Contiguous runs are event groups.  Build every maximal window that
        # fits so tail/random/mixed sampling cannot bisect an event.
        boundaries = _event_group_boundaries(clinical_segments)
        group_lengths = boundaries[1:] - boundaries[:-1]
        oversized = group_lengths > kept_clinical_tokens
        if oversized.any():
            largest = int(group_lengths.max().item())
            raise ValueError(
                "A clinical event group cannot fit without being split: "
                f"largest_group={largest}, available_tokens={kept_clinical_tokens}."
            )

        windows = []
        stop_group = 0
        used = 0
        for start_group in range(len(group_lengths)):
            while stop_group < len(group_lengths):
                candidate = int(group_lengths[stop_group].item())
                if used + candidate > kept_clinical_tokens:
                    break
                used += candidate
                stop_group += 1
            windows.append(
                (int(boundaries[start_group].item()), int(boundaries[stop_group].item()))
            )
            used -= int(group_lengths[start_group].item())

        tail_start = min(i for i, (_, stop) in enumerate(windows) if stop == clinical_length)
        if strategy == "tail":
            window_index = tail_start
        elif strategy == "mixed_window":
            if not 0.0 <= tail_window_probability <= 1.0:
                raise ValueError("tail_window_probability must be in [0, 1].")
            use_tail = torch.rand((), generator=generator).item() < tail_window_probability
            window_index = (
                tail_start
                if use_tail
                else int(torch.randint(len(windows), (1,), generator=generator).item())
            )
        elif strategy == "random_window":
            window_index = int(
                torch.randint(len(windows), (1,), generator=generator).item()
            )
        else:
            raise ValueError(f"Unknown truncation strategy: {strategy!r}")

        start_offset, stop_offset = windows[window_index]
        clinical_start = background_length + start_offset
        clinical_stop = background_length + stop_offset
        background_indices = torch.arange(background_length, dtype=torch.long)
        clinical_indices = torch.arange(clinical_start, clinical_stop, dtype=torch.long)
        indices = torch.cat((background_indices, clinical_indices))
        for field in sequence_tensor_fields(result):
            result[field] = result[field][indices]
        metadata["clinical_window_started_mid_history"] = start_offset > 0

    if return_metadata:
        return result, metadata
    return result
