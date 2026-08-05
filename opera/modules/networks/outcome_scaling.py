"""Configuration helpers for fixed cross-outcome loss scaling."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any


def resolve_outcome_reference_scales(
    config: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Return a copied config with an optional diagnostic JSON artifact loaded."""
    settings = dict(config or {})
    reference_file = settings.pop("outcome_reference_scale_file", None)
    if reference_file is None:
        return settings
    path = Path(str(reference_file))
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    scales = payload.get("outcome_reference_scales", payload)
    if not isinstance(scales, Mapping):
        raise ValueError(
            f"Outcome reference artifact {path} must contain a mapping or an "
            "outcome_reference_scales mapping."
        )
    if settings.get("outcome_reference_scales"):
        raise ValueError(
            "Set either outcome_reference_scale_file or outcome_reference_scales, not both."
        )
    settings["outcome_reference_scales"] = dict(scales)
    return settings
