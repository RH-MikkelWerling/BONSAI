"""Input-semantics contracts persisted across BONSAI training stages."""

from __future__ import annotations

from collections.abc import Mapping


VALID_NUMERIC_VALUE_CONTROLS = frozenset({"observed", "masked"})


def validate_numeric_value_control(value: str) -> str:
    """Return a normalized numeric-value policy or raise a clear error."""
    normalized = str(value).lower()
    if normalized not in VALID_NUMERIC_VALUE_CONTROLS:
        expected = ", ".join(sorted(VALID_NUMERIC_VALUE_CONTROLS))
        raise ValueError(
            f"numeric_value_control must be one of {{{expected}}}, got {value!r}."
        )
    return normalized


def checkpoint_numeric_value_control(
    hparams: Mapping | None,
    *,
    legacy_default: str = "observed",
) -> str:
    """Read the saved policy, with a guarded fallback for legacy checkpoints.

    Early no-values checkpoints recorded their intent only in
    ``training_stage``.  Supporting that exact marker lets those checkpoints
    remain usable without guessing from model architecture: both observed and
    masked runs intentionally use the same FiLM architecture.
    """
    hparams = hparams if isinstance(hparams, Mapping) else {}
    metadata = hparams.get("checkpoint_metadata", {})
    metadata = metadata if isinstance(metadata, Mapping) else {}
    contract = metadata.get("input_contract", {})
    contract = contract if isinstance(contract, Mapping) else {}

    saved = contract.get("numeric_value_control")
    if saved is None:
        saved = metadata.get("numeric_value_control")
    if saved is not None:
        return validate_numeric_value_control(saved)

    training_stage = str(metadata.get("training_stage", "")).lower()
    if "no_values" in training_stage or "no-values" in training_stage:
        return "masked"
    return validate_numeric_value_control(legacy_default)


def resolve_numeric_value_control(
    requested: str | None,
    source_hparams: Mapping | None = None,
    *,
    default: str = "observed",
) -> str:
    """Resolve an explicit policy or inherit it from the source checkpoint."""
    if requested in (None, "inherit", "checkpoint", "saved"):
        if source_hparams:
            return checkpoint_numeric_value_control(
                source_hparams, legacy_default=default
            )
        return validate_numeric_value_control(default)
    return validate_numeric_value_control(requested)


def input_contract_metadata(numeric_value_control: str) -> dict:
    """Build the canonical checkpoint metadata fragment."""
    return {
        "input_contract": {
            "numeric_value_control": validate_numeric_value_control(
                numeric_value_control
            )
        }
    }
