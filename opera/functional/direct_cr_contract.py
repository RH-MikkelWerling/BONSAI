"""Fail-fast structural checks for direct competing-risk experiments."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def validate_direct_cr_config(config: Mapping[str, Any]) -> None:
    """Validate invariants that otherwise fail late in a GPU training run."""
    outcomes = set(config.get("outcomes", {}))
    competing = dict(config.get("competing_risk", {}))
    cross = dict(config.get("cross_outcome", {}))
    training = dict(config.get("training", {}))
    model = dict(config.get("model", {}))
    if float(competing.get("loss_weight", 0.0)) <= 0:
        return

    boundaries = [float(value) for value in competing.get("interval_boundaries_days", [])]
    if not boundaries or boundaries != sorted(set(boundaries)) or boundaries[0] <= 0:
        raise ValueError(
            "competing_risk.interval_boundaries_days must be unique, increasing, "
            "positive values."
        )
    horizons = [float(value) for value in training.get("evaluation_horizons_days", boundaries)]
    if not horizons or any(value <= 0 or value > boundaries[-1] for value in horizons):
        raise ValueError(
            "training.evaluation_horizons_days must be positive and no later than "
            "the final competing-risk boundary."
        )

    families = dict(cross.get("outcome_families", {}))
    members = [outcome for values in families.values() for outcome in values]
    duplicate = sorted({name for name in members if members.count(name) > 1})
    missing = sorted(outcomes - set(members))
    extra = sorted(set(members) - outcomes)
    needs_families = (
        competing.get("head_mode") == "family_trunks"
        or cross.get("aggregation") == "hierarchical_support"
        or "family_macro" in str(training.get("eval_monitor_metric", ""))
    )
    if needs_families and (missing or extra or duplicate):
        raise ValueError(
            "outcome_families must cover configured outcomes exactly once; "
            f"missing={missing}, extra={extra}, duplicate={duplicate}."
        )
    if competing.get("head_mode") == "family_trunks" and model.get("pooling") != "cls_last":
        raise ValueError("Production family trunks require model.pooling=cls_last.")

    weighter = str(competing.get("weighter", "uniform")).lower()
    if weighter == "kendall_null":
        if cross.get("aggregation", "macro") != "macro":
            raise ValueError("kendall_null requires cross_outcome.aggregation=macro.")
        if not competing.get("initialize_from_null_hazards", False):
            raise ValueError(
                "kendall_null production runs must initialize the head from the "
                "same training-only null hazards."
            )

    curriculum = dict(competing.get("curriculum", {}))
    if curriculum.get("enabled") and len(outcomes) < 2:
        raise ValueError("Curriculum learning requires at least two outcomes.")
