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
    objective = str(competing.get("objective", "piecewise_exponential")).lower()
    if objective not in {"piecewise_exponential", "horizon_bce"}:
        raise ValueError(
            "competing_risk.objective must be piecewise_exponential or horizon_bce."
        )
    if objective == "horizon_bce" and horizons != boundaries:
        raise ValueError(
            "horizon_bce requires training.evaluation_horizons_days to exactly "
            "match competing_risk.interval_boundaries_days."
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
    head_mode = str(competing.get("head_mode", "linear"))
    if head_mode not in {"linear", "outcome_mlp", "family_trunks"}:
        raise ValueError(
            "competing_risk.head_mode must be linear, outcome_mlp, or family_trunks."
        )
    readout_mode = str(competing.get("readout_mode", "shared"))
    attention_readouts = {
        "outcome_attention",
        "shared_attention_prototypes",
        "factorized_cohort_outcome_prototypes",
    }
    if readout_mode not in {"shared", *attention_readouts}:
        raise ValueError(
            "competing_risk.readout_mode must be shared, outcome_attention, "
            "shared_attention_prototypes, or "
            "factorized_cohort_outcome_prototypes."
        )
    if head_mode in {"outcome_mlp", "family_trunks"} and model.get("pooling") != "cls_last":
        raise ValueError("Specialized competing-risk heads require model.pooling=cls_last.")
    if readout_mode in attention_readouts:
        if model.get("pooling") != "cls_last":
            raise ValueError(
                "Outcome-conditioned attention requires model.pooling=cls_last."
            )
        if head_mode != "linear":
            raise ValueError(
                "The isolated attention-readout ablation requires head_mode=linear."
            )
        if float(competing.get("contrastive_loss_weight", 0.0)) != 0.0:
            raise ValueError(
                "Outcome-conditioned attention requires contrastive_loss_weight=0."
            )
        if training.get("enable_validation_probe", False):
            raise ValueError(
                "Outcome-conditioned attention requires enable_validation_probe=false."
            )
    if readout_mode in {
        "shared_attention_prototypes",
        "factorized_cohort_outcome_prototypes",
    }:
        if int(competing.get("prototype_attention_num_prototypes", 10)) < 1:
            raise ValueError("prototype_attention_num_prototypes must be positive.")
        if float(competing.get("prototype_mixture_temperature", 1.0)) <= 0:
            raise ValueError("prototype_mixture_temperature must be positive.")
    if readout_mode == "factorized_cohort_outcome_prototypes":
        if not config.get("cohorts"):
            raise ValueError(
                "Factorized cohort-outcome attention requires configured cohorts."
            )
        if int(competing.get("prototype_interaction_rank", 8)) < 1:
            raise ValueError("prototype_interaction_rank must be positive.")

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
