"""Fail-fast structural checks for direct competing-risk experiments."""

from __future__ import annotations

from collections.abc import Mapping
import math
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
    if training.get("eval_monitor_metric") == "val/cr_horizon_bce_endpoint_group_macro":
        if (
            objective != "horizon_bce"
            or not cross.get("endpoint_groups")
            or training.get("limit_val_batches", 1.0) != 1.0
        ):
            raise ValueError(
                "Full-tuning BCE selection requires horizon_bce, endpoint groups "
                "and complete validation"
            )
    if objective not in {"piecewise_exponential", "horizon_bce"}:
        raise ValueError(
            "competing_risk.objective must be piecewise_exponential or horizon_bce."
        )
    if objective == "horizon_bce" and horizons != boundaries:
        raise ValueError(
            "horizon_bce requires training.evaluation_horizons_days to exactly "
            "match competing_risk.interval_boundaries_days."
        )
    sampling = dict(training.get("batch_sampling", {}))
    if sampling.get("type") == "ipw_event_horizon_bce":
        if objective != "horizon_bce":
            raise ValueError("IPW event sampling requires horizon_bce")
        sampling_horizons = [float(value) for value in sampling.get("horizons_days", [])]
        if sampling_horizons != horizons:
            raise ValueError("IPW sampling must cover the complete BCE horizon grid")
        if not 0 < float(sampling.get("event_fraction", 0.5)) < 1:
            raise ValueError("IPW event_fraction must be between zero and one")
        if float(config.get("model", {}).get("dapt_anchor_weight", 0.0)) != 0:
            raise ValueError("IPW sampling requires zero DAPT anchor weight")
        if int(sampling.get("min_unique_events", 10)) < 1 or int(
            sampling.get("min_unique_controls", 20)
        ) < 1:
            raise ValueError("IPW sampling support thresholds must be positive")
    if sampling.get("type") == "minority_event_horizon_bce":
        if objective != "horizon_bce":
            raise ValueError("minority_event_horizon_bce requires horizon_bce objective")
        if float(sampling.get("horizon_days", 365)) not in horizons:
            raise ValueError("Sampling horizon must be on the BCE horizon grid")
        if int(sampling.get("max_draws_per_patient", 2)) < 1:
            raise ValueError("Sampling exposure cap must be positive")

    families = dict(cross.get("outcome_families", {}))
    if cross.get("aggregation") == "endpoint_group_macro" or cross.get("endpoint_groups"):
        from opera.functional.endpoint_groups import validate_endpoint_groups
        validate_endpoint_groups(cross.get("endpoint_groups", {}), outcomes)
        if objective != "horizon_bce" or competing.get("weighter", "uniform") != "uniform":
            raise ValueError("Endpoint-group pilot requires uniform horizon BCE.")
        if float(competing.get("contrastive_loss_weight", 0)) != 0:
            raise ValueError("Endpoint-group pilot is direct BCE only.")
        if training.get("limit_val_batches", 1.) != 1.:
            raise ValueError("Endpoint-group selection requires complete tuning validation.")
        if min(int(training.get("selection_min_cases", 20)),
               int(training.get("selection_min_controls", 20))) < 1:
            raise ValueError("Selection support thresholds must be positive.")
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
        "source_anchored_adapters",
        "unanchored_adapters",
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
    if readout_mode in {"source_anchored_adapters", "unanchored_adapters"}:
        fine_names = list(competing.get("fine_names", []))
        if not fine_names or len(set(fine_names)) != len(fine_names):
            raise ValueError("Adapter readout requires unique fine_names")
        if objective != "horizon_bce" or head_mode != "linear":
            raise ValueError("Adapter readout requires horizon_bce and a linear head")
        if model.get("pooling") != "cls_last":
            raise ValueError("Adapter readout requires cls_last pooling")
        if float(competing.get("contrastive_loss_weight", 0)) != 0:
            raise ValueError("Adapter readout requires direct risk training")
        initialization = config.get("adapter_initialization", "shared_bce")
        if initialization not in {"shared_bce", "dapt_fresh"}:
            raise ValueError("adapter_initialization must be shared_bce or dapt_fresh")
        if initialization == "shared_bce" and not config.get("source_opera_ckpt"):
            raise ValueError("Adapter readout requires source_opera_ckpt")
        if initialization == "dapt_fresh" and config.get("source_opera_ckpt"):
            raise ValueError("dapt_fresh adapters must not load source_opera_ckpt")
        if competing.get("adapter_query_mode", "additive") not in {"additive", "interacting"}:
            raise ValueError("adapter_query_mode must be additive or interacting")
        if int(competing.get("adapter_query_hidden_dim", 128)) < 1:
            raise ValueError("adapter_query_hidden_dim must be positive")
        if competing.get("adapter_source_mode", "both") not in {
            "both", "disease_only", "outcome_only", "none"
        }:
            raise ValueError("adapter_source_mode must be both, disease_only, outcome_only, or none")
        shared_init = float(competing.get("adapter_shared_logit_init", 3.0))
        if not math.isfinite(shared_init):
            raise ValueError("adapter_shared_logit_init must be finite")
    adapter_epsilon = training.get("adapter_optimizer_epsilon")
    if adapter_epsilon is not None:
        if readout_mode not in {"source_anchored_adapters", "unanchored_adapters"}:
            raise ValueError("adapter_optimizer_epsilon requires source adapters")
        if not math.isfinite(float(adapter_epsilon)) or float(adapter_epsilon) <= 0:
            raise ValueError("adapter_optimizer_epsilon must be finite and positive")

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
