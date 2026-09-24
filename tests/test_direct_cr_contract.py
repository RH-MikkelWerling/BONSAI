import copy

import pytest

from opera.functional.direct_cr_contract import validate_direct_cr_config


def _config():
    return {
        "outcomes": {"anemia": {}, "aki": {}},
        "model": {"pooling": "cls_last"},
        "competing_risk": {
            "loss_weight": 1.0,
            "head_mode": "family_trunks",
            "weighter": "uniform",
            "interval_boundaries_days": [30, 60, 90, 180, 365, 730],
            "initialize_from_null_hazards": True,
            "curriculum": {"enabled": True},
        },
        "cross_outcome": {
            "aggregation": "hierarchical_support",
            "outcome_families": {"blood": ["anemia"], "renal": ["aki"]},
        },
        "training": {
            "evaluation_horizons_days": [30, 60, 90, 180, 365, 730],
            "eval_monitor_metric": "val/cr_cif_auc_family_macro",
        },
    }


def test_direct_cr_contract_accepts_family_curriculum_production_config():
    validate_direct_cr_config(_config())


def test_direct_cr_contract_accepts_matched_horizon_bce():
    config = _config()
    config["competing_risk"]["objective"] = "horizon_bce"
    validate_direct_cr_config(config)


def test_direct_cr_contract_rejects_horizon_bce_with_unmatched_evaluation_grid():
    config = _config()
    config["competing_risk"]["objective"] = "horizon_bce"
    config["training"]["evaluation_horizons_days"] = [30, 90, 365]
    with pytest.raises(ValueError, match="exactly match"):
        validate_direct_cr_config(config)


def test_direct_cr_contract_rejects_incomplete_family_mapping():
    config = _config()
    config["cross_outcome"]["outcome_families"].pop("renal")
    with pytest.raises(ValueError, match="cover configured outcomes exactly once"):
        validate_direct_cr_config(config)


def test_direct_cr_contract_accepts_kendall_null_as_separate_macro_ablation():
    config = copy.deepcopy(_config())
    config["competing_risk"]["weighter"] = "kendall_null"
    config["cross_outcome"]["aggregation"] = "macro"
    validate_direct_cr_config(config)


def test_direct_cr_contract_rejects_hierarchical_kendall_null_combination():
    config = _config()
    config["competing_risk"]["weighter"] = "kendall_null"
    with pytest.raises(ValueError, match="requires cross_outcome.aggregation=macro"):
        validate_direct_cr_config(config)


def test_direct_cr_contract_accepts_isolated_outcome_attention_ablation():
    config = _config()
    config["competing_risk"]["head_mode"] = "linear"
    config["competing_risk"]["readout_mode"] = "outcome_attention"
    config["competing_risk"]["contrastive_loss_weight"] = 0.0
    config["training"]["enable_validation_probe"] = False
    validate_direct_cr_config(config)


def test_direct_cr_contract_rejects_outcome_attention_with_specialized_head():
    config = _config()
    config["competing_risk"]["readout_mode"] = "outcome_attention"
    config["training"]["enable_validation_probe"] = False
    with pytest.raises(ValueError, match="requires head_mode=linear"):
        validate_direct_cr_config(config)


def test_direct_cr_contract_accepts_shared_attention_prototypes():
    config = _config()
    config["competing_risk"].update(
        {
            "head_mode": "linear",
            "readout_mode": "shared_attention_prototypes",
            "contrastive_loss_weight": 0.0,
            "prototype_attention_num_prototypes": 10,
            "prototype_mixture_temperature": 1.0,
        }
    )
    config["training"]["enable_validation_probe"] = False
    validate_direct_cr_config(config)


def test_direct_cr_contract_rejects_invalid_prototype_count():
    config = _config()
    config["competing_risk"].update(
        {
            "head_mode": "linear",
            "readout_mode": "shared_attention_prototypes",
            "contrastive_loss_weight": 0.0,
            "prototype_attention_num_prototypes": 0,
        }
    )
    config["training"]["enable_validation_probe"] = False
    with pytest.raises(ValueError, match="num_prototypes must be positive"):
        validate_direct_cr_config(config)


def test_direct_cr_contract_accepts_factorized_cohort_outcome_prototypes():
    config = _config()
    config["cohorts"] = {"DLBCL_like": {}, "CLL_SLL": {}}
    config["competing_risk"].update(
        {
            "head_mode": "linear",
            "readout_mode": "factorized_cohort_outcome_prototypes",
            "contrastive_loss_weight": 0.0,
            "prototype_attention_num_prototypes": 10,
            "prototype_interaction_rank": 8,
        }
    )
    config["training"]["enable_validation_probe"] = False
    validate_direct_cr_config(config)


def test_direct_cr_contract_rejects_factorized_attention_without_cohorts():
    config = _config()
    config["competing_risk"].update(
        {
            "head_mode": "linear",
            "readout_mode": "factorized_cohort_outcome_prototypes",
            "contrastive_loss_weight": 0.0,
        }
    )
    config["training"]["enable_validation_probe"] = False
    with pytest.raises(ValueError, match="requires configured cohorts"):
        validate_direct_cr_config(config)
