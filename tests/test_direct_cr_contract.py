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
