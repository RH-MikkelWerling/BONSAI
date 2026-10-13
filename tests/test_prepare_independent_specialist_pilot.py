import copy
from pathlib import Path

import pytest
import yaml

from opera.run.prepare_independent_specialist_pilot import specialist_pilot_config


def _base():
    path = Path("opera/configs/generated/direct_cr_bce_lowrank_endpoint_groups_v1.yaml")
    base = yaml.safe_load(path.read_text(encoding="utf-8"))
    base["competing_risk"]["readout_mode"] = "shared"
    base["cross_outcome"]["aggregation"] = "macro"
    return base


def test_independent_specialist_pilot_narrows_cell_without_mutating_base():
    base = _base()
    original = copy.deepcopy(base)
    pilot = specialist_pilot_config(
        base, parent_cohort="DLBCL_like", fine_cohort="DLBCL",
        outcome="overall_survival", label_cache_dir="/secure/cache",
    )
    assert base == original
    assert list(pilot["cohorts"]) == ["DLBCL_like"]
    assert pilot["cohorts"]["DLBCL_like"]["cohort_fine_value"] == "DLBCL"
    assert list(pilot["outcomes"]) == ["overall_survival"]
    assert pilot["cross_outcome"]["endpoint_groups"] == {"overall_survival": ["overall_survival"]}
    assert pilot["training"]["batch_sampling"]["type"] == "random"
    assert pilot["training"]["epochs"] == 5
    assert pilot["paths"]["outcome_label_cache_dir"] == "/secure/cache"
    assert pilot.get("source_opera_ckpt") is None


def test_independent_specialist_pilot_rejects_wrong_parent():
    with pytest.raises(ValueError, match="does not map"):
        specialist_pilot_config(
            _base(), parent_cohort="MM", fine_cohort="DLBCL",
            outcome="overall_survival",
        )


def test_full_specialist_is_not_marked_as_timing_pilot():
    config = specialist_pilot_config(
        _base(), parent_cohort="CLL_SLL", fine_cohort="CLL",
        outcome="overall_survival", max_epochs=30, full_run=True,
    )
    assert config["training"]["epochs"] == 30
    assert config["training"]["eval_monitor_metric"] == "val/loss"
    assert config["cross_outcome"]["endpoint_groups"] == {}
    assert config["specialist_run"]["purpose"] == "independent_dapt_specialist_full_fit"
    assert "specialist_pilot" not in config
