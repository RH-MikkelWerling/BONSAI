from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import nn
from torch.utils.data import ConcatDataset, Dataset

from opera.modules.lightningmodules.OperaContrastiveModule import (
    OperaContrastiveModule,
)
from opera.modules.networks.competing_risk import (
    PiecewiseExponentialCompetingRiskLoss,
)
from opera.run.finetune_all_horizons import (
    _set_dataset_cohort_index,
    build_specialist_cohort_config,
    configure_specialist_objective,
    resolve_parent_cohort,
    verify_exact_parent_state,
)


class _IndexedDataset(Dataset):
    def __init__(self, cohort_index: int):
        self.cohort_index = cohort_index

    def __len__(self):
        return 1

    def __getitem__(self, index):
        return index


def test_resolve_parent_cohort_uses_canonical_mapping():
    cohorts = {"DLBCL_like": {}, "MM": {}}
    assert resolve_parent_cohort("DLBCL", None, cohorts) == "DLBCL_like"
    with pytest.raises(ValueError, match="maps to"):
        resolve_parent_cohort("DLBCL", "MM", cohorts)


def test_specialist_cohort_replaces_grouped_filter_without_mutating_source():
    source = {
        "data_dir": "/data",
        "cohort_fine_col": "cohort_grouped",
        "cohort_fine_value": "DLBCL_like",
    }
    result = build_specialist_cohort_config(
        source,
        cohort_fine_col="cohort_fine",
        cohort_fine_value="DLBCL",
        outcome="anemia_g3plus",
    )
    assert result["cohort_fine_col"] == "cohort_fine"
    assert result["cohort_fine_value"] == "DLBCL"
    assert source["cohort_fine_col"] == "cohort_grouped"


def test_specialist_cohort_rejects_parent_excluded_outcome():
    with pytest.raises(ValueError, match="excluded"):
        build_specialist_cohort_config(
            {"exclude_outcomes": ["treatment_failure"]},
            cohort_fine_col="cohort_fine",
            cohort_fine_value="HCL",
            outcome="treatment_failure",
        )


def test_nested_datasets_are_routed_to_parent_index():
    left = _IndexedDataset(0)
    right = _IndexedDataset(0)
    count = _set_dataset_cohort_index(ConcatDataset([left, right]), 7)
    assert count == 2
    assert left.cohort_index == right.cohort_index == 7


def test_specialist_objective_removes_multitask_training_policies():
    loss = PiecewiseExponentialCompetingRiskLoss(
        ["a", "b"],
        [30, 90],
        no_competing_outcomes=[],
        cross_outcome_config={
            "curriculum": {"enabled": True, "n_support_tiers": 2}
        },
        weighter="kendall",
    )
    model = SimpleNamespace(
        competing_risk_loss=loss,
        competing_risk_head=nn.Linear(4, 12),
        contrastive_loss_weight=1.0,
        competing_risk_weight=0.5,
        dapt_anchor_weight=0.2,
        dapt_embedding_store={1: torch.zeros(4)},
    )
    selected = configure_specialist_objective(model, "horizon_bce")
    assert selected == "horizon_bce"
    assert loss.objective == "horizon_bce"
    assert loss.weighter_name == "uniform"
    # Retained for strict checkpoint reconstruction, but bypassed while the
    # specialist objective is uniform.
    assert loss.weighter is not None
    assert not loss.curriculum_enabled
    assert model.contrastive_loss_weight == 0.0
    assert model.competing_risk_weight == 1.0
    assert model.dapt_anchor_weight == 0.0
    assert model.dapt_embedding_store is None


def test_exact_parent_state_verification_detects_changes():
    model = nn.Linear(3, 2)
    checkpoint = {
        "state_dict": {
            "model.weight": model.weight.detach().clone(),
            "model.bias": model.bias.detach().clone(),
        }
    }
    report = verify_exact_parent_state(checkpoint, model)
    assert report["n_state_tensors"] == 2
    with torch.no_grad():
        model.bias.add_(1)
    with pytest.raises(RuntimeError, match="changed learned tensors"):
        verify_exact_parent_state(checkpoint, model)


class _ValidationModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.competing_risk_loss = PiecewiseExponentialCompetingRiskLoss(
            ["a", "b"], [30, 90], no_competing_outcomes=[]
        )


def test_validation_can_select_one_outcome_from_full_parent_head():
    module = OperaContrastiveModule(
        model=_ValidationModel(),
        outcome_names=["b"],
        dapt_anchor_weight=0.0,
        enable_validation_probe=False,
        evaluation_horizons_days=[30, 90],
    )
    log_hazards = torch.zeros(3, 2, 2, 3)
    survival = {
        "b": {
            "times": torch.tensor([10.0, 40.0, 100.0]),
            "events": torch.tensor([1, 0, 2]),
        }
    }
    module._accumulate_competing_risk_validation(log_hazards, survival)
    stored = module._val_cr_predictions["b"]
    assert stored["cif"][0].shape == (3, 2)
