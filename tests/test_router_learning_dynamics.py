from types import SimpleNamespace

import torch

from opera.modules.networks.source_anchored_readout import SourceAnchoredReadout
from opera.modules.networks.competing_risk import PiecewiseExponentialCompetingRiskLoss
from opera.run.audit_router_learning_dynamics import projection_diagnostics


def test_legacy_residual_metric_can_be_nonzero_with_no_sources_under_autocast():
    torch.manual_seed(42)
    readout = SourceAnchoredReadout(16, ['A', 'B'], ['x', 'y'], source_mode='none')
    model = SimpleNamespace(source_adapter_readout=readout,
                            competing_risk_head=torch.nn.Linear(16, 2 * 12))
    pooled = torch.randn(32, 16)
    report = projection_diagnostics(model, pooled, torch.zeros(32, dtype=torch.long),
                                    torch.ones(32, 2, dtype=torch.bool), amp_dtype=torch.bfloat16)
    assert report['true_fp32_projected_correction_mean_abs'] == 0
    assert report['same_projection_correction_mean_abs'] == 0
    assert report['zero_adapter_legacy_metric_artifact_mean_abs'] > 0


def test_outcome_home_adapter_receives_gradients_from_patients_of_either_disease():
    torch.manual_seed(42)
    readout = SourceAnchoredReadout(16, ['A', 'B'], ['x', 'y'], source_mode='outcome_only')
    for disease in (0, 1):
        readout.zero_grad(set_to_none=True)
        pooled = torch.randn(1, 16)
        adapted = readout(pooled, torch.tensor([disease]))
        adapted[:, 0].square().mean().backward()
        assert readout.up.grad[2].abs().sum() > 0
        assert readout.up.grad[3].abs().sum() == 0  # borrowed y source is anchored
        assert readout.up.grad[:2].abs().sum() == 0


def test_repeating_patients_does_not_multiply_macro_normalized_adapter_gradient():
    torch.manual_seed(42)
    readout = SourceAnchoredReadout(16, ['A', 'B'], ['x', 'y'], source_mode='outcome_only')
    head = torch.nn.Linear(16, 2 * 2 * 3)
    objective = PiecewiseExponentialCompetingRiskLoss(
        ['x', 'y'], [30, 365], no_competing_outcomes=[], objective='horizon_bce',
        cross_outcome_config={'aggregation': 'endpoint_group_macro',
                              'endpoint_groups': {'x': ['x'], 'y': ['y']}},
    )
    pooled = torch.randn(2, 16)
    gradients = []
    for repeats in (1, 8):
        readout.zero_grad(set_to_none=True)
        adapted = readout(pooled.repeat(repeats, 1), torch.tensor([0, 1]).repeat(repeats))
        hazards = (torch.einsum('boh,okh->bok', adapted, head.weight.reshape(2, 6, 16))
                   + head.bias.reshape(1, 2, 6)).reshape(-1, 2, 2, 3)
        survival = {name: {'times': torch.tensor([10., 365.]).repeat(repeats),
                           'events': torch.tensor([1, 0]).repeat(repeats)} for name in ['x', 'y']}
        objective(hazards, survival)[0].backward()
        gradients.append(readout.up.grad.clone())
    assert torch.allclose(gradients[0], gradients[1], atol=1e-9, rtol=1e-5)


def test_outcome_only_borrowed_updates_reach_other_outcomes_but_not_disabled_diseases():
    torch.manual_seed(7)
    readout = SourceAnchoredReadout(
        16, ['A', 'B'], ['x', 'y'], source_mode='outcome_only', anchored=False,
    )
    readout(torch.randn(2, 16), torch.tensor([0, 1]))[:, 0].square().mean().backward()
    assert readout.up.grad[2].abs().sum() > 0  # own outcome x
    assert readout.up.grad[3].abs().sum() > 0  # borrowed outcome y learns from x
    assert readout.up.grad[:2].abs().sum() == 0  # disease bank remains disabled
