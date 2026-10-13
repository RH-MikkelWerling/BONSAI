import copy

import pytest
import torch

from opera.modules.networks.competing_risk import PiecewiseExponentialCompetingRiskLoss
from opera.modules.networks.source_anchored_readout import SourceAnchoredReadout
from opera.run.preflight_direct_competing_risk import validate_borrowed_update_match, validate_curriculum_off_match


def _reference():
    return {'competing_risk': {'readout_mode': 'source_anchored_adapters',
                              'adapter_bottleneck': 8, 'adapter_residual_scale': .7,
                              'curriculum': {'enabled': True}},
            'training': {'epochs': 30, 'batch_sampling': {'type': 'random'}},
            'paths': {'sequence_data_dir': '/fixed/sequence', 'vocabulary': '/fixed/vocabulary'},
            'seed': 42}


def test_only_gradient_ownership_change_matches_reference():
    reference = _reference()
    actual = copy.deepcopy(reference)
    actual['competing_risk']['readout_mode'] = 'unanchored_adapters'
    actual['competing_risk']['adapter_source_mode'] = 'both'
    validate_borrowed_update_match(actual, reference)


def test_curriculum_off_matches_borrowed_update_reference_without_other_changes():
    reference = _reference()
    reference['competing_risk']['readout_mode'] = 'unanchored_adapters'
    actual = copy.deepcopy(reference)
    actual['competing_risk']['curriculum']['enabled'] = False
    validate_curriculum_off_match(actual, reference)
    actual['training']['optimizer_epsilon'] = 1e-8
    with pytest.raises(ValueError, match='differ'):
        validate_curriculum_off_match(actual, reference)


def test_curriculum_off_activates_every_support_tier_from_first_epoch():
    objective = PiecewiseExponentialCompetingRiskLoss(
        ['x', 'y'], [30, 365], no_competing_outcomes=[], objective='horizon_bce',
        cross_outcome_config={'curriculum': {'enabled': False, 'n_support_tiers': 5,
                                            'warmup_epochs': 1, 'stage_epochs': 1, 'ramp_epochs': 2}},
    )
    objective.set_curriculum_state(0, training=True)
    assert all(objective._curriculum_outcome_multiplier(name) == 1 for name in ['x', 'y'])


@pytest.mark.parametrize('change', ['curriculum', 'rank', 'seed', 'sampling', 'sequence'])
def test_reference_check_rejects_additional_scientific_changes(change):
    reference = _reference()
    actual = copy.deepcopy(reference)
    actual['competing_risk']['readout_mode'] = 'unanchored_adapters'
    if change == 'curriculum':
        actual['competing_risk']['curriculum']['enabled'] = False
    elif change == 'rank':
        actual['competing_risk']['adapter_bottleneck'] = 16
    elif change == 'seed':
        actual['seed'] = 43
    elif change == 'sampling':
        actual['training']['batch_sampling']['type'] = 'weighted'
    else:
        actual['paths']['sequence_data_dir'] = '/other/sequence'
    with pytest.raises(ValueError, match='differ'):
        validate_borrowed_update_match(actual, reference)


def test_full_borrowed_router_updates_both_borrowed_banks_from_one_target_loss():
    torch.manual_seed(42)
    anchored = SourceAnchoredReadout(16, ['A', 'B'], ['x', 'y'], query_mode='interacting')
    live = SourceAnchoredReadout(16, ['A', 'B'], ['x', 'y'], query_mode='interacting', anchored=False)
    live.load_state_dict(anchored.state_dict())
    patient = torch.randn(3, 16)
    fine = torch.zeros(3, dtype=torch.long)
    assert torch.allclose(anchored(patient, fine), live(patient, fine), atol=1e-7, rtol=1e-6)
    head = torch.nn.Linear(16, 12)
    objective = PiecewiseExponentialCompetingRiskLoss(
        ['x', 'y'], [30, 365], no_competing_outcomes=[], objective='horizon_bce')
    survival = {'x': {'times': torch.tensor([10., 365., 365.]),
                      'events': torch.tensor([1, 0, 0])}}
    for readout in (anchored, live):
        adapted = readout(patient, fine)
        hazards = (torch.einsum('boh,okh->bok', adapted, head.weight.reshape(2, 6, 16))
                   + head.bias.reshape(1, 2, 6)).reshape(3, 2, 2, 3)
        objective(hazards, survival)[0].backward()
    for name in ('up', 'down'):
        old, new = getattr(anchored, name), getattr(live, name)
        assert old.grad[1].abs().sum() == 0  # borrowed disease B
        assert old.grad[3].abs().sum() == 0  # borrowed outcome y
        assert new.grad[1].abs().sum() > 0
        assert new.grad[3].abs().sum() > 0
    before = live.up.detach().clone()
    torch.optim.AdamW(live.parameters(), lr=5e-4, eps=1e-6).step()
    assert not torch.equal(before[1], live.up[1])
    assert not torch.equal(before[3], live.up[3])
