import copy
from types import SimpleNamespace

import pytest
import torch

from opera.modules.lightningmodules.OperaContrastiveModule import OperaContrastiveModule
from opera.modules.networks.source_anchored_readout import SourceAnchoredReadout
from opera.run.preflight_direct_competing_risk import validate_adapter_epsilon_match


class _Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = torch.nn.Linear(8, 8)
        self.source_adapter_readout = SourceAnchoredReadout(8, ['A', 'B'], ['x', 'y'])
        self.head = torch.nn.Linear(8, 2)


def _optimizer(model, epsilon, scope="matrices"):
    module = OperaContrastiveModule(
        model, ['x', 'y'], dapt_anchor_weight=0, learning_rate=5e-4,
        encoder_lr_multiplier=.05, optimizer_epsilon=1e-6,
        adapter_optimizer_epsilon=epsilon, adapter_epsilon_scope=scope,
        scheduler_warmup_epochs=0,
    )
    module._trainer = SimpleNamespace(estimated_stepping_batches=30, max_epochs=30)
    return module.configure_optimizers()


def test_lower_epsilon_only_changes_adapter_matrix_updates():
    torch.manual_seed(42)
    baseline = _Model()
    candidate = copy.deepcopy(baseline)
    old, new = _optimizer(baseline, None), _optimizer(candidate, 1e-8)
    assert len(old.param_groups) == 2
    assert len(new.param_groups) == 3
    matrix_ids = {id(candidate.source_adapter_readout.down), id(candidate.source_adapter_readout.up)}
    groups = {id(param): group for group in new.param_groups for param in group['params']}
    assert len(groups) == sum(param.requires_grad for param in candidate.parameters())
    for name, parameter in candidate.named_parameters():
        group = groups[id(parameter)]
        assert group['eps'] == (1e-8 if id(parameter) in matrix_ids else 1e-6)
        assert group['lr'] == pytest.approx(5e-4 * (.05 if name.startswith('encoder.') else 1))
    for model, optimizer in [(baseline, old), (candidate, new)]:
        for parameter in model.parameters():
            parameter.grad = torch.full_like(parameter, 1e-8)
        optimizer.step()
    for name, parameter in candidate.named_parameters():
        prior = dict(baseline.named_parameters())[name]
        if id(parameter) not in matrix_ids:
            torch.testing.assert_close(parameter, prior, rtol=0, atol=0)
        else:
            assert not torch.equal(parameter, prior)


def test_router_scope_gives_every_router_parameter_the_adapter_epsilon():
    torch.manual_seed(42)
    model = _Model()
    model.source_adapter_readout = SourceAnchoredReadout(
        8, ['A', 'B'], ['x', 'y'], query_mode='interacting', query_hidden_dim=4)
    optimizer = _optimizer(model, 1e-8, scope='router')
    groups = {id(param): group for group in optimizer.param_groups for param in group['params']}
    for name, parameter in model.named_parameters():
        expected = 1e-8 if name.startswith('source_adapter_readout.') else 1e-6
        assert groups[id(parameter)]['eps'] == expected, name
    # Gradients far below the default epsilon still move every router parameter.
    before = {n: p.detach().clone() for n, p in model.named_parameters()}
    for parameter in model.parameters():
        parameter.grad = torch.full_like(parameter, 1e-8)
    optimizer.step()
    for name, parameter in model.named_parameters():
        if name.startswith('source_adapter_readout.'):
            assert (parameter - before[name]).abs().max() > 1e-5, name


def test_unknown_epsilon_scope_is_rejected():
    with pytest.raises(ValueError, match='adapter_epsilon_scope'):
        _optimizer(_Model(), 1e-8, scope='everything')


@pytest.mark.parametrize('readout_mode', ['source_anchored_adapters', 'unanchored_adapters'])
def test_epsilon_contract_rejects_other_optimizer_or_curriculum_changes(readout_mode):
    reference = {'competing_risk': {'readout_mode': readout_mode, 'curriculum': {'enabled': True}},
                 'training': {'optimizer_epsilon': 1e-6}}
    actual = copy.deepcopy(reference)
    actual['training']['adapter_optimizer_epsilon'] = 1e-8
    validate_adapter_epsilon_match(actual, reference)
    actual['training']['optimizer_epsilon'] = 1e-8
    with pytest.raises(ValueError, match='differ'):
        validate_adapter_epsilon_match(actual, reference)
    actual['training']['optimizer_epsilon'] = 1e-6
    actual['competing_risk']['curriculum']['enabled'] = False
    with pytest.raises(ValueError, match='differ'):
        validate_adapter_epsilon_match(actual, reference)


@pytest.mark.parametrize('epsilon', [0, -1e-8, float('inf'), float('nan')])
def test_invalid_adapter_epsilon_fails_before_training(epsilon):
    with pytest.raises(ValueError, match='finite and positive'):
        _optimizer(_Model(), epsilon)
