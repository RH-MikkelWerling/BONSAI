import pytest
import torch

from opera.modules.networks.opera_nets import OperaContrastiveModel
from opera.modules.networks.source_anchored_readout import SourceAnchoredReadout
from opera.modules.networks.competing_risk import piecewise_exponential_cif


class _Encoder(torch.nn.Module):
    def forward(self, batch):
        return (batch['input_emb'],)


def _model(mode):
    torch.manual_seed(42)
    return OperaContrastiveModel(
        encoder=_Encoder(), outcome_names=['x', 'y'], hidden_size=16,
        projection_hidden_dim=16, projection_dim=8, pooling='cls_last',
        dapt_anchor_weight=0, outcome_sorted_event_times={'x': torch.tensor([10.]), 'y': torch.tensor([10.])},
        competing_risk_config={
            'loss_weight': 1., 'contrastive_loss_weight': 0., 'objective': 'horizon_bce',
            'head_mode': 'linear', 'readout_mode': 'source_anchored_adapters',
            'fine_names': ['A', 'B'], 'interval_boundaries_days': [30, 365],
            'no_competing_outcomes': [], 'adapter_query_mode': 'interacting',
            'adapter_source_mode': mode, 'adapter_home_floor': .05,
            'adapter_residual_scale': .7, 'adapter_shared_logit_init': 0.,
        },
    )


@pytest.mark.parametrize('mode', ['both', 'disease_only', 'outcome_only', 'none'])
def test_source_mask_through_encoder_router_head_loss_and_cif(mode):
    model = _model(mode)
    batch = {'input_emb': torch.randn(3, 2, 16), 'attention_mask': torch.ones(3, 2, dtype=torch.bool),
             'cohort_fine_index': torch.tensor([0, 1, 0]), 'subject_id': torch.arange(3)}
    survival = {'x': {'times': torch.tensor([10., 365., 365.]), 'events': torch.tensor([1, 0, 0])},
                'y': {'times': torch.tensor([365., 15., 365.]), 'events': torch.tensor([0, 1, 0])}}
    result = model(batch, survival)
    assert torch.isfinite(result['loss'])
    pooled = model.get_objective_representations(batch)
    hazards, routes = model.competing_risk_log_hazards(pooled, batch['cohort_fine_index'], return_routes=True)
    assert routes.shape == (3, 2, 5)
    if mode not in {'both', 'disease_only'}:
        assert torch.count_nonzero(routes[..., 1:3]) == 0
    if mode not in {'both', 'outcome_only'}:
        assert torch.count_nonzero(routes[..., 3:]) == 0
    cif = piecewise_exponential_cif(hazards, [30, 365], [30, 365], time_scale_days=365.25)
    assert cif.shape == (3, 2, 2)
    assert bool((cif[..., 1] >= cif[..., 0]).all())
    result['loss'].backward()
    readout = model.source_adapter_readout
    if mode == 'disease_only':
        assert readout.target_query.weight.grad.abs().sum() > 0
        assert readout.up.grad[2:].abs().sum() == 0
    elif mode == 'outcome_only':
        assert readout.fine_query.weight.grad.abs().sum() > 0
        assert readout.up.grad[:2].abs().sum() == 0
    elif mode == 'none':
        assert torch.equal(routes[..., 0], torch.ones_like(routes[..., 0]))
        assert readout.up.grad is None or readout.up.grad.abs().sum() == 0


def test_source_modes_do_not_change_initial_risk_head_or_parameter_inventory():
    reference = _model('both').state_dict()
    for mode in ['disease_only', 'outcome_only', 'none']:
        state = _model(mode).state_dict()
        assert state.keys() == reference.keys()
        for name, value in reference.items():
            torch.testing.assert_close(state[name], value, rtol=0, atol=0, equal_nan=True, msg=name)


def test_zero_query_embeddings_leave_explicit_home_identity_in_routes():
    readout = SourceAnchoredReadout(16, ['A', 'B'], ['x', 'y'], home_floor=.05, query_mode='interacting')
    with torch.no_grad():
        readout.source_keys.zero_()
        readout.fine_query.weight.zero_()
        readout.target_query.weight.zero_()
    routes = readout.route_weights(torch.zeros(2, 16), torch.tensor([0, 1]))
    assert not torch.equal(routes[0, 0], routes[0, 1])  # outcome home floor
    assert not torch.equal(routes[0, 0], routes[1, 0])  # disease home floor
