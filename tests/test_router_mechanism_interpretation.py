"""Counterexamples that distinguish routing allocation from predictive mechanism."""
import json

import pandas as pd
import pytest
import torch

from opera.modules.networks.source_anchored_readout import SourceAnchoredReadout
from opera.run.router_decision import routing_health


def flat_router(**overrides):
    settings = dict(hidden_size=4, fine_names=[f'D{i:02}' for i in range(24)],
        outcome_names=[f'o{i:02}' for i in range(69)], bottleneck=1,
        residual_scale=.7, home_floor=.05, global_adapter=True)
    settings.update(overrides)
    readout = SourceAnchoredReadout(**settings)
    with torch.no_grad():
        readout.source_keys.zero_()
        readout.shared_logit.zero_()
        readout.up.zero_()
        readout.global_up.zero_()
    return readout


def test_flat_logits_and_home_floors_already_allocate_most_mass_to_outcome_bank():
    readout = flat_router()
    weights = readout.route_weights(torch.zeros(1, 4), torch.tensor([0]))
    # One softmax over 24 disease sources, 69 outcome sources and one null slot.
    assert weights[0, 0, 1:25].sum().item() == pytest.approx(.05 + .9*24/94)
    assert weights[0, 0, 25:].sum().item() == pytest.approx(.05 + .9*69/94)
    assert weights[0, 0, 1].item() == pytest.approx(.05 + .9/94)


def test_noncollapse_health_can_pass_with_zero_corrections_and_minimal_gate_differences(tmp_path):
    readout = flat_router()
    with torch.no_grad():
        readout.patient_query.weight.zero_()
        readout.fine_query.weight.zero_()
        # Four source preferences differ only by a logit of 0.001. No training
        # or patient/outcome dependence, and every adapter output is zero.
        for fine in range(4):
            readout.fine_query.weight[fine, fine] = 1
            readout.source_keys[(fine + 1) % 4, fine] = .002
    patient = torch.randn(4, 4)
    output, weights = readout(patient, torch.arange(4), return_weights=True)
    torch.testing.assert_close(output, patient[:, None, :].expand_as(output), rtol=0, atol=0)
    records = []
    disease_names = ['disease:' + n for n in readout.fine_names]
    outcome_names = ['outcome:' + n for n in readout.outcome_names]
    for fine in range(4):
        for outcome in range(69):
            values = weights[fine, outcome].detach().tolist()
            records.append(dict(fine_index=fine, outcome=readout.outcome_names[outcome],
                disease_weights=json.dumps(dict(zip(disease_names, values[1:25]))),
                outcome_weights=json.dumps(dict(zip(outcome_names, values[25:])))))
    routes, support = tmp_path / 'routes.parquet', tmp_path / 'support.json'
    pd.DataFrame(records).to_parquet(routes)
    support.write_text(json.dumps(dict(sources=disease_names + outcome_names, shrinkage=[1.] * 93)))
    result = routing_health(routes, support)
    assert result['healthy'], result
    assert result['criteria']['top_source_share']['share'] == .25
    assert result['criteria']['distinct_top_borrowed_across_diseases']['count'] == 4


def test_warmup_freezes_query_parameters_but_not_weights_as_patient_input_changes():
    readout = flat_router(router_warmup_epochs=2).train()
    with torch.no_grad():
        readout.patient_query.weight.copy_(torch.eye(4))
        readout.source_keys[0, 0] = 2.
    a = readout.route_weights(torch.zeros(1, 4), torch.tensor([0]))
    b = readout.route_weights(torch.ones(1, 4), torch.tensor([0]))
    assert readout.router_frozen
    assert not a.requires_grad and not b.requires_grad
    assert not torch.equal(a, b)


def test_identical_global_displacements_can_change_two_outcome_logits_in_opposite_directions():
    readout = flat_router(source_mode='none')
    with torch.no_grad():
        readout.global_down.zero_()
        readout.global_down[0, 0] = 1.
        readout.global_up[0, 0] = 1.
    patient = torch.ones(1, 4)
    delta = readout(patient, torch.tensor([0])) - patient[:, None]
    torch.testing.assert_close(delta[:, 0], delta[:, 1], rtol=0, atol=0)
    heads = torch.tensor([[1., 0., 0., 0.], [-1., 0., 0., 0.]])
    change = torch.einsum('boh,oh->bo', delta[:, :2], heads)
    assert change[0, 0] > 0 and change[0, 1] < 0


def test_nonzero_source_vectors_can_cancel_within_the_routed_sum():
    readout = flat_router(home_floor=0., source_mode='disease_only')
    with torch.no_grad():
        readout.down.zero_()
        readout.down[:2, 0, 0] = 1.
        readout.up[0, 0, 0] = 2.
        readout.up[1, 0, 0] = -2.
    patient = torch.ones(1, 4)
    output, weights = readout(patient, torch.tensor([0]), return_weights=True)
    assert readout.up[:2].abs().sum() > 0
    assert weights[0, 0, 1] > 0 and weights[0, 0, 2] > 0
    torch.testing.assert_close(output, patient[:, None].expand_as(output), rtol=0, atol=0)
