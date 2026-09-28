import torch

from opera.modules.networks.source_anchored_readout import SourceAnchoredReadout
from opera.modules.networks.competing_risk import piecewise_exponential_cif


def _readout(anchored=True):
    model = SourceAnchoredReadout(
        12, ["DLBCL", "FL"], ["anemia", "infection"],
        bottleneck=3, anchored=anchored,
    )
    model.set_training_support([10] * 4, [10] * 4, [10] * 4)
    return model


def test_borrowed_sources_do_not_receive_parameter_gradients():
    torch.manual_seed(7)
    model = _readout()
    patient = torch.randn(1, 12, requires_grad=True)
    adapted, weights = model(patient, torch.tensor([0]), return_weights=True)
    adapted[:, 0].square().sum().backward()
    assert model.up.grad[0].abs().sum() > 0  # home DLBCL
    assert model.up.grad[2].abs().sum() > 0  # home anemia
    assert model.up.grad[1].abs().sum() == 0  # borrowed FL
    assert model.up.grad[3].abs().sum() == 0  # borrowed infection
    assert model.source_keys.grad.abs().sum() > 0
    assert patient.grad.abs().sum() > 0
    assert torch.allclose(weights.sum(-1), torch.ones_like(weights[..., 0]))


def test_unanchored_control_updates_borrowed_sources():
    torch.manual_seed(7)
    model = _readout(anchored=False)
    model(torch.randn(1, 12), torch.tensor([0]))[:, 0].square().sum().backward()
    assert model.up.grad[1].abs().sum() > 0
    assert model.up.grad[3].abs().sum() > 0


def test_unknown_fine_falls_back_to_shared_and_removal_is_diagnostic():
    model = _readout()
    patient = torch.randn(2, 12)
    fine = torch.tensor([-1, 0])
    adapted, weights = model(patient, fine, return_weights=True)
    assert torch.allclose(adapted[0], patient[0].expand(2, -1))
    assert torch.all(weights[0, :, 0] == 1)
    removed, removed_weights = model(
        patient, fine, removed_sources=["outcome:anemia"], return_weights=True
    )
    assert torch.all(removed_weights[:, :, 3] == 0)
    assert torch.allclose(removed_weights.sum(-1), torch.ones_like(weights[..., 0]))
    assert not torch.equal(adapted[1], removed[1])


def test_source_identity_and_support_survive_state_roundtrip():
    model = _readout()
    clone = _readout()
    clone.load_state_dict(model.state_dict())
    patient = torch.randn(3, 12)
    fine = torch.tensor([0, 1, -1])
    assert torch.allclose(model(patient, fine), clone(patient, fine))
    assert clone.fine_names == ("DLBCL", "FL")
    assert clone.outcome_names == ("anemia", "infection")
    assert torch.equal(model.source_train_events, clone.source_train_events)


def test_one_route_yields_six_monotone_horizon_predictions():
    model = _readout()
    head = torch.nn.Linear(12, 2 * 2 * 6)
    patient = torch.randn(3, 12)
    adapted, routes = model(patient, torch.tensor([0, 1, -1]), return_weights=True)
    weight = head.weight.reshape(2, 12, 12)
    bias = head.bias.reshape(2, 12)
    hazards = (torch.einsum("boh,okh->bok", adapted, weight) + bias).reshape(3, 2, 2, 6)
    cif = piecewise_exponential_cif(
        hazards, [30, 60, 90, 180, 365], [30, 60, 90, 180, 365, 730],
        time_scale_days=365.25,
    )
    assert routes.shape == (3, 2, 5)
    assert cif.shape == (3, 2, 6)
    assert torch.all(cif[..., 1:] >= cif[..., :-1])
