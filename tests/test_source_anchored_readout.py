import torch
import pytest

from opera.modules.networks.source_anchored_readout import SourceAnchoredReadout
from opera.modules.networks.competing_risk import piecewise_exponential_cif


def _readout(anchored=True, query_mode="additive", source_mode="both"):
    model = SourceAnchoredReadout(
        12, ["DLBCL", "FL"], ["anemia", "infection"],
        bottleneck=3, anchored=anchored, query_mode=query_mode,
        query_hidden_dim=16,
        source_mode=source_mode,
    )
    model.set_training_support([10] * 4, [10] * 4, [10] * 4)
    return model


@pytest.mark.parametrize("mode,active", [
    ("both", (True, True)), ("disease_only", (True, False)),
    ("outcome_only", (False, True)), ("none", (False, False)),
])
def test_source_type_factorial_masks_routes_and_home_gradients(mode, active):
    torch.manual_seed(23)
    model = _readout(query_mode="interacting", source_mode=mode)
    patient = torch.randn(2, 12, requires_grad=True)
    output, weights = model(patient, torch.tensor([0, 1]), return_weights=True)
    assert torch.allclose(weights.sum(-1), torch.ones_like(weights[..., 0]))
    assert bool((weights[..., 1:3] > 0).any()) == active[0]
    assert bool((weights[..., 3:] > 0).any()) == active[1]
    if not active[0]:
        assert torch.count_nonzero(weights[..., 1:3]) == 0
    if not active[1]:
        assert torch.count_nonzero(weights[..., 3:]) == 0
    if mode == "none":
        assert torch.equal(weights[..., 0], torch.ones_like(weights[..., 0]))
        assert torch.allclose(output, patient[:, None, :].expand_as(output))
    output.square().sum().backward()
    if active[0]:
        assert model.up.grad[:2].abs().sum() > 0
    elif model.up.grad is not None:
        assert model.up.grad[:2].abs().sum() == 0
    if active[1]:
        assert model.up.grad[2:].abs().sum() > 0
    elif model.up.grad is not None:
        assert model.up.grad[2:].abs().sum() == 0


def test_invalid_source_mode_rejected():
    with pytest.raises(ValueError, match="source_mode"):
        _readout(source_mode="unexpected")


def test_loose_router_floor_and_support_shrinkage():
    model = SourceAnchoredReadout(
        12, ["DLBCL", "FL"], ["anemia", "infection"],
        residual_scale=.7, home_floor=.02, query_mode="interacting",
    )
    model.set_training_support([10] * 4, [10] * 4, [10] * 4, event_tau=0.0)
    assert torch.equal(model.source_shrinkage, torch.ones(4))
    weights = model.route_weights(torch.randn(3, 12), torch.tensor([0, 1, 0]))
    assert torch.allclose(weights.sum(-1), torch.ones_like(weights[..., 0]))
    assert torch.isfinite(weights).all()


@pytest.mark.parametrize("query_mode", ["additive", "interacting"])
@pytest.mark.parametrize("anchored", [True, False])
@pytest.mark.parametrize("device,dtype", [("cpu", torch.bfloat16), ("cuda", torch.float16)])
def test_autocast_forward_backward(device, dtype, anchored, query_mode):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    model = _readout(anchored=anchored, query_mode=query_mode).to(device)
    patient = torch.randn(3, 12, device=device, requires_grad=True)
    with torch.autocast(device_type=device, dtype=dtype):
        adapted, weights = model(
            patient, torch.tensor([0, 1, -1], device=device), return_weights=True
        )
        loss = adapted.square().mean()
    loss.backward()
    assert adapted.dtype == torch.float32
    assert torch.isfinite(adapted).all()
    assert torch.isfinite(weights).all()
    assert torch.isfinite(patient.grad).all()
    assert model.up.grad is not None and torch.isfinite(model.up.grad).all()
    assert torch.allclose(adapted[2], patient[2].expand(2, -1))


@pytest.mark.parametrize("query_mode", ["additive", "interacting"])
def test_borrowed_sources_do_not_receive_parameter_gradients(query_mode):
    torch.manual_seed(7)
    model = _readout(query_mode=query_mode)
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
    if query_mode == "interacting":
        # A zero final layer makes the two arms identical at initialization.
        # Its gradient is live immediately; upstream interaction gradients
        # become live after the first update.
        final = model.query_interaction[-1]
        assert final.weight.grad is not None
        assert torch.isfinite(final.weight.grad).all()
        assert final.weight.grad.abs().sum() > 0
        with torch.no_grad():
            final.weight.add_(0.1 * final.weight.grad)
        model.zero_grad(set_to_none=True)
        model(patient.detach(), torch.tensor([0]))[:, 0].square().sum().backward()
        first = model.query_interaction[0]
        assert first.weight.grad is not None
        assert torch.isfinite(first.weight.grad).all()
        assert first.weight.grad.abs().sum() > 0


def test_interacting_starts_as_exact_additive_control_without_rng_shift():
    torch.manual_seed(123)
    additive = _readout(query_mode="additive")
    next_additive_draw = torch.rand(3)
    torch.manual_seed(123)
    interacting = _readout(query_mode="interacting")
    next_interacting_draw = torch.rand(3)
    assert torch.equal(next_additive_draw, next_interacting_draw)
    for name, value in additive.state_dict().items():
        assert torch.equal(value, interacting.state_dict()[name]), name
    patient = torch.randn(4, 12)
    fine = torch.tensor([0, 1, 0, -1])
    before_a, route_a = additive(patient, fine, return_weights=True)
    before_i, route_i = interacting(patient, fine, return_weights=True)
    assert torch.equal(route_a, route_i)
    assert torch.equal(before_a, before_i)


@pytest.mark.parametrize("query_mode", ["additive", "interacting"])
def test_target_patient_source_interactions(query_mode):
    torch.manual_seed(17)
    model = _readout(query_mode=query_mode)
    model.home_floor = 0  # Test learned routing, not forced home mass.
    with torch.no_grad():
        model.fine_query.weight.normal_(std=0.5)
        model.target_query.weight.normal_(std=0.5)
        if query_mode == "interacting":
            model.query_interaction[-1].weight.normal_(std=0.1)
    patient = torch.randn(8, 12) * 3
    weights = model.route_weights(patient, torch.zeros(8, dtype=torch.long))
    odds = torch.log(weights[:, :, 1] / weights[:, :, 2])
    target_contrast = odds[:, 0] - odds[:, 1]
    if query_mode == "additive":
        assert target_contrast.std() < 1e-6
    else:
        assert target_contrast.std() > 1e-6
        # Changing disease also changes the target-specific source preference.
        other = model.route_weights(patient, torch.ones(8, dtype=torch.long))
        other_odds = torch.log(other[:, :, 1] / other[:, :, 2])
        assert (target_contrast - (other_odds[:, 0] - other_odds[:, 1])).abs().max() > 1e-6


def test_interacting_query_roundtrip_and_shared_fallback():
    torch.manual_seed(19)
    model = _readout(query_mode="interacting")
    clone = _readout(query_mode="interacting")
    clone.load_state_dict(model.state_dict())
    patient = torch.randn(3, 12)
    fine = torch.tensor([0, 1, -1])
    assert torch.allclose(model(patient, fine), clone(patient, fine))
    assert torch.allclose(model(patient, fine)[2], patient[2].expand(2, -1))
    assert not any(key.startswith("query_interaction.") for key in _readout().state_dict())


def test_interacting_router_config_is_matched_to_additive_control():
    import yaml
    from pathlib import Path
    from opera.functional.direct_cr_contract import validate_direct_cr_config
    from opera.run.generate_sweep_configs import (
        DEFAULT_REGISTRY, _hydra_environment, build_endpoint_group_router_config,
        load_registry,
    )

    root = Path(__file__).resolve().parents[1]
    path = root / "opera/configs/generated/direct_cr_bce_router_interacting_endpoint_groups_v1.yaml"
    actual = yaml.safe_load(path.read_text())
    registry = load_registry(DEFAULT_REGISTRY)
    expected = _hydra_environment(build_endpoint_group_router_config(registry, interacting=True))
    assert actual == expected
    validate_direct_cr_config(actual)
    assert actual["model"]["freeze_encoder"] is True
    actual["competing_risk"].pop("adapter_query_mode")
    actual["competing_risk"].pop("adapter_query_hidden_dim")
    assert actual == _hydra_environment(build_endpoint_group_router_config(registry))


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
