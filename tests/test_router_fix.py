"""Router fix (lock addendum 2026-10-26): global adapter, router warm-up,
bounded logits, and the six stage-3 fits."""

import pytest
import torch

from opera.modules.networks.source_anchored_readout import SourceAnchoredReadout


def _readout(**options):
    model = SourceAnchoredReadout(12, ["DLBCL", "FL"], ["anemia", "infection"],
                                  bottleneck=3, query_mode="interacting",
                                  query_hidden_dim=16, **options)
    model.set_training_support([10] * 4, [10] * 4, [10] * 4)
    return model


def test_the_global_adapter_leaves_every_other_initialisation_unchanged():
    torch.manual_seed(5)
    plain = _readout()
    after_plain = torch.rand(3)
    torch.manual_seed(5)
    fixed = _readout(global_adapter=True)
    after_fixed = torch.rand(3)
    assert torch.equal(after_plain, after_fixed)  # e.g. the risk head built next
    for name, value in plain.state_dict().items():
        assert torch.equal(value, fixed.state_dict()[name]), name
    assert set(fixed.state_dict()) - set(plain.state_dict()) == {"global_down", "global_up"}


def test_the_global_adapter_is_always_on_and_trained_by_every_target():
    torch.manual_seed(0)
    model = _readout(global_adapter=True, source_mode="none")
    with torch.no_grad():
        model.global_up.normal_()
    patient = torch.randn(4, 12)
    output = model(patient, torch.tensor([0, 1, 0, 1]))
    expected = patient + model.residual_scale * torch.nn.functional.linear(
        torch.nn.functional.gelu(torch.nn.functional.linear(patient, model.global_down)),
        model.global_up)
    assert torch.allclose(output[:, 0], expected) and torch.allclose(output[:, 1], expected)
    output.sum().backward()
    assert model.global_down.grad.abs().sum() > 0 and model.global_up.grad.abs().sum() > 0


def test_warm_up_freezes_the_router_but_not_the_adapters():
    torch.manual_seed(0)
    model = _readout(global_adapter=True, router_warmup_epochs=2)
    assert model.router_frozen  # frozen from the start
    patient = torch.randn(4, 12)
    model(patient, torch.tensor([0, 1, 0, 1])).sum().backward()
    for name in ("source_keys", "patient_query.weight", "shared_logit"):
        grad = dict(model.named_parameters())[name].grad
        assert grad is None or grad.abs().sum() == 0, name
    assert model.down.grad.abs().sum() > 0 and model.global_up.grad.abs().sum() > 0
    model.zero_grad()
    model.router_frozen = False
    model(patient, torch.tensor([0, 1, 0, 1])).sum().backward()
    assert model.source_keys.grad.abs().sum() > 0


def test_bounded_logits_keep_every_source_alive():
    torch.manual_seed(0)
    model = _readout(logit_scale=3.0, home_floor=0.0)
    with torch.no_grad():  # push the router hard towards one source
        model.source_keys.zero_()
        model.source_keys[0] = 1000.0
        model.patient_query.weight.copy_(torch.eye(12) * 1000)
    patient = torch.ones(4, 12)
    weights = model.route_weights(patient, torch.tensor([0, 1, 0, 1]))
    bound = torch.exp(torch.tensor(3.0)) / (torch.exp(torch.tensor(3.0)) + 5 * torch.exp(torch.tensor(-3.0)))
    assert weights.max() <= bound + 1e-6 and weights.min() > 0


def test_the_training_module_opens_the_router_after_the_warm_up():
    from types import SimpleNamespace

    from opera.modules.lightningmodules.OperaContrastiveModule import OperaContrastiveModule

    readout = _readout(router_warmup_epochs=2)
    fake = SimpleNamespace(model=SimpleNamespace(source_adapter_readout=readout),
                           trainer=SimpleNamespace(datamodule=None))
    for epoch, frozen in ((0, True), (1, True), (2, False), (5, False)):
        fake.current_epoch = epoch
        OperaContrastiveModule.on_train_epoch_start(fake)
        assert readout.router_frozen is frozen


@pytest.mark.parametrize("key,value", [("adapter_router_warmup_epochs", -1),
                                       ("adapter_router_warmup_epochs", 1.5),
                                       ("adapter_logit_scale", -2.0),
                                       ("adapter_global", "yes")])
def test_invalid_router_fix_settings_are_rejected(key, value):
    from test_paper_run_bundle import base_config, reference

    from opera.functional.direct_cr_contract import validate_direct_cr_config
    from opera.run.prepare_paper_run_bundle import router_config

    config = router_config(base_config(), reference(), source_mode="both", rank=8,
                           query_mode="interacting", query_width=512)
    config["competing_risk"][key] = value
    with pytest.raises(ValueError):
        validate_direct_cr_config(config)


def test_stage_three_adds_six_fits_and_reuses_every_earlier_one():
    from test_paper_run_bundle import base_config, reference

    from opera.run.prepare_paper_run_bundle import build_bundle
    from opera.run.paper_prediction_queue import canonical_models

    lock = {"source_mode": "both", "rank": 8, "query_mode": "interacting", "query_width": 512}
    stage2 = build_bundle(base_config(), reference(), {"L": "/x/best.ckpt"}, selected_router=lock)
    stage3 = build_bundle(base_config(), reference(), {"L": "/x/best.ckpt"},
                          selected_router=lock, router_fix=True)
    assert [e["config"] for e in stage3[: len(stage2)]] == [e["config"] for e in stage2]
    fixes = {canonical_models(e, e["config"], {"selected_router": lock})[0]: e
             for e in stage3[len(stage2):]}
    assert set(fixes) == {"R_AB_L", "RBH_AB_L", "N_A_L", "R_A_L", "R_ABT_L", "R_AB_s43_L"}
    risk = {alias: e["config"]["competing_risk"] for alias, e in fixes.items()}
    locked = next(e["config"]["competing_risk"] for e in stage2
                  if e["name"].endswith("final_source_both"))
    # R_AB is the locked router plus the two changes, nothing else.
    changed = {k for k in risk["R_AB_L"] if risk["R_AB_L"][k] != locked.get(k)}
    assert changed == {"adapter_global", "adapter_router_warmup_epochs"}
    assert risk["RBH_AB_L"]["adapter_borrowing"] is False
    assert risk["N_A_L"]["adapter_source_mode"] == "none"
    assert risk["R_A_L"]["adapter_router_warmup_epochs"] == 0
    assert risk["R_ABT_L"]["adapter_logit_scale"] == 3.0
    assert fixes["R_AB_s43_L"]["seed"] == 43
    assert {k: v for k, v in risk["R_AB_s43_L"].items()} == risk["R_AB_L"]


def test_own_sources_outside_the_mix_are_never_routed_and_always_applied():
    torch.manual_seed(0)
    model = _readout(own_outside=True, home_floor=0.0)
    with torch.no_grad():
        model.up.normal_()
    patient = torch.randn(4, 12)
    fine = torch.tensor([0, 1, 0, 1])
    output, weights = model(patient, fine, return_weights=True)
    rows = torch.arange(4)
    # Own disease and own outcome get no route weight ...
    assert torch.all(weights[rows, :, 1 + fine] == 0)
    for outcome in range(2):
        assert torch.all(weights[:, outcome, 1 + 2 + outcome] == 0)
    # ... but their corrections are applied in full: with every borrowed
    # source removed, the output is patient + scale * (own disease + own outcome).
    alone = model(patient, fine, removed_sources=["disease:DLBCL", "disease:FL",
                                                  "outcome:anemia", "outcome:infection"])
    own = model._own_corrections(patient, fine)
    assert torch.allclose(alone, patient[:, None, :] + model.residual_scale * own, atol=1e-5)
    # Own adapters train with live parameters.
    output.sum().backward()
    assert model.up.grad[0].abs().sum() > 0 and model.up.grad[2].abs().sum() > 0


def test_own_outside_requires_no_home_floor():
    with pytest.raises(ValueError):
        _readout(own_outside=True, home_floor=0.05)


def test_without_borrowing_own_outside_leaves_only_own_and_global():
    torch.manual_seed(0)
    model = _readout(own_outside=True, home_floor=0.0, borrowing=False, global_adapter=True)
    weights = model.route_weights(torch.randn(3, 12), torch.tensor([0, 1, 0]))
    assert torch.allclose(weights[..., 0], torch.ones(3, 2))  # everything on "borrow nothing"


def test_stage_four_adds_the_own_outside_fits():
    from test_paper_run_bundle import base_config, reference

    from opera.run.prepare_paper_run_bundle import build_bundle
    from opera.run.paper_prediction_queue import canonical_models

    lock = {"source_mode": "both", "rank": 8, "query_mode": "interacting", "query_width": 512}
    stage3 = build_bundle(base_config(), reference(), {"L": "/x/best.ckpt"},
                          selected_router=lock, router_fix=True)
    stage4 = build_bundle(base_config(), reference(), {"L": "/x/best.ckpt"},
                          selected_router=lock, router_fix=True, router_own=True)
    assert [e["config"] for e in stage4[: len(stage3)]] == [e["config"] for e in stage3]
    new = {canonical_models(e, e["config"], {"selected_router": lock})[0]: e["config"]["competing_risk"]
           for e in stage4[len(stage3):]}
    assert set(new) == {"R_O_L", "RBH_O_L", "R_O_s43_L"}
    r_ab = next(e["config"]["competing_risk"] for e in stage3 if e["name"].endswith("fix_R_AB"))
    changed = {k for k in new["R_O_L"] if new["R_O_L"][k] != r_ab.get(k)}
    assert changed == {"adapter_own_outside", "adapter_home_floor"}
    assert new["R_O_L"]["adapter_home_floor"] == 0.0
    assert new["RBH_O_L"]["adapter_borrowing"] is False


def test_stage_five_adds_ipw_sampling_fits_and_changes_only_sampling():
    from test_paper_run_bundle import base_config, reference

    from opera.functional.direct_cr_contract import validate_direct_cr_config
    from opera.run.prepare_paper_run_bundle import build_bundle
    from opera.run.paper_prediction_queue import canonical_models

    lock = {"source_mode": "both", "rank": 8, "query_mode": "interacting", "query_width": 512}
    args = dict(selected_router=lock, router_fix=True, router_own=True)
    stage4 = build_bundle(base_config(), reference(), {"L": "/x/best.ckpt"}, **args)
    stage5 = build_bundle(base_config(), reference(), {"L": "/x/best.ckpt"}, **args,
                          sampling_ipw=True)
    assert [e["config"] for e in stage5[: len(stage4)]] == [e["config"] for e in stage4]
    new = {canonical_models(e, e["config"], {"selected_router": lock})[0]: e["config"]
           for e in stage5[len(stage4):]}
    assert set(new) == {"J_IPW_L", "R_AB_IPW_L"}
    r_ab = next(e["config"] for e in stage4 if e["name"].endswith("fix_R_AB"))
    ipw = new["R_AB_IPW_L"]
    assert ipw["competing_risk"] == r_ab["competing_risk"]
    sampling = ipw["training"]["batch_sampling"]
    assert sampling["type"] == "ipw_event_horizon_bce"
    assert sampling["horizons_days"] == ipw["training"]["evaluation_horizons_days"]
    assert ipw["model"]["dapt_anchor_weight"] == 0.0
    for config in new.values():
        validate_direct_cr_config(config)
