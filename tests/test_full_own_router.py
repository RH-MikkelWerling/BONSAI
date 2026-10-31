"""Full own corrections under task routing (stage 8, 31 October 2026, evening):
own corrections as in R_O, only the borrowed part pooled."""

import pytest
import torch

from opera.diagnostics.router_mechanism import COMPONENTS, FrozenRouterHead
from opera.modules.networks.full_own_routed_readout import FullOwnTaskRoutedReadout
from opera.modules.networks.source_anchored_readout import SourceAnchoredReadout
from test_partial_pooling_router import POOLING
from test_router_axis_separation import _model

FINE, OUTCOMES = ["DLBCL", "FL", "MCL", "CLL"], ["anemia", "infection", "death"]


def _readout(values="raw"):
    r = FullOwnTaskRoutedReadout(12, FINE, OUTCOMES, bottleneck=3, query_mode="interacting",
                                 query_hidden_dim=16, global_adapter=True, tau_prior=10.0,
                                 borrow_values=values)
    r.set_training_support([10] * 7, [10] * 7, [10] * 7)
    r.set_cell_support([[0, 10, 90], [5, 5, 5], [40, 0, 1000], [3, 3, 3]])
    with torch.no_grad():
        r.up.normal_(std=3.0); r.global_up.normal_(std=0.3)
        r.fine_query.weight.normal_(); r.target_query.weight.normal_(); r.source_keys.normal_()
    return r.eval()


@pytest.mark.parametrize("values", ["raw", "equalised"])
def test_forward_is_the_component_sum_and_disease_terms_add_up(values):
    r = _readout(values)
    if values == "equalised":
        r.train(); r(torch.randn(8, 12), torch.tensor([0, 1, 2, 3] * 2)); r.eval()
    x, fine = torch.randn(5, 12), torch.tensor([0, 1, 2, 3, -1])
    out, w = r(x, fine, return_weights=True)
    vectors = r.component_vectors(x, fine, w)
    torch.testing.assert_close(out, x[:, None] + r.residual_scale * sum(vectors[c] for c in COMPONENTS))
    base, terms = r.disease_terms(x, fine)
    torch.testing.assert_close(base + terms.sum(2), out)
    # Unknown disease identity: no corrections other than the global adapter.
    for name in ("own_disease", "own_outcome", "borrowed_disease", "borrowed_outcome"):
        assert float(vectors[name][4].abs().max()) == 0


def test_own_corrections_are_the_own_outside_routers_unbounded_and_at_full_weight():
    r = _readout()
    with torch.no_grad():
        r.down.normal_(std=1.0)
    reference = SourceAnchoredReadout(12, FINE, OUTCOMES, bottleneck=3, query_mode="interacting",
                                      query_hidden_dim=16, global_adapter=True, own_outside=True,
                                      home_floor=0.0)
    reference.load_state_dict(r.state_dict(), strict=False)
    x, fine = torch.randn(6, 12), torch.tensor([0, 1, 2, 3, 0, 2])
    vectors = r.component_vectors(x, fine, r.route_weights(x, fine))
    own = vectors["own_disease"] + vectors["own_outcome"]
    torch.testing.assert_close(own, reference._own_corrections(x, fine))
    assert float(own.norm(dim=-1).max()) > 2  # not soft-normalised below 1


def test_borrowing_shrinks_with_support_and_is_zero_for_pool_scale_zero():
    r = _readout()
    x, fine = torch.randn(4, 12), torch.tensor([2, 2, 2, 2])
    vectors = r.component_vectors(x, fine, r.route_weights(x, fine))
    borrowed = (vectors["borrowed_disease"] + vectors["borrowed_outcome"]).norm(dim=-1).mean(0)
    # MCL: anemia 40 events, infection 0 (all borrowed), death 1000 (almost none).
    assert borrowed[1] > borrowed[0] > borrowed[2]
    with torch.no_grad():
        r.pool_scale.zero_()
    vectors = r.component_vectors(x, fine, r.route_weights(x, fine))
    assert float(vectors["borrowed_disease"].abs().max()) == 0


def test_equalised_values_give_borrowed_sources_the_same_typical_size():
    r = _readout("equalised")
    with torch.no_grad():
        r.up[0].mul_(20.0)  # one source ten-plus times larger than the rest
    r.train()
    x, fine = torch.randn(64, 12), torch.randint(0, 4, (64,))
    r(x, fine)
    assert bool(r.source_norm_seen)
    equalised = r._source_values(x, detached=True).float() * r._equaliser()[None, :, None]
    sizes = equalised.norm(dim=-1).mean(0)
    torch.testing.assert_close(sizes, torch.full_like(sizes, float(sizes.mean())), rtol=1e-4, atol=1e-4)
    # Evaluation does not update the running norms.
    before = r.source_norm.clone(); r.eval(); r(torch.randn(8, 12), fine[:8])
    torch.testing.assert_close(r.source_norm, before)


def test_config_builds_the_candidate_and_the_audit_reconstructs_it(tmp_path):
    settings = dict(POOLING, adapter_task_routing=True, adapter_logit_bound=5.0,
                    adapter_borrow_mix="per_source", adapter_own_full=True,
                    adapter_borrow_values="equalised")
    model = _model("both", **settings).eval()
    r = model.source_adapter_readout
    assert isinstance(r, FullOwnTaskRoutedReadout) and r.borrow_values == "equalised"
    r.set_training_support([10] * 4, [10] * 4, [10] * 4); r.set_cell_support([[3, 50], [0, 12]])
    with torch.no_grad():
        r.up.normal_(std=0.5); r.source_keys.normal_(); model.competing_risk_head.weight.normal_(std=0.2)
        r.source_norm.copy_(torch.tensor([1.0, 2.0, 0.5, 3.0])); r.source_norm_seen.fill_(True)
    risk = {"readout_mode": "source_anchored_adapters", "head_mode": "linear", "fine_names": ["A", "B"],
            "interval_boundaries_days": [30, 365], "no_competing_outcomes": [],
            "adapter_query_mode": "interacting", "adapter_source_mode": "both",
            "adapter_residual_scale": 0.7, "adapter_query_hidden_dim": 128, **settings}
    path = tmp_path / "best.ckpt"
    torch.save({"hyper_parameters": {"model_init_config": {"competing_risk_config": risk},
                                     "outcome_names": ["x", "y"]},
                "state_dict": {f"model.{k}": v for k, v in model.state_dict().items()}}, path)
    head = FrozenRouterHead.from_checkpoint(path)
    assert isinstance(head.readout, FullOwnTaskRoutedReadout)
    x, fine, idx = torch.randn(4, 16), torch.tensor([0, 1, 0, 1]), torch.tensor([0, 1])
    torch.testing.assert_close(sum(head.terms(x, fine, idx)[0].values()),
                               model.competing_risk_log_hazards(x, fine).flatten(2)[:, idx])


def test_stage_eight_is_a_one_factor_sweep_and_reuses_everything_else():
    from test_paper_run_bundle import base_config, reference

    from opera.functional.direct_cr_contract import validate_direct_cr_config
    from opera.run.paper_prediction_queue import ROUTED_MODELS, canonical_models
    from opera.run.prepare_paper_run_bundle import (ROUTER_PF, ROUTER_PF_CONTROLS, build_bundle,
                                                    external_routed)

    lock = {"source_mode": "both", "rank": 8, "query_mode": "interacting", "query_width": 512}
    args = dict(selected_router=lock, router_fix=True, router_own=True, sampling_ipw=True,
                router_pp=True, router_pt=True)
    stage7 = build_bundle(base_config(), reference(), {"L": "/x/best.ckpt"}, **args)
    stage8 = build_bundle(base_config(), reference(), {"L": "/x/best.ckpt"}, **args, router_pf=True)
    assert [e["config"] for e in stage8[: len(stage7)]] == [e["config"] for e in stage7]
    new = {canonical_models(e, e["config"], {"selected_router": lock})[0]: e
           for e in stage8[len(stage7):]}
    assert set(new) == {f"{a}_L" for a in {**ROUTER_PF, **ROUTER_PF_CONTROLS}} and set(new) <= ROUTED_MODELS
    assert list(new)[:4] == ["R_PF5E_L", "R_PF_OFF_L", "R_PH_GE_L", "R_PL8_L"]  # decisive first
    assert list(new)[-2:] == ["R_PF5E_LIVE_L", "R_PF5E_LIVE_F10_L"]
    pt5 = next(e["config"] for e in stage7 if e["name"].endswith("fix_R_PT5"))
    anchor = new["R_PF5E_L"]["config"]
    risk = anchor["competing_risk"]
    assert {k for k in risk if risk[k] != pt5["competing_risk"].get(k)} == {
        "adapter_own_full", "adapter_borrow_values"}
    F10 = {"training.train_patient_fraction", "training.train_fraction_seed"}
    for alias in ("R_PF5E_F10_L", "R_PF_OFF_F10_L", "R_PH_GE_F10_L", "R_PF5E_LIVE_F10_L", "R_PL8_F10_L"):
        training = new[alias]["config"]["training"]
        # The J_L_F10 subset: data efficiency uses fraction 0.1 with the first seed (42).
        assert (training["train_patient_fraction"], training["train_fraction_seed"]) == (0.1, 42)
    for alias, entry in new.items():
        validate_direct_cr_config(entry["config"])
        if alias in ("R_PF5E_L", "R_PF5E_s43_L"):
            continue
        changed = {k for k in {**entry["config"]["competing_risk"], **risk}
                   if entry["config"]["competing_risk"].get(k) != risk.get(k)}
        changed |= {f"training.{k}" for k in {**entry["config"]["training"], **anchor["training"]}
                    if entry["config"]["training"].get(k) != anchor["training"].get(k)}
        expected = {
            "R_PF5_L": {"adapter_borrow_values"}, "R_PF5B_L": {"adapter_borrow_values"},
            "R_PF_OFF_L": {"adapter_borrow_values"}, "R_PF10E_L": {"adapter_logit_bound"},
            "R_PF5EP_L": {"adapter_patient_gate"}, "R_PF5EL_L": {"adapter_borrow_amount"},
            "R_PF5ED_L": {"adapter_borrow_from"}, "R_PF5EO_L": {"adapter_borrow_from"},
            "R_PF5E_LR10_L": {"training.router_lr_multiplier"},
            "R_PF5E_W0_L": {"adapter_router_warmup_epochs"},
            "R_PF5E_FROZEN_L": {"adapter_router_warmup_epochs"},
            "R_PF5E_R32_L": {"adapter_bottleneck"},
            "R_PF_OFF_R32_L": {"adapter_borrow_values", "adapter_bottleneck"},
            "R_PH_G_L": {"adapter_borrow_values", "adapter_disease_groups"},
            "R_PH_GE_L": {"adapter_borrow_values", "adapter_disease_groups", "adapter_outcome_groups"},
            "R_PH_GE_PF5E_L": {"adapter_disease_groups", "adapter_outcome_groups"},
            "R_PF5E_LIVE_L": {"adapter_live_sources"},
            "R_PF5E_F10_L": F10, "R_PF_OFF_F10_L": F10 | {"adapter_borrow_values"},
            "R_PH_GE_F10_L": F10 | {"adapter_borrow_values", "adapter_disease_groups",
                                    "adapter_outcome_groups"},
            "R_PF5E_LIVE_F10_L": F10 | {"adapter_live_sources"},
            "R_PL8_L": {"adapter_borrow_values", "adapter_latent_groups"},
            "R_PL8_F10_L": F10 | {"adapter_borrow_values", "adapter_latent_groups"},
        }[alias]
        assert changed == expected, alias
    frozen = new["R_PF5E_FROZEN_L"]["config"]
    assert frozen["competing_risk"]["adapter_router_warmup_epochs"] == frozen["training"]["epochs"]
    assert new["R_PF5E_s43_L"]["seed"] == 43
    national = external_routed({"competing_risk": {}}, "R_PF5E_L")["competing_risk"]
    assert national["adapter_own_full"] is True and national["adapter_borrow_values"] == "equalised"


def test_contrasts_reach_the_numbers_and_the_ladder():
    from opera.run.analyse_locked_predictions import FULL_OWN
    from opera.run.audit_frozen_router_mechanism import ladder_for
    from opera.run.build_paper_numbers import CODES
    from opera.run.paper_experiments import OWN_REFERENCE, removal_bundle
    from pathlib import Path

    assert set(FULL_OWN) <= set(CODES)
    assert ladder_for("R_PF5E_L") == ["J_L", "R_PF_OFF_L", "R_PF5E_FROZEN_L", "R_PF5E_L"]
    assert ladder_for("R_PH_GE_L")[-1] == "R_PH_GE_L"
    assert OWN_REFERENCE["R_PF5_L"] == "R_PF_OFF_L" and OWN_REFERENCE["R_PF5E_R32_L"] == "R_PF_OFF_R32_L"
    assert removal_bundle(Path("/r"), "R_PH_G_L") == Path("/r/bundles/stage8")


def test_off_borrows_nothing_and_bounded_caps_each_source():
    off, bounded = _readout("off"), _readout("bounded")
    bounded.load_state_dict(off.state_dict())
    x, fine = torch.randn(6, 12) * 5, torch.tensor([0, 1, 2, 3, 0, 1])
    v = off.component_vectors(x, fine, off.route_weights(x, fine))
    assert float(v["borrowed_disease"].abs().max()) == 0 and float(v["borrowed_outcome"].abs().max()) == 0
    raw = bounded._source_values(x, detached=True)
    assert float(bounded._borrowed_values(raw).norm(dim=-1).max()) < 1


def _variant(**options):
    r = FullOwnTaskRoutedReadout(12, FINE, OUTCOMES, bottleneck=3, query_mode="interacting",
                                 query_hidden_dim=16, global_adapter=True, tau_prior=10.0, **options)
    r.set_training_support([10] * 7, [10] * 7, [10] * 7)
    r.set_cell_support([[0, 10, 90], [5, 5, 5], [40, 0, 1000], [3, 3, 3]])
    with torch.no_grad():
        r.up.normal_(std=3.0); r.fine_query.weight.normal_(); r.target_query.weight.normal_()
        r.source_keys.normal_()
    return r


def test_patient_gate_starts_as_task_routing_and_can_learn():
    plain, gated = _variant(), _variant(patient_gate=True)
    own_query = gated.patient_query.weight.detach().clone()
    gated.load_state_dict(plain.state_dict(), strict=False)
    with torch.no_grad():  # keep the gated model's own (random) patient query
        gated.patient_query.weight.copy_(own_query)
    assert float(own_query.abs().sum()) > 0 and gated.patient_query.weight.requires_grad
    x, fine = torch.randn(5, 12), torch.tensor([0, 1, 2, 3, 0])
    torch.testing.assert_close(gated.route_weights(x, fine), plain.route_weights(x, fine))
    gated.train(); gated.router_frozen = False
    gated(x, fine).pow(2).sum().backward()
    assert gated.patient_gate.grad is not None and float(gated.patient_gate.grad.abs()) > 0
    with torch.no_grad():
        gated.patient_gate.fill_(1.0)
    w = gated.eval().route_weights(x, torch.tensor([1] * 5))
    assert not torch.allclose(w[0], w[1])  # routing now depends on the patient


def test_learned_amount_starts_at_the_support_value_and_survives_reloading():
    support, learned = _variant(), _variant(borrow_amount="learned")
    fine = torch.tensor([0, 1, 2, 3])
    expected = (support.cell_train_events.float() / (support.cell_train_events.float() + 10.0)).clamp(1e-3, 1 - 1e-3)
    torch.testing.assert_close(learned.own_weight(fine), expected[fine])
    with torch.no_grad():
        learned.own_logit.fill_(2.0)
    learned.set_cell_support([[1, 1, 1]] * 4)  # e.g. on resume: must not reset the learned share
    assert float(learned.own_logit.min()) == 2.0


@pytest.mark.parametrize("bank", ["disease", "outcome"])
def test_borrowing_from_one_bank_only(bank):
    r = _variant(borrow_from=bank)
    w = r.route_weights(torch.randn(4, 12), torch.tensor([0, 1, 2, -1]))
    n = r.n_disease
    other = w[:3, :, 1 + n:] if bank == "disease" else w[:3, :, 1:1 + n]
    assert float(other.abs().max()) == 0
    torch.testing.assert_close(w.sum(-1), torch.ones(4, 3))
    assert float(w[3, :, 0].min()) == 1  # unknown disease keeps the fallback


def test_group_adapters_are_shared_within_a_lineage_and_enter_the_borrowed_components():
    groups = {"DLBCL": "aggressive", "FL": "indolent", "MCL": "aggressive", "CLL": "indolent"}
    endpoints = {"anemia": "blood", "infection": "infection", "death": "death"}
    r = _variant(borrow_values="off", disease_groups=groups, outcome_groups=endpoints).eval()
    with torch.no_grad():
        r.group_up.normal_(std=3.0)
    x = torch.randn(1, 12).expand(4, -1)
    fine = torch.tensor([0, 1, 2, 3])
    v = r.component_vectors(x, fine, r.route_weights(x, fine))
    # Same patient: DLBCL and MCL share the lineage correction, FL and CLL another.
    torch.testing.assert_close(v["borrowed_disease"][0], v["borrowed_disease"][2])
    assert not torch.allclose(v["borrowed_disease"][0], v["borrowed_disease"][1])
    assert float(v["borrowed_outcome"].abs().max()) > 0
    out = r(x, fine)
    torch.testing.assert_close(out, x[:, None] + r.residual_scale * sum(v[c] for c in COMPONENTS))
    base, terms = r.disease_terms(x, fine)
    torch.testing.assert_close(base + terms.sum(2), out)
    with pytest.raises(ValueError, match="No group"):
        _variant(disease_groups={"DLBCL": "x"})


def test_router_learning_rate_multiplier_splits_matrices_from_routing(monkeypatch):
    import opera.modules.lightningmodules.OperaContrastiveModule as lightning

    monkeypatch.setattr(lightning, "optimizer_with_warmup", lambda optimizer, *_: optimizer)
    settings = dict(POOLING, adapter_task_routing=True, adapter_own_full=True,
                    adapter_disease_groups={"A": "g", "B": "g"})
    model = _model("both", **settings)
    lrs = {}
    for multiplier in (1.0, 10.0):
        module = lightning.OperaContrastiveModule(
            model=model, outcome_names=["x", "y"], learning_rate=1e-4,
            adapter_optimizer_epsilon=1e-10, adapter_epsilon_scope="router",
            router_lr_multiplier=multiplier)
        module._trainer = object()  # only passed through to the patched scheduler
        names = {id(p): n for n, p in model.named_parameters()}
        lrs[multiplier] = {names[id(p)]: g["lr"] for g in module.configure_optimizers().param_groups
                           for p in g["params"]}
    assert set(lrs[1.0]) == set(lrs[10.0])  # no parameter dropped
    for name in ("down", "up", "global_down", "global_up", "group_down", "group_up"):
        assert lrs[10.0][f"source_adapter_readout.{name}"] == pytest.approx(1e-4)
    for name in ("source_keys", "fine_query.weight", "target_query.weight", "tau_log_ratio", "pool_scale"):
        assert lrs[10.0][f"source_adapter_readout.{name}"] == pytest.approx(1e-3)
    with pytest.raises(ValueError, match="adapter_epsilon_scope"):
        lightning.OperaContrastiveModule(model=model, outcome_names=["x", "y"], router_lr_multiplier=10.0)


def test_validation_logs_correction_sizes_and_pooling_state():
    settings = dict(POOLING, adapter_task_routing=True, adapter_own_full=True,
                    adapter_borrow_values="raw")
    model = _model("both", **settings)
    model.source_adapter_readout.set_cell_support([[3, 50], [0, 12]])
    batch = {'input_emb': torch.randn(3, 2, 16), 'attention_mask': torch.ones(3, 2, dtype=torch.bool),
             'cohort_fine_index': torch.tensor([0, 1, 0]), 'subject_id': torch.arange(3)}
    survival = {'x': {'times': torch.tensor([10., 365., 365.]), 'events': torch.tensor([1, 0, 0])},
                'y': {'times': torch.tensor([365., 15., 365.]), 'events': torch.tensor([0, 1, 0])}}
    train_logs = model.train()(batch, survival)
    assert {"cr/pool_tau", "cr/pool_scale"} <= set(train_logs) and "cr/correction_norm/own" not in train_logs
    logs = model.eval()(batch, survival)
    for key in ("cr/patient_norm", "cr/correction_norm/own", "cr/correction_norm/borrowed",
                "cr/correction_norm/global"):
        assert torch.isfinite(logs[key])


def test_live_sources_let_borrowers_train_other_diseases_adapters():
    for live in (False, True):
        r = _variant(live_sources=live)
        r.train(); r.router_frozen = False
        x = torch.randn(6, 12)
        r(x, torch.tensor([0] * 6)).pow(2).sum().backward()  # only DLBCL patients
        other = r.down.grad[1:r.n_disease]  # FL, MCL, CLL disease adapters
        assert (float(other.abs().sum()) > 0) == live


def test_workers_claim_stage_eight_in_queue_order_and_earlier_stages_unchanged():
    from opera.run import paper_worker

    rows = [{"phase": "router_fix", "seed": 42, "name": n, **({"queue_order": q} if q else {})}
            for n, q in (("seed42/L/fix_R_AB", None), ("seed42/L/fix_R_PF5", 9),
                         ("seed42/L/fix_R_PF_OFF", 2), ("seed42/L/fix_R_PF5E", 1),
                         ("seed43/L/fix_R_PF5E_s43", 5))]
    ordered = sorted(rows, key=lambda r: (paper_worker.priority(r), r.get("queue_order", 0),
                                          r["seed"], r["name"]))
    assert [r["name"].split("fix_")[1] for r in ordered] == ["R_AB", "R_PF5E", "R_PF_OFF", "R_PF5E_s43", "R_PF5"]
    import inspect
    assert 'item[0].get("queue_order", 0)' in inspect.getsource(paper_worker.todo)


def test_latent_groups_are_task_routed_shared_adapters_in_the_borrowed_components():
    r = _variant(borrow_values="off", latent_groups=3).eval()
    with torch.no_grad():
        r.latent_up.normal_(std=3.0)
    x, fine = torch.randn(5, 12), torch.tensor([0, 1, 2, 3, -1])
    w = r.latent_weights(fine.clamp(0, 3))
    assert w.shape == (5, 3, 3)
    torch.testing.assert_close(w.sum(-1), torch.ones(5, 3))
    v = r.component_vectors(x, fine, r.route_weights(x, fine))
    assert float(v["borrowed_disease"][:4].abs().max()) > 0 and float(v["borrowed_disease"][4].abs().max()) == 0
    out = r(x, fine)
    torch.testing.assert_close(out, x[:, None] + r.residual_scale * sum(v[c] for c in COMPONENTS))
    base, terms = r.disease_terms(x, fine)
    torch.testing.assert_close(base + terms.sum(2), out)
    # Every task trains the latent adapters it uses; routing learns after warm-up.
    r.train(); r.router_frozen = False
    r(x, fine).pow(2).sum().backward()
    assert float(r.latent_up.grad.abs().sum()) > 0 and float(r.latent_keys.grad.abs().sum()) > 0
    assert len(r.pooling_summary()["latent_weights"]) == 4


def test_latent_group_routing_learns_during_the_router_warm_up():
    r = _variant(borrow_values="off", latent_groups=3)
    with torch.no_grad():
        r.latent_up.normal_(std=3.0)
    r.train(); r.router_frozen = True
    r(torch.randn(5, 12), torch.tensor([0, 1, 2, 3, 0])).pow(2).sum().backward()
    assert float(r.latent_keys.grad.abs().sum()) > 0
    assert r.source_keys.grad is None or float(r.source_keys.grad.abs().sum()) == 0  # source routing frozen
