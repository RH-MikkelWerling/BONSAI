"""Partial-pooling router (lock amendment 2026-10-30, R_PP): how much to borrow
(w_own = n/(n + tau)) and from which sources (a learned-temperature mix of
soft-normalised borrowed corrections)."""

import csv
import json
import math
from types import SimpleNamespace

import pytest
import torch

from opera.diagnostics.router_mechanism import (
    COMPONENTS, VARIANTS, FrozenRouterHead, TaskReferences, borrowed_mixture_health,
    variant_terms,
)
from opera.modules.networks.partial_pooling_readout import (
    PartialPoolingReadout, soft_normalize,
)
from opera.modules.networks.source_anchored_readout import SourceAnchoredReadout
from test_router_axis_separation import _model

POOLING = dict(adapter_global=True, adapter_partial_pooling=True, adapter_own_outside=True,
               adapter_home_floor=0.0, adapter_router_warmup_epochs=2, adapter_event_tau=10.0)


def _readout(**options):
    readout = PartialPoolingReadout(12, ["DLBCL", "FL", "MCL"], ["anemia", "infection", "death"],
                                    bottleneck=3, query_mode="interacting", query_hidden_dim=16,
                                    global_adapter=True, tau_prior=10.0, **options)
    readout.set_training_support([10] * 6, [10] * 6, [10] * 6)
    readout.set_cell_support([[0, 10, 90], [5, 5, 5], [40, 0, 1000]])
    return readout


def _randomise(readout, std=0.5):
    with torch.no_grad():
        readout.up.normal_(std=std)
        readout.global_up.normal_(std=std)
        readout.patient_query.weight.normal_(std=0.3)
        readout.source_keys.normal_(std=1.0)
    return readout


@pytest.mark.parametrize("option", [{"own_outside": False}, {"home_floor": 0.05},
                                    {"logit_scale": 3.0}, {"source_mode": "disease_only"},
                                    {"borrowing": False}])
def test_the_design_cannot_be_changed_by_options(option):
    with pytest.raises(ValueError):
        _readout(**option)


def test_own_weight_is_the_partial_pooling_rule():
    readout = _readout()
    fine = torch.tensor([0, 1, 2, -1])
    weight = readout.own_weight(fine)
    events = torch.tensor([[0, 10, 90], [5, 5, 5], [40, 0, 1000], [0, 0, 0]], dtype=torch.float)
    expected = events / (events + 10.0)
    expected[3] = 0  # unknown disease: no corrections
    torch.testing.assert_close(weight, expected)
    assert weight[0, 0] == 0 and weight[2, 2] > 0.99
    with torch.no_grad():
        readout.tau_log_ratio.fill_(math.log(2.0))
    torch.testing.assert_close(readout.own_weight(fine)[0, 1], torch.tensor(10 / 30))


def test_start_matches_the_priors_and_borrowed_weights_exclude_own_sources():
    readout = _readout().eval()
    assert float(readout.tau) == pytest.approx(10.0)
    assert float(readout.temperature) == pytest.approx(math.sqrt(12))
    assert float(readout.pool_scale) == 1.0
    with torch.no_grad():
        readout.patient_query.weight.zero_()  # exact start: queries all zero
    fine = torch.tensor([0, 2])
    weights = readout.route_weights(torch.randn(2, 12), fine)
    torch.testing.assert_close(weights.sum(-1), torch.ones(2, 3))
    assert bool((weights[..., 0] == 0).all())  # no "borrow nothing" slot
    for b, d in enumerate(fine.tolist()):
        for o in range(3):
            assert weights[b, o, 1 + d] == 0 and weights[b, o, 1 + 3 + o] == 0
            borrowed = weights[b, o, 1:][weights[b, o, 1:] > 0]
            torch.testing.assert_close(borrowed, torch.full_like(borrowed, 1 / 4))


def test_soft_normalised_sources_make_the_gate_the_only_lever():
    values = torch.randn(5, 7) * 1e4
    assert bool((soft_normalize(values).norm(dim=-1) <= 1 + 1e-6).all())
    small = torch.randn(5, 7) * 1e-3  # near zero it is almost the identity
    torch.testing.assert_close(soft_normalize(small), small, rtol=1e-5, atol=1e-8)
    readout = _randomise(_readout()).eval()
    x, fine = torch.randn(4, 12), torch.tensor([0, 1, 2, 0])
    weights = readout.route_weights(x, fine)
    before = readout.component_vectors(x, fine, weights)
    with torch.no_grad():
        readout.up.mul_(1e3)  # an adapter can no longer buy influence by growing
    after = readout.component_vectors(x, fine, weights)
    own_w = readout.own_weight(fine)
    bound = (readout.pool_scale * own_w).detach()
    for name in ("own_disease", "own_outcome"):
        assert bool((after[name].norm(dim=-1) <= bound + 1e-5).all())
    borrowed = after["borrowed_disease"] + after["borrowed_outcome"]
    assert bool((borrowed.norm(dim=-1) <= (1 - own_w) + 1e-5).all())
    assert not torch.allclose(before["own_disease"], after["own_disease"])


def test_forward_is_exactly_the_component_sum_and_disease_terms_add_up():
    readout = _randomise(_readout()).eval()
    x, fine = torch.randn(5, 12), torch.tensor([0, 1, 2, 0, -1])
    output, weights = readout(x, fine, return_weights=True)
    vectors = readout.component_vectors(x, fine, weights)
    assert set(vectors) == set(COMPONENTS)
    torch.testing.assert_close(output, x[:, None] + readout.residual_scale * sum(vectors.values()))
    base, terms = readout.disease_terms(x, fine)
    torch.testing.assert_close(base + terms.sum(2), output)
    # Unknown disease keeps the shared-path fallback: only the global correction.
    torch.testing.assert_close(output[4], x[4][None] + readout.residual_scale * vectors["global"][4])


def test_borrowed_adapters_are_not_retrained_by_the_target_but_own_adapters_are():
    readout = _randomise(_readout()).train()
    readout.router_frozen = False
    x = torch.randn(1, 12)
    readout(x, torch.tensor([0]))[0, 1].sum().backward()  # disease 0, outcome 1
    grad = readout.up.grad.norm(dim=(1, 2))
    own = {0, 3 + 1}
    assert all(grad[s] > 0 for s in own)
    assert all(grad[s] == 0 for s in range(6) if s not in own)
    assert readout.tau_log_ratio.grad is not None and readout.tau_log_ratio.grad != 0
    assert readout.temperature_log_ratio.grad is not None and readout.temperature_log_ratio.grad != 0


def test_warm_up_freezes_routing_and_tau_but_not_the_adapters():
    readout = _randomise(_readout()).train()
    readout.router_frozen = True
    readout(torch.randn(3, 12), torch.tensor([0, 1, 2])).sum().backward()
    assert readout.tau_log_ratio.grad is None or readout.tau_log_ratio.grad == 0
    assert readout.temperature_log_ratio.grad is None or readout.temperature_log_ratio.grad == 0
    assert readout.up.grad.abs().sum() > 0


def test_cell_support_must_match_the_roster():
    readout = _readout()
    with pytest.raises(ValueError):
        readout.set_cell_support([[1, 2], [3, 4]])
    with pytest.raises(ValueError):
        readout.set_cell_support([[-1, 0, 0]] * 3)


def _pooling_model():
    model = _model("both", **POOLING).eval()
    readout = model.source_adapter_readout
    readout.set_training_support([10] * 4, [10] * 4, [10] * 4)
    readout.set_cell_support([[3, 50], [0, 12]])
    with torch.no_grad():
        readout.up.normal_(std=0.5)
        readout.global_up.normal_(std=0.3)
        readout.patient_query.weight.normal_(std=0.3)
        readout.source_keys.normal_(std=1.0)
        model.competing_risk_head.weight.normal_(std=0.2)
    return model


def test_the_network_builds_the_partial_pooling_readout_from_its_config():
    model = _pooling_model()
    assert isinstance(model.source_adapter_readout, PartialPoolingReadout)
    assert model.source_adapter_readout.tau_prior == 10.0
    plain = _model("both", adapter_global=True)
    assert type(plain.source_adapter_readout) is SourceAnchoredReadout
    batch = {"input_emb": torch.randn(3, 2, 16), "attention_mask": torch.ones(3, 2, dtype=torch.bool),
             "cohort_fine_index": torch.tensor([0, 1, 0]), "subject_id": torch.arange(3)}
    survival = {"x": {"times": torch.tensor([10., 365., 365.]), "events": torch.tensor([1, 0, 0])},
                "y": {"times": torch.tensor([365., 15., 365.]), "events": torch.tensor([0, 1, 0])}}
    model.train()
    assert torch.isfinite(model(batch, survival)["loss"])


@pytest.mark.parametrize("indices", [[0, 1], [1], [1, 0]])
def test_the_mechanism_audit_reconstructs_the_partial_pooling_forward(indices):
    model = _pooling_model()
    head = FrozenRouterHead(model.source_adapter_readout, model.competing_risk_head.weight,
                            model.competing_risk_head.bias, [30, 365], no_competing_outcomes=[]).eval()
    x, fine, idx = torch.randn(6, 16), torch.tensor([0, 1, 1, 0, 1, 0]), torch.tensor(indices)
    terms, gates = head.terms(x, fine, idx)
    assert set(terms) == {"base", *COMPONENTS}
    production = model.competing_risk_log_hazards(x, fine).flatten(2)[:, idx]
    torch.testing.assert_close(sum(terms.values()), production)
    ref = TaskReferences(head, len(idx))
    ref.add(fine, torch.ones(6, len(idx), dtype=torch.bool), terms, gates,
            head.borrowed_mask(fine, idx))
    variants = dict(variant_terms(head, x, fine, idx, ref.finalize()))
    assert set(variants) == {"FULL", *VARIANTS}
    for value in variants.values():
        assert torch.isfinite(head.cif(value, idx, [30, 365])).all()


def test_the_audit_loads_a_partial_pooling_checkpoint(tmp_path):
    model = _pooling_model()
    risk = {"readout_mode": "source_anchored_adapters", "head_mode": "linear",
            "fine_names": ["A", "B"], "interval_boundaries_days": [30, 365],
            "no_competing_outcomes": [], "adapter_query_mode": "interacting",
            "adapter_source_mode": "both", "adapter_residual_scale": 0.7,
            "adapter_query_hidden_dim": 128, **POOLING}
    state = {f"model.{key}": value for key, value in model.state_dict().items()}
    path = tmp_path / "best.ckpt"
    torch.save({"hyper_parameters": {"model_init_config": {"competing_risk_config": risk},
                                     "outcome_names": ["x", "y"]},
                "state_dict": state}, path)
    head = FrozenRouterHead.from_checkpoint(path)
    assert isinstance(head.readout, PartialPoolingReadout)
    x, fine, idx = torch.randn(4, 16), torch.tensor([0, 1, 0, 1]), torch.tensor([0, 1])
    torch.testing.assert_close(sum(head.terms(x, fine, idx)[0].values()),
                               model.competing_risk_log_hazards(x, fine).flatten(2)[:, idx])


def test_two_sided_health_separates_even_peaked_and_selective_mixtures():
    readout = PartialPoolingReadout(12, [f"D{i}" for i in range(5)], ["a", "b"], bottleneck=3,
                                    global_adapter=True, tau_prior=10.0)
    readout.set_training_support([10] * 7, [10] * 7, [10] * 7)
    weight = torch.zeros(2 * (2 + 1) * 2, 12)
    head = FrozenRouterHead(readout, weight, torch.zeros(len(weight)), [30, 365]).eval()
    x = torch.randn(50, 12)
    fine = torch.arange(50) % 5
    eligible = torch.ones(50, 2, dtype=torch.bool)
    idx = torch.tensor([0, 1])
    with torch.no_grad():
        readout.patient_query.weight.zero_()
    even = borrowed_mixture_health(head, x, fine, eligible, idx)
    assert even["median_entropy_ratio"] == pytest.approx(1.0)
    assert even["not_uniform"] is False and even["not_collapsed"] is False  # one shared top
    with torch.no_grad():
        readout.fine_query.weight.copy_(torch.eye(5, 12) * 100)
        readout.source_keys.copy_(torch.roll(torch.eye(7, 12), 1, 0) * 100)
    peaked = borrowed_mixture_health(head, x, fine, eligible, idx)
    assert peaked["fraction_rows_top_share_above_half"] > 0.5 and peaked["not_collapsed"] is False
    with torch.no_grad():
        # Moderately selective: the preferred source gets about half the weight.
        readout.fine_query.weight.copy_(torch.eye(5, 12) * 2.15)
        readout.source_keys.copy_(torch.roll(torch.eye(7, 12), 1, 0) * 2.15)
    selective = borrowed_mixture_health(head, x, fine, eligible, idx)
    assert selective["not_uniform"] and selective["not_collapsed"]
    assert selective["distinct_top_borrowed_sources_across_diseases"] >= 3


def test_training_setup_counts_cell_events_and_writes_cell_support(tmp_path):
    from opera.run.contrastive_multicohort import _set_source_training_support

    model = SimpleNamespace(source_adapter_readout=_readout())
    dataset = SimpleNamespace(
        fine_index_by_subject={"p1": 0, "p2": 0, "p3": 2},
        outcome_dicts={
            "anemia": {"p1": {"event": 1, "time_days": 10}, "p2": {"event": 0, "time_days": 50},
                       "p3": {"event": 1, "time_days": 5}},
            "death": {"p1": {"event": 1, "time_days": 90}, "p2": {"event": -1, "time_days": 4}},
        })
    _set_source_training_support(model, dataset, str(tmp_path / "source_support.csv"), 10.0)
    events = model.source_adapter_readout.cell_train_events
    assert events.tolist() == [[1, 0, 1], [0, 0, 0], [1, 0, 0]]
    rows = list(csv.DictReader(open(tmp_path / "cell_support.csv")))
    assert len(rows) == 9 and {r["training_events"] for r in rows} == {"0", "1"}


def test_router_provenance_pins_the_subclass_and_keeps_base_payloads(tmp_path):
    from opera.run.contrastive_multicohort import _record_router_code_provenance

    base = SimpleNamespace(source_adapter_readout=SourceAnchoredReadout(
        12, ["DLBCL"], ["anemia"], bottleneck=3))
    _record_router_code_provenance(str(tmp_path / "base"), base)
    payload = json.loads((tmp_path / "base" / "router_code_provenance_INTERNAL.json").read_text())
    assert set(payload) == {"loaded_class_file", "loaded_class_file_sha256", "query_mode",
                            "fine_names", "outcome_names"}
    assert payload["loaded_class_file"].endswith("source_anchored_readout.py")
    pooled = SimpleNamespace(source_adapter_readout=_readout())
    _record_router_code_provenance(str(tmp_path / "pooled"), pooled)
    payload = json.loads((tmp_path / "pooled" / "router_code_provenance_INTERNAL.json").read_text())
    assert payload["loaded_class_file"].endswith("partial_pooling_readout.py")
    assert payload["parent_class_file"].endswith("source_anchored_readout.py")


def test_stage_six_adds_the_two_partial_pooling_fits_and_reuses_every_earlier_one():
    from test_paper_run_bundle import base_config, reference

    from opera.functional.direct_cr_contract import validate_direct_cr_config
    from opera.run.paper_prediction_queue import canonical_models
    from opera.run.prepare_paper_run_bundle import build_bundle

    lock = {"source_mode": "both", "rank": 8, "query_mode": "interacting", "query_width": 512}
    args = dict(selected_router=lock, router_fix=True, router_own=True, sampling_ipw=True)
    stage5 = build_bundle(base_config(), reference(), {"L": "/x/best.ckpt"}, **args)
    stage6 = build_bundle(base_config(), reference(), {"L": "/x/best.ckpt"}, **args, router_pp=True)
    assert [e["config"] for e in stage6[: len(stage5)]] == [e["config"] for e in stage5]
    new = {canonical_models(e, e["config"], {"selected_router": lock})[0]: e["config"]
           for e in stage6[len(stage5):]}
    assert set(new) == {"R_PP_L", "R_PP_s43_L", "R_PP_FROZEN_L"}
    r_ab = next(e["config"]["competing_risk"] for e in stage5 if e["name"].endswith("fix_R_AB"))
    pp = new["R_PP_L"]["competing_risk"]
    assert {k for k in pp if pp[k] != r_ab.get(k)} == {
        "adapter_partial_pooling", "adapter_own_outside", "adapter_home_floor"}
    frozen = new["R_PP_FROZEN_L"]
    assert {k for k in frozen["competing_risk"] if frozen["competing_risk"][k] != pp[k]} == {
        "adapter_router_warmup_epochs"}
    # Frozen for every epoch the fit can run.
    assert frozen["competing_risk"]["adapter_router_warmup_epochs"] >= frozen["training"]["epochs"]
    for config in new.values():
        validate_direct_cr_config(config)
    broken = json.loads(json.dumps(new["R_PP_L"]))
    broken["competing_risk"]["adapter_own_outside"] = False
    with pytest.raises(ValueError):
        validate_direct_cr_config(broken)


def test_a_selected_partial_pooling_router_carries_over_to_the_national_fits():
    from opera.run.prepare_paper_run_bundle import external_routed, routed_fix

    assert routed_fix("R_PP_L") == "R_PP"
    config = external_routed({"competing_risk": {}}, "R_PP_L")
    assert config["competing_risk"]["adapter_partial_pooling"] is True
    assert config["competing_risk"]["adapter_own_outside"] is True
    assert config["competing_risk"]["adapter_global"] is True
    with pytest.raises(ValueError):
        routed_fix("R_UNKNOWN_L")
    with pytest.raises(ValueError):  # the frozen control can never become primary
        routed_fix("R_PP_FROZEN_L")


def test_positive_control_router_finds_the_matching_source_and_it_pays_off():
    """Averaged over all three prespecified seeds (none is picked): for the
    data-poor disease A, whose true risk matches B's, R_PP puts most borrowed
    weight on B and predicts A better than the R_AB-style readout."""
    from opera.diagnostics.router_positive_control import run

    rows = run()
    pp = [r for r in rows if r["design"] == "R_PP"]
    ab = [r for r in rows if r["design"] == "R_AB-style"]
    mean = lambda rs, f: sum(f(r) for r in rs) / len(rs)
    weight_b = mean(pp, lambda r: r["weight_A_from"]["B"])
    assert weight_b > 0.5
    for other in "CD":
        assert weight_b > 2 * mean(pp, lambda r: r["weight_A_from"][other])
    assert mean(pp, lambda r: r["auc_A"]) > 0.75
    assert mean(pp, lambda r: r["auc_A"]) > mean(ab, lambda r: r["auc_A"])
    assert all(r["weight_A_from"]["A"] == 0 for r in pp)  # own source never in the borrowed mix
