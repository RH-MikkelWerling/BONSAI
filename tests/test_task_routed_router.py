"""Task-routed candidates (31 October 2026): routing from disease and outcome
only, bounded logits, optional variance-preserving borrowed mixture."""

import json
import math
from types import SimpleNamespace

import pytest
import torch

from opera.diagnostics.router_mechanism import COMPONENTS, FrozenRouterHead
from opera.modules.networks.task_routed_readout import TaskRoutedPoolingReadout
from test_partial_pooling_router import POOLING
from test_router_axis_separation import _model


def _readout(mix="per_source", bound=5.0):
    r = TaskRoutedPoolingReadout(12, ["DLBCL", "FL", "MCL", "CLL"], ["anemia", "infection", "death"],
                                 bottleneck=3, query_mode="interacting", query_hidden_dim=16,
                                 global_adapter=True, tau_prior=10.0, logit_bound=bound, borrow_mix=mix)
    r.set_training_support([10] * 7, [10] * 7, [10] * 7)
    r.set_cell_support([[0, 10, 90], [5, 5, 5], [40, 0, 1000], [3, 3, 3]])
    with torch.no_grad():
        r.up.normal_(std=0.5); r.global_up.normal_(std=0.3)
        r.fine_query.weight.normal_(std=1.0); r.target_query.weight.normal_(std=1.0)
        r.source_keys.normal_(std=1.0)
    return r.eval()


@pytest.mark.parametrize("mix", ["per_source", "variance_preserving"])
def test_forward_is_the_component_sum_and_disease_terms_add_up(mix):
    r = _readout(mix)
    x, fine = torch.randn(5, 12), torch.tensor([0, 1, 2, 3, -1])
    out, w = r(x, fine, return_weights=True)
    vectors = r.component_vectors(x, fine, w)
    torch.testing.assert_close(out, x[:, None] + r.residual_scale * sum(vectors[c] for c in COMPONENTS))
    base, terms = r.disease_terms(x, fine)
    torch.testing.assert_close(base + terms.sum(2), out)


def test_routing_depends_on_the_task_not_the_patient_and_is_bounded():
    r = _readout(bound=5.0)
    w1 = r.route_weights(torch.randn(3, 12), torch.tensor([1, 1, 1]))
    torch.testing.assert_close(w1[0], w1[1]); torch.testing.assert_close(w1[0], w1[2])
    assert r.patient_query.weight.abs().sum() == 0 and not r.patient_query.weight.requires_grad
    # With cosine scores in [-bound, bound], no source can exceed the bounded maximum share.
    available = 7 - 2
    ceiling = math.exp(5) / (math.exp(5) + (available - 1) * math.exp(-5))
    assert float(w1[..., 1:].max()) <= ceiling + 1e-6
    torch.testing.assert_close(w1.sum(-1), torch.ones(3, 3))


def test_variance_preserving_mix_keeps_an_even_mix_from_shrinking():
    r = _readout("variance_preserving")
    with torch.no_grad():
        r.source_keys.zero_()  # even mix
    x, fine = torch.randn(4, 12), torch.tensor([0, 1, 2, 3])
    w = r.route_weights(x, fine)
    plain = super(TaskRoutedPoolingReadout, r).component_vectors(x, fine, w)
    scaled = r.component_vectors(x, fine, w)
    ratio = scaled["borrowed_disease"].norm(dim=-1) / plain["borrowed_disease"].norm(dim=-1).clamp_min(1e-9)
    torch.testing.assert_close(ratio, torch.full_like(ratio, math.sqrt(5)), rtol=1e-4, atol=1e-4)


def test_config_builds_the_candidate_and_the_audit_reconstructs_it(tmp_path):
    settings = dict(POOLING, adapter_task_routing=True, adapter_logit_bound=3.0,
                    adapter_borrow_mix="variance_preserving")
    model = _model("both", **settings).eval()
    r = model.source_adapter_readout
    assert isinstance(r, TaskRoutedPoolingReadout) and r.logit_bound == 3.0
    r.set_training_support([10] * 4, [10] * 4, [10] * 4); r.set_cell_support([[3, 50], [0, 12]])
    with torch.no_grad():
        r.up.normal_(std=0.5); r.fine_query.weight.normal_(); r.target_query.weight.normal_()
        r.source_keys.normal_(); model.competing_risk_head.weight.normal_(std=0.2)
    risk = {"readout_mode": "source_anchored_adapters", "head_mode": "linear", "fine_names": ["A", "B"],
            "interval_boundaries_days": [30, 365], "no_competing_outcomes": [],
            "adapter_query_mode": "interacting", "adapter_source_mode": "both",
            "adapter_residual_scale": 0.7, "adapter_query_hidden_dim": 128, **settings}
    path = tmp_path / "best.ckpt"
    torch.save({"hyper_parameters": {"model_init_config": {"competing_risk_config": risk},
                                     "outcome_names": ["x", "y"]},
                "state_dict": {f"model.{k}": v for k, v in model.state_dict().items()}}, path)
    head = FrozenRouterHead.from_checkpoint(path)
    assert isinstance(head.readout, TaskRoutedPoolingReadout)
    x, fine, idx = torch.randn(4, 16), torch.tensor([0, 1, 0, 1]), torch.tensor([0, 1])
    torch.testing.assert_close(sum(head.terms(x, fine, idx)[0].values()),
                               model.competing_risk_log_hazards(x, fine).flatten(2)[:, idx])


def test_provenance_pins_the_intermediate_partial_pooling_class(tmp_path):
    from opera.run.contrastive_multicohort import _record_router_code_provenance

    _record_router_code_provenance(str(tmp_path), SimpleNamespace(source_adapter_readout=_readout()))
    payload = json.loads((tmp_path / "router_code_provenance_INTERNAL.json").read_text())
    assert payload["loaded_class_file"].endswith("task_routed_readout.py")
    assert any(p.endswith("partial_pooling_readout.py") for p in payload["intermediate_class_files"])


def test_stage_seven_adds_the_two_candidates_and_reuses_everything_else():
    from test_paper_run_bundle import base_config, reference

    from opera.functional.direct_cr_contract import validate_direct_cr_config
    from opera.run.paper_prediction_queue import canonical_models
    from opera.run.prepare_paper_run_bundle import build_bundle, external_routed

    lock = {"source_mode": "both", "rank": 8, "query_mode": "interacting", "query_width": 512}
    args = dict(selected_router=lock, router_fix=True, router_own=True, sampling_ipw=True, router_pp=True)
    stage6 = build_bundle(base_config(), reference(), {"L": "/x/best.ckpt"}, **args)
    stage7 = build_bundle(base_config(), reference(), {"L": "/x/best.ckpt"}, **args, router_pt=True)
    assert [e["config"] for e in stage7[: len(stage6)]] == [e["config"] for e in stage6]
    new = {canonical_models(e, e["config"], {"selected_router": lock})[0]: e["config"]
           for e in stage7[len(stage6):]}
    assert set(new) == {"R_PT5_L", "R_PT3V_L"}
    pp = next(e["config"]["competing_risk"] for e in stage6 if e["name"].endswith("fix_R_PP"))
    for alias, (bound, mix) in {"R_PT5_L": (5.0, "per_source"), "R_PT3V_L": (3.0, "variance_preserving")}.items():
        risk = new[alias]["competing_risk"]
        assert {k for k in risk if risk[k] != pp.get(k)} == {
            "adapter_task_routing", "adapter_logit_bound", "adapter_borrow_mix"}
        assert (risk["adapter_logit_bound"], risk["adapter_borrow_mix"]) == (bound, mix)
        validate_direct_cr_config(new[alias])
    national = external_routed({"competing_risk": {}}, "R_PT3V_L")["competing_risk"]
    assert national["adapter_task_routing"] is True and national["adapter_borrow_mix"] == "variance_preserving"
