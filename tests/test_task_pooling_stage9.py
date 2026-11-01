"""Stage 9 (1 November 2026): the joint model with task-conditioned attention
pooling; each fit differs from J only in the readout and starts exactly at J."""

import pytest
import torch

from opera.modules.networks.opera_nets import OperaContrastiveModel
from test_paper_run_bundle import base_config, reference
from test_router_axis_separation import _Encoder

LOCK = {"source_mode": "both", "rank": 8, "query_mode": "interacting", "query_width": 512}


def _bundles():
    from opera.run.prepare_paper_run_bundle import build_bundle

    args = dict(selected_router=LOCK, router_fix=True, router_own=True, sampling_ipw=True,
                router_pp=True, router_pt=True, router_pf=True)
    before = build_bundle(base_config(), reference(), {"L": "/x/best.ckpt"}, **args)
    after = build_bundle(base_config(), reference(), {"L": "/x/best.ckpt"}, **args, task_pooling=True)
    return before, after


def test_stage_nine_adds_three_readouts_of_the_joint_model_and_nothing_else():
    from opera.functional.direct_cr_contract import validate_direct_cr_config
    from opera.run.paper_prediction_queue import canonical_models

    before, after = _bundles()
    assert [e["config"] for e in after[: len(before)]] == [e["config"] for e in before]
    new = after[len(before):]
    aliases = {canonical_models(e, e["config"], {"selected_router": LOCK})[0]: e for e in new}
    assert set(aliases) == {"J_OA_L", "J_SP_L", "J_FP_L", "J_FPF_L", "J_OA_OWN_L", "J_OA_OWN_PH_L", "J_FPF_OWN_L",
                            "J_FPF_PH_L", "J_FPF_PH_OWN_PH_L", "J_OA_LR_L", "J_OA_LR_OWN_L"}
    joint = next(e["config"] for e in before if e["name"] == "seed42/L/J_all_diseases_all_outcomes")
    for alias, entry in aliases.items():
        config = entry["config"]
        assert entry["queue_order"] < 0 and entry["phase"] == "router_fix"
        diff = {k for k in {**config["competing_risk"], **joint["competing_risk"]}
                if config["competing_risk"].get(k) != joint["competing_risk"].get(k)}
        expected = {"readout_mode"}
        if alias.startswith("J_FPF"):
            expected |= {"prototype_condition", "fine_names", "fine_column"}
        if alias.startswith("J_FPF_PH"):
            expected |= {"prototype_condition_groups"}
        if "_OWN" in alias:
            expected |= {"attention_own_corrections", "attention_own_rank", "adapter_residual_scale",
                         "fine_names", "fine_column"}
        if "_LR" in alias:
            expected |= {"outcome_attention_query_rank"}
        if alias in ("J_OA_OWN_PH_L", "J_FPF_PH_OWN_PH_L"):
            expected |= {"attention_own_disease_groups"}
        assert diff == expected, (alias, diff)
        assert {k: v for k, v in config.items() if k not in ("competing_risk", "supervision_factorial")} == \
               {k: v for k, v in joint.items() if k not in ("competing_risk", "supervision_factorial")}
        assert config["training"]["enable_validation_probe"] is False
        validate_direct_cr_config(config)


@pytest.mark.parametrize("mode", ["outcome_attention", "shared_attention_prototypes",
                                  "factorized_cohort_outcome_prototypes"])
def test_readout_starts_at_the_joint_model_and_trains(mode):
    def build(readout):
        torch.manual_seed(0)
        return OperaContrastiveModel(
            encoder=_Encoder(), outcome_names=["x", "y"], hidden_size=16, projection_hidden_dim=16,
            projection_dim=8, pooling="cls_last", dapt_anchor_weight=0,
            outcome_sorted_event_times={"x": torch.tensor([10.0]), "y": torch.tensor([10.0])},
            cohort_names=["C1", "C2"],
            competing_risk_config={"loss_weight": 1.0, "contrastive_loss_weight": 0.0,
                                   "objective": "horizon_bce", "head_mode": "linear",
                                   "readout_mode": readout, "interval_boundaries_days": [30, 365],
                                   "no_competing_outcomes": []})
    model = build(mode)
    batch = {"input_emb": torch.randn(4, 5, 16), "attention_mask": torch.ones(4, 5, dtype=torch.bool),
             "cohort_index": torch.tensor([0, 1, 0, 1]), "cohort_fine_index": torch.tensor([0, 1, 0, 1]),
             "subject_id": torch.arange(4)}
    survival = {o: {"times": torch.tensor([10.0, 400.0, 400.0, 20.0]), "events": torch.tensor([1, 0, 0, 1])}
                for o in ("x", "y")}
    representation = model.get_objective_representations(batch)
    assert representation.shape == (4, 2, 16)
    # Zero-initialised gates: every outcome starts from the shared summary vector.
    shared = model._pool(batch)
    torch.testing.assert_close(representation, shared[:, None].expand(-1, 2, -1))
    model.train()
    loss = model(batch, survival)["loss"]
    assert torch.isfinite(loss)
    loss.backward()
    gates = [p for n, p in model.named_parameters() if "residual_gates" in n]
    assert gates and float(gates[0].grad.abs().sum()) > 0  # the attention path can switch on
    model.eval()
    with torch.no_grad():
        hazards = model.competing_risk_log_hazards(representation, batch["cohort_fine_index"])
    assert hazards.shape[:2] == (4, 2) and torch.isfinite(hazards).all()


def test_contrasts_reach_the_numbers():
    from opera.run.analyse_locked_predictions import TASK_POOLING
    from opera.run.build_paper_numbers import CODES

    assert set(TASK_POOLING) <= set(CODES)


@pytest.mark.parametrize("variant", ["fine", "own", "own_groups", "fine_own", "fine_groups"])
def test_diagnosis_conditioning_and_own_corrections_start_at_the_attention_model(variant):
    from opera.functional.direct_cr_contract import needs_fine_index

    readout = "outcome_attention" if variant in ("own", "own_groups") else "factorized_cohort_outcome_prototypes"
    settings = {"loss_weight": 1.0, "contrastive_loss_weight": 0.0, "objective": "horizon_bce",
                "head_mode": "linear", "readout_mode": readout, "interval_boundaries_days": [30, 365],
                "no_competing_outcomes": [], "fine_names": ["A", "B", "C"]}
    if "fine" in variant:
        settings["prototype_condition"] = "fine"
    if "own" in variant:
        settings.update(attention_own_corrections=True, attention_own_rank=4, adapter_residual_scale=0.7)
    if variant == "own_groups":
        settings["attention_own_disease_groups"] = {"A": "g1", "B": "g1", "C": "g2"}
    if variant == "fine_groups":
        settings["prototype_condition_groups"] = {"A": "g1", "B": "g1", "C": "g2"}
    assert needs_fine_index(settings)
    torch.manual_seed(0)
    model = OperaContrastiveModel(
        encoder=_Encoder(), outcome_names=["x", "y"], hidden_size=16, projection_hidden_dim=16,
        projection_dim=8, pooling="cls_last", dapt_anchor_weight=0,
        outcome_sorted_event_times={"x": torch.tensor([10.0]), "y": torch.tensor([10.0])},
        cohort_names=["C1", "C2"], competing_risk_config=settings)
    batch = {"input_emb": torch.randn(5, 6, 16), "attention_mask": torch.ones(5, 6, dtype=torch.bool),
             "cohort_index": torch.tensor([0, 1, 0, 1, 0]), "cohort_fine_index": torch.tensor([0, 1, 2, 1, -1]),
             "subject_id": torch.arange(5)}
    survival = {o: {"times": torch.tensor([10.0, 400.0, 400.0, 20.0, 50.0]),
                    "events": torch.tensor([1, 0, 0, 1, 0])} for o in ("x", "y")}
    representation = model.get_objective_representations(batch)
    shared = model._pool(batch)
    # Attention gates start at zero; own corrections start near zero (small up matrices).
    torch.testing.assert_close(representation, shared[:, None].expand(-1, 2, -1), atol=1e-2, rtol=0)
    model.train()
    loss = model(batch, survival)["loss"]
    loss.backward()
    assert torch.isfinite(loss)
    if "own" in variant:
        own = model.attention_own_corrections
        assert float(own.disease_up.grad.abs().sum()) > 0 and float(own.outcome_up.grad.abs().sum()) > 0
        # An unknown diagnosis (-1) gets the global correction only.
        terms = own.components(representation.detach(), batch["cohort_fine_index"])
        assert float(terms["own_disease"][4].abs().max()) == 0
        if variant == "own_groups":
            assert "group" in terms and float(own.group_up.grad.abs().sum()) > 0
    if "fine" in variant:
        readout = model.outcome_attention_readout
        assert readout.cohort_names == ["A", "B", "C"] and readout.allow_unknown
    if variant == "fine_groups":
        readout = model.outcome_attention_readout
        assert readout.cohort_group_index.tolist() == [0, 0, 1]
        assert float(readout.group_factors.grad.abs().sum()) >= 0  # in the graph
        with torch.no_grad():  # A and B share their lineage's terms
            readout.group_mixture_logits.normal_(); readout.cohort_mixture_logits.zero_()
            readout.cohort_factors.zero_()
        w = readout.mixture_weights(torch.tensor([0, 1, 2]))
        torch.testing.assert_close(w[0], w[1])
        assert not torch.allclose(w[0], w[2])


def test_shared_attention_runs_keep_the_old_cohort_conditioning():
    from opera.functional.direct_cr_contract import needs_fine_index

    assert not needs_fine_index({"readout_mode": "factorized_cohort_outcome_prototypes"})
    assert not needs_fine_index({"readout_mode": "shared"})
    assert needs_fine_index({"readout_mode": "source_anchored_adapters"})


def test_stage_nine_is_claimed_first_and_in_its_order():
    _, after = _bundles()
    stage9 = sorted((e for e in after if e.get("queue_order", 0) < 0), key=lambda e: e["queue_order"])
    assert [e["name"].split("fix_")[1] for e in stage9][:2] == ["J_OA_OWN", "J_OA_LR_OWN"]
    assert max(e["queue_order"] for e in stage9) < min(e["queue_order"] for e in after if e.get("queue_order", 0) > 0)


def test_tied_outcome_queries_share_a_basis_and_start_at_the_joint_model():
    from opera.modules.networks.opera_nets import OutcomeAttentionReadout

    readout = OutcomeAttentionReadout(16, ["a", "b", "c", "d"], num_heads=4, query_rank=2)
    queries = readout.outcome_queries()
    assert queries.shape == (4, 16)
    assert torch.linalg.matrix_rank(queries.detach()) <= 2  # own deviations start at zero
    hidden, shared = torch.randn(3, 5, 16), torch.randn(3, 16)
    out = readout(hidden, shared, torch.ones(3, 5, dtype=torch.bool))
    torch.testing.assert_close(out, shared[:, None].expand(-1, 4, -1))  # zero gates
    with torch.no_grad():
        readout.residual_gates.fill_(1.0)
    readout(hidden, shared, torch.ones(3, 5, dtype=torch.bool)).sum().backward()
    assert float(readout.query_basis.grad.abs().sum()) > 0 and float(readout.query_coefficients.grad.abs().sum()) > 0


@pytest.mark.parametrize("groups", [False, True])
def test_own_corrections_survive_device_and_dtype_moves_and_reloading(groups):
    """Regression (1 Nov 2026): a helper named ``_apply`` shadowed nn.Module._apply,
    so every own-correction fit failed when Lightning moved the model to the GPU."""
    from opera.modules.networks.attention_own_corrections import AttentionOwnCorrections

    module = AttentionOwnCorrections(16, ["A", "B", "C"], ["x", "y"], rank=4,
                                     disease_groups={"A": "g", "B": "g", "C": "h"} if groups else None)
    module.to("cpu"); module.cpu(); module.double(); module.float()
    reloaded = AttentionOwnCorrections(16, ["A", "B", "C"], ["x", "y"], rank=4,
                                       disease_groups={"A": "g", "B": "g", "C": "h"} if groups else None)
    reloaded.load_state_dict(module.state_dict())
    rep, fine = torch.randn(3, 2, 16), torch.tensor([0, 2, -1])
    torch.testing.assert_close(reloaded(rep, fine), module(rep, fine))


@pytest.mark.parametrize("alias", ["J_OA_OWN", "J_OA_LR_OWN", "J_OA_OWN_PH", "J_FPF_OWN", "J_FPF_PH_OWN_PH",
                                   "J_OA", "J_FPF_PH"])
def test_every_stage_nine_model_moves_between_devices_and_dtypes(alias):
    """Every stage-9 config builds a model that survives .to/.cpu/.double, as
    Lightning does at the start and end of every fit."""
    _, after = _bundles()
    config = next(e["config"] for e in after if e["name"].endswith(f"fix_{alias}"))
    risk = dict(config["competing_risk"])
    names = list(risk.get("fine_names", [])) or ["A"]
    torch.manual_seed(0)
    model = OperaContrastiveModel(
        encoder=_Encoder(), outcome_names=list(config["outcomes"]), hidden_size=16, projection_hidden_dim=16,
        projection_dim=8, pooling="cls_last", dapt_anchor_weight=0,
        outcome_sorted_event_times={o: torch.tensor([10.0]) for o in config["outcomes"]},
        cohort_names=sorted(config["cohorts"]), competing_risk_config=dict(risk, fine_names=names))
    model.to("cpu"); model.double(); model.float(); model.cpu()
    batch = {"input_emb": torch.randn(3, 4, 16), "attention_mask": torch.ones(3, 4, dtype=torch.bool),
             "cohort_index": torch.tensor([0, 1, 0]) % len(config["cohorts"]),
             "cohort_fine_index": torch.tensor([0, len(names) - 1, -1]), "subject_id": torch.arange(3)}
    assert torch.isfinite(model.get_objective_representations(batch)).all()


@pytest.mark.parametrize("encoder_lr", [None, 1.0])
def test_national_arm_uses_a_stage_nine_primary_readout(encoder_lr):
    """Lock closure: with a stage-9 primary, the national R fits carry its readout
    on the national encoder, and optionally the winning encoder rate."""
    from opera.functional.direct_cr_contract import validate_direct_cr_config
    from opera.run.paper_prediction_queue import canonical_models
    from opera.run.prepare_paper_run_bundle import build_bundle, task_pooling_design

    assert task_pooling_design("J_OA_LR_OWN_L") == "J_OA_LR_OWN" and task_pooling_design("R_AB_L") is None
    entries = build_bundle(base_config(), reference(), {"L": "/L.ckpt", "E": "/E.ckpt", "EA": "/EA.ckpt"},
                           seeds=(42,), selected_router=LOCK, routed_model="J_OA_LR_OWN_L",
                           external_encoder_lr=encoder_lr)
    external = {e["name"]: e for e in entries if e["phase"] == "external"}
    assert set(external) == {"seed42/E/J", "seed42/E/R", "seed42/EA/J", "seed42/EA/R"}
    local = next(e["config"] for e in _bundles()[1] if e["name"].endswith("fix_J_OA_LR_OWN"))
    for origin in ("E", "EA"):
        routed = external[f"seed42/{origin}/R"]["config"]
        risk = routed["competing_risk"]
        assert risk["readout_mode"] == "outcome_attention" and risk["outcome_attention_query_rank"] == 8
        assert risk["attention_own_corrections"] is True and routed["dapt_ckpt"] == f"/{origin}.ckpt"
        # The same readout settings as the local primary.
        keys = {k for k in local["competing_risk"] if k.startswith(("attention_", "outcome_attention", "readout"))}
        assert {k: risk[k] for k in keys} == {k: local["competing_risk"][k] for k in keys}
        assert canonical_models(external[f"seed42/{origin}/R"], routed, {"selected_router": LOCK}) == [f"R_{origin}"]
        for mode in ("J", "R"):
            training = external[f"seed42/{origin}/{mode}"]["config"]["training"]
            assert training.get("encoder_lr_multiplier") == encoder_lr if encoder_lr else \
                "encoder_lr_multiplier" not in training or training["encoder_lr_multiplier"] != 1.0
        validate_direct_cr_config(routed)


def test_router_only_analyses_refuse_a_stage_nine_primary(tmp_path, monkeypatch):
    import argparse
    from opera.run import paper_experiments

    env = {"OPERA_PAPER_ROUTED_MODEL": "J_OA_LR_OWN_L", "OPERA_PAPER_ROOT": str(tmp_path)}
    args = argparse.Namespace(action="source-removal", split="tuning", execute=False, phase=None, stage=9)
    with pytest.raises(ValueError, match="has no sources"):
        paper_experiments.build_commands(args, env)


@pytest.mark.parametrize("alias", ["R_E", "R_EA", "R_L", "J_OA_OWN_L"])
@pytest.mark.parametrize("explicit_routes", [False, True])
def test_reading_aliases_never_request_nonexistent_routes(alias, explicit_routes):
    from argparse import Namespace
    from opera.run.paper_prediction_queue import export_flags

    args = Namespace(export_routes=explicit_routes)
    flags = export_flags([alias], args, {"competing_risk": {"readout_mode": "outcome_attention"}})
    assert flags == {"export_embeddings": True, "export_routes": False}
    routed = export_flags(["R_AB_L"], args,
                          {"competing_risk": {"readout_mode": "source_anchored_adapters"}})
    assert routed == {"export_embeddings": True, "export_routes": True}
