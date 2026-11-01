"""Stage 12 (1 November 2026): seed repeat, capacity and ablations of the
leading reading design; each changes exactly what it names."""

import pytest
import torch

from opera.modules.networks.attention_own_corrections import AttentionOwnCorrections
from test_paper_run_bundle import base_config, reference
from test_task_pooling_stage9 import LOCK

ARGS = dict(selected_router=LOCK, router_fix=True, router_own=True, sampling_ipw=True, router_pp=True,
            router_pt=True, router_pf=True, task_pooling=True, encoder_lr=True, encoder_lr_reading=True)


def _bundles():
    from opera.run.prepare_paper_run_bundle import build_bundle

    before = build_bundle(base_config(), reference(), {"L": "/x/best.ckpt"}, **ARGS)
    after = build_bundle(base_config(), reference(), {"L": "/x/best.ckpt"}, **ARGS, task_ablation=True)
    return before, after


def test_stage_twelve_changes_exactly_what_each_fit_names():
    from opera.functional.direct_cr_contract import validate_direct_cr_config
    from opera.run.paper_prediction_queue import canonical_models

    before, after = _bundles()
    assert [e["config"] for e in after[: len(before)]] == [e["config"] for e in before]
    new = {canonical_models(e, e["config"], {"selected_router": LOCK})[0]: e for e in after[len(before):]}
    assert set(new) == {"J_OA_OWN_s43_L", "J_OA_OWN_R32_L", "J_OA_OWN_NODIS_L", "J_OA_OWN_NOOUT_L",
                        "J_OA_OWN_F10_L"}
    source = {e["name"].split("fix_")[1]: e["config"] for e in before if "fix_J_OA" in e["name"]}
    expected = {
        "J_OA_OWN_R32_L": ("J_OA_OWN", {"competing_risk.attention_own_rank"}),
        "J_OA_OWN_NODIS_L": ("J_OA_OWN", {"competing_risk.attention_own_disease"}),
        "J_OA_OWN_NOOUT_L": ("J_OA_OWN", {"competing_risk.attention_own_outcome"}),
        "J_OA_OWN_F10_L": ("J_OA_OWN", {"training.train_patient_fraction", "training.train_fraction_seed"}),
    }
    for alias, entry in new.items():
        validate_direct_cr_config(entry["config"])
        assert entry["queue_order"] < min(e.get("queue_order", 0) for e in before)
        if alias.endswith("_s43_L"):
            assert entry["seed"] == 43
            continue
        design, changed = expected[alias]
        a, b = entry["config"], source[design]
        diff = {f"{part}.{k}" for part in ("competing_risk", "training")
                for k in {**a[part], **b[part]} if a[part].get(k) != b[part].get(k)}
        assert diff == changed, (alias, diff)
    assert new["J_OA_OWN_R32_L"]["config"]["competing_risk"]["attention_own_rank"] == 32
    assert new["J_OA_OWN_F10_L"]["config"]["training"]["train_patient_fraction"] == 0.1


@pytest.mark.parametrize("use_disease,use_outcome", [(False, True), (True, False)])
def test_ablations_drop_exactly_one_term_and_freeze_it(use_disease, use_outcome):
    m = AttentionOwnCorrections(16, ["A", "B"], ["x", "y"], rank=4, use_disease=use_disease, use_outcome=use_outcome)
    m.to("cpu"); m.double(); m.float()
    with torch.no_grad():
        m.disease_up.normal_(); m.outcome_up.normal_(); m.global_up.normal_()
    terms = m.components(torch.randn(3, 2, 16), torch.tensor([0, 1, -1]))
    assert ("own_disease" in terms) == use_disease and ("own_outcome" in terms) == use_outcome
    assert m.disease_up.requires_grad == use_disease and m.outcome_up.requires_grad == use_outcome


def test_contrasts_reach_the_numbers():
    from opera.run.analyse_locked_predictions import TASK_ABLATION
    from opera.run.build_paper_numbers import CODES

    assert set(TASK_ABLATION) <= set(CODES)


@pytest.mark.parametrize("alias", ["J_OA_OWN_s43", "J_OA_OWN_R32", "J_OA_OWN_NODIS", "J_OA_OWN_NOOUT",
                                   "J_OA_OWN_F10"])
def test_every_stage_twelve_model_builds_trains_and_moves(alias):
    from opera.modules.networks.opera_nets import OperaContrastiveModel
    from test_router_axis_separation import _Encoder

    config = next(e["config"] for e in _bundles()[1] if e["name"].endswith(f"fix_{alias}"))
    risk = dict(config["competing_risk"])
    outcomes = list(config["outcomes"])
    torch.manual_seed(0)
    model = OperaContrastiveModel(
        encoder=_Encoder(), outcome_names=outcomes, hidden_size=16, projection_hidden_dim=16, projection_dim=8,
        pooling="cls_last", dapt_anchor_weight=0, outcome_sorted_event_times={o: torch.tensor([10.0]) for o in outcomes},
        cohort_names=sorted(config["cohorts"]), competing_risk_config=risk)
    model.to("cpu"); model.double(); model.float()
    names = risk["fine_names"]
    batch = {"input_emb": torch.randn(4, 5, 16), "attention_mask": torch.ones(4, 5, dtype=torch.bool),
             "cohort_index": torch.tensor([0, 0, 0, 0]), "cohort_fine_index": torch.tensor([0, len(names) - 1, 0, -1]),
             "subject_id": torch.arange(4)}
    survival = {o: {"times": torch.tensor([10.0, 400.0, 20.0, 400.0]), "events": torch.tensor([1, 0, 1, 0])}
                for o in outcomes}
    model.train()
    loss = model(batch, survival)["loss"]
    loss.backward()
    assert torch.isfinite(loss)
    own = model.attention_own_corrections
    assert own.disease_down.shape[1] == risk["attention_own_rank"]
