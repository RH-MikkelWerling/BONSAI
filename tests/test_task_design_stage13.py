"""Stage 13 (1 November 2026): support shrinkage and diagnosis x outcome
corrections for the leading reading design."""

from types import SimpleNamespace

import pytest
import torch

from opera.modules.networks.attention_own_corrections import AttentionOwnCorrections
from test_paper_run_bundle import base_config, reference
from test_task_ablation_stage12 import ARGS
from test_task_pooling_stage9 import LOCK


def _module(**options):
    torch.manual_seed(0)
    m = AttentionOwnCorrections(16, ["A", "B"], ["x", "y", "z"], rank=4, **options)
    with torch.no_grad():
        m.disease_up.normal_(); m.outcome_up.normal_(); m.global_up.normal_()
    return m


def test_shrinkage_scales_own_corrections_by_cell_support_and_leaves_global():
    plain, shrunk = _module(), _module(shrink_tau=20.0)
    events = torch.tensor([[0.0, 20.0, 1000.0], [5.0, 5.0, 5.0]])
    shrunk.set_cell_support(events); plain.set_cell_support(events)
    rep, fine = torch.randn(4, 3, 16), torch.tensor([0, 0, 1, -1])
    a, b = plain.components(rep, fine), shrunk.components(rep, fine)
    w = (events / (events + 20.0))[fine.clamp(0)]
    for term in ("own_disease", "own_outcome"):
        torch.testing.assert_close(b[term][:3], a[term][:3] * w[:3, :, None])
    assert float(b["own_disease"][0, 0].abs().max()) == 0  # a cell without events gets no own correction
    torch.testing.assert_close(b["global"], a["global"])
    reloaded = _module(shrink_tau=20.0); reloaded.load_state_dict(shrunk.state_dict())
    torch.testing.assert_close(reloaded.cell_train_events, events)  # support is saved with the checkpoint


def test_cross_correction_starts_shared_and_can_become_outcome_specific():
    plain, cross = _module(), _module(cross=True)
    rep = torch.randn(1, 1, 16).expand(2, 3, 16).contiguous()  # same representation for every outcome
    fine = torch.tensor([0, 1])
    torch.testing.assert_close(cross.components(rep, fine)["own_disease"], plain.components(rep, fine)["own_disease"])
    cross.components(rep, fine)["own_disease"].sum().backward()
    assert float(cross.outcome_bottleneck_scale.grad.abs().sum()) > 0
    with torch.no_grad():
        cross.outcome_bottleneck_scale.normal_()
    d = cross.components(rep, fine)["own_disease"]
    assert not torch.allclose(d[:, 0], d[:, 1])  # the diagnosis correction now differs by outcome


def test_training_support_counts_events_per_diagnosis_and_outcome(tmp_path):
    from opera.run.contrastive_multicohort import _set_attention_own_support

    own = _module(shrink_tau=10.0)
    dataset = SimpleNamespace(fine_index_by_subject={1: 0, 2: 0, 3: 1, 4: -1}, outcome_dicts={
        "x": {1: {"event": 1, "time_days": 5}, 2: {"event": 0, "time_days": 9}, 3: {"event": 1, "time_days": 2},
              4: {"event": 1, "time_days": 1}},
        "z": {1: {"event": 1, "time_days": 3}, 2: {"event": 1, "time_days": 3}, 3: {"event": -1, "time_days": 3}}})
    _set_attention_own_support(SimpleNamespace(attention_own_corrections=own), dataset, str(tmp_path / "s.csv"))
    torch.testing.assert_close(own.cell_train_events, torch.tensor([[1.0, 0.0, 2.0], [1.0, 0.0, 0.0]]))
    assert (tmp_path / "s.csv").read_text().count("\n") == 7


def _bundles():
    from opera.run.prepare_paper_run_bundle import build_bundle

    before = build_bundle(base_config(), reference(), {"L": "/x/best.ckpt"}, **ARGS, task_ablation=True)
    after = build_bundle(base_config(), reference(), {"L": "/x/best.ckpt"}, **ARGS, task_ablation=True,
                         task_design=True)
    return before, after


def test_stage_thirteen_changes_exactly_what_each_fit_names():
    from opera.functional.direct_cr_contract import validate_direct_cr_config
    from opera.run.paper_prediction_queue import canonical_models

    before, after = _bundles()
    assert [e["config"] for e in after[: len(before)]] == [e["config"] for e in before]
    new = {canonical_models(e, e["config"], {"selected_router": LOCK})[0]: e for e in after[len(before):]}
    base = next(e["config"] for e in before if e["name"].endswith("fix_J_OA_OWN"))
    expected = {"J_OA_OWN_SW20_L": {"attention_own_shrink_tau"}, "J_OA_OWN_SW100_L": {"attention_own_shrink_tau"},
                "J_OA_OWN_X_L": {"attention_own_cross"},
                "J_OA_OWN_X_SW_L": {"attention_own_cross", "attention_own_shrink_tau"},
                "J_OA_OWN_PE_L": {"attention_own_outcome_groups"},
                "J_OA_OWN_PH_X_L": {"attention_own_cross", "attention_own_disease_groups"},
                "J_OA_OWN_PE_X_SW_L": {"attention_own_outcome_groups", "attention_own_cross",
                                       "attention_own_shrink_tau"},
                "J_OA_OWN_MAX_L": {"attention_own_outcome_groups", "attention_own_disease_groups",
                                   "attention_own_cross", "attention_own_shrink_tau", "attention_own_rank"}}
    assert set(new) == set(expected)
    for alias, entry in new.items():
        risk, ref = entry["config"]["competing_risk"], base["competing_risk"]
        assert {k for k in {**risk, **ref} if risk.get(k) != ref.get(k)} == expected[alias], alias
        # Before every earlier fit except the stage-12 seed repeat, which runs first.
        assert entry["queue_order"] < min(e.get("queue_order", 0) for e in before
                                          if not e["name"].endswith("_s43"))
        validate_direct_cr_config(entry["config"])
    first = min(after, key=lambda e: e.get("queue_order", 0))
    assert first["name"].endswith("_s43")


@pytest.mark.parametrize("alias", ["J_OA_OWN_SW20", "J_OA_OWN_X", "J_OA_OWN_X_SW", "J_OA_OWN_PE", "J_OA_OWN_PH_X",
                                   "J_OA_OWN_PE_X_SW", "J_OA_OWN_MAX"])
def test_every_stage_thirteen_model_trains_and_moves(alias):
    from opera.modules.networks.opera_nets import OperaContrastiveModel
    from test_router_axis_separation import _Encoder

    config = next(e["config"] for e in _bundles()[1] if e["name"].endswith(f"fix_{alias}"))
    risk, outcomes = dict(config["competing_risk"]), list(config["outcomes"])
    torch.manual_seed(0)
    model = OperaContrastiveModel(
        encoder=_Encoder(), outcome_names=outcomes, hidden_size=16, projection_hidden_dim=16, projection_dim=8,
        pooling="cls_last", dapt_anchor_weight=0, outcome_sorted_event_times={o: torch.tensor([10.0]) for o in outcomes},
        cohort_names=sorted(config["cohorts"]), competing_risk_config=risk)
    model.attention_own_corrections.set_cell_support(torch.full((len(risk["fine_names"]), len(outcomes)), 30.0))
    model.to("cpu"); model.double(); model.float()
    batch = {"input_emb": torch.randn(4, 5, 16), "attention_mask": torch.ones(4, 5, dtype=torch.bool),
             "cohort_index": torch.zeros(4, dtype=torch.long),
             "cohort_fine_index": torch.tensor([0, len(risk["fine_names"]) - 1, 0, -1]), "subject_id": torch.arange(4)}
    survival = {o: {"times": torch.tensor([10.0, 400.0, 20.0, 400.0]), "events": torch.tensor([1, 0, 1, 0])}
                for o in outcomes}
    model.train(); loss = model(batch, survival)["loss"]; loss.backward()
    assert torch.isfinite(loss)


def test_contrasts_reach_the_numbers():
    from opera.run.analyse_locked_predictions import TASK_DESIGN
    from opera.run.build_paper_numbers import CODES

    assert set(TASK_DESIGN) <= set(CODES)


def test_endpoint_family_corrections_are_shared_within_a_family():
    m = AttentionOwnCorrections(16, ["A", "B"], ["x", "y", "z"], rank=4,
                                outcome_groups={"x": "liver", "y": "liver", "z": "infection"})
    m.to("cpu"); m.double(); m.float()
    with torch.no_grad():
        m.outcome_group_up.normal_()
    rep = torch.randn(1, 1, 16).expand(2, 3, 16).contiguous()  # same representation for every outcome
    family = m.components(rep, torch.tensor([0, -1]))["outcome_group"]
    torch.testing.assert_close(family[0, 0], family[0, 1])      # x and y share the liver correction
    assert not torch.allclose(family[0, 0], family[0, 2])
    assert float(family[1].abs().max()) == 0                    # unknown diagnosis: no own-side terms
    family.sum().backward()
    assert float(m.outcome_group_up.grad.abs().sum()) > 0
    with pytest.raises(ValueError, match="No endpoint family"):
        AttentionOwnCorrections(16, ["A"], ["x", "y"], rank=4, outcome_groups={"x": "liver"})


def test_checkpoints_from_before_the_support_buffer_still_load():
    """Stage-9 fits (J_OA_OWN among them) were saved without cell_train_events;
    resuming and evaluating them must load strictly and leave the outputs unchanged."""
    old = _module()
    state = {k: v for k, v in old.state_dict().items() if k != "cell_train_events"}
    new = AttentionOwnCorrections(16, ["A", "B"], ["x", "y", "z"], rank=4)
    new.load_state_dict(state)  # strict
    rep, fine = torch.randn(3, 3, 16), torch.tensor([0, 1, -1])
    torch.testing.assert_close(new(rep, fine), old(rep, fine))
    assert float(new.cell_train_events.abs().sum()) == 0
    wrapped = torch.nn.Sequential(AttentionOwnCorrections(16, ["A", "B"], ["x", "y", "z"], rank=4))
    wrapped.load_state_dict({f"0.{k}": v for k, v in state.items()})  # nested prefix, as in the model
