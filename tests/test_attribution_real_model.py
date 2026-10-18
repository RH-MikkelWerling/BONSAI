"""Integrated gradients through the real OPERA model classes.

The routed model (source-anchored adapters with an interacting query) is the
candidate paper model, so every property the attribution analysis relies on is
checked on it, not only on a stand-in.
"""

import pytest
import torch

from opera.interpretability.token_attribution import (
    attributable_mask, baseline_tensor, combined_target, event_ids,
    gradients_through_encoder, integrated_gradients, mean_content,
    reference_embeddings, replaced_target, risk_target, split_embeddings)
from test_token_attribution import FIXED, OUTCOMES, SAMPLES, TinyModel, collate


def routed_model(*, freeze=False, seed=5):
    from opera.modules.networks.opera_nets import OperaContrastiveModel

    torch.manual_seed(seed)
    encoder = TinyModel(router=False).encoder
    grids = {o: torch.tensor([30.0, 180.0, 365.0]) for o in OUTCOMES}
    probabilities = {o: torch.tensor([0.3, 0.3, 0.4]) for o in OUTCOMES}
    config = {
        "loss_weight": 1.0, "contrastive_loss_weight": 0.0,
        "interval_boundaries_days": [180.0, 365.0, 730.0],
        "no_competing_outcomes": ["overall_survival"],
        "readout_mode": "source_anchored_adapters", "fine_names": ["A", "B"],
        "adapter_query_mode": "interacting", "adapter_query_hidden_dim": 16,
        "adapter_residual_scale": 0.7, "adapter_home_floor": 0.05,
        "adapter_shared_logit_init": 0.0,
    }
    model = OperaContrastiveModel(
        encoder, OUTCOMES, hidden_size=16, projection_hidden_dim=16, projection_dim=8,
        outcome_sorted_event_times=grids, outcome_event_time_probs=probabilities,
        competing_risk_config=config, dapt_anchor_weight=0.0, pooling="cls_last",
        freeze_encoder=freeze,
    ).eval()
    readout = model.source_adapter_readout
    with torch.no_grad():
        # Trained routers are far from initialisation: make routes input-dependent.
        readout.up.normal_(std=0.5)
        readout.down.normal_(std=0.5)
        readout.source_keys.normal_(std=2.0)
        readout.patient_query.weight.normal_(std=1.0)
        readout.query_interaction[-1].weight.normal_(std=0.5)
    return model


def prepared(model, samples=SAMPLES):
    batch = collate(samples)
    batch["cohort_fine_index"] = torch.tensor([0, 1, 0][: len(samples)])
    mask = attributable_mask(batch, FIXED)
    full, content = split_embeddings(model.encoder, batch)
    reference = mean_content(model.encoder, [batch], FIXED)
    return batch, mask, full, content, reference, baseline_tensor(content, mask, reference)


COLUMNS = [("risk", "overall_survival", 365.0), ("logit", "overall_survival", 365.0),
           ("logit", "infection", 365.0), ("borrowed", "infection", None)]


def test_routed_model_reads_injected_embeddings_exactly_as_its_own_input():
    model = routed_model()
    batch, mask, full, *_ = prepared(model)
    target = combined_target(model, COLUMNS)
    with torch.no_grad():
        torch.testing.assert_close(target(batch), target({**batch, "token_embeddings": full}))


def test_routed_attributions_are_complete_for_risk_log_odds_and_router_weights():
    model = routed_model()
    batch, mask, full, content, _, baseline = prepared(model)
    attributions, f_input, f_base = integrated_gradients(
        combined_target(model, COLUMNS), batch, full, content, baseline, mask, steps=128)
    difference = f_input - f_base
    gap = (attributions.sum(-1) - difference).abs() / difference.abs().clamp_min(1e-3)
    assert gap.max() < 0.05
    # The router weights depend on the patient, so they get event credit too.
    assert attributions[:, 3].abs().sum() > 0
    assert torch.allclose(torch.logit(f_input[:, 0]), f_input[:, 1], atol=1e-5)


def test_routing_changes_what_the_prediction_rests_on():
    model = routed_model()
    batch, mask, full, content, _, baseline = prepared(model)
    routed, *_ = integrated_gradients(risk_target(model, OUTCOMES, 365), batch, full,
                                      content, baseline, mask, steps=32)
    sources = [*("disease:" + n for n in model.source_adapter_readout.fine_names),
               *("outcome:" + n for n in model.source_adapter_readout.outcome_names)]
    shared, *_ = integrated_gradients(
        risk_target(model, OUTCOMES, 365, removed_sources=sources), batch, full, content,
        baseline, mask, steps=32)
    assert not torch.allclose(routed, shared, atol=1e-6)


def test_timeline_only_evaluation_input_is_the_attribution_reference():
    model = routed_model()
    batch, mask, full, content, reference, baseline = prepared(model)
    target = combined_target(model, COLUMNS)
    _, _, f_base = integrated_gradients(target, batch, full, content, baseline, mask, steps=2)
    timeline = reference_embeddings(model.encoder, batch, reference, FIXED)
    with torch.no_grad():
        torch.testing.assert_close(target({**batch, "token_embeddings": timeline}), f_base)


def test_frozen_real_model_needs_the_gradient_switch_and_stays_frozen():
    model = routed_model(freeze=True)
    batch, mask, full, content, _, baseline = prepared(model)
    target = risk_target(model, OUTCOMES, 365)
    with pytest.raises(RuntimeError):
        integrated_gradients(target, batch, full, content, baseline, mask, steps=2)
    with gradients_through_encoder(model):
        attributions, f_input, f_base = integrated_gradients(
            target, batch, full, content, baseline, mask, steps=64)
    assert model.freeze_encoder
    assert attributions.abs().sum() > 0
    assert (attributions.sum(-1) - (f_input - f_base)).abs().max() < 1e-3


def test_padded_patients_get_the_attributions_they_get_alone():
    model = routed_model()
    batch, mask, full, content, reference, baseline = prepared(model)
    together, *_ = integrated_gradients(risk_target(model, OUTCOMES, 365), batch, full,
                                        content, baseline, mask, steps=8, chunk=3)
    for row, sample in enumerate(SAMPLES):
        alone_batch = collate([sample])
        alone_batch["cohort_fine_index"] = batch["cohort_fine_index"][row:row + 1]
        alone_mask = attributable_mask(alone_batch, FIXED)
        alone_full, alone_content = split_embeddings(model.encoder, alone_batch)
        alone, *_ = integrated_gradients(
            risk_target(model, OUTCOMES, 365), alone_batch, alone_full, alone_content,
            baseline_tensor(alone_content, alone_mask, reference), alone_mask, steps=8)
        width = alone.shape[-1]
        torch.testing.assert_close(together[row, :, :width], alone[0], atol=1e-6, rtol=1e-4)
        assert torch.all(together[row, :, width:] == 0)


def test_removal_checks_measure_exactly_the_attribution_when_the_model_is_linear():
    batch = collate(SAMPLES)
    mask = attributable_mask(batch, FIXED)
    torch.manual_seed(0)
    full = torch.randn(*batch["code"].shape, 4)
    content = 0.5 * full
    baseline = baseline_tensor(content, mask, torch.zeros(4))
    weight = torch.randn(batch["code"].shape[1], 4)

    def linear(local):
        return (local["token_embeddings"] * weight).sum((1, 2))[:, None]

    attributions, f_input, f_base = integrated_gradients(linear, batch, full, content,
                                                         baseline, mask, steps=1)
    events = event_ids(batch, mask)
    for row in range(len(SAMPLES)):
        for event in range(int(events[row].max()) + 1):
            chosen = (events == event) & (torch.arange(len(SAMPLES))[:, None] == row)
            after = replaced_target(linear, batch, full, content, baseline, chosen[None])
            drop = f_input[row, 0] - after[0, row, 0]
            credit = attributions[row, 0][events[row] == event].sum()
            assert torch.allclose(drop, credit, atol=1e-5)


def test_explainer_runs_every_column_on_the_routed_model():
    from opera.modules.datamodules.ContrastiveDataModule import contrastive_collate
    from opera.run.explain_direct_competing_risk import Explainer, column_name
    from test_token_attribution import _engine_sample

    model = routed_model()
    explainer = Explainer(model, {"[PAD]": 0, "[CLS]": 1, "[SEP]": 2, "[UNK]": 3,
                                  "[MASK]": 4, "BG_SEX//F": 5,
                                  **{f"CODE{i}": 6 + i for i in range(10)}},
                          collate=contrastive_collate, columns=COLUMNS,
                          check_columns=[COLUMNS[0]], steps=32, ks=(1,), random_repeats=2)
    explainer.fit_baseline([prepared(model)[0]])
    sample = _engine_sample(SAMPLES[0]["code"].tolist(), SAMPLES[0]["abspos"].tolist(), 1, 7)
    rows = explainer.explain(sample, meta={"subject_id": 7, "weight": 1.0})
    names = {r["column"] for r in rows["patients"]}
    assert names == {column_name(*c) for c in COLUMNS if c[1] == "overall_survival"}
    assert all(abs(r["completeness_gap"]) <= 0.05 for r in rows["patients"])
    assert {r["perturbation"] for r in rows["faithfulness"]} == {"replace", "delete"}


def test_fast_integration_matches_a_fine_reference_on_the_routed_model():
    model = routed_model()
    batch, mask, full, content, _, baseline = prepared(model)
    columns = [("logit", o, 365.0) for o in OUTCOMES] + [("risk", o, 365.0) for o in OUTCOMES]
    target = combined_target(model, columns)
    reference, f_input, f_base = integrated_gradients(
        target, batch, full, content, baseline, mask, steps=256, rule="midpoint",
        batched=False)
    fast, *_ = integrated_gradients(target, batch, full, content, baseline, mask, steps=8)
    looped, *_ = integrated_gradients(target, batch, full, content, baseline, mask, steps=8,
                                      batched=False)
    torch.testing.assert_close(fast, looped, atol=1e-7, rtol=1e-5)  # one backward = many
    scale = reference.abs().max()
    assert (fast - reference).abs().max() / scale < 1e-4  # 8 Gauss nodes vs 256 midpoints
    gap = (fast.sum(-1) - (f_input - f_base)).abs() / (f_input - f_base).abs().clamp_min(1e-3)
    assert gap.max() < 0.01
