"""Every stage-8 fit, built from its bundle config through the real model code:
one training step, validation logging, optimizer groups and the audit's
checkpoint reconstruction (31 October 2026)."""

import pytest
import torch

from opera.diagnostics.router_mechanism import COMPONENTS, FrozenRouterHead
from opera.modules.networks.full_own_routed_readout import FullOwnTaskRoutedReadout
from opera.modules.networks.opera_nets import OperaContrastiveModel
from test_paper_run_bundle import base_config, reference
from test_router_axis_separation import _Encoder

LOCK = {"source_mode": "both", "rank": 8, "query_mode": "interacting", "query_width": 512}
H = 16


def _stage8():
    from opera.run.paper_prediction_queue import canonical_models
    from opera.run.prepare_paper_run_bundle import build_bundle

    args = dict(selected_router=LOCK, router_fix=True, router_own=True, sampling_ipw=True,
                router_pp=True, router_pt=True)
    before = build_bundle(base_config(), reference(), {"L": "/x/best.ckpt"}, **args)
    after = build_bundle(base_config(), reference(), {"L": "/x/best.ckpt"}, **args, router_pf=True)
    return {canonical_models(e, e["config"], {"selected_router": LOCK})[0]: e["config"]
            for e in after[len(before):]}


FITS = _stage8()


def _build(config):
    outcomes = list(config["outcomes"])
    torch.manual_seed(0)
    model = OperaContrastiveModel(
        encoder=_Encoder(), outcome_names=outcomes, hidden_size=H, projection_hidden_dim=H,
        projection_dim=8, pooling="cls_last", dapt_anchor_weight=0,
        outcome_sorted_event_times={o: torch.tensor([10.0]) for o in outcomes},
        competing_risk_config=config["competing_risk"])
    r = model.source_adapter_readout
    r.set_training_support([30] * r.n_sources, [10] * r.n_sources, [30] * r.n_sources)
    r.set_cell_support([[3 + 40 * (d + o) for o in range(r.n_outcome)] for d in range(r.n_disease)])
    return model, outcomes


def _batch(model, outcomes, n=6):
    r = model.source_adapter_readout
    fine = torch.arange(n) % r.n_disease
    batch = {"input_emb": torch.randn(n, 2, H), "attention_mask": torch.ones(n, 2, dtype=torch.bool),
             "cohort_fine_index": fine, "subject_id": torch.arange(n)}
    survival = {o: {"times": torch.tensor([10.0, 400.0] * (n // 2)), "events": torch.tensor([1, 0] * (n // 2))}
                for o in outcomes}
    return batch, survival


def test_stage_eight_has_every_design():
    assert len(FITS) == 25 and all(name.startswith(("R_PF", "R_PH", "R_PL")) for name in FITS)


@pytest.mark.parametrize("alias", sorted(FITS))
def test_fit_trains_logs_optimises_and_reconstructs(alias, tmp_path, monkeypatch):
    import opera.modules.lightningmodules.OperaContrastiveModule as lightning

    config = FITS[alias]
    risk = config["competing_risk"]
    model, outcomes = _build(config)
    r = model.source_adapter_readout
    assert isinstance(r, FullOwnTaskRoutedReadout)
    with torch.no_grad():  # trained-looking adapters so every path carries signal
        r.up.normal_(std=0.5)
        for name in ("group_up", "latent_up"):
            if getattr(r, name, None) is not None:
                getattr(r, name).normal_(std=0.5)

    # One training step after the warm-up (routing live), finite gradients.
    model.train(); r.router_frozen = False
    batch, survival = _batch(model, outcomes)
    loss = model(batch, survival)["loss"]
    assert torch.isfinite(loss)
    loss.backward()
    grads = {n: p.grad for n, p in model.named_parameters() if p.grad is not None}
    assert all(torch.isfinite(g).all() for g in grads.values()), alias
    borrows = (risk.get("adapter_borrow_values") != "off" or risk.get("adapter_disease_groups")
               or risk.get("adapter_latent_groups"))
    readout = {n.split("source_adapter_readout.")[1]: g for n, g in grads.items() if "source_adapter_readout." in n}
    assert float(readout["up"].abs().sum()) > 0  # own corrections train
    if risk.get("adapter_borrow_values") != "off":
        assert float(readout["source_keys"].abs().sum()) > 0, "routing receives no gradient"
        assert float(readout["pool_scale"].abs()) > 0
    if risk.get("adapter_disease_groups"):
        assert float(readout["group_up"].abs().sum()) > 0
    if risk.get("adapter_latent_groups"):
        assert float(readout["latent_keys"].abs().sum()) > 0
    if risk.get("adapter_patient_gate"):
        assert float(readout["patient_gate"].abs()) > 0

    # Validation logging: correction sizes, finite; borrowed zero only when nothing is borrowed.
    model.eval()
    with torch.no_grad():
        logs = model(batch, survival)
    for key in ("cr/patient_norm", "cr/correction_norm/own", "cr/correction_norm/borrowed",
                "cr/correction_norm/global", "cr/pool_tau", "cr/pool_scale"):
        assert torch.isfinite(logs[key]), key
    assert (float(logs["cr/correction_norm/borrowed"]) > 0) == bool(borrows), alias

    # Optimizer: every trainable parameter exactly once, router LR as configured.
    monkeypatch.setattr(lightning, "optimizer_with_warmup", lambda optimizer, *_: optimizer)
    training = config["training"]
    module = lightning.OperaContrastiveModule(
        model=model, outcome_names=outcomes, learning_rate=1e-4,
        adapter_optimizer_epsilon=training.get("adapter_optimizer_epsilon"),
        adapter_epsilon_scope=training.get("adapter_epsilon_scope", "matrices"),
        router_lr_multiplier=float(training.get("router_lr_multiplier", 1.0)))
    module._trainer = object()
    groups = module.configure_optimizers().param_groups
    seen = [id(p) for g in groups for p in g["params"]]
    assert sorted(seen) == sorted(id(p) for p in model.parameters() if p.requires_grad)
    lr = {id(p): g["lr"] for g in groups for p in g["params"]}
    assert lr[id(r.source_keys)] == pytest.approx(1e-4 * float(training.get("router_lr_multiplier", 1.0)))
    assert lr[id(r.down)] == pytest.approx(1e-4)

    # The audit rebuilds the readout from a checkpoint and reproduces the model's hazards.
    path = tmp_path / "best.ckpt"
    torch.save({"hyper_parameters": {"model_init_config": {"competing_risk_config": dict(
                    risk, fine_names=list(r.fine_names))}, "outcome_names": outcomes},
                "state_dict": {f"model.{k}": v for k, v in model.state_dict().items()}}, path)
    head = FrozenRouterHead.from_checkpoint(path)
    assert type(head.readout) is type(r)
    x = torch.randn(4, H)
    fine = torch.arange(4) % r.n_disease
    idx = torch.arange(len(outcomes))
    terms = head.terms(x, fine, idx)[0]
    assert set(COMPONENTS) <= set(terms)
    torch.testing.assert_close(sum(terms.values()), model.competing_risk_log_hazards(x, fine).flatten(2)[:, idx])
    summary = head.readout.pooling_summary()
    assert summary["borrow_values"] == risk["adapter_borrow_values"]
