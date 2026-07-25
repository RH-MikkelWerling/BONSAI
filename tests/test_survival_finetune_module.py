import torch
import torch.nn.functional as F
import lightning as L
from torch import nn
from torch.utils.data import DataLoader, Dataset
import pytest
from omegaconf import OmegaConf

from opera.functional.ipcw import (
    compute_ipcw_train_weights,
    summarize_ipcw_weights,
)
from opera.modules.lightningmodules.SurvivalFinetuneModule import (
    SurvivalFinetuneModule,
    cox_batch_signal_counts,
    cox_partial_likelihood_loss,
    exact_breslow_cox_loss,
)
from opera.run.survival_finetune import (
    _validate_cox_support,
    resolve_survival_monitor,
)


def _outcomes():
    return {
        1: {"time_days": 10.0, "event": 1, "label": 1},
        2: {"time_days": 40.0, "event": 0, "label": 0},
        3: {"time_days": 5.0, "event": 0, "label": 0},
        4: {"time_days": 12.0, "event": 2, "label": 0},
        5: {"time_days": 50.0, "event": 1, "label": 0},
        6: {"time_days": 45.0, "event": 2, "label": 0},
    }


def test_compute_ipcw_train_weights_cases_controls_and_normalization():
    weights = compute_ipcw_train_weights(_outcomes(), horizon_hours=30 * 24)

    assert weights[1] > 0.0
    assert weights[2] > 0.0
    assert weights[3] == 0.0
    assert weights[4] == 0.0
    assert weights[6] > 0.0
    assert torch.isclose(
        torch.tensor(list(weights.values())).mean(),
        torch.tensor(1.0),
        atol=1e-6,
    )


def test_cumulative_incidence_weights_keep_competing_events_as_controls():
    weights = compute_ipcw_train_weights(
        _outcomes(),
        horizon_hours=30 * 24,
        estimand="cumulative_incidence",
    )

    assert weights[1] > 0.0
    assert weights[2] > 0.0
    assert weights[3] == 0.0  # administrative censoring before the horizon
    assert weights[4] > 0.0  # competing event is a known negative
    assert weights[6] > 0.0
    assert torch.isclose(
        torch.tensor(list(weights.values())).mean(),
        torch.tensor(1.0),
        atol=1e-6,
    )


def test_ipcw_weight_diagnostics_report_effective_sample_size():
    weights = compute_ipcw_train_weights(
        _outcomes(),
        horizon_hours=30 * 24,
        estimand="cumulative_incidence",
    )
    diagnostics = summarize_ipcw_weights(_outcomes(), weights)

    assert diagnostics["n_total"] == 6
    assert diagnostics["n_cases_nonzero"] == 1
    assert diagnostics["n_controls_nonzero"] == 4
    assert 0 < diagnostics["effective_sample_size"] <= 6
    assert diagnostics["mean_weight"] == pytest.approx(1.0)


def test_ipcw_cif_bce_module_uses_weighted_binary_loss():
    module = SurvivalFinetuneModule(
        BatchScoreModel([0.3, -0.2, 1.1, -1.0]),
        "ipcw_cif_bce",
        horizon_days=20.0,
    )
    loss = module.training_step(_batch(weights=[1.0, 0.0, 1.0, 1.0]), 0)
    assert torch.isfinite(loss)


def test_ipcw_cif_validation_logs_full_cohort_weighted_auc(monkeypatch):
    module = SurvivalFinetuneModule(
        BatchScoreModel([4.0, 1.0, 0.5, 0.0]),
        "ipcw_cif_bce",
        horizon_days=20.0,
    )
    batch = _batch()
    batch["target"] = torch.tensor([[1], [0], [0], [0]], dtype=torch.long)
    batch["event"] = torch.tensor([[1], [2], [0], [0]], dtype=torch.long)
    captured = {}

    def capture(name, value, **kwargs):
        captured[name] = float(value.detach())

    monkeypatch.setattr(module, "log", capture)
    module.validation_step(batch, 0)
    module.on_validation_epoch_end()

    assert captured["val/AUROC"] == pytest.approx(1.0)


def test_cox_partial_likelihood_loss_scalar_grad_and_no_events():
    risk = torch.tensor([3.0, 2.0, 1.0], requires_grad=True)
    times = torch.tensor([5.0, 10.0, 15.0])
    events = torch.tensor([1, 0, 1])

    loss = cox_partial_likelihood_loss(risk, times, events)

    assert loss.ndim == 0
    assert loss.requires_grad
    loss.backward()
    assert risk.grad is not None
    assert torch.isfinite(risk.grad).all()

    no_event_risk = torch.tensor([1.0, 2.0], requires_grad=True)
    no_event_loss = cox_partial_likelihood_loss(
        no_event_risk,
        torch.tensor([1.0, 2.0]),
        torch.tensor([0, 0]),
    )
    assert no_event_loss.requires_grad
    assert no_event_loss.item() == 0.0


def test_cox_partial_likelihood_loss_rewards_correct_ordering():
    times = torch.tensor([5.0, 10.0, 20.0])
    events = torch.tensor([1, 1, 0])
    correct = cox_partial_likelihood_loss(
        torch.tensor([3.0, 2.0, 0.0]),
        times,
        events,
    )
    incorrect = cox_partial_likelihood_loss(
        torch.tensor([0.0, 2.0, 3.0]),
        times,
        events,
    )

    assert correct < incorrect


def test_exact_breslow_cox_matches_naive_full_risk_sets_with_ties():
    risk = torch.tensor([0.7, -0.4, 1.2, 0.1, -0.8], requires_grad=True)
    times = torch.tensor([10.0, 10.0, 8.0, 5.0, 3.0])
    # A competing death tied with the first primary event remains in that
    # event's risk set but never contributes a primary-event numerator.
    events = torch.tensor([1, 2, 1, 0, 1])

    exact = exact_breslow_cox_loss(risk, times, events)
    exact_gradient = torch.autograd.grad(exact, risk, retain_graph=True)[0]
    naive = cox_partial_likelihood_loss(risk, times, events)
    naive_gradient = torch.autograd.grad(naive, risk)[0]

    assert torch.allclose(exact, naive, atol=1e-6)
    assert torch.allclose(exact_gradient, naive_gradient, atol=1e-6)


def test_cached_score_gradient_matches_direct_full_cohort_model_gradient():
    features = torch.tensor(
        [[1.0, 0.2], [0.5, -0.4], [-0.2, 0.7], [0.1, -0.8]]
    )
    times = torch.tensor([4.0, 7.0, 11.0, 13.0])
    events = torch.tensor([1, 2, 1, 0])
    direct_model = nn.Linear(2, 1, bias=False)
    cached_model = nn.Linear(2, 1, bias=False)
    cached_model.load_state_dict(direct_model.state_dict())

    direct_scores = direct_model(features).reshape(-1)
    direct_loss = exact_breslow_cox_loss(direct_scores, times, events)
    direct_loss.backward()

    cached_scores = cached_model(features).reshape(-1).detach().requires_grad_(True)
    score_gradient = torch.autograd.grad(
        exact_breslow_cox_loss(cached_scores, times, events),
        cached_scores,
    )[0]
    for microbatch in (slice(0, 2), slice(2, 4)):
        recomputed = cached_model(features[microbatch]).reshape(-1)
        (recomputed * score_gradient[microbatch]).sum().backward()

    assert torch.allclose(
        cached_model.weight.grad,
        direct_model.weight.grad,
        atol=1e-6,
    )


class _ExactCoxDataset(Dataset):
    def __init__(self):
        self.features = torch.tensor(
            [[1.0, 0.2], [0.5, -0.4], [-0.2, 0.7], [0.1, -0.8]]
        )
        self.times = torch.tensor([4.0, 7.0, 11.0, 13.0])
        self.events = torch.tensor([1, 2, 1, 0])

    def __len__(self):
        return len(self.times)

    def __getitem__(self, index):
        return {
            "features": self.features[index],
            "subject_id": torch.tensor(index + 1),
            "target": torch.tensor(0),
            "time_days": self.times[index],
            "event": self.events[index],
        }


class _FeatureScoreModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(2, 1, bias=False)

    def forward(self, batch):
        return self.linear(batch["features"])


def test_lightning_exact_cached_cox_runs_two_pass_optimizer_step():
    loader = DataLoader(_ExactCoxDataset(), batch_size=2, shuffle=False)
    model = _FeatureScoreModel()
    initial = model.linear.weight.detach().clone()
    module = SurvivalFinetuneModule(
        model,
        "cox_exact_cached",
        learning_rate=0.05,
    )
    trainer = L.Trainer(
        accelerator="cpu",
        devices=1,
        max_epochs=1,
        logger=False,
        enable_checkpointing=False,
        enable_model_summary=False,
        enable_progress_bar=False,
    )

    trainer.fit(module, train_dataloaders=loader, val_dataloaders=loader)

    assert not torch.equal(model.linear.weight.detach(), initial)


def test_cox_batch_signal_counts_distinguishes_events_without_comparators():
    n_events, n_comparable = cox_batch_signal_counts(
        torch.tensor([20.0, 10.0, 5.0]),
        torch.tensor([1, 0, 0]),
    )
    assert n_events.item() == 1
    assert n_comparable.item() == 0

    n_events, n_comparable = cox_batch_signal_counts(
        torch.tensor([5.0, 10.0, 20.0]),
        torch.tensor([1, 0, 0]),
    )
    assert n_events.item() == 1
    assert n_comparable.item() == 1


class BatchScoreModel(nn.Module):
    def __init__(self, scores):
        super().__init__()
        self.scores = nn.Parameter(torch.tensor(scores, dtype=torch.float32))

    def forward(self, batch):
        return self.scores[: batch["target"].shape[0]].unsqueeze(1)


def _batch(weights=None):
    if weights is None:
        weights = [1.0, 1.0, 1.0, 1.0]
    return {
        "target": torch.tensor([[1], [1], [0], [0]], dtype=torch.long),
        "time_days": torch.tensor([[5.0], [10.0], [15.0], [20.0]]),
        "event": torch.tensor([[1], [1], [0], [0]], dtype=torch.long),
        "ipcw_weight": torch.tensor(weights, dtype=torch.float32).unsqueeze(1),
    }


def test_survival_finetune_module_cox_training_step_and_ordering():
    good = SurvivalFinetuneModule(BatchScoreModel([4.0, 3.0, 1.0, 0.0]), "cox")
    bad = SurvivalFinetuneModule(BatchScoreModel([0.0, 1.0, 3.0, 4.0]), "cox")

    good_loss = good.training_step(_batch(), 0)
    bad_loss = bad.training_step(_batch(), 0)

    assert good_loss.ndim == 0
    assert good_loss < bad_loss


def test_survival_finetune_module_cox_rejects_pos_weight():
    with pytest.raises(ValueError, match="pos_weight"):
        SurvivalFinetuneModule(
            BatchScoreModel([1.0, 0.0]),
            "cox",
            pos_weight=torch.tensor([2.0]),
        )


def test_survival_finetune_module_competing_events_do_not_crash_and_logs_cindex():
    module = SurvivalFinetuneModule(BatchScoreModel([4.0, 3.0, 1.0, 0.0]), "cox")
    batch = _batch()
    batch["event"][2] = 2
    loss = module.validation_step(batch, 0)
    module.on_validation_epoch_end()
    assert torch.isfinite(loss)


def test_ipcw_bce_zero_weight_blocks_gradient_and_unit_weights_match_bce():
    model = BatchScoreModel([0.3, -0.2, 1.1, -1.0])
    module = SurvivalFinetuneModule(model, "ipcw_bce", horizon_days=20.0)
    batch = _batch(weights=[1.0, 0.0, 1.0, 1.0])

    loss = module.training_step(batch, 0)
    loss.backward()

    assert model.scores.grad[1].item() == 0.0

    logits = torch.tensor([0.3, -0.2, 1.1, -1.0])
    labels = _batch()["target"].reshape(-1).float()
    unit_module = SurvivalFinetuneModule(
        BatchScoreModel(logits.tolist()),
        "ipcw_bce",
        horizon_days=20.0,
    )
    unit_loss = unit_module.training_step(_batch(), 0)
    expected = F.binary_cross_entropy_with_logits(logits, labels)
    assert torch.isclose(unit_loss, expected)


def test_ipcw_bce_loss_changes_when_ipcw_weight_is_halved():
    logits = [0.3, -0.2, 1.1, -1.0]
    module = SurvivalFinetuneModule(
        BatchScoreModel(logits),
        "ipcw_bce",
        horizon_days=20.0,
    )
    full_loss = module.training_step(_batch(weights=[1.0, 1.0, 1.0, 1.0]), 0)
    half_loss = module.training_step(_batch(weights=[0.5, 1.0, 1.0, 1.0]), 0)
    raw = F.binary_cross_entropy_with_logits(
        torch.tensor(logits),
        _batch()["target"].reshape(-1).float(),
        reduction="none",
    )
    expected = (raw * torch.tensor([0.5, 1.0, 1.0, 1.0])).mean()
    assert torch.isclose(half_loss, expected)
    assert not torch.isclose(full_loss, half_loss)


@pytest.mark.parametrize(
    ("training_mode", "expected"),
    [
        ("cox", ("val/concordance_index", "max")),
        ("cox_exact_cached", ("val/concordance_index", "max")),
        ("ipcw_bce", ("val/loss", "min")),
        ("ipcw_cif_bce", ("val/loss", "min")),
    ],
)
def test_survival_monitor_auto_is_estimand_appropriate(training_mode, expected):
    cfg = OmegaConf.create(
        {
            "training_mode": training_mode,
            "training": {"eval_monitor_metric": "auto"},
        }
    )
    assert resolve_survival_monitor(cfg) == expected


def test_cox_support_fails_before_training_without_comparable_events():
    with pytest.raises(ValueError, match="no comparable primary events"):
        _validate_cox_support(
            "tuning",
            {
                1: {"time_days": 10.0, "event": 1},
                2: {"time_days": 5.0, "event": 0},
            },
        )

    summary = _validate_cox_support(
        "tuning",
        {
            1: {"time_days": 10.0, "event": 1},
            2: {"time_days": 20.0, "event": 0},
        },
    )
    assert summary["n_comparable_primary_events"] == 1
