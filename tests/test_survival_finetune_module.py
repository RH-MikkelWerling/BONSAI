import torch
import torch.nn.functional as F
from torch import nn
import pytest

from opera.functional.ipcw import compute_ipcw_train_weights
from opera.modules.lightningmodules.SurvivalFinetuneModule import (
    SurvivalFinetuneModule,
    cox_partial_likelihood_loss,
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
    nonzero = torch.tensor([w for w in weights.values() if w > 0.0])
    assert torch.isclose(nonzero.mean(), torch.tensor(1.0), atol=1e-6)


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
    module = SurvivalFinetuneModule(model, "ipcw_bce")
    batch = _batch(weights=[1.0, 0.0, 1.0, 1.0])

    loss = module.training_step(batch, 0)
    loss.backward()

    assert model.scores.grad[1].item() == 0.0

    logits = torch.tensor([0.3, -0.2, 1.1, -1.0])
    labels = _batch()["target"].reshape(-1).float()
    unit_module = SurvivalFinetuneModule(BatchScoreModel(logits.tolist()), "ipcw_bce")
    unit_loss = unit_module.training_step(_batch(), 0)
    expected = F.binary_cross_entropy_with_logits(logits, labels)
    assert torch.isclose(unit_loss, expected)


def test_ipcw_bce_loss_changes_when_ipcw_weight_is_halved():
    logits = [0.3, -0.2, 1.1, -1.0]
    module = SurvivalFinetuneModule(BatchScoreModel(logits), "ipcw_bce")
    full_loss = module.training_step(_batch(weights=[1.0, 1.0, 1.0, 1.0]), 0)
    half_loss = module.training_step(_batch(weights=[0.5, 1.0, 1.0, 1.0]), 0)
    assert not torch.isclose(full_loss, half_loss)
