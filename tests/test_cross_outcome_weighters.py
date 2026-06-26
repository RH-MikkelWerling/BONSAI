import pytest
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from torch import nn
from types import MethodType

from opera.modules.networks.cross_outcome_weighters import (
    FAMOWeighter,
    UniformWeighter,
    build_cross_outcome_weighter,
    effective_number_outcome_factors,
)
from opera.modules.networks.opera_nets import (
    MultiOutcomeSurvivalLoss,
    outcome_eligibility_mask,
)


def _survival_batch():
    return {
        "a": {
            "times": torch.tensor([5.0, 10.0, 20.0, 30.0]),
            "events": torch.tensor([1, 1, 1, 1]),
        },
        "b": {
            "times": torch.tensor([3.0, 8.0, 18.0, 25.0]),
            "events": torch.tensor([1, 1, 1, 1]),
        },
    }


def test_kendall_pooled_matches_frozen_pre_refactor_reference():
    grids = {
        "a": torch.tensor([5.0, 10.0, 20.0, 30.0]),
        "b": torch.tensor([3.0, 8.0, 18.0, 25.0]),
    }
    torch.manual_seed(4)
    embeddings = F.normalize(torch.randn(4, 8), dim=-1)
    refactored = MultiOutcomeSurvivalLoss(
        ["a", "b"],
        outcome_sorted_event_times=grids,
        cross_outcome_config={
            "weighter": "kendall",
            "aggregation": "pooled",
            "class_balanced": False,
        },
    )
    values = torch.tensor([0.2, -0.15])
    refactored.log_sigma.data.copy_(values)

    actual = refactored(embeddings, _survival_batch())["loss"]

    # Captured from the pre-refactor implementation on this fixed-seed batch.
    assert actual.item() == pytest.approx(5.6688337326049805, rel=1e-6, abs=1e-7)


def test_legacy_kendall_state_dict_migrates_strictly():
    grids = {"a": torch.tensor([1.0, 2.0])}

    class Wrapper(nn.Module):
        def __init__(self):
            super().__init__()
            self.contrastive_loss = MultiOutcomeSurvivalLoss(
                ["a"],
                outcome_sorted_event_times=grids,
            )

    model = Wrapper()
    legacy_state = model.state_dict()
    legacy_state["contrastive_loss.log_sigma"] = torch.tensor([0.35])
    del legacy_state["contrastive_loss.weighter.log_sigma"]

    model.load_state_dict(legacy_state, strict=True)

    assert model.contrastive_loss.log_sigma.item() == pytest.approx(0.35)


def test_uniform_weighter_assigns_equal_active_weights():
    losses = torch.tensor([1.0, float("nan"), 3.0])

    weights = UniformWeighter().weights(losses)

    assert weights.tolist() == [1.0, 0.0, 1.0]


def test_famo_shifts_task_distribution_toward_slower_task():
    weighter = FAMOWeighter(
        n_outcomes=2,
        learning_rate=0.1,
        gamma=0.0,
    )
    weighter.train()
    weighter.weights(torch.tensor([10.0, 10.0]))

    for losses in (
        torch.tensor([5.0, 9.0]),
        torch.tensor([2.0, 8.0]),
        torch.tensor([1.0, 7.0]),
    ):
        weighter.update(losses)
        weighter.weights(losses)

    allocation = torch.softmax(weighter.task_logits, dim=0)
    assert allocation[1] > allocation[0]


def test_famo_eval_before_training_is_finite_and_read_only():
    weighter = FAMOWeighter(n_outcomes=2)
    weighter.eval()

    weights = weighter.weights(torch.tensor([2.0, 4.0]))

    assert torch.isfinite(weights).all()
    assert torch.isnan(weighter.initial_losses).all()


def test_famo_config_fails_closed_without_post_step_integration():
    with pytest.raises(ValueError, match="same-batch losses after"):
        build_cross_outcome_weighter(
            ["a", "b"],
            {"weighter": "famo", "aggregation": "macro"},
        )


def test_effective_number_factor_increases_rare_outcome_and_respects_cap():
    factors = effective_number_outcome_factors(
        ["rare", "balanced"],
        {
            "rare": {"positive": 1, "negative": 9999},
            "balanced": {"positive": 5000, "negative": 5000},
        },
        cap=10.0,
    )

    assert factors[0].item() > factors[1].item()
    assert factors[0].item() == pytest.approx(10.0)
    assert factors.max().item() <= 10.0


@pytest.mark.parametrize("weighter", ["uniform", "kendall"])
def test_no_eligible_outcome_produces_zero_without_nan(weighter):
    loss_fn = MultiOutcomeSurvivalLoss(
        ["a"],
        outcome_sorted_event_times={"a": torch.tensor([1.0, 2.0])},
        competing_event_handling="censor",
        cross_outcome_config={
            "weighter": weighter,
            "aggregation": "pooled",
            "class_balanced": False,
        },
    )
    embeddings = F.normalize(torch.randn(3, 4), dim=-1).requires_grad_(True)
    survival = {
        "a": {
            "times": torch.full((3,), -1.0),
            "events": torch.full((3,), -1),
        }
    }

    result = loss_fn(embeddings, survival)
    result["loss"].backward()

    assert torch.isfinite(result["loss"])
    assert result["loss"].item() == 0.0
    assert torch.isfinite(embeddings.grad).all()


def test_sentinel_row_has_zero_gradient_for_that_outcome():
    loss_fn = MultiOutcomeSurvivalLoss(
        ["a"],
        outcome_sorted_event_times={"a": torch.tensor([5.0, 20.0, 30.0])},
        effective_pair_normalization=False,
        cross_outcome_config={
            "weighter": "uniform",
            "aggregation": "macro",
            "class_balanced": False,
        },
    )
    embeddings = F.normalize(torch.randn(4, 6), dim=-1).requires_grad_(True)
    survival = {
        "a": {
            "times": torch.tensor([5.0, -1.0, 20.0, 30.0]),
            "events": torch.tensor([1, -1, 1, 1]),
        }
    }

    terms, _ = loss_fn.compute_per_outcome_losses(embeddings, survival)
    gradient = torch.autograd.grad(terms["a"]["loss"], embeddings)[0]

    assert outcome_eligibility_mask(
        survival["a"]["times"],
        survival["a"]["events"],
    ).tolist() == [True, False, True, True]
    assert torch.equal(gradient[1], torch.zeros_like(gradient[1]))
    assert gradient[[0, 2, 3]].abs().sum().item() > 0.0


def test_uniform_macro_and_pooled_differ_only_by_active_outcome_scale():
    embeddings = torch.ones(200, 2, requires_grad=True)

    def run(aggregation):
        loss_fn = MultiOutcomeSurvivalLoss(
            ["large", "small"],
            outcome_sorted_event_times={
                "large": torch.tensor([1.0]),
                "small": torch.tensor([1.0]),
            },
            cross_outcome_config={
                "weighter": "uniform",
                "aggregation": aggregation,
                "class_balanced": False,
            },
        )

        def fake_terms(self, values, *args, **kwargs):
            return {
                "large": {
                    "aggregation_loss": values[:200, 0].mean(),
                    "n_effective_pairs": torch.tensor(1.0),
                },
                "small": {
                    "aggregation_loss": values[:20, 1].mean(),
                    "n_effective_pairs": torch.tensor(1.0),
                },
            }, {}

        loss_fn.compute_per_outcome_losses = MethodType(fake_terms, loss_fn)
        loss = loss_fn(embeddings, {})["loss"]
        gradient = torch.autograd.grad(loss, embeddings, retain_graph=True)[0]
        return loss, gradient

    macro_loss, macro_gradient = run("macro")
    pooled_loss, pooled_gradient = run("pooled")

    assert macro_gradient[:, 0].abs().sum().item() == pytest.approx(
        macro_gradient[:, 1].abs().sum().item()
    )
    assert pooled_gradient[:, 0].abs().sum().item() == pytest.approx(
        pooled_gradient[:, 1].abs().sum().item()
    )
    assert torch.allclose(pooled_gradient, 2.0 * macro_gradient)
    assert pooled_loss.item() == pytest.approx(2.0 * macro_loss.item())


def test_production_contrastive_configs_use_uniform_macro():
    for path in (
        "opera/configs/contrastive.yaml",
        "opera/configs/contrastive_multicohort.yaml",
        "opera/configs/leukemia_contrastive.yaml",
    ):
        config = OmegaConf.load(path)
        assert config.cross_outcome.weighter == "uniform"
        assert config.cross_outcome.aggregation == "macro"


def test_production_joint_config_matches_contrastive_task_balancing():
    config = OmegaConf.load("opera/configs/joint_finetune.yaml")

    assert config.cross_outcome.weighter == "uniform"
    assert config.cross_outcome.aggregation == "macro"
    assert config.cross_outcome.positive_class_weighted is True
    assert config.cross_outcome.require_both_classes_per_batch is True


def test_zero_informative_pairs_do_not_apply_kendall_regularizer():
    loss_fn = MultiOutcomeSurvivalLoss(
        ["a"],
        outcome_sorted_event_times={"a": torch.tensor([1.0, 2.0])},
        competing_event_handling="censor",
        cross_outcome_config={
            "weighter": "kendall",
            "aggregation": "pooled",
            "class_balanced": False,
        },
    )
    loss_fn.log_sigma.data.fill_(0.5)
    embeddings = F.normalize(torch.randn(2, 4), dim=-1)
    survival = {
        "a": {
            "times": torch.tensor([1.0, 2.0]),
            "events": torch.tensor([1, 2]),
        }
    }

    result = loss_fn(embeddings, survival)

    assert result["n_effective_pairs/a"].item() == 0.0
    assert result["loss"].item() == 0.0
