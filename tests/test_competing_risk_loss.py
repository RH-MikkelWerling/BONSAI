import math

import pytest
import torch

from opera.modules.networks.competing_risk import (
    PiecewiseExponentialCompetingRiskLoss,
    estimate_piecewise_null_log_hazards,
    summarize_competing_risk_support,
    validate_competing_risk_sampling,
)


def _survival(times, events):
    return {
        "endpoint": {
            "times": torch.tensor(times, dtype=torch.float32),
            "events": torch.tensor(events, dtype=torch.long),
        }
    }


def test_piecewise_exponential_uses_exact_exposure_time():
    loss_fn = PiecewiseExponentialCompetingRiskLoss(
        ["endpoint"],
        [30.0, 90.0],
        time_scale_days=1.0,
        no_competing_outcomes=[],
    )
    log_hazards = torch.full((1, 1, 2, 3), math.log(0.01))

    early, _ = loss_fn(log_hazards, _survival([47.0], [2]))
    late, _ = loss_fn(log_hazards, _survival([89.0], [2]))

    # Same interval and event hazard, but 42 additional exact days exposed to
    # the two cause-specific hazards.
    assert late.item() - early.item() == pytest.approx(42.0 * 0.02, rel=1e-5)


def test_piecewise_exponential_matches_closed_form_target_event_loss():
    loss_fn = PiecewiseExponentialCompetingRiskLoss(
        ["endpoint"],
        [30.0],
        time_scale_days=1.0,
        no_competing_outcomes=[],
    )
    target_rate = 0.02
    death_rate = 0.01
    log_hazards = torch.tensor(
        [
            [
                [
                    [math.log(target_rate), math.log(target_rate)],
                    [
                        math.log(death_rate),
                        math.log(death_rate),
                    ],
                ]
            ]
        ]
    )

    loss, diagnostics = loss_fn(log_hazards, _survival([10.0], [1]))

    expected = 10.0 * (target_rate + death_rate) - math.log(target_rate)
    assert loss.item() == pytest.approx(expected)
    assert diagnostics["cr/n_target/endpoint"].item() == 1
    assert diagnostics["cr/n_death/endpoint"].item() == 0


def test_overall_survival_uses_one_cause_and_rejects_event_two():
    loss_fn = PiecewiseExponentialCompetingRiskLoss(
        ["overall_survival"],
        [30.0],
        time_scale_days=1.0,
    )
    log_hazards = torch.tensor(
        [[[[math.log(0.02), math.log(0.02)], [math.log(100.0), math.log(100.0)]]]]
    )

    loss, _ = loss_fn(
        log_hazards,
        {
            "overall_survival": {
                "times": torch.tensor([10.0]),
                "events": torch.tensor([1]),
            }
        },
    )
    assert loss.item() == pytest.approx(10.0 * 0.02 - math.log(0.02))

    with pytest.raises(ValueError, match="without a competing cause"):
        loss_fn(
            log_hazards,
            {
                "overall_survival": {
                    "times": torch.tensor([10.0]),
                    "events": torch.tensor([2]),
                }
            },
        )


def test_piecewise_exponential_masks_missing_outcomes_and_backpropagates():
    loss_fn = PiecewiseExponentialCompetingRiskLoss(
        ["endpoint"],
        [30.0, 90.0],
        no_competing_outcomes=[],
    )
    log_hazards = torch.zeros(3, 1, 2, 3, requires_grad=True)
    survival = _survival([15.0, -1.0, 100.0], [1, -1, 0])

    loss, diagnostics = loss_fn(log_hazards, survival)
    loss.backward()

    assert torch.isfinite(loss)
    assert diagnostics["cr/n_valid/endpoint"].item() == 2
    assert log_hazards.grad is not None
    assert torch.isfinite(log_hazards.grad).all()


def test_piecewise_exponential_supports_mixed_precision_log_hazards():
    loss_fn = PiecewiseExponentialCompetingRiskLoss(
        ["endpoint"],
        [30.0, 90.0],
        no_competing_outcomes=[],
    )
    log_hazards = torch.zeros(3, 1, 2, 3, dtype=torch.float16, requires_grad=True)

    loss, diagnostics = loss_fn(
        log_hazards,
        _survival([15.0, 45.0, 100.0], [1, 2, 0]),
    )
    loss.backward()

    assert torch.isfinite(loss)
    assert diagnostics["cr/n_target/endpoint"].item() == 1
    assert diagnostics["cr/n_death/endpoint"].item() == 1
    assert log_hazards.grad is not None
    assert torch.isfinite(log_hazards.grad).all()


@pytest.mark.parametrize(
    "boundaries",
    [[0.0, 30.0], [30.0, 30.0], [90.0, 30.0]],
)
def test_piecewise_exponential_rejects_invalid_boundaries(boundaries):
    with pytest.raises(ValueError, match="strictly increasing and positive"):
        PiecewiseExponentialCompetingRiskLoss(["endpoint"], boundaries)


def test_competing_risk_likelihood_requires_natural_sampling():
    validate_competing_risk_sampling(
        {"loss_weight": 1.0},
        {"type": "random"},
    )
    with pytest.raises(ValueError, match="natural-distribution random"):
        validate_competing_risk_sampling(
            {"loss_weight": 1.0},
            {"type": "event_aware"},
        )

    # Contrastive-only ablations may still use event-aware batches.
    validate_competing_risk_sampling(
        {"loss_weight": 0.0},
        {"type": "event_aware"},
    )


def test_competing_risk_support_counts_exact_intervals():
    class Dataset:
        outcome_dicts = {
            "endpoint": {
                1: {"time_days": 3.0, "event": 1},
                2: {"time_days": 3.1, "event": 2},
                3: {"time_days": 20.0, "event": 0},
            }
        }

    rows = summarize_competing_risk_support(Dataset(), ["endpoint"], [3.0, 14.0])

    assert rows[0]["n_target"] == 1
    assert rows[1]["n_death"] == 1
    assert rows[2]["n_censored"] == 1


def test_piecewise_null_hazards_use_training_events_and_person_time():
    class Dataset:
        outcome_dicts = {
            "endpoint": {
                1: {"time_days": 5.0, "event": 1},
                2: {"time_days": 15.0, "event": 2},
                3: {"time_days": 20.0, "event": 0},
            }
        }

    fitted = estimate_piecewise_null_log_hazards(
        Dataset(), ["endpoint"], [10.0], time_scale_days=1.0,
        no_competing_outcomes=[]
    )
    # First interval: 25 person-days and one target event. Second interval:
    # 15 person-days and one competing event.
    assert math.exp(fitted["endpoint"][0][0]) == pytest.approx(1.0 / 25.0)
    assert math.exp(fitted["endpoint"][1][1]) == pytest.approx(1.0 / 15.0)


def test_kendall_null_normalizes_against_fixed_train_hazards():
    objective = PiecewiseExponentialCompetingRiskLoss(
        ["endpoint"], [10.0], no_competing_outcomes=[], time_scale_days=1.0,
        weighter="kendall_null",
        null_log_hazards={"endpoint": [[math.log(0.02)] * 2, [math.log(0.01)] * 2]},
    )
    hazards = torch.tensor(
        [[[[math.log(0.02)] * 2, [math.log(0.01)] * 2]]], requires_grad=True
    )
    loss, diagnostics = objective(hazards, _survival([5.0], [1]))
    # Matching the null gives ratio one; Kendall starts at precision 0.5.
    assert diagnostics["cr/loss_over_null/endpoint"].item() == pytest.approx(1.0)
    assert loss.item() == pytest.approx(0.5)
    loss.backward()
    assert hazards.grad is not None


def test_kendall_survival_weights_are_checkpoint_parameters():
    objective = PiecewiseExponentialCompetingRiskLoss(
        ["a", "b"], [10.0], no_competing_outcomes=[], weighter="kendall"
    )
    assert "weighter.log_sigma" in objective.state_dict()


def test_kendall_uses_positive_day_density_nll_for_annual_hazards():
    annual_rate = 10.0
    objective = PiecewiseExponentialCompetingRiskLoss(
        ["endpoint"], [], no_competing_outcomes=["endpoint"],
        time_scale_days=365.25, weighter="kendall"
    )
    hazards = torch.tensor([[[[math.log(annual_rate)], [-12.0]]]])
    loss, _, terms = objective(
        hazards, _survival([1.0], [1]), return_per_outcome=True
    )
    assert terms["endpoint"]["full_likelihood_loss"].item() < 0.0
    assert terms["endpoint"]["aggregation_loss"].item() > 0.0
    assert loss.item() > 0.0


def test_likelihood_is_batch_partition_invariant():
    loss_fn = PiecewiseExponentialCompetingRiskLoss(
        ["endpoint"],
        [30.0, 90.0],
        no_competing_outcomes=[],
    )
    torch.manual_seed(4)
    log_hazards = torch.randn(8, 1, 2, 3) - 2.0
    times = torch.tensor([5.0, 20.0, 35.0, 70.0, 95.0, 120.0, 200.0, 400.0])
    events = torch.tensor([1, 0, 2, 1, 0, 2, 1, 0])

    full, _ = loss_fn(
        log_hazards,
        {"endpoint": {"times": times, "events": events}},
    )
    first, _ = loss_fn(
        log_hazards[:3],
        {"endpoint": {"times": times[:3], "events": events[:3]}},
    )
    second, _ = loss_fn(
        log_hazards[3:],
        {"endpoint": {"times": times[3:], "events": events[3:]}},
    )

    partitioned = (3 * first + 5 * second) / 8
    assert partitioned.item() == pytest.approx(full.item(), rel=1e-6)


def test_support_curriculum_renormalizes_training_and_validation_uses_all_outcomes():
    settings = {
        "aggregation": "hierarchical_support",
        "outcome_families": {"rich": ["a"], "middle": ["b"], "rare": ["c"]},
        "event_location_counts": {"a": 100, "b": 50, "c": 10},
        "curriculum": {
            "enabled": True,
            "n_support_tiers": 3,
            "warmup_epochs": 1,
            "stage_epochs": 1,
            "ramp_epochs": 1,
        },
    }
    loss_fn = PiecewiseExponentialCompetingRiskLoss(
        ["a", "b", "c"], [30.0], no_competing_outcomes=[], cross_outcome_config=settings
    )
    hazards = torch.zeros(2, 3, 2, 2)
    survival = {
        name: {
            "times": torch.tensor([10.0, 40.0]),
            "events": torch.tensor([1, 0]),
        }
        for name in ("a", "b", "c")
    }

    loss_fn.set_curriculum_state(0, training=True)
    _, warmup = loss_fn(hazards, survival)
    assert warmup["cr/family_weight/rich"].item() == pytest.approx(1.0)
    assert "cr/family_weight/middle" not in warmup
    assert "cr/family_weight/rare" not in warmup

    loss_fn.set_curriculum_state(1, training=True)
    _, ramp = loss_fn(hazards, survival)
    assert ramp["cr/curriculum/tier_1_multiplier"].item() == pytest.approx(1.0)
    assert "cr/family_weight/middle" in ramp
    assert "cr/family_weight/rare" not in ramp
    assert sum(
        ramp[key].item() for key in ramp if key.startswith("cr/family_weight/")
    ) == pytest.approx(1.0)

    loss_fn.set_curriculum_state(0, training=False)
    _, validation = loss_fn(hazards, survival)
    assert set(
        key.removeprefix("cr/family_weight/")
        for key in validation
        if key.startswith("cr/family_weight/")
    ) == {"rich", "middle", "rare"}


def test_competing_risk_can_return_differentiable_per_outcome_terms():
    loss_fn = PiecewiseExponentialCompetingRiskLoss(
        ["a", "b"], [30.0], no_competing_outcomes=[]
    )
    hazards = torch.zeros(3, 2, 2, 2, requires_grad=True)
    survival = {
        name: {
            "times": torch.tensor([10.0, 40.0, 20.0]),
            "events": torch.tensor([1, 0, 2]),
        }
        for name in ("a", "b")
    }

    _, _, terms = loss_fn(hazards, survival, return_per_outcome=True)
    gradient = torch.autograd.grad(terms["a"]["loss"], hazards)[0]

    assert set(terms) == {"a", "b"}
    assert terms["a"]["valid_mask"].tolist() == [True, True, True]
    assert torch.isfinite(gradient).all()
    assert gradient[:, 1].abs().sum().item() == 0.0
# Component losses must retain the full-likelihood scale used by the model.
def test_per_outcome_competing_risk_components_sum_to_full_likelihood():
    from opera.modules.networks.competing_risk import PiecewiseExponentialCompetingRiskLoss

    objective = PiecewiseExponentialCompetingRiskLoss(
        ["lab"], [30.0], no_competing_outcomes=(), time_scale_days=1.0
    )
    hazards = torch.tensor(
        [[[[0.0, -0.2], [-0.4, -0.5]]], [[[0.1, 0.2], [-0.3, -0.1]]]],
        requires_grad=True,
    )
    survival = {"lab": {"times": torch.tensor([10.0, 45.0]), "events": torch.tensor([1, 2])}}
    _, _, terms = objective(hazards, survival, return_per_outcome=True)
    term = terms["lab"]
    expected = term["exposure_loss"] + term["primary_event_loss"] + term["competing_event_loss"]
    assert torch.allclose(term["full_likelihood_loss"], expected)
