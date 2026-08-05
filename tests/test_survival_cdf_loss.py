"""
Tests for the survival loss fixes:
  Fix 1 - Kaplan-Meier event-mass time similarity
  Fix 2 - Signal-weighted anchor mean
  Fix 3 - KM-based admin-censoring similarity
"""

import pytest
import pandas as pd
import torch
import torch.nn.functional as F

from opera.modules.networks.opera_nets import (
    MultiOutcomeSurvivalLoss,
    SurvivalSoftContrastiveLoss,
)
from opera.modules.datamodules.ContrastiveDataModule import (
    _km_event_time_probabilities,
    compute_sorted_event_times,
)
from opera.modules.datamodules.MultiCohortContrastiveDataModule import (
    compute_pooled_event_time_probability_grids,
    compute_pooled_sorted_event_times,
)


def _norm(x: torch.Tensor) -> torch.Tensor:
    return F.normalize(x, dim=-1)


def _km_q(
    sorted_et: torch.Tensor, t: float, probs: torch.Tensor | None = None
) -> float:
    if probs is None:
        probs = torch.full((sorted_et.numel(),), 1.0 / max(sorted_et.numel(), 1))
    probs = probs / probs.sum().clamp_min(1e-12)
    pos = torch.searchsorted(sorted_et, torch.tensor(t), right=True).item()
    if pos <= 0:
        return 0.0
    return min(float(probs[:pos].sum().item()), 1.0)


def test_km_event_mass_spreads_dense_event_regions_more_than_sparse_regions():
    sorted_et = torch.tensor(
        [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 700.0, 800.0]
    )
    q_diff_early = abs(_km_q(sorted_et, 10.0) - _km_q(sorted_et, 20.0))
    q_diff_late = abs(_km_q(sorted_et, 700.0) - _km_q(sorted_et, 710.0))
    assert q_diff_early > q_diff_late


def test_km_event_mass_is_batch_size_independent():
    sorted_et = torch.tensor([50.0, 100.0, 200.0, 400.0, 600.0])
    assert _km_q(sorted_et, 100.0) == pytest.approx(0.4, abs=1e-6)
    assert _km_q(sorted_et, 200.0) == pytest.approx(0.6, abs=1e-6)


def test_km_event_mass_clamps_out_of_range_times():
    sorted_et = torch.tensor([100.0, 200.0, 300.0])
    assert _km_q(sorted_et, 1.0) == pytest.approx(0.0)
    assert _km_q(sorted_et, 9999.0) == pytest.approx(1.0)


def test_km_event_mass_empty_sorted_events_does_not_crash():
    sorted_et = torch.tensor([], dtype=torch.float32)
    times = torch.tensor([10.0, 20.0, 30.0])
    events = torch.tensor([1, 0, 1])
    torch.manual_seed(0)
    emb = _norm(torch.randn(3, 8))
    loss_fn = SurvivalSoftContrastiveLoss(km_time_scale=0.25)
    result = loss_fn(emb, times, events, sorted_event_times=sorted_et)
    assert torch.isfinite(result)
    assert result.item() == pytest.approx(0.0)


def test_rare_outcome_two_events_are_distinguished():
    sorted_et = torch.tensor([100.0, 500.0])
    assert _km_q(sorted_et, 100.0) == pytest.approx(0.5, abs=1e-6)
    assert _km_q(sorted_et, 500.0) == pytest.approx(1.0, abs=1e-6)
    assert _km_q(sorted_et, 50.0) == pytest.approx(0.0, abs=1e-6)


def test_km_event_probabilities_change_similarity_geometry():
    sorted_et = torch.tensor([10.0, 50.0, 100.0])
    uniform_probs = torch.tensor([1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0])
    km_probs = torch.tensor([0.8, 0.1, 0.1])
    loss_fn = SurvivalSoftContrastiveLoss(km_time_scale=0.25)
    _, q_uniform, _ = loss_fn._event_grid(sorted_et, uniform_probs, torch.device("cpu"))
    _, q_km, _ = loss_fn._event_grid(sorted_et, km_probs, torch.device("cpu"))

    uniform_gap = q_uniform[2] - q_uniform[1]
    km_gap = q_km[2] - q_km[1]

    assert uniform_gap.item() == pytest.approx(1.0 / 3.0)
    assert km_gap.item() == pytest.approx(0.1)


def test_signal_weighted_loss_is_finite_and_has_gradient():
    sorted_et = torch.tensor([30.0, 90.0, 180.0, 365.0, 730.0])
    times = torch.tensor([30.0, 90.0, 180.0, 365.0])
    events = torch.tensor([1, 1, 1, 1])
    torch.manual_seed(42)
    emb = _norm(torch.randn(4, 8))
    loss_fn = SurvivalSoftContrastiveLoss(km_time_scale=0.25)
    result = loss_fn(emb, times, events, sorted_event_times=sorted_et)
    assert torch.isfinite(result)
    assert result.requires_grad
    assert result.item() > 0


def test_contrastive_cross_entropy_decomposes_into_entropy_and_kl():
    sorted_et = torch.tensor([30.0, 90.0, 180.0, 365.0])
    times = torch.tensor([30.0, 90.0, 180.0, 365.0])
    events = torch.ones(4, dtype=torch.long)
    emb = _norm(torch.randn(4, 8, generator=torch.Generator().manual_seed(19)))
    loss_fn = SurvivalSoftContrastiveLoss(km_time_scale=0.25)

    cross_entropy, diagnostics = loss_fn(
        emb,
        times,
        events,
        sorted_event_times=sorted_et,
        return_diagnostics=True,
    )

    assert diagnostics["excess_loss"].item() >= -1e-6
    assert cross_entropy.item() == pytest.approx(
        diagnostics["target_entropy"].item() + diagnostics["excess_loss"].item(),
        abs=1e-6,
    )
    assert diagnostics["uniform_baseline"] >= diagnostics["target_entropy"]


def test_kl_and_cross_entropy_terms_have_identical_embedding_gradients():
    sorted_et = {"mortality": torch.tensor([30.0, 90.0, 180.0, 365.0])}
    survival = {
        "mortality": {
            "times": torch.tensor([30.0, 90.0, 180.0, 365.0]),
            "events": torch.ones(4, dtype=torch.long),
        }
    }
    base = _norm(torch.randn(4, 8, generator=torch.Generator().manual_seed(23)))

    def gradient(term):
        emb = base.detach().clone().requires_grad_(True)
        loss_fn = MultiOutcomeSurvivalLoss(
            outcome_names=["mortality"],
            outcome_sorted_event_times=sorted_et,
            effective_pair_normalization=False,
            cross_outcome_config={
                "weighter": "uniform",
                "aggregation": "macro",
                "contrastive_term": term,
            },
        )
        result = loss_fn(emb, survival)
        return result["loss"].detach(), torch.autograd.grad(result["loss"], emb)[0]

    ce_loss, ce_gradient = gradient("cross_entropy")
    kl_loss, kl_gradient = gradient("kl")

    assert kl_loss < ce_loss
    assert torch.allclose(ce_gradient, kl_gradient, atol=1e-6, rtol=1e-5)


def test_initial_kl_scaling_uses_fixed_reference_without_detaching_loss():
    embeddings = _norm(torch.randn(4, 8)).requires_grad_(True)
    survival = {
        "mortality": {
            "times": torch.tensor([30.0, 90.0, 180.0, 365.0]),
            "events": torch.ones(4, dtype=torch.long),
        }
    }
    loss_fn = MultiOutcomeSurvivalLoss(
        outcome_names=["mortality"],
        outcome_sorted_event_times={"mortality": survival["mortality"]["times"]},
        effective_pair_normalization=False,
        cross_outcome_config={
            "weighter": "uniform",
            "aggregation": "macro",
            "contrastive_term": "kl",
            "outcome_scale_mode": "initial_kl",
            "outcome_reference_scales": {"mortality": 2.0},
        },
    )
    terms, logs = loss_fn.compute_per_outcome_losses(embeddings, survival)

    assert torch.allclose(
        terms["mortality"]["aggregation_loss"],
        terms["mortality"]["excess_loss"] / 2.0,
    )
    assert logs["outcome_scale_factor/mortality"].item() == pytest.approx(0.5)
    torch.autograd.grad(terms["mortality"]["aggregation_loss"], embeddings)


def test_initial_kl_scaling_requires_complete_positive_references():
    with pytest.raises(ValueError, match="finite positive reference"):
        MultiOutcomeSurvivalLoss(
            outcome_names=["mortality"],
            outcome_sorted_event_times={"mortality": torch.tensor([1.0])},
            cross_outcome_config={
                "outcome_scale_mode": "initial_kl",
                "outcome_reference_scales": {},
            },
        )


def test_noisy_anchor_perturb_moves_loss_less_than_signal_rich():
    sorted_et = torch.tensor([10.0, 30.0, 90.0, 180.0, 365.0, 730.0])
    times = torch.tensor([180.0, 2.0, 30.0, 90.0, 365.0, 730.0])
    events = torch.tensor([1, 0, 1, 1, 1, 1])

    torch.manual_seed(7)
    base_emb = _norm(torch.randn(6, 8))
    loss_fn = SurvivalSoftContrastiveLoss(km_time_scale=0.25)
    base_loss = loss_fn(base_emb, times, events, sorted_event_times=sorted_et).item()

    g = torch.Generator()
    emb_rich = base_emb.clone()
    emb_rich[0] = _norm(torch.randn(8, generator=g.manual_seed(99)))
    loss_rich = loss_fn(emb_rich, times, events, sorted_event_times=sorted_et).item()

    emb_noisy = base_emb.clone()
    emb_noisy[1] = _norm(torch.randn(8, generator=g.manual_seed(99)))
    loss_noisy = loss_fn(emb_noisy, times, events, sorted_event_times=sorted_et).item()

    assert abs(loss_rich - base_loss) > abs(loss_noisy - base_loss)


def test_admin_censoring_uses_future_event_distribution():
    sorted_et = torch.tensor([10.0, 50.0, 100.0, 300.0, 500.0])
    loss_fn = SurvivalSoftContrastiveLoss(km_time_scale=0.25)
    event_grid, km_grid, event_probs = loss_fn._event_grid(
        sorted_et, None, torch.device("cpu")
    )
    dist = loss_fn._patient_quantile_distributions(
        torch.tensor([5.0, 600.0]),
        torch.tensor([0, 0]),
        event_grid,
        km_grid,
        event_probs,
    )

    assert dist[0].nonzero().numel() == sorted_et.numel()
    assert dist[0, 0].item() == pytest.approx(0.0)
    assert dist[1, -1].item() == pytest.approx(1.0)
    assert dist[0, 1].item() == pytest.approx(1.0 / sorted_et.numel())


def test_admin_censoring_uses_km_tail_probabilities_when_available():
    sorted_et = torch.tensor([10.0, 50.0, 100.0])
    event_probs = torch.tensor([0.7, 0.2, 0.1])
    loss_fn = SurvivalSoftContrastiveLoss(km_time_scale=0.25)
    event_grid, km_grid, probs = loss_fn._event_grid(
        sorted_et,
        event_probs,
        torch.device("cpu"),
    )
    dist = loss_fn._patient_quantile_distributions(
        torch.tensor([20.0]),
        torch.tensor([0]),
        event_grid,
        km_grid,
        probs,
    )

    assert dist[0, 0].item() == pytest.approx(0.0)
    assert dist[0, 1].item() == pytest.approx(0.0)
    assert dist[0, 2].item() == pytest.approx(2.0 / 3.0)
    assert dist[0, 3].item() == pytest.approx(1.0 / 3.0)


def test_competing_death_before_first_event_maps_to_first_event_slot():
    # A competing-death patient at time 5 < first event at 30 should land on
    # slot 1 (first real event slot), not slot 0 (km_grid sentinel = 0.0).
    sorted_et = torch.tensor([30.0, 90.0])
    loss_fn = SurvivalSoftContrastiveLoss(
        km_time_scale=0.25,
        competing_event_handling="censor",
    )
    event_grid, km_grid, probs = loss_fn._event_grid(
        sorted_et, None, torch.device("cpu")
    )
    dist = loss_fn._patient_quantile_distributions(
        torch.tensor([5.0]),
        torch.tensor([2]),
        event_grid,
        km_grid,
        probs,
    )

    # Slot 0 is the sentinel (km_grid[0] = 0.0) and must never hold mass.
    assert dist[0, 0].item() == pytest.approx(0.0)
    # Mass falls on slot 1 (first real event time grid slot).
    assert dist[0, 1].item() == pytest.approx(1.0)


def test_km_event_time_probabilities_normalize_primary_event_mass():
    records = [
        {"time_days": 10.0, "event": 1},
        {"time_days": 20.0, "event": 0},
        {"time_days": 30.0, "event": 1},
        {"time_days": 30.0, "event": 2},
    ]
    times, probs = _km_event_time_probabilities(records)

    assert times.tolist() == [10.0, 30.0]
    assert probs.sum().item() == pytest.approx(1.0)
    assert torch.all(probs > 0)


def test_km_event_time_probabilities_treat_censoring_as_risk_set_exit_only():
    with_censor, _ = _km_event_time_probabilities(
        [
            {"time_days": 10.0, "event": 1},
            {"time_days": 10.0, "event": 0},
            {"time_days": 20.0, "event": 1},
        ]
    )
    without_censor, _ = _km_event_time_probabilities(
        [
            {"time_days": 10.0, "event": 1},
            {"time_days": 20.0, "event": 1},
        ]
    )

    assert with_censor.tolist() == without_censor.tolist()


def test_multi_outcome_end_to_end():
    sorted_ets = {
        "mortality": torch.tensor([30.0, 90.0, 180.0, 365.0, 730.0]),
        "aki": torch.tensor([5.0, 10.0, 20.0, 30.0]),
    }
    loss_fn = MultiOutcomeSurvivalLoss(
        outcome_names=["mortality", "aki"],
        outcome_sorted_event_times=sorted_ets,
    )
    torch.manual_seed(0)
    emb = _norm(torch.randn(8, 16))
    outcome_survival = {
        "mortality": {
            "times": torch.tensor([30.0, 90.0, 180.0, 365.0, 730.0, 60.0, 0.0, 730.0]),
            "events": torch.tensor([1, 1, 0, 1, 0, 1, 0, 0]),
        },
        "aki": {
            "times": torch.tensor([5.0, 10.0, 20.0, 30.0, 5.0, 15.0, -1.0, -1.0]),
            "events": torch.tensor([1, 0, 1, 1, 1, 0, -1, -1]),
        },
    }
    result = loss_fn(emb, outcome_survival)
    assert "loss" in result
    assert torch.isfinite(result["loss"])
    assert result["loss"].requires_grad
    assert result["loss"].item() > 0
    for name in ["mortality", "aki"]:
        assert f"loss/{name}" in result
        assert f"loss_sigma_input/{name}" in result
        assert f"sigma/{name}" in result
        assert f"n_effective_pairs/{name}" in result
        assert f"effective_pair_fraction/{name}" in result


def test_missing_sorted_event_times_raises_valueerror():
    loss_fn = MultiOutcomeSurvivalLoss(
        outcome_names=["mortality"],
        outcome_sorted_event_times={},
    )
    emb = _norm(torch.randn(4, 8))
    outcome_survival = {
        "mortality": {
            "times": torch.tensor([10.0, 20.0, 30.0, 40.0]),
            "events": torch.tensor([1, 1, 0, 1]),
        }
    }
    with pytest.raises(ValueError, match="sorted_event_times"):
        loss_fn(emb, outcome_survival)


def test_gradient_flows_to_embeddings():
    sorted_et = {"mortality": torch.tensor([30.0, 90.0, 180.0, 365.0])}
    loss_fn = MultiOutcomeSurvivalLoss(
        outcome_names=["mortality"],
        outcome_sorted_event_times=sorted_et,
    )
    torch.manual_seed(0)
    raw = torch.randn(4, 8, requires_grad=True)
    emb = F.normalize(raw, dim=-1)
    outcome_survival = {
        "mortality": {
            "times": torch.tensor([30.0, 90.0, 180.0, 365.0]),
            "events": torch.tensor([1, 1, 1, 1]),
        }
    }
    result = loss_fn(emb, outcome_survival)
    result["loss"].backward()
    assert raw.grad is not None
    assert not torch.all(raw.grad == 0)


def test_compute_sorted_event_times_returns_sorted_float32(tmp_path):
    path = tmp_path / "mortality.parquet"
    pd.DataFrame(
        {
            "split": ["train", "train", "train", "tuning"],
            "event": [1, 0, 1, 1],
            "time_days": [30.0, 10.0, 5.0, 1.0],
        }
    ).to_parquet(path)

    result = compute_sorted_event_times({"mortality": {"path": str(path)}})

    assert list(result) == ["mortality"]
    assert result["mortality"].dtype == torch.float32
    assert result["mortality"].ndim == 1
    assert result["mortality"].tolist() == [5.0, 30.0]


def test_compute_sorted_event_times_excludes_competing_deaths(tmp_path):
    path = tmp_path / "tx_failure.parquet"
    pd.DataFrame(
        {
            "split": ["train", "train", "train", "train"],
            "event": [1, 0, 2, 1],
            "time_days": [30.0, 10.0, 15.0, 90.0],
        }
    ).to_parquet(path)

    result = compute_sorted_event_times({"tx_failure": {"path": str(path)}})

    assert result["tx_failure"].tolist() == [30.0, 90.0]


def test_competing_death_is_hard_negative_by_default():
    sorted_et = torch.tensor([30.0, 90.0, 180.0, 365.0])
    times = torch.tensor([90.0, 60.0])
    events = torch.tensor([1, 2])

    loss_fn = SurvivalSoftContrastiveLoss(km_time_scale=0.25)
    weights, n_eff = loss_fn._compute_pair_weights(times, events, sorted_et)

    event_free_weights, _ = loss_fn._compute_pair_weights(
        times,
        torch.tensor([1, 0]),
        sorted_et,
    )
    assert weights[0, 1].item() == pytest.approx(event_free_weights[0, 1].item())
    assert weights[1, 0].item() == pytest.approx(event_free_weights[1, 0].item())
    assert n_eff.item() > 0.0


def test_same_subject_pairs_are_excluded_from_contrastive_weights():
    sorted_et = torch.tensor([30.0, 90.0, 180.0, 365.0])
    times = torch.tensor([30.0, 30.0, 180.0])
    events = torch.tensor([1, 1, 0])
    subject_ids = torch.tensor([10, 10, 11])
    embeddings = F.normalize(torch.randn(3, 8), dim=-1)
    loss_fn = SurvivalSoftContrastiveLoss(km_time_scale=0.25)

    weights, _ = loss_fn._compute_pair_weights(
        times,
        events,
        sorted_et,
        subject_ids=subject_ids,
    )
    loss = loss_fn(
        embeddings,
        times,
        events,
        sorted_event_times=sorted_et,
        subject_ids=subject_ids,
    )

    assert weights[0, 1].item() == pytest.approx(0.0)
    assert weights[1, 0].item() == pytest.approx(0.0)
    assert torch.isfinite(loss)


def test_batch_containing_only_duplicate_subject_has_no_informative_pairs():
    sorted_et = torch.tensor([30.0, 90.0])
    times = torch.tensor([30.0, 30.0])
    events = torch.tensor([1, 1])
    subject_ids = torch.tensor([10, 10])
    embeddings = F.normalize(torch.randn(2, 8), dim=-1)
    loss_fn = SurvivalSoftContrastiveLoss(km_time_scale=0.25)

    loss, diagnostics = loss_fn(
        embeddings,
        times,
        events,
        sorted_event_times=sorted_et,
        subject_ids=subject_ids,
        return_diagnostics=True,
    )

    assert loss.item() == pytest.approx(0.0)
    assert diagnostics["n_effective_pairs"].item() == pytest.approx(0.0)


def test_competing_death_censor_mode_reproduces_conservative_zero_weight():
    sorted_et = torch.tensor([30.0, 90.0, 180.0, 365.0])
    times = torch.tensor([90.0, 60.0])
    events = torch.tensor([1, 2])

    loss_fn = SurvivalSoftContrastiveLoss(
        km_time_scale=0.25,
        competing_event_handling="censor",
        competing_event_weight=0.0,
    )
    weights, n_eff = loss_fn._compute_pair_weights(times, events, sorted_et)

    assert weights[0, 1].item() == pytest.approx(0.0)
    assert weights[1, 0].item() == pytest.approx(0.0)
    assert n_eff.item() == pytest.approx(0.0)


def test_competing_death_exclude_mode_always_removes_pairs():
    sorted_et = torch.tensor([30.0, 90.0, 180.0])
    times = torch.tensor([90.0, 60.0])
    events = torch.tensor([1, 2])
    loss_fn = SurvivalSoftContrastiveLoss(
        competing_event_handling="exclude",
        competing_event_weight=1.0,
    )

    weights, n_eff = loss_fn._compute_pair_weights(times, events, sorted_et)

    assert weights.sum().item() == pytest.approx(0.0)
    assert n_eff.item() == pytest.approx(0.0)


def test_competing_event_gamma_sensitivity_increases_primary_competing_weight():
    sorted_et = torch.tensor([30.0, 90.0, 180.0, 365.0])
    times = torch.tensor([90.0, 60.0])
    events = torch.tensor([1, 2])
    conservative = SurvivalSoftContrastiveLoss(
        km_time_scale=0.25,
        competing_event_handling="censor",
        competing_event_weight=0.0,
    )
    sensitivity = SurvivalSoftContrastiveLoss(
        km_time_scale=0.25,
        competing_event_handling="censor",
        competing_event_weight=0.5,
    )

    w0, _ = conservative._compute_pair_weights(times, events, sorted_et)
    w1, _ = sensitivity._compute_pair_weights(times, events, sorted_et)

    assert w0[0, 1].item() == pytest.approx(0.0)
    assert w1[0, 1].item() > w0[0, 1].item()


def test_competing_event_reliability_mode_is_explicitly_reserved():
    with pytest.raises(NotImplementedError, match="reliability"):
        SurvivalSoftContrastiveLoss(competing_event_handling="reliability")


def test_loss_with_all_three_event_types_is_finite():
    sorted_et = torch.tensor([30.0, 90.0, 180.0, 365.0])
    times = torch.tensor([90.0, 180.0, 60.0, 200.0, 30.0])
    events = torch.tensor([1, 1, 2, 0, 1])
    torch.manual_seed(7)
    emb = F.normalize(torch.randn(5, 8), dim=-1)

    loss_fn = SurvivalSoftContrastiveLoss(km_time_scale=0.25)
    result = loss_fn(emb, times, events, sorted_event_times=sorted_et)

    assert torch.isfinite(result)
    assert result.requires_grad
    assert result.item() >= 0


def test_compute_pooled_sorted_event_times_returns_sorted_float32(tmp_path):
    for cohort, times in {"a": [20.0, 5.0], "b": [10.0]}.items():
        outcome_dir = tmp_path / cohort / "outcomes"
        outcome_dir.mkdir(parents=True)
        pd.DataFrame({"subject_id": list(range(len(times)))}).to_csv(
            tmp_path / cohort / "population_full.csv", index=False
        )
        pd.DataFrame(
            {
                "subject_id": list(range(len(times))),
                "split": ["train"] * len(times),
                "event": [1] * len(times),
                "time_days": times,
            }
        ).to_parquet(outcome_dir / "mortality.parquet")

    result = compute_pooled_sorted_event_times(
        {
            "a": {"data_dir": str(tmp_path / "a")},
            "b": {"data_dir": str(tmp_path / "b")},
        },
        {"mortality": {"outcome_file": "mortality.parquet"}},
    )

    assert result["mortality"].dtype == torch.float32
    assert result["mortality"].ndim == 1
    assert result["mortality"].tolist() == [5.0, 10.0, 20.0]


def test_pooled_event_grid_strict_mode_rejects_missing_configured_cell(tmp_path):
    with pytest.raises(FileNotFoundError, match="aki_30d"):
        compute_pooled_event_time_probability_grids(
            {"dlbcl": {"data_dir": str(tmp_path / "dlbcl")}},
            {
                "aki_30d": {
                    "outcome_file": "aki_first_documented.parquet",
                    "n_hours_start_include": 1,
                    "n_hours_end_include": 720,
                }
            },
            require_all_configured_cells=True,
        )
