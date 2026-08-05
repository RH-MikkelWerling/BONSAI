"""Cross-outcome weighting for OPERA contrastive objectives.

The weighters in this module operate on one scalar loss per configured outcome.
Inactive outcomes are represented by non-finite entries and receive zero weight.
Pair construction and patient-level eligibility remain the responsibility of the
survival contrastive loss.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence

import torch
from torch import nn


class CrossOutcomeWeighter(nn.Module, ABC):
    """Interface for combining active per-outcome losses."""

    @abstractmethod
    def weights(self, per_outcome_losses: torch.Tensor) -> torch.Tensor:
        """Return a non-negative coefficient for each configured outcome."""

    def update(self, per_outcome_losses: torch.Tensor) -> None:
        """Update state from detached losses. Stateless weighters do nothing."""

    def regularizer(self, active_mask: torch.Tensor) -> torch.Tensor:
        """Return an optional additive regularizer over active outcomes."""
        return torch.zeros((), device=active_mask.device)


class UniformWeighter(CrossOutcomeWeighter):
    """Assign the same coefficient to every active outcome."""

    def weights(self, per_outcome_losses: torch.Tensor) -> torch.Tensor:
        active = torch.isfinite(per_outcome_losses)
        return active.to(dtype=per_outcome_losses.dtype)


class KendallWeighter(CrossOutcomeWeighter):
    """Homoscedastic uncertainty weighting from Kendall et al. (2018)."""

    def __init__(self, n_outcomes: int):
        super().__init__()
        self.log_sigma = nn.Parameter(torch.zeros(n_outcomes))

    def weights(self, per_outcome_losses: torch.Tensor) -> torch.Tensor:
        active = torch.isfinite(per_outcome_losses)
        precision = 0.5 * torch.exp(-2.0 * self.log_sigma)
        return torch.where(active, precision, torch.zeros_like(precision))

    def regularizer(self, active_mask: torch.Tensor) -> torch.Tensor:
        return self.log_sigma[active_mask].sum()


class FAMOWeighter(CrossOutcomeWeighter):
    """FAMO task weighting core without per-task model gradients.

    A complete FAMO integration requires task losses from the same batch before
    and after the shared model optimizer step. OPERA's automatic optimization
    path does not currently provide that lifecycle. This class remains
    available for isolated algorithm work, but the config factory rejects FAMO
    until a post-step hook is implemented and tested.

    FAMO tends to retain pressure on slowly improving and rare tasks. It is
    included as a pro-rare comparator for the confound panel, not as the
    production default.
    """

    def __init__(
        self,
        n_outcomes: int,
        learning_rate: float = 0.025,
        gamma: float = 0.001,
        beta1: float = 0.9,
        beta2: float = 0.999,
        epsilon: float = 1e-8,
    ):
        super().__init__()
        self.learning_rate = float(learning_rate)
        self.gamma = float(gamma)
        self.beta1 = float(beta1)
        self.beta2 = float(beta2)
        self.epsilon = float(epsilon)

        self.register_buffer("task_logits", torch.zeros(n_outcomes))
        self.register_buffer(
            "initial_losses",
            torch.full((n_outcomes,), float("nan")),
        )
        self.register_buffer(
            "previous_losses",
            torch.full((n_outcomes,), float("nan")),
        )
        self.register_buffer("adam_first_moment", torch.zeros(n_outcomes))
        self.register_buffer("adam_second_moment", torch.zeros(n_outcomes))
        self.register_buffer("update_step", torch.zeros((), dtype=torch.long))

    def _normalized_losses(
        self,
        per_outcome_losses: torch.Tensor,
        initialize: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        detached = per_outcome_losses.detach().to(self.task_logits.device)
        active = torch.isfinite(detached)
        if initialize:
            needs_initial = active & ~torch.isfinite(self.initial_losses)
            self.initial_losses[needs_initial] = detached[needs_initial].clamp_min(
                self.epsilon
            )
        denominator = torch.where(
            torch.isfinite(self.initial_losses),
            self.initial_losses,
            detached.clamp_min(self.epsilon),
        ).clamp_min(self.epsilon)
        normalized = detached.clamp_min(self.epsilon) / denominator
        normalized = torch.where(
            active,
            normalized,
            torch.full_like(normalized, float("nan")),
        )
        return normalized, active, denominator

    def weights(self, per_outcome_losses: torch.Tensor) -> torch.Tensor:
        normalized, active, initial = self._normalized_losses(
            per_outcome_losses,
            initialize=self.training,
        )
        result = torch.zeros_like(per_outcome_losses)
        n_active = int(active.sum().item())
        if n_active == 0:
            return result

        active_logits = self.task_logits[active]
        task_distribution = torch.softmax(active_logits, dim=0)
        inverse_gap = task_distribution / normalized[active].clamp_min(self.epsilon)
        normalization = inverse_gap.sum().clamp_min(self.epsilon)

        result[active] = (
            float(n_active)
            * inverse_gap.to(per_outcome_losses.device)
            / normalization.to(per_outcome_losses.device)
            / initial[active].to(per_outcome_losses.device)
        )

        if self.training:
            self.previous_losses[active] = normalized[active]
        return result

    @torch.no_grad()
    def update(self, per_outcome_losses: torch.Tensor) -> None:
        normalized, active, _ = self._normalized_losses(
            per_outcome_losses,
            initialize=True,
        )
        joint = active & torch.isfinite(self.previous_losses)
        if int(joint.sum().item()) < 2:
            return

        previous = self.previous_losses[joint].clamp_min(self.epsilon)
        current = normalized[joint].clamp_min(self.epsilon)
        relative_decrease = torch.log(previous) - torch.log(current)

        logits = self.task_logits[joint]
        distribution = torch.softmax(logits, dim=0)
        centered = relative_decrease - torch.sum(distribution * relative_decrease)
        gradient = distribution * centered + self.gamma * logits

        first = self.adam_first_moment[joint]
        second = self.adam_second_moment[joint]
        first = self.beta1 * first + (1.0 - self.beta1) * gradient
        second = self.beta2 * second + (1.0 - self.beta2) * gradient.square()
        self.update_step.add_(1)
        step = int(self.update_step.item())
        first_hat = first / (1.0 - self.beta1**step)
        second_hat = second / (1.0 - self.beta2**step)

        self.task_logits[joint] = logits - self.learning_rate * first_hat / (
            second_hat.sqrt() + self.epsilon
        )
        self.adam_first_moment[joint] = first
        self.adam_second_moment[joint] = second


def effective_number_outcome_factors(
    outcome_names: Sequence[str],
    class_counts: Mapping[str, Mapping[str, int | float]],
    *,
    beta: float = 0.9999,
    cap: float = 50.0,
) -> torch.Tensor:
    """Compute capped outcome factors from global positive and negative counts.

    Cui et al. (2019) define a class weight proportional to
    ``(1 - beta) / (1 - beta**n)``. OPERA has a pairwise scalar loss per
    outcome, so this function averages the two class weights and divides by the
    corresponding balanced-cohort value. The result is an outcome-level
    normalization, not a change to the pair geometry.
    """
    if not 0.0 <= beta < 1.0:
        raise ValueError("class_balanced_beta must be in [0, 1).")
    if cap < 1.0:
        raise ValueError("class_balanced_cap must be at least 1.")

    def class_weight(count: float) -> float:
        if count <= 0:
            raise ValueError("Class counts must be positive.")
        if beta == 0.0:
            return 1.0
        return (1.0 - beta) / (1.0 - beta**count)

    factors = []
    for name in outcome_names:
        if name not in class_counts:
            raise ValueError(
                f"Missing global positive and negative counts for outcome {name!r}."
            )
        counts = class_counts[name]
        positive = float(counts.get("positive", 0))
        negative = float(counts.get("negative", 0))
        total = positive + negative
        if positive <= 0 or negative <= 0:
            raise ValueError(
                f"Outcome {name!r} requires positive counts for both classes."
            )

        observed_weight = 0.5 * (class_weight(positive) + class_weight(negative))
        balanced_weight = class_weight(total / 2.0)
        factors.append(min(observed_weight / balanced_weight, cap))
    return torch.tensor(factors, dtype=torch.float32)


def build_cross_outcome_weighter(
    outcome_names: Sequence[str],
    config: Mapping[str, object] | None,
) -> tuple[CrossOutcomeWeighter, str, torch.Tensor]:
    """Build a weighter, aggregation mode, and fixed class-balance factors."""
    settings = dict(config or {})
    name = str(settings.get("weighter", "kendall")).lower()
    aggregation = str(settings.get("aggregation", "pooled")).lower()
    if aggregation not in {"macro", "pooled", "hierarchical_support"}:
        raise ValueError(
            "cross_outcome.aggregation must be one of: macro, pooled, "
            "hierarchical_support."
        )

    n_outcomes = len(outcome_names)
    if name == "uniform":
        weighter: CrossOutcomeWeighter = UniformWeighter()
    elif name == "kendall":
        weighter = KendallWeighter(n_outcomes)
    elif name == "famo":
        raise ValueError(
            "FAMO is not enabled in the training config path. A valid FAMO "
            "update requires same-batch losses after the shared optimizer step, "
            "which OperaContrastiveModule does not yet expose."
        )
    else:
        raise ValueError(
            "cross_outcome.weighter must be one of: uniform, kendall, famo."
        )

    factors = torch.ones(n_outcomes, dtype=torch.float32)
    if bool(settings.get("class_balanced", False)):
        counts = settings.get("class_counts")
        if not isinstance(counts, Mapping):
            raise ValueError(
                "cross_outcome.class_counts is required when class_balanced is true."
            )
        factors = effective_number_outcome_factors(
            outcome_names,
            counts,
            beta=float(settings.get("class_balanced_beta", 0.9999)),
            cap=float(settings.get("class_balanced_cap", 50.0)),
        )
    return weighter, aggregation, factors
