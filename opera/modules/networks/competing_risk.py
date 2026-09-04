"""Continuous-time piecewise-exponential competing-risk objective."""

from __future__ import annotations

import math
from bisect import bisect_left
from collections.abc import Mapping, Sequence

import torch
from torch import nn

from opera.modules.networks.cross_outcome_weighters import KendallWeighter


def validate_competing_risk_sampling(
    competing_risk_config: Mapping[str, object] | None,
    batch_sampling: Mapping[str, object] | None,
) -> None:
    """Reject outcome-enriched sampling for an unweighted absolute likelihood."""
    settings = dict(competing_risk_config or {})
    if float(settings.get("loss_weight", 0.0)) <= 0:
        return
    sampler_type = str(dict(batch_sampling or {}).get("type", "random")).lower()
    if sampler_type not in {"none", "random"}:
        raise ValueError(
            "The competing-risk likelihood requires natural-distribution random "
            "minibatches. Outcome-enriched sampling changes the fitted hazards. "
            "Use training.batch_sampling.type=random, disable the likelihood, "
            "or implement explicit inverse-sampling weights."
        )


def summarize_competing_risk_support(
    dataset,
    outcome_names: Sequence[str],
    interval_boundaries_days: Sequence[float],
) -> list[dict[str, int | float | str | None]]:
    """Count training observations by outcome, event type, and time interval."""
    boundaries = [float(value) for value in interval_boundaries_days]
    datasets = getattr(dataset, "datasets", [dataset])
    counts: dict[tuple[str, int], dict[str, int]] = {}
    for name in outcome_names:
        for interval in range(len(boundaries) + 1):
            counts[(name, interval)] = {
                "n_valid": 0,
                "n_target": 0,
                "n_death": 0,
                "n_censored": 0,
            }

    for child in datasets:
        outcome_dicts = getattr(child, "outcome_dicts", {})
        for name in outcome_names:
            for record in outcome_dicts.get(name, {}).values():
                time = record.get("time_days")
                event = record.get("event", record.get("label", -1))
                if time is None or event is None:
                    continue
                time = float(time)
                event = int(event)
                if time < 0 or event not in {0, 1, 2}:
                    continue
                interval = bisect_left(boundaries, time)
                row = counts[(name, interval)]
                row["n_valid"] += 1
                if event == 1:
                    row["n_target"] += 1
                elif event == 2:
                    row["n_death"] += 1
                else:
                    row["n_censored"] += 1

    rows = []
    for (name, interval), count in counts.items():
        rows.append(
            {
                "outcome": name,
                "interval": interval,
                "start_day": 0.0 if interval == 0 else boundaries[interval - 1],
                "end_day": (
                    boundaries[interval] if interval < len(boundaries) else None
                ),
                **count,
            }
        )
    return rows


def estimate_piecewise_null_log_hazards(
    dataset,
    outcome_names: Sequence[str],
    interval_boundaries_days: Sequence[float],
    *,
    no_competing_outcomes: Sequence[str] = ("overall_survival",),
    time_scale_days: float = 365.25,
    minimum_log_hazard: float = -12.0,
) -> dict[str, list[list[float]]]:
    """Fit train-only intercept hazards by events divided by person-time.

    The returned rates use the same ``time_scale_days`` unit as the model loss.
    They form a censoring- and competing-risk-aware null reference without
    looking at validation data or patient covariates.
    """
    boundaries = [float(value) for value in interval_boundaries_days]
    starts = [0.0, *boundaries]
    ends = [*boundaries, None]
    no_competing = set(no_competing_outcomes)
    datasets = getattr(dataset, "datasets", [dataset])
    result: dict[str, list[list[float]]] = {}
    for name in outcome_names:
        exposure_days = [0.0] * len(starts)
        events = [[0.0] * len(starts), [0.0] * len(starts)]
        for child in datasets:
            for record in getattr(child, "outcome_dicts", {}).get(name, {}).values():
                time = record.get("time_days")
                event = record.get("event", record.get("label", -1))
                if time is None or event is None:
                    continue
                time = float(time)
                event = int(event)
                if time < 0 or event not in {0, 1, 2}:
                    continue
                for interval, (start, end) in enumerate(zip(starts, ends)):
                    exposure_days[interval] += max(
                        0.0, min(time, end) - start if end is not None else time - start
                    )
                event_interval = bisect_left(boundaries, time)
                if event == 1:
                    events[0][event_interval] += 1.0
                elif event == 2 and name not in no_competing:
                    events[1][event_interval] += 1.0

        log_rates: list[list[float]] = [[], []]
        for cause in range(2):
            for interval, person_days in enumerate(exposure_days):
                if name in no_competing and cause == 1:
                    log_rate = minimum_log_hazard
                elif person_days <= 0 or events[cause][interval] <= 0:
                    log_rate = minimum_log_hazard
                else:
                    rate = events[cause][interval] / (person_days / time_scale_days)
                    log_rate = max(minimum_log_hazard, min(6.0, math.log(rate)))
                log_rates[cause].append(log_rate)
        result[name] = log_rates
    return result


class PiecewiseExponentialCompetingRiskLoss(nn.Module):
    """Exact-time competing-risk likelihood with locally constant hazards.

    Event codes are ``0`` for administrative censoring, ``1`` for the target
    event, and ``2`` for death before the target event. Times remain continuous:
    interval boundaries only determine where the hazard is allowed to change.
    """

    def __init__(
        self,
        outcome_names: Sequence[str],
        interval_boundaries_days: Sequence[float],
        *,
        no_competing_outcomes: Sequence[str] = ("overall_survival",),
        time_scale_days: float = 365.25,
        smoothness_weight: float = 0.0,
        cross_outcome_config: Mapping[str, object] | None = None,
        weighter: str = "uniform",
        null_log_hazards: Mapping[str, Sequence[Sequence[float]]] | None = None,
    ):
        super().__init__()
        boundaries = torch.as_tensor(interval_boundaries_days, dtype=torch.float32)
        if boundaries.ndim != 1:
            raise ValueError("interval_boundaries_days must be one-dimensional.")
        if boundaries.numel() and (
            bool((boundaries <= 0).any())
            or bool((boundaries[1:] <= boundaries[:-1]).any())
        ):
            raise ValueError(
                "interval_boundaries_days must be strictly increasing and positive."
            )
        if time_scale_days <= 0:
            raise ValueError("time_scale_days must be positive.")
        if smoothness_weight < 0:
            raise ValueError("smoothness_weight must be non-negative.")

        self.outcome_names = list(outcome_names)
        self.outcome_index = {
            name: index for index, name in enumerate(self.outcome_names)
        }
        unknown = set(no_competing_outcomes) - set(self.outcome_names)
        if unknown:
            raise ValueError(
                "no_competing_outcomes contains unknown outcomes: "
                + ", ".join(sorted(unknown))
            )
        self.no_competing_outcomes = set(no_competing_outcomes)
        self.time_scale_days = float(time_scale_days)
        self.smoothness_weight = float(smoothness_weight)
        self.weighter_name = str(weighter).lower()
        if self.weighter_name not in {"uniform", "kendall", "kendall_null"}:
            raise ValueError(
                "competing_risk.weighter must be 'uniform', 'kendall', or "
                "'kendall_null'."
            )
        self.weighter = (
            KendallWeighter(len(self.outcome_names))
            if self.weighter_name in {"kendall", "kendall_null"}
            else None
        )
        settings = dict(cross_outcome_config or {})
        self.aggregation = str(settings.get("aggregation", "macro"))
        self.outcome_family = {
            outcome: family
            for family, members in dict(settings.get("outcome_families", {})).items()
            for outcome in members
        }
        self.family_weights = dict(settings.get("family_weights", {}))
        self.support_tau_locations = float(settings.get("support_tau_locations", 100.0))
        self.event_location_counts = dict(settings.get("event_location_counts", {}))
        curriculum = dict(settings.get("curriculum", {}))
        self.curriculum_enabled = bool(curriculum.get("enabled", False))
        self.curriculum_n_tiers = int(curriculum.get("n_support_tiers", 3))
        self.curriculum_warmup_epochs = int(curriculum.get("warmup_epochs", 0))
        self.curriculum_ramp_epochs = int(curriculum.get("ramp_epochs", 0))
        self.curriculum_stage_epochs = int(
            curriculum.get("stage_epochs", self.curriculum_warmup_epochs)
        )
        if self.curriculum_n_tiers < 1:
            raise ValueError("n_support_tiers must be positive.")
        if min(
            self.curriculum_warmup_epochs,
            self.curriculum_ramp_epochs,
            self.curriculum_stage_epochs,
        ) < 0:
            raise ValueError("Curriculum epoch counts must be non-negative.")
        ranked = sorted(
            self.outcome_names,
            key=lambda name: (-float(self.event_location_counts.get(name, 0.0)), name),
        )
        self.curriculum_tier = {
            name: min(
                self.curriculum_n_tiers - 1,
                index * self.curriculum_n_tiers // max(1, len(ranked)),
            )
            for index, name in enumerate(ranked)
        }
        self._curriculum_epoch = 0
        self._curriculum_training = False
        self.register_buffer(
            "boundaries",
            boundaries / self.time_scale_days,
            persistent=True,
        )
        self.register_buffer(
            "null_log_hazards",
            torch.full((len(self.outcome_names), 2, self.n_intervals), float("nan")),
            persistent=True,
        )
        if null_log_hazards is not None:
            self.set_null_log_hazards(null_log_hazards)

    def set_null_log_hazards(
        self, values: Mapping[str, Sequence[Sequence[float]]]
    ) -> None:
        """Install train-only intercept hazards used by ``kendall_null``."""
        tensor = self.null_log_hazards.detach().clone()
        for name, value in values.items():
            if name not in self.outcome_index:
                continue
            candidate = torch.as_tensor(value, dtype=tensor.dtype, device=tensor.device)
            if tuple(candidate.shape) != (2, self.n_intervals):
                raise ValueError(
                    f"Null hazards for {name!r} must have shape "
                    f"(2, {self.n_intervals}); got {tuple(candidate.shape)}."
                )
            tensor[self.outcome_index[name]] = candidate
        self.null_log_hazards.copy_(tensor)

    @property
    def n_intervals(self) -> int:
        return int(self.boundaries.numel()) + 1

    def set_curriculum_state(self, epoch: int, *, training: bool) -> None:
        """Set curriculum state explicitly for gradient-cached training."""
        self._curriculum_epoch = max(0, int(epoch))
        self._curriculum_training = bool(training)

    def _curriculum_outcome_multiplier(self, outcome: str) -> float:
        """Stage outcomes by empirical training support, never clinical labels."""
        if not self.curriculum_enabled or not self._curriculum_training:
            return 1.0
        tier = self.curriculum_tier[outcome]
        if tier == 0:
            return 1.0
        start = self.curriculum_warmup_epochs + (tier - 1) * self.curriculum_stage_epochs
        if self._curriculum_epoch < start:
            return 0.0
        if self.curriculum_ramp_epochs == 0:
            return 1.0
        return min(
            1.0,
            float(self._curriculum_epoch - start + 1)
            / float(self.curriculum_ramp_epochs),
        )

    def _exposure(self, times: torch.Tensor) -> torch.Tensor:
        """Return exact time at risk in every interval, in scaled time units."""
        scaled = times / self.time_scale_days
        starts = torch.cat([scaled.new_zeros(1), self.boundaries.to(scaled)])
        if self.boundaries.numel():
            widths = self.boundaries[1:] - self.boundaries[:-1]
            widths = torch.cat([self.boundaries[:1], widths]).to(scaled)
            finite = torch.clamp(
                scaled.unsqueeze(1) - starts[:-1].unsqueeze(0),
                min=0.0,
            )
            finite = torch.minimum(finite, widths.unsqueeze(0))
        else:
            finite = scaled.new_empty((scaled.numel(), 0))
        tail = torch.clamp(scaled - starts[-1], min=0.0).unsqueeze(1)
        return torch.cat([finite, tail], dim=1)

    def forward(
        self,
        log_hazards: torch.Tensor,
        outcome_survival: Mapping[str, Mapping[str, torch.Tensor]],
        *,
        return_per_outcome: bool = False,
        _skip_aggregation: bool = False,
    ) -> (
        tuple[torch.Tensor, dict[str, torch.Tensor]]
        | tuple[
            torch.Tensor,
            dict[str, torch.Tensor],
            dict[str, dict[str, torch.Tensor]],
        ]
    ):
        """Return macro-average likelihood and per-outcome diagnostics.

        ``log_hazards`` has shape ``(batch, outcomes, 2, intervals)``.
        Cause zero is the target event and cause one is competing death.
        """
        expected = (
            log_hazards.shape[0],
            len(self.outcome_names),
            2,
            self.n_intervals,
        )
        if tuple(log_hazards.shape) != expected:
            raise ValueError(
                f"log_hazards must have shape {expected}; "
                f"got {tuple(log_hazards.shape)}."
            )

        # Clamping the log-rate is a numerical guard only. Rates are per
        # ``time_scale_days`` and remain strictly positive.
        safe_log_hazards = log_hazards.clamp(min=-12.0, max=6.0)
        hazards = torch.exp(safe_log_hazards)
        losses: list[torch.Tensor] = []
        loss_names: list[str] = []
        diagnostics: dict[str, torch.Tensor] = {}
        per_outcome: dict[str, dict[str, torch.Tensor]] = {}
        null_reference_losses: dict[str, torch.Tensor] = {}

        for name, outcome_index in self.outcome_index.items():
            survival = outcome_survival.get(name)
            if not survival:
                continue
            times = survival.get("times")
            events = survival.get("events")
            if times is None or events is None:
                continue
            times = times.to(log_hazards).reshape(-1)
            events = events.to(device=log_hazards.device, dtype=torch.long).reshape(-1)
            valid = torch.isfinite(times) & (times >= 0) & (events >= 0)
            if not valid.any():
                continue

            times = times[valid]
            events = events[valid]
            if name in self.no_competing_outcomes and bool((events == 2).any()):
                raise ValueError(
                    f"Outcome {name!r} is configured without a competing cause "
                    "but contains event=2 observations."
                )

            outcome_hazards = hazards[valid, outcome_index]
            if name in self.no_competing_outcomes:
                total_hazard = outcome_hazards[:, 0]
            else:
                total_hazard = outcome_hazards.sum(dim=1)

            exposure = self._exposure(times)
            cumulative_hazard = (exposure * total_hazard).sum(dim=1)
            interval = torch.bucketize(times / self.time_scale_days, self.boundaries)
            row = torch.arange(times.numel(), device=times.device)
            target = events == 1
            competing = events == 2
            active_log_hazards = safe_log_hazards[valid, outcome_index]
            target_log_hazard = active_log_hazards[row, 0, interval]
            competing_log_hazard = active_log_hazards[row, 1, interval]
            # Components share the same valid-observation denominator, so they
            # sum exactly to the unsmoothed full likelihood.  This is important
            # for gradient diagnostics: event-conditional means would silently
            # rescale rare outcomes and manufacture apparent alignment.
            exposure_loss = cumulative_hazard.mean()
            primary_event_loss = -(
                target.to(cumulative_hazard.dtype)
                * target_log_hazard.to(cumulative_hazard.dtype)
            ).mean()
            competing_event_loss = -(
                competing.to(cumulative_hazard.dtype)
                * competing_log_hazard.to(cumulative_hazard.dtype)
            ).mean()
            likelihood_loss = (
                exposure_loss + primary_event_loss + competing_event_loss
            )
            # A continuous density's NLL depends on its time unit. Hazards are
            # parameterized per ``time_scale_days`` (normally per year), which
            # can make common early-event NLLs negative and invalid as inputs
            # to Kendall weighting. Convert event-density terms to per-day NLL
            # for cross-task aggregation. This adds only the exact Jacobian
            # constant and does not alter fitted hazards or their gradients.
            event_fraction = (target | competing).to(likelihood_loss.dtype).mean()
            unit_adjustment = event_fraction * math.log(self.time_scale_days)
            aggregation_loss = likelihood_loss + unit_adjustment
            if self.weighter_name == "kendall_null" and not _skip_aggregation:
                reference = self.null_log_hazards[outcome_index].to(log_hazards)
                if not torch.isfinite(reference).all():
                    raise RuntimeError(
                        "kendall_null requires train-only null hazards; "
                        f"none were installed for {name!r}."
                    )
                reference_rates = reference.exp()
                reference_total = (
                    reference_rates[0]
                    if name in self.no_competing_outcomes
                    else reference_rates.sum(dim=0)
                )
                null_cumulative = (exposure * reference_total.unsqueeze(0)).sum(dim=1)
                null_loss = (
                    null_cumulative
                    - target.to(null_cumulative.dtype) * reference[0, interval]
                    - competing.to(null_cumulative.dtype) * reference[1, interval]
                ).mean() + unit_adjustment
                if not torch.isfinite(null_loss) or float(null_loss) <= 0.0:
                    raise RuntimeError(
                        f"Training-null likelihood for {name!r} must be positive; "
                        f"got {float(null_loss):.6g}. Consider a finer time unit "
                        "or inspect this outcome's training support."
                    )
                null_reference_losses[name] = null_loss.detach()
            outcome_loss = (
                aggregation_loss if self.weighter is not None else likelihood_loss
            )
            smoothness_loss = outcome_loss * 0.0
            if self.smoothness_weight and self.n_intervals > 1:
                active_log_hazards = log_hazards[valid, outcome_index]
                if name in self.no_competing_outcomes:
                    active_log_hazards = active_log_hazards[:, :1]
                smoothness = active_log_hazards.diff(dim=-1).square().mean()
                smoothness_loss = self.smoothness_weight * smoothness
                outcome_loss = outcome_loss + smoothness_loss
                diagnostics[f"cr/smoothness/{name}"] = smoothness.detach()

            losses.append(outcome_loss)
            loss_names.append(name)
            per_outcome[name] = {
                "loss": outcome_loss,
                "full_likelihood_loss": likelihood_loss,
                "aggregation_loss": aggregation_loss,
                "exposure_loss": exposure_loss,
                "primary_event_loss": primary_event_loss,
                "competing_event_loss": competing_event_loss,
                "smoothness_loss": smoothness_loss,
                "valid_mask": valid,
                "target_mask": target,
                "competing_mask": competing,
            }
            diagnostics[f"cr/loss/{name}"] = outcome_loss.detach()
            diagnostics[f"cr/n_valid/{name}"] = valid.sum().float().detach()
            diagnostics[f"cr/n_target/{name}"] = target.sum().float().detach()
            diagnostics[f"cr/n_death/{name}"] = competing.sum().float().detach()

        if not losses:
            zero = log_hazards.sum() * 0.0
            diagnostics["cr/n_active_outcomes"] = zero.detach()
            if return_per_outcome:
                return zero, diagnostics, per_outcome
            return zero, diagnostics

        if _skip_aggregation:
            total = torch.stack(losses).mean()
            if return_per_outcome:
                return total, diagnostics, per_outcome
            return total, diagnostics

        if self.weighter is not None:
            if self.aggregation != "macro":
                raise ValueError(
                    "Kendall survival weighting currently requires "
                    "cross_outcome.aggregation=macro."
                )
            active_losses = torch.stack(losses)
            if self.weighter_name == "kendall_null":
                normalized = []
                for name, loss in zip(loss_names, losses):
                    null_loss = null_reference_losses[name]
                    normalized.append(loss / null_loss.clamp_min(1e-8))
                    diagnostics[f"cr/null_loss/{name}"] = null_loss
                    diagnostics[f"cr/loss_over_null/{name}"] = normalized[-1].detach()
                active_losses = torch.stack(normalized)
            per_outcome_losses = active_losses.new_full(
                (len(self.outcome_names),), float("nan")
            )
            active_indices = torch.as_tensor(
                [self.outcome_index[name] for name in loss_names],
                device=active_losses.device,
            )
            per_outcome_losses[active_indices] = active_losses
            active = torch.isfinite(per_outcome_losses)
            weights = self.weighter.weights(per_outcome_losses)
            finite_losses = torch.where(
                active, per_outcome_losses, torch.zeros_like(per_outcome_losses)
            )
            total = torch.sum(weights * finite_losses) + self.weighter.regularizer(active)
            for name in loss_names:
                index = self.outcome_index[name]
                diagnostics[f"cr/outcome_weight/{name}"] = weights[index].detach()
                diagnostics[f"cr/log_sigma/{name}"] = self.weighter.log_sigma[
                    self.outcome_index[name]
                ].detach()
            diagnostics["cr/outcome_weight_mean"] = weights[active].mean().detach()
            diagnostics["cr/log_sigma_mean"] = self.weighter.log_sigma[active].mean().detach()
            if self.weighter_name == "kendall_null":
                diagnostics["cr/loss_over_null_mean"] = active_losses.mean().detach()
        elif self.aggregation == "hierarchical_support":
            outcome_multipliers = {
                name: self._curriculum_outcome_multiplier(name) for name in loss_names
            }
            active_families = sorted(
                {
                    self.outcome_family[name]
                    for name in loss_names
                    if outcome_multipliers[name] > 0.0
                }
            )
            if not active_families:
                raise RuntimeError("Curriculum disabled every active outcome family.")
            family_denominator = sum(
                float(self.family_weights.get(family, 1.0))
                for family in active_families
            )
            total = losses[0] * 0.0
            for family in active_families:
                indices = [
                    i
                    for i, name in enumerate(loss_names)
                    if self.outcome_family[name] == family
                    and outcome_multipliers[name] > 0.0
                ]
                support = losses[0].new_tensor([
                    ((float(self.event_location_counts.get(loss_names[i], 0.0)) /
                      (float(self.event_location_counts.get(loss_names[i], 0.0)) + self.support_tau_locations)) ** 0.5)
                    * outcome_multipliers[loss_names[i]]
                    for i in indices
                ])
                if float(support.sum().item()) <= 0:
                    support = torch.ones_like(support)
                support = support / support.sum()
                family_weight = (
                    float(self.family_weights.get(family, 1.0))
                    / family_denominator
                )
                family_loss = torch.sum(
                    support * torch.stack([losses[i] for i in indices])
                )
                total = total + family_weight * family_loss
                family_key = family.lower().replace(" ", "_").replace("&", "and")
                diagnostics[f"cr/family_loss/{family_key}"] = family_loss.detach()
                diagnostics[f"cr/family_weight/{family_key}"] = losses[0].new_tensor(
                    family_weight
                )
            for tier in range(self.curriculum_n_tiers):
                tier_values = [
                    outcome_multipliers[name]
                    for name in loss_names
                    if self.curriculum_tier[name] == tier
                ]
                if tier_values:
                    diagnostics[f"cr/curriculum/tier_{tier}_multiplier"] = losses[
                        0
                    ].new_tensor(sum(tier_values) / len(tier_values))
        else:
            total = torch.stack(losses).mean()
        diagnostics["cr/n_active_outcomes"] = torch.tensor(
            float(len(losses)),
            device=log_hazards.device,
        )
        diagnostics["cr/loss"] = total.detach()
        if return_per_outcome:
            return total, diagnostics, per_outcome
        return total, diagnostics
