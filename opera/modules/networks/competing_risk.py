"""Continuous-time piecewise-exponential competing-risk objective."""

from __future__ import annotations

import math
from bisect import bisect_left
from collections.abc import Mapping, Sequence

import torch
from torch import nn

from opera.modules.networks.cross_outcome_weighters import KendallWeighter


def piecewise_exponential_cif(
    log_hazards: torch.Tensor,
    interval_boundaries_days: Sequence[float] | torch.Tensor,
    horizons_days: Sequence[float] | torch.Tensor,
    *,
    time_scale_days: float = 365.25,
    no_competing_outcome_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Convert piecewise cause-specific hazards to target-event CIFs.

    ``log_hazards`` is ``(batch, outcomes, 2, intervals)``.  The returned
    tensor is ``(batch, outcomes, horizons)`` and integrates the target hazard
    while reducing event-free survival by both target and competing hazards.
    For outcomes without a competing cause, pass a boolean outcome mask; their
    second hazard is then excluded from both survival and incidence.
    """
    if log_hazards.ndim != 4 or log_hazards.shape[2] != 2:
        raise ValueError(
            "log_hazards must have shape (batch, outcomes, 2, intervals)."
        )
    if time_scale_days <= 0:
        raise ValueError("time_scale_days must be positive.")
    boundaries = torch.as_tensor(
        interval_boundaries_days,
        dtype=log_hazards.dtype,
        device=log_hazards.device,
    )
    horizons = torch.as_tensor(
        horizons_days,
        dtype=log_hazards.dtype,
        device=log_hazards.device,
    )
    if boundaries.ndim != 1 or horizons.ndim != 1:
        raise ValueError("Boundaries and horizons must be one-dimensional.")
    if boundaries.numel() + 1 != log_hazards.shape[-1]:
        raise ValueError("Hazard interval count does not match the boundaries.")
    if boundaries.numel() and (
        bool((boundaries <= 0).any())
        or bool((boundaries[1:] <= boundaries[:-1]).any())
    ):
        raise ValueError("Boundaries must be positive and strictly increasing.")
    if horizons.numel() and bool((horizons <= 0).any()):
        raise ValueError("Horizons must be positive.")

    rates = log_hazards.float().clamp(min=-12.0, max=6.0).exp()
    target_rates = rates[:, :, 0]
    competing_rates = rates[:, :, 1]
    if no_competing_outcome_mask is not None:
        mask = torch.as_tensor(
            no_competing_outcome_mask,
            dtype=torch.bool,
            device=log_hazards.device,
        )
        if tuple(mask.shape) != (log_hazards.shape[1],):
            raise ValueError("no_competing_outcome_mask must have shape (outcomes,).")
        competing_rates = competing_rates.masked_fill(mask.view(1, -1, 1), 0.0)
    total_rates = target_rates + competing_rates

    starts = torch.cat([boundaries.new_zeros(1), boundaries]) / time_scale_days
    outputs = []
    for horizon in horizons / time_scale_days:
        survival = torch.ones_like(target_rates[:, :, 0])
        incidence = torch.zeros_like(survival)
        for interval in range(log_hazards.shape[-1]):
            if interval + 1 < len(starts):
                stop = starts[interval + 1]
                duration = torch.clamp(
                    torch.minimum(horizon, stop) - starts[interval], min=0.0
                )
            else:
                duration = torch.clamp(horizon - starts[interval], min=0.0)
            total = total_rates[:, :, interval]
            target = target_rates[:, :, interval]
            event_probability = -torch.expm1(-total * duration)
            target_share = torch.where(total > 0, target / total, torch.zeros_like(total))
            incidence = incidence + survival * target_share * event_probability
            survival = survival * torch.exp(-total * duration)
        outputs.append(incidence)
    if not outputs:
        return log_hazards.new_empty((*log_hazards.shape[:2], 0), dtype=torch.float32)
    return torch.stack(outputs, dim=-1).to(log_hazards.dtype)


def validate_competing_risk_sampling(
    competing_risk_config: Mapping[str, object] | None,
    batch_sampling: Mapping[str, object] | None,
) -> None:
    """Reject outcome-enriched sampling for an unweighted absolute likelihood."""
    settings = dict(competing_risk_config or {})
    if float(settings.get("loss_weight", 0.0)) <= 0:
        return
    sampler_type = str(dict(batch_sampling or {}).get("type", "random")).lower()
    if sampler_type == "ipw_event_horizon_bce":
        if str(settings.get("objective", "piecewise_exponential")) != "horizon_bce":
            raise ValueError("IPW event sampling requires horizon_bce")
        if float(settings.get("contrastive_loss_weight", 0.0)) != 0.0:
            raise ValueError("IPW event sampling requires direct BCE only")
        return
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
    event_pseudocount: float = 0.0,
) -> dict[str, list[list[float]]]:
    """Fit train-only intercept hazards by events divided by person-time.

    The returned rates use the same ``time_scale_days`` unit as the model loss.
    They form a censoring- and competing-risk-aware null reference without
    looking at validation data or patient covariates.
    """
    boundaries = [float(value) for value in interval_boundaries_days]
    if event_pseudocount < 0:
        raise ValueError("event_pseudocount must be non-negative.")
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
                elif person_days <= 0 or (
                    events[cause][interval] <= 0 and event_pseudocount == 0
                ):
                    log_rate = minimum_log_hazard
                else:
                    rate = (events[cause][interval] + event_pseudocount) / (
                        person_days / time_scale_days
                    )
                    log_rate = max(minimum_log_hazard, min(6.0, math.log(rate)))
                log_rates[cause].append(log_rate)
        result[name] = log_rates
    return result


class PiecewiseExponentialCompetingRiskLoss(nn.Module):
    """Competing-risk objectives over locally constant cause-specific hazards.

    Event codes are ``0`` for administrative censoring, ``1`` for the target
    event, and ``2`` for death before the target event. Times remain continuous:
    interval boundaries determine where the hazard is allowed to change.

    ``objective="piecewise_exponential"`` fits the exact-time likelihood.
    ``objective="horizon_bce"`` converts the same hazards to cumulative
    incidence at every configured boundary and optimizes all known binary
    horizon labels jointly. The latter is an objective ablation: architecture,
    competing-event semantics, monotonic CIF construction, and evaluation all
    remain unchanged.
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
        objective: str = "piecewise_exponential",
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
        self.objective = str(objective).lower()
        self.require_sampling_weights = False
        if self.objective not in {"piecewise_exponential", "horizon_bce"}:
            raise ValueError(
                "competing_risk.objective must be 'piecewise_exponential' or "
                "'horizon_bce'."
            )
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
        self.endpoint_groups = dict(settings.get("endpoint_groups", {}))
        if self.aggregation == "endpoint_group_macro":
            from opera.functional.endpoint_groups import validate_endpoint_groups
            validate_endpoint_groups(self.endpoint_groups, self.outcome_names)
            if self.objective != "horizon_bce" or self.weighter_name != "uniform":
                raise ValueError("endpoint_group_macro requires uniform horizon BCE.")
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

    def _horizon_targets(
        self,
        times: torch.Tensor,
        events: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return cumulative binary targets and censor-aware observation mask."""
        horizons = self.boundaries.to(times) * self.time_scale_days
        observed_time = times.unsqueeze(1)
        event_code = events.unsqueeze(1)
        positive = (event_code == 1) & (observed_time <= horizons)
        # Status is known when the primary event occurred, follow-up reaches the
        # horizon, or a competing death permanently rules out the primary event.
        known = (
            positive
            | (observed_time >= horizons)
            | ((event_code == 2) & (observed_time <= horizons))
        )
        return positive.to(dtype=times.dtype), known

    def _horizon_cif(
        self,
        log_hazards: torch.Tensor,
        outcome_name: str,
    ) -> torch.Tensor:
        """Return target-event CIF at every configured finite boundary."""
        no_competing = torch.tensor(
            [outcome_name in self.no_competing_outcomes],
            dtype=torch.bool,
            device=log_hazards.device,
        )
        return piecewise_exponential_cif(
            log_hazards.unsqueeze(1),
            self.boundaries * self.time_scale_days,
            self.boundaries * self.time_scale_days,
            time_scale_days=self.time_scale_days,
            no_competing_outcome_mask=no_competing,
        )[:, 0].float()

    @staticmethod
    def _horizon_bce_cells(
        cif: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """Return per-cell BCE for CIF probabilities.

        ``F.binary_cross_entropy`` is banned inside an autocast region, and the
        CIF is a probability rather than a logit, so the logit round-trip that
        ``binary_cross_entropy_with_logits`` would need is avoided by writing
        the loss out directly in float32.
        """
        probabilities = cif.float().clamp(min=1e-6, max=1.0 - 1e-6)
        labels = targets.to(probabilities)
        return -(
            labels * probabilities.log() + (1.0 - labels) * torch.log1p(-probabilities)
        )

    @classmethod
    def _masked_horizon_bce(
        cls,
        cif: torch.Tensor,
        targets: torch.Tensor,
        known: torch.Tensor,
        weights: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if not bool(known.any()):
            raise ValueError("Horizon BCE requires at least one known horizon label.")
        cells = cls._horizon_bce_cells(cif, targets)
        if weights is None:
            return cells[known].mean()
        weighted_known = known.to(cells) * weights[:, None]
        return (cells * weighted_known).sum() / weighted_known.sum()

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
            weights = survival.get("weights")
            if self.require_sampling_weights and self._curriculum_training and weights is None:
                raise ValueError("IPW sampling requires per-patient weights in training")
            if weights is not None:
                if self.objective != "horizon_bce":
                    raise ValueError("Sampling weights are supported only for horizon BCE")
                weights = weights.to(device=log_hazards.device, dtype=torch.float32).reshape(-1)
                if weights.numel() != times.numel() or not bool(
                    (torch.isfinite(weights) & (weights > 0)).all()
                ):
                    raise ValueError("Sampling weights must be finite, positive, and batch-aligned")
            valid = torch.isfinite(times) & (times >= 0) & (events >= 0)
            if not valid.any():
                continue

            times = times[valid]
            events = events[valid]
            if weights is not None:
                weights = weights[valid]
            if name in self.no_competing_outcomes and bool((events == 2).any()):
                raise ValueError(
                    f"Outcome {name!r} is configured without a competing cause "
                    "but contains event=2 observations."
                )

            target = events == 1
            competing = events == 2
            active_log_hazards = safe_log_hazards[valid, outcome_index]
            horizon_targets = None
            horizon_known = None
            if self.objective == "horizon_bce":
                horizon_targets, horizon_known = self._horizon_targets(times, events)
                if not bool(horizon_known.any()):
                    continue
                cif = self._horizon_cif(active_log_hazards, name)
                if self.aggregation == "endpoint_group_macro":
                    cells = self._horizon_bce_cells(cif, horizon_targets)
                    if weights is None:
                        counts = horizon_known.sum(dim=0)
                        likelihood_loss = (
                            (cells * horizon_known).sum(dim=0)[counts > 0]
                            / counts[counts > 0]
                        ).mean()
                    else:
                        weighted_known = horizon_known.to(cells) * weights[:, None]
                        counts = weighted_known.sum(dim=0)
                        likelihood_loss = (
                            (cells * weighted_known).sum(dim=0)[counts > 0]
                            / counts[counts > 0]
                        ).mean()
                else:
                    likelihood_loss = self._masked_horizon_bce(
                        cif, horizon_targets, horizon_known, weights
                    )
                # Retain the component contract used by gradient diagnostics.
                # Horizon BCE is one indivisible component rather than an
                # exposure term plus event-density terms.
                exposure_loss = likelihood_loss * 0.0
                primary_event_loss = likelihood_loss
                competing_event_loss = likelihood_loss * 0.0
                aggregation_loss = likelihood_loss
                cell_loss = self._horizon_bce_cells(cif, horizon_targets)
                for horizon_index, horizon in enumerate(
                    self.boundaries * self.time_scale_days
                ):
                    known_at_horizon = horizon_known[:, horizon_index]
                    suffix = f"{int(round(float(horizon)))}d"
                    diagnostics[f"cr/horizon_bce/n_known/{name}/{suffix}"] = (
                        known_at_horizon.sum().float().detach()
                    )
                    if bool(known_at_horizon.any()):
                        horizon_cells = cell_loss[known_at_horizon, horizon_index]
                        if weights is None:
                            horizon_loss = horizon_cells.mean()
                        else:
                            horizon_weights = weights[known_at_horizon]
                            horizon_loss = (
                                (horizon_cells * horizon_weights).sum()
                                / horizon_weights.sum()
                            )
                        diagnostics[f"cr/horizon_bce/loss/{name}/{suffix}"] = horizon_loss.detach()
            else:
                outcome_hazards = hazards[valid, outcome_index]
                if name in self.no_competing_outcomes:
                    total_hazard = outcome_hazards[:, 0]
                else:
                    total_hazard = outcome_hazards.sum(dim=1)

                exposure = self._exposure(times)
                cumulative_hazard = (exposure * total_hazard).sum(dim=1)
                interval = torch.bucketize(times / self.time_scale_days, self.boundaries)
                row = torch.arange(times.numel(), device=times.device)
                target_log_hazard = active_log_hazards[row, 0, interval]
                competing_log_hazard = active_log_hazards[row, 1, interval]
                # Components share the same valid-observation denominator, so
                # they sum exactly to the unsmoothed full likelihood. Event-
                # conditional means would silently rescale rare outcomes.
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
                # Convert annual event-density terms to per-day NLL for stable
                # cross-task Kendall aggregation. This is a constant Jacobian.
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
                if self.objective == "horizon_bce":
                    reference_cif = self._horizon_cif(
                        reference.unsqueeze(0).expand(times.numel(), -1, -1),
                        name,
                    )
                    null_loss = self._masked_horizon_bce(
                        reference_cif, horizon_targets, horizon_known
                    )
                else:
                    reference_rates = reference.exp()
                    reference_total = (
                        reference_rates[0]
                        if name in self.no_competing_outcomes
                        else reference_rates.sum(dim=0)
                    )
                    null_cumulative = (
                        exposure * reference_total.unsqueeze(0)
                    ).sum(dim=1)
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
            uses_learned_weighter = self.weighter_name in {
                "kendall",
                "kendall_null",
            }
            outcome_loss = aggregation_loss if uses_learned_weighter else likelihood_loss
            smoothness_loss = outcome_loss * 0.0
            if self.smoothness_weight and self.n_intervals > 1:
                active_log_hazards = log_hazards[valid, outcome_index]
                if name in self.no_competing_outcomes:
                    active_log_hazards = active_log_hazards[:, :1]
                if weights is None:
                    smoothness = active_log_hazards.diff(dim=-1).square().mean()
                else:
                    smoothness_per_patient = (
                        active_log_hazards.diff(dim=-1).square().mean(dim=(1, 2))
                    )
                    smoothness = (
                        (smoothness_per_patient * weights).sum() / weights.sum()
                    )
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
                "horizon_bce_loss": (
                    likelihood_loss
                    if self.objective == "horizon_bce"
                    else likelihood_loss * 0.0
                ),
                "smoothness_loss": smoothness_loss,
                "valid_mask": valid,
                "target_mask": target,
                "competing_mask": competing,
                "horizon_known_mask": horizon_known,
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

        if self.weighter_name in {"kendall", "kendall_null"}:
            if self.weighter is None:
                raise RuntimeError(
                    "Learned competing-risk weighting was selected without a "
                    "configured weighter module."
                )
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
            curriculum_multipliers = per_outcome_losses.new_tensor(
                [
                    self._curriculum_outcome_multiplier(name)
                    for name in self.outcome_names
                ]
            )
            active = torch.isfinite(per_outcome_losses) & (
                curriculum_multipliers > 0
            )
            weighting_input = torch.where(
                active,
                per_outcome_losses,
                torch.full_like(per_outcome_losses, float("nan")),
            )
            weights = self.weighter.weights(weighting_input) * curriculum_multipliers
            finite_losses = torch.where(
                active, per_outcome_losses, torch.zeros_like(per_outcome_losses)
            )
            # Apply the same curriculum ramp to Kendall's log-sigma term. A
            # task at multiplier zero must be genuinely absent, while a task
            # entering at 0 < m < 1 should contribute both likelihood and
            # uncertainty regularization proportionally.
            regularizer = torch.sum(
                torch.where(
                    active,
                    self.weighter.log_sigma * curriculum_multipliers,
                    torch.zeros_like(self.weighter.log_sigma),
                )
            )
            total = torch.sum(weights * finite_losses) + regularizer
            for name in loss_names:
                index = self.outcome_index[name]
                diagnostics[f"cr/outcome_weight/{name}"] = weights[index].detach()
                diagnostics[f"cr/curriculum/outcome_multiplier/{name}"] = (
                    curriculum_multipliers[index].detach()
                )
                diagnostics[f"cr/log_sigma/{name}"] = self.weighter.log_sigma[
                    self.outcome_index[name]
                ].detach()
            diagnostics["cr/outcome_weight_mean"] = weights[active].mean().detach()
            diagnostics["cr/log_sigma_mean"] = self.weighter.log_sigma[active].mean().detach()
            if self.weighter_name == "kendall_null":
                diagnostics["cr/loss_over_null_mean"] = active_losses.mean().detach()
        elif self.aggregation == "endpoint_group_macro":
            group_losses = []
            for members in self.endpoint_groups.values():
                indices = [i for i, name in enumerate(loss_names)
                           if name in members and self._curriculum_outcome_multiplier(name) > 0]
                if indices:
                    # Fixed grade denominator preserves curriculum ramp amplitude.
                    group_losses.append(sum(
                        losses[i] * self._curriculum_outcome_multiplier(loss_names[i])
                        for i in indices
                    ) / sum(name in loss_names for name in members))
            total = torch.stack(group_losses).mean() if group_losses else log_hazards.sum() * 0.
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
