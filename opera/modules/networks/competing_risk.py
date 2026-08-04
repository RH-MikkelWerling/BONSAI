"""Continuous-time piecewise-exponential competing-risk objective."""

from __future__ import annotations

from bisect import bisect_left
from collections.abc import Mapping, Sequence

import torch
from torch import nn


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
        self.register_buffer(
            "boundaries",
            boundaries / self.time_scale_days,
            persistent=True,
        )

    @property
    def n_intervals(self) -> int:
        return int(self.boundaries.numel()) + 1

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
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
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
        diagnostics: dict[str, torch.Tensor] = {}

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
            # Autocast may promote the cumulative-hazard reduction to float32
            # while leaving the selected log hazards in float16/bfloat16.
            # Constructing the term functionally avoids dtype-sensitive indexed
            # assignment and keeps this path friendly to torch.compile.
            event_term = (
                target.to(cumulative_hazard.dtype)
                * target_log_hazard.to(cumulative_hazard.dtype)
                + competing.to(cumulative_hazard.dtype)
                * competing_log_hazard.to(cumulative_hazard.dtype)
            )

            outcome_loss = (cumulative_hazard - event_term).mean()
            if self.smoothness_weight and self.n_intervals > 1:
                active_log_hazards = log_hazards[valid, outcome_index]
                if name in self.no_competing_outcomes:
                    active_log_hazards = active_log_hazards[:, :1]
                smoothness = active_log_hazards.diff(dim=-1).square().mean()
                outcome_loss = outcome_loss + self.smoothness_weight * smoothness
                diagnostics[f"cr/smoothness/{name}"] = smoothness.detach()

            losses.append(outcome_loss)
            diagnostics[f"cr/loss/{name}"] = outcome_loss.detach()
            diagnostics[f"cr/n_valid/{name}"] = valid.sum().float().detach()
            diagnostics[f"cr/n_target/{name}"] = target.sum().float().detach()
            diagnostics[f"cr/n_death/{name}"] = competing.sum().float().detach()

        if not losses:
            zero = log_hazards.sum() * 0.0
            diagnostics["cr/n_active_outcomes"] = zero.detach()
            return zero, diagnostics

        total = torch.stack(losses).mean()
        diagnostics["cr/n_active_outcomes"] = torch.tensor(
            float(len(losses)),
            device=log_hazards.device,
        )
        diagnostics["cr/loss"] = total.detach()
        return total, diagnostics
