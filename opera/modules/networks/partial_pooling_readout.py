"""Partial-pooling router: how much to borrow, and from which sources.

Lock amendment 2026-10-30 (R_PP). The R_AB audit found the routing close to an
even mix: a source's influence is gate x adapter magnitude, so magnitude could
stand in for the gate, and fixed 1/16 logit scaling kept logits close. This
readout keeps every R_AB component and changes three things:

1. Source corrections are soft-normalised, ``a / sqrt(|a|^2 + 1)``: bounded by
   1, so a source's influence is set by its weight. Adapters start near zero,
   so training still starts at the joint model's predictions.
2. Own disease and own outcome corrections sit outside the borrowed mix with
   weight ``w_own = n_t / (n_t + tau)`` (``n_t``: training events of the target
   disease x outcome; ``tau`` learned). Borrowed sources share ``1 - w_own``,
   and there is no "borrow nothing" slot, so how much to borrow has one control.
3. The fixed ``1/sqrt(hidden)`` logit scaling is a learned temperature.

New parameters: ``tau``, the temperature and one pooled-correction scale. The
first two are ``prior x exp(p)`` with ``p = 0`` at the start, so weight decay
shrinks them towards their priors. Weight slot 0 (the parent's shared slot) is
kept for shape compatibility and is always 0, except as the parent's
unknown-disease fallback (no corrections).

The base class stays byte-identical, so existing checkpoints and their
router-code provenance are unaffected.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
from torch import nn
from torch.nn import functional as F

from opera.modules.networks.source_anchored_readout import SourceAnchoredReadout

COMPONENTS = ("global", "own_disease", "own_outcome", "borrowed_disease", "borrowed_outcome")
_MASKED = -1e4


def soft_normalize(values: torch.Tensor) -> torch.Tensor:
    """Bounded magnitude, ``|a| / sqrt(|a|^2 + 1) < 1``, smooth at zero."""
    return values * torch.rsqrt(values.pow(2).sum(-1, keepdim=True) + 1.0)


class PartialPoolingReadout(SourceAnchoredReadout):
    def __init__(
        self,
        hidden_size: int,
        fine_names: Sequence[str],
        outcome_names: Sequence[str],
        *,
        tau_prior: float = 20.0,
        **kwargs,
    ) -> None:
        fixed = {"own_outside": True, "home_floor": 0.0, "logit_scale": 0.0,
                 "anchored": True, "borrowing": True, "disease_embedding": False}
        for key, value in fixed.items():
            if key in kwargs and kwargs[key] != value:
                raise ValueError(f"The partial-pooling router requires {key}={value}")
            kwargs[key] = value
        if kwargs.get("source_mode", "both") != "both":
            raise ValueError("The partial-pooling router borrows from diseases and outcomes")
        if not math.isfinite(tau_prior) or tau_prior <= 0:
            raise ValueError("tau_prior must be positive")
        super().__init__(hidden_size, fine_names, outcome_names, **kwargs)
        self.tau_prior = float(tau_prior)
        self.temperature_prior = float(hidden_size) ** 0.5
        self.tau_log_ratio = nn.Parameter(torch.zeros(()))
        self.temperature_log_ratio = nn.Parameter(torch.zeros(()))
        self.pool_scale = nn.Parameter(torch.ones(()))
        self.register_buffer(
            "cell_train_events", torch.zeros(self.n_disease, self.n_outcome, dtype=torch.long))

    @property
    def tau(self) -> torch.Tensor:
        return self.tau_prior * self.tau_log_ratio.exp()

    @property
    def temperature(self) -> torch.Tensor:
        return self.temperature_prior * self.temperature_log_ratio.exp()

    def set_cell_support(self, events) -> None:
        """Training events per (disease, outcome), from eligible training records."""
        events = torch.as_tensor(events, dtype=torch.long)
        if events.shape != (self.n_disease, self.n_outcome) or bool((events < 0).any()):
            raise ValueError("Cell support must be nonnegative (disease, outcome) counts")
        self.cell_train_events.copy_(events.to(self.cell_train_events.device))

    def own_weight(self, fine_index: torch.Tensor) -> torch.Tensor:
        """``w_own`` per (patient, target outcome); 0 for unknown disease."""
        fine_index = fine_index.long().reshape(-1).to(self.cell_train_events.device)
        known = (fine_index >= 0) & (fine_index < self.n_disease)
        events = self.cell_train_events[fine_index.clamp(0, self.n_disease - 1)].float()
        # How much to borrow is part of the router: frozen during its warm-up.
        tau = self.tau.detach() if self.training and self.router_frozen else self.tau
        weight = events / (events + tau)
        return weight * known[:, None].to(weight.dtype)

    def _query(self, patient, safe_index):
        patient_query = self.patient_query(patient)[:, None, :]
        disease_query = self.fine_query(safe_index)[:, None, :]
        outcome_query = self.target_query.weight[None, :, :]
        query = patient_query + disease_query + outcome_query
        if self.query_interaction is not None:
            shape = (len(patient), self.n_outcome, self.hidden_size)
            joint = torch.cat((patient_query.expand(shape), disease_query.expand(shape),
                               outcome_query.expand(shape)), dim=-1)
            query = query + self.query_interaction(joint)
        return query

    def route_weights(self, patient: torch.Tensor, fine_index: torch.Tensor) -> torch.Tensor:
        """(batch, target outcome, 1 + sources): a distribution over borrowed
        sources only. Own sources and slot 0 carry no weight."""
        if patient.ndim != 2 or patient.shape[1] != self.hidden_size:
            raise ValueError("Expected (batch, hidden) patient representations")
        fine_index = fine_index.long().reshape(-1).to(patient.device)
        if fine_index.numel() != len(patient):
            raise ValueError("fine_index must have one value per patient")
        known = (fine_index >= 0) & (fine_index < self.n_disease)
        safe_index = fine_index.clamp(0, self.n_disease - 1)
        scores = torch.einsum("boh,sh->bos", self._query(patient, safe_index),
                              self.source_keys) / self.temperature
        unavailable = (self.source_shrinkage <= 0)[None, None, :]
        own = torch.zeros(len(patient), self.n_outcome, self.n_sources,
                          dtype=torch.bool, device=patient.device)
        own[torch.arange(len(patient), device=patient.device), :, safe_index] = True
        rows = torch.arange(self.n_outcome, device=patient.device)
        own[:, rows, self.n_disease + rows] = True
        scores = scores.masked_fill(own | unavailable, _MASKED)
        slot = torch.full((len(patient), self.n_outcome, 1), _MASKED,
                          dtype=scores.dtype, device=patient.device)
        logits = torch.cat((slot, scores), dim=-1)
        if self.training and self.router_frozen:
            logits = logits.detach()
        weights = torch.softmax(logits, dim=-1)
        fallback = torch.zeros_like(weights)
        fallback[..., 0] = 1
        return torch.where(known[:, None, None], weights, fallback)

    def _source_values(self, patient, *, detached: bool) -> torch.Tensor:
        """Soft-normalised, shrunk source corrections, (batch, source, hidden)."""
        down = self.down.detach() if detached else self.down
        up = self.up.detach() if detached else self.up
        values = torch.einsum("bh,srh->bsr", patient, down)
        values = torch.einsum("bsr,shr->bsh", F.gelu(values), up)
        return soft_normalize(values) * self.source_shrinkage[None, :, None]

    def component_vectors(
        self, patient: torch.Tensor, fine_index: torch.Tensor, weights: torch.Tensor,
        outcome_indices: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """The correction split into COMPONENTS, each (batch, target, hidden)
        before ``residual_scale``. The forward pass sums exactly these, so the
        mechanism audit reconstructs predictions from the same arithmetic."""
        fine_index = fine_index.long().reshape(-1).to(patient.device)
        known = (fine_index >= 0) & (fine_index < self.n_disease)
        safe_index = fine_index.clamp(0, self.n_disease - 1)
        targets = (torch.arange(self.n_outcome, device=patient.device)
                   if outcome_indices is None else outcome_indices.long().to(patient.device))
        if weights.shape != (len(patient), len(targets), self.n_sources + 1):
            raise ValueError("weights must be (batch, targets, 1 + sources)")
        live = self._source_values(patient, detached=False).float()
        borrowed = self._source_values(patient, detached=True).float()
        own_w = self.own_weight(fine_index)[:, targets].float()
        scale = self.pool_scale.float()
        own_scale = (scale * own_w)[..., None]
        borrow_scale = (scale * (1 - own_w) * known[:, None].float())[..., None]
        batch = torch.arange(len(patient), device=patient.device)
        own_disease = live[batch, safe_index][:, None, :].expand(-1, len(targets), -1)
        own_outcome = live[:, self.n_disease + targets]
        gates = weights[..., 1:].float()
        n = self.n_disease
        vectors = {
            "own_disease": own_scale * own_disease,
            "own_outcome": own_scale * own_outcome,
            "borrowed_disease": borrow_scale * torch.einsum(
                "bos,bsh->boh", gates[..., :n], borrowed[:, :n]),
            "borrowed_outcome": borrow_scale * torch.einsum(
                "bos,bsh->boh", gates[..., n:], borrowed[:, n:]),
        }
        general = torch.zeros_like(patient, dtype=torch.float32)
        if self.global_down is not None:
            general = F.linear(F.gelu(F.linear(patient, self.global_down)), self.global_up).float()
        vectors["global"] = general[:, None, :].expand(-1, len(targets), -1)
        return vectors

    def forward(
        self, patient: torch.Tensor, fine_index: torch.Tensor, *,
        removed_sources: Sequence[str] = (), return_weights: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        weights = self.route_weights(patient, fine_index)
        if removed_sources:
            names = (*("disease:" + n for n in self.fine_names),
                     *("outcome:" + n for n in self.outcome_names))
            unknown = set(removed_sources) - set(names)
            if unknown:
                raise ValueError(f"Unknown source names: {sorted(unknown)}")
            keep = torch.ones(self.n_sources + 1, device=patient.device)
            for name in removed_sources:
                keep[1 + names.index(name)] = 0
            weights = weights * keep
            weights = weights / weights.sum(-1, keepdim=True).clamp_min(1e-8)
        vectors = self.component_vectors(patient, fine_index, weights)
        residual = sum(vectors[name] for name in COMPONENTS)
        output = patient[:, None, :] + self.residual_scale * residual.to(patient.dtype)
        return (output, weights) if return_weights else output

    def disease_terms(
        self, patient: torch.Tensor, fine_index: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Disease-source terms (batch, outcome, disease, hidden) with
        ``base + terms.sum(2)`` equal to the forward output (evaluation only)."""
        output, weights = self(patient, fine_index, return_weights=True)
        fine_index = fine_index.long().reshape(-1).to(patient.device)
        known = (fine_index >= 0) & (fine_index < self.n_disease)
        safe_index = fine_index.clamp(0, self.n_disease - 1)
        n = self.n_disease
        values = self._source_values(patient, detached=True)[:, :n].float()
        own_w = self.own_weight(fine_index).float()
        scale = self.pool_scale.float()
        borrow = (scale * (1 - own_w) * known[:, None].float())[..., None, None]
        terms = borrow * weights[..., 1:1 + n, None].float() * values[:, None]
        own = F.one_hot(safe_index, n).float() * known[:, None].float()
        terms = terms + (scale * own_w)[..., None, None] * own[:, None, :, None] * values[:, None]
        terms = self.residual_scale * terms
        return output - terms.sum(2).to(output.dtype), terms

    def pooling_summary(self) -> dict:
        """Learned pooling quantities, for the audit report."""
        with torch.no_grad():
            events = self.cell_train_events.float()
            own = events / (events + self.tau)
            return {"tau": float(self.tau), "tau_prior": self.tau_prior,
                    "temperature": float(self.temperature),
                    "temperature_prior": self.temperature_prior,
                    "pool_scale": float(self.pool_scale),
                    "own_weight_by_cell": own.tolist(),
                    "cell_train_events": self.cell_train_events.tolist()}
