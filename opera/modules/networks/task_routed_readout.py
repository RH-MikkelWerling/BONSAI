"""Task-routed partial pooling (router candidates R_PT5 and R_PT3V, 31 October 2026).

The partial-pooling router (R_PP) collapsed to one source per prediction within
a few steps of its warm-up ending. Two causes, both reproduced in a many-source
toy (20 diseases in lineages of four):

1. The patient term dominated the routing query. The disease and outcome query
   embeddings start at zero, the patient projection at full size, so routing
   followed a direction common to all patients, sending everyone to the same
   few sources. Without it, data-poor tasks put most borrowed weight on their
   true relatives.
2. With bounded source corrections and unbounded logits, concentrating the gate
   was the fastest way to a large correction, so routing snapped to one-hot.

This readout routes from the target task only (disease and outcome, with their
interaction) and bounds the logits: ``bound * cos(query, key)``, so no source
can take all the weight, while relatives can still dominate. Optionally the
borrowed mixture is variance-preserving: divided by ``sqrt(sum pi^2)``, so an
even mixture of unrelated unit corrections keeps unit size and the gate is not
pushed to concentrate for magnitude. Own weight ``n / (n + tau)``, bounded
corrections and everything else are inherited from R_PP unchanged.

The learned temperature is not used (the bound replaces it); the patient
projection is kept, frozen at zero, so checkpoints share R_PP's tensors.
"""
from __future__ import annotations

from collections.abc import Sequence

import torch
from torch.nn import functional as F

from opera.modules.networks.partial_pooling_readout import _MASKED, PartialPoolingReadout

MIXES = ("per_source", "variance_preserving")
QUERY_INIT_STD = 0.02
NORMALIZE_EPS = 1e-4


class TaskRoutedPoolingReadout(PartialPoolingReadout):
    def __init__(self, hidden_size: int, fine_names: Sequence[str],
                 outcome_names: Sequence[str], *, logit_bound: float = 5.0,
                 borrow_mix: str = "per_source", **kwargs) -> None:
        if not logit_bound > 0:
            raise ValueError("logit_bound must be positive")
        if borrow_mix not in MIXES:
            raise ValueError(f"borrow_mix must be one of {MIXES}")
        super().__init__(hidden_size, fine_names, outcome_names, **kwargs)
        self.logit_bound = float(logit_bound)
        self.borrow_mix = borrow_mix
        with torch.no_grad():
            self.patient_query.weight.zero_()
        self.patient_query.weight.requires_grad_(False)
        self.temperature_log_ratio.requires_grad_(False)
        # The parent starts the disease and outcome queries at zero. Normalising a
        # zero query has a ~1/eps gradient, which overflows in 16-bit mixed
        # precision, so every step was skipped and routing never moved (first
        # launch, 31 October). Small random queries start near an even mix with
        # ordinary gradients; a forked RNG keeps every other initialisation equal.
        with torch.random.fork_rng(devices=[]):
            torch.nn.init.normal_(self.fine_query.weight, std=QUERY_INIT_STD)
            torch.nn.init.normal_(self.target_query.weight, std=QUERY_INIT_STD)

    def _task_query(self, fine_index):
        """(batch, target outcome, hidden) from disease and outcome only."""
        disease_query = self.fine_query(fine_index)[:, None, :]
        outcome_query = self.target_query.weight[None, :, :]
        query = disease_query + outcome_query
        if self.query_interaction is not None:
            shape = (len(fine_index), self.n_outcome, self.hidden_size)
            joint = torch.cat((torch.zeros(shape, device=query.device, dtype=query.dtype),
                               disease_query.expand(shape), outcome_query.expand(shape)), dim=-1)
            query = query + self.query_interaction(joint)
        return query

    def route_weights(self, patient: torch.Tensor, fine_index: torch.Tensor) -> torch.Tensor:
        if patient.ndim != 2 or patient.shape[1] != self.hidden_size:
            raise ValueError("Expected (batch, hidden) patient representations")
        fine_index = fine_index.long().reshape(-1).to(patient.device)
        if fine_index.numel() != len(patient):
            raise ValueError("fine_index must have one value per patient")
        known = (fine_index >= 0) & (fine_index < self.n_disease)
        safe_index = fine_index.clamp(0, self.n_disease - 1)
        query = F.normalize(self._task_query(safe_index).float(), dim=-1, eps=NORMALIZE_EPS)
        keys = F.normalize(self.source_keys.float(), dim=-1, eps=NORMALIZE_EPS)
        scores = self.logit_bound * torch.einsum("boh,sh->bos", query, keys)
        own = torch.zeros(len(patient), self.n_outcome, self.n_sources,
                          dtype=torch.bool, device=patient.device)
        own[torch.arange(len(patient), device=patient.device), :, safe_index] = True
        rows = torch.arange(self.n_outcome, device=patient.device)
        own[:, rows, self.n_disease + rows] = True
        scores = scores.masked_fill(own | (self.source_shrinkage <= 0)[None, None, :], _MASKED)
        slot = torch.full((len(patient), self.n_outcome, 1), _MASKED,
                          dtype=scores.dtype, device=patient.device)
        logits = torch.cat((slot, scores), dim=-1)
        if self.training and self.router_frozen:
            logits = logits.detach()
        weights = torch.softmax(logits, dim=-1)
        fallback = torch.zeros_like(weights)
        fallback[..., 0] = 1
        return torch.where(known[:, None, None], weights, fallback)

    def _mix_scale(self, weights):
        """1 / sqrt(sum pi^2) per (patient, target) for the variance-preserving mix."""
        if self.borrow_mix != "variance_preserving":
            return None
        return weights[..., 1:].float().pow(2).sum(-1, keepdim=True).clamp_min(1e-8).rsqrt()

    def component_vectors(self, patient, fine_index, weights, outcome_indices=None):
        vectors = super().component_vectors(patient, fine_index, weights, outcome_indices)
        scale = self._mix_scale(weights)
        if scale is not None:
            for key in ("borrowed_disease", "borrowed_outcome"):
                vectors[key] = vectors[key] * scale
        return vectors

    def disease_terms(self, patient, fine_index):
        base, terms = super().disease_terms(patient, fine_index)
        scale = self._mix_scale(self.route_weights(patient, fine_index))
        if scale is None:
            return base, terms
        # Rescale only the borrowed disease terms (not the own disease's term),
        # keeping base + terms equal to the forward output.
        fine_index = fine_index.long().reshape(-1).to(patient.device)
        own = F.one_hot(fine_index.clamp(0, self.n_disease - 1), self.n_disease).bool()
        borrowed = (~own)[:, None, :, None]
        adjusted = torch.where(borrowed, terms * scale[..., None], terms)
        return base + (terms - adjusted).sum(2).to(base.dtype), adjusted

    def pooling_summary(self) -> dict:
        summary = super().pooling_summary()
        summary.update(logit_bound=self.logit_bound, borrow_mix=self.borrow_mix,
                       routing="task (disease x outcome)")
        return summary
