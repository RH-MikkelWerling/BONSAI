"""Patient-dependent transfer from fine-disease and outcome source adapters.

Source names are fixed in the constructor and saved in model_init_config. A
borrowed adapter uses detached *parameters*, not a detached output: gradients
still reach the patient representation and the router. Home adapters are used
with live parameters only for their own eligible supervision.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn
from torch.nn import functional as F


class SourceAnchoredReadout(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        fine_names: Sequence[str],
        outcome_names: Sequence[str],
        *,
        bottleneck: int = 8,
        residual_scale: float = 0.1,
        home_floor: float = 0.1,
        anchored: bool = True,
    ) -> None:
        super().__init__()
        self.fine_names = tuple(sorted(fine_names))
        self.outcome_names = tuple(outcome_names)
        if not self.fine_names or len(set(self.fine_names)) != len(self.fine_names):
            raise ValueError("fine_names must be nonempty and unique")
        if not self.outcome_names or len(set(self.outcome_names)) != len(self.outcome_names):
            raise ValueError("outcome_names must be nonempty and unique")
        if bottleneck < 1 or not 0 < residual_scale <= 1 or not 0 <= home_floor < 0.5:
            raise ValueError("Invalid adapter bottleneck, residual scale, or home floor")
        self.hidden_size = hidden_size
        self.n_disease = len(self.fine_names)
        self.n_outcome = len(self.outcome_names)
        self.n_sources = self.n_disease + self.n_outcome
        self.residual_scale = float(residual_scale)
        self.home_floor = float(home_floor)
        self.anchored = bool(anchored)

        self.down = nn.Parameter(torch.empty(self.n_sources, bottleneck, hidden_size))
        self.up = nn.Parameter(torch.empty(self.n_sources, hidden_size, bottleneck))
        nn.init.normal_(self.down, std=0.02)
        nn.init.normal_(self.up, std=1e-3)
        self.patient_query = nn.Linear(hidden_size, hidden_size, bias=False)
        self.fine_query = nn.Embedding(self.n_disease, hidden_size)
        self.target_query = nn.Embedding(self.n_outcome, hidden_size)
        self.source_keys = nn.Parameter(torch.empty(self.n_sources, hidden_size))
        nn.init.normal_(self.source_keys, std=0.02)
        nn.init.zeros_(self.fine_query.weight)
        nn.init.zeros_(self.target_query.weight)
        nn.init.normal_(self.patient_query.weight, std=0.02)
        self.shared_logit = nn.Parameter(torch.tensor(3.0))
        self.register_buffer("source_shrinkage", torch.ones(self.n_sources))
        self.register_buffer("source_train_patients", torch.zeros(self.n_sources, dtype=torch.long))
        self.register_buffer("source_train_events", torch.zeros(self.n_sources, dtype=torch.long))
        self.register_buffer("source_train_observations", torch.zeros(self.n_sources, dtype=torch.long))

    def set_training_support(
        self, patients: Sequence[int], events: Sequence[int], observations: Sequence[int],
        *, event_tau: float = 20.0,
    ) -> None:
        if event_tau < 0:
            raise ValueError("event_tau must be nonnegative")
        values = [patients, events, observations]
        if any(len(value) != self.n_sources for value in values):
            raise ValueError("Support vectors must match the source inventory")
        if any(int(item) < 0 for value in values for item in value):
            raise ValueError("Support counts cannot be negative")
        self.source_train_patients.copy_(torch.as_tensor(patients, dtype=torch.long))
        self.source_train_events.copy_(torch.as_tensor(events, dtype=torch.long))
        self.source_train_observations.copy_(torch.as_tensor(observations, dtype=torch.long))
        count = self.source_train_events.float()
        self.source_shrinkage.copy_(count / (count + event_tau).clamp_min(1.0))

    def route_weights(self, patient: torch.Tensor, fine_index: torch.Tensor) -> torch.Tensor:
        """Return (batch, target outcome, shared + named sources) weights."""
        if patient.ndim != 2 or patient.shape[1] != self.hidden_size:
            raise ValueError("Expected (batch, hidden) patient representations")
        fine_index = fine_index.long().reshape(-1).to(patient.device)
        if fine_index.numel() != len(patient):
            raise ValueError("fine_index must have one value per patient")
        known = (fine_index >= 0) & (fine_index < self.n_disease)
        safe_index = fine_index.clamp(0, self.n_disease - 1)
        query = (
            self.patient_query(patient)[:, None, :]
            + self.fine_query(safe_index)[:, None, :]
            + self.target_query.weight[None, :, :]
        )
        scores = torch.einsum("boh,sh->bos", query, self.source_keys) / self.hidden_size**0.5
        scores = scores.masked_fill(self.source_shrinkage[None, None, :] <= 0, -1e4)
        shared = self.shared_logit.expand(len(patient), self.n_outcome, 1)
        weights = torch.softmax(torch.cat((shared, scores), dim=-1), dim=-1)
        if self.home_floor:
            home = torch.zeros_like(weights)
            row = torch.arange(len(patient), device=patient.device)[:, None]
            outcome = torch.arange(self.n_outcome, device=patient.device)[None, :]
            disease_available = self.source_shrinkage[safe_index] > 0
            outcome_available = self.source_shrinkage[self.n_disease + outcome] > 0
            home[row, outcome, 1 + safe_index[:, None]] = (
                self.home_floor * known[:, None] * disease_available[:, None]
            )
            home[row, outcome, 1 + self.n_disease + outcome] = (
                self.home_floor * outcome_available
            )
            weights = weights * (1 - home.sum(-1, keepdim=True)) + home
        # Unknown fine identity is an explicit shared-path fallback.
        fallback = torch.zeros_like(weights)
        fallback[..., 0] = 1
        return torch.where(known[:, None, None], weights, fallback)

    def forward(
        self, patient: torch.Tensor, fine_index: torch.Tensor, *,
        removed_sources: Sequence[str] = (), return_weights: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        weights = self.route_weights(patient, fine_index)
        source_names = (*("disease:" + n for n in self.fine_names),
                        *("outcome:" + n for n in self.outcome_names))
        if removed_sources:
            unknown = set(removed_sources) - set(source_names)
            if unknown:
                raise ValueError(f"Unknown source names: {sorted(unknown)}")
            keep = torch.ones(self.n_sources + 1, device=patient.device)
            for name in removed_sources:
                keep[1 + source_names.index(name)] = 0
            weights = weights * keep
            weights = weights / weights.sum(-1, keepdim=True).clamp_min(1e-8)

        down = self.down.detach() if self.anchored else self.down
        up = self.up.detach() if self.anchored else self.up
        borrowed = torch.einsum("bh,srh->bsr", patient, down)
        borrowed = torch.einsum("bsr,shr->bsh", F.gelu(borrowed), up)
        borrowed = borrowed * self.source_shrinkage[None, :, None]
        residual = torch.einsum("bos,bsh->boh", weights[..., 1:], borrowed)

        if self.anchored:
            fine_index = fine_index.long().reshape(-1).to(patient.device)
            valid = (fine_index >= 0) & (fine_index < self.n_disease)
            # Only the home disease rows use live disease parameters.
            if bool(valid.any()):
                rows = torch.nonzero(valid, as_tuple=True)[0]
                source = fine_index[rows]
                hidden = F.gelu(torch.einsum("bh,brh->br", patient[rows], self.down[source]))
                home = torch.einsum("br,bhr->bh", hidden, self.up[source])
                home = home * self.source_shrinkage[source, None]
                adjustment = (home - borrowed[rows, source])[:, None, :]
                home_weight = weights[rows].gather(
                    2, (1 + source)[:, None, None].expand(-1, self.n_outcome, 1)
                )
                residual = residual.index_add(0, rows, home_weight * adjustment)
            # Outcome o has live parameters only in its own target loss.
            for outcome in range(self.n_outcome):
                source = self.n_disease + outcome
                hidden = F.gelu(F.linear(patient, self.down[source]))
                home = F.linear(hidden, self.up[source]) * self.source_shrinkage[source]
                residual[:, outcome] = residual[:, outcome] + (
                    weights[:, outcome, 1 + source, None]
                    * (home - borrowed[:, source])
                )

        output = patient[:, None, :] + self.residual_scale * residual
        return (output, weights) if return_weights else output
