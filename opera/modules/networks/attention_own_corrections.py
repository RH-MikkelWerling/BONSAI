"""Own corrections on outcome-conditioned representations (stage 9, 1 November 2026).

Task-conditioned pooling gives every outcome its own patient summary,
(batch, outcome, hidden). This module adds the corrections that carried
R_AB's gain over the joint model, now applied to those summaries: an own
disease adapter (the patient's diagnosis), an own outcome adapter, one global
adapter, and optionally one adapter per disease lineage (hierarchical
shrinkage, as R_PH_G). Live parameters, full weight, unbounded, as in R_O.
Corrections start near zero (small ``up`` matrices), so a fit starts at the
attention-pooled joint model. A patient without a known diagnosis gets the
global correction only, as in the adapter readouts.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch
from torch import nn
from torch.nn import functional as F


def _adapters(count, rank, hidden):
    down = nn.Parameter(torch.empty(count, rank, hidden))
    up = nn.Parameter(torch.empty(count, hidden, rank))
    nn.init.normal_(down, std=0.02)
    nn.init.normal_(up, std=1e-3)
    return down, up


class AttentionOwnCorrections(nn.Module):
    def __init__(self, hidden_size: int, fine_names: Sequence[str], outcome_names: Sequence[str], *,
                 rank: int = 8, residual_scale: float = 0.7,
                 disease_groups: Mapping[str, str] | None = None,
                 use_disease: bool = True, use_outcome: bool = True,
                 cross: bool = False, shrink_tau: float | None = None,
                 outcome_groups: Mapping[str, str] | None = None) -> None:
        super().__init__()
        if rank < 1 or not 0 < residual_scale <= 1:
            raise ValueError("Need a positive rank and residual_scale in (0, 1]")
        if not fine_names or len(set(fine_names)) != len(fine_names):
            raise ValueError("Own corrections need unique fine_names")
        self.fine_names, self.outcome_names = list(fine_names), list(outcome_names)
        self.n_disease, self.n_outcome = len(self.fine_names), len(self.outcome_names)
        self.hidden_size, self.residual_scale = int(hidden_size), float(residual_scale)
        # Stage-12 ablations (1 Nov 2026): drop the diagnosis or the outcome term.
        # The adapters still exist (unchanged checkpoint layout) but are frozen.
        self.use_disease, self.use_outcome = bool(use_disease), bool(use_outcome)
        # Stage 13 (1 Nov 2026). ``cross``: each outcome rescales the diagnosis
        # adapter's bottleneck, so the diagnosis correction differs by outcome
        # (a diagnosis x outcome correction, pooled through the shared adapters).
        # ``shrink_tau``: the own corrections of a diagnosis x outcome cell are
        # scaled by n / (n + tau), n its training events (fixed prior: learned
        # from the training loss, shrinkage drifts to none).
        self.cross = bool(cross)
        if shrink_tau is not None and not shrink_tau > 0:
            raise ValueError("shrink_tau must be positive")
        self.shrink_tau = None if shrink_tau is None else float(shrink_tau)
        self.register_buffer("cell_train_events", torch.zeros(len(fine_names), len(outcome_names)))
        with torch.random.fork_rng(devices=[]):  # every other initialisation unchanged
            self.disease_down, self.disease_up = _adapters(self.n_disease, rank, hidden_size)
            self.outcome_down, self.outcome_up = _adapters(self.n_outcome, rank, hidden_size)
            self.global_down, self.global_up = _adapters(1, rank, hidden_size)
            self.group_down = self.group_up = None
            self.group_names = []
            if disease_groups:
                missing = [n for n in self.fine_names if n not in disease_groups]
                if missing:
                    raise ValueError(f"No group for disease {missing}")
                self.group_names = sorted({disease_groups[n] for n in self.fine_names})
                self.register_buffer("group_index", torch.tensor(
                    [self.group_names.index(disease_groups[n]) for n in self.fine_names]))
                self.group_down, self.group_up = _adapters(len(self.group_names), rank, hidden_size)
        # Stage 13: one correction per endpoint family, shared by its outcomes
        # (the outcome-side counterpart of the lineage corrections).
        self.outcome_group_down = self.outcome_group_up = None
        self.outcome_group_names = []
        if outcome_groups:
            missing = [o for o in self.outcome_names if o not in outcome_groups]
            if missing:
                raise ValueError(f"No endpoint family for outcome {missing}")
            self.outcome_group_names = sorted({outcome_groups[o] for o in self.outcome_names})
            self.register_buffer("outcome_group_index", torch.tensor(
                [self.outcome_group_names.index(outcome_groups[o]) for o in self.outcome_names]))
            with torch.random.fork_rng(devices=[]):
                self.outcome_group_down, self.outcome_group_up = _adapters(
                    len(self.outcome_group_names), rank, hidden_size)
        self.outcome_bottleneck_scale = None
        if self.cross:
            # Starts at zero: the diagnosis correction starts identical for every outcome.
            self.outcome_bottleneck_scale = nn.Parameter(torch.zeros(len(outcome_names), rank))
        for used, params in ((self.use_disease, (self.disease_down, self.disease_up)),
                             (self.use_outcome, (self.outcome_down, self.outcome_up))):
            if not used:
                for param in params:
                    param.requires_grad_(False)

    def set_cell_support(self, events) -> None:
        """Training events per (diagnosis, outcome), from eligible training records."""
        events = torch.as_tensor(events, dtype=torch.float32)
        if events.shape != (self.n_disease, self.n_outcome) or bool((events < 0).any()):
            raise ValueError("Cell support must be nonnegative (diagnosis, outcome) counts")
        self.cell_train_events.copy_(events.to(self.cell_train_events.device))

    def shrink_weights(self, safe_index: torch.Tensor) -> torch.Tensor | None:
        """(batch, outcome, 1) weights n / (n + tau) for the own corrections, or None."""
        if self.shrink_tau is None:
            return None
        events = self.cell_train_events[safe_index].float()
        return (events / (events + self.shrink_tau))[..., None]

    # Not ``_apply``: that name is nn.Module's device/dtype hook (.to/.cuda/.cpu),
    # and shadowing it made every fit fail when the model moved to the GPU.
    @staticmethod
    def _adapt(representation, down, up):
        """Per-patient adapters: representation (b, o, h); down (b, r, h); up (b, h, r)."""
        hidden = F.gelu(torch.einsum("boh,brh->bor", representation, down))
        return torch.einsum("bor,bhr->boh", hidden, up)

    def components(self, representation: torch.Tensor, fine_index: torch.Tensor) -> dict:
        if representation.ndim != 3 or representation.shape[1:] != (self.n_outcome, self.hidden_size):
            raise ValueError("Expected (batch, outcome, hidden) representations")
        fine_index = fine_index.long().reshape(-1).to(representation.device)
        known = ((fine_index >= 0) & (fine_index < self.n_disease)).to(representation.dtype)[:, None, None]
        safe = fine_index.clamp(0, self.n_disease - 1)
        outcome = torch.einsum("bor,ohr->boh", F.gelu(torch.einsum("boh,orh->bor", representation, self.outcome_down)),
                               self.outcome_up)
        general = torch.einsum("bor,hr->boh", F.gelu(torch.einsum("boh,rh->bor", representation, self.global_down[0])),
                               self.global_up[0])
        terms = {"global": general}
        shrink = self.shrink_weights(safe)
        shrink = known if shrink is None else known * shrink.to(representation.dtype)
        if self.use_disease:
            if self.cross:
                hidden = F.gelu(torch.einsum("boh,brh->bor", representation, self.disease_down[safe]))
                hidden = hidden * (1 + self.outcome_bottleneck_scale[None].to(hidden.dtype))
                disease = torch.einsum("bor,bhr->boh", hidden, self.disease_up[safe])
            else:
                disease = self._adapt(representation, self.disease_down[safe], self.disease_up[safe])
            terms["own_disease"] = shrink * disease
        if self.use_outcome:
            terms["own_outcome"] = shrink * outcome
        # Lineage and endpoint-family corrections are pooled over their members:
        # not shrunk per cell.
        if self.group_down is not None:
            group = self.group_index[safe]
            terms["group"] = known * self._adapt(representation, self.group_down[group], self.group_up[group])
        if self.outcome_group_down is not None:
            index = self.outcome_group_index
            family = torch.einsum("bor,ohr->boh", F.gelu(torch.einsum(
                "boh,orh->bor", representation, self.outcome_group_down[index])), self.outcome_group_up[index])
            terms["outcome_group"] = known * family
        return terms

    def forward(self, representation: torch.Tensor, fine_index: torch.Tensor) -> torch.Tensor:
        terms = self.components(representation, fine_index)
        return representation + self.residual_scale * sum(terms.values()).to(representation.dtype)
