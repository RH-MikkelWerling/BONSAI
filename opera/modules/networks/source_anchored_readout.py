"""Patient-dependent transfer from fine-disease and outcome source adapters.

Source names are fixed in the constructor and saved in model_init_config. A
borrowed adapter uses detached *parameters*, not a detached output: gradients
still reach the patient representation and the router. Home adapters are used
with live parameters only for their own eligible supervision.

Router fix (lock addendum 2026-10-26; all off by default). On stage-2 tuning
fits the router put ~90% of every prediction's weight on one outcome's adapter
within the first epoch: it chose before the adapters had learned anything, and
the only way to get a correction that helps every target was to borrow one.

* ``global_adapter``: one more adapter with live parameters, trained by every
  target and always applied, outside the routed mix. A general correction
  gets its own home; sources are then routed only for what they add.
* ``router_warmup_epochs`` (set per epoch by the training module through
  ``router_frozen``): routing stays at its starting distribution, and the
  router receives no gradient, while the adapters learn.
* ``logit_scale``: scores are ``s * cos(query, key)`` instead of an unbounded
  dot product, so no source can take all the weight and the others keep a
  gradient.
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
        query_mode: str = "additive",
        query_hidden_dim: int = 128,
        source_mode: str = "both",
        borrowing: bool = True,
        disease_embedding: bool = False,
        global_adapter: bool = False,
        router_warmup_epochs: int = 0,
        logit_scale: float = 0.0,
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
        if query_mode not in {"additive", "interacting"}:
            raise ValueError("query_mode must be additive or interacting")
        if query_hidden_dim < 1:
            raise ValueError("query_hidden_dim must be positive")
        self.query_mode = query_mode
        if source_mode not in {"both", "disease_only", "outcome_only", "none"}:
            raise ValueError("source_mode must be both, disease_only, outcome_only, or none")
        self.source_mode = source_mode
        self.use_disease_sources = source_mode in {"both", "disease_only"}
        self.use_outcome_sources = source_mode in {"both", "outcome_only"}
        # Decomposition controls (off by default; parameter-identical when off):
        # without borrowing, a prediction may use only its own disease's and its
        # own outcome's adapter; the disease embedding gives a shared readout
        # the disease identity and nothing else.
        self.borrowing = bool(borrowing)
        self.disease_embedding = (
            nn.Embedding(self.n_disease, hidden_size) if disease_embedding else None
        )
        if router_warmup_epochs < 0 or logit_scale < 0:
            raise ValueError("router_warmup_epochs and logit_scale must be nonnegative")
        self.router_warmup_epochs = int(router_warmup_epochs)
        self.router_frozen = self.router_warmup_epochs > 0
        self.logit_scale = float(logit_scale)
        if self.disease_embedding is not None:
            nn.init.zeros_(self.disease_embedding.weight)

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
        # Keep the old mode parameter-identical for existing checkpoints. The
        # nonlinear query permits patient x disease x target source preferences;
        # no dropout, so gradient-cache replay remains deterministic.
        self.query_interaction = None
        if query_mode == "interacting":
            # Preserve the RNG stream for all subsequently constructed shared
            # tensors (notably the risk head). Start at exactly the additive
            # function, then let the interaction branch learn away from zero.
            with torch.random.fork_rng(devices=[]):
                self.query_interaction = nn.Sequential(
                    nn.Linear(3 * hidden_size, query_hidden_dim),
                    nn.GELU(),
                    nn.Linear(query_hidden_dim, hidden_size),
                )
            nn.init.zeros_(self.query_interaction[-1].weight)
            nn.init.zeros_(self.query_interaction[-1].bias)
        self.shared_logit = nn.Parameter(torch.tensor(3.0))
        self.global_down = self.global_up = None
        if global_adapter:
            # Forked RNG, as for the query interaction: every other tensor
            # (notably the risk head built after this readout) is initialised
            # exactly as without the global adapter.
            with torch.random.fork_rng(devices=[]):
                self.global_down = nn.Parameter(torch.empty(bottleneck, hidden_size))
                self.global_up = nn.Parameter(torch.empty(hidden_size, bottleneck))
                nn.init.normal_(self.global_down, std=0.02)
                nn.init.normal_(self.global_up, std=1e-3)
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
        patient_query = self.patient_query(patient)[:, None, :]
        disease_query = self.fine_query(safe_index)[:, None, :]
        outcome_query = self.target_query.weight[None, :, :]
        query = patient_query + disease_query + outcome_query
        if self.query_interaction is not None:
            shape = (len(patient), self.n_outcome, self.hidden_size)
            joint = torch.cat((patient_query.expand(shape),
                               disease_query.expand(shape),
                               outcome_query.expand(shape)), dim=-1)
            query = query + self.query_interaction(joint)
        if self.logit_scale:
            scores = self.logit_scale * torch.einsum(
                "boh,sh->bos", F.normalize(query, dim=-1), F.normalize(self.source_keys, dim=-1))
        else:
            scores = torch.einsum("boh,sh->bos", query, self.source_keys) / self.hidden_size**0.5
        enabled = torch.tensor(
            [self.use_disease_sources] * self.n_disease
            + [self.use_outcome_sources] * self.n_outcome,
            device=patient.device, dtype=torch.bool,
        )
        scores = scores.masked_fill(
            (~enabled | (self.source_shrinkage <= 0))[None, None, :], -1e4
        )
        if not self.borrowing:
            own = torch.zeros(len(patient), self.n_outcome, self.n_sources,
                              dtype=torch.bool, device=patient.device)
            own[torch.arange(len(patient), device=patient.device), :, safe_index] = True
            outcome_rows = torch.arange(self.n_outcome, device=patient.device)
            own[:, outcome_rows, self.n_disease + outcome_rows] = True
            scores = scores.masked_fill(~own, -1e4)
        shared = self.shared_logit.expand(len(patient), self.n_outcome, 1)
        logits = torch.cat((shared, scores), dim=-1)
        if self.router_frozen:
            logits = logits.detach()  # warm-up: no gradient reaches the router
        weights = torch.softmax(logits, dim=-1)
        if self.home_floor:
            home = torch.zeros_like(weights)
            row = torch.arange(len(patient), device=patient.device)[:, None]
            outcome = torch.arange(self.n_outcome, device=patient.device)[None, :]
            disease_available = (self.source_shrinkage[safe_index] > 0) & self.use_disease_sources
            outcome_available = (self.source_shrinkage[self.n_disease + outcome] > 0) & self.use_outcome_sources
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
        # Autocast makes einsum half precision, whereas support/router-weight
        # products promote home corrections to float32. Accumulate both paths
        # in float32 so index_add has matching dtypes and stable summation.
        residual = torch.einsum("bos,bsh->boh", weights[..., 1:], borrowed).float()

        if self.anchored:
            fine_index = fine_index.long().reshape(-1).to(patient.device)
            valid = (fine_index >= 0) & (fine_index < self.n_disease)
            # Only the home disease rows use live disease parameters.
            if self.use_disease_sources and bool(valid.any()):
                rows = torch.nonzero(valid, as_tuple=True)[0]
                source = fine_index[rows]
                hidden = F.gelu(torch.einsum("bh,brh->br", patient[rows], self.down[source]))
                home = torch.einsum("br,bhr->bh", hidden, self.up[source])
                home = home * self.source_shrinkage[source, None]
                adjustment = (home - borrowed[rows, source])[:, None, :]
                home_weight = weights[rows].gather(
                    2, (1 + source)[:, None, None].expand(-1, self.n_outcome, 1)
                )
                correction = (home_weight * adjustment).to(residual.dtype)
                residual = residual.index_add(0, rows, correction)
            # Outcome o has live parameters only in its own target loss.
            for outcome in range(self.n_outcome if self.use_outcome_sources else 0):
                source = self.n_disease + outcome
                hidden = F.gelu(F.linear(patient, self.down[source]))
                home = F.linear(hidden, self.up[source]) * self.source_shrinkage[source]
                residual[:, outcome] = residual[:, outcome] + (
                    weights[:, outcome, 1 + source, None]
                    * (home - borrowed[:, source])
                )

        if self.global_down is not None:
            general = F.linear(F.gelu(F.linear(patient, self.global_down)), self.global_up)
            residual = residual + general[:, None, :].to(residual.dtype)
        output = patient[:, None, :] + self.residual_scale * residual
        if self.disease_embedding is not None:
            fine = fine_index.long().reshape(-1).to(patient.device)
            known = ((fine >= 0) & (fine < self.n_disease)).to(output.dtype)
            shift = self.disease_embedding(fine.clamp(0, self.n_disease - 1))
            output = output + (known[:, None] * shift)[:, None, :].to(output.dtype)
        return (output, weights) if return_weights else output

    def disease_terms(
        self, patient: torch.Tensor, fine_index: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """The readout split into its disease-source terms (evaluation only).

        Returns ``base`` (batch, outcome, hidden) and ``terms`` (batch, outcome,
        disease, hidden) with ``base + terms.sum(2)`` equal to the forward
        output. A term is the source's routed correction, weight x adapter
        output; the home disease's term uses the same adapter values as its live
        correction. Dropping a term removes that source's contribution and
        changes nothing else (no renormalisation of the other weights).
        """
        if not self.use_disease_sources:
            raise ValueError("This readout has no disease sources")
        output, weights = self(patient, fine_index, return_weights=True)
        n = self.n_disease
        hidden = F.gelu(torch.einsum("bh,srh->bsr", patient, self.down[:n]))
        correction = torch.einsum("bsr,shr->bsh", hidden, self.up[:n])
        correction = correction * self.source_shrinkage[None, :n, None]
        terms = self.residual_scale * (
            weights[..., 1:1 + n, None].float() * correction[:, None].float())
        return output - terms.sum(2).to(output.dtype), terms


def borrowing_design(fine_names: Sequence[str], fine_index: torch.Tensor):
    """Source masks of the borrowing design, (variant, batch, disease).

    ``full``: every source; ``none``: the patient's own disease only (no
    borrowing); ``in:X``: own disease and X (what X adds alone); ``out:X``:
    every source but X (what only X provides).
    """
    names = list(fine_names)
    n = len(names)
    fine_index = fine_index.long().reshape(-1)
    known = (fine_index >= 0) & (fine_index < n)
    own = F.one_hot(fine_index.clamp(0, n - 1), n).float() * known[:, None]
    eye = torch.eye(n, device=fine_index.device)
    full = torch.ones_like(own)
    masks = [full, own,
             *((own + eye[i]).clamp(max=1) for i in range(n)),
             *(full - eye[i] for i in range(n))]
    labels = ["full", "none", *(f"in:{x}" for x in names), *(f"out:{x}" for x in names)]
    return labels, torch.stack(masks)
