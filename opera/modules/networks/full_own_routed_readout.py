"""Task routing on top of full own corrections (router candidates R_PF5 and
R_PF5E, 31 October 2026, evening).

R_PT5, R_PT3V and their frozen control all finished ~0.004 below R_AB on the
tuning macro AUC, and learned routing recovered only ~0.001 of that. Their
learned tau fell (6-11 against a prior of 20; median own weight 0.83-0.90) and
the shared correction scale grew from 1 to 1.7: the fits asked for more of
their own correction, not for less borrowing. The partial-pooling readout
capped exactly that: own corrections were soft-normalised (norm below 1) and
multiplied by ``w_own``, whereas R_AB's own adapters are unbounded at full
weight, and own corrections carry R_AB's gain over the joint model (own-only
control 0.7832 against R_AB 0.7836).

This readout applies the own disease and own outcome corrections exactly as the
own-outside router (R_O): live parameters, full weight, unbounded. Only what is
borrowed is pooled: task-level routing with bounded logits (as R_PT5) over the
other sources, weighted ``1 - w_own = tau / (n + tau)`` and one learned scale.
Borrowed values are

* ``raw``: the sources' corrections as they are (as R_O borrows them),
* ``equalised``: each source divided by a running mean of its correction norm
  and multiplied by the mean over sources, so borrowed sources have the same
  typical size and a gate is that source's share of the borrowed correction,
* ``bounded``: soft-normalised as in R_PT (norm below 1), or
* ``off``: nothing is borrowed (the own-only control RBH_O in this code).

Design sweep factors (stage 8; each default is the anchor design):

* ``patient_gate``: routing query ``unit(task query) + g * unit(patient query)``
  with a learned scalar ``g`` starting at 0, so routing can become
  patient-specific only as far as it helps (R_PP failed with the patient term
  at full size from the start).
* ``borrow_amount``: ``support`` (``tau / (n + tau)``, one learned tau) or
  ``learned`` (one free borrowed share per disease x outcome, started at the
  support value).
* ``borrow_from``: ``both`` source banks, or ``disease`` or ``outcome`` only.
* ``live_sources``: borrowed sources use live parameters, so borrowers also
  train them; routing then learns soft clusters of tasks that share adapters (a
  task-level mixture of experts), the learned counterpart of the known groups.
* ``disease_groups`` / ``outcome_groups``: hierarchical shrinkage over known
  structure ({name: group}). One more live adapter per disease lineage and per
  endpoint group, applied at full weight to every member and trained by all of
  them, so a rare disease's correction starts from its lineage's. These enter
  the borrowed components (borrowing by structure rather than by routing), so
  the audit's borrowed interventions remove them too.
* ``latent_groups``: K shared group adapters (live, trained by every task that
  uses them), and task-level routing over the K groups (bound x cos). The
  router learns the grouping instead of being given it: the learned
  counterpart of the known lineage groups, without reusing (and disturbing)
  other diseases' own adapters as ``live_sources`` does. Part of the borrowed
  components, like the known groups. The group routing is never frozen and
  uses its own bound (10): in the many-source toy, with a warm-up every group
  saw the same even mixture, learned the same correction and the groups never
  separated (same-lineage diseases shared a top group 0.20 of the time, other
  lineages 0.12); trained from the start with bound 10, 0.58 against 0.14.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from opera.modules.networks.partial_pooling_readout import _MASKED, soft_normalize
from opera.modules.networks.task_routed_readout import NORMALIZE_EPS, TaskRoutedPoolingReadout

BORROW_VALUES = ("raw", "equalised", "bounded", "off")
BORROW_AMOUNTS = ("support", "learned")
BORROW_FROM = ("both", "disease", "outcome")
NORM_MOMENTUM = 0.01
LATENT_BOUND = 10.0


class FullOwnTaskRoutedReadout(TaskRoutedPoolingReadout):
    def __init__(self, hidden_size, fine_names, outcome_names, *,
                 borrow_values: str = "raw", patient_gate: bool = False,
                 borrow_amount: str = "support", borrow_from: str = "both",
                 disease_groups: dict | None = None, outcome_groups: dict | None = None,
                 live_sources: bool = False, latent_groups: int = 0, **kwargs) -> None:
        if borrow_values not in BORROW_VALUES:
            raise ValueError(f"borrow_values must be one of {BORROW_VALUES}")
        if borrow_amount not in BORROW_AMOUNTS:
            raise ValueError(f"borrow_amount must be one of {BORROW_AMOUNTS}")
        if borrow_from not in BORROW_FROM:
            raise ValueError(f"borrow_from must be one of {BORROW_FROM}")
        if kwargs.get("borrow_mix", "per_source") != "per_source":
            raise ValueError("Full own corrections use the per-source borrowed mix")
        super().__init__(hidden_size, fine_names, outcome_names, **kwargs)
        self.borrow_values = borrow_values
        self.borrow_amount = borrow_amount
        self.borrow_from = borrow_from
        self.use_patient_gate = bool(patient_gate)
        self.live_sources = bool(live_sources)
        # Running mean correction norm per source (equalised values only).
        self.register_buffer("source_norm", torch.zeros(self.n_sources))
        self.register_buffer("source_norm_seen", torch.zeros((), dtype=torch.bool))
        if self.use_patient_gate:
            self.patient_gate = nn.Parameter(torch.zeros(()))
            with torch.random.fork_rng(devices=[]):
                nn.init.normal_(self.patient_query.weight, std=0.02)
            self.patient_query.weight.requires_grad_(True)
        if borrow_amount == "learned":
            # Logit of the own share per disease x outcome, set from the
            # training support (at the tau prior) the first time it is known.
            self.own_logit = nn.Parameter(torch.zeros(self.n_disease, self.n_outcome))
            self.register_buffer("own_logit_set", torch.zeros((), dtype=torch.bool))
        self.disease_groups = dict(disease_groups or {})
        self.outcome_groups = dict(outcome_groups or {})
        self.group_names = []
        self.group_down = self.group_up = None
        if self.disease_groups or self.outcome_groups:
            index = {}
            def lookup(groups, names, prefix):
                missing = [n for n in names if n not in groups]
                if missing:
                    raise ValueError(f"No group for {prefix} {missing}")
                return [index.setdefault(f"{prefix}:{groups[n]}", len(index)) for n in names]
            disease = (lookup(self.disease_groups, self.fine_names, "disease_group")
                       if self.disease_groups else [-1] * self.n_disease)
            outcome = (lookup(self.outcome_groups, self.outcome_names, "outcome_group")
                       if self.outcome_groups else [-1] * self.n_outcome)
            self.group_names = list(index)
            self.register_buffer("disease_group_index", torch.tensor(disease, dtype=torch.long))
            self.register_buffer("outcome_group_index", torch.tensor(outcome, dtype=torch.long))
            rank, hidden = self.down.shape[1], self.hidden_size
            self.group_down = nn.Parameter(torch.empty(len(index), rank, hidden))
            self.group_up = nn.Parameter(torch.empty(len(index), hidden, rank))
            with torch.random.fork_rng(devices=[]):
                nn.init.normal_(self.group_down, std=0.02)
                nn.init.normal_(self.group_up, std=1e-3)
        if int(latent_groups) < 0:
            raise ValueError("latent_groups must be nonnegative")
        self.n_latent = int(latent_groups)
        self.latent_down = self.latent_up = self.latent_keys = None
        if self.n_latent:
            rank, hidden = self.down.shape[1], self.hidden_size
            self.latent_down = nn.Parameter(torch.empty(self.n_latent, rank, hidden))
            self.latent_up = nn.Parameter(torch.empty(self.n_latent, hidden, rank))
            self.latent_keys = nn.Parameter(torch.empty(self.n_latent, hidden))
            with torch.random.fork_rng(devices=[]):
                nn.init.normal_(self.latent_down, std=0.02)
                nn.init.normal_(self.latent_up, std=1e-3)
                nn.init.normal_(self.latent_keys, std=1.0)

    def latent_weights(self, safe_index, targets=None):
        """(batch, target, K) task-level weights over the latent groups."""
        query = self._task_query(safe_index).float()
        if targets is not None:
            query = query[:, targets]
        query = F.normalize(query, dim=-1, eps=NORMALIZE_EPS)
        keys = F.normalize(self.latent_keys.float(), dim=-1, eps=NORMALIZE_EPS)
        # Never frozen, and its own bound (see the module docstring).
        logits = LATENT_BOUND * torch.einsum("bth,kh->btk", query, keys)
        return torch.softmax(logits, dim=-1)

    def _latent_corrections(self, patient, safe_index, known, targets):
        if self.latent_down is None:
            return torch.zeros(len(patient), len(targets), self.hidden_size,
                               device=patient.device, dtype=torch.float32)
        weights = self.latent_weights(safe_index, targets)
        values = torch.einsum("bh,krh->bkr", patient, self.latent_down)
        values = torch.einsum("bkr,khr->bkh", F.gelu(values), self.latent_up).float()
        return known.float()[:, None, None] * torch.einsum("btk,bkh->bth", weights, values)

    def _group_corrections(self, patient, safe_index, known, targets):
        """Lineage and endpoint-group corrections, each (batch, target, hidden)."""
        zero = torch.zeros(len(patient), len(targets), self.hidden_size,
                           device=patient.device, dtype=torch.float32)
        if self.group_down is None:
            return zero, zero
        values = torch.einsum("bh,grh->bgr", patient, self.group_down)
        values = torch.einsum("bgr,ghr->bgh", F.gelu(values), self.group_up).float()
        mask = known.float()[:, None, None]
        disease, outcome = zero, zero
        group = self.disease_group_index[safe_index]
        if bool((group >= 0).all()):
            batch = torch.arange(len(patient), device=patient.device)
            disease = mask * values[batch, group][:, None, :].expand(-1, len(targets), -1)
        group = self.outcome_group_index[targets]
        if bool((group >= 0).all()):
            outcome = mask * values[:, group]
        return disease, outcome

    def set_cell_support(self, events) -> None:
        super().set_cell_support(events)
        if self.borrow_amount == "learned" and not bool(self.own_logit_set):
            with torch.no_grad():
                share = (self.cell_train_events.float()
                         / (self.cell_train_events.float() + self.tau_prior)).clamp(1e-3, 1 - 1e-3)
                self.own_logit.copy_(torch.log(share / (1 - share)))
                self.own_logit_set.fill_(True)

    def own_weight(self, fine_index: torch.Tensor) -> torch.Tensor:
        if self.borrow_amount == "support":
            return super().own_weight(fine_index)
        fine_index = fine_index.long().reshape(-1).to(self.own_logit.device)
        known = (fine_index >= 0) & (fine_index < self.n_disease)
        logit = self.own_logit[fine_index.clamp(0, self.n_disease - 1)]
        # How much to borrow is part of the router: frozen during its warm-up.
        if self.training and self.router_frozen:
            logit = logit.detach()
        return torch.sigmoid(logit.float()) * known[:, None].float()

    def _task_query(self, fine_index, patient=None):
        query = super()._task_query(fine_index)
        if not self.use_patient_gate or patient is None:
            return query
        task = F.normalize(query.float(), dim=-1, eps=NORMALIZE_EPS)
        person = F.normalize(self.patient_query(patient).float(), dim=-1, eps=NORMALIZE_EPS)
        return task + self.patient_gate.float() * person[:, None, :]

    def route_weights(self, patient: torch.Tensor, fine_index: torch.Tensor) -> torch.Tensor:
        if self.use_patient_gate:
            weights = self._gated_route_weights(patient, fine_index)
        else:
            weights = super().route_weights(patient, fine_index)
        if self.borrow_from == "both":
            return weights
        keep = torch.ones(self.n_sources + 1, dtype=weights.dtype, device=weights.device)
        keep[1 + self.n_disease:] = 0 if self.borrow_from == "disease" else 1
        keep[1:1 + self.n_disease] = 0 if self.borrow_from == "outcome" else 1
        kept = weights * keep
        total = kept[..., 1:].sum(-1, keepdim=True)
        # Unknown-disease fallback (slot 0, kept at 1) passes through unchanged.
        return torch.where(total > 0, kept / total.clamp_min(1e-12), kept)

    def _gated_route_weights(self, patient, fine_index):
        """As the task-routed weights, with the gated patient term in the query."""
        if patient.ndim != 2 or patient.shape[1] != self.hidden_size:
            raise ValueError("Expected (batch, hidden) patient representations")
        fine_index = fine_index.long().reshape(-1).to(patient.device)
        if fine_index.numel() != len(patient):
            raise ValueError("fine_index must have one value per patient")
        known = (fine_index >= 0) & (fine_index < self.n_disease)
        safe_index = fine_index.clamp(0, self.n_disease - 1)
        query = F.normalize(self._task_query(safe_index, patient), dim=-1, eps=NORMALIZE_EPS)
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

    def _borrowed_values(self, raw: torch.Tensor) -> torch.Tensor:
        """Borrowed source corrections from the raw (detached) ones."""
        if self.borrow_values == "bounded":
            return soft_normalize(raw)
        if self.borrow_values == "off":
            return torch.zeros_like(raw)
        factor = self._equaliser()
        return raw if factor is None else raw * factor[None, :raw.shape[1], None]

    def _source_values(self, patient, *, detached: bool) -> torch.Tensor:
        """Unbounded source corrections, (batch, source, hidden), as R_O applies them."""
        down = self.down.detach() if detached else self.down
        up = self.up.detach() if detached else self.up
        values = torch.einsum("bh,srh->bsr", patient, down)
        values = torch.einsum("bsr,shr->bsh", F.gelu(values), up)
        return values * self.source_shrinkage[None, :, None]

    @torch.no_grad()
    def _update_source_norm(self, borrowed: torch.Tensor) -> None:
        batch = borrowed.float().norm(dim=-1).mean(0)
        if bool(self.source_norm_seen):
            self.source_norm.lerp_(batch, NORM_MOMENTUM)
        else:
            self.source_norm.copy_(batch)
            self.source_norm_seen.fill_(True)

    def _equaliser(self) -> torch.Tensor | None:
        """(source,) multipliers for borrowed values; None for raw values. All
        ones until the first training batch has been seen."""
        if self.borrow_values != "equalised":
            return None
        norm = self.source_norm.float().clamp_min(1e-6)
        live = self.source_shrinkage > 0
        target = norm[live].mean() if bool(live.any()) else norm.new_ones(())
        return target / norm

    def component_vectors(self, patient, fine_index, weights, outcome_indices=None):
        fine_index = fine_index.long().reshape(-1).to(patient.device)
        known = (fine_index >= 0) & (fine_index < self.n_disease)
        safe_index = fine_index.clamp(0, self.n_disease - 1)
        targets = (torch.arange(self.n_outcome, device=patient.device)
                   if outcome_indices is None else outcome_indices.long().to(patient.device))
        if weights.shape != (len(patient), len(targets), self.n_sources + 1):
            raise ValueError("weights must be (batch, targets, 1 + sources)")
        live = self._source_values(patient, detached=False).float()
        borrowed = live if self.live_sources else self._source_values(patient, detached=True).float()
        if self.training and self.borrow_values == "equalised":
            self._update_source_norm(borrowed)
        borrowed = self._borrowed_values(borrowed)
        own_w = self.own_weight(fine_index)[:, targets].float()
        borrow_scale = (self.pool_scale.float() * (1 - own_w) * known[:, None].float())[..., None]
        mask = known.float()[:, None, None]
        batch = torch.arange(len(patient), device=patient.device)
        gates = weights[..., 1:].float()
        n = self.n_disease
        group_disease, group_outcome = self._group_corrections(patient, safe_index, known, targets)
        group_disease = group_disease + self._latent_corrections(patient, safe_index, known, targets)
        vectors = {
            # Unknown disease identity keeps the shared-path fallback: no corrections.
            "own_disease": mask * live[batch, safe_index][:, None, :].expand(-1, len(targets), -1),
            "own_outcome": mask * live[:, self.n_disease + targets],
            "borrowed_disease": group_disease + borrow_scale * torch.einsum(
                "bos,bsh->boh", gates[..., :n], borrowed[:, :n]),
            "borrowed_outcome": group_outcome + borrow_scale * torch.einsum(
                "bos,bsh->boh", gates[..., n:], borrowed[:, n:]),
        }
        general = torch.zeros_like(patient, dtype=torch.float32)
        if self.global_down is not None:
            general = F.linear(F.gelu(F.linear(patient, self.global_down)), self.global_up).float()
        vectors["global"] = general[:, None, :].expand(-1, len(targets), -1)
        return vectors

    def disease_terms(self, patient, fine_index):
        """Disease-source terms (batch, outcome, disease, hidden) with
        ``base + terms.sum(2)`` equal to the forward output (evaluation only)."""
        output, weights = self(patient, fine_index, return_weights=True)
        fine_index = fine_index.long().reshape(-1).to(patient.device)
        known = (fine_index >= 0) & (fine_index < self.n_disease)
        safe_index = fine_index.clamp(0, self.n_disease - 1)
        n = self.n_disease
        raw = self._source_values(patient, detached=True)[:, :n].float()
        values = self._borrowed_values(raw)
        own_w = self.own_weight(fine_index).float()
        borrow = (self.pool_scale.float() * (1 - own_w) * known[:, None].float())[..., None, None]
        terms = borrow * weights[..., 1:1 + n, None].float() * values[:, None]
        own = F.one_hot(safe_index, n).float() * known[:, None].float()
        terms = terms + own[:, None, :, None] * raw[:, None]
        terms = self.residual_scale * terms
        return output - terms.sum(2).to(output.dtype), terms

    def pooling_summary(self) -> dict:
        summary = super().pooling_summary()
        summary.update(own="full weight, unbounded (as R_O)", borrow_values=self.borrow_values,
                       borrow_amount=self.borrow_amount, borrow_from=self.borrow_from,
                       groups=self.group_names, live_sources=self.live_sources,
                       latent_groups=self.n_latent,
                       source_norm=self.source_norm.tolist())
        if self.use_patient_gate:
            summary["patient_gate"] = float(self.patient_gate)
        if self.borrow_amount == "learned":
            with torch.no_grad():
                summary["own_weight_by_cell"] = torch.sigmoid(self.own_logit).tolist()
        if self.n_latent:
            # Learned grouping: (disease, outcome, K) weights over the latent groups.
            with torch.no_grad():
                fine = torch.arange(self.n_disease, device=self.latent_keys.device)
                summary["latent_weights"] = self.latent_weights(fine).tolist()
        return summary
