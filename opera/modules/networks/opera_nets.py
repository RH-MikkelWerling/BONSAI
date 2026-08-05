"""
OPERA Networks — Outcome-guided EHR foundation model adaptation.

Architecture
============
BonsaiEncoder (frozen or trainable)
        ↓
  768-d CLS pooling  (or BiGRU pooling from BONSAI)
        ↓
  SharedProjectionHead  →  128-d  l2-normalised embedding
        ↓
  Per-outcome projection heads (optional, for outcome-specific spaces)

The contrastive loss is a multi-outcome SupCon variant weighted by learnable
log-sigma parameters (Kendall et al. 2018, "Multi-Task Learning Using
Uncertainty to Weigh Losses for Scene Understanding").

NOTE: The projection dimension (default 128) and the number / identity of
outcomes are the main knobs to tune.  Search for "# TUNE:" comments below.
"""

from typing import Dict, List, Mapping, Optional, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from opera.compat.bonsai import BonsaiEncoder, BiGRU, encoder_hidden_state
from opera.modules.networks.cross_outcome_weighters import (
    KendallWeighter,
    build_cross_outcome_weighter,
)
from opera.modules.networks.competing_risk import (
    PiecewiseExponentialCompetingRiskLoss,
)


def outcome_eligibility_mask(
    times: torch.Tensor,
    events: torch.Tensor,
) -> torch.Tensor:
    """Return rows with observable time and event values."""
    return (times >= 0) & (events >= 0)


# ═══════════════════════════════════════════════════════════════════════════
# Projection heads
# ═══════════════════════════════════════════════════════════════════════════


class ProjectionHead(nn.Module):
    """Two-layer MLP projection head (SimCLR-style)."""

    def __init__(
        self,
        input_dim: int = 768,
        hidden_dim: int = 256,  # TUNE: intermediate projection width
        output_dim: int = 128,  # TUNE: contrastive embedding dimension
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.net(x), dim=-1)


class FamilyProjectionHead(nn.Module):
    """Shared projection trunk with a small final head per outcome family."""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, families):
        super().__init__()
        family_names = list(dict.fromkeys(str(name) for name in families))
        if not family_names:
            raise ValueError("Family projection requires at least one family.")
        self.trunk = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.GELU())
        self.heads = nn.ModuleDict(
            {name: nn.Linear(hidden_dim, output_dim) for name in family_names}
        )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        shared = self.trunk(x)
        return {
            family: F.normalize(head(shared), dim=-1)
            for family, head in self.heads.items()
        }


# ═══════════════════════════════════════════════════════════════════════════
# Multi-outcome contrastive loss (SupCon + Kendall weighting)
# ═══════════════════════════════════════════════════════════════════════════


class _LegacySurvivalSoftContrastiveLoss(nn.Module):
    """
    Survival-informed soft contrastive loss for a *single* outcome.

    Rather than binarising outcomes (event / no-event) and using SupCon,
    we use the full time-to-event information to construct *continuous*
    pair weights.  Two patients are "similar" in proportion to how close
    their survival times are, discounted by censoring uncertainty.

    Pair weight  w(i,j) = time_similarity(i,j) * reliability(i,j)
                        * dapt_prior(i,j)                            [optional]

    where:
        time_similarity = exp( -|q_i - q_j| / cdf_scale )
            q_i = empirical-CDF position of patient i's event/censoring time
            The CDF transform makes the loss scale-invariant while preserving
            event-density information from the training population.
            Patients in opposite quartiles get weight exp(-4) ≈ 0.018;
            adjacent-quartile patients get exp(-1) ≈ 0.37.
        reliability     = 1.0  if both primary events observed
                        = 1.0  if primary event + competing death (confirmed non-event)
                        = 0.5  if primary event observed before admin-censored patient
                        = 0.3  if both competing deaths
                        = 0.2  if competing death + admin censored
                        = 0.1–0.3 if both admin censored (scaled by min quantile)

    The soft contrastive loss then maximises the weighted log-similarity:
        L = -Σ_i  Σ_j  w(i,j) * log_softmax( sim(z_i,z_j) / τ )_j

    This degrades gracefully:
      - When all patients have the same survival time, all pairs are equal
        (uniform contrastive, no gradient).
      - When survival times are spread widely, distant-time pairs get
        near-zero weight, recovering SupCon-like behaviour.

    cdf_scale is shared across outcomes because raw times are transformed to
    empirical event-time CDF positions in [0, 1].

    Reference framing: a continuous generalisation of
        Khosla et al. "Supervised Contrastive Learning" (NeurIPS 2020)
    combined with the censoring-aware pair construction from
        Lee et al. "DeepHit" (AAAI 2018).
    """

    def __init__(
        self,
        temperature: float = 0.07,  # TUNE: contrastive temperature
        cdf_scale: float = 0.25,  # quartile distance gives exp(-1)
        min_weight_threshold: float = 1e-4,  # ignore near-zero-weight pairs
    ):
        super().__init__()
        self.temperature = temperature
        self.cdf_scale = cdf_scale
        self.min_weight_threshold = min_weight_threshold

    def forward(
        self,
        embeddings: torch.Tensor,
        times: torch.Tensor,
        events: torch.Tensor,
        sorted_event_times: torch.Tensor,
        dapt_weights: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            embeddings  : (B, D) l2-normalised embeddings.
            times       : (B,)   time-to-event or censoring in days.
            events      : (B,)   1 = primary event, 0 = admin censored,
                                 2 = competing death (confirmed non-primary-event).
            sorted_event_times: sorted observed training event times for this outcome.
            dapt_weights: (B, B) optional pre-computed cohort similarity
                                 from DAPT embeddings, values in (0, 1].
        Returns:
            Scalar loss.
        """
        device = embeddings.device
        B = embeddings.size(0)
        if B <= 1:
            return torch.tensor(0.0, device=device, requires_grad=True)

        # -- 1. Time similarity (population CDF, precomputed) -------------
        # Observed events use an inclusive event quantile. Censored patients
        # use a lower-bound quantile: the fraction of events they outlived.
        sorted_et = sorted_event_times.to(device)
        n_et = max(sorted_et.numel(), 1)
        positions = torch.searchsorted(
            sorted_et.contiguous(),
            times.float().contiguous(),
            right=True,
        )
        q = (positions.float() / n_et).clamp(0.0, 1.0)

        q_i = q.unsqueeze(1)
        q_j = q.unsqueeze(0)
        time_sim = torch.exp(-torch.abs(q_i - q_j) / self.cdf_scale)
        # -- 2. Reliability mask (competing-risk-aware) -------------------
        # Use explicit boolean masks — avoids broken arithmetic when event=2.
        obs_i = (events == 1).unsqueeze(1).float()  # (B, 1) primary event
        obs_j = (events == 1).unsqueeze(0).float()  # (1, B)
        cens_i = (events == 0).unsqueeze(1).float()  # (B, 1) admin censored
        cens_j = (events == 0).unsqueeze(0).float()  # (1, B)
        comp_i = (events == 2).unsqueeze(1).float()  # (B, 1) competing death
        comp_j = (events == 2).unsqueeze(0).float()  # (1, B)

        # Pair reliability by combination:
        #   obs  + obs  → 1.0  both primary events, timing fully known
        #   obs  + comp → 1.0  competing death is a confirmed non-event;
        #                      ordering certain regardless of time direction
        #   obs  + cens → 0.5  only when obs happened before cens (ordering known)
        #   comp + comp → 0.3  both confirmed non-events, max both-censored weight
        #   comp + cens → 0.2  one confirmed, one uncertain
        #   cens + cens → 0.1–0.3  both uncertain, scaled by min quantile
        both_obs = obs_i * obs_j
        obs_comp = obs_i * comp_j + comp_i * obs_j
        obs_cens_ord = (
            obs_i * cens_j * (q_i <= q_j).float()
            + cens_i * obs_j * (q_j <= q_i).float()
        )
        both_comp = comp_i * comp_j
        comp_cens = comp_i * cens_j + cens_i * comp_j
        both_cens = cens_i * cens_j
        both_cens_reliability = (0.1 + 0.2 * torch.minimum(q_i, q_j)).clamp(0.1, 0.3)

        reliability = (
            1.0 * both_obs
            + 1.0 * obs_comp
            + 0.5 * obs_cens_ord
            + 0.3 * both_comp
            + 0.2 * comp_cens
            + both_cens_reliability * both_cens
        )  # (B, B), in [0.1, 1.0]

        # ── 3. Pair weights ────────────────────────────────────────────────
        pair_weights = time_sim * reliability  # (B, B)

        if dapt_weights is not None:
            pair_weights = pair_weights * dapt_weights.to(device)

        # Zero out diagonal (self-pairs)
        mask_self = torch.eye(B, device=device, dtype=torch.bool)
        pair_weights = pair_weights.masked_fill(mask_self, 0.0)

        # Skip anchors where all pair weights are negligible
        row_sum = pair_weights.sum(dim=1)
        valid = row_sum > self.min_weight_threshold
        if valid.sum() == 0:
            return torch.tensor(0.0, device=device, requires_grad=True)

        # Normalise weights per anchor so they sum to 1
        pair_weights = pair_weights / (row_sum.unsqueeze(1) + 1e-12)

        # ── 4. Soft contrastive objective ──────────────────────────────────
        sim = torch.mm(embeddings, embeddings.t()) / self.temperature  # (B, B)

        # Numerically stable log-softmax
        logits_max = sim.detach().max(dim=1, keepdim=True)[0]
        logits = sim - logits_max
        exp_logits = torch.exp(logits).masked_fill(mask_self, 0.0)
        log_softmax = logits - torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-12)

        # Weighted cross-entropy over valid anchors
        per_anchor_loss = -(pair_weights[valid] * log_softmax[valid]).sum(dim=1)
        anchor_weights = row_sum[valid] / (row_sum[valid].sum() + 1e-12)
        loss = (anchor_weights * per_anchor_loss).sum()
        return loss


class SurvivalSoftContrastiveLoss(nn.Module):
    """Survival-soft contrastive loss with KM event-mass similarity.

    Admin-censored patients are encoded as a conditional distribution over
    plausible future primary-event locations under the Kaplan-Meier event-time
    mass. Production excludes competing deaths from target-event imputation
    because the proper competing-risk likelihood models death explicitly.
    ``hard_negative`` retains the historical cause-specific-censoring geometry
    as an ablation.
    """

    def __init__(
        self,
        temperature: float = 0.07,
        km_time_scale: float = 0.25,
        cdf_scale: Optional[float] = None,
        min_weight_threshold: float = 1e-4,
        competing_event_weight: float = 0.0,
        competing_event_handling: str = "hard_negative",
    ):
        super().__init__()
        allowed = {"censor", "exclude", "hard_negative", "reliability"}
        if competing_event_handling not in allowed:
            raise ValueError(
                "competing_event_handling must be one of "
                f"{sorted(allowed)}; got {competing_event_handling!r}."
            )
        if competing_event_handling == "reliability":
            raise NotImplementedError(
                "competing_event_handling='reliability' is reserved for a "
                "future reliability-weighted competing-risk sensitivity run."
            )
        self.temperature = temperature
        if cdf_scale is not None:
            km_time_scale = cdf_scale
        self.km_time_scale = km_time_scale
        self.cdf_scale = km_time_scale
        self.min_weight_threshold = min_weight_threshold
        self.competing_event_weight = competing_event_weight
        self.competing_event_handling = competing_event_handling

    def _event_grid(
        self,
        sorted_event_times: torch.Tensor,
        event_time_probs: Optional[torch.Tensor],
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return event times, cumulative event-mass grid, and event masses.

        NOTE: when event_time_probs is None this builds an empirical event CDF
        (equal mass 1/N per observed event), NOT a true Kaplan-Meier estimator.
        A true KM would apply a risk-set correction: d_i / n_i at each event
        time where n_i counts all patients still under observation (including
        censored).  The empirical CDF ignores censored patients in the
        denominator, so it overestimates early event rates and compresses
        quantile spacing in censoring-heavy outcomes (e.g. aki_30d).
        The bias is acceptable for representation learning but should not be
        compared to a published KM curve.  Proper KM grids can be injected via
        the outcome_event_time_probs argument to MultiOutcomeSurvivalLoss.
        """
        times = sorted_event_times.to(device).float().contiguous()
        if times.numel() == 0:
            probs = torch.tensor([1.0], device=device)
            return (
                torch.tensor([], device=device),
                probs,
                probs,
            )
        if event_time_probs is None:
            probs = torch.full(
                (times.numel(),), 1.0 / float(times.numel()), device=device
            )
        else:
            probs = event_time_probs.to(device).float().contiguous()
            if probs.numel() != times.numel():
                raise ValueError(
                    "event_time_probs must match sorted_event_times length; "
                    f"got {probs.numel()} and {times.numel()}."
                )
            probs = probs / probs.sum().clamp_min(1e-12)
        km_grid = torch.cat(
            [
                torch.tensor([0.0], device=device),
                torch.cumsum(probs, dim=0).clamp(0.0, 1.0),
            ]
        )
        prob_grid = torch.cat([torch.tensor([0.0], device=device), probs])
        return times, km_grid, prob_grid

    def _patient_quantile_distributions(
        self,
        times: torch.Tensor,
        events: torch.Tensor,
        event_time_grid: torch.Tensor,
        km_grid: torch.Tensor,
        event_time_probs: torch.Tensor,
    ) -> torch.Tensor:
        B = times.numel()
        G = km_grid.numel()
        dist = torch.zeros((B, G), device=times.device, dtype=torch.float32)
        positions = torch.searchsorted(
            event_time_grid,
            times.float().contiguous(),
            right=True,
        )
        # clamp to [1, G-1]: slot 0 is the km_grid 0.0 sentinel (before any
        # events) and must never hold patient mass.
        exact_idx = positions.clamp(1, G - 1)

        distribution_events = events
        if self.competing_event_handling == "hard_negative":
            # Recode competing deaths (event=2) as administratively censored at
            # death time.  This is equivalent to cause-specific hazard censoring
            # and requires NON-INFORMATIVE censoring: the hazard of competing
            # death must be independent of the primary-event hazard conditional
            # on covariates.  In leukemia registries this assumption may be
            # violated (treatment-related death is correlated with disease
            # aggressiveness which also drives AKI, infections, etc.).
            # Run a sensitivity with competing_event_handling: censor,
            # competing_event_weight: 0.0 to assess the impact.
            distribution_events = torch.where(
                events == 2,
                torch.zeros_like(events),
                events,
            )

        exact_mask = distribution_events == 1
        if self.competing_event_handling in {"censor", "exclude"}:
            exact_mask = exact_mask | (events == 2)
        if exact_mask.any():
            dist[exact_mask, exact_idx[exact_mask]] = 1.0

        cens_mask = distribution_events == 0
        if cens_mask.any():
            cens_idx = torch.where(cens_mask)[0]
            # (C, E): True when the event-time grid point is strictly after the
            # patient's censoring time, i.e. the patient could still have the event
            future = event_time_grid.unsqueeze(0) > times[cens_idx].unsqueeze(1)
            # Prepend False for prob_grid[0] (the "before any events" sentinel)
            tail_full = torch.cat(
                [
                    torch.zeros(
                        len(cens_idx), 1, dtype=torch.bool, device=times.device
                    ),
                    future,
                ],
                dim=1,
            )  # (C, G)
            masked = event_time_probs.unsqueeze(0) * tail_full.float()  # (C, G)
            row_sums = masked.sum(dim=1, keepdim=True).clamp_min(1e-12)  # (C, 1)
            normalized = masked / row_sums  # (C, G)
            # Patients censored past the last observed event time get all mass at G-1
            no_future = ~tail_full.any(dim=1)  # (C,)
            if no_future.any():
                normalized[no_future] = 0.0
                normalized[no_future, -1] = 1.0
            dist[cens_idx] = normalized
        return dist

    def _compute_pair_weights(
        self,
        times: torch.Tensor,
        events: torch.Tensor,
        sorted_event_times: torch.Tensor,
        event_time_probs: Optional[torch.Tensor] = None,
        dapt_weights: Optional[torch.Tensor] = None,
        subject_ids: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        device = times.device
        if sorted_event_times.numel() == 0:
            return (
                torch.zeros((times.numel(), times.numel()), device=device),
                torch.tensor(0.0, device=device),
            )
        event_time_grid, km_grid, event_time_probs = self._event_grid(
            sorted_event_times,
            event_time_probs,
            device,
        )
        km_dist = self._patient_quantile_distributions(
            times.float(),
            events.long(),
            event_time_grid,
            km_grid,
            event_time_probs,
        )
        km_kernel = torch.exp(
            -torch.abs(km_grid.unsqueeze(1) - km_grid.unsqueeze(0)) / self.km_time_scale
        )
        pair_weights = km_dist @ km_kernel @ km_dist.t()

        comp = events == 2
        if self.competing_event_handling in {"censor", "exclude"} and comp.any():
            # OR: downweight any pair involving at least one competing-event patient
            # (XOR would miss competing+competing pairs, leaving them unreliably weighted)
            any_comp = comp.unsqueeze(1) | comp.unsqueeze(0)
            competing_weight = (
                0.0
                if self.competing_event_handling == "exclude"
                else float(self.competing_event_weight)
            )
            pair_weights = torch.where(
                any_comp,
                pair_weights * competing_weight,
                pair_weights,
            )

        excluded_pairs = torch.eye(times.numel(), device=device, dtype=torch.bool)
        if subject_ids is not None:
            ids = subject_ids.to(device).reshape(-1)
            excluded_pairs |= ids.unsqueeze(0) == ids.unsqueeze(1)
        pair_weights = pair_weights.masked_fill(excluded_pairs, 0.0)
        if dapt_weights is not None:
            pair_weights = pair_weights * dapt_weights.to(device)

        weight_sum = pair_weights.sum()
        n_eff = (weight_sum * weight_sum) / (pair_weights.square().sum() + 1e-12)
        return pair_weights, n_eff.detach()

    def forward(
        self,
        embeddings: torch.Tensor,
        times: torch.Tensor,
        events: torch.Tensor,
        sorted_event_times: torch.Tensor,
        event_time_probs: Optional[torch.Tensor] = None,
        dapt_weights: Optional[torch.Tensor] = None,
        subject_ids: Optional[torch.Tensor] = None,
        return_diagnostics: bool = False,
    ) -> torch.Tensor:
        device = embeddings.device
        B = embeddings.size(0)
        if B <= 1:
            zero = torch.tensor(0.0, device=device, requires_grad=True)
            if return_diagnostics:
                return zero, {"n_effective_pairs": torch.tensor(0.0, device=device)}
            return zero

        pair_weights, n_eff = self._compute_pair_weights(
            times=times,
            events=events,
            sorted_event_times=sorted_event_times,
            event_time_probs=event_time_probs,
            dapt_weights=dapt_weights,
            subject_ids=subject_ids,
        )
        excluded_pairs = torch.eye(B, device=device, dtype=torch.bool)
        if subject_ids is not None:
            ids = subject_ids.to(device).reshape(-1)
            excluded_pairs |= ids.unsqueeze(0) == ids.unsqueeze(1)
        row_sum = pair_weights.sum(dim=1)
        valid = row_sum > self.min_weight_threshold
        if valid.sum() == 0:
            zero = torch.tensor(0.0, device=device, requires_grad=True)
            if return_diagnostics:
                return zero, {
                    "n_effective_pairs": n_eff,
                    "target_entropy": zero.detach(),
                    "excess_loss": zero,
                    "uniform_baseline": zero.detach(),
                    "contrastive_headroom": zero.detach(),
                }
            return zero

        pair_weights = pair_weights / (row_sum.unsqueeze(1) + 1e-12)
        sim = torch.mm(embeddings, embeddings.t()) / self.temperature
        logits_max = sim.detach().max(dim=1, keepdim=True)[0]
        logits = sim - logits_max
        exp_logits = torch.exp(logits).masked_fill(excluded_pairs, 0.0)
        log_softmax = logits - torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-12)
        per_anchor_loss = -(pair_weights[valid] * log_softmax[valid]).sum(dim=1)
        anchor_weights = row_sum[valid] / (row_sum[valid].sum() + 1e-12)
        loss = (anchor_weights * per_anchor_loss).sum()
        target_entropy_per_anchor = -(
            pair_weights[valid] * torch.log(pair_weights[valid].clamp_min(1e-12))
        ).sum(dim=1)
        target_entropy = (anchor_weights * target_entropy_per_anchor).sum()
        candidate_counts = (~excluded_pairs[valid]).sum(dim=1).clamp_min(1)
        uniform_baseline = (
            anchor_weights * torch.log(candidate_counts.to(loss.dtype))
        ).sum()
        excess_loss = loss - target_entropy
        contrastive_headroom = (uniform_baseline - target_entropy).clamp_min(0.0)
        if not loss.requires_grad:
            loss.requires_grad_()
        if return_diagnostics:
            return loss, {
                "n_effective_pairs": n_eff,
                "target_entropy": target_entropy.detach(),
                "excess_loss": excess_loss,
                "uniform_baseline": uniform_baseline.detach(),
                "contrastive_headroom": contrastive_headroom.detach(),
            }
        return loss


class _LegacyMultiOutcomeSurvivalLoss(nn.Module):
    """
    Aggregates per-outcome survival-soft-contrastive losses using learnable
    log-variance (Kendall et al. 2018):

        L_total = Σ_k  [ (1 / 2σ_k²) · L_k  +  log(σ_k) ]

    Replaces the binary MultiOutcomeContrastiveLoss with the full
    time-to-event formulation.

    Scientific value of σ_k:
        Low  σ_k  → outcome k strongly structures the embedding space.
        High σ_k  → outcome k is noisy / uninformative for representation.

    The σ values are a publishable finding *independent of AUC gains*:
    they reveal which clinical outcomes encode the latent patient structure
    that the model learns.

    DAPT-prior cohort similarity:
        If ``dapt_embeddings`` is provided, pair weights are further
        modulated by the cosine similarity of the patients' DAPT-stage
        representations.  This implements the meta-learning prior:
        "learn primarily from patients whose clinical histories resemble mine,
        but allow cross-disease signal to flow with a λ-floor."

        The DAPT representations are *frozen* and pre-computed; they serve
        as a fixed prior, not a trainable component.
    """

    def __init__(
        self,
        outcome_names: List[str],
        temperature: float = 0.07,
        km_time_scale: float = 0.25,
        outcome_sorted_event_times: Optional[Dict[str, torch.Tensor]] = None,
        dapt_lambda_floor: float = 0.55,  # TUNE: cross-disease floor weight
        outcome_event_time_probs: Optional[Dict[str, torch.Tensor]] = None,
        competing_event_weight: float = 0.0,
        competing_event_handling: str = "hard_negative",
        effective_pair_normalization: bool = True,
    ):
        super().__init__()
        self.outcome_names = outcome_names
        self.n_outcomes = len(outcome_names)
        self.dapt_lambda_floor = dapt_lambda_floor
        self.outcome_sorted_event_times = outcome_sorted_event_times or {}
        self.outcome_event_time_probs = outcome_event_time_probs or {}
        self.effective_pair_normalization = effective_pair_normalization

        self.survival_con = SurvivalSoftContrastiveLoss(
            temperature=temperature,
            km_time_scale=km_time_scale,
            competing_event_weight=competing_event_weight,
            competing_event_handling=competing_event_handling,
        )

        # Learnable log-variance per outcome  (initialised to 0 → σ = 1)
        self.log_sigma = nn.Parameter(torch.zeros(self.n_outcomes))

    def _compute_dapt_weights(
        self,
        subject_ids: torch.Tensor,
        dapt_embedding_store: Optional[Dict],
    ) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Compute (B, B) cohort-similarity weight matrix from pre-computed
        DAPT embeddings.  Returns None if no store is provided.
        """
        if dapt_embedding_store is None or not dapt_embedding_store:
            return None, None

        ids = subject_ids.cpu().tolist()
        template = torch.as_tensor(next(iter(dapt_embedding_store.values())))
        embs = []
        known = []
        for sid in ids:
            if sid in dapt_embedding_store:
                embs.append(torch.as_tensor(dapt_embedding_store[sid]))
                known.append(True)
            else:
                # Unknown subject: keep shape only, then make its pairs neutral.
                embs.append(torch.zeros_like(template))
                known.append(False)

        embs = torch.stack(embs).float()
        embs = F.normalize(embs, dim=-1)
        known_mask = torch.tensor(known, dtype=torch.bool, device=embs.device)

        # Cosine similarity in [-1, 1] → map to [0, 1]
        cos_sim = torch.mm(embs, embs.t())
        cos_sim = (cos_sim + 1.0) / 2.0

        # Apply λ-floor: even cross-disease pairs contribute at rate λ
        dapt_weights = self.dapt_lambda_floor + (1.0 - self.dapt_lambda_floor) * cos_sim

        # Missing DAPT embeddings should not imply moderate similarity.
        # Pairs involving an unknown subject get weight 1.0, so the
        # outcome/time/censoring weights are left unchanged for that pair.
        pair_known = known_mask.unsqueeze(0) & known_mask.unsqueeze(1)
        dapt_weights = torch.where(
            pair_known, dapt_weights, torch.ones_like(dapt_weights)
        )
        return dapt_weights, known_mask  # (B, B), (B,)

    def forward(
        self,
        embeddings: torch.Tensor,
        outcome_survival: Dict[str, Dict[str, torch.Tensor]],
        subject_ids: Optional[torch.Tensor] = None,
        dapt_embedding_store: Optional[Dict] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            embeddings       : (B, D) l2-normalised.
            outcome_survival : dict mapping outcome name →
                               {"times": (B,) float, "events": (B,) int}
                               times = -1 and events = -1 mean missing.
            subject_ids      : (B,) int tensor of subject IDs (for DAPT lookup).
            dapt_embedding_store : dict {subject_id: embedding_tensor} or None.
        Returns:
            dict with keys "loss", "loss/<outcome>", "sigma/<outcome>",
                           "precision/<outcome>", "n_valid_pairs/<outcome>".
        """
        device = embeddings.device
        total_loss = torch.tensor(0.0, device=device, requires_grad=True)
        log_dict: Dict[str, torch.Tensor] = {}

        # Pre-compute DAPT weights once for the whole batch
        dapt_weights = None
        dapt_known_mask = None
        if subject_ids is not None and dapt_embedding_store is not None:
            dapt_weights, dapt_known_mask = self._compute_dapt_weights(
                subject_ids, dapt_embedding_store
            )
            if dapt_weights is not None:
                log_dict["dapt/weight_mean"] = dapt_weights.mean().detach()
                log_dict["dapt/weight_std"] = dapt_weights.std(unbiased=False).detach()
                log_dict["dapt/weight_min"] = dapt_weights.min().detach()
                log_dict["dapt/weight_max"] = dapt_weights.max().detach()
                if dapt_known_mask is not None:
                    log_dict["dapt/coverage"] = dapt_known_mask.float().mean().detach()

        for k, name in enumerate(self.outcome_names):
            survival_k = outcome_survival.get(name, {})
            times_k = survival_k.get("times", None)
            events_k = survival_k.get("events", None)

            if times_k is None or events_k is None:
                continue

            # Valid mask: both time and event indicator must be non-negative
            valid_mask = (times_k >= 0) & (events_k >= 0)
            if valid_mask.sum() < 2:
                continue

            emb_k = embeddings[valid_mask]
            times_v = times_k[valid_mask]
            events_v = events_k[valid_mask]

            dapt_w_k = None
            if dapt_weights is not None:
                idx = torch.where(valid_mask)[0]
                dapt_w_k = dapt_weights[idx][:, idx]

            sorted_et_k = self.outcome_sorted_event_times.get(name)
            if sorted_et_k is None:
                raise ValueError(
                    f"No sorted_event_times provided for outcome '{name}'. "
                    "Pass outcome_sorted_event_times to MultiOutcomeSurvivalLoss."
                )
            loss_k, diagnostics_k = self.survival_con(
                emb_k,
                times_v,
                events_v,
                sorted_event_times=sorted_et_k,
                event_time_probs=self.outcome_event_time_probs.get(name),
                dapt_weights=dapt_w_k,
                subject_ids=(
                    subject_ids[valid_mask] if subject_ids is not None else None
                ),
                return_diagnostics=True,
            )

            # Kendall weighting
            n_eff = diagnostics_k["n_effective_pairs"].to(device)
            max_pairs = max(
                float(valid_mask.sum().item() * (valid_mask.sum().item() - 1)), 1.0
            )
            effective_pair_fraction = (n_eff / max_pairs).clamp(1e-6, 1.0)
            sigma_loss_k = (
                loss_k * torch.sqrt(effective_pair_fraction)
                if self.effective_pair_normalization
                else loss_k
            )
            precision = 0.5 * torch.exp(-2.0 * self.log_sigma[k])
            weighted_loss_k = precision * sigma_loss_k + self.log_sigma[k]
            total_loss = total_loss + weighted_loss_k

            sigma_k = torch.exp(self.log_sigma[k])
            log_dict[f"loss/{name}"] = loss_k.detach()
            log_dict[f"loss_sigma_input/{name}"] = sigma_loss_k.detach()
            log_dict[f"sigma/{name}"] = sigma_k.detach()
            log_dict[f"precision/{name}"] = precision.detach()
            log_dict[f"n_valid/{name}"] = valid_mask.sum().float().detach()
            log_dict[f"n_effective_pairs/{name}"] = n_eff.detach()
            log_dict[f"effective_pair_fraction/{name}"] = (
                effective_pair_fraction.detach()
            )

        log_dict["loss"] = total_loss
        return log_dict


# ── Backwards-compatible alias ─────────────────────────────────────────────
# Keep the old binary loss available for ablations / unit tests.


class MultiOutcomeSurvivalLoss(_LegacyMultiOutcomeSurvivalLoss):
    """Survival contrastive loss with pluggable cross-outcome aggregation.

    Direct construction without ``cross_outcome_config`` preserves the legacy
    Kendall plus pooled behavior. New training configs can select uniform or
    FAMO weighting explicitly.
    """

    def __init__(
        self,
        outcome_names: List[str],
        temperature: float = 0.07,
        km_time_scale: float = 0.25,
        outcome_sorted_event_times: Optional[Dict[str, torch.Tensor]] = None,
        dapt_lambda_floor: float = 0.55,
        outcome_event_time_probs: Optional[Dict[str, torch.Tensor]] = None,
        competing_event_weight: float = 0.0,
        competing_event_handling: str = "hard_negative",
        effective_pair_normalization: bool = True,
        cross_outcome_config: Optional[Mapping[str, object]] = None,
    ):
        super().__init__(
            outcome_names=outcome_names,
            temperature=temperature,
            km_time_scale=km_time_scale,
            outcome_sorted_event_times=outcome_sorted_event_times,
            dapt_lambda_floor=dapt_lambda_floor,
            outcome_event_time_probs=outcome_event_time_probs,
            competing_event_weight=competing_event_weight,
            competing_event_handling=competing_event_handling,
            effective_pair_normalization=effective_pair_normalization,
        )
        del self.log_sigma
        (
            self.weighter,
            self.aggregation,
            class_balance_factors,
        ) = build_cross_outcome_weighter(outcome_names, cross_outcome_config)
        settings = dict(cross_outcome_config or {})
        families = settings.get("outcome_families", {})
        self.outcome_family = {
            outcome: family
            for family, members in families.items()
            for outcome in members
        }
        self.family_weights = dict(settings.get("family_weights", {}))
        self.support_tau_locations = float(settings.get("support_tau_locations", 100.0))
        self.event_location_counts = dict(settings.get("event_location_counts", {}))
        self.contrastive_term = str(
            settings.get("contrastive_term", "cross_entropy")
        ).lower()
        if self.contrastive_term not in {"cross_entropy", "kl"}:
            raise ValueError(
                "cross_outcome.contrastive_term must be 'cross_entropy' or 'kl'."
            )
        default_scale_mode = (
            "effective_pairs" if effective_pair_normalization else "none"
        )
        self.outcome_scale_mode = str(
            settings.get("outcome_scale_mode", default_scale_mode)
        ).lower()
        if self.outcome_scale_mode not in {"none", "effective_pairs", "initial_kl"}:
            raise ValueError(
                "cross_outcome.outcome_scale_mode must be one of: none, "
                "effective_pairs, initial_kl."
            )
        reference_scales = settings.get("outcome_reference_scales", {})
        if not isinstance(reference_scales, Mapping):
            raise ValueError(
                "cross_outcome.outcome_reference_scales must be a mapping."
            )
        self.outcome_reference_scales = {
            str(name): float(value) for name, value in reference_scales.items()
        }
        if self.outcome_scale_mode == "initial_kl":
            missing_scales = sorted(
                set(outcome_names) - set(self.outcome_reference_scales)
            )
            invalid_scales = sorted(
                name
                for name, value in self.outcome_reference_scales.items()
                if name in outcome_names
                and (not torch.isfinite(torch.tensor(value)) or value <= 0)
            )
            if missing_scales or invalid_scales:
                raise ValueError(
                    "initial_kl scaling requires one finite positive reference per "
                    f"outcome; missing={missing_scales}, invalid={invalid_scales}."
                )
        if self.aggregation == "hierarchical_support":
            missing = set(outcome_names) - set(self.outcome_family)
            extra = set(self.outcome_family) - set(outcome_names)
            if missing or extra:
                raise ValueError(
                    f"outcome_families must cover configured outcomes exactly; "
                    f"missing={sorted(missing)}, extra={sorted(extra)}"
                )
        self.register_buffer(
            "class_balance_factors",
            class_balance_factors,
            persistent=False,
        )

    @property
    def log_sigma(self) -> torch.Tensor:
        """Expose the legacy Kendall parameter for analysis utilities."""
        if not isinstance(self.weighter, KendallWeighter):
            raise AttributeError("log_sigma is only available with Kendall weighting.")
        return self.weighter.log_sigma

    @log_sigma.deleter
    def log_sigma(self) -> None:
        self._parameters.pop("log_sigma", None)

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        old_key = f"{prefix}log_sigma"
        new_key = f"{prefix}weighter.log_sigma"
        if (
            isinstance(self.weighter, KendallWeighter)
            and old_key in state_dict
            and new_key not in state_dict
        ):
            state_dict[new_key] = state_dict.pop(old_key)
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    def compute_per_outcome_losses(
        self,
        embeddings: Union[torch.Tensor, Mapping[str, torch.Tensor]],
        outcome_survival: Dict[str, Dict[str, torch.Tensor]],
        subject_ids: Optional[torch.Tensor] = None,
        dapt_embedding_store: Optional[Dict] = None,
    ) -> tuple[Dict[str, Dict[str, torch.Tensor]], Dict[str, torch.Tensor]]:
        """Return differentiable outcome terms before final aggregation."""
        template_embedding = (
            embeddings
            if isinstance(embeddings, torch.Tensor)
            else next(iter(embeddings.values()))
        )
        device = template_embedding.device
        terms: Dict[str, Dict[str, torch.Tensor]] = {}
        log_dict: Dict[str, torch.Tensor] = {}

        dapt_weights = None
        dapt_known_mask = None
        if subject_ids is not None and dapt_embedding_store is not None:
            dapt_weights, dapt_known_mask = self._compute_dapt_weights(
                subject_ids,
                dapt_embedding_store,
            )
            if dapt_weights is not None:
                dapt_weights = dapt_weights.to(device)
                log_dict["dapt/weight_mean"] = dapt_weights.mean().detach()
                log_dict["dapt/weight_std"] = dapt_weights.std(unbiased=False).detach()
                log_dict["dapt/weight_min"] = dapt_weights.min().detach()
                log_dict["dapt/weight_max"] = dapt_weights.max().detach()
                if dapt_known_mask is not None:
                    log_dict["dapt/coverage"] = dapt_known_mask.float().mean().detach()

        for name in self.outcome_names:
            survival = outcome_survival.get(name, {})
            times = survival.get("times")
            events = survival.get("events")
            if times is None or events is None:
                continue

            valid_mask = outcome_eligibility_mask(times, events)
            log_dict[f"n_valid/{name}"] = valid_mask.sum().float().detach()
            if int(valid_mask.sum().item()) < 2:
                continue

            valid_indices = torch.where(valid_mask)[0]
            outcome_dapt_weights = None
            if dapt_weights is not None:
                outcome_dapt_weights = dapt_weights[valid_indices][:, valid_indices]

            sorted_event_times = self.outcome_sorted_event_times.get(name)
            if sorted_event_times is None:
                raise ValueError(
                    f"No sorted_event_times provided for outcome {name!r}. "
                    "Pass outcome_sorted_event_times to MultiOutcomeSurvivalLoss."
                )
            outcome_embeddings = (
                embeddings
                if isinstance(embeddings, torch.Tensor)
                else embeddings[self.outcome_family[name]]
            )
            loss, diagnostics = self.survival_con(
                outcome_embeddings[valid_mask],
                times[valid_mask],
                events[valid_mask],
                sorted_event_times=sorted_event_times,
                event_time_probs=self.outcome_event_time_probs.get(name),
                dapt_weights=outcome_dapt_weights,
                subject_ids=(
                    subject_ids[valid_mask] if subject_ids is not None else None
                ),
                return_diagnostics=True,
            )

            n_effective_pairs = diagnostics["n_effective_pairs"].to(device)
            n_valid = int(valid_mask.sum().item())
            max_pairs = max(float(n_valid * (n_valid - 1)), 1.0)
            effective_pair_fraction = (n_effective_pairs / max_pairs).clamp(1e-6, 1.0)
            excess_loss = diagnostics["excess_loss"].to(device)
            objective_loss = excess_loss if self.contrastive_term == "kl" else loss
            scale_mode = self.outcome_scale_mode
            if scale_mode == "effective_pairs":
                scale_factor = torch.sqrt(effective_pair_fraction)
            elif scale_mode == "initial_kl":
                scale_factor = objective_loss.new_tensor(
                    1.0 / self.outcome_reference_scales[name]
                )
            else:
                scale_factor = objective_loss.new_tensor(1.0)
            aggregation_loss = objective_loss * scale_factor
            terms[name] = {
                "loss": loss,
                "target_entropy": diagnostics["target_entropy"].to(device),
                "excess_loss": excess_loss,
                "uniform_baseline": diagnostics["uniform_baseline"].to(device),
                "contrastive_headroom": diagnostics["contrastive_headroom"].to(device),
                "aggregation_loss": aggregation_loss,
                "valid_mask": valid_mask,
                "n_effective_pairs": n_effective_pairs,
                "effective_pair_fraction": effective_pair_fraction,
                "outcome_scale_factor": scale_factor,
            }
            log_dict[f"loss/{name}"] = loss.detach()
            log_dict[f"target_entropy/{name}"] = terms[name]["target_entropy"].detach()
            log_dict[f"excess_loss/{name}"] = excess_loss.detach()
            log_dict[f"contrastive_headroom/{name}"] = terms[name][
                "contrastive_headroom"
            ].detach()
            log_dict[f"loss_sigma_input/{name}"] = aggregation_loss.detach()
            log_dict[f"n_effective_pairs/{name}"] = n_effective_pairs.detach()
            log_dict[f"effective_pair_fraction/{name}"] = (
                effective_pair_fraction.detach()
            )
            log_dict[f"outcome_scale_factor/{name}"] = scale_factor.detach()
        return terms, log_dict

    def forward(
        self,
        embeddings: Union[torch.Tensor, Mapping[str, torch.Tensor]],
        outcome_survival: Dict[str, Dict[str, torch.Tensor]],
        subject_ids: Optional[torch.Tensor] = None,
        dapt_embedding_store: Optional[Dict] = None,
    ) -> Dict[str, torch.Tensor]:
        """Compute and aggregate all informative outcome losses in the batch."""
        terms, log_dict = self.compute_per_outcome_losses(
            embeddings,
            outcome_survival,
            subject_ids=subject_ids,
            dapt_embedding_store=dapt_embedding_store,
        )
        template_embedding = (
            embeddings
            if isinstance(embeddings, torch.Tensor)
            else next(iter(embeddings.values()))
        )
        device = template_embedding.device
        per_outcome_losses = torch.full(
            (self.n_outcomes,),
            float("nan"),
            device=device,
            dtype=template_embedding.dtype,
        )

        for index, name in enumerate(self.outcome_names):
            term = terms.get(name)
            if term is None:
                continue
            if float(term["n_effective_pairs"].detach().item()) < 1.0:
                continue
            factor = self.class_balance_factors[index].to(
                device=device,
                dtype=template_embedding.dtype,
            )
            per_outcome_losses[index] = term["aggregation_loss"] * factor
            log_dict[f"class_balance_factor/{name}"] = factor.detach()

        active_mask = torch.isfinite(per_outcome_losses)
        outcome_weights = self.weighter.weights(per_outcome_losses)
        finite_losses = torch.where(
            active_mask,
            per_outcome_losses,
            torch.zeros_like(per_outcome_losses),
        )

        if active_mask.any() and self.aggregation == "hierarchical_support":
            total_loss = template_embedding.sum() * 0.0
            active_families = sorted(
                {
                    self.outcome_family[name]
                    for index, name in enumerate(self.outcome_names)
                    if active_mask[index]
                }
            )
            family_denominator = sum(
                float(self.family_weights.get(family, 1.0))
                for family in active_families
            )
            effective_weights = torch.zeros_like(per_outcome_losses)
            for family in active_families:
                indices = [
                    index
                    for index, name in enumerate(self.outcome_names)
                    if active_mask[index] and self.outcome_family[name] == family
                ]
                support = per_outcome_losses.new_tensor(
                    [
                        (
                            float(
                                self.event_location_counts.get(
                                    self.outcome_names[index], 0.0
                                )
                            )
                            / (
                                float(
                                    self.event_location_counts.get(
                                        self.outcome_names[index], 0.0
                                    )
                                )
                                + self.support_tau_locations
                            )
                        )
                        ** 0.5
                        for index in indices
                    ]
                )
                if float(support.sum().item()) <= 0:
                    support = torch.ones_like(support)
                support = support / support.sum()
                family_weight = (
                    float(self.family_weights.get(family, 1.0)) / family_denominator
                )
                index_tensor = torch.tensor(indices, device=device)
                effective_weights[index_tensor] = family_weight * support
                total_loss = total_loss + family_weight * torch.sum(
                    support * per_outcome_losses[index_tensor]
                )
                family_key = family.lower().replace(" ", "_").replace("&", "and")
                log_dict[f"family_weight/{family_key}"] = per_outcome_losses.new_tensor(
                    family_weight
                )
                log_dict[f"family_loss/{family_key}"] = torch.sum(
                    support * per_outcome_losses[index_tensor]
                ).detach()
                for diagnostic_name in (
                    "loss",
                    "target_entropy",
                    "excess_loss",
                    "uniform_baseline",
                    "contrastive_headroom",
                ):
                    values = torch.stack(
                        [
                            terms[self.outcome_names[index]].get(
                                diagnostic_name,
                                terms[self.outcome_names[index]]["aggregation_loss"],
                            )
                            for index in indices
                        ]
                    )
                    log_dict[f"family_{diagnostic_name}/{family_key}"] = torch.sum(
                        support * values
                    ).detach()
                family_headroom = log_dict[f"family_contrastive_headroom/{family_key}"]
                log_dict[f"family_relative_excess/{family_key}"] = (
                    log_dict[f"family_excess_loss/{family_key}"]
                    / family_headroom.clamp_min(1e-6)
                ).detach()
                for local_index, outcome_index in enumerate(indices):
                    log_dict[f"support_weight/{self.outcome_names[outcome_index]}"] = (
                        support[local_index].detach()
                    )
            outcome_weights = effective_weights
        elif active_mask.any():
            total_loss = torch.sum(outcome_weights * finite_losses)
            total_loss = total_loss + self.weighter.regularizer(active_mask)
            if self.aggregation == "macro":
                total_loss = total_loss / active_mask.sum().to(total_loss.dtype)
        else:
            # No valid pairs in any outcome. Apply regularizer so learnable
            # sigma/precision parameters still receive a gradient, and add
            # an embeddings anchor so embeddings.grad is not None.
            total_loss = (
                self.weighter.regularizer(active_mask) + template_embedding.sum() * 0.0
            )

        for index, name in enumerate(self.outcome_names):
            log_dict[f"cross_outcome_weight/{name}"] = outcome_weights[index].detach()
            if isinstance(self.weighter, KendallWeighter):
                log_dict[f"sigma/{name}"] = torch.exp(
                    self.weighter.log_sigma[index]
                ).detach()
                log_dict[f"precision/{name}"] = outcome_weights[index].detach()

        for diagnostic_name in (
            "loss",
            "target_entropy",
            "excess_loss",
            "uniform_baseline",
            "contrastive_headroom",
        ):
            diagnostic_values = torch.zeros_like(per_outcome_losses)
            for index, name in enumerate(self.outcome_names):
                if active_mask[index]:
                    diagnostic_values[index] = terms[name].get(
                        diagnostic_name,
                        terms[name]["aggregation_loss"],
                    )
            log_dict[f"contrastive/{diagnostic_name}"] = torch.sum(
                outcome_weights * diagnostic_values
            ).detach()
        headroom = log_dict["contrastive/contrastive_headroom"]
        log_dict["contrastive/relative_excess"] = (
            log_dict["contrastive/excess_loss"] / headroom.clamp_min(1e-6)
        ).detach()

        log_dict["loss"] = total_loss
        return log_dict


class SupervisedContrastiveLoss(nn.Module):
    """Binary SupCon loss — kept for ablation experiments."""

    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        device = embeddings.device
        B = embeddings.size(0)
        if B <= 1:
            return torch.tensor(0.0, device=device, requires_grad=True)
        sim = torch.mm(embeddings, embeddings.t()) / self.temperature
        labels = labels.view(-1, 1)
        mask_pos = (labels == labels.t()).float()
        mask_self = torch.eye(B, device=device)
        mask_pos = mask_pos - mask_self
        n_pos = mask_pos.sum(dim=1)
        valid = n_pos > 0
        if valid.sum() == 0:
            return torch.tensor(0.0, device=device, requires_grad=True)
        logits_max, _ = sim.detach().max(dim=1, keepdim=True)
        logits = sim - logits_max
        exp_logits = torch.exp(logits) * (1 - mask_self)
        log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-12)
        mean_log_prob = (mask_pos * log_prob).sum(dim=1) / (n_pos + 1e-12)
        return -mean_log_prob[valid].mean()


class MultiOutcomeContrastiveLoss(nn.Module):
    """Binary multi-outcome SupCon — kept for ablation experiments."""

    def __init__(self, outcome_names: List[str], temperature: float = 0.07):
        super().__init__()
        self.outcome_names = outcome_names
        self.n_outcomes = len(outcome_names)
        self.sup_con = SupervisedContrastiveLoss(temperature=temperature)
        self.log_sigma = nn.Parameter(torch.zeros(self.n_outcomes))

    def forward(
        self,
        embeddings: torch.Tensor,
        outcome_labels: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        total_loss = torch.tensor(0.0, device=embeddings.device, requires_grad=True)
        log_dict: Dict[str, torch.Tensor] = {}
        for k, name in enumerate(self.outcome_names):
            labels_k = outcome_labels[name]
            valid_mask = labels_k >= 0
            if valid_mask.sum() < 2:
                continue
            loss_k = self.sup_con(embeddings[valid_mask], labels_k[valid_mask])
            precision = 0.5 * torch.exp(-2 * self.log_sigma[k])
            total_loss = total_loss + precision * loss_k + self.log_sigma[k]
            log_dict[f"loss/{name}"] = loss_k.detach()
            log_dict[f"sigma/{name}"] = torch.exp(self.log_sigma[k]).detach()
            log_dict[f"precision/{name}"] = precision.detach()
        log_dict["loss"] = total_loss
        return log_dict


# ═══════════════════════════════════════════════════════════════════════════
# Full OPERA contrastive model
# ═══════════════════════════════════════════════════════════════════════════


class OperaContrastiveModel(nn.Module):
    """
    Wraps a BonsaiEncoder → pooler → projection head and exposes the
    survival-informed contrastive loss computation.

    Key changes from the binary-SupCon version
    ─────────────────────────────────────────────
    1. Loss is ``MultiOutcomeSurvivalLoss``: uses time-to-event + event
       indicator rather than binarised labels, with censoring-aware pair
       weights.

    2. DAPT-prior cohort similarity: if ``dapt_embedding_store`` is
       provided, pair weights are further modulated by cosine similarity
       of frozen DAPT-stage representations.  This implements the
       meta-learning prior ("learn from similar histories") without making
       diagnosis categories a hard gate.

    3. Uncertainty estimation: ``get_embeddings`` supports
       ``enable_dropout=True`` so that MC-Dropout uncertainty can be
       computed at inference time by calling it multiple times.
    """

    def __init__(
        self,
        encoder: BonsaiEncoder,
        outcome_names: List[str],
        hidden_size: int = 768,
        projection_hidden_dim: int = 256,
        projection_dim: int = 128,
        temperature: float = 0.07,
        km_time_scale: float = 0.25,
        outcome_sorted_event_times: Optional[Dict[str, torch.Tensor]] = None,
        outcome_event_time_probs: Optional[Dict[str, torch.Tensor]] = None,
        dapt_lambda_floor: float = 0.55,  # TUNE: cross-disease floor
        dapt_anchor_weight: float = 0.2,
        competing_event_weight: float = 0.0,
        competing_event_handling: str = "hard_negative",
        effective_pair_normalization: bool = True,
        cross_outcome_config: Optional[Mapping[str, object]] = None,
        projection_mode: str = "shared",
        freeze_encoder: bool = False,
        pooling: str = "cls_last",
        dapt_embedding_store: Optional[Dict] = None,
        competing_risk_config: Optional[Mapping[str, object]] = None,
    ):
        super().__init__()
        self.encoder = encoder
        self.freeze_encoder = freeze_encoder
        self.dapt_embedding_store = dapt_embedding_store  # {subject_id: tensor}
        self.dapt_anchor_weight = dapt_anchor_weight
        competing_risk_settings = dict(competing_risk_config or {})
        self.competing_risk_weight = float(
            competing_risk_settings.get("loss_weight", 0.0)
        )
        if self.competing_risk_weight < 0:
            raise ValueError("competing_risk.loss_weight must be non-negative.")
        self.contrastive_loss_weight = float(
            competing_risk_settings.get("contrastive_loss_weight", 1.0)
        )
        if self.contrastive_loss_weight < 0:
            raise ValueError(
                "competing_risk.contrastive_loss_weight must be non-negative."
            )
        self.model_init_config = {
            "hidden_size": hidden_size,
            "projection_hidden_dim": projection_hidden_dim,
            "projection_dim": projection_dim,
            "temperature": temperature,
            "km_time_scale": km_time_scale,
            "dapt_lambda_floor": dapt_lambda_floor,
            "dapt_anchor_weight": dapt_anchor_weight,
            "competing_event_weight": competing_event_weight,
            "competing_event_handling": competing_event_handling,
            "effective_pair_normalization": effective_pair_normalization,
            "cross_outcome_config": dict(cross_outcome_config or {}),
            "projection_mode": projection_mode,
            "freeze_encoder": freeze_encoder,
            "pooling": pooling,
            "competing_risk_config": competing_risk_settings,
        }

        if freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad = False

        self.pooling = pooling
        if pooling == "bigru":
            self.pooler = BiGRU(hidden_size)
        elif pooling not in {"cls_last", "mean_last_128"}:
            raise ValueError(
                "pooling must be one of: 'cls_last', 'mean_last_128', 'bigru'."
            )

        self.projection_mode = str(projection_mode).lower()
        cross_outcome_settings = dict(cross_outcome_config or {})
        outcome_families = cross_outcome_settings.get("outcome_families", {})
        if self.projection_mode == "shared":
            self.projection = ProjectionHead(
                input_dim=hidden_size,
                hidden_dim=projection_hidden_dim,
                output_dim=projection_dim,
            )
        elif self.projection_mode == "family":
            family_by_outcome = {
                outcome: family
                for family, members in outcome_families.items()
                for outcome in members
            }
            missing = sorted(set(outcome_names) - set(family_by_outcome))
            if missing:
                raise ValueError(
                    "Family projection requires outcome_families to cover every "
                    f"configured outcome; missing={missing}."
                )
            self.projection = FamilyProjectionHead(
                input_dim=hidden_size,
                hidden_dim=projection_hidden_dim,
                output_dim=projection_dim,
                families=outcome_families,
            )
        else:
            raise ValueError("projection_mode must be 'shared' or 'family'.")

        self.contrastive_loss = MultiOutcomeSurvivalLoss(
            outcome_names=outcome_names,
            temperature=temperature,
            km_time_scale=km_time_scale,
            outcome_sorted_event_times=outcome_sorted_event_times,
            outcome_event_time_probs=outcome_event_time_probs,
            dapt_lambda_floor=dapt_lambda_floor,
            competing_event_weight=competing_event_weight,
            competing_event_handling=competing_event_handling,
            effective_pair_normalization=effective_pair_normalization,
            cross_outcome_config=cross_outcome_config,
        )
        self.competing_risk_loss = None
        self.competing_risk_head = None
        if self.competing_risk_weight > 0:
            boundaries = competing_risk_settings.get("interval_boundaries_days")
            if not isinstance(boundaries, (list, tuple)):
                raise ValueError(
                    "competing_risk.interval_boundaries_days is required when "
                    "competing_risk.loss_weight is positive."
                )
            self.competing_risk_loss = PiecewiseExponentialCompetingRiskLoss(
                outcome_names,
                boundaries,
                no_competing_outcomes=[
                    name
                    for name in competing_risk_settings.get(
                        "no_competing_outcomes",
                        ["overall_survival"],
                    )
                    if name in outcome_names
                ],
                time_scale_days=float(
                    competing_risk_settings.get("time_scale_days", 365.25)
                ),
                smoothness_weight=float(
                    competing_risk_settings.get("smoothness_weight", 0.0)
                ),
                cross_outcome_config=cross_outcome_config,
            )
            n_outputs = len(outcome_names) * 2 * self.competing_risk_loss.n_intervals
            self.competing_risk_head = nn.Linear(hidden_size, n_outputs)
            nn.init.normal_(self.competing_risk_head.weight, mean=0.0, std=0.01)
            nn.init.constant_(
                self.competing_risk_head.bias,
                float(competing_risk_settings.get("initial_log_hazard", -2.3)),
            )

    def _pool(
        self,
        batch: dict,
        enable_dropout: bool = False,
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """Return the pooled encoder representation before projection."""
        if enable_dropout:
            # Temporarily set encoder to train mode to activate dropout
            prev_training = self.encoder.training
            self.encoder.train()

        with torch.set_grad_enabled(not self.freeze_encoder):
            outputs = self.encoder(batch)

        if enable_dropout and not prev_training:
            self.encoder.eval()

        hidden = encoder_hidden_state(outputs)  # (B, L, H)

        if self.pooling == "bigru":
            return self.pooler(hidden, batch["attention_mask"], return_embedding=True)

        lengths = batch["attention_mask"].sum(dim=1) - 1
        if self.pooling == "mean_last_128":
            positions = torch.arange(hidden.size(1), device=hidden.device).unsqueeze(0)
            starts = (lengths - 128).clamp_min(0).unsqueeze(1)
            content_ends = lengths.unsqueeze(1)
            recent_content = (positions >= starts) & (positions < content_ends)
            weights = recent_content.unsqueeze(-1).to(hidden.dtype)
            return (hidden * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)
        return hidden[torch.arange(hidden.size(0), device=hidden.device), lengths]

    def get_embeddings(
        self,
        batch: dict,
        return_pre_projection: bool = False,
        enable_dropout: bool = False,
    ) -> torch.Tensor:
        """
        Run encoder → pooling → projection.

        Args:
            batch                : standard BONSAI batch dict.
            return_pre_projection: return pooled hidden state before projection.
            enable_dropout       : if True, keep dropout active regardless of
                                   ``self.training``.  Used for MC-Dropout
                                   uncertainty estimation — call this method
                                   N times with ``enable_dropout=True`` and
                                   compute variance across runs.
        Returns:
            (B, dim) tensor, l2-normalised (or un-projected if requested).
        """
        pooled = self._pool(batch, enable_dropout=enable_dropout)
        if return_pre_projection:
            return pooled

        return self.projection(pooled)

    def get_contrastive_embeddings(
        self,
        batch: dict,
        *,
        family: Optional[str] = None,
        enable_dropout: bool = False,
    ) -> torch.Tensor:
        """Return a specific contrastive space while keeping pooled embeddings canonical."""
        projected = self.get_embeddings(batch, enable_dropout=enable_dropout)
        if isinstance(projected, torch.Tensor):
            return projected
        if family is None:
            raise ValueError("family is required when projection_mode='family'.")
        if family not in projected:
            raise KeyError(f"Unknown projection family {family!r}.")
        return projected[family]

    def _compute_anchor_loss(
        self,
        pooled: torch.Tensor,
        subject_ids: Optional[torch.Tensor],
    ) -> Optional[torch.Tensor]:
        """
        Anchor current pooled encoder states to frozen DAPT representations.

        The anchor acts before the projection head. This protects the encoder's
        EHR representation while allowing the contrastive projection to move.
        """
        if (
            self.dapt_embedding_store is None
            or self.dapt_anchor_weight == 0.0
            or subject_ids is None
        ):
            return None

        if hasattr(subject_ids, "detach"):
            ids = subject_ids.detach().cpu().tolist()
        else:
            ids = list(subject_ids)

        current = []
        target = []
        for idx, sid in enumerate(ids):
            if sid not in self.dapt_embedding_store:
                continue
            current.append(pooled[idx])
            target.append(
                torch.as_tensor(
                    self.dapt_embedding_store[sid],
                    device=pooled.device,
                    dtype=pooled.dtype,
                )
            )

        if not current:
            return None

        current_tensor = F.normalize(torch.stack(current), dim=-1)
        target_tensor = F.normalize(torch.stack(target), dim=-1)
        if current_tensor.shape != target_tensor.shape:
            raise ValueError(
                "DAPT anchor embeddings must match pooled encoder shape; "
                f"got current {tuple(current_tensor.shape)} and "
                f"stored {tuple(target_tensor.shape)}."
            )

        cosine = (current_tensor * target_tensor).sum(dim=-1)
        return (1.0 - cosine).mean()

    def forward(
        self,
        batch: dict,
        outcome_survival: Dict[str, Dict[str, torch.Tensor]],
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            batch            : standard BONSAI batch dict (must include
                               ``subject_id`` for DAPT lookup).
            outcome_survival : dict mapping outcome name →
                               {"times": (B,) float, "events": (B,) int}.
        """
        pooled = self._pool(batch)
        return self.forward_from_pooled(
            pooled, outcome_survival, batch.get("subject_id", None)
        )

    def forward_from_pooled(
        self,
        pooled: torch.Tensor,
        outcome_survival: Dict[str, Dict[str, torch.Tensor]],
        subject_ids: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Compute all OPERA objectives from cached pooled representations."""
        embeddings = self.projection(pooled)
        log_dict = self.contrastive_loss(
            embeddings,
            outcome_survival,
            subject_ids=subject_ids,
            dapt_embedding_store=self.dapt_embedding_store,
        )
        contrastive_loss = log_dict["loss"]
        log_dict["contrastive_loss"] = contrastive_loss.detach()
        total_loss = self.contrastive_loss_weight * contrastive_loss

        if self.competing_risk_head is not None:
            n_intervals = self.competing_risk_loss.n_intervals
            log_hazards = self.competing_risk_head(pooled).reshape(
                pooled.shape[0],
                len(self.contrastive_loss.outcome_names),
                2,
                n_intervals,
            )
            competing_risk_loss, competing_logs = self.competing_risk_loss(
                log_hazards,
                outcome_survival,
            )
            total_loss = total_loss + self.competing_risk_weight * competing_risk_loss
            log_dict.update(competing_logs)

        anchor_loss = self._compute_anchor_loss(pooled, subject_ids)
        if anchor_loss is not None:
            total_loss = total_loss + self.dapt_anchor_weight * anchor_loss
            log_dict["anchor_loss"] = anchor_loss.detach()
        log_dict["loss"] = total_loss
        return log_dict
