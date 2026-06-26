"""
Joint multi-task finetuning network for OPERA.

Architecture
────────────
BonsaiEncoder  (shared, pretrained)
      ↓
 768-d pooled representation  (BiGRU or CLS-last)
      ↓
  Per-outcome classification heads  (one Linear(768→1) each)

The key claim this model tests
────────────────────────────────
A single model trained jointly on *all* patients across *all* disease
cohorts and *all* outcomes simultaneously should outperform:
  (a) separate per-outcome tabular models, because the encoder learns
      shared structure across diseases that tabular models cannot capture
      without massive feature engineering.
  (b) separately finetuned per-cohort foundation models, because joint
      training exposes the encoder to the full disease-outcome structure
      rather than a single cohort slice.

This is where the foundation model's inductive bias — a shared token
space across all EHR events regardless of disease — pays off most
directly.

Loss
────
Multi-task BCE with Kendall uncertainty weighting (same sigma trick as
the contrastive stage):

    L_total = Σ_k  [ (1 / 2σ_k²) * BCE_k  +  log(σ_k) ]

The σ_k values are interpretable: low σ → the model treats this outcome
as high-confidence; high σ → noisy / hard to learn jointly.  This is a
directly comparable quantity to the contrastive sigma values, letting you
ask: "Is the outcome structure the contrastive stage learned consistent
with how hard the joint model finds each outcome?"
"""

from collections.abc import Mapping
from typing import Dict, List, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from opera.compat.bonsai import BonsaiEncoder, BiGRU
from opera.modules.networks.cross_outcome_weighters import (
    KendallWeighter,
    build_cross_outcome_weighter,
)

# NOTE: The historical docstring above describes the original Kendall-only
# joint BCE loss. The implementation now shares the contrastive stage's
# pluggable cross-outcome weighter; production configs use uniform macro
# aggregation plus capped positive-class weighting for rare endpoints.


class JointFinetuneModel(nn.Module):
    """
    Parameters
    ----------
    encoder       : pretrained BonsaiEncoder (weights loaded externally).
    outcome_names : list of outcome names — one head is created per outcome.
    hidden_size   : encoder hidden dimension (default 768).
    pooling       : "bigru" | "cls_last".
    freeze_encoder: if True, only train the heads (useful for ablation).
    dropout       : dropout applied to pooled representation before heads.
                    Also enables MC-Dropout uncertainty at inference.
    """

    def __init__(
        self,
        encoder: BonsaiEncoder,
        outcome_names: List[str],
        hidden_size: int = 768,
        pooling: str = "bigru",
        freeze_encoder: bool = False,
        dropout: float = 0.1,
        cross_outcome_config: Optional[Mapping[str, object]] = None,
    ):
        super().__init__()
        self.encoder = encoder
        self.outcome_names = outcome_names
        self.pooling = pooling
        self.freeze_encoder = freeze_encoder
        self.cross_outcome_config = dict(cross_outcome_config or {})

        if freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad = False

        if pooling == "bigru":
            self.pooler = BiGRU(hidden_size)
        # cls_last needs no extra params

        self.dropout = nn.Dropout(dropout)

        # One lightweight head per outcome
        self.heads = nn.ModuleDict(
            {name: nn.Linear(hidden_size, 1) for name in outcome_names}
        )

        # Share the contrastive stage's cross-outcome weighting semantics.
        settings = self._legacy_default_cross_outcome_config()
        settings.update(self.cross_outcome_config)
        (
            self.weighter,
            self.aggregation,
            class_balance_factors,
        ) = build_cross_outcome_weighter(outcome_names, settings)
        self.register_buffer(
            "class_balance_factors",
            class_balance_factors,
            persistent=False,
        )
        self.require_both_classes_per_batch = bool(
            settings.get("require_both_classes_per_batch", True)
        )
        self.positive_class_weighted = bool(
            settings.get("positive_class_weighted", False)
        )
        self.positive_class_weight_cap = float(
            settings.get(
                "positive_class_weight_cap",
                settings.get("class_balanced_cap", 50.0),
            )
        )
        positive_weights = self._positive_class_weights(settings)
        self.register_buffer(
            "positive_class_weights",
            positive_weights,
            persistent=False,
        )

    @staticmethod
    def _legacy_default_cross_outcome_config() -> dict:
        """Preserve historical construction unless configs opt into uniform macro."""
        return {
            "weighter": "kendall",
            "aggregation": "pooled",
            "class_balanced": False,
        }

    def _positive_class_weights(self, settings: Mapping[str, object]) -> torch.Tensor:
        """Return capped per-outcome BCE positive weights from train counts."""
        weights = torch.ones(len(self.outcome_names), dtype=torch.float32)
        if not self.positive_class_weighted:
            return weights
        counts = settings.get("class_counts", {})
        if not isinstance(counts, Mapping):
            raise ValueError(
                "cross_outcome.class_counts must be a mapping when "
                "positive_class_weighted is true."
            )
        for index, name in enumerate(self.outcome_names):
            outcome_counts = counts.get(name, {})
            if not isinstance(outcome_counts, Mapping):
                continue
            positive = float(outcome_counts.get("positive", 0.0) or 0.0)
            negative = float(outcome_counts.get("negative", 0.0) or 0.0)
            if positive <= 0 or negative <= 0:
                continue
            weights[index] = min(negative / positive, self.positive_class_weight_cap)
        return weights

    @property
    def log_sigma(self) -> torch.Tensor:
        """Expose Kendall log-sigma for older analysis utilities."""
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
        if old_key in state_dict:
            if isinstance(self.weighter, KendallWeighter) and new_key not in state_dict:
                state_dict[new_key] = state_dict[old_key]
            state_dict.pop(old_key)
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    def get_embedding(
        self,
        batch: dict,
        enable_dropout: bool = False,
    ) -> torch.Tensor:
        """
        Returns (B, hidden_size) pooled representation.
        Pass enable_dropout=True for MC-Dropout uncertainty estimation.
        """
        if enable_dropout:
            prev = self.encoder.training
            self.encoder.train()

        with torch.set_grad_enabled(not self.freeze_encoder):
            outputs = self.encoder(batch)
        hidden = outputs[0]  # (B, L, H)

        if enable_dropout and not prev:
            self.encoder.eval()

        if self.pooling == "bigru":
            pooled = self.pooler(hidden, batch["attention_mask"], return_embedding=True)
        else:
            lengths = batch["attention_mask"].sum(dim=1) - 1
            pooled = hidden[
                torch.arange(hidden.size(0), device=hidden.device),
                lengths,
            ]

        return self.dropout(pooled)

    def forward(
        self,
        batch: dict,
        outcome_labels: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """
        Args
        ----
        batch          : standard BONSAI batch dict.
        outcome_labels : {outcome_name: (B,) int tensor, -1 = missing}.

        Returns
        -------
        dict with:
            "loss"            : total weighted BCE
            "loss/<outcome>"  : per-outcome BCE
            "sigma/<outcome>" : learned sigma per outcome
            "logits/<outcome>": raw logits (B,) — for evaluation
        """
        pooled = self.get_embedding(batch)
        device = pooled.device

        log_dict: Dict[str, torch.Tensor] = {}
        per_outcome_losses = torch.full(
            (len(self.outcome_names),),
            float("nan"),
            device=device,
            dtype=pooled.dtype,
        )

        for k, name in enumerate(self.outcome_names):
            labels_k = outcome_labels.get(name)
            if labels_k is None:
                continue

            valid = labels_k >= 0
            log_dict[f"n_valid/{name}"] = valid.sum().float().detach()
            if int(valid.sum().item()) < 2:
                continue

            labels_v = labels_k[valid].float()
            if self.require_both_classes_per_batch and torch.unique(labels_v).numel() < 2:
                continue

            logits_k = self.heads[name](pooled[valid]).squeeze(-1)  # (V,)
            pos_weight = self.positive_class_weights[k].to(
                device=device,
                dtype=pooled.dtype,
            )
            loss_k = F.binary_cross_entropy_with_logits(
                logits_k,
                labels_v,
                reduction="mean",
                pos_weight=pos_weight if self.positive_class_weighted else None,
            )

            factor = self.class_balance_factors[k].to(
                device=device,
                dtype=pooled.dtype,
            )
            per_outcome_losses[k] = loss_k * factor

            log_dict[f"loss/{name}"] = loss_k.detach()
            log_dict[f"class_balance_factor/{name}"] = factor.detach()
            log_dict[f"positive_class_weight/{name}"] = pos_weight.detach()
            log_dict[f"logits/{name}"] = logits_k.detach()

        active_mask = torch.isfinite(per_outcome_losses)
        outcome_weights = self.weighter.weights(per_outcome_losses)
        finite_losses = torch.where(
            active_mask,
            per_outcome_losses,
            torch.zeros_like(per_outcome_losses),
        )
        if active_mask.any():
            total_loss = torch.sum(outcome_weights * finite_losses)
            total_loss = total_loss + self.weighter.regularizer(active_mask)
            if self.aggregation == "macro":
                total_loss = total_loss / active_mask.sum().to(total_loss.dtype)
        else:
            total_loss = pooled.sum() * 0.0

        for index, name in enumerate(self.outcome_names):
            log_dict[f"cross_outcome_weight/{name}"] = (
                outcome_weights[index].detach()
            )
            if isinstance(self.weighter, KendallWeighter):
                log_dict[f"sigma/{name}"] = torch.exp(
                    self.weighter.log_sigma[index]
                ).detach()
                log_dict[f"precision/{name}"] = outcome_weights[index].detach()

        log_dict["loss"] = total_loss
        return log_dict

    def predict(
        self,
        batch: dict,
        outcome_name: str,
    ) -> torch.Tensor:
        """
        Inference-only: return (B,) logits for a single outcome.
        Missing patients (label=-1) are still scored.
        """
        pooled = self.get_embedding(batch)
        return self.heads[outcome_name](pooled).squeeze(-1)
