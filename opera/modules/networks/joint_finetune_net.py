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

from typing import Dict, List, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from opera.compat.bonsai import BonsaiEncoder, BiGRU


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
    ):
        super().__init__()
        self.encoder       = encoder
        self.outcome_names = outcome_names
        self.pooling       = pooling
        self.freeze_encoder = freeze_encoder

        if freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad = False

        if pooling == "bigru":
            self.pooler = BiGRU(hidden_size)
        # cls_last needs no extra params

        self.dropout = nn.Dropout(dropout)

        # One lightweight head per outcome
        self.heads = nn.ModuleDict({
            name: nn.Linear(hidden_size, 1)
            for name in outcome_names
        })

        # Learnable log-variance per outcome (Kendall et al. 2018)
        self.log_sigma = nn.Parameter(torch.zeros(len(outcome_names)))

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
            pooled  = hidden[torch.arange(hidden.size(0)), lengths]

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

        total_loss = torch.tensor(0.0, device=device, requires_grad=True)
        log_dict: Dict[str, torch.Tensor] = {}
        bce = nn.BCEWithLogitsLoss(reduction="mean")

        for k, name in enumerate(self.outcome_names):
            labels_k = outcome_labels.get(name)
            if labels_k is None:
                continue

            valid = labels_k >= 0
            if valid.sum() < 2:
                continue

            logits_k = self.heads[name](pooled[valid]).squeeze(-1)  # (V,)
            loss_k   = bce(logits_k, labels_k[valid].float())

            # Kendall weighting
            precision = 0.5 * torch.exp(-2.0 * self.log_sigma[k])
            total_loss = total_loss + precision * loss_k + self.log_sigma[k]

            log_dict[f"loss/{name}"]   = loss_k.detach()
            log_dict[f"sigma/{name}"]  = torch.exp(self.log_sigma[k]).detach()
            log_dict[f"logits/{name}"] = logits_k.detach()

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
