"""
Multi-Outcome Learning (MOL) Network.

This is the direct-prediction counterpart to OPERA's contrastive approach.
Instead of shaping the embedding geometry, MOL jointly optimises BCE losses
across multiple outcomes through task-specific classification heads.

Architecture
============
BonsaiEncoder (frozen or trainable)
        ↓
  Pooling (CLS-last or BiGRU)  →  (H,) shared representation
        ↓
  Per-outcome classification heads  →  one logit per outcome

Loss weighting strategy
=======================
The default is EQUAL weighting — each outcome's BCE contributes equally.
This keeps MOL clean as a comparison arm: any performance difference vs
OPERA is attributable to contrastive-vs-BCE, not to the weighting trick.

Three weighting modes are available:

  "equal"   — L = Σ_k BCE_k / K                        (default)
  "fixed"   — L = Σ_k w_k · BCE_k     (user-specified per-outcome weights)
  "kendall" — L = Σ_k [(1/2σ_k²)·BCE_k + log(σ_k)]    (learned, opt-in)

The Kendall mode exists for a specific ablation: comparing OPERA-contrastive
with Kendall weighting vs MOL with Kendall weighting isolates the SupCon-vs-
BCE question while controlling for the weighting mechanism.  But it should
NOT be the default, because:

  1. The σ values don't have the same scientific meaning as in OPERA.
     In OPERA they reflect embedding geometry; here they conflate task
     difficulty, label noise, and class imbalance.
  2. It makes MOL less interpretable as a baseline — adding the same
     trick to both methods muddies the comparison.
  3. Learned σ can be unstable with small per-outcome batch counts,
     since BCE on a handful of samples is noisier than SupCon over
     all pairwise similarities in a batch.

Key design decisions
====================
- Each outcome head is a small 2-layer MLP (not a single linear layer),
  giving each task some private capacity while sharing the encoder.
- Missing labels (coded as -1) are masked out of each outcome's BCE loss,
  so subjects contribute to whichever outcomes they have labels for.
- The shared representation is the same pooled vector used in OPERA,
  making the two approaches directly comparable.
"""

from typing import Dict, List, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from opera.compat.bonsai import BonsaiEncoder, BiGRU


# ═══════════════════════════════════════════════════════════════════════════
# Per-outcome classification head
# ═══════════════════════════════════════════════════════════════════════════

class OutcomeHead(nn.Module):
    """
    Small MLP head for a single binary outcome.
    Gives each task some private capacity beyond the shared encoder.
    """

    def __init__(
        self,
        input_dim: int = 768,
        hidden_dim: int = 128,        # TUNE: per-head hidden width
        dropout: float = 0.1,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns (B, 1) logits."""
        return self.net(x)


# ═══════════════════════════════════════════════════════════════════════════
# Multi-outcome loss with configurable weighting
# ═══════════════════════════════════════════════════════════════════════════

class MultiOutcomeBCELoss(nn.Module):
    """
    Aggregates per-outcome BCE losses with one of three weighting strategies.

    Parameters
    ----------
    outcome_names : list of str
    weighting : str
        "equal"   — unweighted mean of per-outcome losses (default).
        "fixed"   — user-specified weights via ``fixed_weights``.
        "kendall" — learned log-variance (Kendall et al. 2018), opt-in for
                    ablation only.
    fixed_weights : dict, optional
        Mapping outcome_name → float weight.  Required when weighting="fixed".
        Does not need to sum to 1 (will be used as-is).
    """

    def __init__(
        self,
        outcome_names: List[str],
        weighting: str = "equal",
        fixed_weights: Optional[Dict[str, float]] = None,
    ):
        super().__init__()
        self.outcome_names = outcome_names
        self.n_outcomes = len(outcome_names)
        self.weighting = weighting

        if weighting == "kendall":
            # Learned log-variance, same parameterisation as OPERA contrastive
            self.log_sigma = nn.Parameter(torch.zeros(self.n_outcomes))
        elif weighting == "fixed":
            if fixed_weights is None:
                raise ValueError("fixed_weights required when weighting='fixed'")
            w = torch.tensor([fixed_weights[n] for n in outcome_names], dtype=torch.float32)
            self.register_buffer("fixed_w", w)
        elif weighting == "equal":
            pass
        else:
            raise ValueError(
                f"Unknown weighting '{weighting}'. Choose 'equal', 'fixed', or 'kendall'."
            )

    def forward(
        self,
        logits_per_outcome: Dict[str, torch.Tensor],
        labels_per_outcome: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        device = next(iter(logits_per_outcome.values())).device
        total_loss = torch.tensor(0.0, device=device, requires_grad=True)
        log_dict: Dict[str, torch.Tensor] = {}
        n_active = 0

        for k, name in enumerate(self.outcome_names):
            labels_k = labels_per_outcome[name]
            logits_k = logits_per_outcome[name].squeeze(-1)

            valid = labels_k >= 0
            if valid.sum() == 0:
                continue

            bce_k = F.binary_cross_entropy_with_logits(
                logits_k[valid], labels_k[valid].float(), reduction="mean"
            )
            log_dict[f"loss/{name}"] = bce_k.detach()

            # ── Apply weighting strategy ─────────────────────────
            if self.weighting == "equal":
                total_loss = total_loss + bce_k
                n_active += 1

            elif self.weighting == "fixed":
                total_loss = total_loss + self.fixed_w[k] * bce_k
                log_dict[f"weight/{name}"] = self.fixed_w[k]

            elif self.weighting == "kendall":
                precision = 0.5 * torch.exp(-2 * self.log_sigma[k])
                weighted = precision * bce_k + self.log_sigma[k]
                total_loss = total_loss + weighted

                sigma_k = torch.exp(self.log_sigma[k])
                log_dict[f"sigma/{name}"] = sigma_k.detach()
                log_dict[f"precision/{name}"] = precision.detach()

        # For equal weighting, normalise by number of active outcomes
        if self.weighting == "equal" and n_active > 0:
            total_loss = total_loss / n_active

        log_dict["loss"] = total_loss
        return log_dict


# ═══════════════════════════════════════════════════════════════════════════
# Full MOL model
# ═══════════════════════════════════════════════════════════════════════════

class MultiOutcomeModel(nn.Module):
    """
    Shared encoder + per-outcome classification heads.

    Parameters
    ----------
    encoder : BonsaiEncoder
    outcome_names : list of str
    hidden_size : int
        Must match encoder hidden dimension.
    head_hidden_dim : int
        Per-outcome head MLP width. TUNE.
    head_dropout : float
    freeze_encoder : bool
        If True, only the heads train.
    pooling : str
        "cls_last" or "bigru".
    weighting : str
        Loss weighting strategy: "equal" (default), "fixed", or "kendall".
    fixed_weights : dict, optional
        Per-outcome weights when weighting="fixed".
    """

    def __init__(
        self,
        encoder: BonsaiEncoder,
        outcome_names: List[str],
        hidden_size: int = 768,
        head_hidden_dim: int = 128,
        head_dropout: float = 0.1,
        freeze_encoder: bool = False,
        pooling: str = "cls_last",
        weighting: str = "equal",
        fixed_weights: Optional[Dict[str, float]] = None,
    ):
        super().__init__()
        self.encoder = encoder
        self.freeze_encoder = freeze_encoder
        self.outcome_names = outcome_names

        if freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad = False

        # Pooling
        self.pooling = pooling
        if pooling == "bigru":
            self.pooler = BiGRU(hidden_size)

        # Per-outcome heads
        self.heads = nn.ModuleDict({
            name: OutcomeHead(
                input_dim=hidden_size,
                hidden_dim=head_hidden_dim,
                dropout=head_dropout,
            )
            for name in outcome_names
        })

        # Loss
        self.loss_fn = MultiOutcomeBCELoss(
            outcome_names, weighting=weighting, fixed_weights=fixed_weights,
        )

    def get_shared_representation(self, batch: dict) -> torch.Tensor:
        """
        Encode + pool → (B, H) shared representation.
        This is the representation that gets compared to OPERA embeddings.
        """
        with torch.set_grad_enabled(not self.freeze_encoder):
            outputs = self.encoder(batch)

        hidden = outputs[0]  # (B, L, H)

        if self.pooling == "bigru":
            pooled = self.pooler(
                hidden, batch["attention_mask"], return_embedding=True
            )
        else:
            lengths = batch["attention_mask"].sum(dim=1) - 1
            pooled = hidden[torch.arange(hidden.size(0)), lengths]

        return pooled

    def forward(
        self,
        batch: dict,
        outcome_labels: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """
        Full forward: encode → pool → per-head logits → weighted loss.
        """
        pooled = self.get_shared_representation(batch)

        logits_per_outcome = {}
        for name in self.outcome_names:
            logits_per_outcome[name] = self.heads[name](pooled)

        return self.loss_fn(logits_per_outcome, outcome_labels)

    def predict(
        self, batch: dict
    ) -> Dict[str, torch.Tensor]:
        """
        Inference: returns per-outcome probabilities (no loss computation).
        """
        pooled = self.get_shared_representation(batch)
        probs = {}
        for name in self.outcome_names:
            logits = self.heads[name](pooled).squeeze(-1)
            probs[name] = torch.sigmoid(logits)
        return probs
