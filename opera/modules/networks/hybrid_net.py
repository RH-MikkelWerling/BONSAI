"""
OPERA Hybrid model — Embedding + Tabular feature concatenation.

This is the "bonus" experiment that concatenates the BONSAI encoder's
pooled representation with external tabular features (e.g. RKKP quality
registry variables) and passes the combined vector through a small MLP.

Architecture
============
    BonsaiEncoder → pool → (H,)
                                  ⊕ → MLP → logit
    Tabular features →     (T,)

The encoder can be frozen (default) so only the MLP trains, or jointly
fine-tuned.  The tabular features are expected as a ``tabular`` key in
the batch dict (a float tensor of shape (B, T)).

This file is OPTIONAL — it exists for the hybrid experiment tier.
"""

from typing import Optional
import torch
import torch.nn as nn
from opera.compat.bonsai import BonsaiEncoder, BiGRU


class HybridClassifier(nn.Module):
    """
    Parameters
    ----------
    encoder : BonsaiEncoder
        Pretrained / DAPT / contrastive encoder.
    hidden_size : int
        Encoder hidden dimension (must match checkpoint).
    tabular_dim : int
        Number of external tabular features (e.g. RKKP variables).
        TUNE: set this to match your tabular feature count.
    mlp_hidden_dims : list of int
        Intermediate MLP widths after concatenation.
        TUNE: default [256, 64] is a reasonable starting point.
    dropout : float
        Dropout in MLP layers.
    freeze_encoder : bool
        If True, encoder weights are frozen (recommended for this experiment).
    pooling : str
        "cls_last" or "bigru".
    """

    def __init__(
        self,
        encoder: BonsaiEncoder,
        hidden_size: int = 768,
        tabular_dim: int = 87,  # TUNE: number of RKKP features
        mlp_hidden_dims: Optional[list] = None,
        dropout: float = 0.1,
        freeze_encoder: bool = True,
        pooling: str = "cls_last",
    ):
        super().__init__()
        self.encoder = encoder
        self.freeze_encoder = freeze_encoder

        if freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad = False

        # Pooling
        self.pooling = pooling
        if pooling == "bigru":
            self.pooler = BiGRU(hidden_size)

        # Tabular feature normalization
        self.tabular_norm = nn.BatchNorm1d(tabular_dim)

        # MLP on concatenated features
        if mlp_hidden_dims is None:
            mlp_hidden_dims = [256, 64]  # TUNE: MLP architecture

        concat_dim = hidden_size + tabular_dim
        layers = []
        in_dim = concat_dim
        for h_dim in mlp_hidden_dims:
            layers.extend(
                [
                    nn.Linear(in_dim, h_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.LayerNorm(h_dim),
                ]
            )
            in_dim = h_dim
        layers.append(nn.Linear(in_dim, 1))
        self.mlp = nn.Sequential(*layers)

    def forward(self, batch: dict) -> torch.Tensor:
        """
        Args:
            batch: standard BONSAI batch dict, plus:
                   batch["tabular"]: (B, tabular_dim) float tensor
        Returns:
            (B, 1) logits.
        """
        # Encoder
        with torch.set_grad_enabled(not self.freeze_encoder):
            outputs = self.encoder(batch)
        hidden = outputs[0]

        # Pool
        if self.pooling == "bigru":
            pooled = self.pooler(hidden, batch["attention_mask"], return_embedding=True)
        else:
            lengths = batch["attention_mask"].sum(dim=1) - 1
            pooled = hidden[torch.arange(hidden.size(0)), lengths]

        # Tabular
        tabular = self.tabular_norm(batch["tabular"].float())

        # Concatenate and classify
        combined = torch.cat([pooled, tabular], dim=-1)
        logits = self.mlp(combined)
        return logits
