"""Linear-probe readout for frozen OPERA/BONSAI encoders."""

from __future__ import annotations

import torch

from opera.compat.bonsai import BonsaiEncoder


class BonsaiLinearProbe(BonsaiEncoder):
    """Frozen-encoder classifier with masked mean pooling and a linear head.

    Inputs are ordinary BONSAI/OPERA batches. The output is one logit per
    patient. The scientific purpose is a strict representation-quality probe:
    the encoder is held fixed and only a linear classifier over pooled token
    embeddings is trained.
    """

    def __init__(self, config):
        super().__init__(config)
        self.classifier = torch.nn.Linear(config.hidden_size, 1)

    def pooled_embedding(self, batch: dict) -> torch.Tensor:
        """Return masked mean pooled encoder embeddings for one batch."""
        outputs = super().forward(batch)
        hidden = outputs[0]
        mask = batch["attention_mask"].unsqueeze(-1).to(hidden.dtype)
        denom = mask.sum(dim=1).clamp_min(1.0)
        return (hidden * mask).sum(dim=1) / denom

    def forward(self, batch: dict, **kwargs) -> torch.Tensor:
        pooled = self.pooled_embedding(batch)
        return self.classifier(pooled)
