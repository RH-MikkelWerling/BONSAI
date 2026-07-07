"""OPERA-specific sequence pooling layers.

BONSAI's native finetuning head now reads the prediction token directly.
OPERA retains BiGRU pooling as an explicit experimental ablation.
"""

from __future__ import annotations

import torch


class BiGRU(torch.nn.Module):
    """Pool the final valid sequence states with a bidirectional GRU."""

    def __init__(self, hidden_size: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.rnn_hidden_size = hidden_size // 2
        self.rnn = torch.nn.GRU(
            hidden_size,
            self.rnn_hidden_size,
            batch_first=True,
            bidirectional=True,
        )
        self.norm = torch.nn.LayerNorm(hidden_size)
        self.classifier = torch.nn.Linear(hidden_size, 1)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        return_embedding: bool = False,
    ) -> torch.Tensor:
        lengths = attention_mask.sum(dim=1).to(dtype=torch.long, device="cpu")
        packed = torch.nn.utils.rnn.pack_padded_sequence(
            hidden_states,
            lengths,
            batch_first=True,
            enforce_sorted=False,
        )
        output, _ = self.rnn(packed)
        output, _ = torch.nn.utils.rnn.pad_packed_sequence(output, batch_first=True)
        last_sequence_idx = lengths - 1
        batch_indices = torch.arange(output.shape[0], device=output.device)
        forward_output = output[
            batch_indices,
            last_sequence_idx.to(output.device),
            : self.rnn_hidden_size,
        ]
        backward_output = output[:, 0, self.rnn_hidden_size :]
        pooled = self.norm(torch.cat((forward_output, backward_output), dim=-1))
        return pooled if return_embedding else self.classifier(pooled)


__all__ = ["BiGRU"]
