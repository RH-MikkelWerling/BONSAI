from typing import Optional

import torch
import torch.nn as nn


class EhrEmbeddings(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        hidden_size: int,
        max_seqlen: int,
        value_bin_vocab_size: int = 0,
    ):
        super().__init__()

        # Initialize embeddings
        self.code_embedding = nn.Embedding(vocab_size, hidden_size, padding_idx=0)
        self.segment_embedding = nn.Embedding(max_seqlen, hidden_size, padding_idx=0)
        self.age_embedding = Time2Vec(hidden_size, clip_range=100)
        self.abspos_embedding = Time2Vec(hidden_size, clip_range=100)
        self.value_bin_vocab_size = int(value_bin_vocab_size)
        if self.value_bin_vocab_size > 0:
            self.value_bin_embedding = nn.Embedding(
                self.value_bin_vocab_size,
                hidden_size,
                padding_idx=0,
            )
            self.value_projection = nn.Linear(1, hidden_size)
        else:
            self.value_bin_embedding = None
            self.value_projection = None

    def forward(
        self,
        code: torch.LongTensor,
        age: torch.Tensor,
        abspos: torch.Tensor,
        segment: torch.LongTensor,
        value_bin: Optional[torch.LongTensor] = None,
        value_normalized: Optional[torch.Tensor] = None,
        value_present: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        embeddings = self.code_embedding(code)

        embeddings += self.age_embedding(age)
        embeddings += self.abspos_embedding(abspos)
        embeddings += self.segment_embedding(segment)
        if value_bin is not None or value_normalized is not None:
            if self.value_bin_embedding is None or self.value_projection is None:
                raise ValueError(
                    "Batch contains numeric value tensors, but the model was "
                    "created with value_bin_vocab_size=0."
                )
            if value_bin is None:
                value_bin = torch.zeros_like(code)
            if value_normalized is None:
                value_normalized = torch.zeros_like(age)
            if value_present is None:
                value_present = value_bin != 0
            value_present = value_present.bool().unsqueeze(-1)
            value_normalized = torch.nan_to_num(value_normalized.float()).unsqueeze(-1)
            value_embeddings = self.value_bin_embedding(value_bin.long())
            value_embeddings += self.value_projection(value_normalized)
            embeddings += value_embeddings * value_present.to(embeddings.dtype)

        return embeddings


class Time2Vec(nn.Module):
    """Time2Vec embedding layer that combines linear and periodic components.

    This layer transforms temporal inputs using a combination of linear and periodic embeddings:
    - First component (i=0): linear transformation w0*t + phi0
    - Remaining components: periodic transformations f(w*t + phi)

    The linear component can be clipped to a specified range.

    Parameters:
        output_dim: int
            Dimension of the output embedding vector. Default: 768
        function: callable
            Periodic function to use (e.g., torch.cos). Default: torch.cos
        clip_range: float, optional
            -Minimum/maximum value for clipping the linear component

    Forward Input:
        tau: torch.Tensor
            Input temporal values of shape (batch_size, sequence_length)

    Returns:
        torch.Tensor: Concatenated linear and periodic embeddings
            of shape (batch_size, sequence_length, output_dim)
    """

    def __init__(
        self,
        output_dim: int = 768,
        function: callable = torch.cos,
        clip_range: Optional[float] = None,
    ):
        """
        Parameters:
            output_dim: int - dimension of the output
            function: callable - function to use for the time2vec transformation
            clip_min: float - minimum value of the output
            clip_max: float - maximum value of the output
        """
        super().__init__()
        self.f = function
        self.clip_range = clip_range
        # for i = 0
        self.w0 = torch.nn.Parameter(torch.randn(1, 1))
        self.phi0 = torch.nn.Parameter(torch.randn(1))
        # for 1 <= i <= k (output_dim)
        self.w = torch.nn.Parameter(torch.randn(1, output_dim - 1))
        self.phi = torch.nn.Parameter(torch.randn(output_dim - 1))

    def forward(self, tau: torch.Tensor) -> torch.Tensor:
        tau = tau.unsqueeze(2)  # (batch_size, sequence_length, 1)

        linear_1 = torch.matmul(tau, self.w0) + self.phi0
        linear_2 = torch.matmul(tau, self.w)

        if self.clip_range is not None:
            linear_1 = torch.clamp(linear_1, -self.clip_range, self.clip_range)

        periodic = self.f(linear_2 + self.phi)

        return torch.cat((linear_1, periodic), dim=-1)
