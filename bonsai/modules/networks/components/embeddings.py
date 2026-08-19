import math
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
        value_embedding_mode: str = "legacy",
        abspos_encoding: str = "legacy",
    ):
        super().__init__()

        # Initialize embeddings
        self.code_embedding = nn.Embedding(vocab_size, hidden_size, padding_idx=0)
        self.segment_embedding = nn.Embedding(max_seqlen, hidden_size, padding_idx=0)
        self.age_embedding = Time2Vec(hidden_size, clip_range=100)
        if abspos_encoding == "legacy":
            self.abspos_embedding = Time2Vec(hidden_size, clip_range=100)
        elif abspos_encoding == "fourier":
            self.abspos_embedding = AbsposFourierEncoding(hidden_size)
        elif abspos_encoding in {
            "sequence_relative_fourier",
            "sequence_relative_fourier_with_gaps",
        }:
            self.abspos_embedding = RelativeAbsposFourierEncoding(hidden_size)
            self.time_gap_embedding = (
                LogTimeGapEncoding(hidden_size)
                if abspos_encoding == "sequence_relative_fourier_with_gaps"
                else None
            )
        elif abspos_encoding == "none":
            self.abspos_embedding = ZeroTimeEncoding(hidden_size)
        else:
            raise ValueError(
                "Unknown abspos_encoding "
                f"{abspos_encoding!r}; expected 'legacy', 'fourier', "
                "'sequence_relative_fourier', "
                "'sequence_relative_fourier_with_gaps', or 'none'."
            )
        self.abspos_encoding = abspos_encoding
        self.value_bin_vocab_size = int(value_bin_vocab_size)
        self.value_embedding_mode = value_embedding_mode
        if value_embedding_mode not in {"legacy", "combined_binning", "film"}:
            raise ValueError(f"Unknown value_embedding_mode: {value_embedding_mode!r}")
        if value_embedding_mode == "film":
            if self.value_bin_vocab_size != 0:
                raise ValueError(
                    "film value embedding requires value_bin_vocab_size=0."
                )
            self.continuous_value_embedding = ContinuousValueEmbedding(hidden_size)
        else:
            self.continuous_value_embedding = None
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
        numeric_value: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        embeddings = self.code_embedding(code)

        if numeric_value is not None:
            if self.continuous_value_embedding is None:
                raise ValueError(
                    "Batch contains numeric_value, but the model was not created "
                    "with value_embedding_mode='film'."
                )
            embeddings = self.continuous_value_embedding(numeric_value, embeddings)

        embeddings += self.age_embedding(age)
        if self.abspos_encoding.startswith("sequence_relative_fourier"):
            valid = code != 0
            clinical = valid & (segment > 0)
            fallback = abspos.masked_fill(~valid, float("inf")).amin(dim=1)
            anchor = abspos.masked_fill(~clinical, float("inf")).amin(dim=1)
            anchor = torch.where(torch.isfinite(anchor), anchor, fallback)
            anchor = torch.where(torch.isfinite(anchor), anchor, torch.zeros_like(anchor))
            abspos_input = abspos - anchor.unsqueeze(1)
            # Background tokens describe the patient at birth and already use
            # the age channel. They are not part of elapsed clinical time.
            abspos_input = abspos_input.masked_fill(~clinical, 0.0)
        else:
            abspos_input = abspos
        embeddings += self.abspos_embedding(abspos_input)
        if getattr(self, "time_gap_embedding", None) is not None:
            delta_hours = torch.zeros_like(abspos)
            delta_hours[:, 1:] = (abspos[:, 1:] - abspos[:, :-1]).clamp_min(0)
            previous_clinical = torch.nn.functional.pad(
                clinical[:, :-1], (1, 0), value=False
            )
            delta_hours = delta_hours.masked_fill(
                ~(clinical & previous_clinical), 0.0
            )
            embeddings += self.time_gap_embedding(delta_hours)
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
            if self.value_embedding_mode == "combined_binning":
                # The normalized bin representative is projected in place of
                # the [VAL] code embedding. Temporal/segment features remain.
                value_embeddings = self.value_projection(value_normalized)
                embeddings = torch.where(value_present, value_embeddings, embeddings)
            elif self.value_embedding_mode == "legacy":
                value_embeddings = self.value_bin_embedding(value_bin.long())
                value_embeddings += self.value_projection(value_normalized)
                embeddings += value_embeddings * value_present.to(embeddings.dtype)
            else:
                raise ValueError(
                    "Binned value tensors are incompatible with "
                    f"value_embedding_mode={self.value_embedding_mode!r}."
                )

        return embeddings


class ContinuousValueEmbedding(nn.Module):
    """Collaborator-style FiLM fusion for normalized continuous values."""

    def __init__(self, hidden_size: int):
        super().__init__()
        self.value_projection = nn.Sequential(
            nn.Linear(1, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
        )
        self.gamma = nn.Linear(hidden_size, hidden_size)
        self.beta = nn.Linear(hidden_size, hidden_size)

    def forward(
        self,
        values: torch.Tensor,
        concept_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        present = torch.isfinite(values).unsqueeze(-1)
        safe_values = torch.where(
            torch.isfinite(values), values, torch.zeros_like(values)
        )
        value_embeddings = self.value_projection(safe_values.float().unsqueeze(-1))
        value_embeddings = value_embeddings.to(concept_embeddings.dtype)
        fused = self.gamma(concept_embeddings) * value_embeddings
        fused = fused + self.beta(concept_embeddings)
        return torch.where(present, fused, concept_embeddings)


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
        # Absolute position is expressed in hours since the Unix epoch and is
        # therefore around 450,000 for contemporary records. FP16 cannot
        # represent values above 65,504, so autocasting this small transform to
        # FP16 produces ``inf`` before the cosine and consequently NaN model
        # losses on Volta GPUs. Keep Time2Vec in FP32; the surrounding
        # transformer can still use mixed precision.
        output_dtype = self.w.dtype
        with torch.autocast(device_type=tau.device.type, enabled=False):
            tau_float = tau.float().unsqueeze(2)
            linear_1 = torch.matmul(tau_float, self.w0.float()) + self.phi0.float()
            linear_2 = torch.matmul(tau_float, self.w.float())

            if self.clip_range is not None:
                linear_1 = torch.clamp(
                    linear_1,
                    -self.clip_range,
                    self.clip_range,
                )

            periodic = self.f(linear_2 + self.phi.float())
            output = torch.cat((linear_1, periodic), dim=-1)

        return output.to(dtype=output_dtype)


class AbsposFourierEncoding(nn.Module):
    """Fixed-frequency calendar encoding for Unix-epoch hours.

    The first channel is normalized calendar time. Remaining channels are
    paired sine/cosine features with geometrically spaced periods; an even
    output dimension has one final zero-filled slot.
    """

    def __init__(
        self,
        output_dim: int = 768,
        min_period_years: float = 1.0,
        max_period_years: float = 80.0,
        linear_ref_years: float = 12.0,
        linear_scale_years: float = 15.0,
    ):
        super().__init__()
        if output_dim < 1:
            raise ValueError("output_dim must be positive.")
        if min_period_years <= 0 or max_period_years <= min_period_years:
            raise ValueError(
                "Period range must satisfy 0 < min_period_years < max_period_years."
            )
        if linear_scale_years <= 0:
            raise ValueError("linear_scale_years must be positive.")

        num_pairs = (output_dim - 1) // 2
        self.output_dim = output_dim
        self.num_pairs = num_pairs
        self.register_buffer("epoch_2000_hours", torch.tensor(30.0 * 8766.0))
        self.register_buffer("hours_per_year", torch.tensor(8766.0))
        self.register_buffer("linear_ref", torch.tensor(float(linear_ref_years)))
        self.register_buffer("linear_scale", torch.tensor(float(linear_scale_years)))

        if num_pairs:
            periods = torch.logspace(
                math.log10(min_period_years),
                math.log10(max_period_years),
                num_pairs,
            )
            frequencies = (2.0 * math.pi) / periods
        else:
            periods = torch.empty(0)
            frequencies = torch.empty(0)
        self.register_buffer("periods", periods)
        self.register_buffer("frequencies", frequencies)
        self.phi = nn.Parameter(torch.zeros(num_pairs))

    def forward(self, tau: torch.Tensor) -> torch.Tensor:
        output_dtype = self.phi.dtype
        with torch.autocast(device_type=tau.device.type, enabled=False):
            tau_years = (
                tau.float() - self.epoch_2000_hours.float()
            ) / self.hours_per_year.float()
            linear = (
                (tau_years - self.linear_ref.float()) / self.linear_scale.float()
            ).unsqueeze(-1)
            angles = (
                tau_years.unsqueeze(-1) * self.frequencies.float() + self.phi.float()
            )
            periodic = torch.stack((torch.sin(angles), torch.cos(angles)), dim=-1)
            periodic = periodic.flatten(start_dim=-2)
            output = torch.cat((linear, periodic), dim=-1)
            if output.shape[-1] < self.output_dim:
                output = torch.cat((output, torch.zeros_like(linear)), dim=-1)
        return output.to(dtype=output_dtype)


class RelativeAbsposFourierEncoding(AbsposFourierEncoding):
    """Fixed Fourier features for hours relative to each sequence endpoint."""

    def forward(self, tau: torch.Tensor) -> torch.Tensor:
        output_dtype = self.phi.dtype
        with torch.autocast(device_type=tau.device.type, enabled=False):
            tau_years = tau.float() / self.hours_per_year.float()
            linear = (tau_years / self.linear_scale.float()).unsqueeze(-1)
            angles = tau_years.unsqueeze(-1) * self.frequencies.float() + self.phi.float()
            periodic = torch.stack((torch.sin(angles), torch.cos(angles)), dim=-1)
            output = torch.cat((linear, periodic.flatten(start_dim=-2)), dim=-1)
            if output.shape[-1] < self.output_dim:
                output = torch.cat((output, torch.zeros_like(linear)), dim=-1)
        return output.to(dtype=output_dtype)


class ZeroTimeEncoding(nn.Module):
    """A parameter-free absolute-calendar ablation with a stable output shape."""

    def __init__(self, output_dim: int):
        super().__init__()
        self.output_dim = int(output_dim)

    def forward(self, tau: torch.Tensor) -> torch.Tensor:
        return tau.new_zeros((*tau.shape, self.output_dim))


class LogTimeGapEncoding(nn.Module):
    """Trainable encoding of log(1 + days since the preceding event group)."""

    def __init__(self, output_dim: int):
        super().__init__()
        self.time2vec = Time2Vec(output_dim, clip_range=20)

    def forward(self, delta_hours: torch.Tensor) -> torch.Tensor:
        log_days = torch.log1p(delta_hours.float().clamp_min(0) / 24.0)
        return self.time2vec(log_days)
