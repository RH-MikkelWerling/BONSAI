"""Compare legacy and Fourier absolute-position encodings on a checkpoint."""

from __future__ import annotations

import argparse
import math
from datetime import datetime, timezone
from pathlib import Path

import torch

from bonsai.modules.networks.components.embeddings import (
    AbsposFourierEncoding,
    Time2Vec,
)


def _monthly_grid() -> torch.Tensor:
    values = []
    year, month = 2003, 1
    while (year, month) <= (2026, 7):
        timestamp = datetime(year, month, 1, tzinfo=timezone.utc).timestamp()
        values.append(timestamp / 3600.0)
        month += 1
        if month == 13:
            year += 1
            month = 1
    return torch.tensor(values)


def _find_tensor(state: dict[str, torch.Tensor], suffix: str) -> torch.Tensor:
    matches = [value for key, value in state.items() if key.endswith(suffix)]
    if len(matches) != 1:
        raise KeyError(f"Expected one checkpoint tensor ending in {suffix!r}.")
    return matches[0]


def _translation_std(encoder, grid: torch.Tensor, gap_years: float) -> float:
    output = encoder(grid)
    shifted = encoder(grid + gap_years * 8766.0)
    if isinstance(encoder, AbsposFourierEncoding):
        stop = 1 + 2 * encoder.num_pairs
        output, shifted = output[:, 1:stop], shifted[:, 1:stop]
    else:
        output, shifted = output[:, 1:], shifted[:, 1:]
    return float((output * shifted).sum(dim=-1).std())


def _print_metrics(name, encoder, grid, code_norm):
    with torch.no_grad():
        output = encoder(grid)
    linear = output[:, 0]
    saturated = (linear.abs() >= 100.0 - 1e-6).float().mean()
    mean_norm = output.norm(dim=-1).mean()
    if isinstance(encoder, AbsposFourierEncoding):
        periods = encoder.periods
    else:
        periods = (2.0 * math.pi / encoder.w.detach().abs().flatten()) / 8766.0
    gap_stds = {
        label: _translation_std(encoder, grid, gap)
        for label, gap in (("1_month", 1 / 12), ("1_year", 1), ("5_years", 5))
    }
    print(f"\n{name}")
    print(f"  linear range: [{linear.min():.4f}, {linear.max():.4f}]")
    print(f"  saturated fraction: {saturated:.6f}")
    print(f"  mean output L2 norm: {mean_norm:.4f}")
    print(f"  ratio to mean code norm ({code_norm:.4f}): {mean_norm / code_norm:.4f}")
    print(f"  translation-invariance std: {gap_stds}")
    print(f"  implied period range (years): [{periods.min():.4f}, {periods.max():.4f}]")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    state = checkpoint.get("state_dict", checkpoint)
    code_weight = _find_tensor(state, "embeddings.code_embedding.weight")
    hidden_size = code_weight.shape[1]
    code_norm = float(code_weight.norm(dim=-1).mean())

    legacy = Time2Vec(hidden_size, clip_range=100)
    legacy.load_state_dict(
        {
            name: _find_tensor(state, f"embeddings.abspos_embedding.{name}")
            for name in ("w0", "phi0", "w", "phi")
        }
    )
    fourier = AbsposFourierEncoding(hidden_size)
    grid = _monthly_grid()
    _print_metrics("legacy (checkpoint weights)", legacy, grid, code_norm)
    _print_metrics("fourier (fresh initialization)", fourier, grid, code_norm)


if __name__ == "__main__":
    main()
