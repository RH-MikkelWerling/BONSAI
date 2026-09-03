"""Compare legacy and Fourier absolute-position encodings on a checkpoint."""

from __future__ import annotations

import argparse
import math
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from bonsai.functional.checkpointing import get_saved_encoder_config
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
    # Time2Vec consumes [batch, sequence], just like the model forward pass.
    return torch.tensor(values).unsqueeze(0)


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
        output, shifted = output[..., 1:stop], shifted[..., 1:stop]
    else:
        output, shifted = output[..., 1:], shifted[..., 1:]
    return float((output * shifted).sum(dim=-1).std())


def _print_metrics(name, encoder, grid, code_norm):
    with torch.no_grad():
        output = encoder(grid)
    linear = output[..., 0]
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
    parser.add_argument(
        "checkpoint",
        type=Path,
        nargs="+",
        help="One or more learned Time2Vec checkpoints.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional PNG path for a side-by-side learned encoding plot.",
    )
    args = parser.parse_args()
    grid = _monthly_grid()
    learned_runs = []
    summary_rows = []
    for path in args.checkpoint:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        state = checkpoint.get("state_dict", checkpoint)
        code_weight = _find_tensor(state, "embeddings.code_embedding.weight")
        hidden_size = code_weight.shape[1]
        code_norm = float(code_weight.norm(dim=-1).mean())
        hparams = checkpoint.get("hyper_parameters", {})
        try:
            config = get_saved_encoder_config(hparams)
        except (KeyError, TypeError, ValueError):
            config = hparams
        mode = str(config.get("abspos_encoding", "legacy"))
        if mode not in {"legacy", "scaled_time2vec"}:
            raise ValueError(
                f"Checkpoint {path} uses abspos_encoding={mode!r}; choose the "
                "original legacy-Time2Vec checkpoint, not a Fourier checkpoint."
            )
        learned = Time2Vec(
            hidden_size,
            clip_range=100,
            input_scale=1e-3 if mode == "scaled_time2vec" else 1.0,
        )
        learned.load_state_dict(
            {
                name: _find_tensor(state, f"embeddings.abspos_embedding.{name}")
                for name in ("w0", "phi0", "w", "phi")
            }
        )
        label = f"{path.parent.name}: {mode}"
        _print_metrics(label, learned, grid, code_norm)
        with torch.no_grad():
            encoded = learned(grid).squeeze(0).numpy()
        learned_runs.append((label, encoded))
        summary_rows.append(
            {
                "checkpoint": str(path),
                "abspos_encoding": mode,
                "input_scale": learned.input_scale,
                "mean_abspos_norm": float(np.linalg.norm(encoded, axis=1).mean()),
                "mean_code_norm": code_norm,
                "abspos_to_code_norm_ratio": float(
                    np.linalg.norm(encoded, axis=1).mean() / code_norm
                ),
                "saturated_fraction": float((np.abs(encoded[:, 0]) >= 100 - 1e-6).mean()),
            }
        )

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        years = np.linspace(2003, 2026 + 6 / 12, grid.shape[1])
        fig, axes = plt.subplots(
            len(learned_runs), 2, figsize=(10, 3.2 * len(learned_runs)), squeeze=False
        )
        for row, (label, encoded) in enumerate(learned_runs):
            axes[row, 0].plot(years, encoded[:, 0], linewidth=1.6)
            axes[row, 0].set_title(label)
            axes[row, 0].set_ylabel("Linear channel")
            image = axes[row, 1].imshow(
                encoded[:, 1:].T,
                origin="lower",
                aspect="auto",
                extent=(years.min(), years.max(), 1, encoded.shape[1] - 1),
                cmap="coolwarm",
            )
            axes[row, 1].set_title("Periodic Time2Vec channels")
            axes[row, 1].set_ylabel("Embedding dimension")
            fig.colorbar(image, ax=axes[row, 1], fraction=0.04)
        for axis in axes[-1]:
            axis.set_xlabel("Calendar year")
        fig.tight_layout()
        fig.savefig(args.output, dpi=200)
        fig.savefig(args.output.with_suffix(".pdf"))
        plt.close(fig)
        pd.DataFrame(summary_rows).to_csv(
            args.output.with_name(f"{args.output.stem}_summary.csv"), index=False
        )
        print(f"Wrote plot to {args.output}")


if __name__ == "__main__":
    main()
