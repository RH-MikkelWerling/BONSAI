"""
Analyse the learned Kendall sigma values from an OPERA contrastive run.

This script is specifically designed to extract the scientifically
interesting σ_k parameters — useful even if the contrastive stage
doesn't improve downstream performance.

Usage:
    python -m opera.run.analyse_sigmas \
        contrastive_ckpt=/path/to/contrastive/best.ckpt \
        training_log_dir=/path/to/contrastive_runs/version_0/ \
        output_dir=./sigma_analysis
"""

import sys
import torch
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List

from opera.visualization.embedding_plots import (
    plot_sigma_evolution,
    plot_sigma_barplot,
)


def extract_sigma_values(ckpt_path: str) -> Dict[str, float]:
    """Extract final σ values from a contrastive checkpoint."""
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state_dict = ckpt["state_dict"]

    # Find log_sigma parameters
    log_sigma_key = None
    for k in state_dict:
        if "log_sigma" in k:
            log_sigma_key = k
            break

    if log_sigma_key is None:
        raise ValueError("No log_sigma found in checkpoint. Is this a contrastive checkpoint?")

    log_sigma = state_dict[log_sigma_key]
    sigma = torch.exp(log_sigma).numpy()

    # Try to find outcome names from the checkpoint
    # They should be in the hyper_parameters or model structure
    hparams = ckpt.get("hyper_parameters", {})
    outcome_names = hparams.get("outcome_names", None)

    if outcome_names is None:
        # Fall back to generic names
        outcome_names = [f"outcome_{i}" for i in range(len(sigma))]

    return {name: float(s) for name, s in zip(outcome_names, sigma)}


def load_training_log(log_dir: str) -> pd.DataFrame:
    """Load training metrics CSV from a Lightning CSVLogger directory."""
    log_dir = Path(log_dir)
    metrics_path = log_dir / "metrics.csv"
    if not metrics_path.exists():
        # Try one level down
        candidates = list(log_dir.glob("**/metrics.csv"))
        if candidates:
            metrics_path = candidates[0]
        else:
            raise FileNotFoundError(f"No metrics.csv found in {log_dir}")

    return pd.read_csv(metrics_path)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Analyse OPERA contrastive sigma values")
    parser.add_argument("--contrastive_ckpt", required=True, help="Path to contrastive checkpoint")
    parser.add_argument("--training_log_dir", default=None, help="Path to CSVLogger output dir")
    parser.add_argument("--output_dir", default="./sigma_analysis", help="Output directory")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Extract final σ values ───────────────────────────────────────
    print("Extracting sigma values from checkpoint...")
    sigma_values = extract_sigma_values(args.contrastive_ckpt)

    print("\nFinal learned σ values:")
    print("-" * 40)
    for name, val in sorted(sigma_values.items(), key=lambda x: x[1]):
        precision = 0.5 * np.exp(-2 * np.log(val))
        print(f"  {name:30s}  σ={val:.4f}  precision={precision:.4f}")

    # ── Save as CSV ──────────────────────────────────────────────────
    sigma_df = pd.DataFrame([
        {"outcome": name, "sigma": val,
         "precision": 0.5 * np.exp(-2 * np.log(val)),
         "log_sigma": np.log(val)}
        for name, val in sigma_values.items()
    ])
    sigma_df.to_csv(output_dir / "sigma_values.csv", index=False)

    # ── Bar plot of final σ values ───────────────────────────────────
    plot_sigma_barplot(
        sigma_values,
        save_path=str(output_dir / "sigma_barplot.png"),
    )
    print(f"\nSigma bar plot saved to {output_dir / 'sigma_barplot.png'}")

    # ── Evolution plots if training log available ────────────────────
    if args.training_log_dir:
        print("\nLoading training log for evolution plots...")
        try:
            log_df = load_training_log(args.training_log_dir)
            outcome_names = list(sigma_values.keys())

            plot_sigma_evolution(
                log_df, outcome_names,
                save_path=str(output_dir / "sigma_evolution.png"),
            )
            print(f"Sigma evolution plot saved to {output_dir / 'sigma_evolution.png'}")
        except Exception as e:
            print(f"Could not generate evolution plot: {e}")

    print(f"\nAnalysis complete. Results in: {output_dir}")


if __name__ == "__main__":
    main()
