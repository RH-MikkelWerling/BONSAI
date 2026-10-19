"""Check the finished local pretraining run before it is frozen into a bundle.

Runs the same encoder audit the paper workers apply (architecture, vocabulary,
completion, strict load and forward pass) and summarises the training curve:
completed epochs, the selected epoch, and validation loss at epoch 10 and at the
selected epoch (for comparison with the 10-epoch PHAIR encoder).

    python -m opera.run.audit_local_checkpoint --run-dir <run>/version_0 \
        --vocabulary <OPERA_SEQUENCE_DATA_DIR>/vocabulary.pt
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def curve_summary(metrics_path: Path) -> dict:
    """Per-epoch validation metrics from the Lightning CSV log."""
    metrics = pd.read_csv(metrics_path)
    validation = [c for c in metrics.columns if c.startswith("val")]
    per_epoch = (metrics.dropna(subset=validation, how="all")
                 .groupby("epoch")[validation].last().reset_index())
    loss = "val/loss" if "val/loss" in per_epoch else validation[0]
    best = per_epoch.loc[per_epoch[loss].idxmin()]
    summary = {
        "logged_epochs": int(per_epoch.epoch.max()) + 1,
        "selection_metric": loss,
        "best_epoch": int(best.epoch) + 1,
        "best_value": float(best[loss]),
        "curve": {int(e) + 1: float(v) for e, v in zip(per_epoch.epoch, per_epoch[loss])},
    }
    at_ten = per_epoch.loc[per_epoch.epoch == 9, loss]
    if not at_ten.empty:
        summary["value_at_epoch_10"] = float(at_ten.iloc[0])
        summary["improvement_after_epoch_10"] = float(at_ten.iloc[0] - best[loss])
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True,
                        help="Directory with best.ckpt, checkpoint_metadata.json, metrics.csv")
    parser.add_argument("--vocabulary", type=Path, required=True)
    parser.add_argument("--minimum-epochs", type=int, default=10)
    parser.add_argument("--width", type=int, default=256)
    args = parser.parse_args()
    import torch

    from opera.run.run_paper_queue import audit_encoder

    report = {"run_dir": str(args.run_dir)}
    sidecar = args.run_dir / "checkpoint_metadata.json"
    report["sidecar"] = json.loads(sidecar.read_text()) if sidecar.is_file() else None
    for name in ("best.ckpt", "epoch_010.ckpt", "last.ckpt"):
        path = args.run_dir / name
        if path.is_file():
            checkpoint = torch.load(path, map_location="cpu", weights_only=False)
            report[name] = {"epoch": int(checkpoint["epoch"]) + 1,
                            "global_step": int(checkpoint.get("global_step", -1))}
        else:
            report[name] = "missing"
    metrics = args.run_dir / "metrics.csv"
    report["curve"] = curve_summary(metrics) if metrics.is_file() else "metrics.csv missing"
    config = {"dapt_ckpt": str(args.run_dir / "best.ckpt"),
              "paths": {"vocabulary": str(args.vocabulary)}}
    try:
        audit = audit_encoder(config, "L", expected_width=args.width,
                              minimum_local_epochs=args.minimum_epochs)
        report["encoder_audit"] = {k: audit[k] for k in ("status", "architecture")}
    except (ValueError, KeyError, RuntimeError) as error:
        report["encoder_audit"] = f"FAILED: {error}"
    print(json.dumps(report, indent=2, default=str))


if __name__ == "__main__":
    main()
