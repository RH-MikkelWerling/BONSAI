"""Encoder-width probe: the joint model fine-tuned from another local encoder.

Takes the stage-1 joint config (all diseases, all outcomes) unchanged except
for the encoder checkpoint, so the only difference from the paper's joint
model is the encoder's width and pretraining length. Runs outside the paper
queue, in its own directory, with the same audit, preflight and trainer.

    python -m opera.run.width_probe --bundle-dir <root>/bundles/stage1 \
        --encoder-checkpoint <512 run>/version_0/best.ckpt --width 512 \
        --label w512_e10 --output-dir <root>/width_probe --execute

Compare its tuning results with the 256 joint model (and, for equal
pretraining length, with a probe from the 256 run's epoch_010.ckpt).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

JOINT = "J_all_diseases_all_outcomes"


def probe_config(bundle_dir, encoder_checkpoint, seed=42):
    path = Path(bundle_dir) / "configs" / f"seed{seed}" / "L" / f"{JOINT}.yaml"
    if not path.is_file():
        raise FileNotFoundError(f"No stage-1 joint config at {path}")
    config = yaml.safe_load(path.read_text())
    config["dapt_ckpt"] = str(Path(encoder_checkpoint).resolve())
    return config


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--bundle-dir", type=Path, required=True)
    parser.add_argument("--encoder-checkpoint", type=Path, required=True)
    parser.add_argument("--width", type=int, required=True)
    parser.add_argument("--label", required=True, help="e.g. w512_e10, w256_e10")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--minimum-epochs", type=int, default=10)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    from opera.run.run_paper_queue import audit_encoder, train_run

    config = probe_config(args.bundle_dir, args.encoder_checkpoint)
    audit = audit_encoder(config, "L", expected_width=args.width,
                          minimum_local_epochs=args.minimum_epochs)
    run_dir = args.output_dir / args.label
    config_path = run_dir / "config" / f"{JOINT}.yaml"
    print(json.dumps({"run_dir": str(run_dir), "encoder": str(args.encoder_checkpoint),
                      "width": args.width, "audit": audit.get("status")}, indent=2))
    if not args.execute:
        print("Preview only; use --execute to train.")
        return
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text("# @package _global_\n" + yaml.safe_dump(config, sort_keys=False))
    train_run({}, config_path, run_dir, audit)


if __name__ == "__main__":
    main()
