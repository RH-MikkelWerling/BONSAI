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


def compare(output_dir, threshold=0.01, reference="w256_e10"):
    """Tuning score of every probe in ``output_dir`` and the prespecified call.

    The score is the one each completed fit used to select its checkpoint
    (tuning macro IPCW AUC for the paper configs). Rule fixed in advance: keep
    the 256-wide encoder unless a wider one gains at least ``threshold``.
    """
    from bonsai.functional.checkpointing import verified_training_selection

    rows = []
    for run in sorted(p for p in Path(output_dir).iterdir() if p.is_dir()):
        try:
            selection = verified_training_selection(run)
        except (ValueError, FileNotFoundError) as error:
            rows.append({"label": run.name, "status": str(error)})
            continue
        audit = json.loads((run / "encoder_audit.json").read_text())
        rows.append({"label": run.name, "status": "complete",
                     "monitor": selection["monitor"], "mode": selection["mode"],
                     "score": float(selection["best_model_score"]),
                     "best_epoch": selection["best_epoch"],
                     "encoder": audit.get("checkpoint") or audit.get("dapt_ckpt")})
    scores = {r["label"]: r for r in rows if r.get("status") == "complete"}
    decision = {"rule": f"keep 256 unless a wider encoder gains >= {threshold} "
                        "(tuning, joint model, equal pretraining length)",
                "probes": rows}
    if reference in scores:
        base = scores[reference]
        sign = 1 if base["mode"] == "max" else -1
        for label, row in scores.items():
            if label != reference:
                gain = sign * (row["score"] - base["score"])
                decision[f"gain_{label}_vs_{reference}"] = round(gain, 4)
                decision["decision"] = ("switch to " + label if gain >= threshold
                                        else "keep 256")
    else:
        decision["decision"] = f"pending: {reference} not complete"
    (Path(output_dir) / "width_decision.json").write_text(json.dumps(decision, indent=2))
    return decision


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--compare", action="store_true",
                        help="Compare the completed probes in --output-dir and record the call")
    parser.add_argument("--bundle-dir", type=Path)
    parser.add_argument("--encoder-checkpoint", type=Path)
    parser.add_argument("--width", type=int)
    parser.add_argument("--label", help="e.g. w512_e10, w256_e10")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--minimum-epochs", type=int, default=10)
    parser.add_argument("--threshold", type=float, default=0.01)
    parser.add_argument("--epochs", type=int,
                        help="Cap fine-tuning epochs (use the same value for every probe)")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if args.compare:
        print(json.dumps(compare(args.output_dir, args.threshold), indent=2))
        return
    if not (args.bundle_dir and args.encoder_checkpoint and args.width and args.label):
        parser.error("--bundle-dir, --encoder-checkpoint, --width and --label are required")
    from opera.run.run_paper_queue import audit_encoder, train_run

    config = probe_config(args.bundle_dir, args.encoder_checkpoint)
    if args.epochs:
        config["training"]["epochs"] = args.epochs
    # The 256 run is still training when the 512 probe finishes: its named
    # epoch-10 milestone is accepted here (and only here).
    audit = audit_encoder(config, "L", expected_width=args.width,
                          minimum_local_epochs=args.minimum_epochs, allow_milestone=True)
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
