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


AUC_METRIC = "val/cr_cif_auc_macro"
GROUP_AUC_METRIC = "val/cr_cif_auc_endpoint_group_macro"
SELECTION_LOSS = "val/cr_horizon_bce_endpoint_group_macro"


def tuning_metric_at(run_dir, epoch, metric):
    """A metric logged at ``epoch`` in the run's CSV logs (last record wins)."""
    import csv

    value = None
    for path in sorted(Path(run_dir).rglob("metrics.csv")):
        with path.open() as handle:
            for record in csv.DictReader(handle):
                if record.get(metric) in (None, "") or record.get("epoch") in (None, ""):
                    continue
                if int(float(record["epoch"])) == int(epoch):
                    value = float(record[metric])
    return value


def curves(output_dir):
    """Per probe and epoch: tuning loss (selection), macro AUC; read-only.

    Safe while the probes train. Shows where each fit's tuning loss bottomed
    out, which is the evidence for the epoch budget and patience.
    """
    import csv

    report = {}
    for run in sorted(p for p in Path(output_dir).iterdir() if p.is_dir()):
        epochs = {}
        for path in sorted(run.rglob("metrics.csv")):
            with path.open() as handle:
                for record in csv.DictReader(handle):
                    if record.get("epoch") in (None, ""):
                        continue
                    row = epochs.setdefault(int(float(record["epoch"])), {})
                    for key, name in (("loss", SELECTION_LOSS), ("auc_macro", AUC_METRIC)):
                        if record.get(name) not in (None, ""):
                            row[key] = round(float(record[name]), 4)
        rows = [{"epoch": e + 1, **v} for e, v in sorted(epochs.items()) if "loss" in v]
        if rows:
            best = min(rows, key=lambda r: r["loss"])
            report[run.name] = {"epochs_validated": len(rows),
                                "best_epoch_by_loss": best["epoch"],
                                "auc_at_best": best.get("auc_macro"), "curve": rows}
    return report


def compare(output_dir, threshold=0.01, reference="w256_e10"):
    """Tuning macro AUC of every probe in ``output_dir`` and the prespecified call.

    Each fit keeps the checkpoint its own selection metric chose (the tuning
    loss in the paper configs); the decision compares tuning macro AUC
    (``val/cr_cif_auc_macro``) at that selected epoch. Rule fixed in advance:
    keep the 256-wide encoder unless a wider one gains at least ``threshold``.
    The endpoint-group macro AUC and the selection score are reported beside it.
    """
    from bonsai.functional.checkpointing import verified_training_selection

    rows = []
    for run in sorted(p for p in Path(output_dir).iterdir() if p.is_dir()):
        try:
            selection = verified_training_selection(run)
        except (ValueError, FileNotFoundError) as error:
            rows.append({"label": run.name, "status": str(error)})
            continue
        epoch = selection["best_epoch"]
        auc = tuning_metric_at(run, epoch, AUC_METRIC)
        if auc is None:
            rows.append({"label": run.name,
                         "status": f"no {AUC_METRIC} logged at selected epoch {epoch}"})
            continue
        audit = json.loads((run / "encoder_audit.json").read_text())
        rows.append({"label": run.name, "status": "complete",
                     "selected_epoch": epoch,
                     "tuning_auc_macro": auc,
                     "tuning_auc_endpoint_group_macro": tuning_metric_at(
                         run, epoch, GROUP_AUC_METRIC),
                     "selection_monitor": selection["monitor"],
                     "selection_score": float(selection["best_model_score"]),
                     "encoder": audit.get("checkpoint") or audit.get("dapt_ckpt")})
    scores = {r["label"]: r for r in rows if r.get("status") == "complete"}
    decision = {"rule": f"keep 256 unless a wider encoder gains >= {threshold} tuning "
                        "macro AUC at its selected epoch (joint model, equal "
                        "pretraining length)",
                "probes": rows}
    if reference in scores:
        base = scores[reference]
        for label, row in scores.items():
            if label != reference:
                gain = row["tuning_auc_macro"] - base["tuning_auc_macro"]
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
    parser.add_argument("--curves", action="store_true",
                        help="Print each probe's tuning curve so far (read-only, safe while training)")
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
    if args.curves:
        print(json.dumps(curves(args.output_dir), indent=2))
        return
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
