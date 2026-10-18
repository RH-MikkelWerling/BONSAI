"""One-epoch, two-batch GPU smoke test in an isolated non-paper directory."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import subprocess
import sys

import yaml

from opera.run.paper_prediction_queue import bundle_runs
from opera.run.run_paper_queue import audit_encoder


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle-dir", type=Path, required=True)
    parser.add_argument("--runs-dir", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--minimum-local-pretraining-epochs", type=int, default=20)
    parser.add_argument("--local-hidden-size", type=int, default=256)
    args = parser.parse_args()
    _, rows = bundle_runs(args.bundle_dir)
    row, path, base = next(
        item
        for item in rows
        if item[0]["phase"] == "supervision" and item[0]["regime"] == "J"
    )
    config = copy.deepcopy(base)
    config["training"].update(
        epochs=1,
        limit_train_batches=2,
        limit_val_batches=1.0,
        early_stopping_patience=None,
    )
    config["paper_smoke_only"] = True
    output = args.runs_dir / row["scientific_hash"][:20]
    if not args.execute:
        print(
            json.dumps(
                {
                    "mode": "preview",
                    "purpose": "not_a_paper_fit",
                    "directory": str(output),
                    "training_batches": 2,
                    "epochs": 1,
                    "validation": "full_tuning",
                },
                indent=2,
            )
        )
        return
    if output.exists():
        raise FileExistsError("Use a fresh smoke directory")
    audit = audit_encoder(config, "L", expected_width=args.local_hidden_size,
                          minimum_local_epochs=args.minimum_local_pretraining_epochs)
    output.mkdir(parents=True)
    config_path = output / "smoke.yaml"
    config_path.write_text(
        "# @package _global_\n" + yaml.safe_dump(config, sort_keys=False)
    )
    with (output / "smoke.log").open("w") as log:
        subprocess.run(
            [
                sys.executable,
                "-m",
                "opera.run.contrastive_multicohort",
                "--config-path",
                str(output.resolve()),
                "--config-name",
                "smoke",
                f"hydra.run.dir={output.resolve()}",
            ],
            stdout=log,
            stderr=subprocess.STDOUT,
            check=True,
        )
    import torch

    checkpoint = torch.load(
        output / "best.ckpt", map_location="cpu", weights_only=False
    )
    if int(checkpoint.get("global_step", 0)) < 1:
        raise RuntimeError("Smoke test never took an optimizer step")
    # Reload all supervised weights through the evaluation model path on CPU.
    from omegaconf import OmegaConf
    from opera.diagnostics.representation_gradient_conflict import _load_model

    names = sorted(config["outcomes"])
    _load_model(
        output / "best.ckpt",
        OmegaConf.create(config),
        names,
        None,
        None,
        torch.device("cpu"),
    )
    if not (output / "training_complete.json").is_file():
        raise RuntimeError("Smoke test lacks completion marker")
    report = {
        "status": "optimizer_validation_checkpoint_reload_passed",
        "not_a_paper_result": True,
        "global_step": int(checkpoint["global_step"]),
        "encoder_audit": audit,
    }
    (output / "smoke_result.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
