"""GPU smoke test of the stage-8 router fits (31 October 2026), before launch.

Each fit trains for a few dozen batches from its exact bundle config, in the
production precision and on the real data, with routing live from the first
step (the router warm-up set to 0; the frozen control stays frozen), in an
isolated directory that is never a paper fit. ``--check`` then reports, per
fit: whether training finished; the mixed-precision loss scale (a scale that
fell to near zero means steps were skipped, as with the zero-query overflow
of 31 October); whether parameters that start at exact values moved (tau,
borrowed scale, patient gate); the validation correction sizes; and whether
the checkpoint reloads through the evaluation and audit paths. Aggregates only.

    python -m opera.run.router_smoke run --bundle-dir .../bundles/stage8 \\
        --runs-dir .../router_smoke --shard 1/6 --execute
    python -m opera.run.router_smoke check --bundle-dir ... --runs-dir ...
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import subprocess
import sys
from pathlib import Path

import yaml

STAGE8_PREFIXES = ("fix_R_PF", "fix_R_PH", "fix_R_PL")


def stage8_rows(bundle_dir):
    bundle = json.loads((Path(bundle_dir) / "bundle.json").read_text())
    rows = [row for row in bundle["runs"]
            if row["phase"] == "router_fix" and row["name"].split("/")[-1].startswith(STAGE8_PREFIXES)]
    return sorted(rows, key=lambda row: row.get("queue_order", 0))


def alias(row):
    return row["name"].split("fix_", 1)[1]


def run(args):
    rows = stage8_rows(args.bundle_dir)
    if args.fit:
        rows = [row for row in rows if alias(row) in set(args.fit)]
    if args.shard:
        index, total = (int(x) for x in args.shard.split("/"))
        rows = rows[index - 1::total]
    from opera.run.paper_experiments import load_environment

    # The worker's environment: .env values, with the scheduler's GPU taking precedence.
    env = load_environment(args.env_file)
    for row in rows:
        name = alias(row)
        source = Path(args.bundle_dir) / row["config_path"]
        config = yaml.safe_load(source.read_text())
        frozen_control = name.endswith("FROZEN")
        config = copy.deepcopy(config)
        config["training"].update(epochs=1, limit_train_batches=int(args.train_batches),
                                  limit_val_batches=float(args.val_fraction),
                                  early_stopping_patience=None)
        config["competing_risk"]["adapter_router_warmup_epochs"] = 1 if frozen_control else 0
        config["paper_smoke_only"] = True
        output = Path(args.runs_dir) / name
        print(f"{name}: {output}", flush=True)
        if not args.execute:
            continue
        if output.exists():
            raise FileExistsError(f"Use a fresh smoke directory: {output}")
        output.mkdir(parents=True)
        (output / "smoke.yaml").write_text("# @package _global_\n" + yaml.safe_dump(config, sort_keys=False))
        with (output / "smoke.log").open("w") as log:
            for command in (
                [sys.executable, "-m", "opera.run.preflight_direct_competing_risk",
                 "--config-dir", str(output.resolve()), "--config-name", "smoke",
                 "--require-existing-paths"],
                [sys.executable, "-u", "-m", "opera.run.contrastive_multicohort",
                 "--config-path", str(output.resolve()), "--config-name", "smoke",
                 f"hydra.run.dir={output.resolve()}"],
            ):
                result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, env=env)
                if result.returncode:
                    print(f"  FAILED (see {output / 'smoke.log'})", flush=True)
                    break
            else:
                print("  finished", flush=True)


def _loss_scale(checkpoint):
    for key, value in checkpoint.items():
        if isinstance(value, dict) and "scale" in value and ("amp" in key.lower() or "precision" in key.lower()):
            return float(value["scale"])
    return None


def check_one(row, runs_dir):
    import pandas as pd
    import torch

    name = alias(row)
    output = Path(runs_dir) / name
    report = {"fit": name}
    if not (output / "training_complete.json").is_file():
        log = output / "smoke.log"
        tail = log.read_text(errors="replace").strip().splitlines()[-1:] if log.is_file() else ["not run"]
        return report | {"status": "FAIL: no completion marker", "detail": tail[0][:120]}
    checkpoint = torch.load(output / "best.ckpt", map_location="cpu", weights_only=False)
    config = yaml.safe_load((output / "smoke.yaml").read_text())
    risk = config["competing_risk"]
    problems = []
    report["steps"] = int(checkpoint.get("global_step", 0))
    if report["steps"] < 1:
        problems.append("no optimizer step")
    scale = _loss_scale(checkpoint)
    report["loss_scale"] = scale
    if scale is not None and scale < 1.0:
        problems.append(f"loss scale collapsed to {scale:g}: steps skipped")
    state = {k.split("source_adapter_readout.")[-1]: v for k, v in checkpoint["state_dict"].items()
             if "source_adapter_readout." in k}
    borrows_routed = risk.get("adapter_borrow_values", "raw") != "off"
    frozen = name.endswith("FROZEN")
    if borrows_routed:
        report["pool_scale"] = float(state["pool_scale"])
        if report["pool_scale"] == 1.0:
            problems.append("borrowed scale never moved")
        report["tau_log_ratio"] = float(state["tau_log_ratio"])
        if not frozen and risk.get("adapter_borrow_amount", "support") == "support" and report["tau_log_ratio"] == 0.0:
            problems.append("tau never moved")
    if risk.get("adapter_patient_gate"):
        report["patient_gate"] = float(state["patient_gate"])
        if report["patient_gate"] == 0.0:
            problems.append("patient gate never moved")
    for value_name, tensor in state.items():
        if torch.is_floating_point(tensor) and not torch.isfinite(tensor).all():
            problems.append(f"non-finite {value_name}")
    logs = sorted(output.rglob("metrics.csv"))
    if logs:
        metrics = pd.concat([pd.read_csv(path) for path in logs], ignore_index=True)
        for column, key in (("val/cr_cif_auc_macro", "auc"), ("val/cr/patient_norm", "patient"),
                            ("val/cr/correction_norm/own", "own"),
                            ("val/cr/correction_norm/borrowed", "borrowed"),
                            ("val/cr/correction_norm/global", "global")):
            if column in metrics and metrics[column].notna().any():
                report[key] = float(metrics[column].dropna().iloc[-1])
        for key in ("own", "borrowed", "global", "patient"):
            if key in report and not math.isfinite(report[key]):
                problems.append(f"non-finite {key} correction")
        if "own" not in report:
            problems.append("no correction-size logs")
        elif report["own"] <= 0:
            problems.append("own correction is zero")
        borrows = borrows_routed or risk.get("adapter_disease_groups") or risk.get("adapter_latent_groups")
        if "borrowed" in report and bool(borrows) != (report["borrowed"] > 0):
            problems.append("borrowed correction size does not match the design")
    else:
        problems.append("no metrics.csv")
    try:
        from opera.diagnostics.router_mechanism import FrozenRouterHead

        head = FrozenRouterHead.from_checkpoint(output / "best.ckpt")
        summary = head.readout.pooling_summary()
        saturation = summary.get("saturation", {})
        report["route_entropy_vs_uniform"] = (
            f"{saturation.get('median_task_entropy', float('nan')):.2f}/"
            f"{saturation.get('median_uniform_entropy', float('nan')):.2f}")
    except Exception as error:  # noqa: BLE001 - reported, not raised
        problems.append(f"audit reload failed: {type(error).__name__}: {error}"[:160])
    try:
        from omegaconf import OmegaConf

        from opera.diagnostics.representation_gradient_conflict import _load_model

        _load_model(output / "best.ckpt", OmegaConf.create(config), sorted(config["outcomes"]),
                    None, None, torch.device("cpu"))
    except Exception as error:  # noqa: BLE001
        problems.append(f"evaluation reload failed: {type(error).__name__}: {error}"[:160])
    return report | {"status": "ok" if not problems else "FAIL: " + "; ".join(problems)}


def check(args):
    import pandas as pd

    rows = [check_one(row, args.runs_dir) for row in stage8_rows(args.bundle_dir)]
    frame = pd.DataFrame(rows)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_colwidth", 90)
    print(frame.to_string(index=False, float_format=lambda v: f"{v:.4g}"))
    failed = [row["fit"] for row in rows if row["status"] != "ok"]
    print(f"\n{len(rows) - len(failed)}/{len(rows)} fits passed" + (f"; failed: {', '.join(failed)}" if failed else ""))
    return 1 if failed else 0


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=("run", "check"))
    parser.add_argument("--bundle-dir", type=Path, required=True)
    parser.add_argument("--runs-dir", type=Path, required=True)
    parser.add_argument("--fit", action="append", default=[], help="only these aliases (e.g. R_PF5E)")
    parser.add_argument("--shard", help="i/n: every n-th fit starting at i (one shard per GPU)")
    parser.add_argument("--train-batches", type=int, default=40)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    args = parser.parse_args()
    if args.command == "run":
        run(args)
        return 0
    return check(args)


if __name__ == "__main__":
    sys.exit(main())
