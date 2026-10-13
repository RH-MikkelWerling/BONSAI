"""Compact validation-only run inventory; never exports predictions/checkpoints.

Usage: python -m opera.run.bundle_recent_validation --root RUN_ROOT
       --root OTHER_RUN_ROOT --output recent_validation.tar.gz
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import re
import subprocess
import tarfile
from datetime import datetime, timezone
from pathlib import Path

SAFE_NAMES = {
    "config.yaml",
    "overrides.yaml",
    "hparams.yaml",
    "checkpoint_metadata.json",
    "training_complete.json",
    "competing_risk_interval_support.csv",
    "competing_risk_null_log_hazards.csv",
    "source_training_support.csv",
}
SEALED = re.compile(r"(^|[/_.-])(held_?out|test|evaluation)([/_.-]|$)", re.I)


def sanitized_metrics(data: str) -> tuple[str, list[dict], list[str]]:
    reader = csv.DictReader(io.StringIO(data))
    fields = [
        k
        for k in reader.fieldnames or []
        if k in {"epoch", "step"} or k.startswith(("train/", "val/"))
    ]
    rows = [{k: row.get(k, "") for k in fields} for row in reader]
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue(), rows, fields


def metric_summary(rows: list[dict], fields: list[str]) -> list[dict]:
    summaries = []
    for name in fields:
        if not name.startswith("val/"):
            continue
        values = []
        for row in rows:
            try:
                value = float(row.get(name, ""))
            except (TypeError, ValueError):
                continue
            if math.isfinite(value):
                values.append((value, row.get("epoch"), row.get("step")))
        if values:
            minimum, maximum = min(values), max(values)
            summaries.append(
                dict(
                    metric=name,
                    minimum=minimum[0],
                    maximum=maximum[0],
                    maximum_epoch=maximum[1],
                    minimum_epoch=minimum[1],
                    last=values[-1][0],
                    last_epoch=values[-1][1],
                )
            )
    return summaries


def build_bundle(roots: list[Path], output: Path, repo: Path) -> dict:
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    for root in roots:
        if not root.is_dir():
            raise FileNotFoundError(root)
    payloads: dict[str, bytes] = {}
    inventory, summaries, manifests, skipped = [], [], [], []

    def add(name, data):
        payloads[name] = data
        manifests.append(
            dict(path=name, bytes=len(data), sha256=hashlib.sha256(data).hexdigest())
        )

    for root_index, root in enumerate(roots):
        # Accept a suite root or an individual run directory.
        runs = (
            [root]
            if any(
                (root / n).exists()
                for n in (
                    "checkpoint_metadata.json",
                    "training_complete.json",
                    ".hydra",
                    "metrics.csv",
                )
            )
            else sorted(p for p in root.iterdir() if p.is_dir())
        )
        for run in runs:
            run_id = f"root{root_index}/{run.name}"
            included = 0
            config_text = ""
            metrics_files = 0
            for path in sorted(run.rglob("*")):
                if not path.is_file() or path.is_symlink():
                    continue
                rel = path.relative_to(run).as_posix()
                if SEALED.search(rel) or "prediction" in rel.lower():
                    skipped.append(
                        dict(
                            run=run_id, path=rel, reason="sealed or patient predictions"
                        )
                    )
                    continue
                if (
                    path.name not in SAFE_NAMES
                    and path.name != "metrics.csv"
                    and not path.name.endswith("_metadata.json")
                ):
                    continue
                # Some root-level tabular metadata describes held-out evaluations.
                if (
                    path.name.endswith("_metadata.json")
                    and path.name != "checkpoint_metadata.json"
                ):
                    meta = json.loads(path.read_text())
                    if meta.get("evaluation_split") not in {
                        "tuning",
                        "validation",
                        "val",
                    }:
                        skipped.append(
                            dict(
                                run=run_id,
                                path=rel,
                                reason="metadata not explicitly validation",
                            )
                        )
                        continue
                data = path.read_bytes()
                if path.name == "metrics.csv":
                    clean, rows, fields = sanitized_metrics(data.decode("utf-8-sig"))
                    data = clean.encode()
                    metrics_files += 1
                    summaries.extend(
                        dict(run=run_id, metrics_file=rel, **item)
                        for item in metric_summary(rows, fields)
                    )
                if rel == ".hydra/config.yaml" or (
                    path.name == "config.yaml" and not config_text
                ):
                    config_text = data.decode("utf-8")
                add(f"runs/{run_id}/{rel}", data)
                included += 1
            if not included:
                continue

            def setting(key):
                match = re.search(
                    r"^\s*" + re.escape(key) + r":\s*(.+)$", config_text, re.M
                )
                return match.group(1).strip() if match else ""

            inventory.append(
                dict(
                    run=run_id,
                    source_directory=str(run.resolve()),
                    complete=(run / "training_complete.json").exists(),
                    locked=(run / ".launch_lock").exists(),
                    metrics_files=metrics_files,
                    objective=setting("objective"),
                    readout=setting("readout_mode"),
                    query=setting("adapter_query_mode") or "unspecified/legacy",
                    frozen=setting("freeze_encoder"),
                    aggregation=setting("aggregation"),
                    monitor=setting("eval_monitor_metric"),
                    seed=setting("seed"),
                    source_checkpoint=setting("source_opera_ckpt"),
                )
            )
    if not inventory:
        raise ValueError("No supported run artifacts found")

    def csv_bytes(rows):
        stream = io.StringIO(newline="")
        fields = list(rows[0]) if rows else ["run"]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
        return stream.getvalue().encode()

    add("run_inventory.csv", csv_bytes(inventory))
    add("validation_metric_extrema.csv", csv_bytes(summaries))
    add("excluded_artifacts.csv", csv_bytes(skipped))
    sources = [
        "opera/modules/networks/competing_risk.py",
        "opera/modules/networks/source_anchored_readout.py",
        "opera/modules/networks/opera_nets.py",
        "opera/modules/lightningmodules/OperaContrastiveModule.py",
        "opera/run/contrastive_multicohort.py",
        "opera/run/generate_sweep_configs.py",
        "opera/configs/experiment_registry.yaml",
        "opera/analysis/bayesian_rarity.py",
        "docs/recent_experiment_review.md",
    ]
    for name in sources:
        path = repo / name
        if path.is_file():
            add(f"current_checkout/{name}", path.read_bytes())
    for command, name in [
        (["git", "rev-parse", "HEAD"], "git_revision.txt"),
        (["git", "status", "--short"], "git_status.txt"),
    ]:
        try:
            result = subprocess.run(command, cwd=repo, capture_output=True, check=False)
            add(f"current_checkout/{name}", result.stdout + result.stderr)
        except OSError:
            pass
    note = dict(
        created_utc=datetime.now(timezone.utc).isoformat(),
        runs=len(inventory),
        policy="Validation only; no predictions, patient tables, logs, embeddings or checkpoints.",
        limitations=[
            "Metric extrema are NOT selected-checkpoint performance.",
            "Live runs are non-atomic snapshots; rerun after completion.",
            "Current checkout is not proof of the code used in historical runs.",
            "Missing per-outcome/cohort validation metrics require subsequent server evaluation.",
        ],
    )
    add("README.json", json.dumps(note, indent=2).encode())
    payloads["manifest.json"] = json.dumps(manifests, indent=2).encode()
    output.parent.mkdir(parents=True, exist_ok=True)
    with (
        output.open("xb") as stream,
        tarfile.open(fileobj=stream, mode="w:gz") as archive,
    ):
        for name, data in sorted(payloads.items()):
            info = tarfile.TarInfo(f"recent_validation/{name}")
            info.size = len(data)
            archive.addfile(info, io.BytesIO(data))
    return note


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[2]
    note = build_bundle([p.resolve() for p in args.root], args.output.resolve(), repo)
    print(json.dumps(note, indent=2))
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
