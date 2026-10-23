"""Execute one disjoint paper queue inside an existing GPU allocation."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path

import yaml


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def select_runs(
    bundle, phase, decisions=None, *, regime=None, shard_count=1, shard_index=0
):
    if shard_count < 1 or not 0 <= shard_index < shard_count:
        raise ValueError("Invalid shard count/index")
    decisions = decisions or {}
    required = {
        "rank_screen": {"source_mode"},
        "query_screen": {"source_mode", "rank"},
    }.get(phase, set())
    if required - set(decisions):
        raise ValueError(
            f"Phase {phase} needs recorded tuning decisions: {sorted(required)}"
        )
    selected, seen = [], set()
    for row in bundle["runs"]:
        if row["phase"] != phase or (regime and row["regime"] != regime):
            continue
        if any(key in decisions and decisions[key] != value
               for key, value in row["condition"].items()):
            continue
        if row["scientific_hash"] in seen:
            continue
        seen.add(row["scientific_hash"])
        selected.append(row)
    if not selected:
        raise ValueError(
            "No matching runs; final/external phases need a bundle prepared with --router-lock"
        )
    return selected[shard_index::shard_count]


def load_run_config(row, bundle_dir):
    """Verify a bundle config against its hashes and resolve it with Hydra."""
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf

    OmegaConf.register_new_resolver("version", lambda: 0, replace=True)
    path = (Path(bundle_dir) / row["config_path"]).resolve()
    if file_hash(path) != row["config_sha256"]:
        raise ValueError(f"Generated config changed: {path}")
    config = yaml.safe_load(path.read_text())
    if (
        hashlib.sha256(json.dumps(config, sort_keys=True, default=str).encode()).hexdigest()
        != row["scientific_hash"]
    ):
        raise ValueError("Scientific config differs from its bundle identity")
    with initialize_config_dir(config_dir=str(path.parent), version_base="1.2"):
        resolved = OmegaConf.to_container(compose(config_name=path.stem), resolve=True)
    return path, config, resolved


def encoder_key(config, row):
    """Cache key for one encoder audit (checkpoint, vocabulary, origin)."""
    return (
        config["dapt_ckpt"],
        (config.get("paths") or {}).get("vocabulary"),
        row["origin"],
    )


NOT_ESTIMABLE = "not_estimable.json"


class NotEstimable(RuntimeError):
    """The fit cannot be trained by design (e.g. no tuning patients to select on)."""


def not_estimable_reason(log_text):
    """The reason a training log ended in NoPatientsInSplit, or None."""
    found = re.findall(r"NoPatientsInSplit: (.+)", log_text)
    return found[-1].strip() if found else None


class TrainingStalled(RuntimeError):
    """The training process wrote nothing to its log for too long and was stopped."""


# A fit's progress bar writes to its log continuously while it trains or
# validates; this long without output means it is stuck (seen once, in the
# data-loader shutdown after early stopping of a tiny specialist).
STALL_MINUTES = float(os.environ.get("OPERA_PAPER_STALL_MINUTES", "120"))


def wait_with_watchdog(process, log_path, *, stall_seconds, poll_seconds=30):
    """Wait for ``process``; stop its whole process group if the log goes quiet."""
    while process.poll() is None:
        time.sleep(poll_seconds)
        quiet = time.time() - Path(log_path).stat().st_mtime
        if process.poll() is None and quiet > stall_seconds:
            for sig, grace in ((signal.SIGTERM, 60), (signal.SIGKILL, 0)):
                try:
                    os.killpg(process.pid, sig)
                except ProcessLookupError:
                    break
                try:
                    process.wait(timeout=grace or None)
                    break
                except subprocess.TimeoutExpired:
                    continue
            raise TrainingStalled(
                f"No log output for {quiet / 60:.0f} minutes; training stopped "
                f"(retried from last.ckpt): {log_path}")
    return process.returncode


def train_run(row, path, run_dir, audit_record):
    """Preflight and train one bundle run in ``run_dir``; resumes its last.ckpt.

    A fit whose disease has no patients in a split it needs is recorded in
    ``not_estimable.json`` and raises NotEstimable instead of failing.
    """
    run_dir = Path(run_dir)
    (run_dir / "encoder_audit.json").write_text(json.dumps(audit_record, indent=2) + "\n")
    subprocess.run(
        [
            sys.executable,
            "-m",
            "opera.run.preflight_direct_competing_risk",
            "--config-dir",
            str(path.parent),
            "--config-name",
            path.stem,
            "--require-existing-paths",
        ],
        check=True,
    )
    log_path = run_dir / "queue_training.log"
    start = log_path.stat().st_size if log_path.exists() else 0
    command = [
        sys.executable,
        "-u",
        "-m",
        "opera.run.contrastive_multicohort",
        "--config-path",
        str(path.parent),
        "--config-name",
        path.stem,
        f"hydra.run.dir={run_dir.resolve()}",
    ]
    with log_path.open("a") as log:
        # Own process group, so a stalled fit is stopped with its data loaders.
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                   start_new_session=True)
        returncode = wait_with_watchdog(process, log_path,
                                        stall_seconds=STALL_MINUTES * 60)
    if returncode:
        with log_path.open("rb") as handle:
            handle.seek(start)
            reason = not_estimable_reason(handle.read().decode(errors="replace"))
        if reason:
            (run_dir / NOT_ESTIMABLE).write_text(
                json.dumps({"name": row["name"], "reason": reason}, indent=2) + "\n"
            )
            raise NotEstimable(reason)
        raise subprocess.CalledProcessError(returncode, command)
    if not (run_dir / "training_complete.json").is_file():
        raise RuntimeError(f"Training exited without completion marker: {run_dir}")
    from bonsai.functional.checkpointing import verified_training_selection
    verified_training_selection(run_dir)


def audit_encoder(
    config,
    origin,
    *,
    expected_width=256,
    minimum_local_epochs=10,
    minimum_dapt_epochs=10,
    allow_milestone=False,
):
    """Strict CPU reconstruction plus completion/provenance gates.

    ``allow_milestone`` (width probe only) accepts a named milestone checkpoint
    (``epoch_010.ckpt``) of a run that is still training: its completed epochs
    come from the checkpoint itself and must match its name. Paper runs always
    require the completed run's metadata sidecar.
    """
    import torch
    from bonsai.functional.checkpointing import (
        extract_encoder_state_dict,
        get_saved_encoder_config,
    )
    from opera.compat.bonsai import build_bonsai_encoder

    checkpoint_path = Path(config["dapt_ckpt"])
    vocabulary_path = Path(config["paths"]["vocabulary"])
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    architecture = get_saved_encoder_config(checkpoint["hyper_parameters"])
    if architecture["hidden_size"] != expected_width:
        raise ValueError("Paper run refuses an encoder with the wrong hidden width")
    # RI takes its architecture and vocabulary from the local encoder. Local
    # encoders: 4 layers, 64-dimensional heads (4 at width 256, 8 at 512).
    if origin in {"L", "RI", "FZ"}:
        expected = {
            "num_layers": 4,
            "num_attention_heads": expected_width // 64,
            "abspos_encoding": "scaled_time2vec",
        }
        if any(architecture.get(key) != value for key, value in expected.items()):
            raise ValueError(
                "Local encoder differs from the confirmed collaborator architecture"
            )
    vocabulary = torch.load(vocabulary_path, map_location="cpu", weights_only=False)
    if architecture["vocab_size"] != len(vocabulary):
        raise ValueError("Encoder and local vocabulary sizes differ")
    metadata = checkpoint["hyper_parameters"].get("checkpoint_metadata", {})
    sidecar_path = checkpoint_path.parent / "checkpoint_metadata.json"
    milestone = re.fullmatch(r"epoch_(\d+)\.ckpt", checkpoint_path.name)
    if allow_milestone and milestone and origin == "L":
        completed = int(checkpoint.get("epoch", -1)) + 1
        if completed != int(milestone.group(1)):
            raise ValueError(
                f"Milestone checkpoint {checkpoint_path.name} records {completed} completed epochs"
            )
        if completed < minimum_local_epochs:
            raise ValueError(
                f"Encoder fit is not complete enough: {completed} epochs, require {minimum_local_epochs}"
            )
    elif origin in {"L", "RI", "FZ", "EA"}:
        if not sidecar_path.is_file():
            raise ValueError(
                "Wait for completed pretraining/DAPT and its metadata sidecar"
            )
        sidecar = json.loads(sidecar_path.read_text())
        completed = sidecar.get(
            "completed_epochs", sidecar.get("extra", {}).get("completed_epochs", 0)
        )
        minimum = minimum_local_epochs if origin in {"L", "RI", "FZ"} else minimum_dapt_epochs
        if int(completed) < minimum:
            raise ValueError(
                f"Encoder fit is not complete enough: {completed} epochs, require {minimum}"
            )
    if origin == "E" and metadata.get("training_stage") != "external_encoder_export":
        raise ValueError(
            "Direct external arm requires the audited vocabulary-remapped encoder export"
        )
    if (
        origin == "EA"
        and metadata.get("training_stage") != "domain_adaptive_pretraining"
    ):
        raise ValueError("EA requires an actual external-initialized DAPT checkpoint")
    if origin in {"E", "EA"}:
        transfer = metadata.get("source_pretraining_transfer") or {}
        vocabulary_hash = file_hash(vocabulary_path)
        if transfer.get("target_vocabulary_sha256") != vocabulary_hash:
            raise ValueError(
                "External arm vocabulary differs from the audited transfer vocabulary"
            )
    if origin == "EA":
        direct_path = config.get("paper_run", {}).get("direct_external_checkpoint")
        if not direct_path or transfer.get("checkpoint_sha256") != file_hash(
            direct_path
        ):
            raise ValueError(
                "DAPT did not initialize from this matrix's exact direct external export"
            )
    encoder = build_bonsai_encoder(architecture, attn_type="sdpa")
    encoder.load_state_dict(
        extract_encoder_state_dict(checkpoint["state_dict"]), strict=True
    )
    encoder.eval()
    with torch.no_grad():
        states = encoder.encode(
            {
                "code": torch.tensor([[vocabulary["[CLS]"], vocabulary["[SEP]"]]]),
                "age": torch.zeros(1, 2),
                "abspos": torch.zeros(1, 2),
                "segment": torch.zeros(1, 2, dtype=torch.long),
                "attention_mask": torch.ones(1, 2, dtype=torch.bool),
            }
        )
    if not torch.isfinite(states).all():
        raise ValueError("Encoder forward audit produced nonfinite states")
    return {
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": file_hash(checkpoint_path),
        "vocabulary_sha256": file_hash(vocabulary_path),
        "architecture": architecture,
        "origin": origin,
        "status": "strict_load_and_forward_passed",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle-dir", type=Path, required=True)
    parser.add_argument(
        "--phase",
        choices=[
            "supervision",
            "source_screen",
            "rank_screen",
            "query_screen",
            "final_sources",
            "external",
            "pretraining_control",
            "router_controls",
            "data_efficiency",
        ],
        required=True,
    )
    parser.add_argument("--decisions", type=Path)
    parser.add_argument("--regime", choices=["S", "D", "O", "J"])
    parser.add_argument("--runs-dir", type=Path, required=True)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--max-runs", type=int)
    parser.add_argument("--minimum-local-pretraining-epochs", type=int, default=10)
    parser.add_argument("--minimum-dapt-epochs", type=int, default=10)
    parser.add_argument("--external-hidden-size", type=int, default=256)
    parser.add_argument("--local-hidden-size", type=int, default=256)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    bundle = json.loads((args.bundle_dir / "bundle.json").read_text())
    decisions = json.loads(args.decisions.read_text()) if args.decisions else None
    rows = select_runs(
        bundle,
        args.phase,
        decisions,
        regime=args.regime,
        shard_count=args.shard_count,
        shard_index=args.shard_index,
    )
    if args.max_runs is not None:
        if args.max_runs < 1:
            raise ValueError("max-runs must be positive")
        rows = rows[: args.max_runs]
    if not args.execute:
        print(json.dumps({"mode": "dry_run_no_training", "runs": rows}, indent=2))
        return
    from dotenv import load_dotenv

    load_dotenv()
    args.runs_dir.mkdir(parents=True, exist_ok=True)
    audited = {}
    for row in rows:
        path, _, resolved = load_run_config(row, args.bundle_dir)
        key = encoder_key(resolved, row)
        if key not in audited:
            audited[key] = audit_encoder(
                resolved,
                row["origin"],
                expected_width=(args.local_hidden_size if row["origin"] in {"L", "RI", "FZ"}
                                else args.external_hidden_size),
                minimum_local_epochs=args.minimum_local_pretraining_epochs,
                minimum_dapt_epochs=args.minimum_dapt_epochs,
            )
        run_dir = args.runs_dir / row["run_directory"]
        run_dir.mkdir(parents=True, exist_ok=True)
        from opera.run.paper_worker import Claim
        from bonsai.functional.checkpointing import verified_training_selection
        claim = Claim(run_dir, "fixed-queue", heartbeat_seconds=60)
        if not claim.acquire():
            print(f"Already claimed elsewhere: {run_dir}", flush=True)
            continue
        try:
            if (run_dir / "training_complete.json").is_file():
                verified_training_selection(run_dir)
            elif (run_dir / NOT_ESTIMABLE).is_file():
                print(f"Not estimable: {row['name']}", flush=True)
                continue
            else:
                train_run(row, path, run_dir, audited[key])
        except NotEstimable as reason:
            print(f"Not estimable {row['name']}: {reason}", flush=True)
            continue
        finally:
            claim.release()
        print(f"Completed/reused {row['name']}: {run_dir}", flush=True)

if __name__ == "__main__":
    main()
