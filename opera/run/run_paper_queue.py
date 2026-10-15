"""Execute one disjoint paper queue inside an existing GPU allocation."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
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
        if any(decisions.get(key) != value for key, value in row["condition"].items()):
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


def train_run(row, path, run_dir, audit_record):
    """Preflight and train one bundle run in ``run_dir``; resumes its last.ckpt."""
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
    with (run_dir / "queue_training.log").open("a") as log:
        subprocess.run(
            [
                sys.executable,
                "-u",
                "-m",
                "opera.run.contrastive_multicohort",
                "--config-path",
                str(path.parent),
                "--config-name",
                path.stem,
                f"hydra.run.dir={run_dir.resolve()}",
            ],
            stdout=log,
            stderr=subprocess.STDOUT,
            check=True,
        )
    if not (run_dir / "training_complete.json").is_file():
        raise RuntimeError(f"Training exited without completion marker: {run_dir}")
    from bonsai.functional.checkpointing import verified_training_selection
    verified_training_selection(run_dir)


def audit_encoder(
    config,
    origin,
    *,
    expected_width=256,
    minimum_local_epochs=20,
    minimum_dapt_epochs=20,
):
    """Strict CPU reconstruction plus completion/provenance gates."""
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
    # RI takes its architecture and vocabulary from the local encoder.
    if origin in {"L", "RI"} and expected_width == 256:
        expected = {
            "num_layers": 4,
            "num_attention_heads": 4,
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
    if origin in {"L", "RI", "EA"}:
        if not sidecar_path.is_file():
            raise ValueError(
                "Wait for completed pretraining/DAPT and its metadata sidecar"
            )
        sidecar = json.loads(sidecar_path.read_text())
        completed = sidecar.get(
            "completed_epochs", sidecar.get("extra", {}).get("completed_epochs", 0)
        )
        minimum = minimum_local_epochs if origin in {"L", "RI"} else minimum_dapt_epochs
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
        ],
        required=True,
    )
    parser.add_argument("--decisions", type=Path)
    parser.add_argument("--regime", choices=["S", "D", "O", "J"])
    parser.add_argument("--runs-dir", type=Path, required=True)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--max-runs", type=int)
    parser.add_argument("--minimum-local-pretraining-epochs", type=int, default=20)
    parser.add_argument("--minimum-dapt-epochs", type=int, default=20)
    parser.add_argument("--external-hidden-size", type=int, default=256)
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
                expected_width=256 if row["origin"] in {"L", "RI"} else args.external_hidden_size,
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
            else:
                train_run(row, path, run_dir, audited[key])
        finally:
            claim.release()
        print(f"Completed/reused {row['name']}: {run_dir}", flush=True)

if __name__ == "__main__":
    main()
