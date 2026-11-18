"""Collect the reproducibility facts the OPERA paper still lacks (run on the analysis server).

Read-only. Writes one JSON file with:
  * environment: Python and key package versions, torch/CUDA, GPU (nvidia-smi), R sessionInfo();
  * training settings: every hparams.yaml / checkpoint_metadata.json / bundle.json under --roots,
    reduced to the fields the paper reports (learning rate, encoder multiplier, epochs, batch, warm-up,
    freeze/initialisation, architecture sizes), keyed by path;
  * parameter counts: for each --checkpoint, the number of parameters by top-level module prefix.
No patient data are read or written: only configuration files and model weights' shapes.

    python scripts/collect_reproducibility_facts.py \
        --roots data/paper_256_v3 data/models \
        --checkpoint <cohort encoder epoch_010.ckpt> --checkpoint <R_EA best.ckpt> --checkpoint <J_EA best.ckpt> \
        --output logs/reproducibility_facts_20261118.json
"""
from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
from importlib import metadata
from pathlib import Path

PACKAGES = ["torch", "lightning", "pytorch-lightning", "numpy", "pandas", "polars", "scikit-learn", "xgboost",
            "lifelines", "scikit-survival", "torchmetrics", "hydra-core", "umap-learn", "timeseriesflattener"]
KEYS = {"learning_rate", "lr", "encoder_lr_multiplier", "encoder_lr", "epochs", "max_epochs", "batch_size",
        "accumulate_grad_batches", "warmup_epochs", "warmup", "eps", "weight_decay", "freeze_encoder",
        "encoder_initialization", "early_stopping_patience", "eval_monitor_metric", "seed", "hidden_size",
        "num_hidden_layers", "num_attention_heads", "intermediate_size", "vocab_size", "max_position_embeddings",
        "abspos_encoding", "source_checkpoint", "global_step", "epoch", "train_patient_fraction", "precision",
        "gradient_cache", "optimizer", "scheduler"}


def run(cmd):
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=120).stdout.strip()
    except Exception as error:  # noqa: BLE001
        return f"unavailable: {error!r}"


def environment():
    versions = {}
    for name in PACKAGES:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    env = {"python": sys.version, "platform": platform.platform(), "packages": versions,
           "nvidia_smi": run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv"]),
           "R_sessionInfo": run(["Rscript", "-e", "sessionInfo()"])}
    try:
        import torch
        env["torch_cuda"] = torch.version.cuda
        env["cudnn"] = torch.backends.cudnn.version()
    except Exception as error:  # noqa: BLE001
        env["torch_cuda"] = f"unavailable: {error!r}"
    return env


def reduce(obj, out, prefix=""):
    if isinstance(obj, dict):
        for key, value in obj.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            if key in KEYS and not isinstance(value, (dict, list)):
                out[path] = value
            reduce(value, out, path)
    elif isinstance(obj, list):
        for i, value in enumerate(obj):
            reduce(value, out, f"{prefix}[{i}]")
    return out


def settings(roots):
    import yaml
    found = {}
    for root in roots:
        for path in Path(root).rglob("*"):
            if path.name not in {"hparams.yaml", "checkpoint_metadata.json", "bundle.json", "config.yaml"}:
                continue
            if path.name == "config.yaml" and ".hydra" not in path.parts and "hydra" not in path.parts:
                continue
            try:
                text = path.read_text()
                data = json.loads(text) if path.suffix == ".json" else yaml.safe_load(text)
            except Exception as error:  # noqa: BLE001
                found[str(path)] = {"error": repr(error)}
                continue
            reduced = reduce(data, {})
            if reduced:
                found[str(path)] = reduced
    return found


def parameters(checkpoints):
    import torch
    counts = {}
    for ckpt in checkpoints:
        state = torch.load(ckpt, map_location="cpu", weights_only=False)
        state = state.get("state_dict", state)
        by_prefix, total = {}, 0
        for name, tensor in state.items():
            if not hasattr(tensor, "numel"):
                continue
            parts = name.split(".")
            prefix = ".".join(parts[:2]) if parts[0] in {"model", "net", "module"} else parts[0]
            by_prefix[prefix] = by_prefix.get(prefix, 0) + tensor.numel()
            total += tensor.numel()
        counts[str(ckpt)] = {"total": total, "by_prefix": dict(sorted(by_prefix.items()))}
    return counts


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--roots", nargs="*", default=[])
    parser.add_argument("--checkpoint", action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"Refusing to overwrite {args.output}")
    result = {"environment": environment(), "settings": settings(args.roots),
              "parameters": parameters(args.checkpoint) if args.checkpoint else {}}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=1, default=str))
    print(f"wrote {args.output}: {len(result['settings'])} config files, {len(result['parameters'])} checkpoints")


if __name__ == "__main__":
    main()
