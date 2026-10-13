"""Check that completed Fourier/Time2Vec pretraining runs form a matched pair."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml
from omegaconf import OmegaConf
import torch


def verify(fourier: Path, time2vec: Path, sequence_data: Path) -> dict:
    runs = {"fourier": fourier, "scaled_time2vec": time2vec}
    configs = {}
    metadata = {}
    for encoding, root in runs.items():
        best = root / "version_0" / "best.ckpt"
        sidecar = root / "version_0" / "checkpoint_metadata.json"
        resolved = root / "hydra" / "config.yaml"
        for path in (best, sidecar, resolved):
            if not path.is_file() or path.stat().st_size == 0:
                raise ValueError(f"Missing completed {encoding} artifact: {path}")
        record = json.loads(sidecar.read_text(encoding="utf-8"))
        model = dict(record.get("model_config", {}))
        if model.get("abspos_encoding") != encoding:
            raise ValueError(f"{encoding} checkpoint sidecar has wrong absolute-time encoding")
        if record.get("extra", {}).get("completed_epochs", 0) < 20:
            raise ValueError(f"{encoding} pretraining did not complete 20 epochs")
        metadata[encoding] = record
        configs[encoding] = yaml.safe_load(resolved.read_text(encoding="utf-8"))
    f_model = dict(metadata["fourier"]["model_config"])
    t_model = dict(metadata["scaled_time2vec"]["model_config"])
    f_model.pop("abspos_encoding")
    t_model.pop("abspos_encoding")
    if f_model != t_model:
        raise ValueError("Pretraining model configs differ beyond abspos_encoding")
    for key in ("dataset", "seed"):
        f = configs["fourier"].get(key)
        t = configs["scaled_time2vec"].get(key)
        if f != t:
            raise ValueError(f"Resolved pretraining configs differ in {key}")
    for key in ("dir", "train_split", "val_split", "vocab", "population"):
        f = configs["fourier"].get("paths", {}).get(key)
        t = configs["scaled_time2vec"].get("paths", {}).get(key)
        if f != t:
            raise ValueError(f"Pretraining data paths differ in {key}")
    operational = {"auto_resume", "checkpoint_milestone_epochs"}
    f_train = {key: value for key, value in configs["fourier"].get("training", {}).items()
               if key not in operational}
    t_train = {key: value for key, value in configs["scaled_time2vec"].get("training", {}).items()
               if key not in operational}
    if f_train != t_train:
        raise ValueError("Pretraining objective/window/optimizer settings differ")
    for encoding, config in configs.items():
        recorded_dir = Path(str(OmegaConf.select(OmegaConf.create(config), "paths.dir")))
        if recorded_dir.resolve() != sequence_data.resolve():
            raise ValueError(f"{encoding} was pretrained on {recorded_dir}, not {sequence_data}")
    for key in ("pretraining_target_policy", "pretraining_window_policy", "numeric_value_control"):
        f = metadata["fourier"].get("checkpoint_metadata", {}).get(key)
        t = metadata["scaled_time2vec"].get("checkpoint_metadata", {}).get(key)
        if f != t:
            raise ValueError(f"Pretraining checkpoint policies differ in {key}")
    for name in ("subject_data_train.pt", "subject_data_tuning.pt", "vocabulary.pt"):
        if not (sequence_data / name).is_file():
            raise ValueError(f"Missing final sequence-data file: {sequence_data / name}")
    vocabulary = torch.load(sequence_data / "vocabulary.pt", map_location="cpu", weights_only=False)
    if int(f_model["vocab_size"]) != len(vocabulary):
        raise ValueError("Pretraining checkpoint vocabulary size differs from final sequence data")
    return {"status": "matched_pretraining_pair", "epochs": 20,
            "fourier_checkpoint": str(fourier / "version_0" / "best.ckpt"),
            "time2vec_checkpoint": str(time2vec / "version_0" / "best.ckpt"),
            "sequence_data": str(sequence_data)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fourier-run", type=Path, required=True)
    parser.add_argument("--time2vec-run", type=Path, required=True)
    parser.add_argument("--sequence-data", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(verify(args.fourier_run, args.time2vec_run, args.sequence_data), indent=2))


if __name__ == "__main__":
    main()
