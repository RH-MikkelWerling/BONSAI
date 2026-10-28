"""Is the national checkpoint usable as it is? Run before external-data and DAPT.

Reads only the checkpoint's hyperparameters, its code embedding and the
vocabulary (CPU, seconds), and checks what would otherwise fail late or not at
all:

- the vocabulary file is valid (contiguous ids, special tokens) and has as many
  tokens as the checkpoint's code embedding has rows (a vocabulary of the right
  size but in another order cannot be detected here: compare its sha256 with
  the file saved by the national run);
- the encoder is causal and has the width the paper fits use;
- the value representation the data config produces matches the encoder
  (joined bins: no value-bin input; separate bins: a value-bin vocabulary),
  which would otherwise leave the value input silently empty;
- which settings were filled in by defaults rather than recorded (time
  encoding, RoPE layout), so they can be confirmed with the national run.

    python -m opera.run.check_external_checkpoint --checkpoint <best.ckpt> \\
        --vocabulary <vocabulary.pt> --data-config daly_care_joined_bins_data
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
import yaml

from bonsai.functional.model_config import has_explicit_rope_layout
from bonsai.functional.pretrain_transfer import (
    _CODE_WEIGHT,
    _raw_source_config,
    _vocabulary,
    resolve_source_encoder_config,
)

REPO = Path(__file__).resolve().parents[2]


def numeric_value_mode(data_config: str) -> str | None:
    """The data config's numeric_value_mode, following its defaults list."""
    path = REPO / "configs" / f"{data_config}.yaml"
    if not path.is_file():
        return None
    config = yaml.safe_load(path.read_text()) or {}
    if "numeric_value_mode" in config:
        return config["numeric_value_mode"]
    for default in config.get("defaults", []):
        name = default if isinstance(default, str) else next(iter(default), "")
        name = str(name).split("@")[0].strip()
        if name and name != "_self_":
            found = REPO / "configs" / f"{name}.yaml"
            if found.is_file():
                mode = (yaml.safe_load(found.read_text()) or {}).get("numeric_value_mode")
                if mode:
                    return mode
    return None


def check(checkpoint_path, vocabulary_path, data_config, *, width=256, rope_layout=None):
    problems, notes = [], []
    vocabulary = _vocabulary(Path(vocabulary_path))
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict) or not {"state_dict", "hyper_parameters"} <= set(checkpoint):
        raise SystemExit("Not a BONSAI Lightning pretraining checkpoint "
                         "(needs state_dict and hyper_parameters)")
    raw = _raw_source_config(checkpoint["hyper_parameters"])
    config = resolve_source_encoder_config(checkpoint["hyper_parameters"], rope_layout)
    embedding = next((v for k, v in checkpoint["state_dict"].items()
                      if k.endswith(_CODE_WEIGHT)), None)

    if embedding is None:
        problems.append("no code embedding in the checkpoint's state_dict")
    elif embedding.shape[0] != len(vocabulary):
        problems.append(f"code embedding has {embedding.shape[0]} rows but the vocabulary "
                        f"{len(vocabulary)} tokens: wrong vocabulary file")
    if config["vocab_size"] != len(vocabulary):
        problems.append(f"checkpoint vocab_size {config['vocab_size']} differs from the "
                        f"vocabulary ({len(vocabulary)})")
    if config["hidden_size"] != width:
        problems.append(f"hidden size {config['hidden_size']}, but the paper fits use "
                        f"{width} (OPERA_PAPER_EXTERNAL_WIDTH)")
    if not config.get("causal"):
        problems.append("encoder is not causal; DAPT and the paper fits need a causal encoder")

    mode = numeric_value_mode(data_config)
    bins = int(config.get("value_bin_vocab_size") or 0)
    if mode is None:
        notes.append(f"numeric_value_mode of data config {data_config!r} not found; "
                     "confirm it matches the national tokenisation")
    elif mode == "joined_binning" and bins:
        problems.append(f"data config {data_config!r} joins value bins into codes, but the "
                        f"encoder expects a value-bin input ({bins} bins): use the national "
                        "run's separate-bins data config")
    elif mode != "joined_binning" and not bins:
        problems.append(f"data config {data_config!r} uses {mode!r}, but the encoder has no "
                        "value-bin input: values would be dropped")
    if "num_attention_heads" not in raw:
        notes.append(f"attention head count not recorded; inferred {config['num_attention_heads']} "
                     "from original BONSAI's fixed head_dim=64; confirm the training implementation")
    if "abspos_encoding" not in raw:
        notes.append(f"time encoding not recorded; assumed {config['abspos_encoding']!r} "
                     "(confirm with the national run)")
    if not has_explicit_rope_layout(raw) and rope_layout is None:
        notes.append("RoPE layout not recorded; current convention assumed "
                     "(set OPERA_PAPER_EXTERNAL_SOURCE_ROPE_LAYOUT if the run predates a3fa6ac)")

    report = {
        "checkpoint": str(checkpoint_path),
        "vocabulary": str(vocabulary_path),
        "vocabulary_sha256": hashlib.sha256(Path(vocabulary_path).read_bytes()).hexdigest(),
        "vocabulary_size": len(vocabulary),
        "encoder": {k: config.get(k) for k in (
            "hidden_size", "num_layers", "num_attention_heads", "max_seqlen", "causal",
            "abspos_encoding", "value_embedding_mode", "value_bin_vocab_size",
            "rope_interleaved")},
        "data_config": data_config, "numeric_value_mode": mode,
        "problems": problems, "notes": notes,
    }
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--vocabulary", type=Path, required=True)
    parser.add_argument("--data-config", default="daly_care_joined_bins_data")
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--source-rope-layout")
    args = parser.parse_args()
    report = check(args.checkpoint, args.vocabulary, args.data_config, width=args.width,
                   rope_layout=args.source_rope_layout)
    print(json.dumps(report, indent=2, default=str))
    if report["problems"]:
        raise SystemExit("National checkpoint not ready:\n  " + "\n  ".join(report["problems"]))


if __name__ == "__main__":
    main()
