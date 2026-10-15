"""CPU audit of an external BONSAI checkpoint before local domain adaptation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from bonsai.functional.pretrain_transfer import (
    prepare_pretrain_transfer,
    source_pretrain_model_config,
)
from bonsai.modules.networks.bonsai_nets import BonsaiPretrain


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--source-vocab", type=Path, required=True)
    parser.add_argument("--target-vocab", type=Path, required=True)
    parser.add_argument("--config-name", default="pretrain_external_bonsai_dapt_v1")
    parser.add_argument("--config-dir", type=Path, default=Path(__file__).resolve().parents[2] / "configs")
    parser.add_argument("--override", action="append", default=[])
    parser.add_argument("--min-token-coverage", type=float, default=0.9)
    parser.add_argument("--output-json", type=Path)
    args = parser.parse_args()

    OmegaConf.register_new_resolver("version", lambda: 0, replace=True)
    with initialize_config_dir(config_dir=str(args.config_dir.resolve()), version_base="1.2"):
        config = compose(config_name=args.config_name, overrides=args.override)
    if config.paths.get("initialization_use_source_architecture", False):
        source_model_config = source_pretrain_model_config(
            args.checkpoint, config.paths.get("initialization_rope_layout")
        )
        for key in (
            "max_seqlen", "hidden_size", "num_layers", "num_attention_heads",
            "bias", "dropout", "attention_dropout", "causal",
            "value_bin_vocab_size", "value_embedding_mode", "abspos_encoding",
            "rope_interleaved",
        ):
            config.model[key] = source_model_config[key]
    vocabulary = torch.load(args.target_vocab, map_location="cpu", weights_only=False)
    model = BonsaiPretrain(
        vocab_size=len(vocabulary),
        max_seqlen=int(config.model.max_seqlen),
        hidden_size=int(config.model.hidden_size),
        num_layers=int(config.model.num_layers),
        num_attention_heads=int(config.model.num_attention_heads),
        bias=bool(config.model.bias),
        dropout=float(config.model.dropout),
        attention_dropout=float(config.model.attention_dropout),
        causal=bool(config.model.causal),
        attn_type=str(config.model.attn_type),
        value_bin_vocab_size=int(config.model.get("value_bin_vocab_size", 0)),
        value_embedding_mode=str(config.model.get("value_embedding_mode", "legacy")),
        abspos_encoding=str(config.model.get("abspos_encoding", "legacy")),
        rope_interleaved=config.model.get("rope_interleaved"),
    )
    report = prepare_pretrain_transfer(
        model, args.checkpoint, args.source_vocab, args.target_vocab,
        min_token_coverage=args.min_token_coverage, load_weights=True,
        source_rope_layout=config.paths.get("initialization_rope_layout"),
    )
    model.eval()
    with torch.no_grad():
        states = model.encode({
            "code": torch.tensor([[vocabulary["[CLS]"], vocabulary["[SEP]"]]]),
            "age": torch.zeros((1, 2), dtype=torch.float32),
            "abspos": torch.zeros((1, 2), dtype=torch.float32),
            "segment": torch.zeros((1, 2), dtype=torch.long),
            "attention_mask": torch.ones((1, 2), dtype=torch.bool),
        })
    if states.shape != (1, 2, model.hparams["hidden_size"]) or not bool(
        torch.isfinite(states).all()
    ):
        raise ValueError("Transferred encoder failed the two-token forward smoke test")
    report["forward_smoke"] = "passed"
    print(json.dumps(report, indent=2))
    if args.output_json is not None:
        if args.output_json.exists():
            raise FileExistsError(f"Audit output already exists: {args.output_json}")
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
