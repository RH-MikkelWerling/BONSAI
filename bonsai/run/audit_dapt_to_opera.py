"""Fail-fast CPU audit of a local-vocabulary DAPT checkpoint for OPERA."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

from bonsai.functional.checkpointing import extract_encoder_state_dict, get_saved_encoder_config
from bonsai.functional.pretrain_transfer import _vocabulary
from opera.compat.bonsai import build_bonsai_encoder


def audit_dapt_to_opera(checkpoint_path: Path, vocabulary_path: Path) -> dict:
    checkpoint_path = Path(checkpoint_path)
    vocabulary_path = Path(vocabulary_path)
    vocabulary = _vocabulary(vocabulary_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    hparams = checkpoint["hyper_parameters"]
    metadata = hparams.get("checkpoint_metadata", {})
    if metadata.get("training_stage") != "domain_adaptive_pretraining":
        raise ValueError("Checkpoint is not marked as external-source domain adaptation")
    transfer = metadata.get("source_pretraining_transfer") or {}
    digest = hashlib.sha256(vocabulary_path.read_bytes()).hexdigest()
    if transfer.get("target_vocabulary_sha256") != digest:
        raise ValueError("Local vocabulary differs from the DAPT initialization vocabulary")
    config = get_saved_encoder_config(hparams)
    if config["vocab_size"] != len(vocabulary):
        raise ValueError("DAPT encoder vocabulary size differs from local vocabulary")
    encoder = build_bonsai_encoder(config, vocab_size=len(vocabulary))
    encoder.load_state_dict(extract_encoder_state_dict(checkpoint["state_dict"]), strict=True)
    encoder.eval()
    with torch.no_grad():
        states = encoder.encode({
            "code": torch.tensor([[vocabulary["[CLS]"], vocabulary["[SEP]"]]]),
            "age": torch.zeros((1, 2)),
            "abspos": torch.zeros((1, 2)),
            "segment": torch.zeros((1, 2), dtype=torch.long),
            "attention_mask": torch.ones((1, 2), dtype=torch.bool),
        })
    if states.shape != (1, 2, config["hidden_size"]) or not torch.isfinite(states).all():
        raise ValueError("OPERA encoder forward smoke test failed")
    return {
        "status": "opera_strict_load_and_forward_passed",
        "checkpoint": str(checkpoint_path.resolve()),
        "vocabulary": str(vocabulary_path.resolve()),
        "vocabulary_sha256": digest,
        "source_checkpoint_sha256": transfer.get("checkpoint_sha256"),
        "model_config": config,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--vocabulary", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(audit_dapt_to_opera(args.checkpoint, args.vocabulary), indent=2))


if __name__ == "__main__":
    main()
