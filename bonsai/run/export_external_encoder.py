"""Export an audited external encoder in the local vocabulary, without DAPT.

This is an encoder-initialization artifact, not a trainer-resume checkpoint.
Source heads and optimizer state are excluded. Local-only token rows have a
recorded random seed; initialize local DAPT from this SAME export to hold the
remapping initialization fixed in direct-versus-DAPT comparisons.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from bonsai.functional.model_config import normalize_bonsai_model_config
from bonsai.functional.pretrain_transfer import (
    ROPE_LAYOUTS,
    _vocabulary,
    prepare_pretrain_transfer,
    source_pretrain_model_config,
)
from bonsai.modules.networks.bonsai_nets import BonsaiBase


def export_external_encoder(
    checkpoint,
    source_vocab,
    target_vocab,
    output_dir,
    *,
    seed=42,
    minimum_token_coverage=0.9,
    source_rope_layout=None,
):
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise FileExistsError("Use a fresh immutable external-export directory")
    vocabulary = _vocabulary(Path(target_vocab))
    # The export runs on SDPA; the source RoPE layout must be preserved
    # explicitly so the weights keep their trained position behaviour.
    source_config = source_pretrain_model_config(checkpoint, source_rope_layout)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        encoder = BonsaiBase(
            **normalize_bonsai_model_config(
                source_config, vocab_size=len(vocabulary), attn_type="sdpa"
            )
        )
    report = prepare_pretrain_transfer(
        encoder,
        checkpoint,
        source_vocab,
        target_vocab,
        min_token_coverage=minimum_token_coverage,
        load_weights=True,
        source_rope_layout=source_rope_layout,
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
        raise ValueError("Transferred encoder failed CPU forward audit")
    metadata = {
        "training_stage": "external_encoder_export",
        "model_class": "BonsaiBase",
        "source_pretraining_transfer": report,
        "local_token_initialization_seed": seed,
        "local_dapt_performed": False,
        "trainer_resume_supported": False,
    }
    payload = {
        "hyper_parameters": {
            "model_config": dict(encoder.hparams),
            "checkpoint_metadata": metadata,
        },
        "state_dict": {
            "model." + name: value for name, value in encoder.state_dict().items()
        },
    }
    output_dir.mkdir(parents=True)
    torch.save(payload, output_dir / "encoder.ckpt")
    torch.save(vocabulary, output_dir / "vocabulary.pt")
    (output_dir / "checkpoint_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )
    return {
        "checkpoint": str(output_dir / "encoder.ckpt"),
        "vocabulary": str(output_dir / "vocabulary.pt"),
        "status": "remapped_strict_encoder_export_forward_passed_no_dapt",
        **metadata,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--source-vocab", type=Path, required=True)
    parser.add_argument("--target-vocab", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--minimum-token-coverage", type=float, default=0.9)
    parser.add_argument(
        "--source-rope-layout",
        choices=sorted(ROPE_LAYOUTS),
        help="RoPE layout of the source training commit; required for a "
        "FlashAttention checkpoint that does not record it (upstream BONSAI "
        "is non_interleaved before 2026-09-29/a3fa6ac, interleaved after)",
    )
    args = parser.parse_args()
    print(
        json.dumps(
            export_external_encoder(
                args.checkpoint,
                args.source_vocab,
                args.target_vocab,
                args.output_dir,
                seed=args.seed,
                minimum_token_coverage=args.minimum_token_coverage,
                source_rope_layout=args.source_rope_layout,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
