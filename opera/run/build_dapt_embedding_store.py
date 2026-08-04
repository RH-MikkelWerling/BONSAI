"""
Precompute a DAPT embedding store for the DAPT-prior contrastive mechanisms.

``MultiOutcomeSurvivalLoss`` supports two mechanisms — pairwise similarity
modulation (``dapt_lambda_floor``) and an anchor loss (``dapt_anchor_weight``)
— that both require a ``dapt_embedding_store`` dict of
``{subject_id: pre-projection DAPT embedding}``. Neither mechanism can do
anything without this file existing; this script builds it once from a DAPT
checkpoint, over the exact same pooled cohort population contrastive training
will use, and saves it as a ``.pt`` file.

Usage:
    python -m opera.run.build_dapt_embedding_store \\
        dapt_ckpt=/path/to/dapt/best.ckpt \\
        output_path=/path/to/dapt_embeddings.pt

Reuses the same cohort/outcome config structure as contrastive_multicohort.yaml
(default config name below) — override --config-name to point at a different
cohort set, e.g. contrastive.yaml for a single-cohort run.
"""

import hydra
import torch
from pathlib import Path
from dotenv import load_dotenv
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

from bonsai.functional.checkpointing import (
    extract_encoder_state_dict,
    get_saved_encoder_config,
)
from bonsai.functional.versioning import generate_unused_run_id
from opera.compat.bonsai import build_bonsai_encoder, encoder_hparams
from opera.functional.extract import build_dapt_embedding_store
from opera.modules.datamodules.ContrastiveDataModule import contrastive_collate
from opera.modules.datamodules.MultiCohortContrastiveDataModule import (
    MultiCohortContrastiveDataModule,
)
from opera.modules.networks.opera_nets import OperaContrastiveModel

load_dotenv()
OmegaConf.register_new_resolver(
    "version", lambda: generate_unused_run_id(), use_cache=True, replace=True
)


@hydra.main(
    config_path="../configs",
    config_name="generated/joint_opera_full_panel",
    version_base="1.2",
)
def main(cfg: DictConfig) -> None:
    output_path = cfg.get("output_path")
    if output_path is None:
        raise ValueError(
            "output_path=... is required (where to save the .pt embedding store)."
        )
    output_path = Path(output_path)
    if output_path.exists() and not bool(cfg.get("overwrite", False)):
        print(
            f"DAPT embedding store already exists at {output_path}; reusing it "
            "(set overwrite=true to rebuild)."
        )
        return
    device = (
        cfg.hardware.accelerator
        if cfg.hardware.accelerator != "auto"
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )

    # ── Load encoder from DAPT checkpoint (same path as contrastive*.py) ──
    ckpt = torch.load(cfg.dapt_ckpt, map_location="cpu", weights_only=False)
    pretrain_hparams = ckpt["hyper_parameters"]
    vocab = torch.load(cfg.paths.vocabulary, weights_only=False)
    model_cfg = get_saved_encoder_config(pretrain_hparams)
    encoder = build_bonsai_encoder(model_cfg, vocab_size=len(vocab))
    encoder_state = extract_encoder_state_dict(ckpt["state_dict"])
    encoder.load_state_dict(encoder_state, strict=True)

    outcome_configs = OmegaConf.to_container(cfg.outcomes, resolve=True)
    outcome_names = sorted(outcome_configs.keys())
    cohort_configs = OmegaConf.to_container(cfg.cohorts, resolve=True)

    # Only the encoder + pooling matter here (return_pre_projection=True
    # skips the projection head entirely), so the loss/outcome-time machinery
    # this constructor also builds is unused but harmless.
    model = OperaContrastiveModel(
        encoder=encoder,
        outcome_names=outcome_names,
        hidden_size=model_cfg["hidden_size"],
        pooling=cfg.model.get("pooling", "cls_last"),
    )

    # ── Pooled population, same cohort-loading path as training ──────────
    data_module = MultiCohortContrastiveDataModule(
        cohort_configs=cohort_configs,
        outcome_configs=outcome_configs,
        predict_token_id=vocab["[CLS]"],
        batch_size=cfg.training.batch_size,
        num_workers=cfg.hardware.num_workers,
        require_all_configured_cells=cfg.training.get(
            "require_all_configured_cells", True
        ),
        require_min_followup_train=cfg.training.get(
            "require_min_followup_train", False
        ),
        max_len=encoder_hparams(encoder)["max_seqlen"],
        batch_sampling={"type": "none"},
    )
    data_module.setup("fit")

    dataloader = DataLoader(
        data_module.train_dataset,
        batch_size=cfg.training.batch_size,
        shuffle=False,
        num_workers=cfg.hardware.num_workers,
        collate_fn=contrastive_collate,
    )

    build_dapt_embedding_store(
        model=model,
        dataloader=dataloader,
        device=device,
        save_path=output_path,
    )


if __name__ == "__main__":
    main()
