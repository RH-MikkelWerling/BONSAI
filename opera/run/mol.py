"""
Multi-Outcome Learning (MOL) training.

This is the direct-prediction counterpart to OPERA contrastive learning.
Uses the same data pipeline (ContrastiveDataModule) and checkpoint loading,
but replaces the SupCon objective with per-outcome BCE + Kendall weighting.

The encoder from this stage can be used for downstream finetuning via
opera.run.finetune with encoder_source="mol".

Usage:
    python -m opera.run.mol \
        dapt_ckpt=/path/to/dapt/best.ckpt \
        dataset=hematology_cohort
"""

import hydra
import lightning as L
import torch
from dotenv import load_dotenv
from omegaconf import DictConfig, OmegaConf
from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping

from bonsai.functional.pathing import get_experiment_output_path
from bonsai.functional.checkpointing import (
    extract_encoder_state_dict,
    get_saved_encoder_config,
    save_checkpoint_metadata_sidecar,
)
from opera.compat.bonsai import build_bonsai_encoder, encoder_hparams
from opera.modules.networks.mol_net import MultiOutcomeModel
from opera.modules.lightningmodules.MOLModule import MOLModule
from opera.modules.datamodules.ContrastiveDataModule import ContrastiveDataModule

load_dotenv()


@hydra.main(
    config_path="../configs",
    config_name="generated/multi_outcome_full_panel",
    version_base="1.2",
)
def main(cfg: DictConfig) -> None:
    logger = CSVLogger(get_experiment_output_path(), name="mol_runs")
    model_save_dir = logger.log_dir

    # ── Load encoder from DAPT/pretrain checkpoint ───────────────────
    ckpt = torch.load(cfg.dapt_ckpt, map_location="cpu", weights_only=False)
    pretrain_hparams = ckpt["hyper_parameters"]
    vocab = torch.load(cfg.paths.vocabulary)

    model_cfg = get_saved_encoder_config(pretrain_hparams)
    encoder = build_bonsai_encoder(model_cfg, vocab_size=len(vocab))

    # Load encoder weights (same logic as contrastive.py)
    encoder_state = extract_encoder_state_dict(ckpt["state_dict"])
    encoder.load_state_dict(encoder_state, strict=True)

    # ── Outcome configuration ────────────────────────────────────────
    outcome_configs = OmegaConf.to_container(cfg.outcomes, resolve=True)
    outcome_names = sorted(outcome_configs.keys())

    # ── Build MOL model ──────────────────────────────────────────────
    fixed_weights = None
    if cfg.model.weighting == "fixed" and cfg.model.get("fixed_weights"):
        fixed_weights = OmegaConf.to_container(cfg.model.fixed_weights, resolve=True)

    model = MultiOutcomeModel(
        encoder=encoder,
        outcome_names=outcome_names,
        hidden_size=model_cfg["hidden_size"],
        head_hidden_dim=cfg.model.head_hidden_dim,
        head_dropout=cfg.model.head_dropout,
        freeze_encoder=cfg.model.freeze_encoder,
        pooling=cfg.model.pooling,
        weighting=cfg.model.weighting,
        fixed_weights=fixed_weights,
    )

    # ── Data (reuses ContrastiveDataModule — same format) ────────────
    data_module = ContrastiveDataModule(
        path_train_data=cfg.paths.train_split,
        path_val_data=cfg.paths.val_split,
        path_population=cfg.paths.population,
        outcome_configs=outcome_configs,
        predict_token_id=vocab["[CLS]"],
        batch_size=cfg.training.batch_size,
        num_workers=cfg.hardware.num_workers,
        require_min_followup_train=True,
        require_min_followup_val=True,
        max_len=encoder_hparams(encoder)["max_seqlen"],
    )

    # ── Lightning module ─────────────────────────────────────────────
    lightning_module = MOLModule(
        model=model,
        outcome_names=outcome_names,
        learning_rate=cfg.training.learning_rate,
        encoder_lr_multiplier=cfg.training.encoder_lr_multiplier,
        optimizer_epsilon=cfg.training.optimizer_epsilon,
        scheduler_warmup_epochs=cfg.training.scheduler_warmup_epochs,
        checkpoint_metadata={
            "training_stage": "non_contrastive_hematology_adaptation",
            "source_checkpoint": cfg.dapt_ckpt,
            "dataset": cfg.dataset,
            "outcome_set": outcome_names,
        },
    )

    # ── Callbacks ────────────────────────────────────────────────────
    callbacks = [
        ModelCheckpoint(
            dirpath=model_save_dir,
            monitor="val/loss",
            mode="min",
            save_top_k=1,
            filename="best",
            enable_version_counter=False,
            save_last=True,
        ),
    ]
    if cfg.training.get("early_stopping_patience"):
        callbacks.append(
            EarlyStopping(
                monitor="val/loss",
                patience=cfg.training.early_stopping_patience,
                mode="min",
            )
        )

    trainer = L.Trainer(
        accelerator=cfg.hardware.accelerator,
        accumulate_grad_batches=cfg.training.accumulate_grad_batches,
        devices=cfg.hardware.num_devices,
        limit_val_batches=cfg.training.limit_val_batches,
        limit_train_batches=cfg.training.limit_train_batches,
        callbacks=callbacks,
        logger=[logger],
        max_epochs=cfg.training.epochs,
        num_nodes=cfg.hardware.num_nodes,
        precision=cfg.hardware.precision,
    )

    trainer.fit(model=lightning_module, datamodule=data_module)
    save_checkpoint_metadata_sidecar(model_save_dir, lightning_module)


if __name__ == "__main__":
    main()
