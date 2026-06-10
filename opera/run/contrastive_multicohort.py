"""
OPERA multi-cohort contrastive learning stage.

Identical to contrastive.py but uses MultiCohortContrastiveDataModule
to train across all disease cohorts simultaneously.  The DAPT-prior
cohort similarity weights handle soft cross-disease pair weighting.

Usage:
    python -m opera.run.contrastive_multicohort \\
        dapt_ckpt=/path/to/dapt/best.ckpt \\
        dapt_embedding_store=/path/to/dapt_embeddings.pt
"""

import hydra
import lightning as L
import torch
from dotenv import load_dotenv
from omegaconf import DictConfig, OmegaConf
from transformers import ModernBertConfig
from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch.callbacks import ModelCheckpoint

from bonsai.functional.pathing import get_experiment_output_path
from bonsai.functional.checkpointing import (
    get_saved_encoder_config,
    save_checkpoint_metadata_sidecar,
)
from opera.compat.bonsai import BonsaiEncoder
from opera.modules.networks.opera_nets import OperaContrastiveModel
from opera.modules.lightningmodules.OperaContrastiveModule import OperaContrastiveModule
from opera.modules.datamodules.MultiCohortContrastiveDataModule import (
    MultiCohortContrastiveDataModule,
    compute_pooled_event_time_probability_grids,
)

load_dotenv()


@hydra.main(
    config_path="../configs",
    config_name="contrastive_multicohort",
    version_base="1.2",
)
def main(cfg: DictConfig) -> None:
    logger = CSVLogger(
        get_experiment_output_path(), name="contrastive_multicohort_runs"
    )
    model_save_dir = logger.log_dir

    # ── Load encoder from DAPT checkpoint ─────────────────────────────
    ckpt = torch.load(cfg.dapt_ckpt, map_location="cpu", weights_only=False)
    pretrain_hparams = ckpt["hyper_parameters"]

    # Vocabulary: use any cohort's vocab (shared token space)
    first_cohort = next(iter(cfg.cohorts.values()))
    import os

    vocab_path = os.path.join(first_cohort["data_dir"], "vocabulary.pt")
    vocab = torch.load(vocab_path)

    model_cfg = get_saved_encoder_config(pretrain_hparams)
    for key in ("vocab_size", "pad_token_id", "cls_token_id", "sep_token_id"):
        model_cfg.pop(key, None)

    encoder = BonsaiEncoder(
        ModernBertConfig(
            **model_cfg,
            vocab_size=len(vocab),
            pad_token_id=0,
            cls_token_id=1,
            sep_token_id=2,
        )
    )

    state_dict = ckpt["state_dict"]
    encoder_state = {}
    for k, v in state_dict.items():
        if k.startswith("model."):
            clean = k[len("model.") :]
            if clean.startswith("head.") or clean.startswith("decoder."):
                continue
            encoder_state[clean] = v
    encoder.load_state_dict(encoder_state, strict=False)

    # ── DAPT-prior embedding store (optional) ─────────────────────────
    dapt_embedding_store = None
    if cfg.get("dapt_embedding_store") is not None:
        print(f"Loading DAPT embedding store from {cfg.dapt_embedding_store} ...")
        dapt_embedding_store = torch.load(cfg.dapt_embedding_store, map_location="cpu")
        print(f"  Loaded {len(dapt_embedding_store)} patient embeddings.")

    # ── Build model ────────────────────────────────────────────────────
    outcome_configs = OmegaConf.to_container(cfg.outcomes, resolve=True)
    outcome_names = sorted(outcome_configs.keys())

    cohort_configs = OmegaConf.to_container(cfg.cohorts, resolve=True)
    outcome_sorted_event_times, outcome_event_time_probs = (
        compute_pooled_event_time_probability_grids(
            cohort_configs,
            outcome_configs,
            split="train",
        )
    )
    print(
        f"KM event-time grids computed for {len(outcome_sorted_event_times)} outcomes."
    )
    for name, t in outcome_sorted_event_times.items():
        print(f"  {name}: {len(t)} pooled KM-weighted training event locations")

    model = OperaContrastiveModel(
        encoder=encoder,
        outcome_names=outcome_names,
        hidden_size=model_cfg["hidden_size"],
        projection_hidden_dim=cfg.model.projection_hidden_dim,
        projection_dim=cfg.model.projection_dim,
        temperature=cfg.model.temperature,
        outcome_sorted_event_times=outcome_sorted_event_times,
        outcome_event_time_probs=outcome_event_time_probs,
        dapt_lambda_floor=cfg.model.get("dapt_lambda_floor", 0.3),
        dapt_anchor_weight=cfg.model.get("dapt_anchor_weight", 0.0),
        competing_event_weight=cfg.model.get("competing_event_weight", 0.0),
        effective_pair_normalization=cfg.model.get(
            "effective_pair_normalization", True
        ),
        freeze_encoder=cfg.model.freeze_encoder,
        pooling=cfg.model.pooling,
        dapt_embedding_store=dapt_embedding_store,
    )

    # ── Data ───────────────────────────────────────────────────────────
    data_module = MultiCohortContrastiveDataModule(
        cohort_configs=cohort_configs,
        outcome_configs=outcome_configs,
        predict_token_id=vocab["[CLS]"],
        batch_size=cfg.training.batch_size,
        num_workers=cfg.hardware.num_workers,
    )

    # ── Lightning ──────────────────────────────────────────────────────
    lightning_module = OperaContrastiveModule(
        model=model,
        outcome_names=outcome_names,
        learning_rate=cfg.training.learning_rate,
        encoder_lr_multiplier=cfg.training.encoder_lr_multiplier,
        optimizer_epsilon=cfg.training.optimizer_epsilon,
        scheduler_warmup_epochs=cfg.training.scheduler_warmup_epochs,
        dapt_anchor_weight=cfg.model.get("dapt_anchor_weight", 0.0),
        checkpoint_metadata={
            "training_stage": "opera_contrastive_adaptation",
            "source_checkpoint": cfg.dapt_ckpt,
            "cohort_set": sorted(cfg.cohorts.keys()),
            "outcome_set": outcome_names,
        },
    )

    ckpt_callback = ModelCheckpoint(
        dirpath=model_save_dir,
        monitor="val/loss",  # sigma-weighted contrastive loss: consistent with training objective
        mode="min",
        save_top_k=1,
        filename="best",
        enable_version_counter=False,
        save_last=True,
    )

    trainer = L.Trainer(
        accelerator=cfg.hardware.accelerator,
        accumulate_grad_batches=cfg.training.accumulate_grad_batches,
        devices=cfg.hardware.num_devices,
        limit_val_batches=cfg.training.limit_val_batches,
        limit_train_batches=cfg.training.limit_train_batches,
        callbacks=[ckpt_callback],
        logger=[logger],
        max_epochs=cfg.training.epochs,
        num_nodes=cfg.hardware.num_nodes,
        precision=cfg.hardware.precision,
    )

    trainer.fit(model=lightning_module, datamodule=data_module)
    save_checkpoint_metadata_sidecar(model_save_dir, lightning_module)


if __name__ == "__main__":
    main()
