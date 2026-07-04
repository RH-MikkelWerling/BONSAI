"""
OPERA Joint Multi-Task Finetuning.

Trains a single shared model on ALL patients across ALL cohorts and ALL
outcomes simultaneously.  This is the experimental condition that most
directly tests the foundation model's advantage over tabular approaches:

    Tabular (XGBoost, LR) with disease dummies:
        - Needs explicit disease-indicator features
        - Feature vectors are sparse across diseases
        - Cannot share representations across cohorts — just shares a
          decision boundary in a hand-engineered feature space
        - Adding a new cohort requires retraining with new columns

    Foundation model (joint finetune):
        - Same token space for all diseases
        - Encoder learns what is shared (e.g. LDH trajectory, AKI
          mechanism) vs disease-specific (e.g. DLBCL response patterns)
        - Adding a new cohort is just more data
        - Per-outcome heads are tiny — the encoder does the heavy lifting

Evaluation
──────────
After training, per-cohort × per-outcome evaluation is run by the
standard sweep.py — this script only trains the shared model and saves
the checkpoint.  The sweep config should include a "joint_finetuning"
variant pointing at this checkpoint with encoder_source="joint".

Usage
─────
python -m opera.run.joint_finetune \\
    encoder_ckpt=/path/to/contrastive/best.ckpt \\
    encoder_source=contrastive
"""

import torch
import hydra
import lightning as L
from dotenv import load_dotenv
from omegaconf import DictConfig, OmegaConf
from transformers import ModernBertConfig
from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping

from bonsai.functional.pathing import get_experiment_output_path
from bonsai.functional.checkpointing import (
    get_saved_encoder_config,
    save_checkpoint_metadata_sidecar,
)
from opera.compat.bonsai import BonsaiEncoder

from opera.modules.datamodules.MultiCohortContrastiveDataModule import (
    MultiCohortContrastiveDataModule,
    compute_pooled_class_counts,
)
from opera.modules.networks.joint_finetune_net import JointFinetuneModel
from opera.modules.lightningmodules.JointFinetuneModule import JointFinetuneModule
from opera.run.finetune import load_encoder_state_dict

load_dotenv()


def _cross_outcome_config(cfg: DictConfig) -> dict:
    """Resolve joint loss weighting config and auto-fill train class counts."""
    settings = OmegaConf.to_container(
        cfg.get("cross_outcome", {}) or {},
        resolve=True,
    )
    settings = dict(settings or {})
    batch_sampling = cfg.get("training", {}).get("batch_sampling", {}) or {}
    sampler_type = str(batch_sampling.get("type", "event_aware")).lower()
    if sampler_type in {"event_aware", "survival_event_aware"} and settings.get(
        "positive_class_weighted", False
    ):
        raise ValueError(
            "Joint finetuning cannot combine event-aware sampling with positive "
            "class weighting. Choose one imbalance correction so rare events "
            "are not amplified twice."
        )
    needs_counts = bool(settings.get("class_balanced", False)) or bool(
        settings.get("positive_class_weighted", False)
    )
    if needs_counts and not settings.get("class_counts"):
        settings["class_counts"] = compute_pooled_class_counts(
            cohort_configs={
                name: {
                    "data_dir": c["data_dir"],
                    "registry_start_date": c.get("registry_start_date"),
                }
                for name, c in cfg.cohorts.items()
            },
            outcome_configs=OmegaConf.to_container(cfg.outcomes, resolve=True),
            split="train",
            require_min_followup=True,
            require_all_configured_cells=cfg.training.get(
                "require_all_configured_cells", True
            ),
        )
    return settings


@hydra.main(
    config_path="../configs",
    config_name="joint_finetune",
    version_base="1.2",
)
def main(cfg: DictConfig) -> None:
    logger = CSVLogger(get_experiment_output_path(), name="joint_finetune_runs")
    model_save_dir = logger.log_dir

    # ── Load encoder ───────────────────────────────────────────────────
    encoder_state, pretrain_hparams = load_encoder_state_dict(
        cfg.encoder_ckpt, cfg.encoder_source
    )
    vocab = torch.load(cfg.paths.vocabulary, weights_only=False)

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
    missing, unexpected = encoder.load_state_dict(encoder_state, strict=True)

    outcome_names = sorted(cfg.outcomes.keys())
    cross_outcome_config = _cross_outcome_config(cfg)

    joint_model = JointFinetuneModel(
        encoder=encoder,
        outcome_names=outcome_names,
        hidden_size=model_cfg.get("hidden_size", 768),
        pooling=cfg.model.get("pooling", "bigru"),
        freeze_encoder=cfg.model.get("freeze_encoder", False),
        dropout=cfg.model.get("dropout", 0.1),
        cross_outcome_config=cross_outcome_config,
    )

    lightning_module = JointFinetuneModule(
        model=joint_model,
        outcome_names=outcome_names,
        learning_rate=cfg.training.learning_rate,
        encoder_lr_multiplier=cfg.training.get("encoder_lr_multiplier", 0.1),
        optimizer_epsilon=cfg.training.optimizer_epsilon,
        scheduler_warmup_epochs=cfg.training.scheduler_warmup_epochs,
        checkpoint_metadata={
            "training_stage": "joint_finetuning",
            "source_checkpoint": cfg.encoder_ckpt,
            "encoder_source": cfg.encoder_source,
            "cohort_set": sorted(cfg.cohorts.keys()),
            "outcome_set": outcome_names,
            "cross_outcome": cross_outcome_config,
        },
    )

    # ── Data — reuse MultiCohortContrastiveDataModule ──────────────────
    # It already loads all cohorts + binary outcome labels, which is
    # exactly what joint finetuning needs.  Survival fields (time_days,
    # event) are loaded too but simply ignored by JointFinetuneModel.
    data_module = MultiCohortContrastiveDataModule(
        cohort_configs={
            name: {
                "data_dir": c["data_dir"],
                "registry_start_date": c.get("registry_start_date"),
            }
            for name, c in cfg.cohorts.items()
        },
        outcome_configs=OmegaConf.to_container(cfg.outcomes, resolve=True),
        predict_token_id=vocab["[CLS]"],
        batch_size=cfg.training.batch_size,
        num_workers=cfg.hardware.num_workers,
        require_min_followup_train=True,
        require_min_followup_val=True,
        require_all_configured_cells=cfg.training.get(
            "require_all_configured_cells", True
        ),
        eligibility_scope="final",
        max_len=encoder.config.max_position_embeddings,
        batch_sampling=cfg.training.get("batch_sampling", {}),
    )

    # ── Callbacks ──────────────────────────────────────────────────────
    callbacks = [
        ModelCheckpoint(
            dirpath=model_save_dir,
            monitor="val/auroc_macro",
            mode="max",
            save_top_k=1,
            filename="best",
            enable_version_counter=False,
            save_last=True,
        ),
        EarlyStopping(
            monitor="val/auroc_macro",
            patience=cfg.training.get("early_stopping_patience", 5),
            mode="max",
        ),
    ]

    trainer = L.Trainer(
        accelerator=cfg.hardware.accelerator,
        devices=cfg.hardware.num_devices,
        accumulate_grad_batches=cfg.training.accumulate_grad_batches,
        max_epochs=cfg.training.epochs,
        callbacks=callbacks,
        logger=[logger],
        num_nodes=cfg.hardware.num_nodes,
        precision=cfg.hardware.precision,
        limit_val_batches=cfg.training.limit_val_batches,
        limit_train_batches=cfg.training.limit_train_batches,
    )

    resume_ckpt = cfg.paths.get("resume_ckpt") or None
    trainer.fit(model=lightning_module, datamodule=data_module, ckpt_path=resume_ckpt)
    save_checkpoint_metadata_sidecar(model_save_dir, lightning_module)
    print(f"\nJoint finetune complete. Checkpoint: {model_save_dir}/best.ckpt")


if __name__ == "__main__":
    main()
