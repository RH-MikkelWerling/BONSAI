"""
OPERA Hybrid fine-tuning — EHR embeddings + RKKP tabular features.

This is the optional "bonus" experiment that tests whether concatenating
tabular quality registry variables onto the foundation model embeddings
improves performance.

Usage:
    python -m opera.run.hybrid \
        encoder_ckpt=/path/to/contrastive/best.ckpt \
        encoder_source=contrastive \
        dataset=hematology_cohort \
        outcome=treatment_failure \
        tabular.path=/path/to/rkkp_features.parquet
"""

import pandas as pd
import hydra
import lightning as L
import torch
from dotenv import load_dotenv
from omegaconf import DictConfig, OmegaConf
from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping

from bonsai.functional.pathing import get_experiment_output_path
from opera.compat.bonsai import build_bonsai_encoder
from bonsai.modules.lightningmodules.FinetuneModule import FinetuneModule
from bonsai.functional.outcomes import (
    save_binarized_split_summary,
    split_and_binarize_outcomes,
)
from bonsai.functional.loss import get_loss_weight
from bonsai.functional.sampling import get_sampler
from bonsai.functional.checkpointing import (
    get_saved_encoder_config,
    save_checkpoint_metadata_sidecar,
)
from opera.functional.outcomes import (
    attach_prediction_censor_abspos,
    filter_outcome_eligibility,
    filter_registry_eligible_outcomes,
)

from opera.run.finetune import load_encoder_state_dict
from opera.modules.networks.hybrid_net import HybridClassifier
from opera.modules.datamodules.HybridDataModule import HybridDataModule

load_dotenv()


@hydra.main(
    config_path="../configs",
    config_name="hybrid",
    version_base="1.2",
)
def main(cfg: DictConfig) -> None:
    logger = CSVLogger(get_experiment_output_path(), name="hybrid_runs")
    model_save_dir = logger.log_dir

    # ── Load encoder ─────────────────────────────────────────────────
    encoder_state, pretrain_hparams = load_encoder_state_dict(
        cfg.encoder_ckpt, cfg.encoder_source
    )
    vocab = torch.load(cfg.paths.vocabulary)

    model_cfg = get_saved_encoder_config(pretrain_hparams)
    encoder = build_bonsai_encoder(model_cfg, vocab_size=len(vocab))
    encoder.load_state_dict(encoder_state, strict=True)

    # ── Outcomes ─────────────────────────────────────────────────────
    outcomes = pd.read_parquet(cfg.paths.outcome)
    outcomes = filter_outcome_eligibility(
        outcomes,
        cfg.paths.get("eligibility"),
        cohort=cfg.dataset,
        outcome_name=cfg.outcome,
    )
    outcomes = attach_prediction_censor_abspos(outcomes)
    outcomes = filter_registry_eligible_outcomes(
        outcomes,
        cfg.labels.get("registry_start_date"),
        cohort=cfg.get("dataset"),
        outcome_name=cfg.get("outcome"),
    )

    competing_df = None
    competing_path = cfg.paths.get("competing_outcome")
    if competing_path:
        competing_df = pd.read_parquet(competing_path)

    train_outcomes, val_outcomes, test_outcomes = split_and_binarize_outcomes(
        outcomes,
        train_key=cfg.labels.train_key,
        val_key=cfg.labels.val_key,
        test_key=cfg.labels.test_key,
        n_hours_start_include=cfg.labels.n_hours_start_include,
        n_hours_end_include=cfg.labels.n_hours_end_include,
        require_min_followup_train=cfg.labels.get("require_min_followup_train", True),
        require_min_followup_val=cfg.labels.get("require_min_followup_val", True),
        require_min_followup_test=cfg.labels.get("require_min_followup_test", True),
        outcome_name=cfg.outcome,
        competing_event_df=competing_df,
    )
    save_binarized_split_summary(
        outcomes=outcomes,
        split_outputs={
            cfg.labels.train_key: train_outcomes,
            cfg.labels.val_key: val_outcomes,
            cfg.labels.test_key: test_outcomes,
        },
        require_min_followup_by_split={
            cfg.labels.train_key: cfg.labels.get("require_min_followup_train", True),
            cfg.labels.val_key: cfg.labels.get("require_min_followup_val", True),
            cfg.labels.test_key: cfg.labels.get("require_min_followup_test", True),
        },
        path=f"{model_save_dir}/label_split_summary.csv",
        n_hours_end_include=cfg.labels.n_hours_end_include,
        outcome_name=cfg.outcome,
    )
    train_labels = [v["label"] for v in train_outcomes.values()]

    # ── Tabular features ─────────────────────────────────────────────
    feature_columns = OmegaConf.to_container(cfg.tabular.feature_columns, resolve=True)

    data_module = HybridDataModule(
        path_train_data=cfg.paths.train_split,
        path_val_data=cfg.paths.val_split,
        path_population=cfg.paths.population,
        path_tabular=cfg.tabular.path,
        feature_columns=feature_columns,
        train_outcomes=train_outcomes,
        val_outcomes=val_outcomes,
        test_outcomes=test_outcomes,
        predict_token_id=vocab["[CLS]"],
        max_len=int(cfg.training.get("max_len") or model_cfg["max_seqlen"]),
        batch_size=cfg.training.batch_size,
        num_workers=cfg.hardware.num_workers,
        train_sampler=get_sampler(
            weight_fn=cfg.training.sampling_weight_fn, labels=train_labels
        ),
    )

    # ── Build hybrid model ───────────────────────────────────────────
    model = HybridClassifier(
        encoder=encoder,
        hidden_size=model_cfg["hidden_size"],
        tabular_dim=len(feature_columns),
        mlp_hidden_dims=OmegaConf.to_container(cfg.hybrid.mlp_hidden_dims),
        dropout=cfg.hybrid.dropout,
        freeze_encoder=cfg.hybrid.freeze_encoder,
        pooling=cfg.hybrid.pooling,
    )

    lightning_module = FinetuneModule(
        model=model,
        learning_rate=cfg.training.learning_rate,
        optimizer_epsilon=cfg.training.optimizer_epsilon,
        scheduler_warmup_epochs=cfg.training.scheduler_warmup_epochs,
        checkpoint_metadata={
            "training_stage": "hybrid_finetuning",
            "source_checkpoint": cfg.encoder_ckpt,
            "encoder_source": cfg.encoder_source,
            "dataset": cfg.dataset,
            "outcome": cfg.outcome,
            "split_identifier": (
                f"{cfg.labels.train_key}:{cfg.labels.val_key}:{cfg.labels.test_key}"
            ),
        },
        pos_weight=get_loss_weight(
            cfg.training.loss_weight_function,
            labels=train_labels,
        ),
    )

    callbacks = [
        ModelCheckpoint(
            dirpath=model_save_dir,
            monitor=cfg.training.eval_monitor_metric,
            mode="max",
            save_top_k=1,
            filename="best",
            enable_version_counter=False,
            save_last=True,
        ),
        EarlyStopping(
            monitor=cfg.training.eval_monitor_metric,
            patience=cfg.training.early_stopping_patience,
            mode="max",
        ),
    ]

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
