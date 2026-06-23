"""OPERA survival and IPCW-BCE finetuning runner."""

import logging

import hydra
import lightning as L
import pandas as pd
import torch
from dotenv import load_dotenv
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger
from omegaconf import DictConfig
from transformers import ModernBertConfig

from bonsai.functional.checkpointing import (
    get_saved_encoder_config,
    save_checkpoint_metadata_sidecar,
)
from bonsai.functional.outcomes import (
    save_binarized_split_summary,
    split_and_binarize_outcomes,
)
from bonsai.functional.pathing import get_experiment_output_path
from opera.compat.bonsai import BonsaiFinetune
from opera.functional.ipcw import attach_ipcw_weights, compute_ipcw_train_weights
from opera.functional.outcomes import (
    attach_prediction_censor_abspos,
    filter_outcome_eligibility,
    filter_registry_eligible_outcomes,
)
from opera.modules.datamodules.SurvivalFinetuneDataModule import (
    SurvivalFinetuneDataModule,
)
from opera.modules.lightningmodules.SurvivalFinetuneModule import (
    SurvivalFinetuneModule,
)
from opera.run.finetune import load_encoder_state_dict

LOGGER = logging.getLogger(__name__)

load_dotenv()


def _validate_encoder_load(missing, unexpected) -> None:
    """Fail when checkpoint handoff misses anything except the new task head."""
    unexpected = list(unexpected)
    missing = list(missing)
    non_head_missing = [key for key in missing if not key.startswith("cls.")]
    if unexpected or non_head_missing:
        raise RuntimeError(
            "Encoder checkpoint did not load cleanly into the survival finetune "
            f"model. Missing non-head keys: {non_head_missing}; "
            f"unexpected keys: {unexpected}."
        )


@hydra.main(
    config_path="../configs",
    config_name="survival_finetune",
    version_base="1.2",
)
def main(cfg: DictConfig) -> None:
    logger = CSVLogger(get_experiment_output_path(), name="survival_finetune_runs")
    model_save_dir = logger.log_dir

    encoder_state, pretrain_hparams = load_encoder_state_dict(
        cfg.encoder_ckpt,
        cfg.encoder_source,
    )

    vocab = torch.load(cfg.paths.vocabulary)
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
        cohort=cfg.dataset,
        outcome_name=cfg.outcome,
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
        require_min_followup_train=cfg.labels.get("require_min_followup_train", False),
        require_min_followup_val=cfg.labels.get("require_min_followup_val", False),
        require_min_followup_test=cfg.labels.get("require_min_followup_test", False),
        outcome_name=cfg.outcome,
        competing_event_df=competing_df,
    )

    if cfg.training_mode == "ipcw_bce":
        train_ipcw = compute_ipcw_train_weights(
            train_outcomes,
            horizon_hours=cfg.labels.n_hours_end_include,
        )
        attach_ipcw_weights(train_outcomes, train_ipcw)
        val_ipcw = compute_ipcw_train_weights(
            val_outcomes,
            horizon_hours=cfg.labels.n_hours_end_include,
        )
        attach_ipcw_weights(val_outcomes, val_ipcw)
    elif cfg.training_mode != "cox":
        raise ValueError("training_mode must be one of {'cox', 'ipcw_bce'}.")

    save_binarized_split_summary(
        outcomes=outcomes,
        split_outputs={
            cfg.labels.train_key: train_outcomes,
            cfg.labels.val_key: val_outcomes,
            cfg.labels.test_key: test_outcomes,
        },
        require_min_followup_by_split={
            cfg.labels.train_key: cfg.labels.get("require_min_followup_train", False),
            cfg.labels.val_key: cfg.labels.get("require_min_followup_val", False),
            cfg.labels.test_key: cfg.labels.get("require_min_followup_test", False),
        },
        path=f"{model_save_dir}/label_split_summary.csv",
        n_hours_end_include=cfg.labels.n_hours_end_include,
        outcome_name=cfg.outcome,
    )

    data_module = SurvivalFinetuneDataModule(
        batch_size=cfg.training.batch_size,
        num_workers=cfg.hardware.num_workers,
        path_train_data=cfg.paths.train_split,
        path_val_data=cfg.paths.val_split,
        path_population=cfg.paths.population,
        train_outcomes=train_outcomes,
        val_outcomes=val_outcomes,
        test_outcomes=test_outcomes,
        predict_token_id=vocab["[CLS]"],
        train_sampler=None,
        batch_sampling=cfg.training.get("batch_sampling", {}),
    )

    model_cfg = get_saved_encoder_config(pretrain_hparams)
    for key in ("vocab_size", "pad_token_id", "cls_token_id", "sep_token_id"):
        model_cfg.pop(key, None)
    if cfg.get("model"):
        for key, value in cfg.model.items():
            if key not in model_cfg:
                model_cfg[key] = value

    model = BonsaiFinetune(
        ModernBertConfig(
            **model_cfg,
            vocab_size=len(vocab),
            pad_token_id=0,
            cls_token_id=1,
            sep_token_id=2,
        ),
    )
    missing, unexpected = model.load_state_dict(encoder_state, strict=False)
    _validate_encoder_load(missing, unexpected)
    LOGGER.info(
        "Loaded encoder weights with %d missing and %d unexpected keys.",
        len(missing),
        len(unexpected),
    )

    lightning_module = SurvivalFinetuneModule(
        model=model,
        training_mode=cfg.training_mode,
        learning_rate=cfg.training.learning_rate,
        optimizer_epsilon=cfg.training.optimizer_epsilon,
        scheduler_warmup_epochs=cfg.training.scheduler_warmup_epochs,
        checkpoint_metadata={
            "training_stage": "survival_finetuning",
            "training_mode": cfg.training_mode,
            "source_checkpoint": cfg.encoder_ckpt,
            "encoder_source": cfg.encoder_source,
            "dataset": cfg.dataset,
            "outcome": cfg.outcome,
            "n_hours_end_include": cfg.labels.n_hours_end_include,
            "split_identifier": (
                f"{cfg.labels.train_key}:{cfg.labels.val_key}:{cfg.labels.test_key}"
            ),
        },
        pos_weight=None,
    )

    monitor = (
        "val/AUROC" if cfg.training_mode == "ipcw_bce" else "val/concordance_index"
    )
    callbacks = [
        ModelCheckpoint(
            dirpath=model_save_dir,
            monitor=monitor,
            mode="max",
            save_top_k=1,
            filename="best",
            enable_version_counter=False,
            save_last=True,
        ),
    ]
    if cfg.training.get("early_stopping_patience"):
        callbacks.append(
            EarlyStopping(
                monitor=monitor,
                patience=cfg.training.early_stopping_patience,
                mode="max",
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
    torch.save(
        {"train": train_outcomes, "val": val_outcomes, "test": test_outcomes},
        f"{model_save_dir}/outcome_splits.pt",
    )


if __name__ == "__main__":
    main()
