"""
OPERA fine-tuning stage.

Loads an encoder from any prior stage (contrastive, DAPT, or base pretrain)
and fine-tunes a classification head on a specific cohort + outcome.

This is deliberately flexible: the config specifies which checkpoint to
load and which outcome to target, so you can run it on different cohort
specifications without code changes.

Usage:
    python -m opera.run.finetune \
        encoder_ckpt=/path/to/contrastive/best.ckpt \
        encoder_source=contrastive \
        dataset=hematology_cohort \
        outcome=treatment_failure
"""

import pandas as pd
import hydra
import lightning as L
import torch
from typing import Optional
from dotenv import load_dotenv
from omegaconf import DictConfig, OmegaConf
from transformers import ModernBertConfig
from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping

from bonsai.functional.pathing import get_experiment_output_path
from bonsai.modules.datamodules.FinetuneDataModule import FinetuneDataModule
from bonsai.modules.lightningmodules.FinetuneModule import FinetuneModule
from opera.compat.bonsai import BonsaiFinetune
from opera.modules.networks.linear_probe_net import BonsaiLinearProbe
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
from opera.functional.linear_probe import freeze_encoder_for_linear_probe
from opera.functional.outcomes import (
    attach_prediction_censor_abspos,
    filter_registry_eligible_outcomes,
)

load_dotenv()


def load_encoder_state_dict(
    ckpt_path: str, source: str, model_config: Optional[dict] = None
) -> dict:
    """
    Extract encoder weights from different checkpoint types.

    Parameters
    ----------
    ckpt_path : str
    source : str
        One of "pretrain", "dapt", "contrastive", "mol", "random_init".
    """
    if source in {"random_init", "none", "scratch", "no_pretraining"}:
        if model_config is None:
            raise ValueError(
                "random_init finetuning requires model architecture config."
            )
        return {}, dict(model_config)

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state_dict = ckpt["state_dict"]
    hparams = ckpt["hyper_parameters"]

    encoder_state = {}

    if source in ("pretrain", "dapt"):
        # PretrainModule: weights are "model.XXX"
        for k, v in state_dict.items():
            if k.startswith("model."):
                clean = k[len("model.") :]
                if clean.startswith("head.") or clean.startswith("decoder."):
                    continue
                encoder_state[clean] = v

    elif source in ("contrastive", "mol", "joint"):
        # Both OperaContrastiveModule and MOLModule store encoder as "model.encoder.XXX"
        for k, v in state_dict.items():
            if k.startswith("model.encoder."):
                clean = k[len("model.encoder.") :]
                encoder_state[clean] = v

    else:
        raise ValueError(f"Unknown encoder source: {source}")

    return encoder_state, hparams


@hydra.main(
    config_path="../configs",
    config_name="finetune",
    version_base="1.2",
)
def main(cfg: DictConfig) -> None:
    logger = CSVLogger(get_experiment_output_path(), name="finetune_runs")
    model_save_dir = logger.log_dir

    # ── Load encoder ─────────────────────────────────────────────────
    encoder_state, pretrain_hparams = load_encoder_state_dict(
        cfg.encoder_ckpt,
        cfg.encoder_source,
        model_config=OmegaConf.to_container(cfg.model, resolve=True),
    )

    vocab = torch.load(cfg.paths.vocabulary)
    outcomes = pd.read_parquet(cfg.paths.outcome)
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
            cfg.labels.train_key: cfg.labels.get("require_min_followup_train", False),
            cfg.labels.val_key: cfg.labels.get("require_min_followup_val", True),
            cfg.labels.test_key: cfg.labels.get("require_min_followup_test", True),
        },
        path=f"{model_save_dir}/label_split_summary.csv",
        n_hours_end_include=cfg.labels.n_hours_end_include,
        outcome_name=cfg.outcome,
    )

    train_labels = [v["label"] for v in train_outcomes.values()]

    data_module = FinetuneDataModule(
        batch_size=cfg.training.batch_size,
        num_workers=cfg.hardware.num_workers,
        path_train_data=cfg.paths.train_split,
        path_val_data=cfg.paths.val_split,
        path_population=cfg.paths.population,
        train_outcomes=train_outcomes,
        val_outcomes=val_outcomes,
        test_outcomes=test_outcomes,
        predict_token_id=vocab["[CLS]"],
        train_sampler=get_sampler(
            weight_fn=cfg.training.sampling_weight_fn, labels=train_labels
        ),
    )

    # ── Build finetune model and load encoder weights ────────────────
    model_cfg = get_saved_encoder_config(pretrain_hparams)
    for key in ("vocab_size", "pad_token_id", "cls_token_id", "sep_token_id"):
        model_cfg.pop(key, None)
    # Override with any explicit finetune model config
    non_architecture_model_keys = {"freeze_encoder", "trainable_prefixes", "head_type"}
    if cfg.get("model"):
        for k, v in cfg.model.items():
            if k in non_architecture_model_keys:
                continue
            if k not in model_cfg:  # don't override pretrain architecture
                model_cfg[k] = v

    head_type = cfg.model.get(
        "head_type",
        "linear_probe" if cfg.model.get("freeze_encoder", False) else "finetune_head",
    )
    model_class = BonsaiLinearProbe if head_type == "linear_probe" else BonsaiFinetune
    model = model_class(
        ModernBertConfig(
            **model_cfg,
            vocab_size=len(vocab),
            pad_token_id=0,
            cls_token_id=1,
            sep_token_id=2,
        ),
    )

    if encoder_state:
        # The finetuning head is new, but all encoder tensors must match.
        missing, unexpected = model.load_state_dict(encoder_state, strict=False)
        allowed_missing_prefixes = (
            ("classifier.",) if head_type == "linear_probe" else ("cls.",)
        )
        meaningful_missing = [
            key for key in missing if not key.startswith(allowed_missing_prefixes)
        ]
        if meaningful_missing or unexpected:
            raise RuntimeError(
                "Encoder checkpoint is incompatible with the finetuning model. "
                f"Missing encoder keys: {meaningful_missing[:10]}; "
                f"unexpected keys: {unexpected[:10]}"
            )
        print("Loaded encoder weights with a newly initialized prediction head.")
    else:
        print("No encoder checkpoint loaded; using random initialization.")
    linear_probe_metadata = {}
    if cfg.model.get("freeze_encoder", False):
        linear_probe_metadata = freeze_encoder_for_linear_probe(
            model,
            trainable_prefixes=tuple(
                cfg.model.get("trainable_prefixes")
                or (["classifier."] if head_type == "linear_probe" else ["cls."])
            ),
        )
        print(
            "Frozen encoder for linear probe. "
            f"Trainable params: {linear_probe_metadata['n_trainable_parameters']}; "
            f"frozen params: {linear_probe_metadata['n_frozen_parameters']}"
        )

    training_stage = cfg.get(
        "training_stage",
        "linear_probe"
        if cfg.model.get("freeze_encoder", False)
        else "per_task_finetuning",
    )

    lightning_module = FinetuneModule(
        model=model,
        learning_rate=cfg.training.learning_rate,
        optimizer_epsilon=cfg.training.optimizer_epsilon,
        scheduler_warmup_epochs=cfg.training.scheduler_warmup_epochs,
        checkpoint_metadata={
            "training_stage": training_stage,
            "source_checkpoint": cfg.encoder_ckpt,
            "encoder_source": cfg.encoder_source,
            "encoder_frozen": bool(cfg.model.get("freeze_encoder", False)),
            "head_type": head_type,
            "dataset": cfg.dataset,
            "outcome": cfg.outcome,
            "split_identifier": (
                f"{cfg.labels.train_key}:{cfg.labels.val_key}:{cfg.labels.test_key}"
            ),
            **linear_probe_metadata,
        },
        pos_weight=get_loss_weight(
            cfg.training.loss_weight_function,
            labels=train_labels,
        ),
    )

    # ── Callbacks ────────────────────────────────────────────────────
    callbacks = [
        ModelCheckpoint(
            dirpath=model_save_dir,
            monitor=cfg.training.eval_monitor_metric,
            mode="max" if "AUROC" in cfg.training.eval_monitor_metric else "min",
            save_top_k=1,
            filename="best",
            enable_version_counter=False,
            save_last=True,
        ),
    ]
    if cfg.training.get("early_stopping_patience"):
        callbacks.append(
            EarlyStopping(
                monitor=cfg.training.eval_monitor_metric,
                patience=cfg.training.early_stopping_patience,
                mode="max" if "AUROC" in cfg.training.eval_monitor_metric else "min",
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

    trainer.fit(
        model=lightning_module,
        datamodule=data_module,
    )
    save_checkpoint_metadata_sidecar(model_save_dir, lightning_module)

    # ── Save test outcomes for evaluation ────────────────────────────
    torch.save(
        {"train": train_outcomes, "val": val_outcomes, "test": test_outcomes},
        f"{model_save_dir}/outcome_splits.pt",
    )


if __name__ == "__main__":
    main()
