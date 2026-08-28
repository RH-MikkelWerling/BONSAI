"""OPERA survival and IPCW-BCE finetuning runner."""

import logging
from typing import Optional

import hydra
import lightning as L
import pandas as pd
import torch
from dotenv import load_dotenv
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger
from omegaconf import DictConfig, OmegaConf

from bonsai.functional.checkpointing import (
    get_saved_encoder_config,
    save_checkpoint_metadata_sidecar,
)
from bonsai.functional.checkpointing import (
    mark_training_complete,
    should_skip_completed_training,
)
from bonsai.functional.outcomes import (
    save_binarized_split_summary,
    split_and_binarize_outcomes,
)
from bonsai.functional.pathing import get_experiment_output_path
from bonsai.functional.input_contract import (
    input_contract_metadata,
    resolve_numeric_value_control,
)
from bonsai.functional.versioning import generate_unused_run_id
from opera.compat.bonsai import build_bonsai_finetune
from opera.functional.ipcw import (
    attach_ipcw_weights,
    compute_ipcw_train_weights,
    summarize_ipcw_weights,
)
from opera.functional.outcomes import (
    attach_prediction_censor_abspos,
    filter_outcome_eligibility,
    filter_registry_eligible_outcomes,
)
from opera.modules.datamodules.SurvivalFinetuneDataModule import (
    SurvivalFinetuneDataModule,
    resolve_survival_batch_sampler_type,
)
from opera.modules.lightningmodules.SurvivalFinetuneModule import (
    SurvivalFinetuneModule,
)
from opera.run.finetune import load_encoder_state_dict
from opera.evaluation.cohorts import population_subject_ids

LOGGER = logging.getLogger(__name__)

# configs/core/base_train.yaml's default `run_id: ${version:}` needs this
# resolver. bonsai/run/{pretrain,finetune,train}.py each register it as an
# import-time side effect, but none of those modules are imported by this
# entry point (or by any other opera/run/*.py Hydra script), so running this
# script directly without an explicit `run_id=...` CLI override previously
# raised `UnsupportedInterpolationType` before any config could be read.
OmegaConf.register_new_resolver(
    "version", lambda: generate_unused_run_id(), use_cache=True, replace=True
)

load_dotenv()


def _validate_encoder_load(missing, unexpected, encoder_source: str) -> None:
    """Validate checkpoint handoff, or accept intentional random initialization."""
    if encoder_source in {"random_init", "none", "scratch", "no_pretraining"}:
        # An empty state dict is the definition of this control: every model
        # parameter should retain its freshly initialized value.
        return
    unexpected = list(unexpected)
    missing = list(missing)
    non_head_missing = [key for key in missing if not key.startswith("finetune_head.")]
    if unexpected or non_head_missing:
        raise RuntimeError(
            "Encoder checkpoint did not load cleanly into the survival finetune "
            f"model. Missing non-head keys: {non_head_missing}; "
            f"unexpected keys: {unexpected}."
        )


def resolve_survival_finetune_max_len(
    cfg: DictConfig,
    encoder_max_seqlen: Optional[int] = None,
) -> int:
    """Resolve a sequence length that does not exceed the saved encoder limit."""
    value = cfg.training.get("max_len")
    if value is None:
        value = cfg.model.get("max_seqlen", 8192)
    value = int(value)
    if encoder_max_seqlen is not None:
        value = min(value, int(encoder_max_seqlen))
    return value


def resolve_survival_monitor(cfg: DictConfig) -> tuple[str, str]:
    """Resolve an estimand-appropriate checkpoint-selection metric."""
    configured = str(cfg.training.get("eval_monitor_metric", "auto"))
    if configured != "auto":
        mode = "max" if configured in {"val/AUROC", "val/concordance_index"} else "min"
        return configured, mode
    if cfg.training_mode in {"cox", "cox_exact_cached"}:
        return "val/concordance_index", "max"
    return "val/loss", "min"


def _validate_and_report_ipcw_weights(
    split_name: str,
    outcomes: dict,
    weights: dict,
) -> dict:
    """Fail on absent training signal and warn on unstable IPCW support."""
    diagnostics = summarize_ipcw_weights(outcomes, weights)
    if diagnostics["n_nonzero"] == 0:
        raise ValueError(f"{split_name} has no IPCW-observed subjects.")
    if split_name == "train" and (
        diagnostics["n_cases_nonzero"] == 0 or diagnostics["n_controls_nonzero"] == 0
    ):
        raise ValueError(
            "IPCW training requires at least one observed case and control; "
            f"{split_name} diagnostics={diagnostics}."
        )
    if split_name != "train" and (
        diagnostics["n_cases_nonzero"] == 0 or diagnostics["n_controls_nonzero"] == 0
    ):
        LOGGER.warning(
            "%s has only one IPCW-observed class. Weighted validation loss is "
            "defined, but validation AUROC is not estimable: %s.",
            split_name,
            diagnostics,
        )
    if diagnostics["effective_sample_fraction"] < 0.1:
        LOGGER.warning(
            "%s IPCW effective sample fraction is %.3f; estimates may be unstable.",
            split_name,
            diagnostics["effective_sample_fraction"],
        )
    if diagnostics["max_weight"] > 20.0:
        LOGGER.warning(
            "%s IPCW maximum normalized weight is %.2f; inspect positivity and "
            "administrative follow-up before interpreting this cell.",
            split_name,
            diagnostics["max_weight"],
        )
    return {"split": split_name, **diagnostics}


def _survival_support_summary(split_name: str, outcomes: dict) -> dict:
    """Summarize primary events and Cox-comparable events for one split."""
    records = list(outcomes.values())
    times = [float(record["time_days"]) for record in records]
    events = [int(record["event"]) for record in records]
    comparable_events = sum(
        event == 1 and any(other_time > time for other_time in times)
        for time, event in zip(times, events)
    )
    return {
        "split": split_name,
        "n_total": len(records),
        "n_primary_events": sum(event == 1 for event in events),
        "n_competing_events": sum(event == 2 for event in events),
        "n_admin_censored": sum(event == 0 for event in events),
        "n_comparable_primary_events": int(comparable_events),
    }


def _validate_cox_support(split_name: str, outcomes: dict) -> dict:
    diagnostics = _survival_support_summary(split_name, outcomes)
    if diagnostics["n_primary_events"] == 0:
        raise ValueError(
            f"Cox {split_name} split has no observed primary events: {diagnostics}."
        )
    if diagnostics["n_comparable_primary_events"] == 0:
        raise ValueError(
            f"Cox {split_name} split has no comparable primary events: {diagnostics}."
        )
    return diagnostics


def build_survival_finetune_data_module(
    cfg: DictConfig,
    vocab: dict,
    train_outcomes: dict,
    val_outcomes: dict,
    test_outcomes: dict,
    encoder_max_seqlen: Optional[int] = None,
    numeric_value_control: str = "observed",
) -> SurvivalFinetuneDataModule:
    """Construct the datamodule exactly as the survival runner uses it."""
    return SurvivalFinetuneDataModule(
        batch_size=cfg.training.batch_size,
        num_workers=cfg.hardware.num_workers,
        path_train_data=cfg.paths.train_split,
        path_val_data=cfg.paths.val_split,
        path_predict_data=cfg.paths.get("test_split"),
        subject_data_paths=cfg.paths.get("subject_data_paths"),
        path_population=cfg.paths.population,
        train_outcomes=train_outcomes,
        val_outcomes=val_outcomes,
        predict_outcomes=test_outcomes,
        predict_token_id=vocab["[CLS]"],
        max_len=resolve_survival_finetune_max_len(cfg, encoder_max_seqlen),
        train_sampler=None,
        batch_sampling=cfg.training.get("batch_sampling", {}),
        training_mode=cfg.get("training_mode", "cox"),
        numeric_value_control=numeric_value_control,
    )


@hydra.main(
    config_path="../configs",
    config_name="survival_finetune",
    version_base="1.2",
)
def main(cfg: DictConfig) -> None:
    L.seed_everything(int(cfg.seed), workers=True)
    if (
        cfg.training_mode == "cox_exact_cached"
        and int(cfg.training.accumulate_grad_batches) != 1
    ):
        raise ValueError(
            "cox_exact_cached performs its own full-cohort gradient accumulation; "
            "training.accumulate_grad_batches must be 1."
        )
    if cfg.training_mode == "cox_exact_cached" and (
        int(cfg.hardware.num_nodes) != 1 or int(cfg.hardware.num_devices) != 1
    ):
        raise ValueError(
            "cox_exact_cached currently requires one process (num_nodes=1 and "
            "num_devices=1) so every exact risk set contains the full cohort."
        )
    if (
        cfg.training_mode == "cox_exact_cached"
        and float(cfg.training.limit_train_batches) != 1.0
    ):
        raise ValueError(
            "cox_exact_cached requires training.limit_train_batches=1.0; "
            "subsampling would no longer produce full-cohort risk sets."
        )
    logger = CSVLogger(get_experiment_output_path(), name="survival_finetune_runs")
    model_save_dir = get_experiment_output_path()
    if should_skip_completed_training(model_save_dir, cfg):
        return

    encoder_state, pretrain_hparams = load_encoder_state_dict(
        cfg.encoder_ckpt,
        cfg.encoder_source,
        model_config=OmegaConf.to_container(cfg.model, resolve=True),
    )
    model_cfg = (
        get_saved_encoder_config(pretrain_hparams)
        if encoder_state
        # Vocab size is not known until the vocabulary is loaded below. Keep
        # the complete random-init architecture mapping here; the model
        # builder normalizes it together with the actual vocabulary size.
        else dict(pretrain_hparams)
    )
    numeric_value_control = resolve_numeric_value_control(
        cfg.training.get("numeric_value_control", "inherit"),
        pretrain_hparams if encoder_state else None,
    )
    LOGGER.info(
        "Resolved input contract: numeric_value_control=%s",
        numeric_value_control,
    )

    vocab = torch.load(cfg.paths.vocabulary)
    outcomes = pd.read_parquet(cfg.paths.outcome)
    outcomes = filter_outcome_eligibility(
        outcomes,
        cfg.paths.get("eligibility"),
        cohort=cfg.dataset,
        outcome_name=cfg.outcome,
        eligibility_scope="ascertainment",
    )
    outcomes = attach_prediction_censor_abspos(outcomes)
    outcomes = filter_registry_eligible_outcomes(
        outcomes,
        cfg.labels.get("registry_start_date"),
        cohort=cfg.dataset,
        outcome_name=cfg.outcome,
    )
    membership_ids = population_subject_ids(
        cfg.paths.population,
        cohort_fine_col=cfg.get("cohort_fine_col"),
        cohort_fine_value=cfg.get("cohort_fine_value"),
    )
    if membership_ids is not None:
        outcomes = outcomes[outcomes["subject_id"].isin(membership_ids)].copy()
    if outcomes.empty:
        raise ValueError(
            "No eligible outcome rows remain after cohort membership filtering."
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

    if cfg.training_mode in {"ipcw_bce", "ipcw_cif_bce"}:
        estimand = (
            "cumulative_incidence"
            if cfg.training_mode == "ipcw_cif_bce"
            else "net_risk"
        )
        train_ipcw = compute_ipcw_train_weights(
            train_outcomes,
            horizon_hours=cfg.labels.n_hours_end_include,
            estimand=estimand,
        )
        ipcw_diagnostics = [
            _validate_and_report_ipcw_weights(
                cfg.labels.train_key,
                train_outcomes,
                train_ipcw,
            )
        ]
        attach_ipcw_weights(train_outcomes, train_ipcw)
        val_ipcw = compute_ipcw_train_weights(
            val_outcomes,
            horizon_hours=cfg.labels.n_hours_end_include,
            estimand=estimand,
        )
        ipcw_diagnostics.append(
            _validate_and_report_ipcw_weights(
                cfg.labels.val_key,
                val_outcomes,
                val_ipcw,
            )
        )
        attach_ipcw_weights(val_outcomes, val_ipcw)
        pd.DataFrame(ipcw_diagnostics).to_csv(
            f"{model_save_dir}/ipcw_weight_summary.csv",
            index=False,
        )
    elif cfg.training_mode not in {"cox", "cox_exact_cached"}:
        raise ValueError(
            "training_mode must be one of {'cox', 'cox_exact_cached', "
            "'ipcw_bce', 'ipcw_cif_bce'}."
        )
    else:
        cox_diagnostics = [
            _validate_cox_support(cfg.labels.train_key, train_outcomes),
            _validate_cox_support(cfg.labels.val_key, val_outcomes),
            _survival_support_summary(cfg.labels.test_key, test_outcomes),
        ]
        pd.DataFrame(cox_diagnostics).to_csv(
            f"{model_save_dir}/survival_support_summary.csv",
            index=False,
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
            cfg.labels.val_key: cfg.labels.get("require_min_followup_val", False),
            cfg.labels.test_key: cfg.labels.get("require_min_followup_test", False),
        },
        path=f"{model_save_dir}/label_split_summary.csv",
        n_hours_end_include=cfg.labels.n_hours_end_include,
        outcome_name=cfg.outcome,
    )

    data_module = build_survival_finetune_data_module(
        cfg,
        vocab,
        train_outcomes,
        val_outcomes,
        test_outcomes,
        encoder_max_seqlen=model_cfg["max_seqlen"],
        numeric_value_control=numeric_value_control,
    )

    if cfg.get("model"):
        for key, value in cfg.model.items():
            if key not in model_cfg:
                model_cfg[key] = value

    model = build_bonsai_finetune(
        model_cfg,
        vocab_size=len(vocab),
        predict_token_id=vocab["[CLS]"],
    )
    missing, unexpected = model.load_state_dict(encoder_state, strict=False)
    _validate_encoder_load(missing, unexpected, str(cfg.encoder_source))
    LOGGER.info(
        "Loaded encoder weights with %d missing and %d unexpected keys.",
        len(missing),
        len(unexpected),
    )

    monitor, monitor_mode = resolve_survival_monitor(cfg)
    lightning_module = SurvivalFinetuneModule(
        model=model,
        training_mode=cfg.training_mode,
        learning_rate=cfg.training.learning_rate,
        encoder_lr_multiplier=cfg.training.get("encoder_lr_multiplier", 1.0),
        optimizer_epsilon=cfg.training.optimizer_epsilon,
        scheduler_warmup_epochs=cfg.training.scheduler_warmup_epochs,
        checkpoint_metadata={
            "training_stage": "survival_finetuning",
            "training_mode": cfg.training_mode,
            "survival_estimand": {
                "cox": "cause_specific_hazard",
                "cox_exact_cached": "cause_specific_hazard",
                "ipcw_bce": "net_risk",
                "ipcw_cif_bce": "cumulative_incidence",
            }[cfg.training_mode],
            "source_checkpoint": cfg.encoder_ckpt,
            "encoder_source": cfg.encoder_source,
            "dataset": cfg.dataset,
            "outcome": cfg.outcome,
            "n_hours_end_include": cfg.labels.n_hours_end_include,
            "split_identifier": (
                f"{cfg.labels.train_key}:{cfg.labels.val_key}:{cfg.labels.test_key}"
            ),
            "selection_split": cfg.labels.val_key,
            "selection_metric": monitor,
            "selection_mode": monitor_mode,
            **input_contract_metadata(numeric_value_control),
            "seed": int(cfg.seed),
        },
        pos_weight=None,
        horizon_days=(
            None
            if cfg.training_mode in {"cox", "cox_exact_cached"}
            else float(cfg.labels.n_hours_end_include) / 24.0
        ),
    )

    callbacks = [
        ModelCheckpoint(
            dirpath=model_save_dir,
            monitor=monitor,
            mode=monitor_mode,
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
                mode=monitor_mode,
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
        use_distributed_sampler=(
            resolve_survival_batch_sampler_type(
                cfg.training_mode,
                cfg.training.get("batch_sampling", {}),
            )
            == "none"
        ),
    )

    trainer.fit(model=lightning_module, datamodule=data_module)
    save_checkpoint_metadata_sidecar(model_save_dir, lightning_module)
    mark_training_complete(model_save_dir, cfg)
    torch.save(
        {"train": train_outcomes, "val": val_outcomes, "test": test_outcomes},
        f"{model_save_dir}/outcome_splits.pt",
    )


if __name__ == "__main__":
    main()
