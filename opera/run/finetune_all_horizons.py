"""Fine-tune one cohort-fine/outcome specialist across every time horizon.

This runner intentionally complements, rather than replaces,
``opera.run.finetune``. The older runner remains the fixed-horizon binary
baseline. This runner starts from a trained multi-outcome direct-risk model,
preserves its complete prediction path exactly, and optimizes one outcome on
one fine diagnosis using the parent's full time grid.

For factorized cohort/outcome attention, patients in a fine diagnosis are
routed through their known ``cohort_grouped`` parent at initialization. This
is an exact use of the learned parent function, not an estimated fine-cohort
embedding. Full end-to-end fine-tuning then adapts that path to the fine
diagnosis.
"""

from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping

import lightning as L
import torch
from dotenv import load_dotenv
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import ConcatDataset

from bonsai.functional.checkpointing import (
    clean_lightning_state_dict,
    get_saved_encoder_config,
    mark_training_complete,
    resolve_resume_checkpoint,
    save_checkpoint_metadata_sidecar,
    should_skip_completed_training,
)
from bonsai.functional.input_contract import resolve_numeric_value_control
from opera.diagnostics.representation_gradient_conflict import _load_model
from opera.functional.outcome_transfer import file_hash
from opera.functional.cohort_groups import fine_to_grouped
from opera.modules.datamodules.MultiCohortContrastiveDataModule import (
    MultiCohortContrastiveDataModule,
)
from opera.modules.lightningmodules.OperaContrastiveModule import OperaContrastiveModule


def _parse_csv_floats(value: str | None) -> list[float] | None:
    if value in (None, ""):
        return None
    result = [float(item.strip()) for item in value.split(",") if item.strip()]
    if not result or any(item <= 0 for item in result):
        raise ValueError("Evaluation horizons must be positive comma-separated days.")
    if any(right <= left for left, right in zip(result, result[1:])):
        raise ValueError("Evaluation horizons must be strictly increasing.")
    return result


def resolve_parent_cohort(
    cohort_fine: str,
    configured_parent: str | None,
    available_cohorts: Mapping[str, Any],
) -> str:
    """Resolve and validate the grouped parent used by the source model."""
    canonical = fine_to_grouped(cohort_fine)
    parent = configured_parent or canonical
    if parent is None:
        raise ValueError(
            f"No canonical cohort_grouped parent is registered for {cohort_fine!r}; "
            "provide --parent-cohort explicitly."
        )
    if canonical is not None and configured_parent is not None and parent != canonical:
        raise ValueError(
            f"Fine cohort {cohort_fine!r} maps to {canonical!r}, not {parent!r}."
        )
    if parent not in available_cohorts:
        raise ValueError(
            f"Parent cohort {parent!r} is absent from the source config; "
            f"available={sorted(available_cohorts)}."
        )
    return parent


def build_specialist_cohort_config(
    source_cohort: Mapping[str, Any],
    *,
    cohort_fine_col: str,
    cohort_fine_value: str,
    outcome: str,
) -> dict[str, Any]:
    """Copy a grouped source cohort and narrow membership to one diagnosis."""
    result = deepcopy(dict(source_cohort))
    excluded = set(result.get("exclude_outcomes", []))
    if outcome in excluded:
        raise ValueError(
            f"Outcome {outcome!r} is excluded from this parent cohort in the "
            "source configuration."
        )
    result["cohort_fine_col"] = cohort_fine_col
    result["cohort_fine_value"] = cohort_fine_value
    return result


def _set_dataset_cohort_index(dataset: Any, cohort_index: int) -> int:
    """Route every nested specialist dataset through its source parent index."""
    if isinstance(dataset, ConcatDataset):
        return sum(
            _set_dataset_cohort_index(child, cohort_index)
            for child in dataset.datasets
        )
    if not hasattr(dataset, "cohort_index"):
        raise TypeError(
            f"Dataset {type(dataset).__name__} does not expose cohort_index."
        )
    dataset.cohort_index = int(cohort_index)
    return 1


def route_specialist_to_parent(data_module: Any, parent_index: int) -> int:
    """Apply exact parent routing to train and validation datasets."""
    count = _set_dataset_cohort_index(data_module.train_dataset, parent_index)
    if data_module.val_dataset is not None:
        count += _set_dataset_cohort_index(data_module.val_dataset, parent_index)
    return count


def configure_specialist_objective(model: Any, objective: str) -> str:
    """Disable multi-task policies while retaining the parent's prediction path."""
    loss = getattr(model, "competing_risk_loss", None)
    if loss is None or getattr(model, "competing_risk_head", None) is None:
        raise ValueError("The source checkpoint has no direct competing-risk head.")
    selected = loss.objective if objective == "inherit" else objective
    if selected not in {"piecewise_exponential", "horizon_bce"}:
        raise ValueError(
            "Objective must be inherit, piecewise_exponential, or horizon_bce."
        )
    loss.objective = selected
    # A one-outcome specialist has no task-weighting or curriculum problem to
    # solve. Keeping a learned parent Kendall scale would only change its
    # effective learning rate; keeping a parent curriculum tier could suppress
    # the sole target for early epochs.
    loss.weighter_name = "uniform"
    loss.curriculum_enabled = False
    model.contrastive_loss_weight = 0.0
    model.competing_risk_weight = 1.0
    model.dapt_anchor_weight = 0.0
    model.dapt_embedding_store = None
    model_init = getattr(model, "model_init_config", None)
    if isinstance(model_init, dict):
        model_init.setdefault("competing_risk_config", {})["objective"] = selected
    return selected


def verify_exact_parent_state(
    checkpoint: Mapping[str, Any], model: torch.nn.Module
) -> dict[str, int]:
    """Prove that reconstruction changed no learned tensor before training."""
    expected = clean_lightning_state_dict(dict(checkpoint["state_dict"]))
    actual = model.state_dict()
    if set(expected) != set(actual):
        missing = sorted(set(expected) - set(actual))
        unexpected = sorted(set(actual) - set(expected))
        raise RuntimeError(
            "Source reconstruction is not exact: "
            f"missing={missing[:5]}, unexpected={unexpected[:5]}."
        )
    unequal = [
        name
        for name, value in expected.items()
        if not torch.equal(value.detach().cpu(), actual[name].detach().cpu())
    ]
    if unequal:
        raise RuntimeError(
            "Source reconstruction changed learned tensors before specialist "
            f"training: {unequal[:5]}."
        )
    return {
        "n_state_tensors": len(expected),
        "n_state_values": sum(value.numel() for value in expected.values()),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config-name", required=True, help="Source model Hydra config.")
    parser.add_argument(
        "--config-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "configs",
    )
    parser.add_argument("--override", action="append", default=[])
    parser.add_argument("--outcome", required=True)
    parser.add_argument("--cohort-fine", required=True)
    parser.add_argument("--cohort-fine-col", default="cohort_fine")
    parser.add_argument("--parent-cohort", default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--objective",
        choices=["inherit", "piecewise_exponential", "horizon_bce"],
        default="inherit",
    )
    parser.add_argument("--evaluation-horizons", default=None)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--encoder-lr-multiplier", type=float, default=0.1)
    parser.add_argument("--optimizer-epsilon", type=float, default=1e-6)
    parser.add_argument("--scheduler-warmup-epochs", type=int, default=1)
    parser.add_argument("--accumulate-grad-batches", type=int, default=1)
    parser.add_argument("--early-stopping-patience", type=int, default=5)
    parser.add_argument("--monitor", default="val/loss")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--precision", default=None)
    parser.add_argument("--accelerator", default=None)
    parser.add_argument("--devices", type=int, default=None)
    parser.add_argument("--resume-checkpoint", type=Path, default=None)
    parser.add_argument("--auto-resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    positive = {
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "encoder_lr_multiplier": args.encoder_lr_multiplier,
        "optimizer_epsilon": args.optimizer_epsilon,
        "accumulate_grad_batches": args.accumulate_grad_batches,
    }
    invalid = [name for name, value in positive.items() if value <= 0]
    if invalid:
        raise ValueError(f"These arguments must be positive: {invalid}.")
    if args.num_workers < 0 or args.scheduler_warmup_epochs < 0:
        raise ValueError("num-workers and scheduler-warmup-epochs cannot be negative.")
    if args.early_stopping_patience < 0:
        raise ValueError("early-stopping-patience cannot be negative.")


def _run_identity(args: argparse.Namespace, parent: str, objective: str) -> dict:
    return {
        "training_stage": "cohort_outcome_all_horizons_finetuning",
        "source_checkpoint": str(args.checkpoint.resolve()),
        "source_checkpoint_sha256": file_hash(args.checkpoint),
        "source_config_name": args.config_name,
        "target_outcome": args.outcome,
        "cohort_fine": args.cohort_fine,
        "cohort_fine_col": args.cohort_fine_col,
        "parent_cohort_grouped": parent,
        "transfer_mode": "exact_full_parent_path",
        "objective": objective,
        "seed": args.seed,
    }


def main() -> None:
    args = build_parser().parse_args()
    _validate_args(args)
    load_dotenv()
    L.seed_everything(args.seed, workers=True)

    GlobalHydra.instance().clear()
    with initialize_config_dir(
        config_dir=str(args.config_dir.resolve()), version_base="1.2"
    ):
        cfg: DictConfig = compose(
            config_name=args.config_name,
            overrides=list(args.override),
        )

    source_cohorts = OmegaConf.to_container(cfg.cohorts, resolve=True)
    source_outcomes = OmegaConf.to_container(cfg.outcomes, resolve=True)
    if not isinstance(source_cohorts, dict) or not isinstance(source_outcomes, dict):
        raise TypeError("Source cohorts and outcomes must resolve to mappings.")
    if args.outcome not in source_outcomes:
        raise ValueError(
            f"Outcome {args.outcome!r} is absent from the source config; "
            f"available={sorted(source_outcomes)}."
        )
    parent = resolve_parent_cohort(
        args.cohort_fine, args.parent_cohort, source_cohorts
    )

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    source_outcome_names = sorted(source_outcomes)
    # Reconstruction uses the checkpoint's own model_init_config and performs
    # a strict state load. Event-time grids are contrastive-only metadata here.
    empty_grids = {
        name: torch.empty(0, dtype=torch.float32) for name in source_outcome_names
    }
    configured_dapt_store = cfg.get("dapt_embedding_store")
    cfg.dapt_embedding_store = None
    try:
        model = _load_model(
            args.checkpoint,
            cfg,
            source_outcome_names,
            empty_grids,
            empty_grids,
            torch.device("cpu"),
            seed=args.seed,
        )
    finally:
        cfg.dapt_embedding_store = configured_dapt_store
    exact_report = verify_exact_parent_state(checkpoint, model)
    selected_objective = configure_specialist_objective(model, args.objective)

    parent_names = list(model.model_init_config.get("cohort_names", []))
    if parent not in parent_names:
        raise ValueError(
            f"Parent cohort {parent!r} is absent from checkpoint routing; "
            f"checkpoint cohorts={parent_names}."
        )
    parent_index = parent_names.index(parent)

    specialist_cohort = build_specialist_cohort_config(
        source_cohorts[parent],
        cohort_fine_col=args.cohort_fine_col,
        cohort_fine_value=args.cohort_fine,
        outcome=args.outcome,
    )
    vocabulary = torch.load(cfg.paths.vocabulary, map_location="cpu", weights_only=False)
    encoder_config = get_saved_encoder_config(checkpoint["hyper_parameters"])
    max_len = int(encoder_config["max_seqlen"])
    numeric_value_control = resolve_numeric_value_control(
        cfg.training.get("numeric_value_control", "inherit"),
        checkpoint.get("hyper_parameters", {}),
    )
    data_module = MultiCohortContrastiveDataModule(
        cohort_configs={parent: specialist_cohort},
        outcome_configs={args.outcome: source_outcomes[args.outcome]},
        predict_token_id=int(vocabulary["[CLS]"]),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        require_min_followup_train=bool(
            cfg.training.get("require_min_followup_train", False)
        ),
        require_min_followup_val=bool(
            cfg.training.get("require_min_followup_val", False)
        ),
        require_all_configured_cells=True,
        max_len=max_len,
        batch_sampling={"type": "random", "seed": args.seed},
        numeric_value_control=numeric_value_control,
    )
    data_module.setup("fit")
    routed_datasets = route_specialist_to_parent(data_module, parent_index)

    boundaries = [
        float(value) * model.competing_risk_loss.time_scale_days
        for value in model.competing_risk_loss.boundaries.detach().cpu().tolist()
    ]
    evaluation_horizons = _parse_csv_floats(args.evaluation_horizons) or boundaries
    identity = _run_identity(args, parent, selected_objective)
    identity.update(
        {
            "parent_cohort_index": parent_index,
            "interval_boundaries_days": boundaries,
            "evaluation_horizons_days": evaluation_horizons,
            "source_outcome_set": source_outcome_names,
            "readout_mode": model.outcome_readout_mode,
            "head_mode": model.model_init_config["competing_risk_config"].get(
                "head_mode", "linear"
            ),
            "numeric_value_control": numeric_value_control,
        }
    )

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    run_config = {
        **vars(args),
        "checkpoint": str(args.checkpoint.resolve()),
        "config_dir": str(args.config_dir.resolve()),
        "output_dir": str(output_dir),
        "resume_checkpoint": (
            None if args.resume_checkpoint is None else str(args.resume_checkpoint.resolve())
        ),
        "resolved_parent_cohort": parent,
        "resolved_objective": selected_objective,
        "resolved_evaluation_horizons": evaluation_horizons,
    }
    if should_skip_completed_training(output_dir, run_config):
        return

    (output_dir / "initialization_report.json").write_text(
        json.dumps(
            {
                **identity,
                **exact_report,
                "exact_parent_state_verified": True,
                "n_routed_datasets": routed_datasets,
                "n_train_subjects": len(data_module.train_dataset),
                "n_tuning_subjects": (
                    0 if data_module.val_dataset is None else len(data_module.val_dataset)
                ),
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    OmegaConf.save(cfg, output_dir / "source_config_resolved.yaml", resolve=True)

    lightning_module = OperaContrastiveModule(
        model=model,
        outcome_names=[args.outcome],
        learning_rate=args.learning_rate,
        encoder_lr_multiplier=args.encoder_lr_multiplier,
        optimizer_epsilon=args.optimizer_epsilon,
        scheduler_warmup_epochs=args.scheduler_warmup_epochs,
        dapt_anchor_weight=0.0,
        gradient_cache=False,
        enable_validation_probe=False,
        log_per_outcome_metrics=True,
        evaluation_horizons_days=evaluation_horizons,
        checkpoint_metadata=identity,
    )
    monitor_mode = "max" if "auc" in args.monitor.lower() else "min"
    callbacks: list[Any] = [
        ModelCheckpoint(
            dirpath=output_dir,
            monitor=args.monitor,
            mode=monitor_mode,
            save_top_k=1,
            filename="best",
            enable_version_counter=False,
            save_last=True,
        ),
        LearningRateMonitor(logging_interval="step"),
    ]
    if args.early_stopping_patience > 0:
        callbacks.append(
            EarlyStopping(
                monitor=args.monitor,
                patience=args.early_stopping_patience,
                mode=monitor_mode,
            )
        )
    logger = CSVLogger(str(output_dir), name="logs")
    hardware = cfg.get("hardware", {})
    trainer = L.Trainer(
        accelerator=args.accelerator or hardware.get("accelerator", "auto"),
        devices=args.devices or hardware.get("num_devices", 1),
        precision=args.precision or hardware.get("precision", "32-true"),
        max_epochs=args.epochs,
        accumulate_grad_batches=args.accumulate_grad_batches,
        callbacks=callbacks,
        logger=logger,
        log_every_n_steps=10,
        limit_train_batches=cfg.training.get("limit_train_batches", 1.0),
        limit_val_batches=cfg.training.get("limit_val_batches", 1.0),
    )
    resume = resolve_resume_checkpoint(
        output_dir,
        args.resume_checkpoint,
        auto_resume=args.auto_resume,
    )
    trainer.fit(lightning_module, datamodule=data_module, ckpt_path=resume)
    save_checkpoint_metadata_sidecar(
        str(output_dir),
        lightning_module,
        extra={"initialization_report": "initialization_report.json"},
    )
    mark_training_complete(output_dir, run_config)


if __name__ == "__main__":
    main()
