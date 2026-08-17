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

import csv

import hydra
import lightning as L
import torch
from dotenv import load_dotenv
from omegaconf import DictConfig, OmegaConf
from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint

from bonsai.functional.pathing import get_experiment_output_path
from bonsai.functional.versioning import generate_unused_run_id
from bonsai.functional.checkpointing import (
    extract_encoder_state_dict,
    get_saved_encoder_config,
    save_checkpoint_metadata_sidecar,
)
from bonsai.functional.checkpointing import (
    mark_training_complete,
    should_skip_completed_training,
)
from opera.compat.bonsai import build_bonsai_encoder, encoder_hparams
from opera.modules.networks.opera_nets import OperaContrastiveModel
from opera.modules.networks.outcome_scaling import resolve_outcome_reference_scales
from opera.modules.networks.competing_risk import (
    summarize_competing_risk_support,
    validate_competing_risk_sampling,
)
from opera.modules.lightningmodules.OperaContrastiveModule import OperaContrastiveModule
from opera.modules.datamodules.MultiCohortContrastiveDataModule import (
    MultiCohortContrastiveDataModule,
    compute_pooled_event_time_probability_grids,
)

load_dotenv()
OmegaConf.register_new_resolver(
    "version", lambda: generate_unused_run_id(), use_cache=True, replace=True
)
from bonsai.functional.input_contract import (
    input_contract_metadata,
    resolve_numeric_value_control,
)


# These are deliberately kept in the contrastive entry point rather than only
# in the config generator.  A generated YAML file is easy to edit by hand;
# this guard makes it impossible to accidentally route a held-out transfer
# target back into the loss, validation data module, or checkpoint selection.
_TRANSFER_METADATA_FIELDS = (
    "condition",
    "transfer_level",
    "seed",
    "included_outcomes",
    "excluded_outcomes",
    "evaluation_outcomes",
    "related_retained_outcomes",
    "direct_dependencies_excluded",
    "registry_hash",
    "manifest_hash",
    "base_contrastive_config_hash",
    "split_contract",
    "split_contract_hash",
    "source_dapt_checkpoint",
    "selection_outcomes",
)


def _validate_transfer_config(cfg: DictConfig, outcome_names: list[str]) -> dict | None:
    """Validate the narrow outcome-transfer training contract at launch.

    The outcome mapping is the only source passed into the contrastive data
    module below.  For transfer ablations, require it to be exactly the
    explicit included panel and require checkpoint selection to use that same
    panel.  Evaluation labels are intentionally *not* loaded in this process;
    they are consumed later by the frozen-probe evaluator.
    """
    if not bool(cfg.get("transfer_analysis", False)):
        return None

    if bool(cfg.get("launch_blocked", False)):
        raise ValueError(
            "This outcome-transfer condition is blocked: "
            f"{cfg.get('launch_blocked_reason', 'no reason recorded')}"
        )

    condition = cfg.get("transfer_condition")
    if not isinstance(condition, str) or not condition:
        raise ValueError("Transfer config requires a non-empty transfer_condition.")

    included = list(cfg.get("training_outcomes", []))
    excluded = list(cfg.get("training_excluded_outcomes", []))
    selection = list(cfg.get("selection_outcomes", []))
    if not included:
        raise ValueError("Transfer config requires non-empty training_outcomes.")
    overlap = sorted(set(excluded) & set(outcome_names))
    if overlap:
        raise ValueError(
            f"Held-out transfer labels cannot enter contrastive adaptation: {overlap}."
        )
    # The existing contrastive model sorts its internal head names while the
    # generated YAML preserves canonical registry order.  The leakage
    # contract is membership-based, not an incidental dict-order contract.
    if set(outcome_names) != set(included) or len(outcome_names) != len(included):
        raise ValueError(
            "Transfer config outcome mapping must contain exactly training_outcomes; "
            f"mapping={outcome_names}, training_outcomes={included}."
        )
    if selection != included:
        raise ValueError(
            "Transfer checkpoint selection must use exactly training_outcomes; "
            f"selection_outcomes={selection}, training_outcomes={included}."
        )
    if len(set(included)) != len(included) or len(set(excluded)) != len(excluded):
        raise ValueError("Transfer outcome lists cannot contain duplicates.")

    seed = cfg.get("seed")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("Transfer contrastive runs require an integer top-level seed.")
    permitted_seeds = list(cfg.get("transfer_seeds", []))
    if permitted_seeds and seed not in permitted_seeds:
        raise ValueError(
            f"Seed {seed} is not declared by transfer_seeds={permitted_seeds}."
        )

    declared = OmegaConf.to_container(
        # Do not resolve unrelated environment interpolations here.  This
        # guard executes before DAPT I/O and should report an outcome-panel
        # breach even in a config-only dry run without server paths exported.
        cfg.get("transfer_checkpoint_metadata", {}),
        resolve=False,
    )
    if not isinstance(declared, dict):
        raise ValueError("transfer_checkpoint_metadata must be a mapping.")
    missing = [field for field in _TRANSFER_METADATA_FIELDS if field not in declared]
    # ``seed`` is populated at runtime below, so it is the one allowed omission
    # in the deterministic condition-level YAML.
    missing = [field for field in missing if field != "seed"]
    if missing:
        raise ValueError(
            f"Transfer checkpoint metadata is incomplete; missing {sorted(missing)}."
        )
    for key, expected in {
        "condition": condition,
        "included_outcomes": included,
        "excluded_outcomes": excluded,
        "selection_outcomes": included,
        "base_contrastive_config_hash": cfg.get("base_contrastive_config_hash"),
        "split_contract_hash": cfg.get("split_contract_hash"),
    }.items():
        if key in declared and declared[key] != expected:
            raise ValueError(
                f"Transfer metadata field {key!r} disagrees with launch config."
            )
    return declared


def _checkpoint_metadata(
    cfg: DictConfig,
    outcome_names: list[str],
    transfer_metadata: dict | None,
    numeric_value_control: str,
) -> dict:
    """Build inspectable checkpoint provenance for standard and transfer runs."""
    metadata = {
        "training_stage": "opera_contrastive_adaptation",
        "source_checkpoint": str(cfg.dapt_ckpt),
        "cohort_set": sorted(cfg.cohorts.keys()),
        "outcome_set": outcome_names,
        **input_contract_metadata(numeric_value_control),
    }
    if transfer_metadata is None:
        return metadata
    # Copy the condition-level metadata and add the actual runtime seed.  This
    # makes the sidecar and Lightning checkpoint self-describing even when the
    # launcher supplied ``seed=...`` as a Hydra override.
    metadata.update(transfer_metadata)
    metadata.update(
        {
            "condition": str(cfg.transfer_condition),
            "transfer_level": str(cfg.transfer_level),
            "seed": int(cfg.seed),
            "included_outcomes": list(cfg.training_outcomes),
            "excluded_outcomes": list(cfg.training_excluded_outcomes),
            "evaluation_outcomes": list(cfg.evaluation_outcomes),
            "related_retained_outcomes": list(cfg.related_retained_outcomes),
            "direct_dependencies_excluded": list(cfg.direct_dependencies_excluded),
            "registry_hash": str(cfg.registry_hash),
            "manifest_hash": str(cfg.manifest_hash),
            "base_contrastive_config_hash": str(cfg.base_contrastive_config_hash),
            "split_contract": str(cfg.split_contract),
            "split_contract_hash": str(cfg.split_contract_hash),
            "source_dapt_checkpoint": str(cfg.dapt_ckpt),
            "selection_outcomes": list(cfg.selection_outcomes),
        }
    )
    return metadata


@hydra.main(
    config_path="../configs",
    config_name="generated/joint_opera_full_panel",
    version_base="1.2",
)
def main(cfg: DictConfig) -> None:
    # Transfer conditions are run once per declared seed.  Existing production
    # configs have no top-level seed, so leave their stochastic behaviour
    # unchanged unless a seed was explicitly supplied.
    if cfg.get("seed") is not None:
        L.seed_everything(int(cfg.seed), workers=True)

    # Validate the outcome panel before creating a logger, opening a DAPT
    # checkpoint, calculating event-time grids, or constructing a data module.
    # This is deliberately the earliest point at which transfer config fields
    # can be inspected, so a hand-edited YAML cannot make a held-out label
    # influence any training or checkpoint-selection operation.
    outcome_configs = OmegaConf.to_container(cfg.outcomes, resolve=True)
    outcome_names = sorted(outcome_configs.keys())
    validate_competing_risk_sampling(
        OmegaConf.to_container(cfg.get("competing_risk", {}), resolve=True),
        OmegaConf.to_container(
            cfg.training.get("batch_sampling", {}),
            resolve=True,
        ),
    )
    transfer_metadata = _validate_transfer_config(cfg, outcome_names)

    logger = CSVLogger(
        get_experiment_output_path(), name="contrastive_multicohort_runs"
    )
    model_save_dir = get_experiment_output_path()
    if should_skip_completed_training(model_save_dir, cfg):
        return

    # ── Load encoder from DAPT checkpoint ─────────────────────────────
    ckpt = torch.load(cfg.dapt_ckpt, map_location="cpu", weights_only=False)
    pretrain_hparams = ckpt["hyper_parameters"]
    numeric_value_control = resolve_numeric_value_control(
        cfg.training.get("numeric_value_control", "inherit"), pretrain_hparams
    )

    vocab = torch.load(cfg.paths.vocabulary, weights_only=False)

    model_cfg = get_saved_encoder_config(pretrain_hparams)
    encoder = build_bonsai_encoder(model_cfg, vocab_size=len(vocab))
    encoder_state = extract_encoder_state_dict(ckpt["state_dict"])
    encoder.load_state_dict(encoder_state, strict=True)

    # ── DAPT-prior embedding store (optional) ─────────────────────────
    dapt_embedding_store = None
    if cfg.get("dapt_embedding_store") is not None:
        print(f"Loading DAPT embedding store from {cfg.dapt_embedding_store} ...")
        dapt_embedding_store = torch.load(cfg.dapt_embedding_store, map_location="cpu")
        print(f"  Loaded {len(dapt_embedding_store)} patient embeddings.")
    elif bool(cfg.training.get("require_dapt_embedding_store", False)):
        raise ValueError(
            "training.require_dapt_embedding_store=true but "
            "dapt_embedding_store is null. Build the store with "
            "python -m opera.run.build_dapt_embedding_store before training."
        )

    # ── Build model ────────────────────────────────────────────────────
    cohort_configs = OmegaConf.to_container(cfg.cohorts, resolve=True)
    require_all_configured_cells = cfg.training.get(
        "require_all_configured_cells", True
    )
    outcome_sorted_event_times, outcome_event_time_probs = (
        compute_pooled_event_time_probability_grids(
            cohort_configs,
            outcome_configs,
            split="train",
            require_all_configured_cells=require_all_configured_cells,
        )
    )
    print(
        f"KM event-time grids computed for {len(outcome_sorted_event_times)} outcomes."
    )
    for name, t in outcome_sorted_event_times.items():
        print(f"  {name}: {len(t)} pooled KM-weighted training event locations")

    cross_outcome_config = OmegaConf.to_container(
        cfg.get("cross_outcome", {}), resolve=True
    )
    cross_outcome_config = resolve_outcome_reference_scales(cross_outcome_config)
    if cross_outcome_config.get("aggregation") == "hierarchical_support":
        cross_outcome_config["event_location_counts"] = {
            name: int(times.numel())
            for name, times in outcome_sorted_event_times.items()
        }

    model = OperaContrastiveModel(
        encoder=encoder,
        outcome_names=outcome_names,
        hidden_size=model_cfg["hidden_size"],
        projection_hidden_dim=cfg.model.projection_hidden_dim,
        projection_dim=cfg.model.projection_dim,
        temperature=cfg.model.temperature,
        km_time_scale=cfg.model.get("km_time_scale", 0.25),
        outcome_sorted_event_times=outcome_sorted_event_times,
        outcome_event_time_probs=outcome_event_time_probs,
        dapt_lambda_floor=cfg.model.get("dapt_lambda_floor", 0.55),
        dapt_anchor_weight=cfg.model.get("dapt_anchor_weight", 0.2),
        competing_event_weight=cfg.model.get("competing_event_weight", 0.0),
        competing_event_handling=cfg.model.get(
            "competing_event_handling", "hard_negative"
        ),
        effective_pair_normalization=cfg.model.get(
            "effective_pair_normalization", True
        ),
        cross_outcome_config=cross_outcome_config,
        projection_mode=cfg.model.get("projection_mode", "shared"),
        freeze_encoder=cfg.model.freeze_encoder,
        pooling=cfg.model.pooling,
        dapt_embedding_store=dapt_embedding_store,
        competing_risk_config=OmegaConf.to_container(
            cfg.get("competing_risk", {}),
            resolve=True,
        ),
    )

    # ── Data ───────────────────────────────────────────────────────────
    data_module = MultiCohortContrastiveDataModule(
        cohort_configs=cohort_configs,
        outcome_configs=outcome_configs,
        predict_token_id=vocab["[CLS]"],
        batch_size=cfg.training.batch_size,
        num_workers=cfg.hardware.num_workers,
        require_all_configured_cells=require_all_configured_cells,
        require_min_followup_train=cfg.training.get(
            "require_min_followup_train", False
        ),
        max_len=encoder_hparams(encoder)["max_seqlen"],
        batch_sampling=cfg.training.get("batch_sampling", {}),
        logical_batch_size=cfg.training.get("logical_batch_size"),
        logical_val_batch_size=cfg.training.get("logical_val_batch_size"),
        numeric_value_control=numeric_value_control,
    )
    data_module.setup("fit")

    competing_risk_config = OmegaConf.to_container(
        cfg.get("competing_risk", {}),
        resolve=True,
    )
    if float(competing_risk_config.get("loss_weight", 0.0)) > 0:
        support = summarize_competing_risk_support(
            data_module.train_dataset,
            outcome_names,
            competing_risk_config["interval_boundaries_days"],
        )
        support_path = f"{model_save_dir}/competing_risk_interval_support.csv"
        with open(support_path, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(support[0]))
            writer.writeheader()
            writer.writerows(support)
        sparse = [
            row
            for row in support
            if row["n_valid"] > 0 and row["n_target"] + row["n_death"] < 5
        ]
        print(
            f"Competing-risk interval support written to {support_path}; "
            f"{len(sparse)} outcome-intervals have fewer than five observed events."
        )

    if dapt_embedding_store is not None:
        datasets = getattr(
            data_module.train_dataset,
            "datasets",
            [data_module.train_dataset],
        )
        training_subjects = {
            int(subject["subject_id"])
            for dataset in datasets
            for subject in dataset.subjects
        }
        covered = training_subjects.intersection(
            int(subject_id) for subject_id in dapt_embedding_store
        )
        coverage = len(covered) / max(len(training_subjects), 1)
        print(f"DAPT embedding-store training coverage: {coverage:.2%}")
        if (
            bool(cfg.training.get("require_dapt_embedding_store", False))
            and coverage < 0.99
        ):
            raise ValueError(
                "DAPT embedding-store coverage is below 99% for the OPERA "
                f"training population ({coverage:.2%}). Rebuild the store from "
                "the same resolved joint OPERA configuration."
            )

    # ── Lightning ──────────────────────────────────────────────────────
    lightning_module = OperaContrastiveModule(
        model=model,
        outcome_names=outcome_names,
        learning_rate=cfg.training.learning_rate,
        encoder_lr_multiplier=cfg.training.encoder_lr_multiplier,
        optimizer_epsilon=cfg.training.optimizer_epsilon,
        scheduler_warmup_epochs=cfg.training.scheduler_warmup_epochs,
        dapt_anchor_weight=cfg.model.get("dapt_anchor_weight", 0.2),
        gradient_cache=cfg.training.get("logical_batch_size") is not None,
        probe_every_n_epochs=cfg.training.get("probe_every_n_epochs", 1),
        enable_validation_probe=cfg.training.get("enable_validation_probe", True),
        log_per_outcome_metrics=cfg.training.get("log_per_outcome_metrics", True),
        checkpoint_metadata=_checkpoint_metadata(
            cfg,
            outcome_names,
            transfer_metadata,
            numeric_value_control,
        ),
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
        accumulate_grad_batches=(
            1
            if cfg.training.get("logical_batch_size") is not None
            else cfg.training.accumulate_grad_batches
        ),
        devices=cfg.hardware.num_devices,
        limit_val_batches=cfg.training.limit_val_batches,
        limit_train_batches=cfg.training.limit_train_batches,
        callbacks=[ckpt_callback, LearningRateMonitor(logging_interval="step")],
        logger=[logger],
        max_epochs=cfg.training.epochs,
        num_nodes=cfg.hardware.num_nodes,
        precision=cfg.hardware.precision,
        log_every_n_steps=int(cfg.training.get("log_every_n_steps", 10)),
    )

    trainer.fit(model=lightning_module, datamodule=data_module)
    save_checkpoint_metadata_sidecar(model_save_dir, lightning_module)
    mark_training_complete(model_save_dir, cfg)


if __name__ == "__main__":
    main()
