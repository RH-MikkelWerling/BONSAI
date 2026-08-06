"""Generate production Cox/net-risk/CIF sweep configs from one locked registry."""

from __future__ import annotations

import argparse
import copy
from pathlib import Path
from typing import Any
import re

import yaml


DEFAULT_REGISTRY = Path("opera/configs/experiment_registry.yaml")
DEFAULT_OUTPUT_DIR = Path("opera/configs/generated")


class _NoAliasDumper(yaml.SafeDumper):
    """Keep generated YAML reviewable by avoiding anchors for repeated lists."""

    def ignore_aliases(self, data: object) -> bool:
        return True


def _hydra_environment(value: Any) -> Any:
    """Translate shell-style registry variables for Hydra-composed configs."""
    if isinstance(value, str):
        return re.sub(r"\$\{([A-Z][A-Z0-9_]*)\}", r"${oc.env:\1}", value)
    if isinstance(value, dict):
        return {key: _hydra_environment(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_hydra_environment(item) for item in value]
    return value


def load_registry(path: str | Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as handle:
        registry = yaml.safe_load(handle) or {}
    validate_registry(registry)
    return registry


def validate_registry(registry: dict[str, Any]) -> None:
    required = {
        "horizons_days",
        "death_outcome",
        "cohort_groups",
        "outcomes",
        "outcome_families",
        "model_variants",
    }
    missing = required - set(registry)
    if missing:
        raise ValueError(f"Experiment registry is missing: {sorted(missing)}")

    outcomes = list(registry["outcomes"])
    if len(outcomes) != len(set(outcomes)):
        raise ValueError("Experiment registry contains duplicate outcome names.")
    if registry["death_outcome"] not in outcomes:
        raise ValueError("death_outcome must name an outcome in the registry.")

    family_members = [
        outcome
        for members in registry["outcome_families"].values()
        for outcome in members
    ]
    duplicate_family_members = sorted(
        {name for name in family_members if family_members.count(name) > 1}
    )
    if duplicate_family_members:
        raise ValueError(
            f"Outcomes occur in multiple families: {duplicate_family_members}"
        )
    missing_families = sorted(set(outcomes) - set(family_members))
    unknown_family_members = sorted(set(family_members) - set(outcomes))
    if missing_families or unknown_family_members:
        raise ValueError(
            "Outcome-family coverage mismatch; "
            f"missing={missing_families}, unknown={unknown_family_members}"
        )

    horizons = registry["horizons_days"]
    if (
        not isinstance(horizons, list)
        or not horizons
        or any(not isinstance(value, int) or value <= 0 for value in horizons)
        or horizons != sorted(set(horizons))
    ):
        raise ValueError("horizons_days must be unique increasing positive integers.")

    groups = registry["cohort_groups"]
    fine_names: list[str] = []
    for grouped, config in groups.items():
        if not config.get("slug") or not config.get("fine"):
            raise ValueError(f"Cohort group {grouped!r} needs slug and fine mappings.")
        fine_names.extend(config["fine"])
    if len(fine_names) != len(set(fine_names)):
        raise ValueError("A fine cohort is assigned to more than one grouped cohort.")

    for rule_name, rule in registry.get("availability_rules", {}).items():
        unknown_outcomes = sorted(set(rule.get("outcomes", [])) - set(outcomes))
        unknown_groups = sorted(set(rule.get("excluded_grouped", [])) - set(groups))
        if unknown_outcomes or unknown_groups:
            raise ValueError(
                f"Availability rule {rule_name!r} has unknown outcomes/groups: "
                f"{unknown_outcomes}/{unknown_groups}"
            )


def _excluded_outcomes_by_group(registry: dict[str, Any]) -> dict[str, set[str]]:
    result = {name: set() for name in registry["cohort_groups"]}
    for rule in registry.get("availability_rules", {}).values():
        for grouped in rule.get("excluded_grouped", []):
            result[grouped].update(rule.get("outcomes", []))
    return result


def _cohorts(registry: dict[str, Any], level: str) -> dict[str, dict[str, Any]]:
    if level not in {"grouped", "fine"}:
        raise ValueError("level must be grouped or fine")
    data_dir = registry["paths"]["shared_data_dir"]
    population_file = registry["paths"]["cohort_membership_file"]
    exclusions = _excluded_outcomes_by_group(registry)
    result: dict[str, dict[str, Any]] = {}
    for grouped, group_cfg in registry["cohort_groups"].items():
        common = {
            "data_dir": data_dir,
            "population_file": population_file,
            # Display-only metadata for plot/legend grouping (natural-rarity
            # figure). Never affects training/eval population or checkpoint
            # selection — that's cohort_fine_col/cohort_fine_value below.
            "clinical_group": grouped,
            "ipi_score_col": None,
            "registry_start_date": None,
        }
        if exclusions[grouped]:
            common["exclude_outcomes"] = sorted(exclusions[grouped])
        if level == "grouped":
            result[grouped] = {
                **common,
                "cohort_fine_col": registry["cohort_columns"]["grouped"],
                "cohort_fine_value": grouped,
            }
            continue
        for fine in group_cfg["fine"]:
            result[fine] = {
                **common,
                "cohort_fine_col": registry["cohort_columns"]["fine"],
                "cohort_fine_value": fine,
            }
    return result


def _outcomes(
    registry: dict[str, Any], *, training_mode: str, horizon_days: int | None
) -> dict[str, dict[str, Any]]:
    if training_mode in {"cox", "cox_exact_cached"} and horizon_days is not None:
        raise ValueError("Cox generation does not take a fixed horizon.")
    if training_mode in {"ipcw_bce", "ipcw_cif_bce"} and horizon_days is None:
        raise ValueError("IPCW-BCE generation requires a fixed horizon.")
    death = registry["death_outcome"]
    result = {}
    eligibility_files = registry.get("eligibility_files", {})
    for outcome in registry["outcomes"]:
        config: dict[str, Any] = {
            "outcome_file": f"{registry['paths']['outcomes_dir']}/{outcome}.parquet",
            "n_hours_start_include": 1,
            "n_hours_end_include": (
                None if horizon_days is None else int(horizon_days) * 24
            ),
            "registry_start_date": None,
        }
        eligibility_file = eligibility_files.get(outcome)
        if eligibility_file:
            config["eligibility_file"] = eligibility_file
        if outcome != death:
            config["competing_outcome_path"] = (
                f"{registry['paths']['outcomes_dir']}/{death}.parquet"
            )
        result[outcome] = config
    return result


def build_sweep_config(
    registry: dict[str, Any],
    *,
    level: str,
    training_mode: str,
    horizon_days: int | None,
) -> dict[str, Any]:
    if training_mode in {"cox", "cox_exact_cached"}:
        objective = "cox"
    elif training_mode == "ipcw_cif_bce":
        objective = f"ipcw_cif_{horizon_days}d"
    else:
        objective = f"ipcw_{horizon_days}d"
    variants = {
        name: {**config, "training_mode": training_mode}
        for name, config in registry["model_variants"].items()
    }
    return {
        "analysis_level": level,
        "output_dir": f"{registry['paths']['output_root']}/{level}/{objective}",
        "finetune_base_config": "opera/configs/survival_finetune.yaml",
        "seeds": list(registry["seeds"]),
        "rarity_mode": "none",
        "cohorts": _cohorts(registry, level),
        "outcomes": _outcomes(
            registry, training_mode=training_mode, horizon_days=horizon_days
        ),
        "model_variants": variants,
    }


def _adaptation_outcomes(registry: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Full survival endpoint panel shared by OPERA and MOL adaptation."""
    return _outcomes(registry, training_mode="cox", horizon_days=None)


def build_joint_opera_config(registry: dict[str, Any]) -> dict[str, Any]:
    """Shared-data multi-cohort OPERA configuration generated from the registry."""
    cohorts = _cohorts(registry, "grouped")
    return {
        "defaults": ["/core/base_train@", "/hardware/1gpu6cpu@hardware", "_self_"],
        "hydra": {"searchpath": ["file://${oc.env:BONSAI_CONFIG_PATH}"]},
        "dataset": "daly_care_joint_opera",
        "dapt_ckpt": "${oc.env:BONSAI_CHECKPOINT_ROOT}/daly_care_pretrain/best.ckpt",
        "dapt_embedding_store": (
            "${oc.env:BONSAI_CHECKPOINT_ROOT}/daly_care_pretrain/"
            "opera_reference_mean_last_128.pt"
        ),
        # Used only by build_dapt_embedding_store; harmless to training runners.
        "output_path": None,
        "overwrite": False,
        "paths": {
            "vocabulary": f"{registry['paths']['shared_data_dir']}/vocabulary.pt"
        },
        "cohorts": cohorts,
        "outcomes": _adaptation_outcomes(registry),
        "model": {
            "projection_hidden_dim": 256,
            "projection_dim": 128,
            "temperature": 0.07,
            "km_time_scale": 0.25,
            # First-pass OPERA adaptation: preserve pretrained content geometry
            # only weakly, without using embedding proximity to reweight pairs.
            "dapt_lambda_floor": 1.0,
            "dapt_anchor_weight": 0.02,
            "competing_event_handling": "exclude",
            "competing_event_weight": 0.0,
            "effective_pair_normalization": False,
            "projection_mode": "shared",
            "freeze_encoder": False,
            "pooling": "mean_last_128",
        },
        "competing_risk": {
            "loss_weight": 1.0,
            "contrastive_loss_weight": 1.0,
            "interval_boundaries_days": [
                3,
                7,
                14,
                30,
                60,
                90,
                180,
                365,
                730,
                1460,
            ],
            "time_scale_days": 365.25,
            "initial_log_hazard": -2.3,
            "smoothness_weight": 0.01,
            "no_competing_outcomes": ["overall_survival"],
        },
        "cross_outcome": {
            "weighter": "uniform",
            "aggregation": "hierarchical_support",
            "class_balanced": False,
            "outcome_families": registry["outcome_families"],
            "family_weights": {
                "Disease control & survival": 2.0,
                "Treatment trajectory": 2.0,
            },
            "support_tau_locations": 100.0,
            "contrastive_term": "kl",
            "outcome_scale_mode": "none",
        },
        "training": {
            "require_all_configured_cells": True,
            "require_min_followup_train": False,
            "require_dapt_embedding_store": True,
            "batch_size": 8,
            "logical_batch_size": 256,
            "logical_val_batch_size": 256,
            "batch_sampling": {"type": "random"},
            "accumulate_grad_batches": 1,
            "epochs": 20,
            "learning_rate": 5e-5,
            "encoder_lr_multiplier": 0.1,
            "optimizer_epsilon": 1e-6,
            "scheduler_warmup_epochs": 2,
            "log_every_n_steps": 10,
            "probe_every_n_epochs": 5,
            "enable_validation_probe": False,
            "log_per_outcome_metrics": False,
            "limit_val_batches": 1.0,
            "limit_train_batches": 1.0,
        },
    }


def build_joint_opera_ablation_config(
    registry: dict[str, Any], *, projection_mode: str, initial_kl_scaling: bool
) -> dict[str, Any]:
    """Build a diagnostic-gated OPERA projection/scaling ablation."""
    config = copy.deepcopy(build_joint_opera_config(registry))
    config["model"]["projection_mode"] = projection_mode
    if initial_kl_scaling:
        config["cross_outcome"]["outcome_scale_mode"] = "initial_kl"
        config["cross_outcome"]["outcome_reference_scale_file"] = (
            "${oc.env:OPERA_KL_REFERENCE_FILE}"
        )
    return config


def build_direct_cr_config(
    registry: dict[str, Any], *, family_trunks: bool = False, curriculum: bool = False
) -> dict[str, Any]:
    """Direct multi-outcome competing-risk adaptation ablation."""
    config = copy.deepcopy(build_joint_opera_config(registry))
    config["dataset"] = "daly_care_direct_competing_risk"
    config["competing_risk"]["contrastive_loss_weight"] = 0.0
    config["competing_risk"]["head_mode"] = (
        "family_trunks" if family_trunks else "linear"
    )
    config["competing_risk"]["family_trunk_hidden_dim"] = 128
    config["competing_risk"]["family_trunk_dropout"] = 0.1
    config["competing_risk"]["curriculum"] = {
        "enabled": bool(curriculum),
        # Rank outcomes using training-split event-location support. This avoids
        # privileging hand-picked clinical families and preserves every family.
        "n_support_tiers": 5,
        "warmup_epochs": 1,
        "stage_epochs": 1,
        "ramp_epochs": 2,
    }
    config["training"].update(
        {
            "epochs": 12,
            "learning_rate": 2e-4,
            "encoder_lr_multiplier": 0.1,
            "scheduler_warmup_epochs": 1,
            "enable_validation_probe": False,
            "log_per_outcome_metrics": False,
        }
    )
    return config


def build_multi_outcome_config(registry: dict[str, Any]) -> dict[str, Any]:
    """Shared-data full-panel direct multi-outcome ablation configuration."""
    return {
        "defaults": ["/core/base_train@", "/hardware/1gpu6cpu@hardware", "_self_"],
        "hydra": {"searchpath": ["file://${oc.env:BONSAI_CONFIG_PATH}"]},
        "dataset": "daly_care_multi_outcome",
        "dapt_ckpt": "${oc.env:BONSAI_CHECKPOINT_ROOT}/dapt/best.ckpt",
        "paths": {
            "dir": registry["paths"]["shared_data_dir"],
            "train_split": "${paths.dir}/subject_data_train.pt",
            "val_split": "${paths.dir}/subject_data_tuning.pt",
            "vocabulary": "${paths.dir}/vocabulary.pt",
            "population": "${oc.env:BONSAI_COHORT_MEMBERSHIP}",
        },
        "outcomes": {
            name: {**cfg, "path": cfg.pop("outcome_file")}
            for name, cfg in _adaptation_outcomes(registry).items()
        },
        "model": {
            "head_hidden_dim": 128,
            "head_dropout": 0.1,
            "freeze_encoder": False,
            "pooling": "cls_last",
            "weighting": "equal",
        },
        "training": {
            "batch_size": 64,
            "accumulate_grad_batches": 2,
            "epochs": 20,
            "learning_rate": 5e-5,
            "encoder_lr_multiplier": 0.1,
            "optimizer_epsilon": 1e-6,
            "scheduler_warmup_epochs": 2,
            "early_stopping_patience": 5,
            "limit_val_batches": 1.0,
            "limit_train_batches": 1.0,
        },
    }


def generate_configs(
    registry_path: str | Path = DEFAULT_REGISTRY,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
) -> list[Path]:
    registry = load_registry(registry_path)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    objectives = (
        [("cox_exact_cached", None)]
        + [("ipcw_bce", horizon) for horizon in registry["horizons_days"]]
        + [("ipcw_cif_bce", horizon) for horizon in registry["horizons_days"]]
    )
    for level in ("grouped", "fine"):
        for training_mode, horizon in objectives:
            if horizon is None:
                label = "cox"
            elif training_mode == "ipcw_cif_bce":
                label = f"ipcw_cif_{horizon}d"
            else:
                label = f"ipcw_{horizon}d"
            path = output / f"{level}_{label}.yaml"
            config = build_sweep_config(
                registry,
                level=level,
                training_mode=training_mode,
                horizon_days=horizon,
            )
            path.write_text(
                yaml.dump(
                    config,
                    Dumper=_NoAliasDumper,
                    sort_keys=False,
                    allow_unicode=True,
                ),
                encoding="utf-8",
            )
            written.append(path)

    family_path = output / "outcome_families.yaml"
    family_path.write_text(
        yaml.dump(
            {"outcome_families": registry["outcome_families"]},
            Dumper=_NoAliasDumper,
            sort_keys=False,
            allow_unicode=True,
        ),
        encoding="utf-8",
    )
    written.append(family_path)
    for name, config in {
        "joint_opera_full_panel.yaml": build_joint_opera_config(registry),
        "joint_opera_initial_kl.yaml": build_joint_opera_ablation_config(
            registry, projection_mode="shared", initial_kl_scaling=True
        ),
        "joint_opera_family_initial_kl.yaml": build_joint_opera_ablation_config(
            registry, projection_mode="family", initial_kl_scaling=True
        ),
        "direct_cr_full_panel.yaml": build_direct_cr_config(registry),
        "direct_cr_family_trunks.yaml": build_direct_cr_config(
            registry, family_trunks=True
        ),
        "direct_cr_curriculum.yaml": build_direct_cr_config(
            registry, curriculum=True
        ),
        "direct_cr_family_trunks_curriculum.yaml": build_direct_cr_config(
            registry, family_trunks=True, curriculum=True
        ),
        "multi_outcome_full_panel.yaml": build_multi_outcome_config(registry),
    }.items():
        path = output / name
        path.write_text(
            "# @package _global_\n"
            + yaml.dump(
                _hydra_environment(config), Dumper=_NoAliasDumper, sort_keys=False
            ),
            encoding="utf-8",
        )
        written.append(path)
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", default=str(DEFAULT_REGISTRY))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    args = parser.parse_args()
    written = generate_configs(args.registry, args.output_dir)
    print(f"Generated {len(written)} files in {Path(args.output_dir)}")


if __name__ == "__main__":
    main()
