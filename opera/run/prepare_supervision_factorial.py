"""Materialize shared-BCE supervision controls from an approved release config.

No data are read and no training is launched. Existing experiments are untouched.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
from pathlib import Path

import yaml

from opera.functional.cohort_groups import ALL_EVALUATED_FINE, fine_to_grouped
from opera.functional.direct_cr_contract import validate_direct_cr_config

SELECTION_METRIC = "val/cr_horizon_bce_endpoint_group_macro"


def factorial_configs(
    base: dict, *, encoder_checkpoint: str, epochs: int = 30, seed: int = 42
) -> dict[str, dict]:
    """Build S, D, O and J with identical architecture and selection policy."""
    risk = base["competing_risk"]
    if (
        risk.get("objective") != "horizon_bce"
        or risk.get("readout_mode") != "shared"
        or risk.get("head_mode") != "linear"
        or base["model"].get("freeze_encoder", False)
        or base.get("source_opera_ckpt") not in (None, "")
    ):
        raise ValueError("Use the approved trainable shared linear horizon-BCE base")
    if not encoder_checkpoint or epochs < 1:
        raise ValueError(
            "An audited encoder checkpoint and positive epoch budget are required"
        )
    if base["cross_outcome"].get("aggregation") != "endpoint_group_macro":
        raise ValueError("Use the approved endpoint-group-macro shared baseline")
    validate_direct_cr_config(base)
    common = copy.deepcopy(base)
    common["dapt_ckpt"] = encoder_checkpoint
    common["seed"] = seed
    common.setdefault("hardware", {}).update(num_devices=1, num_nodes=1)
    common["competing_risk"].setdefault("curriculum", {})["enabled"] = False
    common["training"].update(
        {
            "epochs": epochs,
            "eval_monitor_metric": SELECTION_METRIC,
            "early_stopping_patience": None,
            "enable_validation_probe": False,
            "limit_val_batches": 1.0,
            "limit_train_batches": 1.0,
            "batch_sampling": {"type": "random", "seed": seed},
            "require_all_configured_cells": True,
            # Restarts continue the same fit from its own last.ckpt.
            "auto_resume": True,
        }
    )
    if (
        float(risk.get("contrastive_loss_weight", 0)) != 0
        or float(common["model"].get("dapt_anchor_weight", 0)) != 0
        or common.get("dapt_embedding_store") is not None
    ):
        raise ValueError(
            "Factorial requires direct BCE without contrastive or DAPT-prior losses"
        )

    def narrow(fine=None, outcome=None):
        config = copy.deepcopy(common)
        if fine is not None:
            parent = fine_to_grouped(fine)
            cohort = config["cohorts"][parent]
            cohort.update(cohort_fine_col="cohort_fine", cohort_fine_value=fine)
            config["cohorts"] = {parent: cohort}
        if outcome is not None:
            config["outcomes"] = {outcome: config["outcomes"][outcome]}
            config["cohorts"] = {
                name: cohort
                for name, cohort in config["cohorts"].items()
                if outcome not in cohort.get("exclude_outcomes", [])
            }
        else:
            # A disease-fixed model must not retain heads unavailable there.
            available = {
                name
                for name in config["outcomes"]
                if any(
                    name not in cohort.get("exclude_outcomes", [])
                    for cohort in config["cohorts"].values()
                )
            }
            config["outcomes"] = {
                name: value
                for name, value in config["outcomes"].items()
                if name in available
            }
        if not config["cohorts"] or not config["outcomes"]:
            return None
        selected = set(config["outcomes"])
        for key in ("endpoint_groups", "outcome_families"):
            config["cross_outcome"][key] = {
                group: [name for name in members if name in selected]
                for group, members in common["cross_outcome"].get(key, {}).items()
                if selected.intersection(members)
            }
        regime = "S" if fine and outcome else "O" if fine else "D" if outcome else "J"
        config["supervision_factorial"] = {
            "regime": regime,
            "fine_cohort": fine,
            "outcome": outcome,
            "initialization": "same_pretrained_encoder_no_pooled_supervised_weights",
            "selection": "full_tuning_horizon_then_grade_then_endpoint_group_mean_bce",
            "budget": "equal_epochs_not_equal_steps_or_compute",
        }
        validate_direct_cr_config(config)
        return config

    result = {"J_all_diseases_all_outcomes": narrow()}
    diseases = [
        fine
        for fine in ALL_EVALUATED_FINE
        if fine_to_grouped(fine) in common["cohorts"]
    ]
    for outcome in common["outcomes"]:
        config = narrow(outcome=outcome)
        if config is not None:
            result[f"D_all_diseases__{outcome}"] = config
    for fine in diseases:
        result[f"O_{fine}__all_outcomes"] = narrow(fine=fine)
        for outcome in common["outcomes"]:
            config = narrow(fine=fine, outcome=outcome)
            if config is not None:
                result[f"S_{fine}__{outcome}"] = config
    return {name: config for name, config in result.items() if config is not None}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-config", type=Path, required=True)
    parser.add_argument("--encoder-checkpoint", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"Use a fresh factorial directory: {args.output_dir}")
    raw = args.base_config.read_bytes()
    configs = factorial_configs(
        yaml.safe_load(raw),
        encoder_checkpoint=args.encoder_checkpoint,
        epochs=args.epochs,
        seed=args.seed,
    )
    args.output_dir.mkdir(parents=True)
    rows = []
    for name, config in configs.items():
        path = args.output_dir / f"{name}.yaml"
        path.write_text(
            "# @package _global_\n" + yaml.safe_dump(config, sort_keys=False)
        )
        rows.append(
            {
                "name": name,
                **config["supervision_factorial"],
                "config": str(path.resolve()),
                "config_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        )
    with (args.output_dir / "manifest.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    report = {
        "status": "configs_materialized_not_preflighted_or_launched",
        "base_config_sha256": hashlib.sha256(raw).hexdigest(),
        "encoder_checkpoint": args.encoder_checkpoint,
        "epochs": args.epochs,
        "seed": args.seed,
        "selection_metric": SELECTION_METRIC,
        "counts": {
            regime: sum(row["regime"] == regime for row in rows)
            for regime in ("S", "D", "O", "J")
        },
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
