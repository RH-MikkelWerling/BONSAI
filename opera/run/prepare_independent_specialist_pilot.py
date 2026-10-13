"""Prepare a DAPT-initialized disease/outcome specialist config."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import yaml

from opera.functional.cohort_groups import fine_to_grouped
from opera.functional.direct_cr_contract import validate_direct_cr_config


def specialist_pilot_config(
    base: dict, *, parent_cohort: str, fine_cohort: str, outcome: str,
    max_epochs: int = 5, label_cache_dir: str | None = None,
    full_run: bool = False,
) -> dict:
    """Narrow plain shared BCE to one cell without importing supervised weights."""
    if max_epochs < 1:
        raise ValueError("Pilot epochs must be positive")
    if fine_to_grouped(fine_cohort) != parent_cohort:
        raise ValueError("Fine cohort does not map to the stated grouped parent")
    if parent_cohort not in base["cohorts"] or outcome not in base["outcomes"]:
        raise ValueError("Specialist cell is absent from the approved base config")
    risk = base["competing_risk"]
    if (risk.get("objective") != "horizon_bce"
            or risk.get("readout_mode") != "shared"
            or risk.get("head_mode") != "linear"
            or base["model"].get("freeze_encoder", False)
            or base.get("source_opera_ckpt") not in (None, "")):
        raise ValueError("Pilot base must be DAPT-fresh, trainable, plain shared BCE")
    if outcome in base["cohorts"][parent_cohort].get("exclude_outcomes", []):
        raise ValueError("Outcome is excluded for the chosen parent cohort")
    groups = {
        name: [outcome] for name, members in base["cross_outcome"]["endpoint_groups"].items()
        if outcome in members
    }
    if len(groups) != 1:
        raise ValueError("Outcome must belong to exactly one approved endpoint group")
    result = copy.deepcopy(base)
    cohort = result["cohorts"][parent_cohort]
    cohort["cohort_fine_col"] = "cohort_fine"
    cohort["cohort_fine_value"] = fine_cohort
    result["cohorts"] = {parent_cohort: cohort}
    result["outcomes"] = {outcome: result["outcomes"][outcome]}
    result["cross_outcome"]["endpoint_groups"] = groups
    result["cross_outcome"]["outcome_families"] = {
        name: [outcome]
        for name, members in result["cross_outcome"].get("outcome_families", {}).items()
        if outcome in members
    }
    result["cross_outcome"]["aggregation"] = "macro"
    if full_run:
        # Full specialists select checkpoints by validation BCE. A singleton
        # endpoint-group AUC is only a report metric and can be undefined in
        # sparse tuning cells; it must not abort otherwise valid training.
        result["cross_outcome"]["endpoint_groups"] = {}
    result["competing_risk"].setdefault("curriculum", {})["enabled"] = False
    result["training"]["batch_sampling"] = {"type": "random", "seed": int(result.get("seed", 42))}
    result["training"]["epochs"] = max_epochs
    result["training"]["eval_monitor_metric"] = "val/loss"
    result["training"]["early_stopping_patience"] = None
    result["training"]["require_all_configured_cells"] = True
    result["paths"]["outcome_label_cache_dir"] = label_cache_dir
    result["specialist_run" if full_run else "specialist_pilot"] = {
        "purpose": (
            "independent_dapt_specialist_full_fit"
            if full_run else "runtime_only_not_primary_scientific_result"
        ),
        "fine_cohort": fine_cohort,
        "outcome": outcome,
        "initialization": "independent_dapt_encoder_no_pooled_supervised_weights",
    }
    validate_direct_cr_config(result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-config", type=Path, required=True)
    parser.add_argument("--parent-cohort", required=True)
    parser.add_argument("--fine-cohort", required=True)
    parser.add_argument("--outcome", required=True)
    parser.add_argument("--max-epochs", type=int, default=5)
    parser.add_argument("--full-run", action="store_true",
                        help="Mark the configuration as a full independent specialist fit.")
    parser.add_argument("--label-cache-dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Pilot config already exists: {args.output}")
    base = yaml.safe_load(args.base_config.read_text(encoding="utf-8"))
    config = specialist_pilot_config(
        base, parent_cohort=args.parent_cohort, fine_cohort=args.fine_cohort,
        outcome=args.outcome, max_epochs=args.max_epochs,
        label_cache_dir=(str(args.label_cache_dir.resolve()) if args.label_cache_dir else None),
        full_run=args.full_run,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "# @package _global_\n" + yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
    )
    key = "specialist_run" if args.full_run else "specialist_pilot"
    print(json.dumps({"status": "specialist_config_written", "path": str(args.output.resolve()),
                      "cell": config[key]}, indent=2))


if __name__ == "__main__":
    main()
