"""Audit exact cohort-by-outcome support before rarity experiments.

The audit uses the same fixed-horizon, competing-event, cohort-membership and
IPCW contracts as survival finetuning.  Candidate tiers depend only on the
training and tuning splits; held-out event counts are reported for uncertainty
assessment but never used to select tasks.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
import yaml

from opera.evaluation.cohort_flow import eligibility_file_path
from opera.evaluation.cohorts import (
    build_evaluation_cohorts,
    population_subject_ids,
)
from opera.functional.ipcw import compute_ipcw_train_weights, summarize_ipcw_weights
from opera.run.generate_sweep_configs import load_registry


SPLITS = ("train", "tuning", "held_out")


def _expand(value: Any) -> Any:
    if isinstance(value, str):
        return os.path.expandvars(value)
    if isinstance(value, dict):
        return {key: _expand(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_expand(item) for item in value]
    return value


def _fine_cohorts(registry: dict[str, Any]) -> list[tuple[str, str]]:
    return [
        (grouped, fine)
        for grouped, group_cfg in registry["cohort_groups"].items()
        for fine in group_cfg["fine"]
    ]


def _excluded(registry: dict[str, Any], grouped: str, outcome: str) -> bool:
    return any(
        outcome in rule.get("outcomes", [])
        and grouped in rule.get("excluded_grouped", [])
        for rule in registry.get("availability_rules", {}).values()
    )


def _candidate_tier(row: dict[str, Any]) -> str:
    train_events = int(row.get("n_primary_events_train", 0))
    tuning_events = int(row.get("n_primary_events_tuning", 0))
    train_controls = int(row.get("n_ipcw_controls_train", 0))
    tuning_controls = int(row.get("n_ipcw_controls_tuning", 0))
    if train_events >= 30 and tuning_events >= 10 and min(train_controls, tuning_controls) >= 10:
        return "scarce_confirmatory"
    if train_events >= 10 and tuning_events >= 3 and min(train_controls, tuning_controls) >= 3:
        return "very_scarce_exploratory"
    return "non_estimable"


def audit_support(
    registry: dict[str, Any],
    *,
    outcomes: Iterable[str],
    horizon_days: int,
) -> pd.DataFrame:
    registry = _expand(registry)
    paths = registry["paths"]
    population_path = paths["cohort_membership_file"]
    data_dir = paths["shared_data_dir"]
    outcomes_dir = Path(paths["outcomes_dir"])
    death_name = registry["death_outcome"]
    death_path = outcomes_dir / f"{death_name}.parquet"
    containing_death = set(registry.get("outcomes_containing_death", [death_name]))
    horizon_hours = int(horizon_days) * 24
    outcome_cache: dict[str, pd.DataFrame] = {}
    death_frame = pd.read_parquet(death_path)
    rows: list[dict[str, Any]] = []

    for outcome in outcomes:
        outcome_path = outcomes_dir / f"{outcome}.parquet"
        outcome_cache[outcome] = pd.read_parquet(outcome_path)
        competing = None if outcome in containing_death else death_frame
        for grouped, fine in _fine_cohorts(registry):
            if _excluded(registry, grouped, outcome):
                continue
            eligibility = eligibility_file_path(
                data_dir,
                fine,
                outcome,
                {
                    "eligibility_file": registry.get("eligibility_files", {}).get(
                        outcome
                    )
                },
            )
            allowed = population_subject_ids(
                population_path,
                cohort_fine_col=registry["cohort_columns"]["fine"],
                cohort_fine_value=fine,
            )
            row: dict[str, Any] = {
                "cohort_grouped": grouped,
                "cohort": fine,
                "outcome": outcome,
                "horizon_days": int(horizon_days),
            }
            for split in SPLITS:
                cohorts = build_evaluation_cohorts(
                    outcome_cache[outcome],
                    split=split,
                    n_hours_start_include=1,
                    n_hours_end_include=horizon_hours,
                    competing_outcomes=competing,
                    eligibility=eligibility,
                    cohort=fine,
                    outcome_name=outcome,
                    allowed_subject_ids=allowed,
                )
                survival = cohorts.survival
                fixed = cohorts.fixed_horizon
                records = dict(survival.records)
                weights = compute_ipcw_train_weights(
                    records,
                    horizon_hours=horizon_hours,
                    estimand="cumulative_incidence",
                )
                ipcw = summarize_ipcw_weights(records, weights)
                survival_frame = survival.to_frame()
                row.update(
                    {
                        f"n_survival_{split}": len(records),
                        f"n_fixed_{split}": len(fixed.records),
                        f"n_primary_events_{split}": survival.n_events,
                        f"primary_event_rate_{split}": (
                            float(survival.n_events / len(records))
                            if records
                            else float("nan")
                        ),
                        f"n_competing_deaths_{split}": int(
                            (survival_frame.get("event", pd.Series(dtype=int)) == 2).sum()
                        ),
                        f"n_ipcw_cases_{split}": int(ipcw["n_cases_nonzero"]),
                        f"n_ipcw_controls_{split}": int(ipcw["n_controls_nonzero"]),
                        f"ipcw_effective_n_{split}": float(ipcw["effective_sample_size"]),
                        f"ipcw_effective_fraction_{split}": float(
                            ipcw["effective_sample_fraction"]
                        ),
                        f"ipcw_max_weight_{split}": float(ipcw["max_weight"]),
                    }
                )
            row["candidate_tier"] = _candidate_tier(row)
            row["small_cohort"] = int(row["n_survival_train"]) < 1000
            row["low_prevalence_train"] = (
                float(row["primary_event_rate_train"]) < 0.10
            )
            row["scarcity_profile"] = (
                "small_cohort_low_prevalence"
                if row["small_cohort"] and row["low_prevalence_train"]
                else "small_cohort_absolute_support"
                if row["small_cohort"]
                else "low_prevalence"
                if row["low_prevalence_train"]
                else "data_rich"
            )
            row["held_out_descriptive_support"] = (
                "adequate"
                if row["n_primary_events_held_out"] >= 10
                else "unstable"
            )
            rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit exact fine-cohort outcome support for rarity experiments."
    )
    parser.add_argument(
        "--registry",
        default="opera/configs/experiment_registry.yaml",
    )
    parser.add_argument("--outcomes", required=True, help="Comma-separated outcomes.")
    parser.add_argument("--horizon-days", type=int, default=90)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    registry = load_registry(args.registry)
    requested = [item.strip() for item in args.outcomes.split(",") if item.strip()]
    unknown = sorted(set(requested) - set(registry["outcomes"]))
    if unknown:
        raise ValueError(f"Unknown outcomes: {unknown}")
    table = audit_support(
        registry,
        outcomes=requested,
        horizon_days=args.horizon_days,
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(output_dir / "cohort_outcome_support.csv", index=False)
    candidates = table[
        (table["candidate_tier"] != "non_estimable") & table["small_cohort"]
    ].sort_values(
        ["candidate_tier", "n_primary_events_train", "cohort", "outcome"],
        ascending=[True, True, True, True],
    )
    candidates.to_csv(output_dir / "candidate_cells.csv", index=False)
    summary = {
        "horizon_days": args.horizon_days,
        "outcomes": requested,
        "n_cells": int(len(table)),
        "tier_counts": table["candidate_tier"].value_counts().to_dict(),
        "n_small_candidates": int(len(candidates)),
        "selection_uses_held_out_events": False,
    }
    (output_dir / "support_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(candidates[
        [
            "cohort",
            "outcome",
            "candidate_tier",
            "n_survival_train",
            "n_primary_events_train",
            "primary_event_rate_train",
            "n_primary_events_tuning",
            "n_primary_events_held_out",
            "ipcw_effective_n_train",
        ]
    ].to_string(index=False))
    print(f"Wrote support audit to {output_dir}")


if __name__ == "__main__":
    main()
