"""Tuning-only inventory of completed approved-panel neural experiments.

Reads aggregate Lightning logs, resolved Hydra configs and selection inventories.
It neither loads checkpoints nor opens patient-level or held-out data.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import yaml

from opera.run.compare_approved_abspos_v2 import _run


PAIRS = (
    ("encoder_plain_shared", "fourier_plain_shared_bce", "time2vec_plain_shared_bce"),
    ("encoder_grouped_shared", "shared", "time2vec_grouped_shared_bce"),
    ("loss_fourier_shared", "fourier_plain_shared_bce", "shared"),
    ("loss_time2vec_shared", "time2vec_plain_shared_bce", "time2vec_grouped_shared_bce"),
    ("readout_family_trunks", "shared", "family_trunks"),
    ("encoder_family_trunks", "family_trunks", "time2vec_family_trunks"),
    ("readout_time2vec_family_trunks", "time2vec_grouped_shared_bce", "time2vec_family_trunks"),
    ("readout_additive_router", "shared", "router_additive_e2e"),
    ("encoder_additive_router", "router_additive_e2e", "time2vec_router_additive_e2e"),
    ("readout_time2vec_additive_router", "time2vec_grouped_shared_bce", "time2vec_router_additive_e2e"),
    ("readout_interacting_router", "shared", "router_interacting_e2e"),
    ("encoder_interacting_router", "router_interacting_e2e", "time2vec_router_interacting_e2e"),
    ("readout_time2vec_interacting_router", "time2vec_grouped_shared_bce", "time2vec_router_interacting_e2e"),
    ("router_query", "router_additive_e2e", "router_interacting_e2e"),
    ("time2vec_router_query", "time2vec_router_additive_e2e", "time2vec_router_interacting_e2e"),
    ("readout_outcome_attention", "shared", "outcome_attention"),
    ("readout_plain_factorized", "fourier_plain_shared_bce", "fourier_plain_factorized_bce"),
)


def _config(run: Path) -> dict:
    path = run / "hydra" / "config.yaml"
    if not path.is_file():
        raise ValueError(f"Missing resolved Hydra config: {path}")
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError(f"Invalid Hydra config: {path}")
    return config


def _contract(config: dict) -> dict:
    competing = config["competing_risk"]
    training = config["training"]
    cross = config["cross_outcome"]
    return {
        "cohorts": config["cohorts"],
        "outcomes": config["outcomes"],
        "sequence_data_dir": config["paths"]["sequence_data_dir"],
        "objective": competing["objective"],
        "interval_boundaries_days": competing["interval_boundaries_days"],
        "evaluation_horizons_days": training["evaluation_horizons_days"],
        "selection_metric": training["eval_monitor_metric"],
        "selection_min_cases": training["selection_min_cases"],
        "selection_min_controls": training["selection_min_controls"],
        "endpoint_groups": cross["endpoint_groups"],
        "outcome_families": cross["outcome_families"],
        "seed": config["seed"],
    }


def _run_summary(label: str, path: Path) -> dict:
    info = _run(path)
    config = _config(path)
    epochs = {int(k): v for k, v in info["per_epoch_endpoint_group_auc"].items()}
    tail = [epochs[k] for k in sorted(epochs)[-5:]]
    info.update(
        label=label,
        five_epoch_mean_auc=sum(tail) / len(tail),
        five_epoch_auc_range=max(tail) - min(tail),
        # Display aliases such as ``interactive_02`` must not turn a
        # Time2Vec run into a purported Fourier run in the exported summary.
        encoder=("scaled_time2vec" if path.name.startswith("time2vec_")
                 or label.startswith("time2vec_") else "fourier"),
        readout_mode=config["competing_risk"]["readout_mode"],
        head_mode=config["competing_risk"]["head_mode"],
        query_mode=config["competing_risk"].get("adapter_query_mode"),
        loss_aggregation=config["cross_outcome"]["aggregation"],
        adapter_initialization=config.get("adapter_initialization"),
    )
    return {"summary": info, "contract": _contract(config)}


def compare(runs: dict[str, Path]) -> dict:
    if not runs:
        raise ValueError("At least one run is required")
    parsed = {label: _run_summary(label, path) for label, path in runs.items()}
    reference_label = next(iter(parsed))
    reference = parsed[reference_label]
    inventory = reference["summary"]["inventory"]
    groups = reference["summary"]["selection_groups"]
    cells = reference["summary"]["selection_cells"]
    if groups is None or cells is None:
        raise ValueError(f"Missing selected tuning support in {reference_label}")
    for label, item in parsed.items():
        if item["contract"] != reference["contract"]:
            different = [key for key in reference["contract"]
                         if item["contract"][key] != reference["contract"][key]]
            raise ValueError(f"Training/evaluation contract differs for {label}: {different}")
        summary = item["summary"]
        if (summary["inventory"] != inventory
                or summary["selection_groups"] != groups
                or summary["selection_cells"] != cells):
            raise ValueError(f"Tuning selection support differs for {label}")
        if not math.isfinite(summary["best_endpoint_group_auc"]):
            raise ValueError(f"Nonfinite selection AUC for {label}")
    summaries = {label: item["summary"] for label, item in parsed.items()}
    common_epochs = set.intersection(*(
        set(summary["per_epoch_endpoint_group_auc"]) for summary in summaries.values()
    ))
    if not common_epochs:
        raise ValueError("No common completed tuning epoch across the selected runs")
    common_epoch = max(common_epochs, key=int)
    for summary in summaries.values():
        summary["common_epoch_auc"] = summary["per_epoch_endpoint_group_auc"][common_epoch]
    contrasts = []
    for question, baseline, candidate in PAIRS:
        if baseline not in summaries or candidate not in summaries:
            continue
        a, b = summaries[baseline], summaries[candidate]
        families = sorted(set(a["family_auc_at_best_epoch"]) &
                          set(b["family_auc_at_best_epoch"]))
        contrasts.append({
            "question": question,
            "baseline": baseline,
            "candidate": candidate,
            "best_auc_delta_candidate_minus_baseline": (
                b["best_endpoint_group_auc"] - a["best_endpoint_group_auc"]),
            "last_auc_delta": b["last_endpoint_group_auc"] - a["last_endpoint_group_auc"],
            "common_epoch_auc_delta": b["common_epoch_auc"] - a["common_epoch_auc"],
            "five_epoch_mean_auc_delta": b["five_epoch_mean_auc"] - a["five_epoch_mean_auc"],
            "family_auc_deltas_at_each_best_epoch": {
                family: b["family_auc_at_best_epoch"][family] -
                a["family_auc_at_best_epoch"][family] for family in families},
        })
    for summary in summaries.values():
        summary.pop("inventory")
    return {
        "status": "tuning_only_descriptive_single_seed",
        "selection_inventory_cells": len(inventory),
        "selection_groups": groups,
        "selection_cells": cells,
        "common_completed_epoch": int(common_epoch),
        "runs": summaries,
        "contrasts": contrasts,
        "interpretation": (
            "Best-epoch contrasts are descriptive and selection-biased; family values "
            "are measured at each arm's own best epoch. Confirm finalists with "
            "paired tuning predictions and replicated seeds; keep held-out closed."),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="append", required=True, metavar="LABEL=RUN_DIR")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    runs = {}
    for item in args.run:
        label, separator, raw_path = item.partition("=")
        if not separator or not label or not raw_path or label in runs:
            parser.error(f"Expected a unique LABEL=RUN_DIR, got {item!r}")
        runs[label] = Path(raw_path)
    result = compare(runs)
    payload = json.dumps(result, indent=2) + "\n"
    if args.output:
        if args.output.exists():
            raise FileExistsError(f"Refusing to overwrite {args.output}")
        args.output.write_text(payload, encoding="utf-8")
    print(payload, end="")


if __name__ == "__main__":
    main()
