"""Prepare train/tuning outcome labels once for repeated neural runs (CPU only)."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from omegaconf import OmegaConf

from opera.modules.datamodules.MultiCohortContrastiveDataModule import (
    MultiCohortContrastiveDataModule,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--cohort", action="append", default=[])
    parser.add_argument("--outcome", action="append", default=[])
    parser.add_argument("--output-json", type=Path)
    args = parser.parse_args()
    cfg = OmegaConf.load(args.config)
    cohorts = OmegaConf.to_container(cfg.cohorts, resolve=True)
    outcomes = OmegaConf.to_container(cfg.outcomes, resolve=True)
    if args.cohort:
        unknown = set(args.cohort) - set(cohorts)
        if unknown:
            raise ValueError(f"Unknown cohorts: {sorted(unknown)}")
        cohorts = {name: cohorts[name] for name in args.cohort}
    if args.outcome:
        unknown = set(args.outcome) - set(outcomes)
        if unknown:
            raise ValueError(f"Unknown outcomes: {sorted(unknown)}")
        outcomes = {name: outcomes[name] for name in args.outcome}
    if not cohorts or not outcomes:
        raise ValueError("Cache warmup requires at least one cohort and outcome")
    module = MultiCohortContrastiveDataModule(
        cohort_configs=cohorts, outcome_configs=outcomes,
        predict_token_id=0, batch_size=1, num_workers=0,
        require_min_followup_train=bool(cfg.training.get("require_min_followup_train", False)),
        require_min_followup_val=bool(cfg.training.get("require_min_followup_val", False)),
        require_all_configured_cells=bool(cfg.training.get("require_all_configured_cells", False)),
        outcome_label_cache_dir=str(args.cache_dir.resolve()),
    )
    started = time.perf_counter()
    counts = []
    for cohort, cohort_cfg in cohorts.items():
        for split in ("train", "tuning"):
            records = module._load_outcomes_for_cohort(
                cohort, cohort_cfg, cohort_cfg["data_dir"], split,
            )
            counts.extend({
                "cohort": cohort, "outcome": outcome, "split": split,
                "n_records": len(subjects),
            } for outcome, subjects in records.items())
            print(f"Prepared {cohort} {split}: {len(records)} outcomes", flush=True)
    report = {
        "status": "train_tuning_outcome_labels_cached",
        "config": str(args.config.resolve()),
        "cache_dir": str(args.cache_dir.resolve()),
        "elapsed_seconds": time.perf_counter() - started,
        "n_cohorts": len(cohorts), "n_outcomes": len(outcomes),
        "n_entries": len(counts), "counts": counts,
        "held_out_loaded": False,
    }
    if args.output_json:
        if args.output_json.exists():
            raise FileExistsError(f"Report already exists: {args.output_json}")
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "counts"}, indent=2))


if __name__ == "__main__":
    main()
