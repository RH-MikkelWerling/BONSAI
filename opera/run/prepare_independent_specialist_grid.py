"""Inventory approved fine-disease by outcome specialist fits, without labels."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import yaml

from opera.functional.cohort_groups import ALL_EVALUATED_FINE, fine_to_grouped


def planned_cells(config: dict, *, exclude_fine: set[str] | None = None) -> list[dict[str, str]]:
    exclude_fine = exclude_fine or set()
    # Plain shared BCE has no router source inventory (`fine_names`). Its
    # supervised fine-disease panel is defined by the canonical cohort map.
    diseases = list(config["competing_risk"].get("fine_names") or ALL_EVALUATED_FINE)
    outcomes = list(config["outcomes"])
    if len(diseases) != len(set(diseases)) or len(outcomes) != len(set(outcomes)):
        raise ValueError("Fine diseases and outcomes must be unique")
    if not exclude_fine <= set(diseases):
        raise ValueError(f"Unknown excluded fine diseases: {sorted(exclude_fine - set(diseases))}")
    cells = []
    for fine in diseases:
        if fine in exclude_fine:
            continue
        grouped = fine_to_grouped(fine)
        if grouped not in config["cohorts"]:
            raise ValueError(f"Fine disease has no approved grouped cohort: {fine}")
        excluded_outcomes = set(config["cohorts"][grouped].get("exclude_outcomes", []))
        for outcome in outcomes:
            if outcome not in excluded_outcomes:
                cells.append({"fine_cohort": fine, "parent_cohort": grouped,
                              "outcome": outcome, "status": "approved_to_attempt_not_guaranteed_estimable"})
    return cells


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-config", type=Path, required=True)
    parser.add_argument("--exclude-fine", action="append", default=[])
    parser.add_argument("--shard-count", type=int, default=1,
                        help="Number of non-overlapping GPU queues to create.")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"Specialist grid already exists: {args.output_dir}")
    if args.shard_count < 1:
        raise ValueError("Shard count must be positive")
    raw = args.base_config.read_bytes()
    config = yaml.safe_load(raw)
    if (config["competing_risk"].get("objective") != "horizon_bce"
            or config["competing_risk"].get("readout_mode") != "shared"
            or config["competing_risk"].get("head_mode") != "linear"):
        raise ValueError("Base must be the approved plain shared horizon-BCE specialist template")
    cells = planned_cells(config, exclude_fine=set(args.exclude_fine))
    args.output_dir.mkdir(parents=True)
    fields = ["fine_cohort", "parent_cohort", "outcome", "status"]
    def write_manifest(path: Path, rows: list[dict[str, str]]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
    write_manifest(args.output_dir / "specialist_manifest.csv", cells)
    # Round-robin over approved cells balances counts and is stable for a
    # fixed manifest; distinct shards never share a disease/outcome fit.
    shards = [cells[index::args.shard_count] for index in range(args.shard_count)]
    for index, rows in enumerate(shards):
        write_manifest(args.output_dir / f"shard_{index:02d}_of_{args.shard_count:02d}" /
                       "specialist_manifest.csv", rows)
    report = {"status": "specialist_grid_planned_not_launched",
              "base_config": str(args.base_config.resolve()),
              "base_config_sha256": hashlib.sha256(raw).hexdigest(),
              "excluded_fine": sorted(set(args.exclude_fine)),
              "n_models_to_attempt": len(cells),
              "shard_count": args.shard_count,
              "models_per_shard": [len(rows) for rows in shards],
              "n_fine_diseases": len({cell["fine_cohort"] for cell in cells}),
              "selection": "single-outcome validation loss; 30 epochs; independent DAPT initialization",
              "caution": "Approved cells are not necessarily estimable at every horizon; classify support before final comparison."}
    (args.output_dir / "manifest_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
