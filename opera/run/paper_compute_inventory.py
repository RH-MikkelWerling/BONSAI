"""Per-fit compute inventory from the GPU workers' attempt records.

Each worker attempt appends wall-clock time, host and GPU to the fit's
``worker_attempts.jsonl``. This sums them per fit, keeping the bundle's
scientific identity (phase, regime, origin, seed, paper model aliases), so
GPU hours can be reported per supervision scope and compared with
performance. Interrupted attempts that were resumed count towards the fit.

    python -m opera.run.paper_compute_inventory --bundle-dir ... --runs-dir ... --output ...
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from opera.run.paper_prediction_queue import bundle_runs, canonical_models
from opera.run.paper_worker import ATTEMPTS


def inventory(bundle_dirs, runs_dir):
    rows = []
    for bundle_dir in bundle_dirs:
        if not (Path(bundle_dir) / "bundle.json").is_file():
            continue
        bundle, entries = bundle_runs(bundle_dir)
        for row, _, config in entries:
            run = Path(runs_dir) / row["run_directory"]
            attempts = []
            path = run / ATTEMPTS
            if path.is_file():
                attempts = [json.loads(line) for line in path.read_text().splitlines() if line]
            wall = sum(a["wall_seconds"] for a in attempts)
            rows.append({
                "name": row["name"],
                "phase": row["phase"],
                "regime": row["regime"],
                "origin": row["origin"],
                "seed": row["seed"],
                "scientific_hash": row["scientific_hash"],
                "paper_models": ",".join(canonical_models(row, config, bundle)),
                "completed": (run / "training_complete.json").is_file(),
                "n_attempts": len(attempts),
                "gpu_hours": wall / 3600.0,
                "completed_attempt_hours": sum(
                    a["wall_seconds"] for a in attempts if a["status"] == "completed"
                ) / 3600.0,
                "gpus": ",".join(sorted({a["gpu"] for a in attempts if a.get("gpu")})),
            })
    frame = pd.DataFrame(rows).drop_duplicates("scientific_hash")
    by_scope = (frame.groupby(["origin", "regime", "phase"], as_index=False)
                .agg(fits=("name", "size"), completed=("completed", "sum"),
                     gpu_hours=("gpu_hours", "sum")))
    return frame, by_scope


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--bundle-dir", type=Path, action="append", required=True)
    parser.add_argument("--runs-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    frame, by_scope = inventory(args.bundle_dir, args.runs_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output_dir / "compute_per_fit.csv", index=False)
    by_scope.to_csv(args.output_dir / "compute_by_scope.csv", index=False)
    print(by_scope.to_string(index=False))


if __name__ == "__main__":
    main()
