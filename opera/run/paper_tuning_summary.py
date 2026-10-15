"""Collect matched tuning checkpoint scores; never select on held-out outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from bonsai.functional.checkpointing import verified_training_selection
from opera.run.paper_prediction_queue import bundle_runs


def summarize(bundle_dir, runs_dir):
    _, rows = bundle_runs(bundle_dir)
    summaries = []
    for row, _, config in rows:
        run = Path(runs_dir) / row["run_directory"]
        metric = config["training"]["eval_monitor_metric"]
        record = {
            "name": row["name"],
            "phase": row["phase"],
            "regime": row["regime"],
            "origin": row["origin"],
            "seed": row["seed"],
            "scientific_hash": row["scientific_hash"],
            "metric": metric,
            "readout": config["competing_risk"]["readout_mode"],
            "score": np.nan,
            "status": "training_incomplete",
        }
        for key in (
            "adapter_source_mode",
            "adapter_bottleneck",
            "adapter_query_mode",
            "adapter_query_hidden_dim",
        ):
            record[key] = config["competing_risk"].get(key)
        # The score is the one the completed fit used to select best.ckpt.
        # Metric logs are not scanned: they can include abandoned attempts.
        try:
            selection = verified_training_selection(run)
        except ValueError as error:
            record["status"] = str(error)
        else:
            if selection["monitor"] != metric:
                record["status"] = "selection_monitor_differs_from_config"
            else:
                record.update(
                    status="complete_tuning_score",
                    score=selection["best_model_score"],
                    selection_mode=selection["mode"],
                    best_epoch=selection["best_epoch"],
                    best_global_step=selection["best_global_step"],
                    best_checkpoint_sha256=selection["best_checkpoint_sha256"],
                )
        summaries.append(record)
    return pd.DataFrame(summaries)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle-dir", type=Path, required=True)
    parser.add_argument("--runs-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError("Use a fresh tuning-summary directory")
    table = summarize(args.bundle_dir, args.runs_dir)
    args.output_dir.mkdir(parents=True)
    table.to_csv(args.output_dir / "tuning_scores.csv", index=False)
    screen = table.loc[table.regime.eq("J") & table.origin.eq("L")]
    screen.to_csv(args.output_dir / "matched_readout_scores.csv", index=False)
    (args.output_dir / "README.txt").write_text(
        "Only matched full-panel J/readout rows are comparable for method screening.\nDo not rank scores across different supervision scopes or target cells.\nNo router winner or decision lock is inferred automatically.\n"
    )
    print(
        json.dumps(
            {
                "rows": len(table),
                "matched_readout_rows": len(screen),
                "held_out_read": False,
            }
        )
    )


if __name__ == "__main__":
    main()
