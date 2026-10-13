"""Compare two checkpoints' routes on exactly matched tuning patients and cells."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from opera.run.plot_router_cell_borrowing import (
    parse_cell, read_cell_routes, verify_route_provenance,
)


def route_matrix(frame: pd.DataFrame, names: list[str]) -> np.ndarray:
    disease = [name for name in names if name.startswith("disease:")]
    outcome = [name for name in names if name.startswith("outcome:")]
    rows = []
    for row in frame.itertuples():
        d, o = json.loads(row.disease_weights), json.loads(row.outcome_weights)
        if set(d) != set(disease) or set(o) != set(outcome):
            raise ValueError("Route source inventory differs from config")
        rows.append([row.shared_weight, *[d[name] for name in disease],
                     *[o[name] for name in outcome]])
    matrix = np.asarray(rows, dtype=float)
    if not np.isfinite(matrix).all() or (matrix < 0).any() or not np.allclose(matrix.sum(1), 1, atol=1e-3):
        raise ValueError("Invalid route weights")
    return matrix


def compare_cell_routes(clean: pd.DataFrame, loose: pd.DataFrame, names: list[str],
                        *, disease: str, outcome: str) -> tuple[dict, pd.DataFrame]:
    if clean.empty or loose.empty:
        raise ValueError("Both route exports must contain tuning patients")
    key = ["subject_id", "cohort", "outcome", "fine_index"]
    a = clean.sort_values(key).reset_index(drop=True)
    b = loose.sort_values(key).reset_index(drop=True)
    if not a[key].equals(b[key]):
        raise ValueError("Route exports have different tuning patient/target rows")
    x, y = route_matrix(a, names), route_matrix(b, names)
    patient_tv = .5 * np.abs(x - y).sum(1)
    mean_x, mean_y = x.mean(0), y.mean(0)
    off_home = np.asarray([i for i, name in enumerate(names)
                           if name not in {"shared", "disease:" + disease,
                                           "outcome:" + outcome}], dtype=int)
    left, right = mean_x[off_home], mean_y[off_home]
    normalized_tv = (.5 * np.abs(left / left.sum() - right / right.sum()).sum()
                     if left.sum() > 0 and right.sum() > 0 else float("nan"))
    summary = {"disease": disease, "outcome": outcome, "n_patients": len(a),
               "mean_paired_patient_route_tv": float(patient_tv.mean()),
               "p90_paired_patient_route_tv": float(np.quantile(patient_tv, .9)),
               "mean_route_tv": float(.5 * np.abs(mean_x - mean_y).sum()),
               "off_home_normalized_mean_route_tv": float(normalized_tv),
               "clean_shared": float(mean_x[0]), "loose_shared": float(mean_y[0]),
               "clean_patient_tv_from_cell_mean": float((.5 * np.abs(x - mean_x).sum(1)).mean()),
               "loose_patient_tv_from_cell_mean": float((.5 * np.abs(y - mean_y).sum(1)).mean())}
    sources = pd.DataFrame({"disease": disease, "outcome": outcome,
                            "source": names, "clean_mean_weight": mean_x,
                            "loose_mean_weight": mean_y,
                            "loose_minus_clean": mean_y - mean_x})
    return summary, sources


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clean-run", type=Path, required=True)
    parser.add_argument("--loose-run", type=Path, required=True)
    parser.add_argument("--clean-export-root", type=Path, required=True)
    parser.add_argument("--loose-export-root", type=Path, required=True)
    parser.add_argument("--cell", action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    clean_config = args.clean_run / "hydra/config.yaml"
    loose_config = args.loose_run / "hydra/config.yaml"
    clean_cfg = yaml.safe_load(clean_config.read_text(encoding="utf-8"))
    loose_cfg = yaml.safe_load(loose_config.read_text(encoding="utf-8"))
    disease_names = sorted(clean_cfg["competing_risk"]["fine_names"])
    outcome_names = sorted(clean_cfg["outcomes"])
    if disease_names != sorted(loose_cfg["competing_risk"]["fine_names"]) or outcome_names != sorted(loose_cfg["outcomes"]):
        raise ValueError("The two runs have different route source inventories")
    names = ["shared", *("disease:" + name for name in disease_names),
             *("outcome:" + name for name in outcome_names)]
    summaries, sources = [], []
    for label in args.cell:
        disease, outcome, index = parse_cell(label, clean_cfg)
        paths = [root / f"{disease}__{outcome}" / "source_route_weights.parquet"
                 for root in (args.clean_export_root, args.loose_export_root)]
        verify_route_provenance(paths[0], clean_config, args.clean_run / "best.ckpt")
        verify_route_provenance(paths[1], loose_config, args.loose_run / "best.ckpt")
        summary, source = compare_cell_routes(
            read_cell_routes(paths[0], fine_index=index, outcome=outcome),
            read_cell_routes(paths[1], fine_index=index, outcome=outcome),
            names, disease=disease, outcome=outcome,
        )
        summaries.append(summary)
        sources.append(source)
    args.output_dir.mkdir(parents=True)
    pd.DataFrame(summaries).to_csv(args.output_dir / "paired_route_similarity.csv", index=False)
    pd.concat(sources, ignore_index=True).to_csv(
        args.output_dir / "paired_source_weight_changes.csv", index=False)
    print(json.dumps({"status": "paired_tuning_route_comparison",
                      "cells": len(summaries), "output_dir": str(args.output_dir.resolve())}, indent=2))


if __name__ == "__main__":
    main()
