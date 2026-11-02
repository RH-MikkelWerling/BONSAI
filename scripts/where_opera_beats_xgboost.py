"""Which disease-outcome cells does OPERA win against per-task XGBoost, and why?

Per-cell paired AUC differences pooled over horizons (the analysis's
contrasts_all_horizons.csv), joined to train-only cell metadata (training
events, patients, endpoint group, lineage), each model's own AUC in the cell
and, optionally, each disease's recorded history before treatment. Prints
summaries by stratum, a precision-weighted regression of the cell gain on cell
characteristics, and the cells won or lost with confidence; writes the tables
as CSV. Aggregates only (cells, never patients). Tuning split by default.

    .venv-v100/bin/python scripts/where_opera_beats_xgboost.py \
        --analysis-root data/paper_256_v3/analysis/tuning_1102 \
        --cell-metadata <OPERA_PAPER_CELL_METADATA> \
        [--patient-metadata <OPERA_PAPER_PATIENT_METADATA>] \
        --output-dir data/paper_256_v3/analysis/tuning_1102/where_opera_wins
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

# OPERA vs per-task XGBoost first; the joint model and pooled XGBoost beside it.
CONTRASTS = {"O1": "OPERA - XGBoost per task", "T2": "Joint - XGBoost per task",
             "O6": "OPERA - XGBoost pooled over diseases"}
EVENT_BINS = [-1, 9, 19, 49, 99, 199, 499, np.inf]
EVENT_LABELS = ["<10", "10-19", "20-49", "50-99", "100-199", "200-499", "500+"]
AUC_BINS = [0, 0.6, 0.7, 0.8, 0.9, 1.0]
AUC_LABELS = ["<0.6", "0.6-0.7", "0.7-0.8", "0.8-0.9", ">=0.9"]


def cell_table(root: Path, metadata: Path, contrast: str, split: str, patients: Path | None):
    analysis = root / "prediction_analysis"
    cells = pd.read_csv(analysis / "contrasts_all_horizons.csv", low_memory=False)
    cells = cells.loc[(cells.split == split) & (cells.metric == "auc") & (cells.contrast == contrast)
                      & cells.gain.notna()].copy()
    if cells.empty:
        return cells
    meta = pd.read_csv(metadata, low_memory=False)
    meta = meta.loc[meta.horizon_days == 365, ["cohort", "outcome", "cohort_group", "endpoint_group",
                                               "n_events_train", *(["n_patients_train"]
                                                                   if "n_patients_train" in meta else [])]]
    cells = cells.merge(meta, on=["cohort", "outcome"], how="left")
    estimates = pd.read_csv(analysis / "estimates.csv", low_memory=False)
    estimates = estimates.loc[(estimates.split == split) & (estimates.metric == "auc")]
    level = (estimates.loc[estimates.model.isin(["X", "OPERA", "J_L", "XP"])]
             .groupby(["cohort", "outcome", "model"]).estimate.mean().unstack("model")
             .add_prefix("auc_").reset_index())
    cases = (estimates.loc[(estimates.horizon_days == 365) & (estimates.model == "X")]
             [["cohort", "outcome", "n_cases"]].rename(columns={"n_cases": "tuning_cases_1y"}))
    cells = cells.merge(level, on=["cohort", "outcome"], how="left").merge(
        cases, on=["cohort", "outcome"], how="left")
    if patients is not None:
        with open(patients, "rb") as handle:
            magic = handle.read(4)
        history = (pd.read_parquet(patients) if magic == b"PAR1"
                   else pd.read_pickle(patients) if magic[:1] == b"\x80"
                   else pd.read_csv(patients, low_memory=False))  # whatever the suffix says
        disease = "cohort_fine" if "cohort_fine" in history else "cohort"
        per_disease = history.groupby(disease)[["history_event_count", "laboratory_event_count"]].median()
        per_disease.columns = ["median_history_events", "median_lab_results"]
        cells = cells.merge(per_disease, left_on="cohort", right_index=True, how="left")
    cells["events_bin"] = pd.cut(cells.n_events_train, EVENT_BINS, labels=EVENT_LABELS)
    if "auc_X" in cells:
        cells["xgboost_auc_bin"] = pd.cut(cells.auc_X, AUC_BINS, labels=AUC_LABELS, include_lowest=True)
    cells["weight"] = 1 / np.square(cells.difference_se.clip(lower=1e-3))
    return cells


def by_stratum(cells: pd.DataFrame, column: str) -> pd.DataFrame:
    def summary(group):
        w = group.weight
        mean = float(np.sum(w * group.gain) / np.sum(w))
        return pd.Series({"cells": len(group), "share_better": float((group.gain > 0).mean()),
                          "confident_wins": int((group.lower > 0).sum()),
                          "confident_losses": int((group.upper < 0).sum()),
                          "median_gain": float(group.gain.median()),
                          "weighted_mean_gain": mean, "weighted_se": float(np.sqrt(1 / np.sum(w)))})
    rows = {key: summary(group) for key, group in cells.groupby(column, observed=True)}
    return pd.DataFrame(rows).T.rename_axis(column).sort_values("weighted_mean_gain", ascending=False)


def weighted_regression(cells: pd.DataFrame) -> pd.DataFrame:
    """Gain on log training events, XGBoost's own AUC, endpoint group and
    lineage (reference levels: the largest group of each), weights 1/se^2."""
    data = cells.dropna(subset=["n_events_train", "auc_X", "endpoint_group", "cohort_group"]).copy()
    if len(data) < 30:
        return pd.DataFrame()
    design = pd.DataFrame({"intercept": 1.0,
                           "log10_training_events": np.log10(data.n_events_train + 1),
                           "xgboost_auc": data.auc_X})
    for column in ("endpoint_group", "cohort_group"):
        reference = data[column].value_counts().idxmax()
        dummies = pd.get_dummies(data[column], prefix=column, dtype=float)
        design = design.join(dummies.drop(columns=f"{column}_{reference}"))
    if "median_lab_results" in data and data.median_lab_results.notna().all():
        design["log10_median_lab_results"] = np.log10(data.median_lab_results + 1)
    x, y, w = design.to_numpy(float), data.gain.to_numpy(float), data.weight.to_numpy(float)
    sw = np.sqrt(w)
    beta, *_ = np.linalg.lstsq(x * sw[:, None], y * sw, rcond=None)
    residual = y - x @ beta
    # Heteroscedasticity-robust (HC1) standard errors for the weighted fit.
    bread = np.linalg.pinv((x * w[:, None]).T @ x)
    meat = (x * (w * residual)[:, None]).T @ (x * (w * residual)[:, None])
    scale = len(y) / max(len(y) - x.shape[1], 1)
    se = np.sqrt(np.diag(bread @ meat @ bread) * scale)
    return pd.DataFrame({"term": design.columns, "coefficient": beta, "robust_se": se,
                         "z": beta / se}).set_index("term")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--analysis-root", type=Path, required=True)
    parser.add_argument("--cell-metadata", type=Path, required=True)
    parser.add_argument("--patient-metadata", type=Path)
    parser.add_argument("--split", default="tuning")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 200, "display.max_columns", 20, "display.max_rows", 80)
    for contrast, label in CONTRASTS.items():
        cells = cell_table(args.analysis_root, args.cell_metadata, contrast, args.split, args.patient_metadata)
        if cells.empty:
            print(f"\n# {contrast} ({label}): not in the analysis")
            continue
        cells.to_csv(args.output_dir / f"cells_{contrast}.csv", index=False)
        print(f"\n# {contrast}: {label} ({len(cells)} cells, {args.split})")
        print(f"better in {np.mean(cells.gain > 0):.0%} of cells; confident wins {int((cells.lower > 0).sum())}, "
              f"confident losses {int((cells.upper < 0).sum())}")
        columns = ["endpoint_group", "cohort_group", "events_bin", "xgboost_auc_bin", "cohort"]
        for column in [c for c in columns if c in cells]:
            table = by_stratum(cells, column)
            table.to_csv(args.output_dir / f"by_{column}_{contrast}.csv")
            print(f"\n## by {column}\n{table.round(4).to_string()}")
        if contrast == "O1":
            model = weighted_regression(cells)
            if not model.empty:
                model.to_csv(args.output_dir / "regression_O1.csv")
                print(f"\n## precision-weighted regression of the cell gain\n{model.round(4).to_string()}")
            show = ["cohort", "outcome", "endpoint_group", "n_events_train", "tuning_cases_1y",
                    "auc_X", "auc_OPERA", "gain", "lower", "upper"]
            show = [c for c in show if c in cells]
            for title, part in (("confident OPERA wins", cells.loc[cells.lower > 0]),
                                ("confident XGBoost wins", cells.loc[cells.upper < 0])):
                part = part.assign(z=part.gain / part.difference_se).sort_values("z", key=abs, ascending=False)
                print(f"\n## {title} (top 20 of {len(part)})\n{part[show].head(20).round(3).to_string(index=False)}")


if __name__ == "__main__":
    main()
