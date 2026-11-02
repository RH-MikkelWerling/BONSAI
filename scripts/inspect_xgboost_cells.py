"""What does per-task XGBoost rely on in the cells where it beats OPERA most?

For the cells XGBoost wins with most confidence (from where_opera_beats_xgboost's
cells_O1.csv), plus any named with --cell, this prints per cell and horizon:

* the 20 features with the largest gain importance (share of the total), each
  flagged if its name suggests information about treatment, outcome, death,
  registries, dates or windows that end at or after the start of treatment;
* each top feature's own tuning AUC (the feature alone, oriented, missing values
  ranked lowest), its share of patients with a value, and whether it alone comes
  within 0.03 of XGBoost's AUC: one feature doing all the work is the signature
  of leakage;
* the size of the hyperparameter search XGBoost won on these same tuning
  patients (its tuning AUC is selected on them, so it is optimistic).

Aggregates only: feature names, importances and per-cell AUCs; no patient rows.

    .venv-v100/bin/python scripts/inspect_xgboost_cells.py \
        --cells data/paper_256_v3/analysis/tuning_1103/where_opera_wins/cells_O1.csv \
        --tabular-dir data/paper_256_v3/tabular_runs --features-dir data/paper_256_v3/features \
        --top 12 --cell RT:treatment_failure --cell DLBCL:treatment_failure
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

FLAGS = {
    "treatment": r"treat|line|regimen|chemo|rituximab|r[-_]?chop|cyclophosph|bendamust|lenalid|bortezom|"
                 r"daratum|ibrutin|venetoc|atc_l0[14]|l01|l04|cytostat|immunother|transplant|radiat",
    "outcome or death": r"outc|death|dead|died|mortal|relapse|progress|failure|second",
    "registry": r"rkkp|adverse|registry|lyfo|cll_reg|myeloma_reg",
    "date or time": r"date|timestamp|year|calendar|days_since|time_since|period",
    "window ending at 0": r"within_0_to_0|_0_to_0_|lookahead|ahead",
}


def flags(name: str) -> str:
    lowered = name.lower()
    return ", ".join(label for label, pattern in FLAGS.items() if re.search(pattern, lowered))


def auc(values: np.ndarray, labels: np.ndarray) -> float:
    """Oriented AUC of one feature (ties averaged; missing ranked lowest)."""
    x = np.where(np.isfinite(values), values, -np.inf)
    pos, neg = labels == 1, labels == 0
    if pos.sum() == 0 or neg.sum() == 0:
        return np.nan
    ranks = pd.Series(x).rank(method="average").to_numpy()
    value = (ranks[pos].sum() - pos.sum() * (pos.sum() + 1) / 2) / (pos.sum() * neg.sum())
    return float(max(value, 1 - value))


def transformed_names(meta: dict, features: pd.DataFrame | None, train_ids) -> list[str]:
    """Names of the columns XGBoost saw, in order: the numeric columns, then a
    missingness indicator for each numeric column with a gap in the training rows
    (SimpleImputer add_indicator), then the one-hot categories. Value columns
    map exactly; indicators are recomputed from the training rows."""
    columns = list(meta.get("feature_columns") or [])
    categorical = set(meta.get("categorical_columns") or [])
    numeric = [c for c in columns if c not in categorical]
    names = list(numeric)
    if features is not None and train_ids is not None:
        rows = features.loc[features.subject_id.isin(train_ids), [c for c in numeric if c in features]]
        names += [f"missing: {c}" for c in numeric if c in rows and rows[c].isna().any()]
    names += [f"category of {c}" for c in columns if c in categorical]
    return names


def tuning_labels(job: dict, split: str = "tuning") -> pd.DataFrame:
    """The analysis's own labels for the cell (same call as the inventory)."""
    from opera.evaluation.cohorts import population_subject_ids
    from opera.run.train_tabular_baselines import outcome_labels

    allowed = population_subject_ids(job["population"], cohort_fine_col="cohort_fine",
                                      cohort_fine_value=job["cohort"])
    labels = outcome_labels(
        job["outcome_path"], split=split, n_hours_start_include=job["n_hours_start_include"],
        n_hours_end_include=job["horizon_days"] * 24, require_min_followup=True, known_status_cif=True,
        competing_outcome_path=job["competing_path"], registry_start_date=job["registry_start_date"],
        cohort=job["cohort"], outcome_name=job["outcome"], eligibility_path=job["eligibility"],
        allowed_subject_ids=allowed)
    return labels.assign(subject_id=labels.subject_id.astype(str))[["subject_id", "label"]]


def inspect(directory: Path, features_dir: Path, xgboost_auc: float | None, top: int) -> None:
    importance = sorted(p for p in directory.glob("*feature_importance.csv") if "xgboost" in p.name.lower())
    if not importance:
        print(f"  no XGBoost importance file in {directory}")
        return
    table = pd.read_csv(importance[0])
    column = "gain_importance" if "gain_importance" in table else table.columns[1]
    table["share"] = table[column].abs() / table[column].abs().sum()
    table = table.head(20).copy()
    metadata = next(iter(directory.glob("*xgboost*metadata.json")), None)
    meta = json.loads(metadata.read_text()) if metadata else {}
    if meta:
        print(f"  tuned on the tuning patients: {meta.get('tune_params') or {}}")
    intent = directory / "intent.json"
    job = json.loads(intent.read_text())["job"] if intent.is_file() else None
    if job is not None:
        try:
            labels = tuning_labels(job)
            from opera.functional.tabular_features import read_feature_table

            name = "all" if job.get("pooled") else job["cohort"]
            features = read_feature_table(features_dir / f"{name}__sequence_matched.parquet")
            features["subject_id"] = features.subject_id.astype(str)
            if meta and table.feature.str.match(r"feature_\d+$").all():
                train = set(tuning_labels(job, "train").subject_id)
                names = transformed_names(meta, features, train)
                position = table.feature.str.removeprefix("feature_").astype(int)
                table["feature"] = [names[i] if i < len(names) else f"feature_{i}" for i in position]
            data = labels.merge(features, on="subject_id", how="left")
            print(f"  tuning patients {len(data)}, cases {int(data.label.sum())}")
            single, present = [], []
            for feature in table.feature:
                indicator = feature.startswith("missing: ")
                raw = feature.removeprefix("missing: ").split("__", 1)[-1]
                if raw in data:
                    values = pd.to_numeric(data[raw], errors="coerce").to_numpy(float)
                    if indicator:
                        values = np.isnan(values).astype(float)
                    single.append(auc(values, data.label.to_numpy()))
                    present.append(float(np.isfinite(values).mean()))
                else:
                    single.append(np.nan)
                    present.append(np.nan)
            table["alone_auc"] = single
            table["has_value"] = present
        except Exception as error:  # report, keep the importances
            print(f"  single-feature AUC unavailable: {type(error).__name__}: {error}")
    table["flags"] = table.feature.map(flags)
    shown = [c for c in ("feature", "share", "alone_auc", "has_value", "flags") if c in table]
    print(table[shown].head(top + 8).round(3).to_string(index=False))
    if xgboost_auc is not None and "alone_auc" in table:
        close = table.loc[table.alone_auc >= xgboost_auc - 0.03, "feature"].tolist()
        if close:
            print(f"  RED FLAG: one feature alone within 0.03 of XGBoost's AUC {xgboost_auc:.3f}: {close}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cells", type=Path, required=True)
    parser.add_argument("--tabular-dir", type=Path, required=True)
    parser.add_argument("--features-dir", type=Path, required=True)
    parser.add_argument("--top", type=int, default=12, help="most confident XGBoost wins to inspect")
    parser.add_argument("--cell", action="append", default=[], help="cohort:outcome, inspected in addition")
    parser.add_argument("--horizons", default="365", help="comma-separated horizons in days")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    pd.set_option("display.width", 220, "display.max_colwidth", 70)
    cells = pd.read_csv(args.cells)
    losses = cells.loc[cells.upper < 0].assign(z=lambda f: f.gain / f.difference_se).sort_values("z")
    chosen = list(losses[["cohort", "outcome"]].head(args.top).itertuples(index=False, name=None))
    chosen += [tuple(c.split(":", 1)) for c in args.cell if tuple(c.split(":", 1)) not in chosen]
    for cohort, outcome in chosen:
        row = cells.loc[(cells.cohort == cohort) & (cells.outcome == outcome)]
        x_auc = float(row.auc_X.iloc[0]) if len(row) and "auc_X" in row and pd.notna(row.auc_X.iloc[0]) else None
        summary = (f"XGBoost {row.auc_X.iloc[0]:.3f} vs OPERA {row.auc_OPERA.iloc[0]:.3f}, "
                   f"gain {row.gain.iloc[0]:+.3f}, training events {row.n_events_train.iloc[0]:.0f}"
                   if len(row) and {"auc_X", "auc_OPERA"} <= set(row) else "not in the cells table")
        print(f"\n### {cohort} / {outcome}: {summary} (all-horizon AUCs)")
        for horizon in args.horizons.split(","):
            directory = args.tabular_dir / f"seed{args.seed}" / cohort / outcome / f"{horizon}d"
            print(f" -- {horizon} days ({directory})")
            inspect(directory, args.features_dir, x_auc, top=12)


if __name__ == "__main__":
    main()
