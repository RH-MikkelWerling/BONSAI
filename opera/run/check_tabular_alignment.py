"""Do the tabular feature matrices cover the neural cohorts at the same index dates?

For every cohort in the bundle, the neural patients and their index dates come
from the cohort's population file (filtered on ``cohort_fine`` exactly as the
fits are); the tabular side is ``<cohort_fine>.<ext>`` in the raw feature
folder. ``all.<ext>`` is compared with the union of the cohorts. Counts only.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from opera.functional.tabular_features import _read_population, read_feature_table
from opera.run.paper_tabular_queue import tabular_jobs

SUFFIXES = {".pkl", ".pickle", ".parquet", ".pq", ".csv"}


def neural_patients(population_path, cohorts, index_column) -> pd.Series:
    """Index date per neural patient of these cohorts (indexed by subject_id)."""
    population = _read_population(population_path)  # index_date from firstline_date
    population = population[population["cohort_fine"].astype(str).isin(cohorts)]
    ids = population["subject_id"].astype("string").str.strip()
    dates = pd.to_datetime(population[index_column], errors="coerce")
    return pd.Series(dates.to_numpy(), index=ids.to_numpy())


def compare(name, matrix_path, neural: pd.Series) -> dict:
    """Patient overlap and index-date agreement for one matrix."""
    row = {"matrix": name, "file": matrix_path.name if matrix_path else None,
           "neural_patients": len(neural)}
    if matrix_path is None:
        return row | {"status": "no matrix for this cohort"}
    features = read_feature_table(matrix_path)
    features = features.rename(columns={"patientid": "subject_id"})
    ids = features["subject_id"].astype("string").str.strip()
    both = ids[ids.isin(neural.index)]
    row |= {
        "matrix_rows": len(features),
        "duplicate_patients": int(ids.duplicated().sum()),
        "in_both": int(both.nunique()),
        "neural_missing_from_matrix": int((~neural.index.isin(ids)).sum()),
        "matrix_not_in_neural_cohort": int((~ids.isin(neural.index)).sum()),
        "columns": features.shape[1],
    }
    if "timestamp" in features:
        stamps = pd.to_datetime(features["timestamp"], errors="coerce")
        stamps.index = ids
        shared = stamps[stamps.index.isin(neural.index)]
        shared = shared[~shared.index.duplicated()]
        days = (shared.dt.normalize() - neural.loc[shared.index].dt.normalize()).dt.days
        row |= {
            "same_index_day": int((days == 0).sum()),
            "index_day_differs": int((days != 0).sum()),
            "index_diff_days_min": days.min(),
            "index_diff_days_median": days.median(),
            "index_diff_days_max": days.max(),
        }
    else:
        row["status"] = "no timestamp column: index date not checked"
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle-dir", type=Path, required=True)
    parser.add_argument("--feature-dir", type=Path, required=True)
    parser.add_argument("--index-date-column", default="index_date")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    populations = {}
    for job in tabular_jobs(args.bundle_dir, (365,)):
        populations.setdefault(job["cohort"], job["population"])
    files = {p.stem: p for p in args.feature_dir.iterdir() if p.suffix.lower() in SUFFIXES}

    rows = [
        compare(cohort, files.get(cohort),
                neural_patients(population, {cohort}, args.index_date_column))
        for cohort, population in sorted(populations.items())
    ]
    # The pooled matrix against every cohort (the cohorts share one population file).
    union = pd.concat([
        neural_patients(population, set(populations), args.index_date_column)
        for population in sorted(set(populations.values()))
    ])
    rows.append(compare("all", files.get("all"), union[~union.index.duplicated()]))
    unused = sorted(set(files) - set(populations) - {"all"})

    table = pd.DataFrame(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.output, index=False)
    with pd.option_context("display.max_columns", None, "display.width", 250):
        print(table.to_string(index=False))
    if unused:
        print(f"\nFiles not matching any bundle cohort (not used): {unused}")
    print(f"\nWritten: {args.output}")


if __name__ == "__main__":
    main()
