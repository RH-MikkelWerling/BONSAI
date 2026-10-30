"""INTERNAL aggregate-only comparison of treatment and MM seed Parquet versions.

No patient identifiers or source free text are emitted. This diagnoses whether
the newer files can be considered for a first-line release; it does not select
an input version or modify any data.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd
import polars as pl
import pyarrow.parquet as pq


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _input(path: Path) -> dict:
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"Expected regular Parquet file, not symlink: {path}")
    return {"path": str(path.resolve()), "bytes": path.stat().st_size,
            "sha256": _sha256(path)}


def _counts(scan: pl.LazyFrame, column: str) -> list[dict]:
    if column not in scan.collect_schema():
        return []
    grouped = (scan.group_by(pl.col(column).cast(pl.String).fill_null("[missing]"))
               .agg(pl.len().alias("n_rows")).sort("n_rows", descending=True)
               .collect(engine="streaming"))
    # Source and code labels only; never emit free-text indication or patient ID.
    return [{"category": row[column], "n_rows": row["n_rows"]}
            for row in grouped.head(100).to_dicts()]


def _treatment(path: Path) -> tuple[dict, set[str]]:
    scan = pl.scan_parquet(path)
    schema = {name: str(dtype) for name, dtype in scan.collect_schema().items()}
    if "patientid" not in schema:
        raise ValueError(f"Missing patientid in {path}")
    summary = {"input": _input(path), "schema": schema,
               "n_rows": int(scan.select(pl.len()).collect().item()),
               "n_patients": int(scan.select(pl.col("patientid").n_unique()).collect().item()),
               "source_rows": _counts(scan, "source"),
               "tx_code_rows_top100": _counts(scan, "tx_code")}
    for column in ("patientid", "date_start", "tx_code", "tx_text", "source",
                   "dx_code", "dx_text"):
        if column in schema:
            summary[f"n_missing_{column}"] = int(scan.select(
                pl.col(column).is_null().sum()).collect().item())
    if "date_start" in schema:
        column = pl.col("date_start")
        if scan.collect_schema()["date_start"] == pl.String:
            column = column.str.to_date(strict=False)
        else:
            column = column.cast(pl.Date, strict=False)
        dates = scan.select(column.alias("date"))
        bounds = dates.select(pl.col("date").min().alias("minimum"),
                              pl.col("date").max().alias("maximum")).collect().to_dicts()[0]
        summary["date_min"] = str(bounds["minimum"]) if bounds["minimum"] else None
        summary["date_max"] = str(bounds["maximum"]) if bounds["maximum"] else None
        summary["year_rows"] = _counts(dates.with_columns(
            pl.col("date").dt.year().alias("year")), "year")
    ids = set(scan.select(pl.col("patientid").cast(pl.String).drop_nulls().unique())
              .collect(engine="streaming").to_series().to_list())
    return summary, ids


def _line_one(path: Path) -> tuple[dict, pd.DataFrame]:
    columns = pq.ParquetFile(path).schema_arrow.names
    required = {"patientid", "line"}
    if not required.issubset(columns):
        raise ValueError(f"Missing MM seed columns in {path}: {sorted(required - set(columns))}")
    date_column = "date_start_line" if "date_start_line" in columns else "date_start"
    if date_column not in columns:
        raise ValueError(f"No line-level or treatment-row start date in {path}")
    selected = sorted(required | {date_column} |
                      ({"regimen"} if "regimen" in columns else set()) |
                      ({"row_acc"} if "row_acc" in columns else set()))
    data = pd.read_parquet(path, columns=selected)
    line_one = data.loc[pd.to_numeric(data["line"], errors="coerce").eq(1)].copy()
    line_one["patientid"] = line_one["patientid"].astype("string").str.strip()
    line_one["date_start_line"] = pd.to_datetime(line_one[date_column], errors="coerce").dt.normalize()
    valid = line_one.loc[line_one["patientid"].notna() &
                         line_one["date_start_line"].notna()].copy()
    chosen = (valid.sort_values(["patientid", "date_start_line"], kind="stable")
              .drop_duplicates("patientid"))
    if "row_acc" in valid:
        accepted = valid.loc[valid["row_acc"].eq(True).fillna(False)]
        accepted_start = (accepted.groupby("patientid")["date_start_line"].min()
                          .rename("accepted_line_one_start"))
        chosen = chosen.join(accepted_start, on="patientid")
    info = {"input": _input(path), "date_source_field": date_column,
            "date_derivation": ("stored_line_start" if date_column == "date_start_line"
                                else "minimum_date_start_among_line_1_rows_per_patient"),
            "schema": {name: str(pq.ParquetFile(path).schema_arrow.field(name).type)
                                              for name in columns},
            "n_all_rows": len(data), "n_line_one_rows": len(line_one),
            "n_line_one_valid_rows": len(valid), "n_line_one_patients": len(chosen),
            "n_line_one_patients_with_multiple_rows": int(valid["patientid"].duplicated(keep=False)
                                                           .groupby(valid["patientid"]).any().sum())}
    if "row_acc" in valid:
        info["n_line_one_rows_row_acc_true"] = int(valid["row_acc"].eq(True).fillna(False).sum())
        info["n_line_one_patients_without_row_acc_true"] = int(
            chosen["accepted_line_one_start"].isna().sum())
    keep = ["patientid", "date_start_line"] + (["regimen"] if "regimen" in chosen else [])
    if "accepted_line_one_start" in chosen:
        keep.append("accepted_line_one_start")
    return info, chosen[keep]


def _date_shift_summary(old: pd.Series, new: pd.Series) -> dict:
    days = (new - old).dt.days.dropna()
    absolute = days.abs()
    return {"n_compared": len(days),
            "n_equal": int(days.eq(0).sum()),
            "n_new_earlier": int(days.lt(0).sum()),
            "n_new_later": int(days.gt(0).sum()),
            "n_abs_at_most_1_day": int(absolute.le(1).sum()),
            "n_abs_at_most_7_days": int(absolute.le(7).sum()),
            "n_abs_at_most_30_days": int(absolute.le(30).sum()),
            "n_abs_over_365_days": int(absolute.gt(365).sum()),
            "median_signed_days": float(days.median()) if len(days) else None,
            "median_absolute_days": float(absolute.median()) if len(days) else None,
            "p90_absolute_days": float(absolute.quantile(0.9)) if len(days) else None}


def compare(old_treatment: Path | None = None, new_treatment: Path | None = None,
            old_mm: Path | None = None, new_mm: Path | None = None) -> dict:
    if (old_treatment is None) != (new_treatment is None):
        raise ValueError("Supply both treatment versions or neither")
    if (old_mm is None) != (new_mm is None):
        raise ValueError("Supply both MM versions or neither")
    if old_treatment is None and old_mm is None:
        raise ValueError("Supply at least one old/new pair")
    result = {"status": "internal_version_comparison_not_input_approval"}
    if old_treatment is not None and new_treatment is not None:
        old_tx, old_ids = _treatment(old_treatment)
        new_tx, new_ids = _treatment(new_treatment)
        result["treatment"] = {
            "old": old_tx, "new": new_tx,
            "columns_old_only": sorted(set(old_tx["schema"]) - set(new_tx["schema"])),
            "columns_new_only": sorted(set(new_tx["schema"]) - set(old_tx["schema"])),
            "n_patients_shared": len(old_ids & new_ids),
            "n_patients_old_only": len(old_ids - new_ids),
            "n_patients_new_only": len(new_ids - old_ids)}
    if old_mm is not None and new_mm is not None:
        old_seed, old_first = _line_one(old_mm)
        new_seed, new_first = _line_one(new_mm)
        shared = old_first.merge(new_first, on="patientid", how="inner", suffixes=("_old", "_new"),
                                 validate="one_to_one")
        date_changed = shared["date_start_line_old"].ne(shared["date_start_line_new"])
        result["mm_seed"] = {"old": old_seed, "new": new_seed,
                             "n_line_one_patients_shared": len(shared),
                             "n_line_one_patients_old_only": len(old_first) - len(shared),
                             "n_line_one_patients_new_only": len(new_first) - len(shared),
                             "n_line_one_dates_changed_shared": int(date_changed.sum()),
                             "date_shift_all_line_one_rows": _date_shift_summary(
                                 shared["date_start_line_old"], shared["date_start_line_new"])}
        if "accepted_line_one_start" in new_first:
            result["mm_seed"]["date_shift_row_acc_true_only"] = _date_shift_summary(
                shared["date_start_line_old"], shared["accepted_line_one_start"])
        if "regimen_old" in shared and "regimen_new" in shared:
            a, b = shared["regimen_old"].astype("string"), shared["regimen_new"].astype("string")
            result["mm_seed"]["n_line_one_regimens_changed_shared"] = int(
                (~(a.eq(b) | (a.isna() & b.isna())).fillna(False)).sum())
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    for name in ("old-treatment", "new-treatment", "old-mm", "new-mm"):
        parser.add_argument(f"--{name}", type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError(f"Refusing to overwrite {args.output}")
    result = compare(args.old_treatment, args.new_treatment, args.old_mm, args.new_mm)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report = {"output": str(args.output)}
    if "treatment" in result:
        report["old_treatment_rows"] = result["treatment"]["old"]["n_rows"]
        report["new_treatment_rows"] = result["treatment"]["new"]["n_rows"]
    if "mm_seed" in result:
        report["mm_shared_line_one"] = result["mm_seed"]["n_line_one_patients_shared"]
        report["mm_changed_line_one_dates"] = result["mm_seed"]["n_line_one_dates_changed_shared"]
    print(json.dumps(report))


if __name__ == "__main__":
    main()
