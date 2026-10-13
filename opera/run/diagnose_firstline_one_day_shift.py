"""INTERNAL aggregate check for a systematic treatment-record date shift.

Deterministically samples shared patients whose *index* moved exactly one day
earlier, then compares old/new treatment rows with equal multiplicity for each
patient/source/tx_code. No identifiers or record-level dates are written.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


def _sample_ids(old_mapped: Path, new_mapped: Path, n: int) -> list[int]:
    columns = ["patientid", "firstline_date"]
    old = pd.read_csv(old_mapped, usecols=columns, dtype={"patientid": "string"})
    new = pd.read_csv(new_mapped, usecols=columns, dtype={"patientid": "string"})
    if old["patientid"].isna().any() or new["patientid"].isna().any() or \
       old["patientid"].duplicated().any() or new["patientid"].duplicated().any():
        raise ValueError("Mapped CSVs need unique nonmissing patient IDs")
    shared = old.merge(new, on="patientid", suffixes=("_old", "_new"), validate="one_to_one")
    delta = (pd.to_datetime(shared["firstline_date_new"], errors="coerce") -
             pd.to_datetime(shared["firstline_date_old"], errors="coerce")).dt.days
    ids = shared.loc[delta.eq(-1), "patientid"].tolist()
    ids.sort(key=lambda value: hashlib.sha256(value.encode()).digest())
    try:
        return [int(value) for value in ids[:n]]
    except ValueError as exc:
        raise ValueError("This audit expects numeric patient IDs matching the Parquet key") from exc


def _records(path: Path, ids: list[int]) -> tuple[pd.DataFrame, dict]:
    columns = ["patientid", "source", "tx_code", "date_start"]
    frame = pd.read_parquet(path, columns=columns, filters=[("patientid", "in", ids)])
    frame = frame.dropna(subset=["patientid", "source", "tx_code", "date_start"]).copy()
    timestamps = pd.to_datetime(frame["date_start"], errors="coerce")
    timezone = timestamps.dt.tz
    # A UTC timestamp and a timezone-naive wall time cannot be subtracted.
    # Compare both the stored clock date and a Copenhagen-calendar sensitivity.
    clock_time = timestamps.dt.tz_localize(None)
    local_time = (timestamps.dt.tz_convert("Europe/Copenhagen").dt.tz_localize(None)
                  if timezone is not None else clock_time)
    assumed_utc_time = (timestamps.dt.tz_convert("Europe/Copenhagen").dt.tz_localize(None)
                        if timezone is not None else
                        timestamps.dt.tz_localize("UTC").dt.tz_convert(
                            "Europe/Copenhagen").dt.tz_localize(None))
    frame["date"] = clock_time.dt.normalize()
    frame["date_copenhagen"] = local_time.dt.normalize()
    frame["date_naive_assumed_utc"] = assumed_utc_time.dt.normalize()
    frame = frame.dropna(subset=["date"])
    frame["patientid"] = frame["patientid"].astype("string")
    frame["source"] = frame["source"].astype("string")
    frame["tx_code"] = frame["tx_code"].astype("string")
    keys = ["patientid", "source", "tx_code"]
    frame = frame.sort_values(keys + ["date"], kind="stable")
    frame["ordinal"] = frame.groupby(keys, dropna=False).cumcount()
    frame["multiplicity"] = frame.groupby(keys, dropna=False)["date"].transform("size")
    metadata = {"date_start_dtype": str(frame["date_start"].dtype),
                "timezone": str(timezone) if timezone is not None else None,
                "hour_0_rows": int(clock_time.dt.hour.eq(0).sum()),
                "hour_22_or_23_rows": int(clock_time.dt.hour.isin([22, 23]).sum())}
    return frame[keys + ["ordinal", "multiplicity", "date", "date_copenhagen",
                         "date_naive_assumed_utc"]], metadata


def diagnose(old_mapped: Path, new_mapped: Path, old_treatment: Path,
             new_treatment: Path, sample_n: int = 500) -> dict:
    ids = _sample_ids(old_mapped, new_mapped, sample_n)
    if not ids:
        raise ValueError("No shared patients had an exactly -1-day index shift")
    old, old_timestamp = _records(old_treatment, ids)
    new, new_timestamp = _records(new_treatment, ids)
    keys = ["patientid", "source", "tx_code", "ordinal"]
    paired = old.merge(new, on=keys, suffixes=("_old", "_new"), validate="one_to_one")
    paired = paired.loc[paired["multiplicity_old"].eq(paired["multiplicity_new"])].copy()
    paired["shift_days"] = (paired["date_new"] - paired["date_old"]).dt.days
    paired["shift_days_copenhagen"] = (
        paired["date_copenhagen_new"] - paired["date_copenhagen_old"]).dt.days
    paired["shift_days_naive_assumed_utc"] = (
        paired["date_naive_assumed_utc_new"] -
        paired["date_naive_assumed_utc_old"]).dt.days
    rows = []
    for source, group in paired.groupby("source", dropna=False):
        delta = group["shift_days"]
        local_delta = group["shift_days_copenhagen"]
        assumed_utc_delta = group["shift_days_naive_assumed_utc"]
        rows.append({"source": str(source), "n_matched_rows": len(group),
                     "n_minus_one_day": int(delta.eq(-1).sum()),
                     "n_equal": int(delta.eq(0).sum()),
                     "n_plus_one_day": int(delta.eq(1).sum()),
                     "n_other_shift": int((~delta.isin([-1, 0, 1])).sum()),
                     "median_shift_days": float(delta.median()),
                     "n_equal_copenhagen_calendar": int(local_delta.eq(0).sum()),
                     "n_minus_one_day_copenhagen_calendar": int(local_delta.eq(-1).sum()),
                     "median_shift_days_copenhagen_calendar": float(local_delta.median()),
                     "n_equal_if_naive_new_is_utc_clock": int(assumed_utc_delta.eq(0).sum()),
                     "median_shift_if_naive_new_is_utc_clock": float(assumed_utc_delta.median())})
    return {"status": "internal_sampled_diagnostic_not_event_adjudication",
            "n_sampled_patients_with_minus_one_day_index": len(ids),
            "n_old_sample_treatment_rows_with_key": len(old),
            "n_new_sample_treatment_rows_with_key": len(new),
            "n_equal_multiplicity_matched_rows": len(paired),
            "old_timestamp_representation": old_timestamp,
            "new_timestamp_representation": new_timestamp,
            "source_date_shifts_INTERNAL": sorted(rows, key=lambda row: -row["n_matched_rows"]),
            "limitation": "Pairs repeated records by chronological rank only when patient/source/tx_code multiplicity matches; source counts can include small internal cells."}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("old-mapped", "new-mapped", "old-treatment", "new-treatment", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--sample-n", type=int, default=500)
    args = parser.parse_args()
    if args.sample_n < 1 or args.output.exists():
        raise ValueError("Use positive sample-n and a new output path")
    report = diagnose(args.old_mapped, args.new_mapped, args.old_treatment,
                      args.new_treatment, args.sample_n)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in (
        "n_sampled_patients_with_minus_one_day_index",
        "n_equal_multiplicity_matched_rows", "output") if key in report} |
        {"output": str(args.output)}))


if __name__ == "__main__":
    main()
