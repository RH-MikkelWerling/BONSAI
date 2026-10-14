"""Read-only era support audit for the remaining LPR3/SMR coverage gates.

The report is descriptive. It does not automatically select a coverage start.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


SOURCES = {
    "hospitalisation": "LPR2_admissions_LPR3_contacts",
    "sepsis": "diagnoses_all",
    "serious_infection_composite": "diagnoses_all",
    "invasive_fungal": "diagnoses_all",
    "herpes_zoster": "diagnoses_all",
    "infections_IV_based": "SMR_administered_medicine",
    "infections_prescription_based": "prescription_dispensing",
    "bone_fracture": "diagnoses_all_comparator",
    "myocardial_infarction": "diagnoses_all_comparator",
}
HORIZONS = (30, 365)
MAX_COMPLETE_INDEX_YEAR = {30: 2025, 365: 2024}


def audit_one(path: Path, outcome: str) -> tuple[list[dict], list[dict]]:
    frame = pd.read_parquet(path, columns=[
        "subject_id", "index_date", "outcome_date", "censor_date",
    ])
    if frame["subject_id"].isna().any() or frame["subject_id"].duplicated().any():
        raise ValueError(f"Missing or duplicate subjects: {path}")
    for name in ("index_date", "outcome_date", "censor_date"):
        frame[name] = pd.to_datetime(frame[name], errors="coerce")
    if frame["index_date"].isna().any() or frame["censor_date"].isna().any():
        raise ValueError(f"Missing index/censor dates: {path}")
    if (frame["censor_date"] < frame["index_date"]).any():
        raise ValueError(f"Censor before index: {path}")
    event = frame["outcome_date"].notna()
    if ((event & (frame["outcome_date"] <= frame["index_date"])) |
            (event & (frame["outcome_date"] > frame["censor_date"]))).any():
        raise ValueError(f"Event outside observed follow-up: {path}")
    frame["index_year"] = frame["index_date"].dt.year
    frame["followup_days"] = (frame["censor_date"] - frame["index_date"]).dt.days
    frame["event_days"] = (frame["outcome_date"] - frame["index_date"]).dt.days
    rows = []
    for year, group in frame.groupby("index_year", sort=True):
        for horizon in HORIZONS:
            if year > MAX_COMPLETE_INDEX_YEAR[horizon]:
                continue
            events = group["event_days"].between(1, horizon)
            full_followup = group["followup_days"].ge(horizon)
            assessable = events | full_followup
            n_assessable = int(assessable.sum())
            rows.append({
                "outcome": outcome,
                "source": SOURCES[outcome],
                "index_year": int(year),
                "horizon_days": horizon,
                "n_eligible": len(group),
                "n_full_followup": int(full_followup.sum()),
                "n_assessable": n_assessable,
                "n_events": int(events.sum()),
                "event_fraction_assessable": (float(events.sum() / n_assessable)
                                               if n_assessable else None),
            })
    recorded = frame.loc[event, "outcome_date"].dt.year.value_counts().sort_index()
    calendar = [{"outcome": outcome, "source": SOURCES[outcome],
                 "event_year": int(year), "n_first_events": int(count)}
                for year, count in recorded.items()]
    return rows, calendar


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outcomes-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise SystemExit(f"Output directory already exists: {args.output_dir}")
    rates, calendar = [], []
    for outcome in SOURCES:
        path = args.outcomes_dir / f"{outcome}.parquet"
        if not path.is_file():
            raise SystemExit(f"Missing ungated outcome: {path}")
        part_rates, part_calendar = audit_one(path, outcome)
        rates.extend(part_rates)
        calendar.extend(part_calendar)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    pd.DataFrame(rates).to_csv(args.output_dir / "event_rates_by_index_year_INTERNAL.csv",
                               index=False)
    pd.DataFrame(calendar).to_csv(args.output_dir / "first_events_by_calendar_year_INTERNAL.csv",
                                  index=False)
    print(json.dumps({
        "outcomes": len(SOURCES),
        "output_dir": str(args.output_dir),
        "interpretation": "Descriptive support only; compare nearby complete years and source provenance before removing a gate",
    }))


if __name__ == "__main__":
    main()
