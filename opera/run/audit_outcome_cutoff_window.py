"""Aggregate, read-only impact audit for competing outcome cutoff dates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def audit_cutoff_window(outcomes_dir: Path, early: str, late: str) -> pd.DataFrame:
    early_date, late_date = pd.Timestamp(early), pd.Timestamp(late)
    if not early_date < late_date:
        raise ValueError("Require --early before --late")
    paths = sorted(path for path in outcomes_dir.glob("*.parquet") if "__" not in path.stem)
    if not paths:
        raise ValueError("No canonical outcome parquets found")
    rows = []
    for path in paths:
        frame = pd.read_parquet(path, columns=["subject_id", "index_date", "outcome_date", "censor_date"])
        index = pd.to_datetime(frame["index_date"], errors="coerce")
        event = pd.to_datetime(frame["outcome_date"], errors="coerce")
        censor = pd.to_datetime(frame["censor_date"], errors="coerce")
        if index.isna().any() or censor.isna().any():
            raise ValueError(f"Invalid index/censor date in {path.name}")
        rows.append({
            "outcome": path.stem,
            "n_rows": len(frame),
            "n_index_in_early_late_window": int(index.between(early_date, late_date, inclusive="left").sum()),
            "n_events_after_early_through_late": int(event.between(early_date, late_date, inclusive="right").sum()),
            "n_events_after_late": int(event.gt(late_date).sum()),
            "n_censor_after_early_through_late": int(censor.between(early_date, late_date, inclusive="right").sum()),
            "n_at_risk_on_early_date": int((index.lt(early_date) & censor.ge(early_date)).sum()),
        })
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outcomes-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--early", default="2026-03-04")
    parser.add_argument("--late", default="2026-03-19")
    args = parser.parse_args()
    result = audit_cutoff_window(args.outcomes_dir, args.early, args.late)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)
    print(json.dumps({
        "outcomes": len(result),
        "early": args.early,
        "late": args.late,
        "outcomes_with_interval_events": int(result["n_events_after_early_through_late"].gt(0).sum()),
        "interval_event_rows_across_outcomes": int(result["n_events_after_early_through_late"].sum()),
        "overall_survival": result.loc[result["outcome"].eq("overall_survival")].to_dict("records"),
        "report": str(args.output),
        "note": "Counts show potential impact, not which source cutoff is valid",
    }))


if __name__ == "__main__":
    main()
