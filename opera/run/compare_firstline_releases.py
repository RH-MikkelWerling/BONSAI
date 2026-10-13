"""Aggregate-only comparison of historical and candidate first-line CSVs."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


FIELDS = ("firstline_date", "cohort_grouped", "cohort_fine", "arm_primary",
          "firstline_source", "cohort_source")


def _profile(frame: pd.DataFrame, ids: pd.Index) -> dict:
    selected = frame.loc[ids]
    result = {}
    for field in ("cohort_grouped", "cohort_fine", "firstline_source", "cohort_source"):
        if field in selected:
            counts = selected[field].astype("string").fillna("[missing]").value_counts()
            result[field] = {str(key): int(value) for key, value in counts.items()}
    return result


def _hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def compare(old_path: Path, new_path: Path) -> dict:
    frames = []
    for path in (old_path, new_path):
        frame = pd.read_csv(path, low_memory=False)
        if "patientid" not in frame or frame["patientid"].isna().any() or frame["patientid"].duplicated().any():
            raise ValueError(f"Missing or duplicate patientid in {path}")
        frame["patientid"] = frame["patientid"].astype("string").str.strip()
        if frame["patientid"].duplicated().any() or frame["patientid"].eq("").any():
            raise ValueError(f"Invalid patientid after normalization in {path}")
        frames.append(frame.set_index("patientid"))
    old, new = frames
    shared = old.index.intersection(new.index)
    report = {
        "old_path": str(old_path.resolve()), "new_path": str(new_path.resolve()),
        "old_sha256": _hash(old_path), "new_sha256": _hash(new_path),
        "old_n": len(old), "new_n": len(new), "shared_n": len(shared),
        "old_only_n": len(old) - len(shared), "new_only_n": len(new) - len(shared),
        "old_only_profile_INTERNAL": _profile(old, old.index.difference(shared)),
        "new_only_profile_INTERNAL": _profile(new, new.index.difference(shared)),
        "fields": {},
    }
    for field in FIELDS:
        if field not in old or field not in new:
            report["fields"][field] = {"status": "missing", "old_present": field in old,
                                       "new_present": field in new}
            continue
        before, after = old.loc[shared, field], new.loc[shared, field]
        if field == "firstline_date":
            before = pd.to_datetime(before, errors="coerce").dt.normalize()
            after = pd.to_datetime(after, errors="coerce").dt.normalize()
        else:
            before = before.astype("string").str.strip().fillna("[missing]")
            after = after.astype("string").str.strip().fillna("[missing]")
        same = before.eq(after).fillna(False) | (before.isna() & after.isna())
        report["fields"][field] = {
            "status": "compared", "changed_shared_n": int((~same).sum()),
            "old_missing_shared_n": int(before.isna().sum()),
            "new_missing_shared_n": int(after.isna().sum()),
            "changed_by_old_cohort_fine_INTERNAL": _profile(
                old, shared[~same.to_numpy()]).get("cohort_fine", {}),
        }
        if field == "firstline_date":
            delta = (after - before).dt.days
            absolute = delta.abs()
            report["fields"][field]["changed_by_over_30d_n"] = int(absolute.gt(30).sum())
            report["fields"][field]["date_shift_summary"] = {
                "n_compared": int(delta.notna().sum()),
                "n_equal": int(delta.eq(0).sum()),
                "n_new_earlier": int(delta.lt(0).sum()),
                "n_new_later": int(delta.gt(0).sum()),
                "n_abs_at_most_1_day": int(absolute.le(1).sum()),
                "n_abs_at_most_7_days": int(absolute.le(7).sum()),
                "n_abs_at_most_30_days": int(absolute.le(30).sum()),
                "n_abs_over_365_days": int(absolute.gt(365).sum()),
                "median_signed_days": float(delta.median()) if delta.notna().any() else None,
                "median_absolute_days": float(absolute.median()) if delta.notna().any() else None,
                "p90_absolute_days": float(absolute.quantile(0.9)) if delta.notna().any() else None,
            }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError(f"Refusing to overwrite {args.output}")
    report = compare(args.old, args.candidate)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("old_n", "new_n", "shared_n", "old_only_n", "new_only_n")}))


if __name__ == "__main__":
    main()
