"""Read-only, aggregate integrity audit of OPERA outcome parquet exports."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import yaml


SPLIT_START = {"tuning": pd.Timestamp("2022-01-01"), "held_out": pd.Timestamp("2023-01-01")}
REQUIRED = {"subject_id", "split", "index_date", "outcome_date", "censor_date"}


def audit_outcome(path: Path) -> dict:
    frame = pd.read_parquet(path)
    missing = sorted(REQUIRED - set(frame.columns))
    if missing:
        return {"outcome": path.stem, "rows": len(frame), "error": f"missing columns: {missing}"}
    dates = {name: pd.to_datetime(frame[name], errors="coerce") for name in
             ("index_date", "outcome_date", "censor_date")}
    split = frame["split"].astype("string")
    event = dates["outcome_date"].notna()
    expected = pd.Series("train", index=frame.index, dtype="string")
    expected.loc[dates["index_date"] >= SPLIT_START["tuning"]] = "tuning"
    expected.loc[dates["index_date"] >= SPLIT_START["held_out"]] = "held_out"
    return {
        "outcome": path.stem,
        "rows": len(frame),
        "unique_subjects": int(frame["subject_id"].nunique(dropna=True)),
        "duplicate_subjects": int(frame["subject_id"].duplicated().sum()),
        "missing_subjects": int(frame["subject_id"].isna().sum()),
        "missing_index": int(dates["index_date"].isna().sum()),
        "missing_censor": int(dates["censor_date"].isna().sum()),
        "censor_before_index": int((dates["censor_date"] < dates["index_date"]).sum()),
        "event_on_or_before_index": int((event & (dates["outcome_date"] <= dates["index_date"])).sum()),
        "event_after_censor": int((event & (dates["outcome_date"] > dates["censor_date"])).sum()),
        "invalid_split": int((~split.isin(["train", "tuning", "held_out"])).sum()),
        "split_index_mismatch": int((split.notna() & dates["index_date"].notna() & (split != expected)).sum()),
        **{f"n_{label}": int((split == label).sum()) for label in SPLIT_START | {"train": None}},
        **{f"events_{label}": int((event & (split == label)).sum()) for label in SPLIT_START | {"train": None}},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outcomes-dir", type=Path, required=True)
    parser.add_argument("--extra-outcomes-dir", type=Path, action="append",
                        help="Additional outcome directory; repeat for second-line and composites")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--registry", type=Path,
        default=Path("opera/configs/experiment_registry.yaml"),
    )
    args = parser.parse_args()
    paths = sorted(path for path in args.outcomes_dir.glob("*.parquet") if "__" not in path.stem)
    for extra_dir in args.extra_outcomes_dir or []:
        if not extra_dir.is_dir():
            raise SystemExit(f"Extra outcomes directory not found: {extra_dir}")
        paths += sorted(path for path in extra_dir.glob("*.parquet")
                        if "__" not in path.stem)
    if not paths:
        raise SystemExit("No canonical outcome parquets found")
    stems = [path.stem for path in paths]
    duplicates = sorted({stem for stem in stems if stems.count(stem) > 1})
    if duplicates:
        raise SystemExit(f"Duplicate canonical outcomes across directories: {duplicates}")
    registry = yaml.safe_load(args.registry.read_text(encoding="utf-8"))
    expected = set(registry["outcomes"])
    found = {path.stem for path in paths}
    inventory = {"missing": sorted(expected - found), "extra": sorted(found - expected)}
    results = [audit_outcome(path) for path in paths]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(results).to_csv(args.output, index=False)
    bad = [row for row in results if "error" in row or any(
        row.get(key, 0) for key in (
            "duplicate_subjects", "missing_subjects", "missing_index", "missing_censor",
            "censor_before_index", "event_on_or_before_index", "event_after_censor",
            "invalid_split", "split_index_mismatch",
        )
    )]
    print(json.dumps({"outcomes": len(results), "flagged": len(bad),
                      "inventory": inventory, "report": str(args.output)}))
    if bad or inventory["missing"] or inventory["extra"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
