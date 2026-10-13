"""Reformat exact aggregate Table 1 exports without rereading patient histories."""

import argparse
import json
from pathlib import Path

import pandas as pd

from opera.functional.table1_release import release_table


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--disclosure-policy",
        choices=["primary-only", "targeted", "conservative"],
        required=True,
    )
    parser.add_argument("--suppression-threshold", type=int, default=5)
    args = parser.parse_args()
    table = pd.read_csv(args.input)
    # Older aggregate files may still contain the now-excluded treatment field.
    table = table.loc[table.variable != "arm_day1_primary"].copy()
    required = {"n_total", "n_observed", "n_missing", "count", "variable", "kind"}
    if not required <= set(table):
        raise ValueError(
            "Use the exact LOCAL_ONLY_internal/table1_values_long.csv, not an already suppressed export"
        )
    result, policy = release_table(
        table, args.suppression_threshold, args.disclosure_policy
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output_dir / "table1_values_long.csv", index=False)
    pooled = result.loc[result.cohort_level == "all"].copy()
    pooled["row"] = pooled.characteristic + pooled.category.map(
        lambda value: f": {value}" if pd.notna(value) else ""
    )
    pooled.pivot(index=["variable", "row"], columns="split", values="display").reindex(
        columns=["Overall", "train", "tuning", "held_out"]
    ).to_csv(args.output_dir / "table1_pooled_display.csv")
    (args.output_dir / "disclosure_policy.json").write_text(
        json.dumps(policy, indent=2), encoding="utf-8"
    )
    print(f"Saved {args.disclosure_policy} review tables to {args.output_dir}")
    if args.disclosure_policy == "primary-only":
        print(
            "NOT CLEARED FOR RELEASE: primary-only masking does not prevent reconstruction from totals or other cells."
        )


if __name__ == "__main__":
    main()
