"""Does a change to the first-line script change anyone's first line?

Compares two first-line exports (``firstline_tbl_mapped_CANDIDATE.csv`` from
``opera/R/run_firstline_candidate_export.R``), typically the script as it ran
and a corrected version, patient by patient. Prints counts only: patients in
either export, and among those in both, how many changed index date, disease
(fine and grouped) or first-line arm. No patient identifiers are printed.

    python -m opera.run.compare_firstline_exports BEFORE.csv AFTER.csv
"""

from __future__ import annotations

import argparse
import json

import pandas as pd

FIELDS = ("firstline_date", "cohort_fine", "cohort_grouped", "arm_primary")


def compare(before: pd.DataFrame, after: pd.DataFrame) -> dict:
    keep = ["patientid", *[f for f in FIELDS if f in before and f in after]]
    joined = before[keep].merge(after[keep], on="patientid", how="outer",
                                suffixes=("_before", "_after"), indicator=True)
    both = joined.loc[joined._merge == "both"]
    report = {
        "patients_before": int(len(before)),
        "patients_after": int(len(after)),
        "only_before": int((joined._merge == "left_only").sum()),
        "only_after": int((joined._merge == "right_only").sum()),
        "in_both": int(len(both)),
    }
    for field in keep[1:]:
        a = both[f"{field}_before"].astype("string").fillna("<NA>")
        b = both[f"{field}_after"].astype("string").fillna("<NA>")
        report[f"changed_{field}"] = int((a != b).sum())
    report["identical"] = (report["only_before"] == report["only_after"] == 0
                           and all(v == 0 for k, v in report.items()
                                   if k.startswith("changed_")))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("before")
    parser.add_argument("after")
    args = parser.parse_args()
    read = lambda path: pd.read_csv(path, dtype={"patientid": str})
    print(json.dumps(compare(read(args.before), read(args.after)), indent=2))


if __name__ == "__main__":
    main()
