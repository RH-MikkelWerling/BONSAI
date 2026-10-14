"""INTERNAL aggregate audit of CLL/MM first-line seed use across disease labels.

This reproduces the script's line-1 seed precedence (earliest date per patient)
and reports where selected seeds cross the eventual mapped disease label.
It does not change or adjudicate any patient-level first-line assignment.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


def _hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _seeds(path: Path, disease: str) -> pd.DataFrame:
    frame = pd.read_parquet(path)
    needed = {"patientid", "line"}
    if not needed.issubset(frame):
        raise ValueError(f"{path}: missing seed columns {sorted(needed - set(frame))}")
    date_column = "date_start_line" if "date_start_line" in frame else "date_start"
    if date_column not in frame:
        raise ValueError(f"{path}: missing date_start_line/date_start")
    result = frame.loc[frame["line"].eq(1), ["patientid", date_column]].copy()
    result["patientid"] = result["patientid"].astype("string").str.strip()
    result["seed_date"] = pd.to_datetime(result[date_column], errors="coerce").dt.normalize()
    result = result.loc[result["patientid"].notna() & result["seed_date"].notna()]
    result = (result.sort_values(["patientid", "seed_date"], kind="stable")
              .drop_duplicates("patientid"))
    result["seed_disease"] = disease
    return result[["patientid", "seed_date", "seed_disease"]]


def audit(mapped_path: Path, cll_path: Path, mm_path: Path,
          old_mapped_path: Path | None = None) -> dict:
    mapped = pd.read_csv(mapped_path, low_memory=False)
    needed = {"patientid", "firstline_date", "cohort_fine", "firstline_source"}
    if not needed.issubset(mapped) or mapped["patientid"].isna().any():
        raise ValueError(f"Mapped CSV lacks {sorted(needed - set(mapped))} or has missing IDs")
    mapped["patientid"] = mapped["patientid"].astype("string").str.strip()
    if mapped["patientid"].duplicated().any():
        raise ValueError("Mapped CSV has duplicate patient IDs")
    mapped["firstline_date"] = pd.to_datetime(mapped["firstline_date"], errors="coerce").dt.normalize()
    cll_seeds, mm_seeds = _seeds(cll_path, "CLL"), _seeds(mm_path, "MM")
    seeds = pd.concat([cll_seeds, mm_seeds], ignore_index=True)
    both_source_ids = len(set(cll_seeds["patientid"]).intersection(mm_seeds["patientid"]))
    # Original R uses arrange(patient_id, seed_date) then distinct(patient_id).
    chosen = seeds.sort_values(["patientid", "seed_date"], kind="stable").drop_duplicates("patientid")
    chosen = chosen.rename(columns={"seed_date": "audit_seed_date",
                                    "seed_disease": "audit_seed_disease"})
    joined = mapped.merge(chosen, on="patientid", how="left", validate="one_to_one")
    cohort = joined["cohort_fine"].astype("string")
    mismatch = (
        (joined["audit_seed_disease"].eq("CLL") & ~cohort.isin(["CLL", "SLL"]))
        | (joined["audit_seed_disease"].eq("MM") & ~cohort.isin(["MM", "PCD", "PCL", "SolM"]))
    ).fillna(False)
    clinical = joined["firstline_source"].eq("clinical_seed").fillna(False)
    near = (joined["firstline_date"] - joined["audit_seed_date"]).dt.days.abs().le(30).fillna(False)
    rows = []
    for disease in ("CLL", "MM"):
        selected = joined["audit_seed_disease"].eq(disease).fillna(False)
        rows.append({
            "seed_disease": disease,
            "n_selected_seed_in_mapped": int(selected.sum()),
            "n_seed_cohort_mismatch": int((selected & mismatch).sum()),
            "n_seed_cohort_mismatch_and_clinical_source": int((selected & mismatch & clinical).sum()),
            "n_seed_cohort_mismatch_clinical_source_near_30d": int((selected & mismatch & clinical & near).sum()),
        })
    result = {
        "status": "internal_diagnostic_not_clinical_adjudication",
        "n_patients_in_both_seed_sources": int(both_source_ids),
        "n_mapped": len(mapped),
        "inputs_sha256": {str(path.resolve()): _hash(path) for path in (mapped_path, cll_path, mm_path)},
        "rows": rows,
    }
    if old_mapped_path is not None:
        old = pd.read_csv(old_mapped_path, low_memory=False)
        required_old = {"patientid", "firstline_date", "cohort_fine", "firstline_source"}
        if not required_old.issubset(old) or old["patientid"].isna().any():
            raise ValueError("Old mapped CSV lacks comparison columns or has missing IDs")
        old["patientid"] = old["patientid"].astype("string").str.strip()
        if old["patientid"].duplicated().any():
            raise ValueError("Old mapped CSV has duplicate patient IDs")
        old["firstline_date"] = pd.to_datetime(old["firstline_date"], errors="coerce").dt.normalize()
        if "seed_date" in old:
            old["seed_date"] = pd.to_datetime(old["seed_date"], errors="coerce").dt.normalize()
        if "seed_date" in joined:
            joined["seed_date"] = pd.to_datetime(joined["seed_date"], errors="coerce").dt.normalize()
        old_columns = ["patientid", "firstline_date", "cohort_fine", "firstline_source"]
        if "seed_date" in old and "seed_date" in joined:
            old_columns.append("seed_date")
        comparison = joined.merge(
            old[old_columns],
            on="patientid", how="left", validate="one_to_one", suffixes=("", "_old"),
            indicator=True,
        )
        shared = comparison["_merge"].eq("both")
        date_changed = comparison["firstline_date"].ne(comparison["firstline_date_old"]).fillna(False)
        source_changed = comparison["firstline_source"].astype("string").ne(
            comparison["firstline_source_old"].astype("string")).fillna(False)
        cohort_changed = comparison["cohort_fine"].astype("string").ne(
            comparison["cohort_fine_old"].astype("string")).fillna(False)
        overlap = []
        for label in ("CLL", "MM", "no_selected_seed"):
            selected = (comparison["audit_seed_disease"].eq(label).fillna(False)
                        if label != "no_selected_seed" else comparison["audit_seed_disease"].isna())
            transitions = comparison.loc[selected & shared & source_changed,
                                         ["firstline_source_old", "firstline_source"]].fillna("[missing]")
            source_transitions = [
                {"old_source": str(old_source), "new_source": str(new_source), "n": int(n)}
                for (old_source, new_source), n in transitions.groupby(
                    ["firstline_source_old", "firstline_source"], dropna=False).size().items()
            ]
            row = {
                "selected_seed": label,
                "n_shared": int((selected & shared).sum()),
                "n_changed_date_shared": int((selected & shared & date_changed).sum()),
                "n_changed_source_shared": int((selected & shared & source_changed).sum()),
                "n_changed_cohort_fine_shared": int((selected & shared & cohort_changed).sum()),
                "n_changed_date_seed_cohort_mismatch_shared": int(
                    (selected & shared & date_changed & mismatch).sum()),
                "source_transitions_INTERNAL": source_transitions,
            }
            if "seed_date_old" in comparison and "seed_date" in comparison:
                old_seed, new_seed = comparison["seed_date_old"], comparison["seed_date"]
                seed_unchanged = old_seed.eq(new_seed) | (old_seed.isna() & new_seed.isna())
                seed_changed = ~seed_unchanged.fillna(False)
                row["n_old_seed_date_present_new_absent"] = int(
                    (selected & shared & old_seed.notna() & new_seed.isna()).sum())
                row["n_old_seed_date_absent_new_present"] = int(
                    (selected & shared & old_seed.isna() & new_seed.notna()).sum())
                row["n_seed_date_changed_shared"] = int((selected & shared & seed_changed).sum())
                row["n_date_changed_and_seed_date_changed_shared"] = int(
                    (selected & shared & date_changed & seed_changed).sum())
            overlap.append(row)
        result["old_new_change_overlap_INTERNAL"] = overlap
        result["inputs_sha256"][str(old_mapped_path.resolve())] = _hash(old_mapped_path)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mapped", type=Path, required=True)
    parser.add_argument("--cll-seed", type=Path, required=True)
    parser.add_argument("--mm-seed", type=Path, required=True)
    parser.add_argument("--old-mapped", type=Path,
                        help="Optional historical mapped CSV for overlap with changed dates/sources")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError(f"Refusing to overwrite {args.output}")
    result = audit(args.mapped, args.cll_seed, args.mm_seed, args.old_mapped)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    summary = {"rows": result["rows"],
               "n_patients_in_both_seed_sources": result["n_patients_in_both_seed_sources"]}
    if "old_new_change_overlap_INTERNAL" in result:
        summary["old_new_change_overlap_INTERNAL"] = result["old_new_change_overlap_INTERNAL"]
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
