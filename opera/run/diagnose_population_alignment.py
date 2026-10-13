"""Read-only source alignment audit; never changes inclusion or split labels."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from opera.run.export_table1 import read_table


def audit_alignment(population, survival):
    for name, frame in (("population", population), ("survival", survival)):
        if "subject_id" not in frame:
            raise ValueError(f"{name} lacks subject_id")
        if frame.subject_id.isna().any() or frame.subject_id.duplicated().any():
            raise ValueError(
                f"{name} has missing or duplicate IDs; resolve before set comparisons"
            )
    pop_ids, os_ids = set(population.subject_id), set(survival.subject_id)
    missing = population.loc[~population.subject_id.isin(os_ids)].copy()
    extra = survival.loc[~survival.subject_id.isin(pop_ids)].copy()
    shared = population.merge(
        survival, on="subject_id", suffixes=("_population", "_survival")
    )
    report = {
        "population_rows": len(population),
        "survival_rows": len(survival),
        "population_id_dtype": str(population.subject_id.dtype),
        "survival_id_dtype": str(survival.subject_id.dtype),
        "shared_patients": len(shared),
        "population_only": len(missing),
        "survival_only": len(extra),
        "survival_split_counts": survival["split"]
        .fillna("<missing>")
        .value_counts()
        .to_dict()
        if "split" in survival
        else None,
    }
    for name, frame in (("population_only", missing), ("survival_only", extra)):
        report[f"{name}_profile"] = {
            column: frame[column]
            .fillna("<missing>")
            .astype(str)
            .value_counts()
            .to_dict()
            for column in ("cohort_grouped", "cohort_fine", "Disease", "split")
            if column in frame
        }
        report[f"{name}_missing_dates"] = {
            column: int(pd.to_datetime(frame[column], errors="coerce").isna().sum())
            for column in ("index_date", "birth_date", "censor_date")
            if column in frame
        }
    for column in ("index_date", "split"):
        left, right = f"{column}_population", f"{column}_survival"
        if {left, right} <= set(shared):
            a, b = shared[left], shared[right]
            if column == "index_date":
                a, b = (
                    pd.to_datetime(a, errors="coerce"),
                    pd.to_datetime(b, errors="coerce"),
                )
            report[f"shared_{column}_conflicts"] = int(
                (a.notna() & b.notna() & a.ne(b)).sum()
            )
    if "split" in survival:
        matched = survival.loc[survival.subject_id.isin(pop_ids)]
        report["shared_invalid_or_missing_splits"] = int(
            (~matched.split.isin(["train", "tuning", "held_out"])).sum()
        )
    return report, missing, extra


def inspect_missing_patient_audit(missing, audit):
    """Summarize available audit evidence without inferring exclusion reasons."""
    if "subject_id" not in audit:
        raise ValueError("Audit file lacks subject_id")
    selected = audit.loc[audit.subject_id.isin(missing.subject_id)].copy()
    summary = {
        "columns": list(audit.columns),
        "matched_rows": len(selected),
        "matched_patients": int(selected.subject_id.nunique()),
        "duplicate_patient_rows": int(selected.subject_id.duplicated().sum()),
        "field_counts": {},
        "date_checks": {},
    }
    for column in selected:
        name = str(column).lower()
        if column != "subject_id" and (
            name == "split"
            or any(
                term in name
                for term in (
                    "reason",
                    "eligible",
                    "eligibility",
                    "exclude",
                    "coverage",
                    "status",
                        "source",
                        "adequate",
                        "observed",
                )
            )
        ):
            summary["field_counts"][column] = (
                selected[column]
                .astype("string")
                .fillna("<missing>")
                .value_counts()
                .to_dict()
            )
    dates = {}
    for column in selected:
        if "date" in str(column).lower():
            dates[column] = pd.to_datetime(selected[column], errors="coerce")
            summary["date_checks"][column] = {
                "missing_or_invalid": int(dates[column].isna().sum()),
                "year_counts": dates[column]
                .dt.year.astype("Int64")
                .astype("string")
                .fillna("<missing>")
                .value_counts()
                .to_dict(),
            }
    if "index_date" in dates:
        for column in ("censor_date", "outcome_date"):
            if column in dates:
                valid = dates[column].notna() & dates["index_date"].notna()
                summary["date_checks"][f"{column}_relative_to_index"] = {
                    "before": int(
                        (valid & dates[column].lt(dates["index_date"])).sum()
                    ),
                    "same": int((valid & dates[column].eq(dates["index_date"])).sum()),
                    "after": int((valid & dates[column].gt(dates["index_date"])).sum()),
                }
    return summary, selected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--population", type=Path, required=True)
    parser.add_argument("--overall-survival", type=Path, required=True)
    parser.add_argument("--sequence-dir", type=Path)
    parser.add_argument(
        "--audit-file",
        type=Path,
        action="append",
        default=[],
        help="Repeatable audit files; defaults to second_line_treatment__audit.parquet if present.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    population, survival = (
        read_table(args.population),
        read_table(args.overall_survival),
    )
    report, missing, extra = audit_alignment(population, survival)
    report["inputs"] = {
        name: {
            "path": str(path.resolve()),
            "size_bytes": path.stat().st_size,
            "mtime_ns": path.stat().st_mtime_ns,
        }
        for name, path in (
            ("population", args.population),
            ("survival", args.overall_survival),
        )
    }
    outcome_rows = []
    audit_paths = args.audit_file or [
        args.overall_survival.parent / "second_line_treatment__audit.parquet"
    ]
    audit_reports, local_audits = {}, []
    for path in audit_paths:
        if not path.exists():
            if args.audit_file:
                raise FileNotFoundError(path)
            continue
        summary, selected = inspect_missing_patient_audit(missing, read_table(path))
        summary["path"] = str(path.resolve())
        audit_reports[str(path.resolve())] = summary
        local_audits.append((path.stem, selected))
    report["missing_patient_audit_evidence"] = audit_reports
    for path in sorted(args.overall_survival.parent.glob("*.parquet")):
        frame = pd.read_parquet(path, columns=["subject_id"])
        ids = set(frame.subject_id)
        outcome_rows.append(
            {
                "outcome": path.stem,
                "rows": len(frame),
                "unique_ids": len(ids),
                "duplicate_ids": int(frame.subject_id.duplicated().sum()),
                "population_covered": int(population.subject_id.isin(ids).sum()),
                "population_only_survival_patients_covered": int(
                    missing.subject_id.isin(ids).sum()
                ),
                "survival_only_patients_covered": int(extra.subject_id.isin(ids).sum()),
            }
        )
    sequence_rows = []
    if args.sequence_dir:
        import torch

        paths = sorted(args.sequence_dir.glob("subject_data_*.pt"))
        if not paths:
            raise ValueError(f"No subject_data_*.pt files in {args.sequence_dir}")
        report["sequence_dir"] = str(args.sequence_dir.resolve())
        for path in paths:
            print(
                f"Reading {path.name} on CPU (may require substantial RAM)", flush=True
            )
            subjects = torch.load(path, map_location="cpu", weights_only=False)
            ids = [subject["subject_id"] for subject in subjects]
            id_set = set(ids)
            missing[f"in_{path.stem}"] = missing.subject_id.isin(id_set)
            extra[f"in_{path.stem}"] = extra.subject_id.isin(id_set)
            sequence_rows.append(
                {
                    "file": path.name,
                    "patients": len(ids),
                    "unique_ids": len(id_set),
                    "population_covered": int(population.subject_id.isin(id_set).sum()),
                    "population_only_survival_patients_covered": int(
                        missing.subject_id.isin(id_set).sum()
                    ),
                }
            )
            del subjects
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for stem, selected in local_audits:
        selected.to_csv(
            args.output_dir / f"LOCAL_ONLY_missing_patients_{stem}.csv", index=False
        )
    (args.output_dir / "alignment_summary.json").write_text(
        json.dumps(report, indent=2, default=str), encoding="utf-8"
    )
    pd.DataFrame(outcome_rows).to_csv(
        args.output_dir / "outcome_coverage.csv", index=False
    )
    if sequence_rows:
        pd.DataFrame(sequence_rows).to_csv(
            args.output_dir / "sequence_coverage.csv", index=False
        )
    # Identifiable records stay on server; only aggregate reports should be shared.
    missing.to_csv(
        args.output_dir / "LOCAL_ONLY_population_without_survival.csv", index=False
    )
    extra.to_csv(
        args.output_dir / "LOCAL_ONLY_survival_without_population.csv", index=False
    )
    print(json.dumps(report, indent=2, default=str))
    print(
        f"Saved diagnostics to {args.output_dir}. Keep LOCAL_ONLY files on the server."
    )


if __name__ == "__main__":
    main()
