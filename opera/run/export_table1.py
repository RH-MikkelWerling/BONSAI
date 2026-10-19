"""Export aggregate Table 1 values from a frozen one-row-per-patient cohort."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

# Only explicit columns are used. No outcomes, sex codes or stages are inferred.
CONTINUOUS = {
    "age": ("Age at first-line treatment, years", 1),
    "observable_history_years": ("Recorded pre-index history span, years", 1),
    "recorded_events": ("Recorded events", 1),
    "hospital_encounters": ("Hospital encounters", 1),
    "n_unique_specialties": ("Distinct contact specialties in the baseline window", 1),
    "distinct_diagnoses": ("Distinct diagnoses", 1),
    "distinct_medications": ("Distinct medications", 1),
    "laboratory_measurements": ("Pre-index laboratory-coded records", 1),
    "comorbidity_score": ("Comorbidity burden", 1),
    "days_since_last_preindex_event": ("Days from last recorded event to treatment", 1),
    "diagnosis_to_index_days": ("Diagnosis to first-line treatment, days", 1),
    "followup_days": ("Observed follow-up duration, years", 365.25),
}
CATEGORICAL = [
    "sex",
    "predominant_region",
    "predominant_region_status",
    "stage",
    "ecog",
    "ipi",
    "iss",
    "r_iss",
    "cytogenetic_risk",
    "treatment_regimen",
]
CLINICAL_CATEGORICAL = {
    "index",
    "ps",
    "disease",
    "arm_primary",
    "treatment_group",
    "treatment_groups",
    "treatment_grouped",
    "treatment_category",
    "regimen",
    "regimen_group",
    "regimen_category",
    "first_line_regimen",
    "first_line_treatment_group",
    "performance_status",
    "performance_score",
    "performance_status_score",
    "who_performance_status",
    "ecog_score",
    "ecog_performance_status",
    "ann_arbor_stage",
    "disease_stage",
    "clinical_stage",
    "b_symptoms",
    "ipi_score",
    "ipi_risk_group",
    "nccn_ipi",
    "nccn_ipi_score",
    "nccn_ipi_group",
    "cohort_source",
    "age_adjusted_ipi",
    "cll_ipi",
    "flipi",
    "flipi_score",
    "flipi2",
    "flipi_risk_group",
    "mipi",
    "mipi_score",
    "mipi_risk_group",
    "iss_stage",
    "r_iss_stage",
}


def clinical_categorical_columns(frame):
    """Recognize field names, but never merge different prognostic systems."""
    selected = set(CATEGORICAL)
    for column in frame:
        normalized = re.sub(r"[^a-z0-9]+", "_", str(column).lower()).strip("_")
        if normalized in CLINICAL_CATEGORICAL or normalized in CATEGORICAL:
            selected.add(column)
    return sorted(selected, key=str)


BINARY = {
    "prior_non_hematological_malignancy": "Prior non-haematological malignancy",
    "death_event": "Death during available follow-up",
    "female": "Female sex",
}


def read_table(path):
    return (
        pd.read_parquet(path)
        if Path(path).suffix.lower() in {".parquet", ".pq"}
        else pd.read_csv(path)
    )


def prepare_survival_table(survival):
    if "subject_id" not in survival or survival.subject_id.duplicated().any():
        raise ValueError("Overall survival must have one record per patient.")
    if "outcome" in survival and not survival.outcome.eq("overall_survival").all():
        raise ValueError(
            "Only overall_survival records may supply follow-up/death summaries."
        )
    if not {"time_days", "event"} <= set(survival):
        from opera.compat.bonsai import binarize_outcomes

        records = binarize_outcomes(
            survival, n_hours_start_include=1, n_hours_end_include=None
        )
        survival = survival.copy()
        survival["time_days"] = survival.subject_id.map(
            lambda sid: records.get(sid, {}).get("time_days", np.nan)
        )
        survival["event"] = survival.subject_id.map(
            lambda sid: records.get(sid, {}).get("event", np.nan)
        )
    keep = [
        c
        for c in ["subject_id", "split", "index_date", "time_days", "event"]
        if c in survival
    ]
    return survival[keep].rename(
        columns={"time_days": "followup_days", "event": "death_event"}
    )


def join_metadata(population, metadata):
    """Patient scope comes only from population; metadata may not add patients."""
    frame = population.copy()
    if (
        "subject_id" not in frame
        or frame.subject_id.isna().any()
        or frame.subject_id.duplicated().any()
    ):
        raise ValueError(
            "Frozen population must have one nonmissing subject_id per patient."
        )
    for source in metadata:
        if (
            "subject_id" not in source
            or source.subject_id.isna().any()
            or source.subject_id.duplicated().any()
        ):
            raise ValueError(
                "Metadata must have one nonmissing subject_id per patient."
            )
        extra = source.set_index("subject_id").reindex(frame.subject_id)
        for column in extra:
            supplied = extra[column].to_numpy()
            if column in frame:
                existing = frame[column]
                supplied = pd.Series(supplied, index=frame.index)
                conflict = (
                    existing.notna()
                    & supplied.notna()
                    & (existing.astype(str) != supplied.astype(str))
                )
                if conflict.any():
                    raise ValueError(
                        f"Conflicting metadata column: {column}; resolve before exporting."
                    )
                frame[column] = existing.fillna(supplied)
            else:
                frame[column] = supplied
    return frame


def restrict_to_survival_population(population, survival):
    """Explicit intersection only; split validation remains mandatory downstream."""
    for source in (population, survival):
        if (
            "subject_id" not in source
            or source.subject_id.isna().any()
            or source.subject_id.duplicated().any()
        ):
            raise ValueError(
                "Population and survival require unique nonmissing subject_id."
            )
    included = population.subject_id.isin(survival.subject_id)
    audit = {
        "input_population_patients": len(population),
        "included_patients": int(included.sum()),
        "excluded_population_patients": int((~included).sum()),
        "survival_only_patients": int(
            (~survival.subject_id.isin(population.subject_id)).sum()
        ),
        "scope": "explicit population/overall-survival intersection",
    }
    return population.loc[included].copy(), audit


SPLITS = ("train", "tuning", "held_out")


def supervised_population(population):
    """Patients with a first-line index date and a supervised split, one row each.

    The diagnosis-indexed population metadata also holds patients who never
    started first-line treatment (no index date) or carry no split; Table 1
    describes the supervised cohort only. Exclusions are counted, never silent.
    """
    if {"subject_id", "index_date", "split"} - set(population):
        raise ValueError("--supervised-only needs subject_id, index_date and split columns")
    index = pd.to_datetime(population["index_date"], errors="coerce")
    split = population["split"].astype("string")
    has_split = split.isin(SPLITS)
    keep = index.notna() & has_split
    included = population.loc[keep].copy()
    duplicated = int(included.subject_id.duplicated().sum())
    audit = {
        "scope": "supervised first-line population (index date and split)",
        "input_population_rows": len(population),
        "excluded_without_index_date": int(index.isna().sum()),
        "excluded_with_index_but_no_split": int((index.notna() & ~has_split).sum()),
        "supervised_patients": int(len(included)),
        "by_split": {k: int(v) for k, v in included.split.value_counts().items()},
    }
    if duplicated:
        raise ValueError(f"{duplicated} supervised patients have more than one row; "
                         f"Table 1 needs one row per patient (audit: {audit})")
    return included, audit


def derive_baseline_fields(frame):
    frame = frame.copy()
    if "age" not in frame and {"birth_date", "index_date"} <= set(frame):
        frame["age"] = (
            pd.to_datetime(frame.index_date, errors="raise")
            - pd.to_datetime(frame.birth_date, errors="raise")
        ).dt.total_seconds() / (365.25 * 86400)
        if frame.age.lt(0).any():
            raise ValueError("Birth date occurs after index date.")
    return frame


def summarize(frame):
    rows = []

    def add(variable, label, values, kind, level=None):
        known = values.notna()
        row = dict(
            variable=variable,
            characteristic=label,
            category=level,
            kind=kind,
            n_total=len(frame),
            n_observed=int(known.sum()),
            n_missing=int((~known).sum()),
        )
        if kind == "continuous":
            valid = values[known].astype(float)
            row.update(
                median=valid.median(), q1=valid.quantile(0.25), q3=valid.quantile(0.75)
            )
            row["display"] = (
                f"{row['median']:.1f} ({row['q1']:.1f}–{row['q3']:.1f})"
                if known.any()
                else "Not available"
            )
        else:
            count = int((values[known] == level).sum())
            row.update(
                count=count,
                percent=100 * count / known.sum() if known.any() else np.nan,
            )
            row["display"] = (
                f"{count:,} ({row['percent']:.1f}%)" if known.any() else "Not available"
            )
        rows.append(row)

    rows.append(
        dict(
            variable="patients",
            characteristic="Patients",
            kind="count",
            n_total=len(frame),
            n_observed=len(frame),
            n_missing=0,
            count=len(frame),
            display=f"{len(frame):,}",
        )
    )
    for column, (label, divisor) in CONTINUOUS.items():
        values = (
            pd.to_numeric(frame[column], errors="raise") / divisor
            if column in frame
            else pd.Series(np.nan, index=frame.index)
        )
        if np.isinf(values.dropna()).any():
            raise ValueError(f"Infinite values in {column}.")
        add(column, label, values, "continuous")
        if column == "age":
            add(
                "age_ge70",
                "Age ≥70 years",
                (values >= 70).where(values.notna()),
                "binary",
                True,
            )
    if {"followup_days", "death_event"} <= set(frame):
        # Reverse Kaplan-Meier: deaths do not shorten the follow-up summary.
        from opera.evaluation.followup import median_followup

        observed = frame.followup_days.notna() & frame.death_event.notna()
        summary = median_followup(frame.followup_days[observed], frame.death_event[observed])
        years = {key: value / 365.25 for key, value in summary.items() if key != "n"}
        rows.append(
            dict(
                variable="followup_reverse_km",
                characteristic="Median follow-up (reverse Kaplan-Meier), years",
                kind="continuous",
                n_total=len(frame),
                n_observed=summary["n"],
                n_missing=len(frame) - summary["n"],
                median=years["median_days"],
                q1=years["q25_days"],
                q3=years["q75_days"],
                display=(
                    f"{years['median_days']:.1f} ({years['q25_days']:.1f}–{years['q75_days']:.1f})"
                    if np.isfinite([years["median_days"], years["q25_days"], years["q75_days"]]).all()
                    else "Not reached"
                ),
            )
        )
    if "index_date" in frame:
        years = pd.to_datetime(frame.index_date, errors="raise").dt.year
        add("index_year", "Calendar year of first-line treatment", years, "continuous")
    for column, label in BINARY.items():
        if column == "female" and ("sex" in frame or "female" not in frame):
            continue
        values = (
            frame[column] if column in frame else pd.Series(np.nan, index=frame.index)
        )
        if not values.dropna().isin([0, 1, False, True]).all():
            raise ValueError(
                f"{column} must be explicitly coded 0/1; no code guessing."
            )
        add(column, label, values, "binary", 1)
    for column in sorted(c for c in frame if str(c).startswith("contact_specialty__")):
        # Share of patients with any pre-index contact in each common specialty.
        values = pd.to_numeric(frame[column], errors="raise")
        add(column, "Pre-index contact: " + column.split("__", 1)[1].replace("_", " "),
            values, "binary", 1)
    for column in [
        "cohort_grouped",
        "cohort_fine",
        *clinical_categorical_columns(frame),
    ]:
        if column not in frame:
            continue
        values = frame[column]
        if column == "predominant_region":
            # Display the region name, not its MEDS code prefix.
            values = values.astype("string").str.split("//", n=1).str[-1]
        for level in sorted(values.dropna().unique(), key=str):
            label = {
                "index": "Prognostic scoring system",
                "nccn_ipi_group": "NCCN-IPI risk group",
                "nccn_ipi_score": "NCCN-IPI score",
                "cohort_source": "Subtype assignment (pass)",
                "ipi": "Prognostic score value (IPI field)",
            }.get(str(column).lower(), column.replace("_", " ").title())
            add(column, label, values, "categorical", level)
    if {"IPI", "index"} <= set(frame):
        # Preserve the system/value pairing; do not compare unlike scores as one scale.
        values = frame["index"].astype("string") + ": " + frame["IPI"].astype("string")
        for level in sorted(values.dropna().unique(), key=str):
            add(
                "prognostic_score_by_system",
                "Prognostic score by scoring system",
                values,
                "categorical",
                level,
            )
    return pd.DataFrame(rows)


def build_tables(frame):
    if frame.empty:
        raise ValueError("Frozen population is empty.")
    if (
        "split" not in frame
        or not frame.split.isin(["train", "tuning", "held_out"]).all()
    ):
        counts = (
            frame["split"].fillna("<missing>").astype(str).value_counts().to_dict()
            if "split" in frame
            else "column absent"
        )
        raise ValueError(
            f"Provide frozen split labels train/tuning/held_out; found: {counts}. "
            "Use --split-manifest for existing patient assignments or --split-map "
            "for explicit label aliases. Missing assignments are not dropped or reconstructed from dates."
        )
    tables = []
    for level in ("all", "cohort_grouped", "cohort_fine"):
        if level != "all" and level not in frame:
            continue
        selections = (
            [("All", frame)]
            if level == "all"
            else list(frame.groupby(level, dropna=False))
        )
        for cohort, cohort_frame in selections:
            for split in ("Overall", "train", "tuning", "held_out"):
                selected = (
                    cohort_frame
                    if split == "Overall"
                    else cohort_frame.loc[cohort_frame.split == split]
                )
                if selected.empty:
                    continue
                table = summarize(selected)
                table["cohort_level"], table["cohort"], table["split"] = (
                    level,
                    str(cohort),
                    split,
                )
                tables.append(table)
    return pd.concat(tables, ignore_index=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--population",
        type=Path,
        required=True,
        help="Frozen included cohort; one row per patient.",
    )
    parser.add_argument(
        "--metadata", "--population-metadata", type=Path, action="append", default=[]
    )
    parser.add_argument(
        "--overall-survival",
        type=Path,
        help="Optional overall_survival export with subject_id, time_days and event (0/1).",
    )
    parser.add_argument(
        "--column-map",
        type=Path,
        help="JSON mapping canonical names to actual source column names.",
    )
    parser.add_argument(
        "--split-manifest",
        type=Path,
        help="Existing frozen table with subject_id and split; must cover the population.",
    )
    parser.add_argument(
        "--split-map",
        help='Explicit JSON label aliases, e.g. {"validation":"tuning","test":"held_out"}.',
    )
    parser.add_argument(
        "--inspect-inputs",
        action="store_true",
        help="Print aggregate input coverage/split labels without generating tables.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--meds-dir", type=Path, help="Event-level MEDS root or data directory; physical shards are pooled before the supervised split is applied."
    )
    parser.add_argument(
        "--meds-code-prefixes",
        type=Path,
        help="JSON with explicit medication/diagnosis/laboratory code-prefix lists.",
    )
    parser.add_argument(
        "--history-lookback-days",
        type=int,
        help="Optional fixed pre-index window; default all available pre-index records.",
    )
    parser.add_argument("--suppression-threshold", type=int, default=5)
    parser.add_argument(
        "--disclosure-policy",
        choices=["primary-only", "targeted", "conservative"],
        default="targeted",
    )
    parser.add_argument(
        "--supervised-only",
        action="store_true",
        help="Keep patients with an index date and a train/tuning/held_out split "
        "(the supervised cohort), counting exclusions.",
    )
    parser.add_argument(
        "--restrict-to-overall-survival",
        action="store_true",
        help="Explicitly restrict population to survival-file IDs and record exclusions. Use only if this defines the analysis cohort.",
    )
    args = parser.parse_args()
    print(f"Table 1 exporter (sex-categorical v2): {Path(__file__).resolve()}")
    print(f"Population input: {args.population.resolve()}")
    if args.inspect_inputs:
        population = read_table(args.population)
        if args.supervised_only:
            population, audit = supervised_population(population)
            print("Population scope:", json.dumps(audit))
        for path in dict.fromkeys(
            [
                args.population,
                *args.metadata,
                *([args.overall_survival] if args.overall_survival else []),
                *([args.split_manifest] if args.split_manifest else []),
            ]
        ):
            source = read_table(path)
            print(f"\n{path}\nRows: {len(source):,}; columns: {list(source.columns)}")
            print(
                "Split labels:",
                source["split"].fillna("<missing>").astype(str).value_counts().to_dict()
                if "split" in source
                else "No split column",
            )
            if "subject_id" in source and "subject_id" in population:
                print(
                    "Population patients covered:",
                    int(population.subject_id.isin(source.subject_id).sum()),
                    "of",
                    len(population),
                    "; duplicate IDs:",
                    int(source.subject_id.duplicated().sum()),
                )
        return
    sources = [read_table(path) for path in args.metadata]
    if args.overall_survival:
        sources.append(prepare_survival_table(read_table(args.overall_survival)))
    if args.split_manifest:
        manifest = read_table(args.split_manifest)
        if {"subject_id", "split"} - set(manifest):
            raise ValueError("Split manifest requires subject_id and split.")
        sources.append(manifest[["subject_id", "split"]])
    population = read_table(args.population)
    scope_audit = {
        "scope": "full supplied population",
        "input_population_patients": len(population),
    }
    if args.restrict_to_overall_survival:
        if not args.overall_survival:
            raise ValueError(
                "--restrict-to-overall-survival requires --overall-survival."
            )
        population, scope_audit = restrict_to_survival_population(
            population, read_table(args.overall_survival)
        )
        print("Population scope:", json.dumps(scope_audit))
    split_mapping = json.loads(args.split_map) if args.split_map else {}
    if split_mapping:
        if not set(split_mapping.values()) <= {"train", "tuning", "held_out"}:
            raise ValueError("Split aliases must map to train/tuning/held_out.")
        for source in [population, *sources]:
            if "split" in source:
                source["split"] = source["split"].replace(split_mapping)
    if args.supervised_only:
        population, scope_audit = supervised_population(population)
        print("Population scope:", json.dumps(scope_audit))
    frame = join_metadata(population, sources)
    mapping = json.loads(args.column_map.read_text()) if args.column_map else {}
    for canonical, source in mapping.items():
        if source not in frame:
            raise ValueError(f"Mapped source column does not exist: {source}")
        if canonical != source and canonical in frame:
            raise ValueError(f"Canonical column already exists: {canonical}")
    frame = frame.rename(
        columns={source: canonical for canonical, source in mapping.items()}
    )
    if split_mapping:
        if not set(split_mapping.values()) <= {"train", "tuning", "held_out"}:
            raise ValueError("Split aliases must map to train/tuning/held_out.")
        if "split" not in frame:
            raise ValueError("No split column to map.")
        frame["split"] = frame["split"].replace(split_mapping)
    frame = derive_baseline_fields(frame)
    if "sex" in frame:
        print(
            "Sex values in joined metadata:",
            frame["sex"].astype("string").fillna("<missing>").value_counts().to_dict(),
        )
    else:
        print(
            "WARNING: no exact 'sex' column in joined metadata. Available columns:",
            list(frame.columns),
        )
    meds_provenance = None
    if args.meds_dir:
        from opera.functional.table1_meds import (
            raw_meds_batches,
            summarize_meds_batches,
        )

        if not args.meds_code_prefixes:
            raise ValueError(
                "--meds-dir requires --meds-code-prefixes; code systems are never guessed."
            )
        prefixes = json.loads(args.meds_code_prefixes.read_text(encoding="utf-8"))
        history = summarize_meds_batches(
            frame, raw_meds_batches(args.meds_dir), prefixes, args.history_lookback_days
        )
        frame = join_metadata(frame, [history])
        history_coverage = {
            str(split): {
                "patients": int(len(group)),
                "patients_with_preindex_events": int(group.recorded_events.notna().sum()),
                "median_days_since_last_preindex_event": float(group.days_since_last_preindex_event.median())
                if group.days_since_last_preindex_event.notna().any() else None,
                "patients_with_region": int(group.predominant_region.notna().sum())
                if "predominant_region" in group else None,
                "patients_with_specialty": int(group.n_unique_specialties.notna().sum())
                if "n_unique_specialties" in group else None,
            }
            for split, group in frame.groupby("split", dropna=False)
        }
        print("MEDS history coverage by supervised split:", json.dumps(history_coverage))
        if any(item["patients_with_preindex_events"] == 0 for item in history_coverage.values()):
            raise ValueError("A supervised split has zero pre-index MEDS history; inspect shard coverage")
        meds_provenance = {
            "root": str(args.meds_dir.resolve()),
            "prefixes": prefixes,
            "lookback_days": args.history_lookback_days,
            "supervised_split_coverage": history_coverage,
            "index_boundary": "strictly before index_date",
            "distinct_definition": "unique complete event-level MEDS codes; laboratory/event counts are event rows, not encounters; binning requires source audit",
            "no_usable_history": "missing, not zero",
        }
    elif args.meds_code_prefixes or args.history_lookback_days:
        raise ValueError("MEDS options require --meds-dir")
    table = build_tables(frame)
    from opera.functional.table1_release import release_table

    release, release_policy = release_table(
        table, args.suppression_threshold, args.disclosure_policy
    )
    internal_dir = args.output_dir / "LOCAL_ONLY_internal"
    release_dir = args.output_dir / "release_review"
    internal_dir.mkdir(parents=True, exist_ok=True)
    release_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(internal_dir / "table1_values_long.csv", index=False)
    internal_pooled = table.loc[table.cohort_level == "all"].copy()
    internal_pooled["row"] = (
        internal_pooled.characteristic
        + internal_pooled.category.map(lambda x: f": {x}" if pd.notna(x) else "")
    )
    internal_pooled.pivot(
        index=["variable", "row"], columns="split", values="display"
    ).reindex(columns=["Overall", "train", "tuning", "held_out"]).to_csv(
        internal_dir / "table1_pooled_display.csv"
    )
    release.to_csv(release_dir / "table1_values_long.csv", index=False)
    pooled = release.loc[release.cohort_level == "all"].copy()
    pooled["row"] = pooled.characteristic + pooled.category.map(
        lambda x: f": {x}" if pd.notna(x) else ""
    )
    pooled.pivot(index=["variable", "row"], columns="split", values="display").reindex(
        columns=["Overall", "train", "tuning", "held_out"]
    ).to_csv(release_dir / "table1_pooled_display.csv")
    (release_dir / "disclosure_policy.json").write_text(
        json.dumps(release_policy, indent=2), encoding="utf-8"
    )
    recognized = (
        set(CONTINUOUS)
        | set(BINARY)
        | set(clinical_categorical_columns(frame))
        | {"cohort_grouped", "cohort_fine", "index_date"}
        | {c for c in frame if str(c).startswith("contact_specialty__")}
    )
    pd.DataFrame(
        [
            {
                "column": c,
                "dtype": str(frame[c].dtype),
                "n_observed": int(frame[c].notna().sum()),
                "n_unique": int(frame[c].nunique()),
                "included_in_summaries": c in recognized,
            }
            for c in frame
        ]
    ).to_csv(internal_dir / "metadata_inventory.csv", index=False)
    (internal_dir / "provenance.json").write_text(
        json.dumps(
            {
                "population": str(args.population.resolve()),
                "metadata": [str(p.resolve()) for p in args.metadata],
                "meds_history": meds_provenance,
                "column_map": mapping,
                "n_patients": len(frame),
                "population_scope": scope_audit,
                "age_derivation": "(index_date - birth_date) / 365.25 days when age is not supplied",
                "split_manifest": str(args.split_manifest.resolve())
                if args.split_manifest
                else None,
                "split_map": split_mapping,
                "overall_survival": str(args.overall_survival.resolve())
                if args.overall_survival
                else None,
                "percent_denominator": "nonmissing patients in the relevant cohort/split",
                "notes": "Baseline fields must refer to index or earlier. followup_days/death_event must be overall-survival follow-up. No imputation. No model predictions used.",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(
        f"Exported {len(frame):,} patients, {len(table):,} aggregate rows to {args.output_dir}"
    )
    print(
        "Only release_review is intended for disclosure review. Keep LOCAL_ONLY_internal and older unsuppressed exports on server."
    )
    print(
        f"Exact formatted table: {(internal_dir / 'table1_pooled_display.csv').resolve()}"
    )


if __name__ == "__main__":
    main()
