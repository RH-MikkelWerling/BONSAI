"""INTERNAL early-treatment TLS sensitivity labels; never overwrite historical outcomes.

Absolute-threshold, same-calendar-day approximation to CTCAE-v6 laboratory
TLS. Not a clinically adjudicated diagnosis. Events occur on days 1-7; the
prediction/evaluation horizon is 30 days. Patient-level outputs stay secure.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

from opera.run.decide_outcome_panel import support_for_outcome
from opera.run.export_candidate_outcomes_v1 import _read_outcome


AXES = ("urate", "potassium", "phosphate", "calcium")
EXPECTED_UNITS = {
    "urate": "µmol/l", "potassium": "mmol/l", "phosphate": "mmol/l",
    "calcium": "mmol/l", "creatinine": "µmol/l",
}
LIMITS = {"urate": 476.0, "potassium": 6.0, "phosphate": 1.45, "calcium": 1.75}
NAMES = ("tls_laboratory_early_v1", "tls_laboratory_with_kidney_early_v1")
STRICT_NAMES = ("tls_laboratory_complete_panel_early_v1",
                "tls_laboratory_with_kidney_complete_panel_early_v1")
MAX_LAB_GAP_DAYS = 180


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_labs(frame: pd.DataFrame) -> pd.DataFrame:
    required = {"subject_id", "analyte", "sample_date", "value", "unit", "ref_upper"}
    if not required.issubset(frame) or frame["subject_id"].isna().any():
        raise ValueError(f"Missing TLS columns or IDs: {sorted(required - set(frame))}")
    frame = frame.loc[frame["analyte"].isin(EXPECTED_UNITS)].copy()
    if set(frame["analyte"]) != set(EXPECTED_UNITS):
        raise ValueError("All five TLS analytes must be represented")
    frame["sample_date"] = pd.to_datetime(frame["sample_date"], errors="coerce").dt.normalize()
    frame["value"] = pd.to_numeric(frame["value"], errors="coerce")
    frame["ref_upper"] = pd.to_numeric(frame["ref_upper"], errors="coerce")
    if frame["sample_date"].isna().any() or (frame["value"].dropna() < 0).any():
        raise ValueError("Invalid TLS lab date or negative value")
    # Clinical exports use both U+00B5 MICRO SIGN and U+03BC GREEK MU for
    # micromol/L. These are spelling variants, not a unit-scale conversion.
    frame["unit"] = (
        frame["unit"].astype("string").str.strip().str.casefold()
        .str.replace("\u03bc", "\u00b5", regex=False)
    )
    # The source may report urate in mmol/L while the absolute TLS threshold
    # is expressed in micromol/L. Convert the measured value, not the label.
    urate_mmol = frame["analyte"].eq("urate") & frame["unit"].eq("mmol/l")
    urate_mmol_median = frame.loc[urate_mmol, "value"].median()
    if pd.notna(urate_mmol_median) and urate_mmol_median > 10:
        raise ValueError("Urate labelled mmol/L has implausible median; check source unit/value scale")
    frame.loc[urate_mmol, "value"] *= 1000.0
    frame.loc[urate_mmol, "unit"] = EXPECTED_UNITS["urate"]
    for analyte, expected in EXPECTED_UNITS.items():
        units = set(frame.loc[frame["analyte"].eq(analyte) & frame["value"].notna(), "unit"].dropna())
        if units != {expected}:
            raise ValueError(f"{analyte}: unit inventory {sorted(units)} != {expected}")
    return frame.loc[frame["value"].notna()].copy()


def _shared_coverage(audit: pd.DataFrame, survival: pd.DataFrame) -> pd.DataFrame:
    """Reuse the R lab engine's analyte-agnostic eligibility contract."""
    required = {"subject_id", "split", "index_date", "eligible", "last_measurement_date"}
    if not required.issubset(audit) or audit["subject_id"].isna().any():
        raise ValueError(f"Invalid shared lab audit; missing {sorted(required - set(audit))}")
    if audit["subject_id"].duplicated().any() or set(audit["subject_id"]) != set(survival["subject_id"]):
        raise ValueError("Shared lab audit must have exactly one row per survival patient")
    gate = audit[list(required)].copy()
    for key in ("index_date", "last_measurement_date"):
        gate[key] = pd.to_datetime(gate[key], errors="coerce").dt.normalize()
    match = gate.merge(survival[["subject_id", "split", "index_date"]],
                       on="subject_id", suffixes=("_audit", "_survival"), validate="one_to_one")
    if (match["split_audit"].astype(str) != match["split_survival"].astype(str)).any() or (
        match["index_date_audit"] != match["index_date_survival"]
    ).any():
        raise ValueError("Shared lab audit and survival disagree on split or index date")
    if gate["eligible"].isna().any() or not gate["eligible"].isin([True, False]).all():
        raise ValueError("Shared lab audit has invalid eligibility flags")
    return gate[["subject_id", "eligible", "last_measurement_date"]]


def _first_dates(days: pd.MultiIndex) -> pd.Series:
    if len(days) == 0:
        return pd.Series(dtype="datetime64[ns]")
    return pd.Series(days.get_level_values("sample_date").to_numpy(),
                     index=days.get_level_values("subject_id")).groupby(level=0).min()


def build_tls(
    labs: pd.DataFrame, survival: pd.DataFrame, coverage_audit: pd.DataFrame,
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    """Primary candidate uses shared lab coverage; complete panel is sensitivity."""
    cohort = survival[["subject_id", "split", "index_date", "censor_date"]].copy()
    gate = _shared_coverage(coverage_audit, survival)
    cohort = cohort.merge(gate, on="subject_id", validate="one_to_one")
    labs = labs.merge(cohort, on="subject_id", validate="many_to_one")
    day = (labs["sample_date"] - labs["index_date"]).dt.days
    labs = labs.loc[day.between(1, 7) & labs["sample_date"].le(labs["censor_date"])].copy()
    labs["abnormal"] = False
    for axis, limit in LIMITS.items():
        is_axis = labs["analyte"].eq(axis)
        labs.loc[is_axis, "abnormal"] = (
            labs.loc[is_axis, "value"].le(limit) if axis == "calcium"
            else labs.loc[is_axis, "value"].ge(limit)
        )
    # Several draws of an axis on a day count as one axis, never two.
    by_day = labs.groupby(["subject_id", "sample_date"])
    coverage = by_day["analyte"].agg(lambda values: set(AXES).issubset(set(values)))
    abnormal = labs.loc[labs["abnormal"] & labs["analyte"].isin(AXES)]
    n_axes = abnormal.groupby(["subject_id", "sample_date"])["analyte"].nunique()
    # A complete panel on another date cannot validate a two-axis signal on
    # an incompletely measured date. Event and negative ascertainment use the
    # same patient-day monitoring requirement.
    signal_days = n_axes[n_axes.ge(2)].index
    strict_days = signal_days.intersection(coverage[coverage].index)
    renal = labs.loc[labs["analyte"].eq("creatinine") & labs["ref_upper"].gt(0)
                     & labs["value"].gt(labs["ref_upper"])]
    renal_days = pd.MultiIndex.from_frame(renal[["subject_id", "sample_date"]].drop_duplicates())
    complete = coverage[coverage].index
    complete_ids = pd.Index(complete.get_level_values("subject_id").unique())
    strict_scaffold = cohort.loc[cohort["subject_id"].isin(complete_ids)].copy()
    shared_scaffold = cohort.loc[cohort["eligible"]].copy()
    if shared_scaffold.empty:
        raise ValueError("No patients pass the shared lab-coverage gate")
    # The stricter kidney sensitivity also requires measured creatinine.
    renal_coverage = by_day["analyte"].agg(lambda values: "creatinine" in set(values))
    renal_complete = coverage & renal_coverage
    renal_ids = pd.Index(renal_complete[renal_complete].index.get_level_values("subject_id").unique())
    strict_kidney_scaffold = cohort.loc[cohort["subject_id"].isin(renal_ids)].copy()
    kidney_days = signal_days.intersection(renal_days)
    strict_kidney_days = strict_days.intersection(renal_days).intersection(renal_complete[renal_complete].index)
    dates_by_name = {
        NAMES[0]: _first_dates(signal_days),
        NAMES[1]: _first_dates(kidney_days),
        STRICT_NAMES[0]: _first_dates(strict_days),
        STRICT_NAMES[1]: _first_dates(strict_kidney_days),
    }
    results = {}
    for name, base in (
        (NAMES[0], shared_scaffold), (NAMES[1], shared_scaffold),
        (STRICT_NAMES[0], strict_scaffold), (STRICT_NAMES[1], strict_kidney_scaffold),
    ):
        result = base.copy()
        dates = dates_by_name[name]
        result["outcome_date"] = (
            pd.to_datetime(result["subject_id"].map(dates)) if len(dates) else pd.NaT
        )
        horizon_end = result["index_date"] + pd.Timedelta(days=30)
        result["censor_date"] = pd.concat([result["censor_date"], horizon_end], axis=1).min(axis=1)
        if name in NAMES:
            ascertain_end = result["last_measurement_date"] + pd.Timedelta(days=MAX_LAB_GAP_DAYS)
            no_event = result["outcome_date"].isna()
            result.loc[no_event, "censor_date"] = pd.concat(
                [result.loc[no_event, "censor_date"], ascertain_end.loc[no_event]], axis=1
            ).min(axis=1)
        results[name] = result[["subject_id", "split", "outcome_date", "index_date", "censor_date"]]
        support_for_outcome(results[name], name, [30])
    audit = pd.DataFrame([{
        "split": split,
        "n_survival": int(cohort["split"].eq(split).sum()),
        "n_any_tls_axis_day1_7": int(cohort.loc[cohort["subject_id"].isin(
            labs.loc[labs["analyte"].isin(AXES), "subject_id"]), "split"].eq(split).sum()),
        "n_shared_lab_eligible": int(shared_scaffold["split"].eq(split).sum()),
        "n_complete_four_axis_same_day": int(strict_scaffold["split"].eq(split).sum()),
        "n_complete_four_axes_plus_creatinine_same_day": int(strict_kidney_scaffold["split"].eq(split).sum()),
        "n_tls_signal_before_coverage_filter": int(cohort.loc[cohort["subject_id"].isin(
            dates_by_name[NAMES[0]].index), "split"].eq(split).sum()),
        "n_kidney_tls_signal_before_coverage_filter": int(cohort.loc[cohort["subject_id"].isin(
            dates_by_name[NAMES[1]].index), "split"].eq(split).sum()),
        "n_tls_events": int(results[NAMES[0]].loc[results[NAMES[0]]["split"].eq(split), "outcome_date"].notna().sum()),
        "n_tls_with_kidney_events": int(results[NAMES[1]].loc[results[NAMES[1]]["split"].eq(split), "outcome_date"].notna().sum()),
        "n_tls_complete_panel_events": int(results[STRICT_NAMES[0]].loc[results[STRICT_NAMES[0]]["split"].eq(split), "outcome_date"].notna().sum()),
        "n_tls_with_kidney_complete_panel_events": int(results[STRICT_NAMES[1]].loc[results[STRICT_NAMES[1]]["split"].eq(split), "outcome_date"].notna().sum()),
    } for split in ("train", "tuning", "held_out")])
    return results, audit


def run(labs_path: Path, survival_path: Path, lab_audit_path: Path, output_dir: Path) -> dict:
    if output_dir.exists():
        raise ValueError(f"Refusing to overwrite {output_dir}")
    labs = _validate_labs(pd.read_parquet(labs_path))
    survival = _read_outcome(survival_path)
    results, audit = build_tls(labs, survival, pd.read_parquet(lab_audit_path))
    output_dir.mkdir(parents=True, exist_ok=False)
    outcomes_dir = output_dir / "outcomes"
    outcomes_dir.mkdir()
    for name, frame in results.items():
        frame.to_parquet(outcomes_dir / f"{name}.parquet", index=False)
    audit.to_csv(output_dir / "tls_monitoring_and_events_INTERNAL.csv", index=False)
    pd.DataFrame([row for name, frame in results.items()
                  for row in support_for_outcome(frame, name, [30])]).to_csv(
        output_dir / "horizon_support_INTERNAL.csv", index=False)
    manifest = {
        "version": 2, "status": "exploratory_shared_lab_gate_tls_not_adjudicated",
        "patient_level_internal_only": True, "prediction_horizon_days": 30,
        "diagnostic_event_days_after_index": [1, 7],
        "lab_axes": list(AXES), "abnormal_limits": LIMITS,
        "laboratory_tls": "at_least_two_distinct_abnormal_axes_same_calendar_day",
        "kidney_involvement": "laboratory_tls_and_creatinine_above_reported_uln_same_day",
        "negative_eligibility": "shared_R_lab_coverage_gate_from_audit_parquet",
        "sensitivity_eligibility": "same_day_complete_four_axis_panel; kidney_label_additionally_requires_creatinine",
        "shared_lab_max_gap_days": MAX_LAB_GAP_DAYS,
        "limitations": ["No 25-percent-baseline-change alternative", "No pre-index spontaneous TLS",
                        "Date-only same-day proxy for 24-hour co-occurrence",
                        "No arrhythmia or seizure routes to clinical TLS",
                        "No adjudication of pre-existing creatinine elevation"],
        "input_sha256": {str(path.resolve()): _sha256(path) for path in (labs_path, survival_path, lab_audit_path)},
        "output_sha256": {name: _sha256(outcomes_dir / f"{name}.parquet") for name in results},
    }
    (output_dir / "manifest_INTERNAL.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labs", type=Path, required=True)
    parser.add_argument("--survival", type=Path, required=True)
    parser.add_argument("--lab-audit", type=Path, required=True,
                        help="A __audit.parquet from the same outcome run's shared R lab gate")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    manifest = run(args.labs, args.survival, args.lab_audit, args.output_dir)
    print(json.dumps({"output_dir": str(args.output_dir), "status": manifest["status"]}))


if __name__ == "__main__":
    main()
