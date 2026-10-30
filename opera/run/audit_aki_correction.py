"""Reconstruct and compare AKI exports without fitting models (secure server only).

The optional episode-table route first reproduces the *old* released endpoint
using its frozen coverage audit and survival follow-up. It refuses to export a
replacement if that reproduction fails. Only development differences are reported.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from opera.run.decide_outcome_panel import REQUIRED, sha256_file

AKI = "aki_first_documented"
HORIZONS = (30, 60, 90, 180, 365, 730)
DATES = ("index_date", "outcome_date", "censor_date")


def read_endpoint(path: Path) -> pd.DataFrame:
    frame = pd.read_parquet(path)
    if set(frame) != set(REQUIRED):
        raise ValueError(f"Require the canonical five-column endpoint: {path}")
    frame = frame.copy()
    frame["subject_id"] = frame["subject_id"].astype("string")
    frame["split"] = frame["split"].astype("string")
    if frame.subject_id.isna().any() or frame.subject_id.duplicated().any():
        raise ValueError("Missing/duplicate endpoint IDs")
    if not frame.split.isin(["train", "tuning", "held_out"]).all():
        raise ValueError("Invalid split")
    for column in DATES:
        original = frame[column]
        frame[column] = pd.to_datetime(original, errors="raise")
        if getattr(frame[column].dt, "tz", None) is not None:
            raise ValueError("Use the same timezone-free calendar dates as the R export")
        if frame[column].dropna().ne(frame[column].dropna().dt.normalize()).any():
            raise ValueError("Endpoint dates must be calendar dates")
    event = frame.outcome_date.notna()
    if (frame.index_date.isna().any() or frame.censor_date.isna().any()
            or frame.censor_date.lt(frame.index_date).any()
            or (event & (frame.outcome_date.le(frame.index_date)
                         | frame.outcome_date.gt(frame.censor_date))).any()):
        raise ValueError("Invalid endpoint chronology")
    return frame.set_index("subject_id").sort_index()


def equal_dates(left: pd.Series, right: pd.Series) -> pd.Series:
    return left.eq(right) | (left.isna() & right.isna())


def require_fixed_population(old: pd.DataFrame, new: pd.DataFrame) -> None:
    if not old.index.equals(new.index):
        raise ValueError("AKI correction changed eligibility/IDs; investigate before release")
    if not old.split.equals(new.split) or not equal_dates(old.index_date, new.index_date).all():
        raise ValueError("AKI correction changed index dates/splits")


def reconstruct(episodes: Path, reference: pd.DataFrame, audit_path: Path,
                survival: pd.DataFrame, max_gap_days: int) -> pd.DataFrame:
    """Mirror events_from_table + .lab_outcome_core for already eligible IDs."""
    if max_gap_days < 0:
        raise ValueError("max_gap_days must be nonnegative")
    audit = pd.read_parquet(audit_path)
    needed = {"subject_id", "eligible", "index_date", "split", "last_measurement_date"}
    if not needed.issubset(audit) or audit.subject_id.isna().any():
        raise ValueError("Missing frozen laboratory coverage audit fields")
    audit["subject_id"] = audit.subject_id.astype("string")
    valid_logical = audit.eligible.dropna().map(lambda value: isinstance(value, (bool, np.bool_)))
    if audit.subject_id.duplicated().any() or not valid_logical.all():
        raise ValueError("Coverage audit requires unique IDs and boolean eligibility")
    eligible = audit.loc[audit.eligible.astype("boolean").fillna(False)].set_index("subject_id").sort_index()
    eligible["split"] = eligible["split"].astype("string")
    if not eligible.index.equals(reference.index):
        raise ValueError("Coverage audit eligibility differs from the released AKI endpoint")
    if (not eligible.split.equals(reference.split)
            or not equal_dates(pd.to_datetime(eligible.index_date), reference.index_date).all()):
        raise ValueError("Coverage audit index/split differs from the released AKI endpoint")
    if not reference.index.isin(survival.index).all():
        raise ValueError("Survival follow-up missing for eligible AKI patients")
    followup = survival.loc[reference.index]
    if (not followup.split.equals(reference.split)
            or not equal_dates(followup.index_date, reference.index_date).all()):
        raise ValueError("Survival index/split differs from AKI")
    events = pd.read_csv(episodes, sep="\t", dtype={"patientid": "string", "aki_start_time": "string"})
    if not {"patientid", "aki_start_time", "aki_episode"}.issubset(events):
        raise ValueError("Expected tab-delimited simplified AKI episode table")
    # R uses onset dates and does not filter on the episode number/stage.
    onset = pd.to_datetime(events.aki_start_time, errors="raise", format="mixed")
    if getattr(onset.dt, "tz", None) is not None:
        raise ValueError("Export onset calendar dates in the original R timezone first")
    events = events.assign(outcome_date=onset.dt.normalize()).dropna(subset=["aki_start_time"])
    events = events.merge(reference[["index_date"]], left_on="patientid", right_index=True)
    events = events.merge(followup[["censor_date"]], left_on="patientid", right_index=True)
    events = events.loc[events.outcome_date.gt(events.index_date)
                        & events.outcome_date.le(events.censor_date)]
    first = events.groupby("patientid").outcome_date.min()
    result = reference.copy()
    result["outcome_date"] = first.reindex(result.index)
    last = pd.to_datetime(eligible.last_measurement_date, errors="raise")
    observation_end = pd.concat([last + pd.Timedelta(days=max_gap_days),
                                 followup.censor_date], axis=1).min(axis=1)
    result["censor_date"] = observation_end.where(result.outcome_date.isna(), followup.censor_date)
    return result


def horizon_state(frame: pd.DataFrame, death: pd.Series, horizon: int) -> pd.Series:
    """Target event, competing death, event-free observed, or censored by h.

    Diagnostic calendar-day states. Actual training/evaluation still uses the
    existing datamodule and competing-risk contract, including eligibility gates.
    """
    end = frame.index_date + pd.Timedelta(days=horizon)
    event = frame.outcome_date.notna() & frame.outcome_date.le(end)
    competing = (death.notna() & death.le(end) & death.le(frame.censor_date)
                 & (frame.outcome_date.isna() | death.lt(frame.outcome_date)))
    state = pd.Series("censored", index=frame.index)
    state.loc[frame.censor_date.ge(end)] = "event_free_observed"
    state.loc[competing] = "competing_death"
    state.loc[event & ~competing] = "target_event"
    return state


def compare(old: pd.DataFrame, new: pd.DataFrame, survival: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    require_fixed_population(old, new)
    development = old.split.isin(["train", "tuning"])
    left, right = old.loc[development], new.loc[development]
    if not left.index.isin(survival.index).all():
        raise ValueError("Missing development survival records")
    mortality = survival.loc[left.index]
    if (not mortality.split.equals(left.split)
            or not equal_dates(mortality.index_date, left.index_date).all()):
        raise ValueError("Survival index/split mismatch")
    changed = pd.DataFrame({
        "split": left.split,
        "old_outcome_date": left.outcome_date, "new_outcome_date": right.outcome_date,
        "old_censor_date": left.censor_date, "new_censor_date": right.censor_date,
        "event_date_changed": ~equal_dates(left.outcome_date, right.outcome_date),
        "censor_date_changed": ~equal_dates(left.censor_date, right.censor_date),
    })
    rows = []
    for split in ("train", "tuning"):
        ids = left.index[left.split.eq(split)]
        for horizon in HORIZONS:
            a = horizon_state(left.loc[ids], mortality.loc[ids].outcome_date, horizon)
            b = horizon_state(right.loc[ids], mortality.loc[ids].outcome_date, horizon)
            old_event, new_event = a.eq("target_event"), b.eq("target_event")
            disagreement = int(a.ne(b).sum())
            added, removed = int((~old_event & new_event).sum()), int((old_event & ~new_event).sum())
            old_n, new_n = int(old_event.sum()), int(new_event.sum())
            rows.append({"split": split, "horizon_days": horizon, "n_eligible": len(ids),
                         "old_events": old_n, "new_events": new_n,
                         "events_added": added, "events_removed": removed,
                         "removed_fraction_of_old_events": removed / old_n if old_n else None,
                         "added_fraction_of_new_events": added / new_n if new_n else None,
                         "status_changed": disagreement,
                         "status_changed_fraction": disagreement / len(ids) if len(ids) else None,
                         "old_known": int(a.ne("censored").sum()),
                         "new_known": int(b.ne("censored").sum()),
                         "observation_status_changed": int(a.eq("censored").ne(b.eq("censored")).sum())})
    report = {"outcome": AKI, "scope": "train_and_tuning_only",
              "held_out_performance_evaluated": False, "fixed_population_and_splits": True,
              "development_patients": len(left),
              "event_dates_changed": int(changed.event_date_changed.sum()),
              "censor_dates_changed": int(changed.censor_date_changed.sum()),
              "development_endpoint_identical": not bool(changed.iloc[:, -2:].any().any()),
              "horizon_comparison": rows}
    return report, changed.loc[changed.event_date_changed | changed.censor_date_changed].reset_index()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old-outcome", type=Path, required=True)
    parser.add_argument("--overall-survival", type=Path, required=True)
    parser.add_argument("--new-outcome", type=Path)
    parser.add_argument("--old-episodes", type=Path)
    parser.add_argument("--corrected-episodes", type=Path)
    parser.add_argument("--coverage-audit", type=Path)
    parser.add_argument("--max-gap-days", type=int, default=180)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--write-patient-differences", action="store_true")
    args = parser.parse_args()
    episode_args = (args.old_episodes, args.corrected_episodes, args.coverage_audit)
    if (args.new_outcome is None and not all(episode_args)) or (args.new_outcome and any(episode_args)):
        parser.error("Supply --new-outcome OR all three episode/coverage arguments")
    if args.output_dir.exists():
        parser.error("Use a fresh audit directory")
    old, survival = read_endpoint(args.old_outcome), read_endpoint(args.overall_survival)
    inputs = {"old_outcome": args.old_outcome, "overall_survival": args.overall_survival}
    if args.new_outcome:
        candidate = args.new_outcome
        new = read_endpoint(candidate)
        inputs["new_outcome"] = candidate
    else:
        reproduced = reconstruct(args.old_episodes, old, args.coverage_audit, survival, args.max_gap_days)
        if not all(equal_dates(old[column], reproduced[column]).all() for column in DATES):
            raise ValueError("Old AKI export not reproduced: check coverage, admin follow-up and lab policy")
        new = reconstruct(args.corrected_episodes, old, args.coverage_audit, survival, args.max_gap_days)
        candidate = args.output_dir / f"{AKI}.parquet"
        inputs.update(old_episodes=args.old_episodes, corrected_episodes=args.corrected_episodes,
                      coverage_audit=args.coverage_audit)
    report, differences = compare(old, new, survival)
    report["inputs"] = {key: {"path": str(path.resolve()), "sha256": sha256_file(path)}
                        for key, path in inputs.items()}
    args.output_dir.mkdir(parents=True, mode=0o700)
    if args.new_outcome is None:
        # Keep the original identifier type, dates and column order in the release schema.
        original = pd.read_parquet(args.old_outcome)
        exported = new.reindex(original.subject_id.astype("string")).reset_index(drop=True)
        exported["subject_id"] = original.subject_id.to_numpy()
        exported[list(original)].to_parquet(candidate, index=False)
        read_endpoint(candidate)  # validate the actual exported bytes
    report["corrected_outcome"] = {"path": str(candidate.resolve()), "sha256": sha256_file(candidate)}
    report["max_gap_days"] = args.max_gap_days if args.new_outcome is None else None
    report["old_endpoint_reproduced"] = args.new_outcome is None
    pd.DataFrame(report["horizon_comparison"]).to_csv(args.output_dir / "horizon_changes_INTERNAL.csv", index=False)
    if args.write_patient_differences:
        differences.to_parquet(args.output_dir / "patient_differences_INTERNAL.parquet", index=False)
    (args.output_dir / "comparison_INTERNAL.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: report[key] for key in ("event_dates_changed", "censor_dates_changed",
                                                "development_endpoint_identical", "corrected_outcome")}, indent=2))


if __name__ == "__main__":
    main()
