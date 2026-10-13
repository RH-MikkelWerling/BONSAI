"""Generate isolated INTERNAL candidate lab outcomes; never overwrite v4 labels.

Inputs are an explicitly normalized DALY-CARE lab snapshot and existing outcome
parquets. These numeric what-if definitions require clinical/source sign-off;
they are not a final CTCAE-v6 release. All patient-level files stay on server.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from opera.run.decide_outcome_panel import support_for_outcome


VERSION = 1
CREATININE_RELATIVE_OUTCOME = "creatinine_rise_gt1p5x_preindex_baseline"
ANALYTES = {
    "neutrophils": ("neutropenia_g2plus", "neutropenia_g3plus"),
    "ALT": ("ALT_increased_g2plus", "ALT_increased_g3plus"),
    "AST": ("AST_increased_g2plus", "AST_increased_g3plus"),
    "bilirubin": ("bilirubin_increased_g2plus", "bilirubin_increased_g3plus"),
    "creatinine": ("creatinine_increased_g2plus", "creatinine_increased_g3plus"),
    "eosinophils": ("eosinophilia",),
}
BASELINE_ANALYTES = frozenset(ANALYTES) - {"neutrophils"}
SCHEMA = ["subject_id", "split", "outcome_date", "index_date", "censor_date"]
LAB_COLUMNS = {"subject_id", "analyte", "sample_date", "value", "unit", "ref_lower", "ref_upper"}
AUDIT_COLUMNS = {"subject_id", "eligible", "last_measurement_date"}
BAD_SPLITS = {"train", "tuning", "held_out"}


def _hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _dates(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    result = frame.copy()
    for column in columns:
        result[column] = pd.to_datetime(result[column], errors="coerce").dt.normalize()
    return result


def _read_outcome(path: Path) -> pd.DataFrame:
    frame = pd.read_parquet(path)
    if set(frame) != set(SCHEMA) or frame["subject_id"].isna().any() or frame["subject_id"].duplicated().any():
        raise ValueError(f"Invalid outcome schema or IDs: {path}")
    frame = _dates(frame[SCHEMA], ["index_date", "outcome_date", "censor_date"])
    if not frame["split"].isin(BAD_SPLITS).all():
        raise ValueError(f"Invalid split in {path}")
    if (frame["index_date"].isna().any() or frame["censor_date"].isna().any()
            or (frame["censor_date"] < frame["index_date"]).any()
            or ((frame["outcome_date"].notna()) & (
                (frame["outcome_date"] <= frame["index_date"])
                | (frame["outcome_date"] > frame["censor_date"]))).any()):
        raise ValueError(f"Invalid dates in {path}")
    return frame


def _load_labs(path: Path, units_path: Path) -> pd.DataFrame:
    contract = yaml.safe_load(units_path.read_text(encoding="utf-8"))
    if contract.get("version") != VERSION or set(contract.get("units", {})) != set(ANALYTES):
        raise ValueError("Unit contract must list exactly the six v1 analytes")
    labs = pd.read_parquet(path)
    if not LAB_COLUMNS.issubset(labs.columns):
        raise ValueError(f"Lab snapshot lacks {sorted(LAB_COLUMNS - set(labs))}")
    labs = labs.loc[labs["analyte"].isin(ANALYTES)].copy()
    if set(labs["analyte"]) != set(ANALYTES):
        raise ValueError("Lab snapshot must contain all six candidate analytes")
    if labs["subject_id"].isna().any() or labs["analyte"].isna().any():
        raise ValueError("Lab snapshot has missing IDs or analytes")
    labs["sample_date"] = pd.to_datetime(labs["sample_date"], errors="coerce").dt.normalize()
    if labs["sample_date"].isna().any():
        raise ValueError("Lab snapshot has invalid sample dates")
    for column in ("value", "ref_lower", "ref_upper"):
        labs[column] = pd.to_numeric(labs[column], errors="coerce")
    if labs["value"].notna().any() and ((labs["value"].dropna() < 0)
                                            | ~np.isfinite(labs["value"].dropna())).any():
        raise ValueError("Lab snapshot has negative or nonfinite cleaned values")
    labs["unit"] = labs["unit"].astype("string").str.strip().str.casefold()
    for analyte, expected in contract["units"].items():
        observed = set(labs.loc[labs["analyte"].eq(analyte), "unit"].dropna())
        if observed != {str(expected).casefold()}:
            raise ValueError(f"{analyte}: cleaned units {sorted(observed)} differ from {expected!r}")
    return labs.loc[labs["value"].notna()]


def _baseline(labs: pd.DataFrame, scaffold: pd.DataFrame, window_days: int) -> pd.DataFrame:
    joined = labs.merge(scaffold[["subject_id", "index_date"]], on="subject_id", validate="many_to_one")
    pre = joined.loc[
        joined["sample_date"].lt(joined["index_date"])
        & joined["sample_date"].ge(joined["index_date"] - pd.Timedelta(days=window_days))
    ]
    # Median of same-day draws removes input-row ordering from baseline choice.
    daily = pre.groupby(["subject_id", "sample_date"], as_index=False)[
        ["value", "ref_lower", "ref_upper"]
    ].median()
    last = daily.sort_values(["subject_id", "sample_date"]).drop_duplicates("subject_id", keep="last")
    return last.rename(columns={
        "sample_date": "baseline_date", "value": "baseline_value",
        "ref_lower": "baseline_ref_lower", "ref_upper": "baseline_ref_upper",
    })


def _gradable_baseline(frame: pd.DataFrame, analyte: str) -> pd.Series:
    valid = frame["baseline_value"].notna() & frame["baseline_value"].ge(0)
    if analyte != "eosinophils":
        valid &= frame["baseline_value"].gt(0)
    if analyte in {"ALT", "AST", "bilirubin"}:
        valid &= frame["baseline_ref_upper"].gt(0)
    elif analyte == "creatinine":
        valid &= frame["baseline_ref_lower"].gt(0)
    return valid


def _candidate_flags(labs: pd.DataFrame, analyte: str) -> tuple[pd.Series, ...]:
    value = labs["value"]
    if analyte == "neutrophils":
        return value.lt(1.0), value.lt(0.5)
    baseline = labs["baseline_value"]
    if analyte == "creatinine_relative":
        # A separate clinical change-from-baseline target, not a CTCAE grade.
        # In particular, this applies even when baseline creatinine is normal
        # or above ULN; CTCAE v6 uses a baseline ratio only below LLN.
        return (value.gt(1.5 * baseline),)
    if analyte in {"ALT", "AST"}:
        abnormal = baseline.gt(labs["baseline_ref_upper"])
        ratio = value / baseline.where(baseline.gt(0))
        normal_ratio = value / labs["ref_upper"].where(labs["ref_upper"].gt(0))
        return (abnormal & ratio.gt(1.5)) | (~abnormal & normal_ratio.gt(3.0)), (
            (abnormal & ratio.gt(2.0)) | (~abnormal & normal_ratio.gt(5.0)))
    if analyte == "bilirubin":
        abnormal = baseline.gt(labs["baseline_ref_upper"])
        ratio = value / baseline.where(baseline.gt(0))
        normal_ratio = value / labs["ref_upper"].where(labs["ref_upper"].gt(0))
        return (abnormal & ratio.gt(1.5)) | (~abnormal & normal_ratio.gt(1.5)), (
            (abnormal & ratio.gt(2.5)) | (~abnormal & normal_ratio.gt(3.0)))
    if analyte == "creatinine":
        below_lln = baseline.lt(labs["baseline_ref_lower"])
        ratio = value / baseline.where(baseline.gt(0))
        normal_ratio = value / labs["ref_upper"].where(labs["ref_upper"].gt(0))
        return (below_lln & ratio.gt(1.5)) | (~below_lln & normal_ratio.gt(1.5)), (
            (below_lln & ratio.gt(3.0)) | (~below_lln & normal_ratio.gt(3.0)))
    if analyte == "eosinophils":
        positive = value.gt(labs["ref_upper"]) & value.gt(baseline)
        return positive, positive
    raise ValueError(f"Unknown analyte: {analyte}")


def _compare(old: pd.DataFrame, candidate: pd.DataFrame, outcome: str) -> list[dict]:
    old_dates = old.set_index("subject_id")["outcome_date"]
    new_dates = candidate.set_index("subject_id")["outcome_date"]
    rows = []
    for split in ("train", "tuning", "held_out"):
        old_split = old.loc[old["split"].eq(split), "subject_id"]
        new_split = candidate.loc[candidate["split"].eq(split), "subject_id"]
        both = pd.Index(old_split).intersection(pd.Index(new_split))
        before, after = old_dates.reindex(both), new_dates.reindex(both)
        rows.append({
            "outcome": outcome, "split": split, "old_eligible": len(old_split),
            "candidate_eligible": len(new_split), "lost_eligibility": len(old_split) - len(both),
            "old_events_all": int(old.loc[old["split"].eq(split), "outcome_date"].notna().sum()),
            "candidate_events_all": int(candidate.loc[candidate["split"].eq(split), "outcome_date"].notna().sum()),
            "old_events_shared": int(before.notna().sum()),
            "candidate_events_shared": int(after.notna().sum()),
            "lost_events_shared": int((before.notna() & after.isna()).sum()),
            "gained_events_shared": int((before.isna() & after.notna()).sum()),
            "shifted_dates_shared": int((before.notna() & after.notna() & before.ne(after)).sum()),
        })
    return rows


def _event_overlap(old: pd.DataFrame, candidate: pd.DataFrame, outcome: str) -> list[dict]:
    """Aggregate patient/event overlap, with eligibility loss kept separate."""
    old_by_id = old.set_index("subject_id")
    new_by_id = candidate.set_index("subject_id")
    rows = []
    for split in ("train", "tuning", "held_out"):
        old_part = old_by_id.loc[old_by_id["split"].eq(split)]
        new_part = new_by_id.loc[new_by_id["split"].eq(split)]
        shared = old_part.index.intersection(new_part.index)
        if not old_part.loc[shared, "index_date"].equals(new_part.loc[shared, "index_date"]):
            raise ValueError(f"{outcome}: index dates differ on shared subjects")
        removed = old_part.index.difference(new_part.index)
        for horizon in (30, 365, None):
            old_days = (old_part["outcome_date"] - old_part["index_date"]).dt.days
            new_days = (new_part["outcome_date"] - new_part["index_date"]).dt.days
            old_event = old_days.notna() if horizon is None else old_days.notna() & old_days.le(horizon)
            new_event = new_days.notna() if horizon is None else new_days.notna() & new_days.le(horizon)
            before = old_event.reindex(shared)
            after = new_event.reindex(shared)
            both = int((before & after).sum())
            only_old = int((before & ~after).sum())
            only_new = int((~before & after).sum())
            union = both + only_old + only_new
            rows.append({
                "outcome": outcome, "split": split,
                "horizon_days": "all" if horizon is None else horizon,
                "old_eligible": len(old_part), "candidate_eligible": len(new_part),
                "shared_eligible": len(shared), "lost_eligibility": len(removed),
                "old_events_on_lost_eligibility": int(old_event.reindex(removed).sum()),
                "both_event_on_shared": both,
                "old_only_event_on_shared": only_old,
                "candidate_only_event_on_shared": only_new,
                "neither_event_on_shared": int((~before & ~after).sum()),
                "jaccard_event_overlap_on_shared": both / union if union else None,
                "candidate_capture_of_old_events_on_shared": (
                    both / (both + only_old) if both + only_old else None
                ),
            })
    return rows


def build_candidate(
    analyte: str, labs: pd.DataFrame, old: pd.DataFrame, audit: pd.DataFrame,
    survival: pd.DataFrame, *, baseline_window_days: int = 365, max_gap_days: int = 180,
) -> tuple[dict[str, pd.DataFrame], dict]:
    if not AUDIT_COLUMNS.issubset(audit.columns) or audit["subject_id"].duplicated().any():
        raise ValueError(f"{analyte}: coverage audit lacks required columns or has duplicate IDs")
    scaffold = old.merge(
        audit[["subject_id", "eligible", "last_measurement_date"]],
        on="subject_id", validate="one_to_one",
    )
    if len(scaffold) != len(old) or not scaffold["eligible"].fillna(False).all():
        raise ValueError(f"{analyte}: old outcome and coverage audit disagree")
    admin = survival[["subject_id", "index_date", "censor_date"]].rename(
        columns={"censor_date": "admin_end", "index_date": "survival_index"})
    scaffold = scaffold.merge(admin, on="subject_id", validate="one_to_one")
    if len(scaffold) != len(old) or not scaffold["index_date"].eq(scaffold["survival_index"]).all():
        raise ValueError(f"{analyte}: old outcome and survival scaffold disagree")
    scaffold["last_measurement_date"] = pd.to_datetime(
        scaffold["last_measurement_date"], errors="coerce").dt.normalize()
    source_analyte = "creatinine" if analyte == "creatinine_relative" else analyte
    raw = labs.loc[labs["analyte"].eq(source_analyte)].copy()
    base = _baseline(raw, scaffold, baseline_window_days) if analyte in BASELINE_ANALYTES or analyte == "creatinine_relative" else None
    if base is not None:
        scaffold = scaffold.merge(base, on="subject_id", how="left", validate="one_to_one")
        gradable = _gradable_baseline(scaffold, analyte)
        n_missing_baseline = int((~gradable).sum())
        scaffold = scaffold.loc[gradable].copy()
    else:
        n_missing_baseline = 0
    candidate_ids = set(scaffold["subject_id"])
    raw_in_followup = raw.merge(
        scaffold[["subject_id", "index_date", "admin_end"]],
        on="subject_id", validate="many_to_one",
    )
    raw_in_followup = raw_in_followup.loc[
        raw_in_followup["sample_date"].gt(raw_in_followup["index_date"])
        & raw_in_followup["sample_date"].le(raw_in_followup["admin_end"])
    ]
    post_ids = set(raw_in_followup["subject_id"])
    entry_ids = set(raw_in_followup.loc[
        raw_in_followup["sample_date"].le(
            raw_in_followup["index_date"] + pd.Timedelta(days=30)
        ), "subject_id"
    ])
    post = raw.merge(scaffold[["subject_id", "index_date", "admin_end"] + (
        ["baseline_value", "baseline_ref_lower", "baseline_ref_upper"] if base is not None else [])],
        on="subject_id", validate="many_to_one")
    post = post.loc[
        post["sample_date"].gt(post["index_date"])
        & post["sample_date"].le(post["admin_end"])
    ].copy()
    if analyte == "creatinine":
        # CTCAE's below-LLN branch only needs a patient baseline, whereas
        # its other branch needs the measurement's ULN.
        post = post.loc[
            post["baseline_value"].lt(post["baseline_ref_lower"])
            | post["ref_upper"].gt(0)
        ].copy()
    elif analyte not in {"neutrophils", "creatinine_relative"}:
        post = post.loc[post["ref_upper"].gt(0)].copy()
    flags = _candidate_flags(post, analyte)
    results = {}
    outcome_names = ((CREATININE_RELATIVE_OUTCOME,) if analyte == "creatinine_relative"
                     else ANALYTES[analyte])
    for i, outcome in enumerate(outcome_names):
        event_dates = post.loc[flags[i]].groupby("subject_id")["sample_date"].min()
        candidate = scaffold[["subject_id", "split", "index_date", "admin_end", "last_measurement_date"]].copy()
        candidate["outcome_date"] = candidate["subject_id"].map(event_dates)
        observed_end = candidate["last_measurement_date"] + pd.Timedelta(days=max_gap_days)
        observed_end = observed_end.fillna(candidate["admin_end"])
        candidate["censor_date"] = np.where(
            candidate["outcome_date"].notna(), candidate["admin_end"],
            pd.concat([observed_end, candidate["admin_end"]], axis=1).min(axis=1),
        )
        candidate = _dates(candidate[SCHEMA], ["index_date", "outcome_date", "censor_date"])
        if (candidate["censor_date"].isna().any()
                or (candidate["censor_date"] < candidate["index_date"]).any()
                or (candidate["outcome_date"].notna() & (
                    (candidate["outcome_date"] <= candidate["index_date"])
                    | (candidate["outcome_date"] > candidate["censor_date"]))).any()):
            raise ValueError(f"{outcome}: invalid candidate chronology")
        results[outcome] = candidate
    return results, {
        "analyte": analyte, "old_eligible": len(old),
        "candidate_eligible": len(scaffold), "excluded_missing_baseline_or_reference": n_missing_baseline,
        "candidate_without_analyte_post_index_draw": len(candidate_ids - post_ids),
        "candidate_without_analyte_entry_draw_30d": len(candidate_ids - entry_ids),
        "post_index_usable_draws": len(post),
    }


def run(
    labs_path: Path, old_dir: Path, output_dir: Path, units_path: Path,
    baseline_window_days: int = 365, max_gap_days: int = 180,
) -> dict:
    if output_dir.exists():
        raise ValueError(f"Refusing to overwrite existing candidate directory: {output_dir}")
    if output_dir.resolve().is_relative_to(old_dir.resolve()):
        raise ValueError("Candidate output directory must be outside old outcomes directory")
    if baseline_window_days <= 0 or max_gap_days <= 0:
        raise ValueError("Windows must be positive")
    labs = _load_labs(labs_path, units_path)
    survival_path = old_dir / "overall_survival.parquet"
    survival = _read_outcome(survival_path)
    output_frames, comparison, overlap, availability = {}, [], [], []
    used_inputs = [labs_path, units_path, survival_path]
    for analyte, outcomes in ANALYTES.items():
        source = old_dir / f"{outcomes[0]}.parquet"
        audit_path = old_dir / f"{outcomes[0]}__audit.parquet"
        old = _read_outcome(source)
        audit = pd.read_parquet(audit_path)
        used_inputs.extend([source, audit_path])
        frames, row = build_candidate(
            analyte, labs, old, audit, survival,
            baseline_window_days=baseline_window_days, max_gap_days=max_gap_days,
        )
        availability.append(row)
        for outcome in outcomes:
            old_path = old_dir / f"{outcome}.parquet"
            old_frame = _read_outcome(old_path)
            used_inputs.append(old_path)
            candidate = frames[outcome]
            if set(old_frame["subject_id"]) != set(old["subject_id"]):
                raise ValueError(f"{analyte}: grade outcomes differ in eligible patient IDs")
            comparison.extend(_compare(old_frame, candidate, outcome))
            overlap.extend(_event_overlap(old_frame, candidate, outcome))
            output_frames[outcome] = candidate
        if len(outcomes) == 2:
            left, right = (frames[name].set_index("subject_id")["outcome_date"] for name in outcomes)
            if not right.dropna().index.isin(left.dropna().index).all():
                raise ValueError(f"{analyte}: candidate G3+ is not nested in G2+")
    creatinine_old = _read_outcome(old_dir / "creatinine_increased_g2plus.parquet")
    creatinine_audit = pd.read_parquet(old_dir / "creatinine_increased_g2plus__audit.parquet")
    relative_frames, relative_availability = build_candidate(
        "creatinine_relative", labs, creatinine_old, creatinine_audit, survival,
        baseline_window_days=baseline_window_days, max_gap_days=max_gap_days,
    )
    relative = relative_frames[CREATININE_RELATIVE_OUTCOME]
    output_frames[CREATININE_RELATIVE_OUTCOME] = relative
    availability.append(relative_availability)
    comparison.extend(_compare(creatinine_old, relative, CREATININE_RELATIVE_OUTCOME))
    overlap.extend(_event_overlap(creatinine_old, relative, CREATININE_RELATIVE_OUTCOME))
    definition_overlap = _event_overlap(
        output_frames["creatinine_increased_g2plus"], relative,
        CREATININE_RELATIVE_OUTCOME,
    )
    output_dir.mkdir(parents=True, exist_ok=False)
    outcomes_dir = output_dir / "outcomes"
    outcomes_dir.mkdir()
    for outcome, frame in output_frames.items():
        frame.to_parquet(outcomes_dir / f"{outcome}.parquet", index=False)
    pd.DataFrame(comparison).to_csv(output_dir / "comparison_counts_INTERNAL.csv", index=False)
    pd.DataFrame(overlap).to_csv(output_dir / "event_overlap_INTERNAL.csv", index=False)
    pd.DataFrame(definition_overlap).to_csv(
        output_dir / "creatinine_definition_overlap_INTERNAL.csv", index=False)
    pd.DataFrame(availability).to_csv(output_dir / "baseline_availability_INTERNAL.csv", index=False)
    horizon_rows = []
    for outcome, frame in output_frames.items():
        horizon_rows.extend(support_for_outcome(frame, outcome, [30, 60, 90, 180, 365, 730]))
    pd.DataFrame(horizon_rows).to_csv(output_dir / "horizon_support_INTERNAL.csv", index=False)
    manifest = {
        "version": VERSION, "status": "candidate_numeric_what_if_not_final_ctcae_v6",
        "definition_revision": 2,
        "creatinine_relative_definition": ">1.5x median creatinine on latest pre-index day within baseline window; not CTCAE v6 or AKI",
        "creatinine_definition_overlap_columns": "old=CTCAE-v6-candidate G2+; candidate=relative-rise target",
        "complete_panel": False, "patient_level_internal_only": True,
        "old_outcomes_dir": str(old_dir.resolve()),
        "baseline_policy": f"median_on_last_pre_index_day_within_{baseline_window_days}d",
        "baseline_missing_policy": "exclude_from_candidate_not_a_negative",
        "monitoring_policy": "retain_old_all_lab_eligibility; analyte_specific_gaps_audited_not_applied",
        "max_gap_days": max_gap_days,
        "changed_outcomes": sorted(output_frames),
        "composites_not_rebuilt": ["hepatotoxicity"],
        "input_sha256": {str(path.resolve()): _hash(path) for path in set(used_inputs)},
        "output_sha256": {
            outcome: _hash(outcomes_dir / f"{outcome}.parquet") for outcome in sorted(output_frames)
        },
    }
    (output_dir / "manifest_INTERNAL.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labs", type=Path, required=True,
                        help="Restricted normalized snapshot from export_candidate_lab_snapshot_v1.R")
    parser.add_argument("--old-outcomes-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--unit-contract", type=Path,
                        default=Path("opera/configs/candidate_lab_units_v1.yaml"))
    parser.add_argument("--baseline-window-days", type=int, default=365)
    parser.add_argument("--max-gap-days", type=int, default=180)
    args = parser.parse_args()
    manifest = run(args.labs, args.old_outcomes_dir, args.output_dir,
                   args.unit_contract, args.baseline_window_days, args.max_gap_days)
    print(json.dumps({
        "output_dir": str(args.output_dir), "n_candidate_outcomes": len(manifest["changed_outcomes"]),
        "status": manifest["status"],
    }))


if __name__ == "__main__":
    main()
