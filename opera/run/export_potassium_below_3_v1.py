"""Versioned lab-only potassium <3.0 mmol/L candidate; no symptom inference.

Reads the already-normalized TLS laboratory snapshot and same-release OPERA
outcomes. Never edits the sealed export. This is the numerical CTCAE-v6 G3+
threshold, not a fully adjudicated clinical grade or a replacement for G2.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from opera.run.audit_outcome_exports import audit_outcome
from opera.run.decide_outcome_panel import sha256_file, support_for_outcome
from opera.run.export_candidate_outcomes_v1 import SCHEMA, _event_overlap, _read_outcome


NAME = "potassium_below_3p0_mmol_l"
MAX_GAP_DAYS = 180


def build(labs: pd.DataFrame, old: pd.DataFrame, survival: pd.DataFrame,
          audit: pd.DataFrame) -> pd.DataFrame:
    required_labs = {"subject_id", "analyte", "sample_date", "value", "unit"}
    required_audit = {"subject_id", "eligible", "last_measurement_date"}
    if not required_labs.issubset(labs) or not required_audit.issubset(audit):
        raise ValueError("Potassium snapshot or outcome audit lacks required columns")
    if audit["subject_id"].duplicated().any() or audit["subject_id"].isna().any():
        raise ValueError("Invalid lab audit IDs")
    potassium = labs.loc[labs["analyte"].eq("potassium")].copy()
    if potassium.empty or potassium["subject_id"].isna().any():
        raise ValueError("Potassium snapshot empty or has missing IDs")
    potassium["sample_date"] = pd.to_datetime(potassium["sample_date"], errors="coerce").dt.normalize()
    potassium["value"] = pd.to_numeric(potassium["value"], errors="coerce")
    # The R snapshot deliberately retains cleaned but nonnumeric source rows.
    # They cannot trigger a numerical event; exclude them, as the TLS exporter
    # does, without silently accepting an unknown unit on a usable result.
    potassium = potassium.loc[potassium["sample_date"].notna()
                              & potassium["value"].notna()].copy()
    if potassium.empty or not np.isfinite(potassium["value"]).all() or (potassium["value"] < 0).any():
        raise ValueError("No valid potassium results or invalid numeric value")
    potassium["unit"] = potassium["unit"].astype("string").str.strip().str.casefold()
    if potassium["unit"].isna().any() or set(potassium["unit"]) != {"mmol/l"}:
        raise ValueError("Potassium results must be normalized to mmol/L")
    scaffold = old.merge(audit[list(required_audit)], on="subject_id", validate="one_to_one")
    if len(scaffold) != len(old) or not scaffold["eligible"].fillna(False).all():
        raise ValueError("Old potassium outcome disagrees with its coverage audit")
    admin = survival[["subject_id", "index_date", "censor_date"]].rename(
        columns={"index_date": "survival_index", "censor_date": "admin_end"})
    scaffold = scaffold.merge(admin, on="subject_id", validate="one_to_one")
    if len(scaffold) != len(old) or not scaffold["index_date"].eq(scaffold["survival_index"]).all():
        raise ValueError("Potassium outcome and survival have different cohorts/index dates")
    scaffold["last_measurement_date"] = pd.to_datetime(
        scaffold["last_measurement_date"], errors="coerce").dt.normalize()
    measurements = potassium.merge(
        scaffold[["subject_id", "index_date", "admin_end"]],
        on="subject_id", validate="many_to_one")
    valid = (measurements["sample_date"].gt(measurements["index_date"])
             & measurements["sample_date"].le(measurements["admin_end"])
             & measurements["value"].lt(3.0))
    first_event = measurements.loc[valid].groupby("subject_id")["sample_date"].min()
    result = scaffold[["subject_id", "split", "index_date", "admin_end", "last_measurement_date"]].copy()
    result["outcome_date"] = result["subject_id"].map(first_event)
    observation_end = (result["last_measurement_date"] + pd.Timedelta(days=MAX_GAP_DAYS)).fillna(
        result["admin_end"])
    result["censor_date"] = pd.concat([observation_end, result["admin_end"]], axis=1).min(axis=1)
    result.loc[result["outcome_date"].notna(), "censor_date"] = result.loc[
        result["outcome_date"].notna(), "admin_end"]
    result = result[SCHEMA]
    if (result["censor_date"].isna().any() or
            (result["censor_date"] < result["index_date"]).any() or
            (result["outcome_date"].notna() & (
                (result["outcome_date"] <= result["index_date"])
                | (result["outcome_date"] > result["censor_date"]))).any()):
        raise ValueError("Invalid potassium outcome chronology")
    return result


def run(labs_path: Path, base_dir: Path, output_dir: Path) -> dict:
    if output_dir.exists():
        raise ValueError(f"Output directory already exists: {output_dir}")
    old_path = base_dir / "hypokalemia_g3plus.parquet"
    g2_path = base_dir / "hypokalemia_g2plus.parquet"
    survival_path = base_dir / "overall_survival.parquet"
    audit_path = base_dir / "hypokalemia_g3plus__audit.parquet"
    old, g2, survival = (_read_outcome(path) for path in (old_path, g2_path, survival_path))
    if (set(old["subject_id"]) != set(g2["subject_id"])
            or not old.set_index("subject_id")[["split", "index_date"]].sort_index().equals(
                g2.set_index("subject_id")[["split", "index_date"]].sort_index())):
        raise ValueError("Historical potassium G2/G3 cohorts or indices differ")
    result = build(pd.read_parquet(labs_path), old, survival, pd.read_parquet(audit_path))
    output_dir.mkdir(parents=True, exist_ok=False)
    outcomes = output_dir / "outcomes"
    outcomes.mkdir()
    target = outcomes / f"{NAME}.parquet"
    result.to_parquet(target, index=False)
    structure = audit_outcome(target)
    if "error" in structure or any(structure.get(key, 0) for key in (
            "duplicate_subjects", "missing_subjects", "missing_index", "missing_censor",
            "censor_before_index", "event_on_or_before_index", "event_after_censor",
            "invalid_split", "split_index_mismatch")):
        raise ValueError(f"New potassium structure failed: {structure}")
    support = pd.DataFrame(support_for_outcome(result, NAME, [30, 60, 90, 180, 365, 730]))
    support.to_csv(output_dir / "horizon_support_INTERNAL.csv", index=False)
    pd.DataFrame(_event_overlap(old, result, NAME)).to_csv(
        output_dir / "overlap_vs_old_g3_INTERNAL.csv", index=False)
    pd.DataFrame(_event_overlap(g2, result, NAME)).to_csv(
        output_dir / "overlap_vs_old_g2_INTERNAL.csv", index=False)
    manifest = {
        "version": 1, "status": "candidate_lab_numeric_not_full_ctcae_grade",
        "outcome": NAME, "threshold": "potassium <3.0 mmol/L on a post-index day",
        "eligibility": "same old R all-lab coverage; no analyte-specific monitoring gate",
        "symptoms_and_intervention": "unobserved; cannot identify full CTCAE G2 or G3",
        "input_sha256": {str(path.resolve()): sha256_file(path)
                         for path in (labs_path, old_path, g2_path, survival_path, audit_path)},
        "output_sha256": sha256_file(target),
    }
    (output_dir / "manifest_INTERNAL.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return {"output_dir": str(output_dir.resolve()), "outcome": NAME}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tls-labs", type=Path, required=True)
    parser.add_argument("--base-outcomes-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.tls_labs, args.base_outcomes_dir, args.output_dir), indent=2))


if __name__ == "__main__":
    main()
