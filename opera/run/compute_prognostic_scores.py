"""IPI and NCCN-IPI from quality-registry components recorded at diagnosis.

Writes one row per patient with ``ipi_computed_score`` / ``ipi_computed_group``
(0-5; low 0-1, low-intermediate 2, high-intermediate 3, high 4-5) and
``nccn_ipi_score`` / ``nccn_ipi_group`` (0-8; low 0-1, low-intermediate 2-3,
high-intermediate 4-5, high >= 6). A score is left missing when any component
it needs is missing; nothing is imputed.

The column map names the registry columns (JSON, canonical -> source):

    age                  age at diagnosis, years
    ldh                  LDH at diagnosis, U/L; the upper limit of normal is
                         205 U/L, or 255 U/L from age 70 (as in LYFO), unless
                         ldh_uln names a column with it
    ldh_ratio            LDH / upper limit of normal, instead of ldh
    stage                Ann Arbor stage (1-4 or I-IV, letters such as "IIIB" allowed)
    ps                   ECOG/WHO performance status (0-4)
    extranodal_sites     IPI: a column with the number of extranodal sites, or
                         a list of per-site Yes/No columns to count
    nccn_extranodal      NCCN-IPI: a list of per-site Yes/No columns (default
                         NCCN_EXTRANODAL_SITES: bone marrow, CNS, liver,
                         stomach, lung, small intestine, colon) or one flag
                         column. Any "Yes" scores 1; otherwise any "No" scores
                         0; all missing leaves the score missing.

    python -m opera.run.compute_prognostic_scores --registry <table> \
        --column-map <map.json> --output <scores.parquet>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROMAN = {"I": 1, "II": 2, "III": 3, "IV": 4}
# LYFO site fields that define NCCN-IPI extranodal disease (Zhou et al. 2014:
# bone marrow, CNS, liver/GI tract, lung).
NCCN_EXTRANODAL_SITES = ["bone_marrow_diagnosis", "CNS_diagnosis", "liver_diagnosis",
                         "ventricle_diagnosis", "lung_diagnosis",
                         "small_intestine_diagnosis", "colon_diagnosis"]


DEFAULT_COLUMNS = {"subject_id": "patientid", "age": "age_diagnosis",
                   "ldh": "LDH_diagnosis", "stage": "AA_stage_diagnosis",
                   "ps": "PS_diagnosis",
                   "nccn_extranodal": NCCN_EXTRANODAL_SITES,
                   # IPI counts all extranodal sites; add any further LYFO site
                   # columns here, otherwise these seven are counted.
                   "extranodal_sites": NCCN_EXTRANODAL_SITES}


def _stage(values):
    text = values.astype("string").str.strip().str.upper()
    digits = pd.to_numeric(text.str.extract(r"^([1-4])", expand=False), errors="coerce")
    roman = text.str.extract(r"^(IV|I{1,3})", expand=False).map(ROMAN)
    return digits.fillna(roman).astype(float)


def _flag(values):
    mapping = {"true": 1.0, "false": 0.0, "1": 1.0, "0": 0.0, "1.0": 1.0, "0.0": 0.0,
               "yes": 1.0, "no": 0.0, "ja": 1.0, "nej": 0.0}
    return values.astype("string").str.strip().str.lower().map(mapping).astype(float)


def _any_site(frame, columns):
    """1 if any site is involved, 0 if none is but some are recorded, else NA."""
    flags = pd.concat([_flag(frame[c]) for c in columns], axis=1)
    return pd.Series(np.where(flags.eq(1).any(axis=1), 1.0,
                              np.where(flags.notna().any(axis=1), 0.0, np.nan)),
                     index=frame.index)


def _groups(score, edges, labels):
    group = pd.cut(score, [-np.inf, *edges, np.inf], labels=labels, right=True)
    return group.astype("string").where(score.notna())


def compute_scores(registry, column_map):
    lists = {k: v for k, v in column_map.items() if isinstance(v, list)}
    frame = registry.rename(columns={v: k for k, v in column_map.items()
                                     if not isinstance(v, list)})
    age = pd.to_numeric(frame.get("age"), errors="coerce")
    if "ldh_ratio" in frame:
        ratio = pd.to_numeric(frame.ldh_ratio, errors="coerce")
    else:
        uln = (pd.to_numeric(frame.ldh_uln, errors="coerce") if "ldh_uln" in frame
               else pd.Series(np.where(age >= 70, 255.0, 205.0), index=frame.index))
        ratio = pd.to_numeric(frame.get("ldh"), errors="coerce") / uln
    stage = _stage(frame["stage"])
    ps = pd.to_numeric(frame.get("ps"), errors="coerce")
    out = pd.DataFrame({"subject_id": frame.subject_id})

    if "extranodal_sites" in lists:
        flags = pd.concat([_flag(frame[c]) for c in lists["extranodal_sites"]], axis=1)
        frame["extranodal_sites"] = flags.sum(axis=1).where(flags.notna().any(axis=1))
    if "extranodal_sites" in frame:
        sites = pd.to_numeric(frame.extranodal_sites, errors="coerce")
        parts = [age > 60, ratio > 1, ps >= 2, stage >= 3, sites > 1]
        known = pd.concat([age, ratio, ps, stage, sites], axis=1).notna().all(axis=1)
        score = sum(part.astype(float) for part in parts).where(known)
        out["ipi_computed_score"] = score
        out["ipi_computed_group"] = _groups(
            score, [1, 2, 3], ["low", "low-intermediate", "high-intermediate", "high"])

    if "nccn_extranodal" in lists or "nccn_extranodal" in frame:
        site = (_any_site(frame, lists["nccn_extranodal"]) if "nccn_extranodal" in lists
                else _flag(frame.nccn_extranodal))
        age_points = np.select([age > 75, age > 60, age > 40], [3, 2, 1], 0)
        ldh_points = np.select([ratio > 3, ratio > 1], [2, 1], 0)
        known = pd.concat([age, ratio, ps, stage, site], axis=1).notna().all(axis=1)
        score = (pd.Series(age_points + ldh_points, index=frame.index)
                 + (stage >= 3).astype(int) + site.fillna(0).astype(int)
                 + (ps >= 2).astype(int)).astype(float).where(known)
        out["nccn_ipi_score"] = score
        out["nccn_ipi_group"] = _groups(
            score, [1, 3, 5], ["low", "low-intermediate", "high-intermediate", "high"])
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--column-map", type=Path,
                        help="JSON overriding DEFAULT_COLUMNS (canonical -> registry name).")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    reader = pd.read_parquet if args.registry.suffix in {".parquet", ".pq"} else pd.read_csv
    registry = reader(args.registry)
    # Defaults are the LYFO-derived names used for the NCCN-IPI previously.
    column_map = {**DEFAULT_COLUMNS,
                  **(json.loads(args.column_map.read_text()) if args.column_map else {})}
    if registry[column_map.get("subject_id", "subject_id")].duplicated().any():
        raise ValueError("One registry row per patient is required")
    scores = compute_scores(registry, column_map)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    scores.to_parquet(args.output, index=False)
    summary = {column: {("missing" if pd.isna(k) else str(k)): int(v)
                        for k, v in scores[column].value_counts(dropna=False).items()}
               for column in scores if column.endswith("_group")}
    print(json.dumps({"output": str(args.output), "patients": len(scores),
                      "groups": summary}, indent=2, default=str))


if __name__ == "__main__":
    main()
