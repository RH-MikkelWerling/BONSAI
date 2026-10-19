"""One row per patient from the DALY-CARE IPI records (Table 1, clinical scores).

The sealed population metadata keeps every record of the server's IPI dataset
per patient (``ipi_records_json``; fields subject_id, sex, PS, IPI, Disease,
index), because a patient can have several. This writes one row per patient:

- ``index``, ``IPI``: the scoring system and its recorded risk category, as in
  the source (used by the clinical-score comparison);
- ``PS``: recorded performance status;
- ``risk_category``: Low / Intermediate / High, the recorded category of the
  disease's own index; for DLBCL the NCCN-IPI computed from registry
  components where it can be (its two intermediate groups merged), otherwise
  the recorded index; ``risk_source`` names which;
- ``performance_status``: 0–1 / 2–4;
- ``ipi_resolution``: how the row was obtained (none, single, agreeing,
  conflicting).

Rule for several records, fixed here and never by row order: a field is kept
when every record that has it agrees; if records disagree, it is left empty
and the patient is counted as conflicting. Empty and conflicting values are
"Not recorded" in the display columns, so Table 1 percentages use every
patient. Only aggregate counts are printed.

    python -m opera.run.flatten_ipi_records \\
        --metadata <sealed>/population_metadata.parquet \\
        --scores <computed>/lyfo_scores.parquet \\
        --output data/paper_256_v2/table1_inputs/clinical_scores.parquet

``--scores`` (repeatable) joins one-row-per-patient tables such as the IPI and
NCCN-IPI computed from LYFO components (opera.run.compute_prognostic_scores),
so Table 1 and the clinical-score comparison read one file.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

NOT_RECORDED = "Not recorded"
# Every system's categories collapsed to three groups (CLL-IPI "very high" and
# R-ISS/NCCN-IPI intermediate splits included). Unmapped labels are printed.
RISK = {"low": "Low", "low-intermediate": "Intermediate", "medium": "Intermediate",
        "intermediate": "Intermediate", "high-intermediate": "Intermediate",
        "high": "High", "very high": "High"}


def _clean(value):
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    text = str(value).strip()
    return text or None


def _agreed(values):
    """The single value all records share, or (None, conflicting)."""
    distinct = {v for v in values if v is not None}
    if len(distinct) == 1:
        return distinct.pop(), False
    return None, len(distinct) > 1


# NCCN-IPI (computed, DLBCL): low 0-1, intermediate 2-5 (low- and
# high-intermediate), high >= 6.
NCCN_RISK = {"low": "Low", "low-intermediate": "Intermediate",
             "high-intermediate": "Intermediate", "high": "High"}


def nccn_risk(groups, cohorts):
    """Collapsed NCCN-IPI for DLBCL patients; empty for other diseases."""
    collapsed = groups.astype("string").str.strip().str.lower().map(NCCN_RISK)
    collapsed = collapsed.fillna(NOT_RECORDED)
    return collapsed.where(cohorts.astype("string") == "DLBCL")


def risk_category(value):
    return RISK.get(str(value).strip().lower(), NOT_RECORDED) if value else NOT_RECORDED


def performance_group(value):
    try:
        score = float(str(value).replace(",", "."))
    except (TypeError, ValueError):
        return NOT_RECORDED
    if score in (0, 1):
        return "0–1"
    if score in (2, 3, 4):
        return "2–4"
    return NOT_RECORDED


def flatten(metadata):
    rows = []
    for subject_id, raw in zip(metadata.subject_id, metadata.ipi_records_json):
        records = [r for r in json.loads(raw or "[]")
                   if any(_clean(r.get(k)) for k in ("index", "IPI", "PS"))]
        pairs = [(_clean(r.get("index")), _clean(r.get("IPI"))) for r in records]
        pair, pair_conflict = _agreed([p if any(p) else None for p in pairs])
        ps, ps_conflict = _agreed([_clean(r.get("PS")) for r in records])
        index, ipi = pair if pair else (None, None)
        resolution = ("none" if not records else "single" if len(records) == 1
                      else "conflicting" if pair_conflict or ps_conflict else "agreeing")
        rows.append({"subject_id": subject_id, "index": index, "IPI": ipi, "PS": ps,
                     "risk_category": risk_category(ipi),
                     "performance_status": performance_group(ps),
                     "ipi_resolution": resolution})
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--metadata", type=Path, required=True,
                        help="Sealed population_metadata.parquet with ipi_records_json")
    parser.add_argument("--scores", type=Path, action="append", default=[],
                        help="One-row-per-patient score table to join (e.g. computed NCCN-IPI)")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"Refusing to overwrite {args.output}")
    import pyarrow.parquet as pq

    available = pq.ParquetFile(args.metadata).schema_arrow.names
    columns = ["subject_id", "ipi_records_json", *(["cohort_fine"] if "cohort_fine" in available else [])]
    metadata = pd.read_parquet(args.metadata, columns=columns)
    if metadata.subject_id.duplicated().any():
        raise SystemExit("Metadata must have one row per patient")
    table = flatten(metadata)
    for path in args.scores:
        scores = pd.read_parquet(path)
        if scores.subject_id.duplicated().any():
            raise SystemExit(f"{path} must have one row per patient")
        clash = (set(scores) & set(table)) - {"subject_id"}
        if clash:
            raise SystemExit(f"{path} repeats columns {sorted(clash)}")
        scores["subject_id"] = scores.subject_id.astype(table.subject_id.dtype)
        table = table.merge(scores, on="subject_id", how="left", validate="one_to_one")
        print(f"joined {path.name}: {int(table[scores.columns[1]].notna().sum()):,} "
              "patients with a value")
    table["risk_source"] = table["index"].where(table.risk_category != NOT_RECORDED)
    if "nccn_ipi_group" in table:
        if "cohort_fine" not in metadata:
            raise SystemExit("NCCN-IPI for Table 1 needs cohort_fine in the metadata (DLBCL only)")
        cohorts = metadata.set_index("subject_id").cohort_fine.reindex(table.subject_id)
        table["nccn_ipi_risk"] = nccn_risk(table.nccn_ipi_group, cohorts.reset_index(drop=True))
        # DLBCL: the NCCN-IPI computed from components decides the risk group
        # where it can be computed; otherwise the recorded index is kept.
        computed = table.nccn_ipi_risk.notna() & (table.nccn_ipi_risk != NOT_RECORDED)
        table.loc[computed, "risk_category"] = table.loc[computed, "nccn_ipi_risk"]
        table.loc[computed, "risk_source"] = "NCCN-IPI (computed)"
        print("nccn_ipi_risk (DLBCL):", table.nccn_ipi_risk.value_counts().to_dict())
        print("risk group source:", table.risk_source.fillna(NOT_RECORDED).value_counts().to_dict())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    table.to_parquet(args.output, index=False)
    print(f"{len(table):,} patients -> {args.output}")
    for column in ("ipi_resolution", "risk_category", "performance_status"):
        print(f"{column}:", table[column].value_counts().to_dict())
    unmapped = table.loc[table.IPI.notna() & (table.risk_category == NOT_RECORDED), "IPI"]
    if len(unmapped):
        print("IPI values without a risk mapping (left Not recorded):",
              sorted(unmapped.unique()))


if __name__ == "__main__":
    main()
