"""Compact, training/tuning-only overlap summary for lab candidate decisions.

Reads aggregate reports from export_candidate_outcomes_v1. No patient rows,
held-out counts, or exact small cells are copied to the output. The broad
bands are descriptive, not prespecified criteria for selecting endpoints.
All output remains INTERNAL until the study's disclosure review permits use.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

from opera.run.export_candidate_outcomes_v1 import ANALYTES, CREATININE_RELATIVE_OUTCOME


LAB_NAMES = set(sum((list(names) for names in ANALYTES.values()), [])) | {CREATININE_RELATIVE_OUTCOME}
COUNTS = ("old_eligible", "candidate_eligible", "shared_eligible", "lost_eligibility",
          "old_events_on_lost_eligibility", "both_event_on_shared",
          "old_only_event_on_shared", "candidate_only_event_on_shared")


def _hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def band(numerator: int, denominator: int, *, min_denominator: int = 20) -> str:
    """Suppress small denominators; otherwise emit a broad descriptive band."""
    if denominator < min_denominator:
        return "withheld"
    ratio = numerator / denominator
    if not 0 <= ratio <= 1:
        raise ValueError("Invalid overlap ratio")
    if ratio < 0.10:
        return "<10%"
    if ratio < 0.25:
        return "10-24%"
    if ratio < 0.50:
        return "25-49%"
    if ratio < 0.75:
        return "50-74%"
    if ratio < 0.90:
        return "75-89%"
    return ">=90%"


def aggregate(frame: pd.DataFrame, name: str, horizon: int) -> dict:
    selected = frame.loc[frame["outcome"].eq(name)
                         & frame["horizon_days"].astype(str).eq(str(horizon))
                         & frame["split"].isin(("train", "tuning"))]
    if len(selected) != 2 or set(selected["split"]) != {"train", "tuning"}:
        raise ValueError(f"Missing or duplicate train/tuning {horizon}-day overlap: {name}")
    if selected[list(COUNTS)].isna().any().any() or (selected[list(COUNTS)] < 0).any().any():
        raise ValueError(f"Invalid counts: {name}")
    counts = selected[list(COUNTS)].sum().astype(int).to_dict()
    if counts["old_eligible"] != counts["shared_eligible"] + counts["lost_eligibility"]:
        raise ValueError(f"Eligibility accounting failed: {name}")
    if counts["candidate_eligible"] < counts["shared_eligible"]:
        raise ValueError(f"Candidate eligibility accounting failed: {name}")
    both = counts["both_event_on_shared"]
    old_only = counts["old_only_event_on_shared"]
    new_only = counts["candidate_only_event_on_shared"]
    old_lost = counts["old_events_on_lost_eligibility"]
    return {
        "old_vs_candidate_event_overlap": band(both, both + old_only + new_only),
        "candidate_events_not_in_old": band(new_only, both + new_only),
        "old_events_missing_from_candidate_on_shared": band(old_only, both + old_only),
        "old_events_lost_with_eligibility": band(old_lost, both + old_only + old_lost),
        "eligibility_retained": band(counts["shared_eligible"], counts["old_eligible"]),
        "candidate_extra_eligibility": band(
            counts["candidate_eligible"] - counts["shared_eligible"],
            counts["candidate_eligible"]),
    }


def run(candidate_dir: Path, output_dir: Path) -> dict:
    if output_dir.exists():
        raise ValueError(f"Output already exists: {output_dir}")
    manifest_path = candidate_dir / "manifest_INTERNAL.json"
    if not manifest_path.is_file():
        raise ValueError(f"Candidate manifest missing: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if set(manifest.get("changed_outcomes", [])) != LAB_NAMES:
        raise ValueError("Candidate inventory does not match the 12 expected lab labels")
    main_path = candidate_dir / "event_overlap_INTERNAL.csv"
    relative_path = candidate_dir / "creatinine_definition_overlap_INTERNAL.csv"
    main = pd.read_csv(main_path)
    relative = pd.read_csv(relative_path)
    required = {"outcome", "split", "horizon_days", *COUNTS}
    for path, frame in ((main_path, main), (relative_path, relative)):
        if not required.issubset(frame):
            raise ValueError(f"{path}: missing columns {sorted(required - set(frame))}")
    if set(main["outcome"]) != LAB_NAMES:
        raise ValueError("Main overlap inventory differs from candidate manifest")
    if set(relative["outcome"]) != {CREATININE_RELATIVE_OUTCOME}:
        raise ValueError("Creatinine definition-overlap inventory is invalid")
    rows = []
    for name in sorted(LAB_NAMES):
        early = aggregate(main, name, 30)
        primary = aggregate(main, name, 365)
        rows.append({"comparison": "existing_vs_candidate", "outcome": name,
                     "eligibility_retained": primary["eligibility_retained"],
                     "candidate_extra_eligibility": primary["candidate_extra_eligibility"],
                     "event_overlap_30d": early["old_vs_candidate_event_overlap"],
                     "event_overlap_365d": primary["old_vs_candidate_event_overlap"],
                     "candidate_unique_events_365d": primary["candidate_events_not_in_old"],
                     "old_events_missing_on_shared_365d": primary[
                         "old_events_missing_from_candidate_on_shared"],
                     "old_events_lost_to_eligibility_365d": primary["old_events_lost_with_eligibility"]})
    early = aggregate(relative, CREATININE_RELATIVE_OUTCOME, 30)
    primary = aggregate(relative, CREATININE_RELATIVE_OUTCOME, 365)
    rows.append({"comparison": "candidate_grade_g2_vs_relative_rise",
                 "outcome": CREATININE_RELATIVE_OUTCOME,
                 "eligibility_retained": primary["eligibility_retained"],
                 "candidate_extra_eligibility": primary["candidate_extra_eligibility"],
                 "event_overlap_30d": early["old_vs_candidate_event_overlap"],
                 "event_overlap_365d": primary["old_vs_candidate_event_overlap"],
                 "candidate_unique_events_365d": primary["candidate_events_not_in_old"],
                 "old_events_missing_on_shared_365d": primary[
                     "old_events_missing_from_candidate_on_shared"],
                 "old_events_lost_to_eligibility_365d": primary["old_events_lost_with_eligibility"]})
    output_dir.mkdir(parents=True, exist_ok=False)
    result = pd.DataFrame(rows)
    result.to_csv(output_dir / "overlap_bands_INTERNAL.csv", index=False)
    headers = list(result)
    lines = ["# Lab candidate overlap, training and tuning only", "",
             "Bands are descriptive, not an inclusion rule. Denominators below 20 are withheld.",
             "Event overlap and candidate-unique-event bands use shared-eligible patients only;",
             "eligibility loss and additional candidate eligibility are shown separately.",
             "The last row compares candidate creatinine G2+",
             "against relative rise, not against the original export.", "",
             "| " + " | ".join(headers) + " |",
             "| " + " | ".join("---" for _ in headers) + " |"]
    lines.extend("| " + " | ".join(str(row[key]) for key in headers) + " |"
                 for row in rows)
    (output_dir / "overlap_bands_INTERNAL.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    provenance = {"candidate_manifest_sha256": _hash(manifest_path),
                  "event_overlap_sha256": _hash(main_path),
                  "creatinine_definition_overlap_sha256": _hash(relative_path),
                  "held_out_used": False, "exact_counts_exported": False,
                  "disclosure_status": "INTERNAL_NOT_CLEARED"}
    (output_dir / "provenance_INTERNAL.json").write_text(
        json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    return {"output_dir": str(output_dir.resolve()), "comparisons": len(rows),
            "held_out_used": False}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lab-candidates-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.lab_candidates_dir, args.output_dir), indent=2))


if __name__ == "__main__":
    main()
