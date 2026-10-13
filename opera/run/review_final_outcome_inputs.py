"""Read-only reconciliation of the assembled panel and isolated candidate audits.

Outputs contain aggregate INTERNAL study information, never patient rows. This
is a triage report, not a clinical sign-off or a training registry.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import yaml

from opera.run.decide_outcome_panel import sha256_file
from opera.run.export_candidate_outcomes_v1 import ANALYTES, CREATININE_RELATIVE_OUTCOME
from opera.run.export_candidate_tls_v1 import NAMES, STRICT_NAMES


LAB_NAMES = set(sum((list(names) for names in ANALYTES.values()), [])) | {CREATININE_RELATIVE_OUTCOME}
TLS_NAMES = set(NAMES + STRICT_NAMES)
SPLITS = {"train", "tuning", "held_out"}


def _csv(path: Path, columns: set[str]) -> pd.DataFrame:
    if not path.is_file():
        raise ValueError(f"Missing required report: {path}")
    frame = pd.read_csv(path)
    absent = columns - set(frame)
    if absent:
        raise ValueError(f"{path}: missing columns {sorted(absent)}")
    return frame


def _support(frame: pd.DataFrame, name: str, horizon: int) -> dict[str, int]:
    selected = frame.loc[frame["outcome"].eq(name) & frame["horizon_days"].eq(horizon)]
    if len(selected) != 3 or set(selected["split"]) != SPLITS:
        raise ValueError(f"Incomplete/duplicate {horizon}-day support for {name}")
    if selected["n_events"].isna().any() or (selected["n_events"] < 0).any():
        raise ValueError(f"Invalid event counts for {name}")
    return dict(zip(selected["split"], selected["n_events"].astype(int)))


def _check_hash(path: Path, expected: str, issues: list[dict], name: str) -> None:
    if not path.is_file():
        issues.append({"severity": "BLOCK", "outcome": name, "problem": "input file missing", "evidence": str(path)})
    elif sha256_file(path) != expected:
        issues.append({"severity": "BLOCK", "outcome": name, "problem": "file hash changed since audit", "evidence": str(path)})


def _manifest_inputs(manifest: dict, expected: set[Path], issues: list[dict], label: str) -> None:
    recorded = manifest.get("input_sha256", {})
    if not isinstance(recorded, dict) or not recorded:
        issues.append({"severity": "BLOCK", "outcome": label,
                       "problem": "candidate input hashes absent", "evidence": ""})
        return
    for required in expected:
        matches = [(Path(key), value) for key, value in recorded.items()
                   if Path(key).resolve() == required.resolve()]
        if len(matches) != 1:
            issues.append({"severity": "BLOCK", "outcome": label,
                           "problem": "candidate uses a different outcome release",
                           "evidence": str(required)})
    for raw_path, expected_hash in recorded.items():
        _check_hash(Path(raw_path), expected_hash, issues, label)


def review(panel: Path, panel_review: Path, base: Path, lab: Path | None,
           tls: Path | None, registry: Path, rules: Path, output: Path) -> dict:
    if output.exists():
        raise ValueError(f"Output already exists: {output}")
    for path in (panel, panel_review, base):
        if not path.is_dir():
            raise ValueError(f"Required directory does not exist: {path}")
    registry_data = yaml.safe_load(registry.read_text(encoding="utf-8"))
    rule_data = yaml.safe_load(rules.read_text(encoding="utf-8"))
    names = registry_data["outcomes"]
    if len(names) != len(set(names)):
        raise ValueError("Duplicate registry outcome")
    issues: list[dict] = []
    review_summary = json.loads((panel_review / "summary_INTERNAL.json").read_text(encoding="utf-8"))
    if Path(review_summary["outcomes_dir"]).resolve() != panel.resolve():
        issues.append({"severity": "BLOCK", "outcome": "panel", "problem": "decision report targets another panel", "evidence": review_summary["outcomes_dir"]})
    decision_inputs_current = True
    for path, field in ((registry, "registry_sha256"), (rules, "rules_sha256")):
        if review_summary.get(field) != sha256_file(path):
            decision_inputs_current = False
            issues.append({"severity": "BLOCK", "outcome": "panel", "problem": f"{field} changed since decision audit", "evidence": str(path)})
    expected = set(names)
    found = {p.stem for p in panel.glob("*.parquet") if "__" not in p.stem}
    for name in sorted(expected - found):
        issues.append({"severity": "BLOCK", "outcome": name, "problem": "panel parquet missing", "evidence": ""})
    for name in sorted(found - expected):
        issues.append({"severity": "BLOCK", "outcome": name, "problem": "unregistered panel parquet", "evidence": ""})
    hashes = _csv(panel_review / "input_files_sha256_INTERNAL.csv", {"outcome", "sha256"})
    if set(hashes["outcome"]) != expected or hashes["outcome"].duplicated().any():
        issues.append({"severity": "BLOCK", "outcome": "panel", "problem": "audited file inventory differs from registry", "evidence": ""})
    for row in hashes.itertuples(index=False):
        _check_hash(panel / f"{row.outcome}.parquet", row.sha256, issues, row.outcome)
    assembly_path = panel / "assembly_manifest_INTERNAL.json"
    if assembly_path.is_file():
        assembly = json.loads(assembly_path.read_text(encoding="utf-8"))
        if assembly.get("registry_sha256") != sha256_file(registry):
            issues.append({"severity": "BLOCK", "outcome": "panel", "problem": "assembly registry hash differs", "evidence": ""})
    else:
        issues.append({"severity": "BLOCK", "outcome": "panel", "problem": "assembly manifest missing", "evidence": ""})
    decisions = _csv(panel_review / "panel_decisions_INTERNAL.csv", {"outcome", "disposition", "reason"})
    support = _csv(panel_review / "horizon_support_INTERNAL.csv", {"outcome", "split", "horizon_days", "n_events"})
    structure = _csv(panel_review / "structure_audit_INTERNAL.csv", {"outcome"})
    if set(decisions["outcome"]) != expected or decisions["outcome"].duplicated().any():
        issues.append({"severity": "BLOCK", "outcome": "panel", "problem": "decision inventory incomplete", "evidence": ""})
    bad_columns = ("duplicate_subjects", "missing_subjects", "missing_index", "missing_censor",
                   "censor_before_index", "event_on_or_before_index", "event_after_censor",
                   "invalid_split", "split_index_mismatch")
    if set(structure["outcome"]) != expected or not set(bad_columns).issubset(structure):
        issues.append({"severity": "BLOCK", "outcome": "panel", "problem": "structure audit incomplete", "evidence": ""})
    else:
        for row in structure.to_dict("records"):
            if any(pd.isna(row[key]) or row[key] != 0 for key in bad_columns):
                issues.append({"severity": "BLOCK", "outcome": row["outcome"], "problem": "structural audit failed", "evidence": ""})
    for row in decisions.to_dict("records"):
        if decision_inputs_current and row["disposition"] != "primary_candidate":
            issues.append({"severity": "REVIEW", "outcome": row["outcome"],
                           "problem": row["disposition"], "evidence": row["reason"]})
        _support(support, row["outcome"], int(rule_data["primary_horizon_days"]))

    candidate_rows = []
    train_floor = int(rule_data["min_train_events"])
    tuning_floor = int(rule_data["min_tuning_events"])
    for label, directory, expected_names, horizon in (
        ("lab", lab, LAB_NAMES, 365), ("tls", tls, TLS_NAMES, 30)
    ):
        if directory is None:
            issues.append({"severity": "BLOCK", "outcome": label,
                           "problem": "candidate audit not supplied", "evidence": "Cannot complete roster review"})
            continue
        manifest_path = directory / "manifest_INTERNAL.json"
        if not manifest_path.is_file():
            issues.append({"severity": "BLOCK", "outcome": label, "problem": "candidate manifest missing", "evidence": str(directory)})
            continue
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        source = base / "overall_survival.parquet"
        inputs = {source}
        if label == "tls":
            audit = base / "anemia_g2plus__audit.parquet"
            inputs.add(audit)
            if manifest.get("prediction_horizon_days") != 30:
                issues.append({"severity": "BLOCK", "outcome": label, "problem": "TLS horizon is not 30 days", "evidence": ""})
        elif Path(manifest.get("old_outcomes_dir", "")).resolve() != base.resolve():
            issues.append({"severity": "BLOCK", "outcome": label, "problem": "lab comparison uses another base release", "evidence": manifest.get("old_outcomes_dir", "")})
        _manifest_inputs(manifest, inputs, issues, label)
        observed = {p.stem for p in (directory / "outcomes").glob("*.parquet")}
        if observed != expected_names:
            issues.append({"severity": "BLOCK", "outcome": label, "problem": "candidate inventory mismatch",
                           "evidence": f"missing={sorted(expected_names-observed)} extra={sorted(observed-expected_names)}"})
        recorded_outputs = manifest.get("output_sha256", {})
        for name in expected_names:
            if name not in recorded_outputs:
                issues.append({"severity": "BLOCK", "outcome": name, "problem": "candidate hash absent", "evidence": ""})
            else:
                _check_hash(directory / "outcomes" / f"{name}.parquet", recorded_outputs[name], issues, name)
        candidate_support = _csv(directory / "horizon_support_INTERNAL.csv", {"outcome", "split", "horizon_days", "n_events"})
        for name in sorted(expected_names):
            counts = _support(candidate_support, name, horizon)
            passes = counts["train"] >= train_floor and counts["tuning"] >= tuning_floor
            candidate_rows.append({"source": label, "outcome": name, "horizon_days": horizon,
                                   "train_events": counts["train"], "tuning_events": counts["tuning"],
                                   "passes_event_floor": passes})
            if not passes:
                issues.append({"severity": "REVIEW", "outcome": name, "problem": "candidate below event floor",
                               "evidence": f"{horizon}-day training/tuning support"})
        if label == "lab":
            availability = _csv(directory / "baseline_availability_INTERNAL.csv",
                                {"analyte", "old_eligible", "candidate_eligible", "excluded_missing_baseline_or_reference"})
            overlap = _csv(directory / "event_overlap_INTERNAL.csv",
                           {"outcome", "split", "horizon_days", "lost_eligibility",
                            "old_only_event_on_shared", "candidate_only_event_on_shared"})
            creatinine_overlap = _csv(directory / "creatinine_definition_overlap_INTERNAL.csv",
                                      {"outcome", "split", "horizon_days", "jaccard_event_overlap_on_shared"})
            analyte_by_outcome = {name: analyte for analyte, outcomes in ANALYTES.items()
                                  for name in outcomes}
            analyte_by_outcome[CREATININE_RELATIVE_OUTCOME] = "creatinine_relative"
            availability_by_analyte = availability.set_index("analyte")
            for row in availability.to_dict("records"):
                if row["excluded_missing_baseline_or_reference"]:
                    issues.append({"severity": "REVIEW", "outcome": row["analyte"],
                                   "problem": "baseline/reference requirement reduces eligibility",
                                   "evidence": "See baseline_availability_INTERNAL.csv"})
            for name in sorted(LAB_NAMES):
                changed = overlap.loc[overlap["outcome"].eq(name) & overlap["split"].eq("train")
                                  & overlap["horizon_days"].astype(str).eq("365")]
                if len(changed) != 1:
                    raise ValueError(f"Missing training day-365 overlap for {name}")
                row = changed.iloc[0]
                evidence = next(item for item in candidate_rows if item["source"] == "lab" and item["outcome"] == name)
                baseline = availability_by_analyte.loc[analyte_by_outcome[name]]
                evidence["old_eligible_all_splits"] = int(baseline["old_eligible"])
                evidence["candidate_eligible_all_splits"] = int(baseline["candidate_eligible"])
                evidence["baseline_or_reference_excluded_all_splits"] = int(
                    baseline["excluded_missing_baseline_or_reference"])
                evidence["old_only_events_shared_train_365d"] = int(row["old_only_event_on_shared"])
                evidence["candidate_only_events_shared_train_365d"] = int(row["candidate_only_event_on_shared"])
                evidence["lost_eligibility_train"] = int(row["lost_eligibility"])
                if row["old_only_event_on_shared"] or row["candidate_only_event_on_shared"]:
                    issues.append({"severity": "REVIEW", "outcome": name,
                                   "problem": "candidate changes event membership",
                                   "evidence": "See event_overlap_INTERNAL.csv"})
            issues.append({"severity": "REVIEW", "outcome": CREATININE_RELATIVE_OUTCOME,
                           "problem": "clinical choice: separate relative-rise target or sensitivity",
                           "evidence": "See creatinine_definition_overlap_INTERNAL.csv"})
            relative = creatinine_overlap.loc[
                creatinine_overlap["outcome"].eq(CREATININE_RELATIVE_OUTCOME)
                & creatinine_overlap["split"].eq("train")
                & creatinine_overlap["horizon_days"].astype(str).eq("365")]
            if len(relative) != 1:
                raise ValueError("Missing training day-365 creatinine-definition overlap")
            evidence = next(item for item in candidate_rows if item["outcome"] == CREATININE_RELATIVE_OUTCOME)
            evidence["jaccard_vs_candidate_creatinine_g2_train_365d"] = relative.iloc[0][
                "jaccard_event_overlap_on_shared"]
        else:
            monitoring = _csv(directory / "tls_monitoring_and_events_INTERNAL.csv",
                              {"split", "n_shared_lab_eligible", "n_complete_four_axis_same_day"})
            if set(monitoring["split"]) != SPLITS:
                raise ValueError("TLS monitoring report lacks a split")
            for row in candidate_rows:
                if row["source"] == "tls":
                    for split in ("train", "tuning"):
                        observed = monitoring.loc[monitoring["split"].eq(split)]
                        if len(observed) != 1:
                            raise ValueError(f"TLS monitoring report has duplicate {split}")
                        row[f"shared_lab_eligible_{split}"] = int(observed.iloc[0]["n_shared_lab_eligible"])
                        row[f"complete_panel_eligible_{split}"] = int(
                            observed.iloc[0]["n_complete_four_axis_same_day"])
            issues.append({"severity": "REVIEW", "outcome": "tls_laboratory_with_kidney_early_v1",
                           "problem": "kidney/proxy definition requires clinical sign-off",
                           "evidence": "Pre-existing creatinine elevation and incomplete axes not adjudicated"})

    output.mkdir(parents=True, exist_ok=False)
    pd.DataFrame(issues, columns=["severity", "outcome", "problem", "evidence"]).to_csv(
        output / "issues_INTERNAL.csv", index=False)
    pd.DataFrame(candidate_rows, columns=["source", "outcome", "horizon_days", "train_events",
                                          "tuning_events", "passes_event_floor"] + sorted({
                                              key for row in candidate_rows for key in row
                                          } - {"source", "outcome", "horizon_days", "train_events",
                                               "tuning_events", "passes_event_floor"})).to_csv(
                                                   output / "candidate_support_INTERNAL.csv", index=False)
    summary = {"status": "BLOCKED" if any(i["severity"] == "BLOCK" for i in issues) else "REVIEW_REQUIRED",
               "panel_outcomes": len(names), "candidate_versions": len(candidate_rows),
               "blocks": sum(i["severity"] == "BLOCK" for i in issues),
               "review_items": sum(i["severity"] == "REVIEW" for i in issues),
               "primary_candidates_in_existing_panel": (
                   int(decisions["disposition"].eq("primary_candidate").sum())
                   if decision_inputs_current else None),
               "note": "Not a final roster; exact small cells remain in INTERNAL CSVs."}
    (output / "summary_INTERNAL.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    lines = ["# Outcome release triage", "", f"Status: {summary['status']}",
             f"Existing panel: {len(names)} outcomes; candidate versions reviewed: {len(candidate_rows)}.",
             f"Blocks: {summary['blocks']}; review items: {summary['review_items']}.",
             "", "This does not freeze the roster. Counts and hashes are in the INTERNAL CSVs.", ""]
    if not decision_inputs_current:
        lines += ["The panel decision report was produced with different registry/rules hashes.",
                  "Its clinical dispositions are suppressed below until that audit is rerun.", ""]
    lines += ["## Every candidate version", "",
              "The 12 lab rows are alternative definitions of existing labels except the separate",
              "creatinine-relative-rise target. The four TLS rows are separate candidates.", "",
              "| Source | Candidate outcome | Horizon | Event floor |", "| --- | --- | ---: | --- |"]
    lines += [f"| {row['source']} | {row['outcome']} | {row['horizon_days']}d | "
              f"{'passes' if row['passes_event_floor'] else 'below floor'} |"
              for row in candidate_rows]
    lines += ["", "## Problems to resolve", ""]
    lines += [f"- {item['severity']}: {item['outcome']} — {item['problem']}"
              for item in issues] or ["- None identified by these automated checks."]
    (output / "triage_INTERNAL.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel-dir", type=Path, required=True)
    parser.add_argument("--panel-review-dir", type=Path, required=True)
    parser.add_argument("--base-dir", type=Path, required=True)
    parser.add_argument("--lab-candidates-dir", type=Path)
    parser.add_argument("--tls-candidates-dir", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--registry", type=Path, default=Path("opera/configs/experiment_registry.yaml"))
    parser.add_argument("--rules", type=Path, default=Path("opera/configs/outcome_panel_decision_v1.yaml"))
    args = parser.parse_args()
    print(json.dumps(review(args.panel_dir, args.panel_review_dir, args.base_dir,
                            args.lab_candidates_dir, args.tls_candidates_dir,
                            args.registry, args.rules, args.output_dir), indent=2))


if __name__ == "__main__":
    main()
