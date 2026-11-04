"""Audit the expanded clinical comparison roster without reading predictions.

Run on the server with its actual score configuration and approved outcome
configuration. Output contains score/population definitions, never patient rows.
This checks completeness; clinical correctness still needs the recorded review.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import yaml


def audit_specs(specs, *, expected_model, outcomes):
    rows = []
    for spec in specs:
        issues = []
        if spec.get("endpoint_role") != "native":
            issues.append("missing_explicit_native_or_proxy_endpoint_mapping")
        if not spec.get("outcome"):
            issues.append("unresolved_outcome")
        elif spec["outcome"] not in outcomes:
            issues.append("outcome_not_in_approved_panel")
        if not spec.get("cohorts"):
            issues.append("empty_or_unresolved_population")
        if not spec.get("original_endpoint"):
            issues.append("original_endpoint_not_documented")
        if spec.get("model") != expected_model:
            issues.append("model_differs_from_selected_primary")
        if spec.get("cutoff_rule") != "cohort_size_matched":
            issues.append("primary_groups_not_cohort_size_matched")
        if spec.get("threshold_split", "train") != "train":
            issues.append("complementary_anchors_not_training_based")
        horizons = spec.get("horizons_days") or [spec.get("horizon_days")]
        if not horizons or any(h not in [30, 60, 90, 180, 365, 730] for h in horizons):
            issues.append("unresolved_or_unregistered_horizon")
        if not spec.get("applicability_review"):
            issues.append("population_applicability_review_not_recorded")
        if not spec.get("lock_provenance"):
            issues.append("definition_lock_provenance_not_recorded")
        rows.append(dict(score_id=spec.get("id"), index=spec.get("score_label"),
                         model=spec.get("model"), cohorts=spec.get("cohorts"),
                         outcome=spec.get("outcome"), original_endpoint=spec.get("original_endpoint"),
                         endpoint_match=spec.get("endpoint_match", "unspecified"),
                         endpoint_proxy_note=spec.get("endpoint_proxy_note"),
                         endpoint_source=spec.get("endpoint_source"),
                         horizons_days=horizons, cutoff_rule=spec.get("cutoff_rule"),
                         applicability_review=spec.get("applicability_review"),
                         status="needs_resolution" if issues else "configuration_complete",
                         issues=issues))
    return dict(status="needs_resolution" if not rows or any(r["issues"] for r in rows)
                else "configuration_complete", scores=rows,
                note="Completeness audit only; it does not certify clinical validity or read test predictions.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--score-config", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--outcomes-config", type=Path, required=True)
    parser.add_argument("--expected-model", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError("Use a fresh output directory")
    from opera.run.analyse_locked_predictions import read_table
    from opera.run.render_clinical_score_panels import expand_index_scores

    config = yaml.safe_load(args.score_config.read_text())
    training = yaml.safe_load(args.outcomes_config.read_text())
    if not training.get("outcomes"):
        raise ValueError("Provide a resolved approved training configuration containing outcomes")
    specs = expand_index_scores(config, read_table(args.metadata))
    result = audit_specs(specs, expected_model=args.expected_model, outcomes=set(training["outcomes"]))
    if config.get("index_endpoint", "native") != "native":
        result.update(status="needs_resolution", configuration_issue="primary_index_endpoint_not_native")
    args.output_dir.mkdir(parents=True)
    (args.output_dir / "clinical_endpoint_audit.json").write_text(json.dumps(result, indent=2) + "\n")
    flat = [{k: json.dumps(v) if isinstance(v, (list, dict)) else v for k, v in row.items()}
            for row in result["scores"]]
    pd.DataFrame(flat).to_csv(args.output_dir / "clinical_endpoint_roster.csv", index=False)
    print(json.dumps({"status": result["status"], "scores": len(flat),
                      "output_dir": str(args.output_dir)}, indent=2))
    if result["status"] != "configuration_complete":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
