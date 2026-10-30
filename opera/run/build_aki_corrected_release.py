"""Publish a one-endpoint amendment to an existing v2 release, without training."""
from __future__ import annotations

import argparse
import copy
import json
import shutil
import tempfile
from pathlib import Path

import pandas as pd
import yaml

from opera.run.audit_aki_correction import AKI, read_endpoint, require_fixed_population
from opera.run.audit_outcome_exports import audit_outcome
from opera.run.decide_outcome_panel import sha256_file
from opera.run.generate_sweep_configs import validate_registry
from opera.run.verify_approved_outcome_training_v2 import BAD_STRUCTURE

MANIFEST = "release_manifest_INTERNAL.json"
REGISTRY = "experiment_registry_primary_v2.yaml"


def validate_release(release: Path) -> tuple[dict, dict]:
    manifest = json.loads((release / MANIFEST).read_text())
    registry = yaml.safe_load((release / REGISTRY).read_text())
    validate_registry(registry)
    if (manifest.get("version") != 2 or manifest.get("held_out_used_for_selection") is not False
            or manifest.get("registry_sha256") != sha256_file(release / REGISTRY)):
        raise ValueError("Invalid parent release/registry identity")
    names = registry["outcomes"]
    if len(names) != 69 or AKI not in names:
        raise ValueError("Expected the existing locked 69-outcome roster including AKI")
    entries = manifest.get("outcomes", [])
    if [entry["outcome"] for entry in entries] != names:
        raise ValueError("Release manifest and registry roster/order differ")
    canonical = {p.stem for p in (release / "outcomes").glob("*.parquet") if "__" not in p.stem}
    if canonical != set(names):
        raise ValueError("Unexpected/missing canonical outcome files")
    for entry in entries:
        if sha256_file(release / "outcomes" / f"{entry['outcome']}.parquet") != entry["sha256"]:
            raise ValueError(f"Outcome release hash mismatch: {entry['outcome']}")
    return manifest, registry


def build(parent: Path, corrected: Path, comparison: Path, output: Path, reason: str) -> dict:
    parent, corrected, output = parent.resolve(), corrected.resolve(), output.resolve()
    if output.exists() or not reason.strip():
        raise ValueError("Require a fresh output directory and a correction rationale")
    manifest, registry = validate_release(parent)
    old = parent / "outcomes" / f"{AKI}.parquet"
    report = json.loads(comparison.read_text())
    if (report.get("outcome") != AKI or report.get("scope") != "train_and_tuning_only"
            or report.get("held_out_performance_evaluated") is not False
            or report.get("inputs", {}).get("old_outcome", {}).get("sha256") != sha256_file(old)
            or report.get("inputs", {}).get("overall_survival", {}).get("sha256")
            != sha256_file(parent / "outcomes" / "overall_survival.parquet")
            or report.get("corrected_outcome", {}).get("sha256") != sha256_file(corrected)):
        raise ValueError("Comparison does not identify these exact old/new release inputs")
    require_fixed_population(read_endpoint(old), read_endpoint(corrected))
    audit = audit_outcome(corrected)
    if "error" in audit or any(audit.get(key, 0) for key in BAD_STRUCTURE):
        raise ValueError("Corrected AKI failed structural audit")
    # Validate everything before publishing. The old release is never modified.
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".aki_release_", dir=output.parent) as temporary:
        staged = Path(temporary) / "release"
        (staged / "outcomes").mkdir(parents=True)
        amended = copy.deepcopy(manifest)
        for entry in amended["outcomes"]:
            name = entry["outcome"]
            source = corrected if name == AKI else parent / "outcomes" / f"{name}.parquet"
            target = staged / "outcomes" / f"{name}.parquet"
            shutil.copy2(source, target)
            if sha256_file(target) != sha256_file(source):
                raise IOError(f"Copy verification failed: {name}")
            entry.update(source=str(source), sha256=sha256_file(target))
        shutil.copy2(parent / REGISTRY, staged / REGISTRY)
        display = parent / "outcome_display_groups_primary_v2.yaml"
        if display.is_file():
            taxonomy = yaml.safe_load(display.read_text())
            members = [name for names in taxonomy.values() for name in names]
            if len(members) != len(registry["outcomes"]) or set(members) != set(registry["outcomes"]):
                raise ValueError("Parent display groups do not cover the unchanged roster exactly")
            shutil.copy2(display, staged / display.name)
            display_audit = parent / "display_group_audit_INTERNAL.json"
            if display_audit.is_file():
                saved = json.loads(display_audit.read_text())
                if (saved["registry_sha256"] != sha256_file(parent / REGISTRY)
                        or saved["output_sha256"] != sha256_file(display)):
                    raise ValueError("Parent display-group audit differs from the taxonomy")
                shutil.copy2(display_audit, staged / display_audit.name)
        audits = [audit_outcome(staged / "outcomes" / f"{name}.parquet") for name in registry["outcomes"]]
        if any("error" in item or any(item.get(key, 0) for key in BAD_STRUCTURE) for item in audits):
            raise ValueError("Amended release has structural violations")
        pd.DataFrame(audits).to_csv(staged / "structure_audit_INTERNAL.csv", index=False)
        shutil.copy2(comparison, staged / "aki_comparison_INTERNAL.json")
        notes_path = parent / "release_notes_INTERNAL.yaml"
        notes = yaml.safe_load(notes_path.read_text()) if notes_path.exists() else {}
        notes = notes or {}
        notes[AKI] = reason
        (staged / "release_notes_INTERNAL.yaml").write_text(yaml.safe_dump(notes, sort_keys=False))
        # Only portable release-local templates are carried forward. Their existing
        # selection metric is preserved; the paper bundle retains its own BCE rule.
        for path in (parent / "generated_configs").glob("*.yaml"):
            config = yaml.safe_load(path.read_text())
            targets = config.get("outcomes", {}) if isinstance(config, dict) else {}
            if (list(targets) == registry["outcomes"]
                    and all(spec.get("outcome_file") == f"${{oc.env:BONSAI_OUTCOMES_DIR}}/{name}.parquet"
                            for name, spec in targets.items())):
                (staged / "generated_configs").mkdir(exist_ok=True)
                shutil.copy2(path, staged / "generated_configs" / path.name)
        amended.update(status="primary_release_candidate_pending_final_signoff",
                       revision="aki_baseline_window_correction_v1",
                       parent_release=str(parent), parent_manifest_sha256=sha256_file(parent / MANIFEST),
                       changed_outcomes=[AKI], unchanged_outcomes=[n for n in registry["outcomes"] if n != AKI],
                       correction_reason=reason, comparison_sha256=sha256_file(comparison),
                       held_out_used_for_selection=False)
        (staged / MANIFEST).write_text(json.dumps(amended, indent=2) + "\n")
        # Do not relabel the parent's horizon-support/clinical sign-off reports as new.
        staged.rename(output)
    return {"release_dir": str(output), "changed_outcomes": [AKI],
            "unchanged_outcomes": 68, "status": amended["status"]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-release", type=Path, required=True)
    parser.add_argument("--corrected-outcome", type=Path, required=True)
    parser.add_argument("--comparison", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--reason", required=True)
    args = parser.parse_args()
    print(json.dumps(build(args.parent_release, args.corrected_outcome, args.comparison,
                           args.output_dir, args.reason), indent=2))


if __name__ == "__main__":
    main()
