"""Fail-closed roster/config handoff check for the version-2 outcome release.

This does not approve clinical definitions or inspect patient rows. Run only
after the structural audit and before starting a new outcome-head training run.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import pandas as pd
import yaml

from opera.run.decide_outcome_panel import sha256_file
from opera.run.generate_sweep_configs import validate_registry


BAD_STRUCTURE = (
    "duplicate_subjects", "missing_subjects", "missing_index", "missing_censor",
    "censor_before_index", "event_on_or_before_index", "event_after_censor",
    "invalid_split", "split_index_mismatch",
)


def verify(release: Path, config_path: Path) -> dict:
    release = release.resolve()
    config_path = config_path.resolve()
    outcomes = release / "outcomes"
    registry_path = release / "experiment_registry_primary_v2.yaml"
    manifest_path = release / "release_manifest_INTERNAL.json"
    audit_path = release / "structure_audit_INTERNAL.csv"
    if not all(path.is_file() for path in (registry_path, manifest_path, audit_path, config_path)):
        raise ValueError("Release registry, manifest, structure audit or generated config is missing")
    if config_path.parent != release / "generated_configs":
        raise ValueError("Training config must be generated inside this release directory")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    registry = yaml.safe_load(registry_path.read_text(encoding="utf-8"))
    validate_registry(registry)
    names = registry["outcomes"]
    if (manifest.get("version") != 2
            or manifest.get("status") != "primary_release_candidate_pending_final_signoff"
            or manifest.get("held_out_used_for_selection") is not False):
        raise ValueError("Not a valid v2 release manifest or held-out selection policy")
    if manifest.get("registry_sha256") != sha256_file(registry_path):
        raise ValueError("Release registry hash differs from manifest")
    entries = manifest.get("outcomes", [])
    if (len(entries) != len(names) or [entry["outcome"] for entry in entries] != names
            or {path.stem for path in outcomes.glob("*.parquet")} != set(names)):
        raise ValueError("Release parquet inventory differs from registry/manifest")
    for entry in entries:
        path = outcomes / f"{entry['outcome']}.parquet"
        if not path.is_file() or sha256_file(path) != entry["sha256"]:
            raise ValueError(f"Release parquet hash mismatch: {entry['outcome']}")
    audit = pd.read_csv(audit_path)
    required = {"outcome", *BAD_STRUCTURE}
    if not required.issubset(audit) or set(audit["outcome"]) != set(names) or len(audit) != len(names):
        raise ValueError("Structure audit does not cover the approved roster exactly")
    if "error" in audit and audit["error"].notna().any():
        raise ValueError("Structure audit contains errors")
    if audit[list(BAD_STRUCTURE)].fillna(0).ne(0).any().any():
        raise ValueError("Structure audit contains chronology, ID or split violations")
    configured_dir = os.environ.get("BONSAI_OUTCOMES_DIR")
    if not configured_dir or Path(configured_dir).resolve() != outcomes.resolve():
        raise ValueError("BONSAI_OUTCOMES_DIR must point to this release's outcomes directory")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    configured = config.get("outcomes", {})
    if list(configured) != names:
        raise ValueError("Generated training config has a different roster/order")
    for name, spec in configured.items():
        if spec.get("outcome_file") != f"${{oc.env:BONSAI_OUTCOMES_DIR}}/{name}.parquet":
            raise ValueError(f"Generated config has a wrong outcome path: {name}")
    cross = config.get("cross_outcome", {})
    if cross.get("aggregation") != "endpoint_group_macro":
        raise ValueError("Selected training config is not endpoint-group macro")
    if cross.get("outcome_families") != registry["outcome_families"]:
        raise ValueError("Training config clinical families differ from approved registry")
    groups = cross.get("endpoint_groups", {})
    members = [name for group in groups.values() for name in group]
    if len(members) != len(names) or set(members) != set(names):
        raise ValueError("Endpoint groups do not cover every approved outcome exactly once")
    grouped_as = {name: group for group, members in groups.items() for name in members}
    for pair in (("alp_elevation_legacy_tier2plus", "alp_elevation_legacy_tier3plus"),
                 ("hyponatremia_g2plus", "sodium_below_125_mmol_l")):
        if all(name in grouped_as for name in pair) and grouped_as[pair[0]] != grouped_as[pair[1]]:
            raise ValueError(f"Nested endpoint pair was split across groups: {pair}")
    if config.get("training", {}).get("eval_monitor_metric") != "val/cr_cif_auc_endpoint_group_macro":
        raise ValueError("Checkpoint selection must use endpoint-group macro AUC")
    if config.get("training", {}).get("selection_min_cases") != 20 or config.get("training", {}).get("selection_min_controls") != 20:
        raise ValueError("Checkpoint selection support thresholds differ from approved config")
    if config.get("source_opera_ckpt"):
        raise ValueError("Old outcome-head checkpoints must not initialize this new roster")
    return {"status": "structure_and_config_verified_pending_clinical_signoff",
            "release_dir": str(release), "outcomes": len(names),
            "clinical_families": len(registry["outcome_families"]),
            "endpoint_groups": len(groups), "config": str(config_path.resolve())}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(verify(args.release_dir, args.config), indent=2))


if __name__ == "__main__":
    main()
