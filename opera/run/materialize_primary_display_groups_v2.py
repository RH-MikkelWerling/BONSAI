"""Create an exact, display-only grouping for an approved outcome release.

This does not change training loss, checkpoint selection, or patient files.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from opera.run.decide_outcome_panel import sha256_file
from opera.run.generate_sweep_configs import validate_registry


ADDITION_GROUPS = {
    "creatinine_rise_gt1p5x_preindex_baseline": "Kidney injury and renal support",
    "potassium_below_3p0_mmol_l": "Sodium and potassium",
}


def build_display_groups(registry: dict, policy: dict, old_groups: dict) -> dict[str, list[str]]:
    names = registry["outcomes"]
    allowed = set(names)
    aliases = policy["rename"]
    source_members = [name for members in old_groups.values() for name in members]
    if len(source_members) != len(set(source_members)):
        raise ValueError("Source display taxonomy has duplicate members")
    groups: dict[str, list[str]] = {}
    for label, members in old_groups.items():
        selected = [aliases.get(name, name) for name in members
                    if aliases.get(name, name) in allowed]
        if selected:
            groups[label] = selected
    for name, label in ADDITION_GROUPS.items():
        if name in allowed:
            if label not in groups:
                raise ValueError(f"Missing clinical display group for {name}: {label}")
            groups[label].append(name)
    assigned = [name for members in groups.values() for name in members]
    if len(assigned) != len(names) or set(assigned) != allowed:
        raise ValueError(f"Display groups do not cover approved roster exactly: "
                         f"missing={sorted(allowed - set(assigned))}, "
                         f"extra={sorted(set(assigned) - allowed)}")
    return groups


def run(release: Path, policy_path: Path, old_groups_path: Path) -> dict:
    registry_path = release / "experiment_registry_primary_v2.yaml"
    output_path = release / "outcome_display_groups_primary_v2.yaml"
    audit_path = release / "display_group_audit_INTERNAL.json"
    if output_path.exists() or audit_path.exists():
        raise ValueError("Display grouping already exists; do not overwrite a release artifact")
    registry = yaml.safe_load(registry_path.read_text(encoding="utf-8"))
    validate_registry(registry)
    policy = yaml.safe_load(policy_path.read_text(encoding="utf-8"))
    old = yaml.safe_load(old_groups_path.read_text(encoding="utf-8"))
    groups = build_display_groups(registry, policy, old)
    output_path.write_text(yaml.safe_dump(groups, sort_keys=False), encoding="utf-8")
    audit = {
        "status": "display_only_exact_coverage",
        "n_outcomes": len(registry["outcomes"]), "n_display_groups": len(groups),
        "registry_sha256": sha256_file(registry_path),
        "policy_sha256": sha256_file(policy_path),
        "source_groups_sha256": sha256_file(old_groups_path),
        "output_sha256": sha256_file(output_path),
    }
    audit_path.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    return audit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-dir", type=Path, required=True)
    parser.add_argument("--policy", type=Path,
                        default=Path("opera/configs/outcome_release_primary_v2.yaml"))
    parser.add_argument("--source-groups", type=Path,
                        default=Path("opera/configs/outcome_display_groups_v1.yaml"))
    args = parser.parse_args()
    print(json.dumps(run(args.release_dir, args.policy, args.source_groups), indent=2))


if __name__ == "__main__":
    main()
