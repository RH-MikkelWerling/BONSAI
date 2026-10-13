"""Audit and materialize proposed training families without changing the release.

The mapping is built from the release-specific display taxonomy, so outcome
renames and the approved roster are resolved before assigning training families.
It writes an immutable proposal and aggregate-only audit to a new directory.
It does not change a training config, loss, checkpoint metric, or patient data.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import yaml

from opera.functional.endpoint_groups import validate_endpoint_groups
from opera.run.decide_outcome_panel import sha256_file
from opera.run.generate_sweep_configs import validate_registry
from opera.run.materialize_primary_display_groups_v2 import build_display_groups


def build_training_families(
    registry: dict, display_groups: dict[str, list[str]],
    endpoint_groups: dict[str, list[str]], policy: dict[str, list[str]],
    *, min_groups: int = 3, max_groups: int = 6,
) -> tuple[dict[str, list[str]], dict]:
    names = registry["outcomes"]
    validate_registry(registry)
    validate_endpoint_groups(endpoint_groups, names)
    display_names = [name for members in display_groups.values() for name in members]
    if len(display_names) != len(names) or set(display_names) != set(names):
        raise ValueError("Display taxonomy does not cover the release exactly once")
    sources = [label for labels in policy.values() for label in labels]
    if len(sources) != len(set(sources)) or set(sources) != set(display_groups):
        raise ValueError("Training-family policy must cover every display group exactly once")
    if not policy or any(not labels for labels in policy.values()):
        raise ValueError("Every training family needs at least one display group")
    families = {
        family: [name for label in labels for name in display_groups[label]]
        for family, labels in policy.items()
    }
    assigned = [name for members in families.values() for name in members]
    if len(assigned) != len(names) or set(assigned) != set(names):
        raise ValueError("Training families do not cover the release exactly once")
    owner = {name: family for family, members in families.items() for name in members}
    family_groups = {family: [] for family in families}
    for group, members in endpoint_groups.items():
        owners = {owner[name] for name in members}
        if len(owners) != 1:
            raise ValueError(f"Endpoint group crosses training families: {group}: {owners}")
        family_groups[next(iter(owners))].append(group)
    counts = {family: len(groups) for family, groups in family_groups.items()}
    outside = {family: count for family, count in counts.items()
               if not min_groups <= count <= max_groups}
    if outside:
        raise ValueError(f"Training families outside {min_groups}-{max_groups} endpoint groups: {outside}")
    audit = {
        "status": "training_family_proposal_not_active",
        "n_outcomes": len(names), "n_endpoint_groups": len(endpoint_groups),
        "n_display_groups": len(display_groups), "n_training_families": len(families),
        "minimum_endpoint_groups": min_groups, "maximum_endpoint_groups": max_groups,
        "families": {
            family: {"n_outcomes": len(families[family]),
                     "n_endpoint_groups": counts[family],
                     "endpoint_groups": family_groups[family],
                     "source_display_groups": policy[family]}
            for family in families
        },
    }
    return families, audit


def load_release_display_groups(
    release: Path, registry: dict, registry_path: Path,
    release_policy_path: Path, display_source_path: Path,
) -> tuple[dict[str, list[str]], dict]:
    display_path = release / "outcome_display_groups_primary_v2.yaml"
    display_audit_path = release / "display_group_audit_INTERNAL.json"
    if display_path.exists() != display_audit_path.exists():
        raise ValueError("Release has only one of display taxonomy and its audit")
    if display_path.exists():
        display = yaml.safe_load(display_path.read_text(encoding="utf-8"))
        display_audit = json.loads(display_audit_path.read_text(encoding="utf-8"))
        if (display_audit.get("status") != "display_only_exact_coverage"
                or display_audit.get("registry_sha256") != sha256_file(registry_path)
                or display_audit.get("output_sha256") != sha256_file(display_path)):
            raise ValueError("Display taxonomy audit belongs to another release")
        return display, {"mode": "existing_audited_release_file",
                         "display_sha256": sha256_file(display_path),
                         "audit_sha256": sha256_file(display_audit_path)}
    release_policy = yaml.safe_load(release_policy_path.read_text(encoding="utf-8"))
    source = yaml.safe_load(display_source_path.read_text(encoding="utf-8"))
    display = build_display_groups(registry, release_policy, source)
    return display, {"mode": "derived_from_checked_in_taxonomy",
                     "release_policy_sha256": sha256_file(release_policy_path),
                     "display_source_sha256": sha256_file(display_source_path)}


def run(release: Path, output: Path, policy_path: Path,
        release_policy_path: Path = Path("opera/configs/outcome_release_primary_v2.yaml"),
        display_source_path: Path = Path("opera/configs/outcome_display_groups_v1.yaml")) -> dict:
    if output.exists():
        raise ValueError(f"Output already exists: {output}")
    registry_path = release / "experiment_registry_primary_v2.yaml"
    base_config_path = release / "generated_configs" / "direct_cr_bce_lowrank_endpoint_groups_v1.yaml"
    registry = yaml.safe_load(registry_path.read_text(encoding="utf-8"))
    display, display_provenance = load_release_display_groups(
        release, registry, registry_path, release_policy_path, display_source_path
    )
    config = yaml.safe_load(base_config_path.read_text(encoding="utf-8"))
    policy = yaml.safe_load(policy_path.read_text(encoding="utf-8"))
    families, audit = build_training_families(
        registry, display, config["cross_outcome"]["endpoint_groups"], policy
    )
    if list(config["outcomes"]) != registry["outcomes"]:
        raise ValueError("Base config outcome roster/order differs from release")
    if config["cross_outcome"]["aggregation"] != "endpoint_group_macro":
        raise ValueError("Expected approved endpoint-group macro base config")
    if config["cross_outcome"]["outcome_families"] != registry["outcome_families"]:
        raise ValueError("Base config family labels differ from the release registry")
    if config["training"]["eval_monitor_metric"] != "val/cr_cif_auc_endpoint_group_macro":
        raise ValueError("Base config selects checkpoints with a different metric")
    audit["source_sha256"] = {
        "registry": sha256_file(registry_path),
        "base_config": sha256_file(base_config_path), "policy": sha256_file(policy_path),
    }
    audit["display_provenance"] = display_provenance
    output.mkdir(parents=True)
    display_snapshot_path = output / "outcome_display_groups_primary_v2.yaml"
    family_path = output / "outcome_training_families_primary_v1.yaml"
    config_path = output / "direct_cr_bce_lowrank_endpoint_groups_training_families_v1.yaml"
    audit_path = output / "training_family_audit_INTERNAL.json"
    display_snapshot_path.write_text(yaml.safe_dump(display, sort_keys=False), encoding="utf-8")
    family_path.write_text(yaml.safe_dump(families, sort_keys=False), encoding="utf-8")
    candidate_config = copy.deepcopy(config)
    candidate_config["cross_outcome"]["outcome_families"] = families
    config_path.write_text(
        "# @package _global_\n" + yaml.safe_dump(candidate_config, sort_keys=False),
        encoding="utf-8",
    )
    audit["families_sha256"] = sha256_file(family_path)
    audit["display_snapshot_sha256"] = sha256_file(display_snapshot_path)
    audit["candidate_config_sha256"] = sha256_file(config_path)
    audit["unchanged_loss"] = config["cross_outcome"]["aggregation"]
    audit["unchanged_selection_metric"] = config["training"]["eval_monitor_metric"]
    audit_path.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    return {"output_dir": str(output.resolve()), "status": audit["status"],
            "n_outcomes": audit["n_outcomes"], "n_endpoint_groups": audit["n_endpoint_groups"],
            "n_training_families": audit["n_training_families"]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--policy", type=Path,
                        default=Path("opera/configs/outcome_training_families_primary_v1.yaml"))
    parser.add_argument("--release-policy", type=Path,
                        default=Path("opera/configs/outcome_release_primary_v2.yaml"))
    parser.add_argument("--display-source", type=Path,
                        default=Path("opera/configs/outcome_display_groups_v1.yaml"))
    args = parser.parse_args()
    print(json.dumps(run(args.release_dir, args.output_dir, args.policy,
                         args.release_policy, args.display_source), indent=2))


if __name__ == "__main__":
    main()
