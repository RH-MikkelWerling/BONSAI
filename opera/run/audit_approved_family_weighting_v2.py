"""Aggregate-only audit of the inherited clinical families versus loss groups."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml


def audit(release: Path) -> dict:
    registry = yaml.safe_load((release / "experiment_registry_primary_v2.yaml").read_text(encoding="utf-8"))
    config = yaml.safe_load((release / "generated_configs" / "direct_cr_bce_lowrank_endpoint_groups_v1.yaml").read_text(encoding="utf-8"))
    names = registry["outcomes"]
    families = registry["outcome_families"]
    groups = config["cross_outcome"]["endpoint_groups"]
    family_members = [name for family in families.values() for name in family]
    owner = {name: family for family, members in families.items() for name in members}
    if len(family_members) != len(names) or set(owner) != set(names):
        raise ValueError("Clinical family inventory does not cover the approved roster exactly")
    members = [name for group in groups.values() for name in group]
    if len(members) != len(names) or set(members) != set(names):
        raise ValueError("Endpoint loss groups do not cover the approved roster exactly")
    cross_family = {group: sorted({owner[name] for name in group_members})
                    for group, group_members in groups.items()
                    if len({owner[name] for name in group_members}) != 1}
    if cross_family:
        raise ValueError(f"Endpoint groups cross clinical families: {cross_family}")
    family_groups = {family: [group for group, group_members in groups.items()
                              if owner[group_members[0]] == family]
                     for family in families}
    total_groups = len(groups)
    return {"status": "family_weighting_review_only", "outcomes": len(names),
            "endpoint_groups": total_groups, "families": len(families),
            "current_loss": "equal_weight_per_endpoint_group",
            "current_checkpoint_selection": "tuning_endpoint_group_macro_auc",
            "equal_family_counterfactual_not_approved": True,
            "family_inventory": {
                family: {"n_outcomes": len(families[family]),
                         "n_endpoint_groups": len(family_groups[family]),
                         "current_loss_share": len(family_groups[family]) / total_groups,
                         "equal_family_loss_share_if_adopted": 1 / len(families),
                         "outcomes": families[family],
                         "endpoint_groups": family_groups[family]}
                for family in families}}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(audit(args.release_dir), indent=2))


if __name__ == "__main__":
    main()
