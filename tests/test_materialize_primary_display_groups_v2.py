from pathlib import Path

import yaml

from opera.run.build_approved_outcome_release_v2 import _registry
from opera.run.materialize_primary_display_groups_v2 import build_display_groups


def test_display_groups_cover_renamed_and_added_release_outcomes_once() -> None:
    source = yaml.safe_load(Path("opera/configs/experiment_registry.yaml").read_text(encoding="utf-8"))
    policy = yaml.safe_load(Path("opera/configs/outcome_release_primary_v2.yaml").read_text(encoding="utf-8"))
    old = yaml.safe_load(Path("opera/configs/outcome_display_groups_v1.yaml").read_text(encoding="utf-8"))
    selected = set(source["outcomes"]) - set(policy["exclude_held"])
    approved = _registry(source, selected, policy["rename"],
                         policy["add_from_lab_candidates"] + policy["add_from_potassium_candidates"])
    groups = build_display_groups(approved, policy, old)
    assigned = [name for names in groups.values() for name in names]
    assert len(assigned) == len(set(assigned)) == len(approved["outcomes"])
    assert "creatinine_rise_gt1p5x_preindex_baseline" in groups["Kidney injury and renal support"]
    assert "potassium_below_3p0_mmol_l" in groups["Sodium and potassium"]
    assert "alp_elevation_legacy_tier2plus" in groups["Hepatic enzymes"]
    assert "infection_diagnosis_or_antimicrobial_use_composite" in groups[
        "Antimicrobial and infection proxies"]
