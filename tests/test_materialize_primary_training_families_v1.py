from pathlib import Path

import pytest
import yaml

from opera.functional.endpoint_groups import endpoint_groups_v1
from opera.run.build_approved_outcome_release_v2 import _registry
from opera.run.materialize_primary_display_groups_v2 import build_display_groups
from opera.run.materialize_primary_training_families_v1 import (
    build_training_families, load_release_display_groups,
)


REMOVED_FROM_APPROVED_ROSTER = {
    "AST_increased_g2plus", "AST_increased_g3plus", "dic_like_coagulopathy",
    "herpes_zoster", "hypermagnesemia_g3plus", "hypofibrinogenemia_g2plus",
    "hypofibrinogenemia_g3plus", "hypoglycemia_g2plus", "hypoglycemia_g3plus",
    "hypokalemia_g2plus", "hypokalemia_g3plus", "hypomagnesemia_g3plus",
    "hys_law_dili", "invasive_fungal", "leukocytosis_g2plus",
    "lymphopenia_g2plus", "lymphopenia_g3plus", "MACE_composite",
    "myocardial_infarction", "tumor_lysis_syndrome",
}


def _approved_inputs():
    source = yaml.safe_load(Path("opera/configs/experiment_registry.yaml").read_text(encoding="utf-8"))
    release_policy = yaml.safe_load(Path("opera/configs/outcome_release_primary_v2.yaml").read_text(encoding="utf-8"))
    display_source = yaml.safe_load(Path("opera/configs/outcome_display_groups_v1.yaml").read_text(encoding="utf-8"))
    family_policy = yaml.safe_load(Path("opera/configs/outcome_training_families_primary_v1.yaml").read_text(encoding="utf-8"))
    selected = set(source["outcomes"]) - REMOVED_FROM_APPROVED_ROSTER
    registry = _registry(source, selected, release_policy["rename"],
                         release_policy["add_from_lab_candidates"]
                         + release_policy["add_from_potassium_candidates"])
    display = build_display_groups(registry, release_policy, display_source)
    return registry, display, endpoint_groups_v1(registry["outcomes"]), family_policy


def test_training_families_cover_69_outcomes_and_52_groups_without_split() -> None:
    registry, display, groups, policy = _approved_inputs()
    families, audit = build_training_families(registry, display, groups, policy)
    assert audit["n_outcomes"] == 69
    assert audit["n_endpoint_groups"] == 52
    assert audit["n_training_families"] == 12
    assert sorted(len(row["endpoint_groups"]) for row in audit["families"].values()) == [
        3, 3, 4, 4, 4, 4, 4, 4, 5, 5, 6, 6
    ]
    assert "bone_fracture" in families["Acute events and care use"]
    assert "infection_diagnosis_or_antimicrobial_use_composite" in families[
        "Infection and antimicrobial use"]


def test_training_families_reject_split_endpoint_group() -> None:
    registry, display, groups, policy = _approved_inputs()
    display["Bilirubin and liver injury"].remove("bilirubin_increased_g3plus")
    display["Sodium and potassium"].append("bilirubin_increased_g3plus")
    with pytest.raises(ValueError, match="crosses training families"):
        build_training_families(registry, display, groups, policy)


def test_training_families_reject_missing_display_group() -> None:
    registry, display, groups, policy = _approved_inputs()
    del policy["Cardiovascular and thrombotic events"]
    with pytest.raises(ValueError, match="cover every display group"):
        build_training_families(registry, display, groups, policy)


def test_missing_release_display_is_derived_in_memory() -> None:
    registry, expected, _, _ = _approved_inputs()
    missing_release = Path("__missing_approved_release_for_test__")
    assert not missing_release.exists()
    actual, provenance = load_release_display_groups(
        missing_release, registry, Path("opera/configs/experiment_registry.yaml"),
        Path("opera/configs/outcome_release_primary_v2.yaml"),
        Path("opera/configs/outcome_display_groups_v1.yaml"),
    )
    assert actual == expected
    assert provenance["mode"] == "derived_from_checked_in_taxonomy"
    assert not missing_release.exists()
