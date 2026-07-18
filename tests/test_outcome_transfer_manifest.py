"""Focused contract tests for the narrow OPERA outcome-transfer experiment."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import yaml

from opera.functional.outcome_transfer import (
    EXPECTED_CONDITIONS,
    build_transfer_config,
    file_hash,
    resolve_transfer_manifest,
)
from opera.run.generate_outcome_transfer_configs import (
    generate_outcome_transfer_configs,
)
from opera.run.generate_sweep_configs import load_registry


REGISTRY_PATH = Path("opera/configs/experiment_registry.yaml")
MANIFEST_PATH = Path("opera/configs/manifests/outcome_transfer.yaml")
BASE_CONFIG_PATH = Path("opera/configs/generated/joint_opera_full_panel.yaml")


def _plan() -> dict:
    return resolve_transfer_manifest(MANIFEST_PATH)


def test_full_opera_uses_every_canonical_production_outcome() -> None:
    registry = load_registry(REGISTRY_PATH)
    plan = _plan()

    full = plan["conditions"]["opera_full"]
    assert full["training_outcomes"] == registry["outcomes"]
    assert full["training_excluded_outcomes"] == []
    assert full["transfer_level"] == "direct_supervision"


def test_resolved_plan_hashes_canonical_base_and_split_contract() -> None:
    plan = _plan()

    assert plan["base_contrastive_config_hash"] == file_hash(BASE_CONFIG_PATH)
    assert plan["split_contract_hash"] == file_hash(plan["split_contract"])


def test_no_g3_excludes_every_and_only_g3plus_and_derives_real_pairs() -> None:
    registry = load_registry(REGISTRY_PATH)
    plan = _plan()
    condition = plan["conditions"]["opera_no_g3"]

    expected_g3 = [name for name in registry["outcomes"] if name.endswith("_g3plus")]
    assert condition["training_excluded_outcomes"] == expected_g3
    assert not set(condition["training_outcomes"]) & set(expected_g3)
    pairs = condition["matched_g2_g3_pairs"]
    assert pairs
    assert {pair["target_outcome"] for pair in pairs} == set(
        condition["primary_evaluation_outcomes"]
    )
    for pair in pairs:
        assert pair["target_outcome"].endswith("_g3plus")
        assert pair["lower_grade_outcome"] == pair["target_outcome"].replace(
            "_g3plus", "_g2plus"
        )
        assert pair["lower_grade_outcome"] in registry["outcomes"]
        assert pair["lower_grade_outcome"] in condition["training_outcomes"]


def test_transfusion_signal_holdout_removes_direct_composite_components_only() -> None:
    plan = _plan()
    condition = plan["conditions"]["opera_no_transfusion_signal"]

    assert set(condition["training_excluded_outcomes"]) == {
        "any_transfusion",
        "RBC_transfusion",
        "platelet_transfusion",
    }
    assert condition["evaluation_outcomes"] == ["any_transfusion"]
    assert {"anemia_g2plus", "anemia_g3plus", "thrombocytopenia_g2plus", "thrombocytopenia_g3plus"} <= set(
        condition["related_retained_outcomes"]
    )
    assert not set(condition["training_excluded_outcomes"]) & set(
        condition["related_retained_outcomes"]
    )


def test_hospitalisation_dependencies_are_verified_and_keep_independent_proxies() -> None:
    plan = _plan()
    condition = plan["conditions"]["opera_no_hospitalisation_signal"]

    assert condition["dependency_resolution_status"] == "verified_archival_source_evidence"
    assert not condition["launch_blocked"]
    assert condition["direct_dependencies_excluded"] == ["hospitalisation"]
    assert condition["training_excluded_outcomes"] == ["hospitalisation"]
    assert {"emergency_stay", "respiratory_support"} <= set(
        condition["related_retained_outcomes"]
    )
    assert {"emergency_stay", "respiratory_support"} <= set(
        condition["training_outcomes"]
    )
    provenance = condition["dependency_provenance"]
    assert provenance["source"].endswith("calculate_adverse_events_update.R")
    assert len(provenance["source_sha256"]) == 64
    assert provenance["evidence_file"].endswith(
        "outcome_transfer_hospitalisation_evidence.yaml"
    )
    assert len(provenance["evidence_sha256"]) == 64
    assert provenance["independent_related_outcomes"]["emergency_stay"]
    assert provenance["independent_related_outcomes"]["respiratory_support"]


def test_family_holdouts_are_exactly_the_canonical_family_members() -> None:
    registry = load_registry(REGISTRY_PATH)
    plan = _plan()
    expected = {
        "opera_no_infection_family": "Infection",
        "opera_no_renal_family": "Renal toxicity",
        "opera_no_cardiovascular_family": "Cardiovascular & thrombotic",
    }

    for condition_name, family in expected.items():
        condition = plan["conditions"][condition_name]
        assert condition["training_excluded_outcomes"] == registry["outcome_families"][family]
        assert condition["evaluation_outcomes"] == registry["outcome_families"][family]
        assert not set(condition["training_outcomes"]) & set(
            registry["outcome_families"][family]
        )
        assert condition["related_retained_outcomes"] == condition["training_outcomes"]


def test_generated_configs_change_only_outcome_panel_and_add_transfer_metadata(
    tmp_path: Path,
) -> None:
    paths = generate_outcome_transfer_configs(output_dir=tmp_path)
    assert len(paths) == 9
    assert {path.name for path in paths} == {
        *(f"{condition}.yaml" for condition in EXPECTED_CONDITIONS),
        "resolved_transfer_plan.json",
        "resolved_transfer_plan.csv",
    }

    base = yaml.safe_load(BASE_CONFIG_PATH.read_text(encoding="utf-8"))
    plan = _plan()
    for condition in EXPECTED_CONDITIONS:
        generated = yaml.safe_load((tmp_path / f"{condition}.yaml").read_text())
        for key, value in base.items():
            if key != "outcomes":
                assert generated[key] == value
        assert list(generated["outcomes"]) == plan["conditions"][condition][
            "training_outcomes"
        ]
        for key in (
            "transfer_analysis",
            "transfer_condition",
            "transfer_level",
            "training_outcomes",
            "training_excluded_outcomes",
            "evaluation_outcomes",
            "related_retained_outcomes",
            "direct_dependencies_excluded",
            "registry_hash",
            "manifest_hash",
            "base_contrastive_config_hash",
            "split_contract_hash",
        ):
            assert key in generated
        checkpoint_metadata = generated["transfer_checkpoint_metadata"]
        assert checkpoint_metadata["base_contrastive_config_hash"] == plan[
            "base_contrastive_config_hash"
        ]
        assert checkpoint_metadata["split_contract_hash"] == plan[
            "split_contract_hash"
        ]


def test_held_out_labels_cannot_reach_training_or_checkpoint_selection() -> None:
    plan = _plan()
    base = yaml.safe_load(BASE_CONFIG_PATH.read_text(encoding="utf-8"))
    for name in EXPECTED_CONDITIONS:
        config = build_transfer_config(plan, name, base)
        held_out = set(config["training_excluded_outcomes"])
        assert not held_out & set(config["outcomes"])
        assert not held_out & set(config["selection_outcomes"])
        assert config["selection_outcomes"] == config["training_outcomes"]
        assert config["transfer_checkpoint_metadata"]["selection_outcomes"] == config[
            "training_outcomes"
        ]


def test_checkpoint_slots_are_fixed_and_dapt_is_not_a_training_slot() -> None:
    plan = _plan()
    assert plan["checkpoint_slot_count"] == 21
    assert len(plan["checkpoint_slots"]) == 21
    assert {slot["condition"] for slot in plan["checkpoint_slots"]} == set(
        EXPECTED_CONDITIONS
    )
    assert {slot["seed"] for slot in plan["checkpoint_slots"]} == {42, 43, 44}
    assert all(slot["condition"] != "dapt" for slot in plan["checkpoint_slots"])
    assert plan["new_contrastive_runs_if_full_reused"] == 18


def test_manifest_resolution_is_deterministic_and_requires_verified_extra_dependencies(
    tmp_path: Path,
) -> None:
    original = yaml.safe_load(MANIFEST_PATH.read_text(encoding="utf-8"))
    broken = deepcopy(original)
    hospital = broken["conditions"]["opera_no_hospitalisation_signal"]["exclude"]
    hospital["direct_dependencies"] = ["emergency_stay"]
    hospital["dependency_provenance"] = None
    manifest_path = tmp_path / "broken.yaml"
    manifest_path.write_text(yaml.safe_dump(broken, sort_keys=False), encoding="utf-8")

    try:
        resolve_transfer_manifest(manifest_path)
    except ValueError as error:
        assert "without verified dependency_provenance" in str(error)
    else:  # pragma: no cover - defensive assertion for an invalid manifest
        raise AssertionError("Unverified direct hospitalisation dependency was accepted.")


def test_hospitalisation_archival_evidence_hash_is_checked_at_resolution(
    tmp_path: Path,
) -> None:
    broken = yaml.safe_load(MANIFEST_PATH.read_text(encoding="utf-8"))
    provenance = broken["conditions"]["opera_no_hospitalisation_signal"][
        "exclude"
    ]["dependency_provenance"]
    provenance["evidence_sha256"] = "0" * 64
    manifest_path = tmp_path / "tampered_evidence.yaml"
    manifest_path.write_text(yaml.safe_dump(broken, sort_keys=False), encoding="utf-8")

    condition = resolve_transfer_manifest(manifest_path)["conditions"][
        "opera_no_hospitalisation_signal"
    ]
    assert condition["launch_blocked"]
    assert condition["dependency_resolution_status"] == "unresolved_source_not_found"
    assert "evidence_sha256" in condition["launch_blocked_reason"]
