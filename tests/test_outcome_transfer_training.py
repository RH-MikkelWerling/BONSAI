"""Launch-time contracts for the narrow OPERA outcome-transfer experiment."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml
from omegaconf import OmegaConf

from opera.functional.outcome_transfer import build_transfer_config, resolve_transfer_manifest
from opera.run.contrastive_multicohort import _validate_transfer_config
from opera.run.generate_outcome_transfer_configs import generate_outcome_transfer_configs
from opera.run.outcome_transfer_train import (
    OutcomeTransferLaunchError,
    _validate_preflight_report,
    check_reusable_full_checkpoint,
    launch_outcome_transfer_conditions,
)


MANIFEST = Path("opera/configs/manifests/outcome_transfer.yaml")
BASE_CONFIG = Path("opera/configs/generated/joint_opera_full_panel.yaml")


def test_ablation_dry_run_creates_one_slot_without_retraining_full(
    tmp_path: Path,
) -> None:
    config_dir = tmp_path / "configs"
    generate_outcome_transfer_configs(output_dir=config_dir)

    rows = launch_outcome_transfer_conditions(
        config_dir=config_dir,
        output_dir=tmp_path / "outputs",
        conditions=["opera_no_g3"],
        seeds=[42],
        execute=False,
    )

    assert len(rows) == 1
    assert rows[0]["condition"] == "opera_no_g3"
    assert rows[0]["status"] == "dry_run"
    assert (tmp_path / "outputs" / "transfer_training_status.csv").exists()
    command = json.loads(rows[0]["command"])
    assert "seed=42" in command
    assert all("outcome=" not in element for element in command)


def test_launcher_rejects_stale_base_or_split_provenance_in_generated_config(
    tmp_path: Path,
) -> None:
    config_dir = tmp_path / "configs"
    generate_outcome_transfer_configs(output_dir=config_dir)
    config_path = config_dir / "opera_no_g3.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    config["base_contrastive_config_hash"] = "stale-base-config"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    with pytest.raises(OutcomeTransferLaunchError, match="base_contrastive_config_hash"):
        launch_outcome_transfer_conditions(
            config_dir=config_dir,
            output_dir=tmp_path / "outputs",
            conditions=["opera_no_g3"],
            seeds=[42],
        )

    generate_outcome_transfer_configs(output_dir=config_dir)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config["transfer_checkpoint_metadata"]["split_contract_hash"] = "stale-split"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    with pytest.raises(OutcomeTransferLaunchError, match="split_contract_hash"):
        launch_outcome_transfer_conditions(
            config_dir=config_dir,
            output_dir=tmp_path / "outputs",
            conditions=["opera_no_g3"],
            seeds=[42],
        )


def test_execute_requires_a_matching_label_only_preflight_before_any_subprocess(
    tmp_path: Path,
) -> None:
    config_dir = tmp_path / "configs"
    generate_outcome_transfer_configs(output_dir=config_dir)
    with pytest.raises(OutcomeTransferLaunchError, match="requires --preflight-report"):
        launch_outcome_transfer_conditions(
            config_dir=config_dir,
            output_dir=tmp_path / "outputs",
            conditions=["opera_no_g3"],
            seeds=[42],
            execute=True,
        )


def test_preflight_gate_requires_every_current_transfer_target_cell(tmp_path: Path) -> None:
    plan = resolve_transfer_manifest(MANIFEST)
    expected_rows = [
        {
            "transfer_condition": condition_name,
            "target_outcome": target,
            "primary_horizon_days": condition["primary_horizon_days"],
            "evaluation_level": "all_hematology",
            "evaluation_group": "all_hematology",
        }
        for condition_name, condition in plan["conditions"].items()
        if condition_name != "opera_full"
        for target in condition["evaluation_outcomes"]
    ]
    report_path = tmp_path / "synthetic_support.json"
    report_path.write_text(
        json.dumps(
            {
                "metadata": {
                    "label_only": True,
                    "registry_hash": plan["registry_hash"],
                    "manifest_hash": plan["manifest_hash"],
                    "base_contrastive_config_hash": plan[
                        "base_contrastive_config_hash"
                    ],
                    "split_contract_hash": plan["split_contract_hash"],
                    "evaluation_target_union": plan["evaluation_target_union"],
                },
                "rows": expected_rows,
            }
        ),
        encoding="utf-8",
    )
    assert _validate_preflight_report(report_path, plan) == report_path

    report_path.write_text(
        json.dumps(
            {
                "metadata": {
                    "label_only": True,
                    "registry_hash": plan["registry_hash"],
                    "manifest_hash": plan["manifest_hash"],
                    "base_contrastive_config_hash": plan[
                        "base_contrastive_config_hash"
                    ],
                    "split_contract_hash": plan["split_contract_hash"],
                    "evaluation_target_union": plan["evaluation_target_union"],
                },
                "rows": expected_rows[:-1],
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(OutcomeTransferLaunchError, match="Preflight report is incomplete"):
        _validate_preflight_report(report_path, plan)


def test_full_checkpoint_reuse_requires_exact_canonical_outcome_panel(
    tmp_path: Path,
) -> None:
    plan = resolve_transfer_manifest(MANIFEST)
    checkpoint = tmp_path / "seed_42" / "best.ckpt"
    checkpoint.parent.mkdir()
    checkpoint.write_bytes(b"metadata sidecar makes checkpoint loading unnecessary")
    (checkpoint.parent / "checkpoint_metadata.json").write_text(
        json.dumps(
            {
                "checkpoint_metadata": {
                    "training_stage": "opera_contrastive_adaptation",
                    "source_checkpoint": "/synthetic/dapt.ckpt",
                    "outcome_set": plan["conditions"]["opera_full"][
                        "training_outcomes"
                    ],
                }
            }
        ),
        encoding="utf-8",
    )

    compatible, detail, _ = check_reusable_full_checkpoint(
        checkpoint,
        plan,
        seed=42,
        # This unit test exercises the deliberate low-level opt-in for a
        # legacy sidecar.  The public launcher requires a {seed}-bound
        # checkpoint template for the same situation.
        allow_legacy_without_seed=True,
    )
    assert compatible
    assert detail == "compatible"

    bad_sidecar = {
        "checkpoint_metadata": {
            "training_stage": "opera_contrastive_adaptation",
            "outcome_set": plan["conditions"]["opera_full"]["training_outcomes"][:-1],
        }
    }
    (checkpoint.parent / "checkpoint_metadata.json").write_text(
        json.dumps(bad_sidecar), encoding="utf-8"
    )
    compatible, detail, _ = check_reusable_full_checkpoint(checkpoint, plan, seed=42)
    assert not compatible
    assert detail == "outcome_panel_mismatch"


def test_tagged_full_checkpoint_requires_base_and_split_hashes(tmp_path: Path) -> None:
    plan = resolve_transfer_manifest(MANIFEST)
    checkpoint = tmp_path / "seed_42" / "best.ckpt"
    checkpoint.parent.mkdir()
    checkpoint.write_bytes(b"sidecar-backed synthetic checkpoint")
    metadata = {
        "training_stage": "opera_contrastive_adaptation",
        "condition": "opera_full",
        "seed": 42,
        "source_dapt_checkpoint": "/synthetic/dapt.ckpt",
        "outcome_set": plan["conditions"]["opera_full"]["training_outcomes"],
        "registry_hash": plan["registry_hash"],
        "manifest_hash": plan["manifest_hash"],
        "base_contrastive_config_hash": plan["base_contrastive_config_hash"],
        "split_contract_hash": plan["split_contract_hash"],
    }
    sidecar = checkpoint.parent / "checkpoint_metadata.json"
    sidecar.write_text(json.dumps({"checkpoint_metadata": metadata}), encoding="utf-8")
    compatible, detail, _ = check_reusable_full_checkpoint(checkpoint, plan, seed=42)
    assert compatible
    assert detail == "compatible"

    metadata["split_contract_hash"] = "stale-split"
    sidecar.write_text(json.dumps({"checkpoint_metadata": metadata}), encoding="utf-8")
    compatible, detail, _ = check_reusable_full_checkpoint(checkpoint, plan, seed=42)
    assert not compatible
    assert detail == "split_contract_hash_mismatch_or_missing"


def test_launcher_does_not_reuse_an_unseeded_legacy_full_checkpoint_across_seeds(
    tmp_path: Path,
) -> None:
    config_dir = tmp_path / "configs"
    generate_outcome_transfer_configs(output_dir=config_dir)
    plan = resolve_transfer_manifest(MANIFEST)
    checkpoint = tmp_path / "legacy_full" / "best.ckpt"
    checkpoint.parent.mkdir()
    checkpoint.write_bytes(b"sidecar-backed synthetic checkpoint")
    (checkpoint.parent / "checkpoint_metadata.json").write_text(
        json.dumps(
            {
                "checkpoint_metadata": {
                    "training_stage": "opera_contrastive_adaptation",
                    "source_checkpoint": "/synthetic/dapt.ckpt",
                    "outcome_set": plan["conditions"]["opera_full"][
                        "training_outcomes"
                    ],
                }
            }
        ),
        encoding="utf-8",
    )

    rows = launch_outcome_transfer_conditions(
        config_dir=config_dir,
        output_dir=tmp_path / "outputs",
        conditions=["opera_full"],
        seeds=[42, 43],
        full_checkpoint_template=str(checkpoint),
    )

    assert [row["status"] for row in rows] == [
        "incompatible_full_checkpoint",
        "incompatible_full_checkpoint",
    ]
    assert {row["detail"] for row in rows} == {"seed_provenance_required"}


def test_launcher_discovers_and_validates_nested_csvlogger_checkpoint(
    tmp_path: Path,
) -> None:
    config_dir = tmp_path / "configs"
    output_dir = tmp_path / "outputs"
    generate_outcome_transfer_configs(output_dir=config_dir)
    plan = resolve_transfer_manifest(MANIFEST)
    condition = "opera_no_g3"
    expected = plan["conditions"][condition]
    checkpoint = (
        output_dir
        / "runs"
        / condition
        / "seed_42"
        / "contrastive_multicohort_runs"
        / "version_0"
        / "best.ckpt"
    )
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"sidecar-backed synthetic checkpoint")
    (checkpoint.parent / "checkpoint_metadata.json").write_text(
        json.dumps(
            {
                "checkpoint_metadata": {
                    "training_stage": "opera_contrastive_adaptation",
                    "condition": condition,
                    "transfer_level": expected["transfer_level"],
                    "seed": 42,
                    "registry_hash": plan["registry_hash"],
                    "manifest_hash": plan["manifest_hash"],
                    "base_contrastive_config_hash": plan[
                        "base_contrastive_config_hash"
                    ],
                    "included_outcomes": expected["training_outcomes"],
                    "excluded_outcomes": expected["training_excluded_outcomes"],
                    "evaluation_outcomes": expected["evaluation_outcomes"],
                    "related_retained_outcomes": expected[
                        "related_retained_outcomes"
                    ],
                    "direct_dependencies_excluded": expected[
                        "direct_dependencies_excluded"
                    ],
                    "selection_outcomes": expected["training_outcomes"],
                    "split_contract": plan["split_contract"],
                    "split_contract_hash": plan["split_contract_hash"],
                    "source_dapt_checkpoint": "/synthetic/dapt.ckpt",
                    "outcome_set": sorted(expected["training_outcomes"]),
                }
            }
        ),
        encoding="utf-8",
    )

    rows = launch_outcome_transfer_conditions(
        config_dir=config_dir,
        output_dir=output_dir,
        conditions=[condition],
        seeds=[42],
        execute=False,
    )
    assert rows[0]["status"] == "already_present"
    assert rows[0]["checkpoint_path"] == str(checkpoint)


def test_runner_rejects_held_out_labels_before_data_module_or_checkpoint_selection() -> None:
    plan = resolve_transfer_manifest(MANIFEST)
    base = yaml.safe_load(BASE_CONFIG.read_text(encoding="utf-8"))
    config = build_transfer_config(plan, "opera_no_transfusion_signal", base)
    config["dapt_ckpt"] = "/synthetic/dapt.ckpt"
    cfg = OmegaConf.create(config)
    # The production runner sorts internal outcome heads, whereas YAML keeps
    # registry order.  Both must still be the same exact outcome panel.
    names = sorted(config["outcomes"])
    metadata = _validate_transfer_config(cfg, names)
    assert metadata is not None
    assert set(metadata["selection_outcomes"]) == set(names)

    held_out = config["training_excluded_outcomes"][0]
    invalid_names = [*names, held_out]
    with pytest.raises(ValueError, match="Held-out transfer labels"):
        _validate_transfer_config(cfg, invalid_names)

    config["transfer_checkpoint_metadata"]["base_contrastive_config_hash"] = "stale"
    with pytest.raises(ValueError, match="base_contrastive_config_hash"):
        _validate_transfer_config(OmegaConf.create(config), names)
