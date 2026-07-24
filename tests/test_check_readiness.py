import pytest

from opera.run.check_readiness import check_manifest_consistency, check_sweep_config

yaml = pytest.importorskip("yaml")
pd = pytest.importorskip("pandas")


def _write_manifest(tmp_path, **overrides):
    manifest = {
        "version": "1.0",
        "primary_endpoints": ["mortality_1y"],
        "primary_contrasts": [["base_pretrain", "opera"]],
        "multiplicity": {"method": "benjamini_hochberg", "alpha": 0.05},
        "min_events": {"test": 10, "train": 20},
        "seeds": [42],
    }
    manifest.update(overrides)
    path = tmp_path / "manifest.yaml"
    path.write_text(yaml.safe_dump(manifest))
    return path


def _write_sweep(tmp_path, **overrides):
    config = {
        "cohorts": {"dlbcl": {"data_dir": "data/dlbcl"}},
        "outcomes": {"mortality_1y": {"n_hours_end_include": 8760}},
        "model_variants": {
            "base_pretrain": {
                "encoder_ckpt": "checkpoints/base/best.ckpt",
                "encoder_source": "contrastive",
            },
            "opera": {
                "encoder_ckpt": "checkpoints/opera/best.ckpt",
                "encoder_source": "contrastive",
            },
        },
        "seeds": [42],
    }
    config.update(overrides)
    path = tmp_path / "sweep.yaml"
    path.write_text(yaml.safe_dump(config))
    return path


def test_check_readiness_reports_placeholder_paths_and_missing_rarity_baseline(
    tmp_path,
):
    config = {
        "rarity_mode": "real",
        "cohorts": {"dlbcl": {"data_dir": "/data/dlbcl"}},
        "outcomes": {"mortality_1y": {"n_hours_end_include": 8760}},
        "model_variants": {
            "opera": {
                "encoder_ckpt": "/ckpts/opera/best.ckpt",
                "encoder_source": "contrastive",
            }
        },
    }
    path = tmp_path / "sweep.yaml"
    path.write_text(yaml.safe_dump(config))

    issues = check_sweep_config(str(path))

    assert any("baseline_model" in issue for issue in issues)
    assert any("placeholder-looking encoder_ckpt" in issue for issue in issues)


def test_check_readiness_passes_minimal_non_placeholder_config(tmp_path):
    config = {
        "cohorts": {"dlbcl": {"data_dir": "data/dlbcl"}},
        "outcomes": {"mortality_1y": {"n_hours_end_include": 8760}},
        "model_variants": {
            "opera": {
                "encoder_ckpt": "checkpoints/opera/best.ckpt",
                "encoder_source": "contrastive",
            }
        },
    }
    path = tmp_path / "sweep.yaml"
    path.write_text(yaml.safe_dump(config))

    assert check_sweep_config(str(path)) == []


def test_check_readiness_reports_unresolved_environment_variables(tmp_path):
    config = {
        "output_dir": "${BONSAI_RESULTS_ROOT}/sweep",
        "cohorts": {"dlbcl": {"data_dir": "${BONSAI_PROCESSED_DATA}/dlbcl"}},
        "outcomes": {"mortality_1y": {"n_hours_end_include": 8760}},
        "model_variants": {
            "opera": {
                "encoder_ckpt": "${BONSAI_CHECKPOINT_ROOT}/opera/best.ckpt",
                "encoder_source": "contrastive",
            }
        },
    }
    path = tmp_path / "sweep.yaml"
    path.write_text(yaml.safe_dump(config))

    issues = check_sweep_config(str(path))

    assert any("output_dir contains an unresolved" in issue for issue in issues)
    assert any("data_dir contains an unresolved" in issue for issue in issues)
    assert any("encoder_ckpt contains an unresolved" in issue for issue in issues)


def test_check_readiness_accepts_legacy_outcome_list(tmp_path):
    data_dir = tmp_path / "data" / "dlbcl"
    outcomes_dir = data_dir / "outcomes"
    outcomes_dir.mkdir(parents=True)
    (outcomes_dir / "mortality_1y.parquet").write_text("placeholder")
    checkpoint = tmp_path / "checkpoints" / "opera" / "best.ckpt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_text("placeholder")
    config = {
        "cohorts": {"dlbcl": {"data_dir": str(data_dir)}},
        "outcomes": ["mortality_1y"],
        "model_variants": {
            "opera": {
                "encoder_ckpt": str(checkpoint),
                "encoder_source": "contrastive",
            }
        },
    }
    path = tmp_path / "sweep.yaml"
    path.write_text(yaml.safe_dump(config))

    assert check_sweep_config(str(path), require_existing_paths=True) == []


def test_check_readiness_validates_survival_training_modes(tmp_path):
    config = {
        "cohorts": {"dlbcl": {"data_dir": "data/dlbcl"}},
        "outcomes": {"treatment_failure": {"n_hours_end_include": None}},
        "model_variants": {
            "cox": {
                "encoder_ckpt": "checkpoints/opera/best.ckpt",
                "encoder_source": "contrastive",
                "training_mode": "cox",
            },
            "ipcw": {
                "encoder_ckpt": "checkpoints/opera/best.ckpt",
                "encoder_source": "contrastive",
                "training_mode": "ipcw_bce",
            },
            "bad": {
                "encoder_ckpt": "checkpoints/opera/best.ckpt",
                "encoder_source": "contrastive",
                "training_mode": "survival",
            },
        },
    }
    path = tmp_path / "sweep.yaml"
    path.write_text(yaml.safe_dump(config))

    issues = check_sweep_config(str(path))

    assert any("invalid training_mode" in issue for issue in issues)
    assert any("IPCW-BCE variants require a fixed horizon" in issue for issue in issues)


def test_check_readiness_validates_competing_outcome_paths(tmp_path):
    data_dir = tmp_path / "data" / "dlbcl"
    outcomes_dir = data_dir / "outcomes"
    outcomes_dir.mkdir(parents=True)
    (outcomes_dir / "mortality.parquet").write_text("placeholder")
    config = {
        "cohorts": {"dlbcl": {"data_dir": str(data_dir)}},
        "outcomes": {
            "mortality_1y": {
                "outcome_file": "mortality.parquet",
                "competing_outcome_file": "death.parquet",
                "n_hours_end_include": 8760,
            }
        },
        "model_variants": {
            "opera": {
                "encoder_ckpt": "checkpoints/opera/best.ckpt",
                "encoder_source": "contrastive",
            }
        },
    }
    path = tmp_path / "sweep.yaml"
    path.write_text(yaml.safe_dump(config))

    issues = check_sweep_config(str(path), require_existing_paths=True)

    assert any("competing outcome file does not exist" in issue for issue in issues)


def test_check_readiness_uses_strict_shared_config_contract(tmp_path):
    config = {
        "cohorts": {
            "dlbcl": {
                "data_dir": "data/dlbcl",
                "registry_start_dat": "2017-01-01",
            }
        },
        "outcomes": {"mortality": {}},
        "model_variants": {
            "random": {"encoder_source": "random_init"},
        },
    }
    path = tmp_path / "sweep.yaml"
    path.write_text(yaml.safe_dump(config))

    issues = check_sweep_config(str(path))

    assert any("unknown fields" in issue for issue in issues)


def test_check_readiness_validates_configured_eligibility_file(tmp_path):
    data_dir = tmp_path / "data" / "dlbcl"
    outcomes_dir = data_dir / "outcomes"
    outcomes_dir.mkdir(parents=True)
    (outcomes_dir / "mortality.parquet").write_text("placeholder")
    (outcomes_dir / "mortality_eligibility.csv").write_text(
        "subject_id,split,eligible,eligibility_reason\n1,held_out,false,\n"
    )
    config = {
        "cohorts": {"dlbcl": {"data_dir": str(data_dir)}},
        "outcomes": {
            "mortality": {
                "eligibility_file": "mortality_eligibility.csv",
            }
        },
        "model_variants": {
            "random": {"encoder_source": "random_init"},
        },
    }
    path = tmp_path / "sweep.yaml"
    path.write_text(yaml.safe_dump(config))

    issues = check_sweep_config(str(path), require_existing_paths=True)

    assert any("non-empty eligibility_reason" in issue for issue in issues)


def test_generated_fine_sweep_config_passes_readiness_check():
    """The primary fine sweep must parse without validation issues (no path check)."""
    from opera.run.check_readiness import check_sweep_config
    from pathlib import Path

    config = str(
        Path(__file__).parents[1] / "opera" / "configs" / "generated" / "fine_cox.yaml"
    )
    issues = check_sweep_config(config, require_existing_paths=False)
    # Filter out expected unresolved-env-var warnings (these are acceptable in CI)
    hard_issues = [i for i in issues if "unresolved environment variable" not in i]
    assert not hard_issues, f"Leukemia sweep config has hard issues: {hard_issues}"


def test_analysis_manifest_validates():
    """The checked-in analysis manifest must pass validation."""
    from opera.config_contracts import validate_analysis_manifest
    from pathlib import Path

    manifest_path = Path(__file__).parents[1] / "opera" / "analysis_manifest.yaml"
    assert manifest_path.exists()
    result = validate_analysis_manifest(manifest_path)
    assert "primary_endpoints" in result
    assert "seeds" in result


def test_manifest_consistency_passes_for_matching_config(tmp_path):
    manifest_path = _write_manifest(tmp_path)
    sweep_path = _write_sweep(tmp_path)

    issues = check_manifest_consistency(str(manifest_path), str(sweep_path))

    assert issues == []


def test_manifest_consistency_flags_missing_primary_endpoint(tmp_path):
    manifest_path = _write_manifest(
        tmp_path,
        primary_endpoints=["mortality_1y", "nonexistent_outcome"],
    )
    sweep_path = _write_sweep(
        tmp_path,
        outcomes={"mortality_1y": {"n_hours_end_include": 8760}},
    )

    issues = check_manifest_consistency(str(manifest_path), str(sweep_path))

    assert issues
    assert any("nonexistent_outcome" in issue for issue in issues)


def test_manifest_consistency_flags_missing_contrast_model(tmp_path):
    manifest_path = _write_manifest(
        tmp_path,
        primary_contrasts=[["base_pretrain", "opera"]],
    )
    sweep_path = _write_sweep(
        tmp_path,
        model_variants={
            "base_pretrain": {
                "encoder_ckpt": "checkpoints/base/best.ckpt",
                "encoder_source": "contrastive",
            },
        },
    )

    issues = check_manifest_consistency(str(manifest_path), str(sweep_path))

    assert issues
    assert any("opera" in issue for issue in issues)


def test_manifest_consistency_flags_seed_mismatch(tmp_path):
    manifest_path = _write_manifest(tmp_path, seeds=[42, 43, 44])
    sweep_path = _write_sweep(tmp_path, seeds=[42])

    issues = check_manifest_consistency(str(manifest_path), str(sweep_path))

    assert issues
    assert any("seed" in issue.lower() for issue in issues)


def _write_shared_data_readiness_config(
    tmp_path,
    monkeypatch,
    *,
    membership: "pd.DataFrame",
    outcomes: "pd.DataFrame",
    membership_suffix: str = ".parquet",
):
    """Build a minimal production-style shared-data config and files."""
    data_dir = tmp_path / "processed"
    outcomes_dir = tmp_path / "outcomes"
    checkpoints_dir = tmp_path / "checkpoints"
    data_dir.mkdir()
    outcomes_dir.mkdir()
    checkpoints_dir.mkdir()
    membership_path = tmp_path / f"cohort_membership{membership_suffix}"
    if membership_suffix == ".csv":
        membership.to_csv(membership_path, index=False)
    else:
        membership.to_parquet(membership_path, index=False)
    outcome_path = outcomes_dir / "endpoint.parquet"
    outcomes.to_parquet(outcome_path, index=False)
    checkpoint = checkpoints_dir / "encoder.ckpt"
    checkpoint.write_text("placeholder")

    monkeypatch.setenv("READINESS_DATA", str(data_dir))
    monkeypatch.setenv("READINESS_MEMBERSHIP", str(membership_path))
    monkeypatch.setenv("READINESS_OUTCOMES", str(outcomes_dir))
    monkeypatch.setenv("READINESS_CHECKPOINTS", str(checkpoints_dir))
    monkeypatch.setenv("READINESS_RESULTS", str(tmp_path / "results"))

    config = {
        "output_dir": "${READINESS_RESULTS}/fine",
        "cohorts": {
            "RARE": {
                "data_dir": "${READINESS_DATA}",
                "population_file": "${READINESS_MEMBERSHIP}",
                "cohort_fine_col": "cohort_fine",
                "cohort_fine_value": "RARE",
            }
        },
        "outcomes": {
            "endpoint": {
                "outcome_file": "${READINESS_OUTCOMES}/endpoint.parquet",
                "n_hours_end_include": 720,
            }
        },
        "model_variants": {
            "opera": {
                "encoder_ckpt": "${READINESS_CHECKPOINTS}/encoder.ckpt",
                "encoder_source": "contrastive",
            }
        },
    }
    config_path = tmp_path / "shared_sweep.yaml"
    config_path.write_text(yaml.safe_dump(config))
    return config_path


def test_check_readiness_supports_shared_global_outcomes_and_csv_membership(
    tmp_path,
    monkeypatch,
):
    """Production shared-data paths expand and count only the selected cohort."""
    membership = pd.DataFrame(
        {
            "subject_id": list(range(1, 11)),
            "cohort_fine": ["RARE"] * 5 + ["OTHER"] * 5,
        }
    )
    outcomes = pd.DataFrame(
        {
            "subject_id": list(range(1, 11)),
            "split": ["held_out"] * 10,
            "event": [1] * 5 + [0] * 5,
        }
    )
    config_path = _write_shared_data_readiness_config(
        tmp_path,
        monkeypatch,
        membership=membership,
        outcomes=outcomes,
        membership_suffix=".csv",
    )

    assert check_sweep_config(str(config_path), require_existing_paths=True) == []


def test_check_readiness_event_counts_are_filtered_to_fine_membership(
    tmp_path,
    monkeypatch,
):
    """Whole-population events must not conceal a zero-event fine cohort."""
    membership = pd.DataFrame(
        {
            "subject_id": list(range(1, 11)),
            "cohort_fine": ["RARE"] * 2 + ["OTHER"] * 8,
        }
    )
    outcomes = pd.DataFrame(
        {
            "subject_id": list(range(1, 11)),
            "split": ["held_out"] * 10,
            "event": [0, 0] + [1] * 8,
        }
    )
    config_path = _write_shared_data_readiness_config(
        tmp_path,
        monkeypatch,
        membership=membership,
        outcomes=outcomes,
    )

    issues = check_sweep_config(str(config_path), require_existing_paths=True)

    assert any(
        "has only 0 held-out events after membership/eligibility filtering" in issue
        for issue in issues
    )


def test_check_readiness_rejects_nonunique_shared_membership(tmp_path, monkeypatch):
    membership = pd.DataFrame(
        {
            "subject_id": [1, 1, 2, 3, 4, 5],
            "cohort_fine": ["RARE"] * 6,
        }
    )
    outcomes = pd.DataFrame(
        {
            "subject_id": [1, 2, 3, 4, 5],
            "split": ["held_out"] * 5,
            "event": [1] * 5,
        }
    )
    config_path = _write_shared_data_readiness_config(
        tmp_path,
        monkeypatch,
        membership=membership,
        outcomes=outcomes,
    )

    issues = check_sweep_config(str(config_path), require_existing_paths=True)

    assert any("duplicate subject_id rows" in issue for issue in issues)
