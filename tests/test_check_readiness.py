import pytest

from opera.run.check_readiness import check_sweep_config

yaml = pytest.importorskip("yaml")


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


def test_check_readiness_accepts_legacy_outcome_list(tmp_path):
    data_dir = tmp_path / "data" / "dlbcl"
    outcomes_dir = data_dir / "outcomes"
    outcomes_dir.mkdir(parents=True)
    (outcomes_dir / "mortality_1y.parquet").write_text("placeholder")
    config = {
        "cohorts": {"dlbcl": {"data_dir": str(data_dir)}},
        "outcomes": ["mortality_1y"],
        "model_variants": {
            "opera": {
                "encoder_ckpt": "checkpoints/opera/best.ckpt",
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
