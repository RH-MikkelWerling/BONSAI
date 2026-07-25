from pathlib import Path

import pytest

from opera.config_contracts import (
    ConfigValidationError,
    SweepConfig,
    load_sweep_config,
    variant_applies_to_outcome,
)
from opera.evaluation.tasks import normalize_outcome_config, outcome_file_path
from opera.functional.outcomes import resolve_registry_start_date


def test_normalize_outcome_config_supports_shared_event_file():
    outcomes = normalize_outcome_config(
        {
            "mortality_1y": {
                "outcome_file": "mortality.parquet",
                "n_hours_end_include": 8760,
            },
            "mortality_2y": {
                "outcome_file": "mortality.parquet",
                "n_hours_end_include": 17520,
            },
        }
    )

    assert outcomes["mortality_1y"]["outcome_file"] == "mortality.parquet"
    assert outcomes["mortality_2y"]["outcome_file"] == "mortality.parquet"
    assert Path(
        outcome_file_path("/data/dlbcl", "mortality_1y", outcomes["mortality_1y"])
    ) == Path("/data/dlbcl/outcomes/mortality.parquet")


def test_normalize_outcome_config_keeps_legacy_list_behavior():
    outcomes = normalize_outcome_config(["treatment_failure"])

    assert outcomes["treatment_failure"]["outcome_file"] == "treatment_failure.parquet"
    assert outcomes["treatment_failure"]["n_hours_end_include"] is None


def test_sweep_contract_rejects_unknown_nested_fields():
    with pytest.raises(ConfigValidationError, match="unknown fields"):
        SweepConfig.from_mapping(
            {
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
        )


def test_sweep_contract_normalizes_legacy_outcomes_and_seeds():
    config = SweepConfig.from_mapping(
        {
            "cohorts": {"dlbcl": {"data_dir": "data/dlbcl"}},
            "outcomes": ["mortality"],
            "model_variants": {
                "random": {"encoder_source": "random_init"},
            },
            "seeds": [42, 43],
        }
    ).to_mapping()

    assert config["outcomes"]["mortality"]["outcome_file"] == "mortality.parquet"
    assert config["seeds"] == [42, 43]


def test_outcome_registry_date_overrides_cohort_default_including_null():
    cohort = {"registry_start_date": "2017-01-01"}

    assert resolve_registry_start_date(cohort, {}) == "2017-01-01"
    assert (
        resolve_registry_start_date(
            cohort,
            {"registry_start_date": "2019-01-01"},
        )
        == "2019-01-01"
    )
    assert (
        resolve_registry_start_date(
            cohort,
            {"registry_start_date": None},
        )
        is None
    )


def test_sweep_contract_normalizes_yaml_registry_dates(tmp_path):
    path = tmp_path / "sweep.yaml"
    path.write_text(
        """
cohorts:
  dlbcl:
    data_dir: data/dlbcl
    registry_start_date: 2017-01-01
outcomes:
  aki_30d:
    registry_start_date: 2018-02-03
model_variants:
  random:
    encoder_source: random_init
"""
    )

    config = load_sweep_config(path)

    assert config.cohorts["dlbcl"].registry_start_date == "2017-01-01"
    assert config.outcomes["aki_30d"].registry_start_date == "2018-02-03"


def test_variant_outcome_filter_is_explicit_and_validated():
    config = SweepConfig.from_mapping(
        {
            "cohorts": {"dlbcl": {"data_dir": "data/dlbcl"}},
            "outcomes": {
                "mortality": {"n_hours_end_include": 8760},
                "treatment_failure": {"n_hours_end_include": None},
            },
            "model_variants": {
                "ipcw": {
                    "encoder_ckpt": "checkpoints/opera.ckpt",
                    "encoder_source": "contrastive",
                    "training_mode": "ipcw_bce",
                    "exclude_outcomes": ["treatment_failure"],
                }
            },
        }
    ).to_mapping()
    variant = config["model_variants"]["ipcw"]

    assert variant_applies_to_outcome(variant, "mortality")
    assert not variant_applies_to_outcome(variant, "treatment_failure")


def test_generated_primary_sweep_has_only_current_model_ladder():
    import yaml

    raw = yaml.safe_load(
        Path("opera/configs/generated/fine_cox.yaml").read_text(encoding="utf-8")
    )
    config = SweepConfig.from_mapping(raw).to_mapping()
    variants = config["model_variants"]

    assert variants["opera"]["training_mode"] == "cox_exact_cached"
    assert variants["multi_outcome"]["training_mode"] == "cox_exact_cached"
    assert "opera_per_grouped" not in variants
