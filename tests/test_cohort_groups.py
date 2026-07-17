"""Tests for opera.functional.cohort_groups and related sweep config contracts."""

from __future__ import annotations

import pytest

from opera.functional.cohort_groups import (
    ALL_EVALUATED_FINE,
    ALL_FINE,
    ALL_GROUPED,
    EXCLUDED_FINE,
    FINE_TO_GROUPED,
    GROUPED_TO_FINE,
    fine_to_grouped,
    grouped_to_fine,
    is_valid_fine,
    is_valid_grouped,
    resolve_training_cohort,
)


# ── Mapping completeness ──────────────────────────────────────────────────────


def test_all_fine_in_fine_to_grouped():
    """Every entry in ALL_FINE must have a mapping."""
    for name in ALL_FINE:
        assert name in FINE_TO_GROUPED, f"{name} missing from FINE_TO_GROUPED"


def test_all_grouped_in_grouped_to_fine():
    """Every entry in ALL_GROUPED must appear as a value in FINE_TO_GROUPED."""
    for name in ALL_GROUPED:
        assert name in GROUPED_TO_FINE, f"{name} missing from GROUPED_TO_FINE"


def test_grouped_to_fine_covers_all_fine():
    """The union of all GROUPED_TO_FINE values must equal ALL_FINE."""
    all_fine_via_grouped = set().union(*GROUPED_TO_FINE.values())
    assert all_fine_via_grouped == set(ALL_FINE)


def test_all_production_fine_cohorts_are_evaluated():
    assert EXCLUDED_FINE == frozenset()
    assert set(ALL_EVALUATED_FINE) == set(ALL_FINE)
    assert len(ALL_FINE) == 24


def test_fine_to_grouped_round_trip():
    """Every fine cohort must round-trip through its grouped parent."""
    for fine, grouped in FINE_TO_GROUPED.items():
        assert fine in GROUPED_TO_FINE[grouped], (
            f"{fine!r} maps to {grouped!r} but is absent from GROUPED_TO_FINE[{grouped!r}]"
        )


def test_one_to_one_cohorts_appear_in_both_namespaces():
    """One-to-one cohorts use the same name at both fine and grouped levels."""
    one_to_one = set(ALL_FINE) & set(ALL_GROUPED)
    expected = {"HL", "HCL", "MM", "MCL", "AMYLOIDOSIS"}
    assert one_to_one == expected, (
        f"Unexpected namespace overlap: {one_to_one ^ expected}"
    )


# ── Known groupings (spot checks from the study count tables) ─────────────────


@pytest.mark.parametrize(
    "fine, expected_grouped",
    [
        ("DLBCL", "DLBCL_like"),
        ("BCL", "Indolent_B_NHL"),
        ("RT", "DLBCL_like"),
        ("TRANSFORMED_FL", "DLBCL_like"),
        ("FL", "Indolent_B_NHL"),
        ("MCL", "MCL"),
        ("LPL", "Indolent_B_NHL"),
        ("EMZL", "Indolent_B_NHL"),
        ("NMZL", "Indolent_B_NHL"),
        ("SMZL", "Indolent_B_NHL"),
        ("AITL", "T_NHL"),
        ("ALCL", "T_NHL"),
        ("PTCL", "T_NHL"),
        ("TCL", "T_NHL"),
        ("CLL", "CLL_SLL"),
        ("SLL", "CLL_SLL"),
        ("BL", "BL_LBL"),
        ("LBL", "BL_LBL"),
        ("MM", "MM"),
        ("PCL", "MM"),
        ("HL", "HL"),
        ("HCL", "HCL"),
        ("AMYLOIDOSIS", "AMYLOIDOSIS"),
        ("SolM", "MM"),
    ],
)
def test_known_fine_to_grouped_mapping(fine: str, expected_grouped: str):
    assert fine_to_grouped(fine) == expected_grouped


def test_known_grouped_to_fine_cll_sll():
    assert grouped_to_fine("CLL_SLL") == frozenset({"CLL", "SLL"})


def test_known_grouped_to_fine_t_nhl():
    assert grouped_to_fine("T_NHL") == frozenset({"AITL", "ALCL", "PTCL", "TCL"})


def test_known_grouped_to_fine_mm():
    assert grouped_to_fine("MM") == frozenset({"MM", "PCL", "SolM"})


def test_known_grouped_to_fine_indolent():
    assert grouped_to_fine("Indolent_B_NHL") == frozenset(
        {"BCL", "FL", "LPL", "EMZL", "NMZL", "SMZL"}
    )


# ── Validation helpers ────────────────────────────────────────────────────────


def test_is_valid_fine():
    assert is_valid_fine("DLBCL")
    assert is_valid_fine("FL")
    assert not is_valid_fine("DLBCL_like")
    assert not is_valid_fine("unknown_cohort")


def test_is_valid_grouped():
    assert is_valid_grouped("DLBCL_like")
    assert is_valid_grouped("Indolent_B_NHL")
    assert not is_valid_grouped("DLBCL")
    assert not is_valid_grouped("unknown_cohort")


def test_fine_to_grouped_unknown_returns_none():
    assert fine_to_grouped("UNKNOWN") is None


def test_grouped_to_fine_unknown_returns_empty():
    assert grouped_to_fine("UNKNOWN") == frozenset()


# ── resolve_training_cohort ───────────────────────────────────────────────────


def test_resolve_training_cohort_fine_name():
    assert resolve_training_cohort("DLBCL") == "DLBCL_like"
    assert resolve_training_cohort("CLL") == "CLL_SLL"
    assert resolve_training_cohort("PCL") == "MM"


def test_resolve_training_cohort_grouped_name():
    assert resolve_training_cohort("DLBCL_like") == "DLBCL_like"
    assert resolve_training_cohort("MM") == "MM"


def test_resolve_training_cohort_override():
    assert (
        resolve_training_cohort("DLBCL", training_cohort_override="custom") == "custom"
    )


def test_resolve_training_cohort_unknown_raises():
    with pytest.raises(KeyError, match="UNKNOWN_COHORT"):
        resolve_training_cohort("UNKNOWN_COHORT")


# ── CohortSpec training_cohort and cohort_fine fields ────────────────────────


def test_cohort_spec_accepts_training_cohort():
    from opera.config_contracts import CohortSpec

    issues: list[str] = []
    spec = CohortSpec.from_mapping(
        "DLBCL",
        {
            "data_dir": "/data/dlbcl_like",
            "training_cohort": "dlbcl_like",
            "cohort_fine_col": "cohort_fine",
            "cohort_fine_value": "DLBCL",
        },
        issues,
    )
    assert not issues
    assert spec.training_cohort == "dlbcl_like"
    assert spec.cohort_fine_col == "cohort_fine"
    assert spec.cohort_fine_value == "DLBCL"


def test_cohort_spec_rejects_partial_cohort_fine():
    """cohort_fine_col and cohort_fine_value must both be set or both absent."""
    from opera.config_contracts import CohortSpec

    issues: list[str] = []
    CohortSpec.from_mapping(
        "DLBCL",
        {
            "data_dir": "/data/dlbcl_like",
            "cohort_fine_col": "cohort_fine",
            # cohort_fine_value missing
        },
        issues,
    )
    assert any("cohort_fine_col and cohort_fine_value" in msg for msg in issues)


def test_cohort_spec_to_mapping_round_trip():
    from opera.config_contracts import CohortSpec

    issues: list[str] = []
    spec = CohortSpec.from_mapping(
        "FL",
        {
            "data_dir": "/data/indolent_b_nhl",
            "training_cohort": "indolent_b_nhl",
            "cohort_fine_col": "cohort_fine",
            "cohort_fine_value": "FL",
            "ipi_score_col": "flipi2",
        },
        issues,
    )
    assert not issues
    mapping = spec.to_mapping()
    assert mapping["training_cohort"] == "indolent_b_nhl"
    assert mapping["cohort_fine_col"] == "cohort_fine"
    assert mapping["cohort_fine_value"] == "FL"
    assert mapping["ipi_score_col"] == "flipi2"


# ── Sweep config validation for leukemia_sweep.yaml ─────────────────────────


def test_generated_fine_sweep_config_structure():
    """The registry-generated sweep is the production fine-cohort surface."""
    from pathlib import Path
    import yaml

    config_path = (
        Path(__file__).parents[1] / "opera" / "configs" / "generated" / "fine_cox.yaml"
    )
    assert config_path.exists(), f"Config not found: {config_path}"
    with open(config_path, encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)

    assert "cohorts" in raw
    assert "outcomes" in raw
    assert "model_variants" in raw

    cohorts = raw["cohorts"]
    expected = set(ALL_EVALUATED_FINE)
    assert set(cohorts.keys()) == expected, (
        f"Cohort mismatch: extra={set(cohorts.keys()) - expected}, "
        f"missing={expected - set(cohorts.keys())}"
    )

    # Grouped cohorts pointing to the correct training_cohort
    assert cohorts["DLBCL"]["training_cohort"] == "DLBCL_like"
    assert cohorts["FL"]["training_cohort"] == "Indolent_B_NHL"
    assert cohorts["CLL"]["training_cohort"] == "CLL_SLL"
    assert cohorts["MM"]["training_cohort"] == "MM"
    assert cohorts["PCL"]["training_cohort"] == "MM"

    # 1:1 cohorts have no cohort_fine_col (they don't need subsetting)
    for name in ("HL", "HCL", "AMYLOIDOSIS", "SolM"):
        assert cohorts[name]["cohort_fine_col"] == "cohort_fine"


def test_generated_joint_opera_config_structure():
    """Joint OPERA uses all registry grouped cohorts through shared data."""
    from pathlib import Path
    import yaml

    config_path = (
        Path(__file__).parents[1] / "opera" / "configs" / "generated" / "joint_opera_full_panel.yaml"
    )
    assert config_path.exists()
    with open(config_path, encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)

    cohorts = raw["cohorts"]
    assert set(cohorts) == set(ALL_GROUPED)


# ── format_variant_path training_cohort placeholder ─────────────────────────


def test_format_variant_path_training_cohort_substitution():
    """format_variant_path must expand {training_cohort} without KeyError."""
    from opera.run.sweep import format_variant_path

    template = "/ckpts/{training_cohort}/best.ckpt"
    result = format_variant_path(
        template,
        cohort="DLBCL",
        outcome="mortality_1y",
        seed=42,
        training_cohort="dlbcl_like",
    )
    assert result == "/ckpts/dlbcl_like/best.ckpt"


def test_format_variant_path_training_cohort_fallback():
    """Without training_cohort, {training_cohort} falls back to cohort name."""
    from opera.run.sweep import format_variant_path

    template = "/ckpts/{training_cohort}/best.ckpt"
    result = format_variant_path(
        template,
        cohort="HL",
        outcome="mortality_1y",
        seed=42,
    )
    assert result == "/ckpts/HL/best.ckpt"


def test_format_variant_path_ordinary_placeholders():
    """Ordinary {cohort}/{outcome}/{seed} placeholders still work."""
    from opera.run.sweep import format_variant_path

    template = "/ckpts/{cohort}/{outcome}/seed{seed}/best.ckpt"
    result = format_variant_path(
        template,
        cohort="cll_sll",
        outcome="treatment_failure",
        seed=43,
    )
    assert result == "/ckpts/cll_sll/treatment_failure/seed43/best.ckpt"
