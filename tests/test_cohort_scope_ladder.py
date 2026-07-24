from pathlib import Path

import pytest

from opera.evaluation.cohort_scope import (
    build_adaptation_breadth_ladder,
    build_ladder_plan,
    build_ladder_plan_from_manifest,
)


def test_adaptation_breadth_ladder_keeps_finetune_cohort_fixed():
    scopes = build_adaptation_breadth_ladder(
        target_cohort="dlbcl",
        cohorts=["dlbcl", "cll", "mm"],
        neighbours={"dlbcl": ["cll"]},
    )

    assert [scope.name for scope in scopes] == [
        "target_only",
        "target_plus_neighbours",
        "full_spectrum",
    ]
    assert [scope.adaptation_cohorts for scope in scopes] == [
        ("dlbcl",),
        ("dlbcl", "cll"),
        ("dlbcl", "cll", "mm"),
    ]
    assert {scope.finetune_cohort for scope in scopes} == {"dlbcl"}
    assert {scope.general_pretraining for scope in scopes} == {"frozen"}

    plan = build_ladder_plan(scopes)
    assert plan["finetune_cohort"] == "dlbcl"
    assert "dataset=dlbcl" in plan["scopes"][1]["hydra_overrides"]["finetune"]
    assert (
        "cohort_scope.adaptation_cohorts=[dlbcl,cll]"
        in plan["scopes"][1]["hydra_overrides"]["contrastive"]
    )


def test_adaptation_breadth_ladder_rejects_unknown_neighbour():
    with pytest.raises(ValueError, match="Neighbour cohorts"):
        build_adaptation_breadth_ladder(
            target_cohort="dlbcl",
            cohorts=["dlbcl"],
            neighbours={"dlbcl": ["cll"]},
        )


def test_ladder_manifest_dry_run_plan(tmp_path):
    manifest = tmp_path / "ladder.yaml"
    manifest.write_text(
        "\n".join(
            [
                "target_cohort: dlbcl",
                "cohorts:",
                "  dlbcl: {}",
                "  cll: {}",
                "neighbours:",
                "  dlbcl:",
                "    - cll",
            ]
        ),
        encoding="utf-8",
    )

    plan = build_ladder_plan_from_manifest(Path(manifest))

    assert plan["target_cohort"] == "dlbcl"
    assert plan["scopes"][2]["adaptation_cohorts"] == ["dlbcl", "cll"]
