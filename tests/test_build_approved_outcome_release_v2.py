from pathlib import Path
from unittest.mock import patch

import pandas as pd
import yaml

from opera.run.build_approved_outcome_release_v2 import _registry, composite_earliest


def test_approved_registry_covers_aliases_and_additions() -> None:
    source = yaml.safe_load(Path("opera/configs/experiment_registry.yaml").read_text(encoding="utf-8"))
    policy = yaml.safe_load(Path("opera/configs/outcome_release_primary_v2.yaml").read_text(encoding="utf-8"))
    selected = set(source["outcomes"]) - set(policy["exclude_held"])
    result = _registry(source, selected, policy["rename"],
                       policy["add_from_lab_candidates"] + policy["add_from_potassium_candidates"])
    assert "MACE_composite" not in result["outcomes"]
    assert "infections_all_definitions" not in result["outcomes"]
    assert "infection_diagnosis_or_antimicrobial_use_composite" in result["outcomes"]
    assert "alp_elevation_legacy_tier2plus" in result["outcomes"]
    assert "creatinine_rise_gt1p5x_preindex_baseline" in result["outcomes"]
    assert "potassium_below_3p0_mmol_l" in result["outcomes"]


def test_release_policy_resolves_all_declared_definition_holds() -> None:
    policy = yaml.safe_load(Path("opera/configs/outcome_release_primary_v2.yaml").read_text(encoding="utf-8"))
    rules = yaml.safe_load(Path("opera/configs/outcome_panel_decision_v1.yaml").read_text(encoding="utf-8"))
    holds = {name for name, spec in rules["clinical_actions"].items()
             if spec["action"] == "definition_hold"}
    included = set(policy["include_held"])
    excluded = set(policy["exclude_held"])
    assert not included & excluded
    assert included | excluded == holds


def test_rebuilt_composite_intersects_eligibility_and_censor() -> None:
    index = pd.Timestamp("2020-01-01")
    a = pd.DataFrame({"subject_id": [1, 2], "split": ["train", "train"],
                      "index_date": [index, index],
                      "outcome_date": [index + pd.Timedelta(days=10), pd.NaT],
                      "censor_date": [index + pd.Timedelta(days=100)] * 2})
    b = pd.DataFrame({"subject_id": [1, 3], "split": ["train", "train"],
                      "index_date": [index, index],
                      "outcome_date": [index + pd.Timedelta(days=80), pd.NaT],
                      "censor_date": [index + pd.Timedelta(days=50)] * 2})
    frames = {Path("alt.parquet"): a, Path("ast.parquet"): b}
    with patch("opera.run.build_approved_outcome_release_v2._read_outcome",
               side_effect=lambda path: frames[path]):
        result = composite_earliest({"ALT": Path("alt.parquet"), "AST": Path("ast.parquet")},
                                    ["ALT", "AST"])
    assert result["subject_id"].tolist() == [1]
    assert result.loc[0, "outcome_date"] == index + pd.Timedelta(days=10)
    assert result.loc[0, "censor_date"] == index + pd.Timedelta(days=50)
