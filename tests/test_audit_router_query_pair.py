import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

import opera.run.audit_router_query_pair as audit_module
from opera.run.audit_router_query_pair import (
    _checkpoint, _flatten, _paired_predictions, _paired_routes,
    _route_target_interaction_probe,
)


def test_current_router_code_identifies_loaded_file():
    info = audit_module._current_router_code()
    assert Path(info["loaded_class_file"]).is_file()
    assert len(info["loaded_class_file_sha256"]) == 64
    assert info["has_interacting_branch"] is True
    assert "historical" in info["scope"]


def test_flatten_exposes_only_intended_query_path():
    assert _flatten({"competing_risk": {"adapter_query_mode": "additive"}}) == {
        "competing_risk.adapter_query_mode": "additive"
    }


def test_checkpoint_checks_actual_interaction_tensors(monkeypatch):
    prefix = "model.source_adapter_readout."
    state = {
        prefix + name: torch.ones(shape)
        for name, shape in {
            "patient_query.weight": (4, 4), "fine_query.weight": (2, 4),
            "target_query.weight": (2, 4), "source_keys": (4, 4),
            "shared_logit": (), "down": (4, 2, 4), "up": (4, 4, 2),
            "source_shrinkage": (4,), "query_interaction.0.weight": (3, 12),
        }.items()
    }
    state["model.competing_risk_head.weight"] = torch.ones(4, 4)
    path = Path("best.ckpt")
    checkpoint = {"epoch": 3, "global_step": 9, "state_dict": state,
                  "hyper_parameters": {"model_init_config": {
                      "competing_risk_config": {
                          "adapter_query_mode": "interacting",
                          "fine_column": "cohort_fine",
                          "fine_names": ["A", "B"],
                      }
                  }}}
    monkeypatch.setattr(audit_module.Path, "is_file", lambda self: True)
    monkeypatch.setattr(audit_module.torch, "load", lambda *args, **kwargs: checkpoint)
    monkeypatch.setattr(audit_module, "_sha256", lambda _: "test-hash")
    info, router = _checkpoint(path, "interacting")
    assert info["completed_epochs"] == 4
    assert info["saved_fine_column"] == "cohort_fine"
    assert info["interaction_parameters"] == 36
    assert "query_interaction.0.weight" in router
    with pytest.raises(ValueError, match="expected 'additive'"):
        _checkpoint(path, "additive")


def test_paired_predictions_requires_same_labels_and_reports_full_precision(monkeypatch):
    first = pd.DataFrame({"subject_id": [1, 2], "cohort": ["x", "x"],
                          "outcome": ["sepsis", "sepsis"], "split": ["tuning"] * 2,
                          "time_days": [3., 4.], "event": [1, 0],
                          "cif_365d": [.2, .3]})
    second = first.copy()
    second.loc[0, "cif_365d"] += 1e-7
    paths = [Path("a.parquet"), Path("b.parquet")]
    frames = {paths[0]: first, paths[1]: second}
    monkeypatch.setattr(audit_module.pd, "read_parquet", lambda path: frames[path].copy())
    result = _paired_predictions(*paths)
    assert result.iloc[0]["fraction_exactly_equal"] == .5
    assert result.iloc[0]["max_abs_difference"] == pytest.approx(1e-7)
    second.loc[0, "event"] = 0
    frames[paths[1]] = second
    with pytest.raises(ValueError, match="labels or follow-up"):
        _paired_predictions(*paths)


def test_paired_routes_detects_small_numerical_difference(monkeypatch):
    first = pd.DataFrame({
        "subject_id": [1], "cohort": ["x"], "outcome": ["sepsis"],
        "split": ["tuning"], "fine_index": [0], "shared_weight": [.5],
        "disease_weights": [json.dumps({"disease:x": .2})],
        "outcome_weights": [json.dumps({"outcome:sepsis": .3})],
    })
    second = first.copy()
    second["shared_weight"] = .499
    second["outcome_weights"] = [json.dumps({"outcome:sepsis": .301})]
    paths = [Path("a.parquet"), Path("b.parquet")]
    frames = {paths[0]: first, paths[1]: second}
    monkeypatch.setattr(audit_module.pd, "read_parquet", lambda path: frames[path].copy())
    result = _paired_routes(*paths).set_index("bank")
    assert result.loc["disease", "fraction_rows_exactly_equal"] == 1.
    assert result.loc["outcome", "max_abs_weight_difference"] == pytest.approx(.001)
    assert result.loc["shared", "max_abs_weight_difference"] == pytest.approx(.001)


def test_actual_route_probe_distinguishes_patient_target_interaction(monkeypatch):
    rows = []
    for subject in range(25):
        for target, offset in (("overall_survival", 0.), ("sepsis", .2)):
            # Source 0:1 log odds differs by a constant across targets in the
            # additive construction, even though it varies across patients.
            odds = .01 * subject + offset
            source_1 = .08
            source_0 = source_1 * np.exp(odds)
            rows.append({
                "subject_id": subject, "cohort": "x", "split": "tuning",
                "outcome": target, "fine_index": 2,
                "disease_weights": json.dumps({"disease:A": source_0,
                                               "disease:B": source_1,
                                               "disease:C": .1}),
            })
    frame = pd.DataFrame(rows)
    monkeypatch.setattr(audit_module.pd, "read_parquet", lambda path: frame.copy())
    additive, pair = _route_target_interaction_probe(Path("unused"))
    assert pair == ("disease:A", "disease:B")
    assert additive["sd_across_patients"] < 1e-12
    frame.loc[frame["outcome"].eq("sepsis"), "disease_weights"] = [
        json.dumps({"disease:A": .08 * np.exp(.01 * subject + .2 + .001 * subject),
                    "disease:B": .08, "disease:C": .1})
        for subject in range(25)
    ]
    interactive, _ = _route_target_interaction_probe(Path("unused"), source_pair=pair)
    assert interactive["sd_across_patients"] > .005
