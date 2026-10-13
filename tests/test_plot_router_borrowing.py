import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from opera.run.plot_router_borrowing import (
    _anchor_target_figure, _source_centered, summarize_routes,
)


def _config():
    return {
        "competing_risk": {"readout_mode": "source_anchored_adapters",
                           "fine_names": ["D1", "D2"]},
        "outcomes": {"x": {}, "y": {}},
        "cross_outcome": {"outcome_families": {"X": ["x"], "Y": ["y"]}},
    }


def _routes():
    rows = []
    for fine_index in (0, 1):
        for outcome in ("x", "y"):
            disease = ({"disease:D1": .2, "disease:D2": .1}
                       if fine_index == 0 else
                       {"disease:D1": .05, "disease:D2": .25})
            outcome_weights = ({"outcome:x": .1, "outcome:y": .1}
                               if outcome == "x" else
                               {"outcome:x": .15, "outcome:y": .05})
            rows.append({
                "subject_id": fine_index + 1, "split": "tuning", "outcome": outcome,
                "fine_index": fine_index, "shared_weight": .5,
                "disease_weights": json.dumps(disease),
                "outcome_weights": json.dumps(outcome_weights),
            })
    return pd.DataFrame(rows)


def test_directed_borrowing_excludes_home_source():
    report = summarize_routes(_routes(), _config(), min_rows=1)
    assert report["n_rows"] == 4
    assert np.isnan(report["disease_matrix"][0, 0])
    assert report["disease_matrix"][0, 1] == pytest.approx(.1)
    assert report["disease_matrix"][1, 0] == pytest.approx(.05)
    assert report["outcome_matrix"][0, 1] == pytest.approx(.1)
    assert report["outcome_matrix"][1, 0] == pytest.approx(.15)
    assert report["outcome_disease_matrix"][0, 0] == pytest.approx(.125)
    assert report["family_matrix"][0, 1] == pytest.approx(.1)
    assert report["family_matrix"][1, 0] == pytest.approx(.15)
    assert report["family_per_adapter_matrix"][0, 1] == pytest.approx(.1)
    assert report["mass_by_target"]["mean_borrowed_disease"].iloc[0] == pytest.approx(.075)


def test_family_overview_removes_source_family_size_effect():
    config = _config()
    config["outcomes"]["z"] = {}
    config["cross_outcome"]["outcome_families"]["Y"].append("z")
    frame = _routes().loc[lambda rows: rows["outcome"].eq("x")].copy()
    frame["outcome_weights"] = json.dumps({"outcome:x": .1,
                                           "outcome:y": .1, "outcome:z": .1})
    frame["shared_weight"] = .4
    report = summarize_routes(frame, config, min_rows=1)
    assert report["family_matrix"][0, 1] == pytest.approx(.2)
    assert report["family_per_adapter_matrix"][0, 1] == pytest.approx(.1)


def test_specificity_removes_global_source_popularity():
    matrix = np.array([[np.nan, .2, .1], [.4, np.nan, .1], [.6, .2, np.nan]])
    centered, baseline = _source_centered(matrix)
    assert baseline.tolist() == pytest.approx([.5, .2, .1])
    assert centered[1, 0] == pytest.approx(-.1)
    assert centered[2, 0] == pytest.approx(.1)
    assert centered[0, 1] == pytest.approx(0.)


def test_anchor_figure_keeps_individual_outcome_targets(monkeypatch):
    from matplotlib.figure import Figure
    saved = []
    monkeypatch.setattr(Figure, "savefig", lambda self, path, **kwargs: saved.append(str(path)))
    summary = summarize_routes(_routes(), _config(), min_rows=1)
    summary["outcome_names"][0] = "sepsis"
    summary["mass_by_target"].loc[
        summary["mass_by_target"]["target_outcome"].eq("x"), "target_outcome"
    ] = "sepsis"
    _anchor_target_figure(summary, Path("unused"))
    assert any(path.endswith("router_borrowing_anchor_targets.png") for path in saved)


def test_rejects_non_tuning_and_bad_mass():
    frame = _routes()
    frame.loc[0, "split"] = "held_out"
    with pytest.raises(ValueError, match="tuning-only"):
        summarize_routes(frame, _config(), min_rows=1)
    frame.loc[0, "split"] = "tuning"
    frame.loc[0, "shared_weight"] = .9
    with pytest.raises(ValueError, match="non-normalized"):
        summarize_routes(frame, _config(), min_rows=1)
