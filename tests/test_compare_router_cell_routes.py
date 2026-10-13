import json

import pandas as pd
import pytest

from opera.run.compare_router_cell_routes import compare_cell_routes


def _routes(shared: float):
    return pd.DataFrame([{
        "subject_id": i, "cohort": "CLL", "outcome": "sepsis", "fine_index": 0,
        "shared_weight": shared,
        "disease_weights": json.dumps({"disease:CLL": .2, "disease:MM": .2}),
        "outcome_weights": json.dumps({"outcome:sepsis": .2,
                                       "outcome:aki": .4 - shared}),
    } for i in range(25)])


def test_route_comparison_requires_same_patients_and_reports_tv():
    names = ["shared", "disease:CLL", "disease:MM", "outcome:aki", "outcome:sepsis"]
    summary, sources = compare_cell_routes(_routes(.2), _routes(.1), names,
                                           disease="CLL", outcome="sepsis")
    assert summary["n_patients"] == 25
    assert summary["mean_paired_patient_route_tv"] == pytest.approx(.1)
    assert summary["mean_route_tv"] == pytest.approx(.1)
    assert sources.set_index("source").loc["shared", "loose_minus_clean"] == pytest.approx(-.1)
    changed = _routes(.1)
    changed.loc[0, "subject_id"] = 999
    with pytest.raises(ValueError, match="different tuning patient"):
        compare_cell_routes(_routes(.2), changed, names, disease="CLL", outcome="sepsis")
