import json

import pandas as pd
import pytest

from opera.run.audit_router_gain_mechanism import route_diagnostics, select_outcomes


def test_selects_positive_neutral_and_negative_without_inventing_losses():
    contrasts = pd.DataFrame({
        "outcome": ["big", "small", "flat", "loss"],
        "estimate": [.1, .02, .001, -.04],
    })
    result = select_outcomes(contrasts, n_each=1).set_index("outcome")
    assert result.loc["big", "selection_reason"] == "largest_gain"
    assert result.loc["flat", "selection_reason"] == "smallest_absolute_gain"
    assert result.loc["loss", "selection_reason"] == "largest_loss"
    no_losses = select_outcomes(contrasts.loc[contrasts.estimate > 0], n_each=1)
    assert "largest_loss" not in set(no_losses.selection_reason)


def test_patient_route_variation_uses_half_l1_and_rejects_bad_splits():
    rows = []
    for i in range(20):
        d = .1 + (.02 if i < 10 else -.02)
        rows.append({"subject_id": i, "cohort": "c", "split": "tuning",
                     "outcome": "sepsis", "fine_index": 0,
                     "shared_weight": .5,
                     "disease_weights": json.dumps({"disease:A": d}),
                     "outcome_weights": json.dumps({"outcome:sepsis": .5 - d})})
    routes = pd.DataFrame(rows)
    selected = pd.DataFrame({"outcome": ["sepsis"]})
    summary = route_diagnostics(routes, selected)
    assert summary.iloc[0]["mean_patient_tv_from_cell_mean"] == pytest.approx(.02)
    routes.loc[0, "split"] = "held_out"
    with pytest.raises(ValueError, match="tuning-only"):
        route_diagnostics(routes, selected)
