import pandas as pd
import pytest

from opera.run.plot_neural_model_scoreboard import summarize_models


def test_scoreboard_outcome_and_family_aggregations_are_distinct():
    rows = []
    for model, scores in (("base", (.6, .7, .8)), ("router", (.7, .8, .9))):
        for outcome, family, score in zip(("a", "b", "c"), ("f1", "f1", "f2"), scores):
            rows.append({"model": model, "outcome": outcome,
                         "outcome_family": family, "estimate": score,
                         "horizon_days": 365})
    summary = summarize_models(pd.DataFrame(rows), "base").set_index("model")
    assert summary.loc["base", "outcome_macro_auc"] == pytest.approx(.7)
    assert summary.loc["base", "family_balanced_auc"] == pytest.approx(.725)
    assert summary.loc["router", "outcome_macro_delta_vs_baseline"] == pytest.approx(.1)


def test_scoreboard_requires_matched_outcomes():
    frame = pd.DataFrame({"model": ["base", "router"], "outcome": ["a", "b"],
                          "outcome_family": ["f", "f"], "estimate": [.7, .8],
                          "horizon_days": [365, 365]})
    with pytest.raises(ValueError, match="same outcomes"):
        summarize_models(frame, "base")
