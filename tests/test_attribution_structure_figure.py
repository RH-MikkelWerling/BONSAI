"""Protect the endpoint-level estimand and alignment in the attribution plate."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from opera.visualization.attribution_overview import figure_attribution_structure


def inputs():
    outcomes = ["anemia_g3plus", "overall_survival", "anemia_g2plus"]
    similarity = pd.DataFrame([
        {"outcome_a": a, "outcome_b": b, "spearman": 1 if a == b else .4}
        for a in outcomes for b in outcomes])
    effects = pd.DataFrame([
        {"outcome": outcomes[0], "restriction": "Without laboratory results", "delta": -.12},
        {"outcome": outcomes[1], "restriction": "Without laboratory results", "delta": -.03}])
    summary = pd.DataFrame([{"group": "All outcomes", "restriction": "Without laboratory results",
                             "mean_delta": -.075}])
    domains = {o: "Disease course and later malignancy" if o == "overall_survival"
               else "Erythroid findings and transfusion"
               for o in outcomes}
    return similarity, effects, summary, domains


def test_individual_effects_align_with_similarity_rows_and_preserve_missingness():
    similarity, effects, summary, domains = inputs()
    fig = figure_attribution_structure(similarity, summary, outcome_restrictions=effects,
                                       domains=domains)
    try:
        fig.canvas.draw()
        matrix_ax, effect_ax, summary_ax = [ax for ax in fig.axes if ax.images]
        rows = fig.attribution_row_order
        assert fig.attribution_outcome_order == fig.restriction_outcome_order
        assert fig.attribution_outcome_order == (
            "overall_survival", "anemia_g2plus", "anemia_g3plus")
        values = effect_ax.images[0].get_array()
        assert values[rows.index("anemia_g3plus"), 0] == pytest.approx(-12)
        assert values[rows.index("overall_survival"), 0] == pytest.approx(-3)
        assert np.ma.is_masked(values[rows.index("anemia_g2plus"), 0])
        assert summary_ax.images[0].get_array()[0, 0] == pytest.approx(-7.5)
        assert matrix_ax.get_ylim() == effect_ax.get_ylim()
        assert matrix_ax.get_position().y0 == effect_ax.get_position().y0
        assert matrix_ax.get_position().y1 == effect_ax.get_position().y1
    finally:
        plt.close(fig)


def test_family_averages_cannot_substitute_for_individual_effects():
    similarity, _, summary, domains = inputs()
    with pytest.raises(ValueError, match="input_restriction_auc.csv"):
        figure_attribution_structure(similarity, summary, domains=domains)


def test_multiple_restriction_horizons_are_not_silently_averaged():
    similarity, effects, summary, domains = inputs()
    with pytest.raises(ValueError, match="Select one restriction horizon"):
        figure_attribution_structure(similarity, summary,
            outcome_restrictions=pd.concat([effects, effects]), domains=domains)


def test_outcomes_missing_attribution_estimates_are_retained_in_both_panels():
    similarity, effects, summary, domains = inputs()
    similarity = similarity.loc[lambda d: d.outcome_a.ne("overall_survival")
                                & d.outcome_b.ne("overall_survival")]
    fig = figure_attribution_structure(similarity, summary, outcome_restrictions=effects,
                                       domains=domains)
    try:
        assert "overall_survival" in fig.attribution_outcome_order
        matrix = fig.axes[0].images[0].get_array()
        row = fig.attribution_row_order.index("overall_survival")
        assert np.ma.getmaskarray(matrix)[row].all()
    finally:
        plt.close(fig)

