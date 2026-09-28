"""Regression checks for scientific encodings and publication exports."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from opera.visualization.classification_plots import plot_roc_comparison, plot_decision_curve
from opera.visualization.comparison_plots import plot_forest, plot_metric_comparison
from opera.visualization.disease_geometry_plots import plot_transfer_matrix
from opera.visualization.style import save_fig
from opera.visualization.survival_plots import plot_decision_curves


def teardown_function():
    plt.close("all")


def test_comparison_keeps_more_than_eight_models():
    results = {
        f"model_{i}": (np.array([0, 1, 0, 1]), np.array([0.1, 0.8, 0.4, 0.6]))
        for i in range(12)
    }
    fig = plot_roc_comparison(results)
    assert len(fig.axes[0].get_legend().get_texts()) == 12
    assert len(fig.axes[0].lines) == 13
    identities = [
        (line.get_color(), str(line._unscaled_dash_pattern))
        for line in fig.axes[0].lines[:-1]
    ]
    assert len(set(identities)) == 12


def test_pr_forest_does_not_invent_prevalence():
    fig = plot_forest(
        {"opera": {"auprc": {"mean": 0.4, "lower": 0.3, "upper": 0.5}}}, metric="auprc"
    )
    assert len(fig.axes[0].lines) == 1


def test_decision_curve_divergent_reference_does_not_expand_upper_limit():
    labels = np.array([0] * 90 + [1] * 10)
    probabilities = np.linspace(0.01, 0.4, 100)
    fig = plot_decision_curve(labels, probabilities)
    assert fig.axes[0].get_ylim()[1] < 0.3


def test_shared_metric_axes_keep_input_order():
    fig = plot_metric_comparison(
        {"opera": {"auroc": 0.8, "auprc": 0.5}, "dapt": {"auroc": 0.7, "auprc": 0.4}}
    )
    assert all(ax.yaxis_inverted() for ax in fig.axes)


def test_transfer_missing_diagonal_is_masked_and_range_not_clipped():
    matrix = pd.DataFrame(
        [[0.8, 0.9], [0.2, np.nan]], index=["a", "b"], columns=["a", "b"]
    )
    fig = plot_transfer_matrix(matrix)
    artist = fig.axes[0].images[0]
    assert artist.get_array().mask[:, 1].all()
    assert artist.norm.vmin <= 0.25
    assert artist.get_array()[1, 0] == 0.25


def test_dca_sorts_and_deduplicates_matched_reference():
    frame = pd.DataFrame(
        {
            "model_family": ["opera", "opera", "dapt", "dapt"],
            "threshold": [0.2, 0.1, 0.2, 0.1],
            "net_benefit_model": [0.08, 0.1, 0.05, 0.09],
            "net_benefit_treat_all": [0.02, 0.03, 0.02, 0.03],
        }
    )
    fig = plot_decision_curves(frame, "event")
    assert all(np.all(np.diff(line.get_xdata()) >= 0) for line in fig.axes[0].lines)
    assert len(fig.axes[0].lines[-2].get_xdata()) == 2


def test_dense_export_keeps_text_editable(tmp_path):
    fig, ax = plt.subplots()
    dense = ax.scatter(np.arange(1001), np.arange(1001))
    sparse = ax.scatter([0], [1])
    ax.set_xlabel("Editable label")
    save_fig(fig, tmp_path / "dense.png")
    save_fig(fig, tmp_path / "dense.svg")
    assert dense.get_rasterized() and not sparse.get_rasterized()
    assert (tmp_path / "dense.pdf").exists()
    svg = (tmp_path / "dense.svg").read_text(encoding="utf-8")
    assert "<text" in svg and "Editable label" in svg
