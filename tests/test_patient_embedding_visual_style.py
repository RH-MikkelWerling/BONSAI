"""Patient plotting regressions that do not require filesystem fixtures."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path

from opera.visualization.style import sequential_cmap
from opera.run import plot_patient_embeddings as patient
from opera.visualization import embedding_plots


def teardown_function():
    plt.close("all")


def test_training_curve_prefers_logged_epoch_loss_over_batch_loss():
    frame = pd.DataFrame({
        "epoch": [0, 0, 1, 1], "train/loss_step": [9., 8., 7., 6.],
        "train/loss_epoch": [np.nan, 2., np.nan, 1.],
    })
    fig = embedding_plots.plot_training_curves(frame, smooth_window=1)
    assert len(fig.axes) == 1
    np.testing.assert_array_equal(fig.axes[0].lines[0].get_ydata(), [2., 1.])


def test_training_curve_does_not_relabel_repeated_batch_losses_as_epochs():
    import pytest
    frame = pd.DataFrame({"epoch": [0, 0, 1], "train/loss": [3., 2., 1.]})
    with pytest.warns(UserWarning, match="multiple values per epoch"):
        fig = embedding_plots.plot_training_curves(frame)
    assert not fig.axes[0].lines


def test_opera_continuous_palette_is_plasma():
    assert sequential_cmap("opera").name == "plasma"
    np.testing.assert_allclose(sequential_cmap("opera")(0.5), plt.get_cmap("plasma")(0.5))


def test_categorical_facets_paginate_without_losing_or_recoloring_groups(monkeypatch):
    values = np.array([f"cohort_{i:02d}" for i in range(25)])
    frame = pd.DataFrame({"component_1": np.arange(25), "component_2": np.arange(25)})
    pages = []
    monkeypatch.setattr(patient, "_plot_categorical_facet_page", lambda *args, **kwargs: pages.append(kwargs))
    paths = patient.plot_categorical_facets(
        frame, values, np.ones(25, dtype=bool), color_by="cohort", method="pca",
        output_path=Path("synthetic.png"), seed=0,
    )
    assert len(paths) == 3
    assert [len(page["levels"]) for page in pages] == [12, 12, 1]
    assert set(sum([page["levels"] for page in pages], [])) == set(values)
    assert pages[0]["colors"] == pages[-1]["colors"]


def test_multioutcome_scatter_has_one_shared_legend_and_no_default_contours():
    rng = np.random.default_rng(5)
    coords = rng.normal(size=(200, 2))
    outcomes = {f"outcome_{i}": rng.integers(-1, 2, len(coords)) for i in range(4)}
    fig = embedding_plots.plot_embedding_multi_outcome(coords, outcomes, method="pca")
    assert len(fig.legends) == 1
    assert all(ax.get_legend() is None for ax in fig.axes)
    assert len({ax.get_xlim() for ax in fig.axes if ax.get_visible()}) == 1


def test_dense_embedding_markers_get_smaller_and_lighter():
    small = embedding_plots._embedding_point_style(100)
    dense = embedding_plots._embedding_point_style(30000)
    assert dense[0] < small[0]
    assert dense[1] < small[1]


def test_high_cardinality_dispatch_returns_every_page_and_skips_centroids(monkeypatch):
    frame = pd.DataFrame({
        "component_1": np.arange(13), "component_2": np.arange(13),
        "cohort": [f"cohort_{i}" for i in range(13)],
    })
    expected = [Path("page_01.png"), Path("page_02.png")]
    monkeypatch.setattr(patient, "plot_categorical_facets", lambda *args, **kwargs: expected)
    monkeypatch.setattr(
        patient, "plot_categorical_centroids",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("Crowded centroid view was generated")),
    )
    assert patient.plot_colored_coordinates(
        frame, color_by="cohort", method="pca", output_dir=Path("preview"),
    ) == expected
