import matplotlib.pyplot as plt
import pytest

from opera.run.render_paper_figure_prototypes import synthetic_embedding_comparison
from opera.visualization.paper_figures import figure_embedding_comparison


def test_embedding_matrix_preserves_coordinates_and_shared_scales():
    coords = synthetic_embedding_comparison()
    fig = figure_embedding_comparison(coords)
    try:
        # Map panels: grid subplots with visible axes (legend columns are hidden).
        panels = [ax for ax in fig.axes
                  if ax.get_subplotspec() is not None and ax.axison and ax.collections]
        assert len(panels) == 20  # five stages x disease, age, history, risk category
        for col in range(5):
            assert panels[col].get_xlim() == panels[5 + col].get_xlim()
            assert panels[col].get_ylim() == panels[15 + col].get_ylim()
        for row in (1, 2, 3):  # patient colours share one scale across stages
            norms = [panels[row * 5 + col].images[-1].norm for col in range(5)]
            assert all(norm is norms[0] for norm in norms)
    finally:
        plt.close(fig)


def test_embedding_matrix_rejects_unmatched_patients_and_metadata():
    coords = synthetic_embedding_comparison()
    with pytest.raises(ValueError, match="identical patients"):
        figure_embedding_comparison(coords.iloc[1:])
    coords.loc[0, "age"] += 1
    with pytest.raises(ValueError, match="metadata must agree"):
        figure_embedding_comparison(coords)


def test_colour_footprint_preserves_empty_space_between_patient_islands():
    import numpy as np
    from opera.visualization.paper_figures import _embedding_surface_mask
    rng = np.random.default_rng(5)
    points = np.vstack([rng.normal((-4, 0), 0.2, (100, 2)),
                        rng.normal((4, 0), 0.2, (100, 2))])
    xx, yy = np.meshgrid(np.linspace(-6, 6, 121), np.linspace(-3, 3, 61))
    # Even if Gaussian effective support/density passed everywhere, the gap
    # and corners must stay blank. A hull would incorrectly fill the gap.
    mask = _embedding_surface_mask(xx, yy, points, np.full(xx.shape, 100),
                                   np.ones(xx.shape), 20)
    assert mask[30, 60]  # midpoint between islands
    assert mask[0, 0]    # empty corner
    assert not mask[30, 20]  # centre of left patient island
    assert not mask[30, 100]  # centre of right patient island
