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
        assert len(panels) == 15  # five stages x disease, risk category, age
        for col in range(5):
            assert panels[col].get_xlim() == panels[5 + col].get_xlim()
            assert panels[col].get_ylim() == panels[10 + col].get_ylim()
        for row in (1, 2):  # surfaces share one colour scale across stages
            norms = [panels[row * 5 + col].collections[-1].norm for col in range(5)]
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
