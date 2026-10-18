"""Manuscript figure functions render from analysis-shaped tables."""

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from opera.run import render_paper_figure_prototypes as proto
from opera.visualization import paper_figures as figures


def test_every_manuscript_figure_renders_and_keeps_unestimable_cells_grey(tmp_path):
    rng = np.random.default_rng(1)
    cells = proto.synthetic_cells(rng)
    cells = cells.loc[cells.cohort.isin(cells.cohort.unique()[:6])]
    curve = proto.synthetic_curve(cells)
    estimates, external = proto.synthetic_external(rng)
    rendered = [
        figures.figure_atlas(proto.synthetic_atlas(cells, rng)),
        figures.figure_pooling(cells, curve),
        figures.figure_tabular(cells, curve),
        figures.figure_external(estimates, pd.concat(
            [external, cells.loc[cells.contrast == "EA-L:J"]])),
        figures.figure_routing(cells, proto.synthetic_gate()),
        figures.figure_horizons(cells),
        figures.figure_calibration_maps(*proto.synthetic_maps(rng, n=300), grid_size=40),
        figures.figure_pooling_cells(cells),
    ]
    routes, disease_groups, outcome_groups = proto.synthetic_routes(rng)
    keep = list(disease_groups)[:6]
    rendered.append(figures.figure_router_usage(
        routes.loc[routes.cohort.isin(keep) & (routes.bank != "disease")
                   | routes.cohort.isin(keep) & routes.source.isin(keep)],
        {d: disease_groups[d] for d in keep}, outcome_groups))
    assert len(rendered[0].axes) >= 1  # the atlas is a single panel
    assert all(len(fig.axes) >= 2 for fig in rendered[1:])
    matrix_axis = rendered[7].axes[0]  # cell-level pooling supplement
    data = matrix_axis.images[0].get_array()
    # NaN gains (not estimable) are masked, so they render as the "bad" grey.
    assert np.ma.isMaskedArray(data) and data.mask.any()
    for fig in rendered:
        plt.close(fig)


def test_calibration_risk_group_and_supplementary_figures_render():
    rng = np.random.default_rng(2)
    cells = proto.synthetic_cells(rng)
    summary, reliability, examples = proto.synthetic_calibration(cells, rng)
    assert len(examples) == 4  # one reliability panel per example, no empty slots
    who, national = proto.synthetic_subgroups(rng)
    rendered = [
        figures.figure_calibration(summary, reliability, examples),
        figures.figure_risk_groups(proto.synthetic_risk_patients(rng, n=600),
                                   outcome_label="Treatment failure", ipi_label="NCCN-IPI"),
        figures.figure_coverage(proto.synthetic_coverage(cells, rng)),
        figures.figure_support(cells),
        figures.figure_rescue(proto.synthetic_rescue(cells, rng)),
        figures.figure_strategy(*proto.synthetic_strategy(cells, rng)),
        figures.figure_subgroup_effects(
            who, contrast_styles={"J-S": ("#343A88", "Joint vs specialist")},
            covariate_order={"age": "Age", "sex": "Sex"}, xlabel="Brier gain"),
    ]
    assert len(rendered[0].axes) == 2 + len(examples)
    assert all(fig.axes for fig in rendered)
    for fig in rendered:
        plt.close(fig)


def test_cumulative_incidence_matches_one_minus_kaplan_meier_without_competing_events():
    time = np.array([1, 2, 2, 3, 4, 5, 6], float)
    event = np.array([1, 0, 1, 1, 0, 1, 0])
    cif = figures.cumulative_incidence(time, event, [0, 1, 2, 3, 6])
    # KM by hand: t=1 7 at risk 1 event; t=2 6 at risk 1 event; t=3 4 at risk 1 event
    km = np.cumprod([1 - 1 / 7, 1 - 1 / 6, 1 - 1 / 4])
    np.testing.assert_allclose(cif[1:4], 1 - km)
    np.testing.assert_allclose(cif[4], 1 - km[-1] * (1 - 1 / 2))


def test_event_free_risk_groups_use_fixed_groups_and_km_complement():
    import pytest
    patients = proto.synthetic_risk_patients(np.random.default_rng(21), n=600)
    fig = figures.figure_risk_groups(patients, outcome_label="Treatment failure",
                                     event_free=True)
    assert "survival" in fig.axes[0].get_ylabel()
    assert len(fig.axes[0].tables) == 1
    for line in fig.axes[0].lines[:-1]:
        assert line.get_ydata()[0] == 1
        assert np.all(np.diff(line.get_ydata()) <= 0)
    plt.close(fig)
    patients.loc[patients.index[0], "event"] = 2
    with pytest.raises(ValueError, match="death included"):
        figures.figure_risk_groups(patients, outcome_label="Treatment failure",
                                  event_free=True)


def test_headline_atlas_benchmarks_and_patient_map_render():
    rng = np.random.default_rng(3)
    cells = proto.synthetic_cells(rng)
    curve = proto.synthetic_curve(cells)
    models = proto.synthetic_model_cells(cells, rng)
    who, _ = proto.synthetic_subgroups(rng)
    atlas = proto.synthetic_atlas(cells, rng)
    gallery = proto.synthetic_gallery(rng, n=600)
    rendered = [
        figures.figure_headline(cells, models, who, proto.synthetic_headline_scores(rng)),
        figures.figure_atlas(atlas, models),
        figures.figure_atlas_matrix(atlas, models),
        figures.figure_benchmarks(cells, curve, *proto.synthetic_calibration(cells, rng)),
        figures.figure_patient_map(*gallery, grid_size=40),
        figures.figure_embedding_gallery(*gallery, page=1, grid_size=30),
    ]
    assert all(fig.axes for fig in rendered)
    for fig in rendered:
        plt.close(fig)


def test_region_map_uses_packaged_boundaries_and_tiles_fallback():
    effects, models, coverage = proto.synthetic_regions(np.random.default_rng(4))
    for geometry in (None, False):
        fig = figures.figure_regions(effects, models, coverage, geometry)
        assert fig.axes[0].patches  # regions drawn
        plt.close(fig)
