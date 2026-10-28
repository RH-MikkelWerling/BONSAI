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


def test_embedding_csv_parquet_identifier_types_do_not_hide_outcomes():
    import pytest
    coords, outcomes = proto.synthetic_gallery(np.random.default_rng(12), n=300)
    coords["subject_id"] = np.arange(len(coords)).astype(str)
    # Same IDs, as numeric Parquet and text CSV inputs from the real renderer.
    ids = outcomes.subject_id.drop_duplicates().tolist()
    mapping = dict(zip(ids, range(len(ids))))
    outcomes["subject_id"] = outcomes.subject_id.map(mapping)
    fig = figures.figure_embedding_gallery(coords, outcomes, per_page=4, grid_size=25,
                                            min_effective_support=1)
    assert any(ax.images for ax in fig.axes)
    plt.close(fig)
    with pytest.raises(ValueError, match="No matching patient IDs"):
        figures.figure_embedding_gallery(coords.assign(subject_id=[f"missing_{i}" for i in range(len(coords))]), outcomes)


def test_routing_bank_panel_selects_requested_metric_and_horizon():
    import pytest
    cells = proto.synthetic_cells(np.random.default_rng(50))
    gate = pd.DataFrame([
        {"contrast": "R2", "metric": metric, "horizon_days": horizon,
         "gain": gain, "lower": gain - 0.001, "upper": gain + 0.001}
        for metric, horizon, gain in [("auc", 30, 0.001), ("brier", 0, 0.002),
                                      ("auc", 0, 0.015)]])
    fig = figures.figure_routing(cells, gate)
    assert fig.axes[1].collections[0].get_offsets()[0, 0] == pytest.approx(0.015)
    plt.close(fig)
    fig = figures.figure_routing(cells, gate, metric="brier")
    assert fig.axes[1].collections[0].get_offsets()[0, 0] == pytest.approx(0.002)
    plt.close(fig)
    with pytest.raises(ValueError, match="one summary"):
        figures.figure_routing(cells, pd.concat([gate, gate]))
    with pytest.raises(ValueError, match="no summary"):
        figures.figure_routing(cells, gate, horizon=730)


def test_routing_bank_panel_marks_unavailable_confidence_intervals():
    cells = proto.synthetic_cells(np.random.default_rng(51))
    gate = pd.DataFrame([{"contrast": "R2", "gain": 0.01,
                          "lower": np.nan, "upper": np.nan}])
    fig = figures.figure_routing(cells, gate)
    assert any("CI unavailable" in t.get_text() for t in fig.axes[1].texts)
    assert len(fig.axes[1].collections[0].get_facecolors()) == 0
    plt.close(fig)


def test_headline_uses_one_year_absolute_auc_and_excludes_pooled_log_horizon():
    cells = proto.synthetic_cells(np.random.default_rng(1))
    cells = pd.concat([cells, cells.loc[cells.horizon_days == 365].assign(horizon_days=0)])
    models = pd.DataFrame({'cohort':['A','A'], 'outcome':['x','x'], 'horizon_days':[0,365],
                           'J':[.2,.9], 'X':[.3,.4]})
    fig = figures.figure_headline(cells, models=models)
    scatter = fig.axes[1].collections[1].get_offsets()
    np.testing.assert_allclose(scatter, [[.4,.9]])
    for line in fig.axes[3].lines:
        x = np.asarray(line.get_xdata())
        if len(x) > 2:
            assert (x > 0).all()
    plt.close(fig)


def test_subgroup_small_and_empty_panels_have_room_for_margins():
    table = pd.DataFrame([dict(covariate='sex', subgroup='Female', contrast='R1',
                              gain=.01, lower=0., upper=.02)])
    styles = {'R1': ('purple', 'Router versus joint')}
    for data in (table, table.iloc[:0]):
        fig = figures.figure_subgroup_effects(
            data, contrast_styles=styles, covariate_order={'sex': 'Sex'}, xlabel='Gain')
        fig.canvas.draw()
        assert fig.subplotpars.bottom < fig.subplotpars.top
        if data.empty:
            assert any('No matching subgroup' in t.get_text() for t in fig.axes[0].texts)
        plt.close(fig)
        fig = figures.figure_who_benefits(
            data, data, contrast_styles=styles, covariate_order={'sex': 'Sex'})
        fig.canvas.draw()
        assert fig.subplotpars.bottom < fig.subplotpars.top
        plt.close(fig)


def test_atlas_matrix_keeps_joint_identity_when_opera_is_available():
    atlas = pd.DataFrame(dict(outcome=['anemia_g2plus'], horizon_days=[365],
        cif_auc=[.65], endpoint_group=['Erythroid findings and transfusion']))
    models = pd.DataFrame(dict(cohort=['DLBCL'], cohort_group=['DLBCL-like'],
        outcome=['anemia_g2plus'], endpoint_group=['Erythroid findings and transfusion'],
        horizon_days=[365], J=[.65], OPERA=[.9]))
    fig = figures.figure_atlas_matrix(atlas, models)
    np.testing.assert_allclose(fig.axes[0].images[0].get_array(), [[.65]])
    fig.canvas.draw()
    plt.close(fig)


def test_patient_map_honours_configured_highlight_order():
    coords, outcomes = proto.synthetic_gallery(np.random.default_rng(41), n=300)
    selected = outcomes.outcome.drop_duplicates().tolist()[-5:][::-1]
    outcomes['highlight_order'] = outcomes.outcome.map(dict(zip(selected, range(5))))
    fig = figures.figure_patient_map(coords, outcomes, grid_size=25,
                                     min_effective_support=1)
    assert [ax.get_title(loc='left') for ax in fig.axes[1:6]] == [
        f'{letter}  {figures._embedding_title(name, width=17)}'
        for letter, name in zip('BCDEF', selected)]
    # Default disease map communicates overlapping patients without KDE regions.
    assert all(type(collection).__name__ != 'QuadContourSet'
               for collection in fig.axes[0].collections)
    fig.canvas.draw()
    plt.close(fig)


def test_long_calibration_titles_fit_their_columns():
    coords, data = proto.synthetic_maps(np.random.default_rng(42), n=300)
    values = list(data.values())
    names = ['Infection diagnosis or antimicrobial use', 'Overall survival',
             'Anemia (G2+)', 'Hospital admission', 'Creatinine increased (G2+)']
    outcomes = {name: values[i % len(values)] for i, name in enumerate(names)}
    fig = figures.figure_calibration_maps(coords, outcomes, grid_size=25)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for ax in fig.axes[:3 * len(names):3]:
        bounds = ax.title.get_window_extent(renderer)
        assert bounds.width <= ax.get_window_extent(renderer).width + 5
    plt.close(fig)
