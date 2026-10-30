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
        routes.loc[routes.cohort.isin(keep)], disease_groups, outcome_groups))
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
                                            min_effective_support=1, mode="observed_surface")
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
    assert [' '.join(ax.get_title(loc='left').split()) for ax in fig.axes[1:6]] == [
        f'{letter} ' + ' '.join(figures._embedding_title(name, width=17).split())
        for letter, name in zip('BCDEF', selected)]
    # The selected design overlays disease contours on the observed patients.
    assert any(type(collection).__name__ == 'QuadContourSet'
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


def test_gallery_default_uses_final_risk_points():
    coords, outcomes = proto.synthetic_gallery(np.random.default_rng(55), n=100)
    fig = figures.figure_embedding_gallery(coords, outcomes, per_page=1)
    displayed = fig.axes[0].collections[0]
    name = outcomes[['outcome', 'domain_order']].drop_duplicates().sort_values(
        ['domain_order', 'outcome']).outcome.iloc[0]
    expected = outcomes.loc[outcomes.outcome == name, 'risk'].to_numpy()
    np.testing.assert_allclose(displayed.get_array(), expected)
    assert displayed.norm.vmin == 0 and displayed.norm.vmax == 1
    plt.close(fig)


def test_horizons_uses_saved_paired_means_with_independent_scales():
    cells = proto.synthetic_cells(np.random.default_rng(56))
    cells['group_mean'] = .012
    cells['group_lower'] = .005
    cells['group_upper'] = .019
    fig = figures.figure_horizons(cells)
    assert all('(mean, 95% CI)' in ax.get_xlabel() for ax in fig.axes)
    assert not fig.axes[0].get_shared_x_axes().joined(fig.axes[0], fig.axes[-1])
    plt.close(fig)


def test_headline_history_accepts_real_covariate_and_contrast_codes():
    rng = np.random.default_rng(57)
    cells = proto.synthetic_cells(rng)
    subgroups = pd.DataFrame([
        dict(covariate='history_event_count', subgroup=label, contrast=contrast,
             gain=.001, lower=0., upper=.002)
        for label in ['≤ 99', '100–999'] for contrast in ['P5', 'T2']])
    fig = figures.figure_headline(cells, subgroups=subgroups)
    panel = next(ax for ax in fig.axes if ax.get_xlabel() == 'Recorded events before treatment')
    assert len(panel.containers) == 2
    assert len(panel.get_xticklabels()) == 2
    plt.close(fig)


def test_performance_forest_uses_saved_domain_means_and_ci():
    frame = pd.DataFrame(dict(endpoint_group=['a', 'a'], contrast=['J-S', 'J-S'],
        gain=[-.1, .1], group_mean=[.03, .03], group_lower=[.01, .01], group_upper=[.05, .05]))
    summary = figures._summary_rows(frame, ['endpoint_group', 'contrast'])
    assert summary['median'].iloc[0] == .03
    assert summary.q25.iloc[0] == .01 and summary.q75.iloc[0] == .05
    fig, ax = plt.subplots()
    figures._diamond_forest(ax, summary, row_key='endpoint_group', row_order=['a'],
        series_key='contrast', series_style={'J-S': ('blue', 'Joint vs specialist')},
        xlabel='AUC gain (median, IQR)')
    assert ax.get_xlabel() == 'AUC gain (mean, 95% CI)'
    plt.close(fig)


def test_patient_map_default_colours_final_predictions():
    coords, outcomes = proto.synthetic_gallery(np.random.default_rng(58), n=100)
    fig = figures.figure_patient_map(coords, outcomes, grid_size=20)
    assert not fig.axes[1].images
    assert fig.axes[1].collections[0].get_array() is not None
    plt.close(fig)


def test_horizon_router_domain_labels_preserve_injected_effects():
    cells = proto.synthetic_cells(np.random.default_rng(60))
    target = 'Disease course and later malignancy'
    cardio = 'Cardiovascular and thrombotic events'
    cells['group_mean'] = np.where(cells.endpoint_group == target, .04,
                                  np.where(cells.endpoint_group == cardio, -.02, .01))
    cells['group_lower'] = cells.group_mean - .005
    cells['group_upper'] = cells.group_mean + .005
    fig = figures.figure_horizons(cells)
    labels = [t.get_text() for t in fig.axes[0].get_yticklabels()]
    for domain, expected in [('Disease course', .04), ('Cardiovascular', -.02)]:
        y = labels.index(domain)
        points = np.concatenate([c.get_offsets() for c in fig.axes[-1].collections])
        selected = points[np.abs(points[:, 1] - y) < .3]
        assert len(selected) == 6
        np.testing.assert_allclose(selected[:, 0], expected)
    plt.close(fig)
