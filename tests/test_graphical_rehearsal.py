import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from opera.visualization import paper_figures as f
from opera.visualization.style import align_panel_titles
from opera.visualization.layout_audit import audit_layout


def test_long_panel_titles_fit_without_reducing_font_size():
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3))
    axes[0].set_title('A  National pretraining and disease-specific corrections', loc='left')
    axes[1].set_title('B  Another panel', loc='left')
    fontsize = axes[0]._left_title.get_fontsize()
    align_panel_titles(fig)
    fig.canvas.draw()
    assert '\n' in axes[0].get_title(loc='left')
    assert axes[0]._left_title.get_fontsize() == fontsize
    assert axes[0]._left_title.get_window_extent().x1 < axes[1].get_window_extent().x0
    plt.close(fig)


def test_threshold_ties_in_plot_match_exported_binary_metrics():
    fig, ax = plt.subplots()
    known = pd.DataFrame(dict(model_risk=[.4,.4,.2,.2],ipi_score=[6,6,0,0]))
    summary=f._clinical_precision_recall_panel(ax,known,pd.Series([1,0,1,0]),
        ipi_label='NCCN-IPI',ipi_high=6,model_threshold=.4)
    assert 'recall 0.50, precision 0.50, specificity 0.50' in summary[-1]
    plt.close(fig)


def test_single_score_and_empty_high_group_do_not_crash_precision_recall():
    for labels in [pd.Series([1,0,1,0]),pd.Series([0,0,0,0])]:
        fig, ax = plt.subplots()
        known=pd.DataFrame(dict(model_risk=[.1,.2,.3,.4],ipi_score=[0,0,0,0]))
        summary=f._clinical_precision_recall_panel(ax,known,labels,
            ipi_label='NCCN-IPI',ipi_high=6,model_threshold=.5)
        assert 'No patients' in summary[0] or 'unavailable' in summary[0]
        plt.close(fig)


def test_audit_finds_actual_text_overlap_but_ignores_nonvisible_ticks():
    fig, ax=plt.subplots()
    ax.text(.3,.5,'overlap one');ax.text(.3,.5,'overlap two')
    ax.set_xlim(0,1);ax.set_xticks([-10,0,1,10]);ax.set_xlim(0,1)
    result=audit_layout(fig)
    assert any(x['first']=='overlap one' and x['second']=='overlap two'
               for x in result['text_collision_candidates'])
    assert not any('-10' in (x['first'],x['second']) for x in result['text_collision_candidates'])
    plt.close(fig)


def test_metric_placeholders_are_not_counted_as_model_fits():
    from opera.run.build_paper_numbers import coverage_numbers
    data=pd.DataFrame([dict(model=m,split='held_out',horizon_days=365,metric='auc',
                           cohort='A',outcome=o,estimate=value)
                       for m in ['J_L','S_L','D_L','O_L'] for o,value in [('a',.7),('b',np.nan)]])
    numbers=coverage_numbers(data,None)
    assert numbers['cells_evaluable']=='1'
    assert numbers['evaluable_specialist_tasks']=='1'
    assert 'narrow_fits' not in numbers


def test_old_clinical_numbers_cannot_silently_enter_manuscript():
    from opera.run.build_paper_numbers import clinical_numbers
    old=pd.DataFrame([dict(manuscript=True,extra_events_flagged_per_1000=20,
                           model_low_per_1000=300,index_low_per_1000=100)])
    assert clinical_numbers(old,None,None)=={}


def test_headline_model_roster_uses_requested_horizon():
    # A national model exported only at 30 days must not empty a 365-day roster.
    models=pd.DataFrame(dict(horizon_days=[365,365,30],J=[.7,.8,.7],
                             OPERA=[.72,.82,.72],R_E=[np.nan,np.nan,.85]))
    fig,ax=plt.subplots()
    f._headline_model_overview(ax,models)
    assert [t.get_text() for t in ax.get_yticklabels()]==['Joint','OPERA (local)']
    assert '2 matched tasks' in ax.get_xlabel()
    plt.close(fig)


def test_router_bank_allocation_preserves_actual_source_counts():
    from opera.run.render_paper_figure_prototypes import synthetic_routes
    routes,disease,outcomes=synthetic_routes(np.random.default_rng(2))
    total = 1 + len(disease) + len(outcomes)
    routes['mean']=1 / total
    fig=f.figure_router_usage(routes,disease,outcomes)
    allocation=next(ax for ax in fig.axes if ax.get_title(loc='left').startswith('C '))
    np.testing.assert_allclose([bar.get_width() for bar in allocation.containers[0]],
                               (len(disease)-1)/total)
    np.testing.assert_allclose([bar.get_width() for bar in allocation.containers[1]],
                               (len(outcomes)-1)/total)
    np.testing.assert_allclose([bar.get_width() for bar in allocation.containers[2]],3/total)
    plt.close(fig)


def test_router_usage_keeps_sources_without_estimable_target_rows():
    from opera.run.render_paper_figure_prototypes import synthetic_routes
    routes, disease, outcomes = synthetic_routes(np.random.default_rng(2))
    removed = 'CLL'
    routes = routes.loc[routes.cohort.ne(removed)]
    fig = f.figure_router_usage(routes, disease, outcomes)
    ax = fig.axes[0]
    assert removed in [label.get_text() for label in ax.get_xticklabels()]
    assert removed not in [label.get_text() for label in ax.get_yticklabels()]
    matrix = ax.images[0].get_array()
    assert matrix.shape == (len(disease)-1, len(disease))
    # Every visible cell retains its original within-bank share.
    target = ax.get_yticklabels()[0].get_text()
    source_index = [label.get_text() for label in ax.get_xticklabels()].index(removed)
    expected = routes.loc[routes.cohort.eq(target) & routes.bank.eq('disease')].copy()
    expected['share'] = expected['mean'] / expected.groupby('outcome')['mean'].transform('sum')
    assert np.isclose(matrix[0, source_index], expected.loc[expected.source.eq(removed), 'share'].mean())
    plt.close(fig)


def test_full_calibration_slope_axis_keeps_extreme_intervals():
    rows = pd.DataFrame(dict(q25=[-.1, .6], median=[1., .9], q75=[300., 1.4]))
    fig, ax = plt.subplots()
    f._full_slope_axis(ax, rows, 'Slope')
    assert ax.get_xlim()[0] < -.1 and ax.get_xlim()[1] > 300.
    assert ax.get_xscale() == 'symlog'
    assert 'all IQRs shown' in ax.get_xlabel()
    plt.close(fig)


def test_strategy_bars_include_joint_and_opera_and_do_not_drop_probability_mass():
    cells = pd.DataFrame(dict(horizon_days=[0], specialist=[.7], joint=[.75],
                              best_tuning=[.72], n_events_train=[50]))
    selection = pd.DataFrame(dict(endpoint_group=['G']*5,
        selected_model=['S','D','O','J','OPERA'], candidate_models=['S_L|D_L|O_L|J_L|OPERA']*5))
    fig = f.figure_strategy(cells, selection)
    bars = fig.axes[-1]
    assert bars.get_legend_handles_labels()[1][-2:] == ['Joint (20% overall)','OPERA (20% overall)']
    assert np.isclose(sum(container[0].get_width() for container in bars.containers), 1.)
    plt.close(fig)


def test_support_keeps_unknown_counts_out_of_cumulative_share():
    cells=pd.DataFrame(dict(cohort=['A','A','A'],cohort_group=['A']*3,
        outcome=['x','y','z'],endpoint_group=['G']*3,horizon_days=[365]*3,
        n_events_train=[0,10,np.nan]))
    fig=f.figure_support(cells)
    curve=fig.axes[-1].lines[0]
    np.testing.assert_allclose(curve.get_ydata(),[.5,1])
    assert any('1 counts unavailable' in t.get_text() for t in [*fig.axes[-1].texts,*fig.texts])
    plt.close(fig)


def test_provisional_clinical_mappings_cannot_enter_held_out_summary(tmp_path):
    import json,pytest
    from opera.run.summarise_attributions import run
    (tmp_path/'attribution_run.json').write_text(json.dumps(dict(split='held_out')))
    with pytest.raises(ValueError,match='restricted to the tuning rehearsal'):
        run(tmp_path,dict(concept_mapping_reviewed=False,allow_provisional_concepts=True),tmp_path/'out')


def test_support_uses_full_one_year_grid_even_when_pooled_rows_are_filtered():
    cells=pd.DataFrame(dict(cohort=['A','A','A'],cohort_group=['A']*3,
        outcome=['x','x','y'],endpoint_group=['G']*3,horizon_days=[0,365,365],
        n_events_train=[100,100,0]))
    fig=f.figure_support(cells)
    np.testing.assert_allclose(fig.axes[-1].lines[0].get_xdata(),[.8,100])
    assert len(fig.axes[0].images[0].get_array().ravel())==2
    plt.close(fig)


def test_support_retains_absent_disease_and_separates_not_applicable():
    cells=pd.DataFrame(dict(cohort=['DLBCL'],cohort_group=['DLBCL-like'],
        outcome=['x'],endpoint_group=['G'],horizon_days=[365],n_events_train=[10]))
    coverage=pd.DataFrame(dict(cohort=['DLBCL','DLBCL','NMZL','NMZL'],
        outcome=['x','y','x','y'],endpoint_group=['G']*4,
        status=['evaluated','not_applicable','fit_failed','no_training_events']))
    fig=f.figure_support(cells,coverage=coverage)
    assert fig.axes[0].images[0].get_array().shape==(2,2)
    assert any('2 counts unavailable' in t.get_text() for t in [*fig.axes[-1].texts,*fig.texts])
    plt.close(fig)


def test_attribution_odds_ratio_display_rejects_probability_scale():
    import pytest
    concepts=pd.DataFrame(dict(scale=['risk']))
    with pytest.raises(ValueError,match='require logit-scale'):
        f.figure_attribution_atlas(pd.DataFrame(),pd.DataFrame(),concepts)
