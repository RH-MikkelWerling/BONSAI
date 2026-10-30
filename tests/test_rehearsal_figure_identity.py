import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import pytest
from opera.visualization import paper_figures as figures


def headline_data():
    return pd.DataFrame([dict(cohort='DLBCL', outcome='survival',horizon_days=h,
        contrast=c, metric='auc',gain=g,lower=g-.01,upper=g+.01,n_events_train=50)
        for h in [0,365] for c,g in [('J-X',.1),('OPERA-X',-.1),('OPERA-S',.2),('OPERA-J',-.2)]])


def test_headline_uses_requested_model_and_blocks_old_clinical_export():
    models=pd.DataFrame([dict(cohort='DLBCL',outcome='survival',horizon_days=365,
                              J=.9,OPERA=.6,X=.8,n_events_train=50)])
    scores=pd.DataFrame([dict(label='NCCN-IPI', index_sensitivity=.4,model_sensitivity=.5)])
    fig=figures.figure_headline(headline_data(),models=models,scores=scores,
        focus_model='OPERA',require_training_scores=True,clinical_model='R_L',
        mark=r'TUNING SET \u2014 not test results')
    labels=[ax.get_ylabel() for ax in fig.axes]
    text=' '.join(t.get_text() for ax in fig.axes for t in ax.texts)
    assert 'OPERA AUC (one year)' in labels
    assert 'OPERA higher AUC in\n0 of 1 cells' in text
    assert 'Clinical refresh pending' in text
    assert fig.texts[0].get_text()=='TUNING SET — not test results'
    assert any(t.get_text()=='Joint model' for t in fig.axes[0].get_yticklabels())
    plt.close(fig)
    with pytest.raises(ValueError,match='lacks requested OPERA'):
        figures.figure_headline(headline_data(),models=models.drop(columns='OPERA'),focus_model='OPERA')
    plt.close('all')


def test_tiny_source_effects_do_not_receive_full_contrast_or_lineage_claims():
    rows=[]
    for target in ['CLL','DLBCL','MM']:
        for source in ['CLL','DLBCL','MM']:
            rows.append(dict(target=target,source=source,endpoint_group='All outcomes',
                target_group='A',source_group='A',effect=.00001,lower=-.001,upper=.001,
                raw_effect=.1,raw_lower=-.3,raw_upper=.4,n_cells=40))
    fig=figures.figure_borrowing_map(pd.DataFrame(rows))
    image=fig.axes[0].images[0]
    assert image.norm.vmax>=1
    assert any('Size of individual-source effects' in ax.get_title(loc='left') for ax in fig.axes)
    plt.close(fig)


def test_raw_borrowing_display_reports_absolute_raw_effect_distribution():
    rows=[]
    for target in ['CLL','DLBCL','MM']:
        for source in ['CLL','DLBCL','MM']:
            rows.append(dict(target=target,source=source,endpoint_group='All outcomes',
                target_group='A',source_group='A',effect=.00001,lower=-.001,upper=.001,
                raw_effect=.1,raw_lower=-.3,raw_upper=.4,n_cells=40))
    fig=figures.figure_borrowing_map(pd.DataFrame(rows),pair_estimate='raw',pair_vmax=2.)
    assert fig.axes[0].images[0].norm.vmax == 2.
    panel=next(ax for ax in fig.axes if 'Size of individual-source effects' in ax.get_title(loc='left'))
    magnitude=panel.lines[0].get_xdata()
    assert all(abs(float(x)-.1)<1e-10 for x in magnitude[1:])
    assert panel.lines[0].get_ydata()[-1] == 1
    plt.close(fig)


def test_source_removal_map_retains_sources_without_target_rows():
    rows=[]
    for target in ['CLL','DLBCL']:
        for source in ['CLL','DLBCL','NMZL']:
            rows.append(dict(target=target,source=source,endpoint_group='All outcomes',
                target_group='A',source_group='0' if source=='NMZL' else 'A',
                effect=.001,lower=-.01,upper=.01,raw_effect=.1,
                raw_lower=-.3,raw_upper=.4,n_cells=40))
    fig=figures.figure_borrowing_map(pd.DataFrame(rows),pair_estimate='raw')
    ax=fig.axes[0]
    assert ax.images[0].get_array().shape == (2,3)
    assert [t.get_text() for t in ax.get_xticklabels()] == ['NMZL','CLL','DLBCL']
    assert [t.get_text() for t in ax.get_yticklabels()] == ['CLL','DLBCL']
    mask=ax.images[0].get_array().mask
    assert not mask[:,0].any()
    assert mask[0,1] and mask[1,2]
    assert figures._borrowing_order(pd.DataFrame(rows))[1] == ['0','A']
    plt.close(fig)
