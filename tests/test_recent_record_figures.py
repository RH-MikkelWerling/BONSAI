import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import pytest

from opera.visualization.paper_figures import figure_recent_record_gaps, figure_recent_record_gap_horizons
from opera.visualization.layout_audit import audit_layout


def summary():
    return pd.DataFrame([dict(restriction=f'Last {d} days hidden', group=family, horizon_days=h,
        mean_delta=-d / h, lower=-d / h - .01, upper=-d / h + .01)
        for h in [30, 60, 90, 180, 365, 730] for d in [7, 14, 21, 28]
        for family in ['All outcomes', 'Infection and antimicrobial use', 'Disease course and later malignancy']])


def test_horizon_overview_and_family_pages_have_distinct_panels_and_shared_scale():
    data = summary()
    overall = figure_recent_record_gap_horizons(data)
    a = figure_recent_record_gaps(data, horizons=[30, 60])
    b = figure_recent_record_gaps(data, horizons=[365, 730])
    assert len(overall.axes[0].get_xticklabels()) == 6
    assert len(a.axes) == 2
    assert a.axes[0].get_xlim() == a.axes[1].get_xlim() == b.axes[0].get_xlim()
    for fig in [overall, a, b]:
        assert not audit_layout(fig)['legend_overflow']
        plt.close(fig)


def test_single_horizon_legacy_tables_remain_renderable():
    data = summary().loc[lambda d: d.horizon_days.eq(365)].drop(columns='horizon_days')
    fig = figure_recent_record_gaps(data)
    assert len(fig.axes) == 1
    plt.close(fig)


def test_multiple_horizons_or_duplicate_estimates_cannot_be_silently_averaged():
    data = summary()
    with pytest.raises(ValueError, match='one or two'):
        figure_recent_record_gaps(data, horizons=[30, 60, 90])
    with pytest.raises(ValueError, match='Duplicate'):
        figure_recent_record_gaps(pd.concat([data, data]), horizons=[365])
