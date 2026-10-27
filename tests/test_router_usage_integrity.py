import numpy as np
import pandas as pd
import pytest
from opera.visualization.paper_figures import borrowed_source_breadth, _route_weights


def test_borrowed_breadth_excludes_own_sources_and_shared_slot():
    rows = pd.DataFrame({'cohort': ['A']*5, 'outcome': ['x']*5, 'target_group': ['G']*5,
                         'bank': ['shared','disease','disease','outcome','outcome'],
                         'source': ['shared','A','B','x','y'], 'w': [.1,.4,.1,.3,.1]})
    result = borrowed_source_breadth(rows)
    assert result.gain.iloc[0] == pytest.approx(2.)
    rows.loc[rows.source=='y','w'] = 0
    assert borrowed_source_breadth(rows).gain.iloc[0] == pytest.approx(1.)
    rows.loc[rows.source=='B','w'] = 0
    assert np.isnan(borrowed_source_breadth(rows).gain.iloc[0])


def test_router_usage_does_not_mix_model_summaries():
    rows = pd.DataFrame({'model':['R','J'], 'cohort':['A','A'], 'outcome':['x','x'],
                         'bank':['shared','shared'], 'source':['shared','shared'], 'mean':[.1,.2]})
    with pytest.raises(ValueError, match='single model'):
        _route_weights(rows)


def test_borrowing_view_preserves_units_and_rejects_duplicate_pairs():
    from opera.visualization.paper_figures import _borrowing_plot_table
    table = pd.DataFrame({'target':['A'], 'source':['B'], 'endpoint_group':['All outcomes'],
                          'effect':[.0005], 'lower':[-.001], 'upper':[.001],
                          'raw_effect':[1.5], 'raw_lower':[-1.], 'raw_upper':[4.]})
    raw = _borrowing_plot_table(table, 'raw')
    assert raw.effect.iloc[0] == 1.5  # Already AUC points: no second x100.
    assert table.effect.iloc[0] == .0005
    with pytest.raises(ValueError, match='Duplicate borrowing'):
        _borrowing_plot_table(pd.concat([table, table]), 'shrunk')
