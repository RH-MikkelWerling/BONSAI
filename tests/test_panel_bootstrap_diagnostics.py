import numpy as np
import pandas as pd
from opera.evaluation.secondary_analysis import effect_summaries
from opera.run.build_paper_plot_tables import panel_contrasts


def test_strict_panel_failure_is_preserved_and_explained_in_plot_table():
    cells = pd.DataFrame([
        dict(split='tuning', seed=42, cohort='A', outcome=o, horizon_days=365,
             contrast='R1', metric='auc', gain=.01) for o in ['x', 'y']])
    draws = cells.drop(columns='gain').copy()
    draws['gain_draws'] = [np.r_[np.full(10, np.nan), np.full(10, .01)],
                          np.r_[np.full(10, .01), np.full(10, np.nan)]]
    metadata = cells[['cohort', 'outcome', 'horizon_days']]
    _, summary = effect_summaries(cells, draws, metadata)
    assert summary.n_valid_bootstrap.eq(0).all()
    assert summary.n_cells_unstable.eq(2).all()
    assert summary.minimum_finite_cells_per_draw.eq(1).all()
    assert summary.gain.eq(.01).all()
    assert summary.lower.isna().all()
    plotted = panel_contrasts(summary, ['R1'], seed=42)
    assert plotted.interval_status.eq('insufficient_draws').all()
    assert plotted.bootstrap_cell_policy.eq('strict_fixed_roster').all()


def test_horizon_pooling_rejects_missing_draws_instead_of_partial_interval():
    import pytest
    from opera.evaluation.secondary_analysis import pool_cells_over_horizons
    cells = pd.DataFrame([dict(split='tuning', seed=42, cohort='A', outcome='x',
                               horizon_days=h, contrast='R1', metric='auc', gain=.01)
                          for h in [30, 365]])
    draws = cells.iloc[:1].drop(columns='gain').copy()
    draws['gain_draws'] = [np.full(100, .01)]
    with pytest.raises(ValueError, match='Missing bootstrap cells'):
        pool_cells_over_horizons(cells, draws)


def test_conflicting_support_metadata_cannot_select_first_row():
    import pytest
    from opera.run.build_paper_plot_tables import _cell_groups
    metadata = pd.DataFrame([dict(cohort='A', outcome='x', horizon_days=365, n_events_train=n)
                             for n in [5, 500]])
    with pytest.raises(ValueError, match='Conflicting train-only metadata'):
        _cell_groups(metadata)
