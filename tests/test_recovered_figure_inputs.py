import pandas as pd
import pytest

from opera.run.build_paper_plot_tables import Missing, origin_estimates, strategy_tables


def test_unselected_strategies_are_missing_not_a_string_accessor_failure():
    estimates = pd.DataFrame([dict(split='held_out', seed=42, cohort='DLBCL',
                                  outcome='overall_survival', horizon_days=365,
                                  model=model, metric='auc', estimate=.7)
                             for model in ['S_L', 'J_L']])
    metadata = pd.DataFrame([dict(cohort='DLBCL', outcome='overall_survival',
                                  horizon_days=365, n_events_train=100)])
    selection = pd.DataFrame([dict(seed=42, cohort='DLBCL', outcome='overall_survival',
                                   status='missing_tuning_candidates', selected_model=float('nan'))])
    with pytest.raises(Missing, match='No tuning-selected'):
        strategy_tables(estimates, selection, metadata, seed=42)


def test_origin_panel_uses_own_correction_alias_and_single_common_roster():
    rows = [dict(split='held_out', seed=42, model=model, group=group, metric='auc',
                 horizon_days=365, estimate=.7, lower=.6, upper=.8, n_cells=988)
            for group in ['encoder_origin', 'encoder_origin_routing']
            for model in ['J_L', 'J_OA_OWN_L', 'J_EA', 'R_EA']]
    result = origin_estimates(pd.DataFrame(rows), seed=42)
    assert len(result) == 4
    assert len(result.loc[result.origin.eq('L') & result.readout.eq('R')]) == 1


def test_horizon_plate_uses_opera_joint_in_fourth_panel():
    import matplotlib.pyplot as plt
    from opera.visualization.paper_figures import figure_horizons
    rows = [dict(contrast=contrast, metric='auc', horizon_days=horizon,
                 endpoint_group='Mortality', gain=.01, group_mean=.01,
                 group_lower=.005, group_upper=.015)
            for contrast in ['D-S', 'O-S', 'J-S', 'OPERA-J']
            for horizon in [30, 365]]
    fig = figure_horizons(pd.DataFrame(rows))
    assert 'OPERA' in fig.axes[3].get_title(loc='left')
    assert len(fig.axes[3].collections) > 0
    plt.close(fig)
