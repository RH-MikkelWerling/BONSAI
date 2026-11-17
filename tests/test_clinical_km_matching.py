import numpy as np
import pandas as pd
import pytest

from opera.evaluation.clinical_km_matching import _prefix_curve, cohort_km_matching
from opera.evaluation.clinical_risk_groups import km_survival


def sample():
    return pd.DataFrame(dict(model_risk=np.arange(12) / 12,
        time_days=[500, 500, 500, 80, 500, 500, 100, 500, 500, 50, 500, 500],
        event=[0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0],
        ipi_group=['Low', 'High', 'Low', 'High', 'Low', 'High'] * 2))


def spec():
    return dict(id='index', horizon_days=365, death_in_endpoint=True, group_order=['Low', 'High'])


def test_every_prefix_km_agrees_with_direct_km_including_ties_and_censoring():
    data = sample()
    data.loc[2:4, 'model_risk'] = .3
    data.loc[1, 'time_days'] = 40  # early censoring matters, not binary event fractions
    for high in [False, True]:
        curve = _prefix_curve(data.model_risk.to_numpy(), data.time_days.to_numpy(),
                              data.event.to_numpy(), 365, high=high)
        for row in curve.itertuples():
            selected = data.model_risk.ge(row.cutoff) if high else data.model_risk.le(row.cutoff)
            assert selected.sum() == row.model_n
            assert row.model_survival == pytest.approx(km_survival(
                data.loc[selected, 'time_days'], data.loc[selected, 'event'], 365))


def test_both_ends_select_largest_feasible_prefix_and_reestimate_bootstrap_target():
    data = sample()
    result, curves = cohort_km_matching(data, spec(), draws=100, minimum_group=1, minimum_at_risk=1)
    for row in result.itertuples():
        curve = curves.loc[curves.end.eq(row.end)]
        feasible = curve.loc[curve.supported & curve.meets_target]
        assert row.model_n == feasible.model_n.max()
        assert row.additional_n == row.model_n - row.index_n
        assert row.grouping_uses_future_outcomes
        assert row.matching_draw_fraction <= 1
        assert row.index_survival_lower < row.index_survival_upper
    assert set(result.bootstrap_rule) == {'paired_patients_reestimate_index_KM_and_reselect_model_prefix'}
    repeated, _ = cohort_km_matching(data, spec(), draws=100, minimum_group=1, minimum_at_risk=1)
    pd.testing.assert_frame_equal(result, repeated)


def test_insufficient_followup_is_not_reported_as_a_match_or_zero_gain():
    data = sample().assign(time_days=20, event=0)
    result, _ = cohort_km_matching(data, spec(), draws=10, minimum_group=1, minimum_at_risk=1)
    assert set(result.status) == {'insufficient_native_followup'}
    assert result.valid_matching_draws.eq(0).all()
    assert result.lower.isna().all()
    assert 'additional_n' not in result


def test_no_feasible_match_and_competing_risk_guard():
    data = sample().assign(model_risk=1.)
    # Clinical Low has no events; the tied full cohort cannot meet that target.
    data['ipi_group'] = np.where(data.event.eq(0), 'Low', 'High')
    result, _ = cohort_km_matching(data, spec(), draws=10, minimum_group=1, minimum_at_risk=1)
    assert result.set_index('end').loc['low', 'status'] == 'no_supported_match'
    with pytest.raises(ValueError, match='CIF'):
        cohort_km_matching(data, {**spec(), 'death_in_endpoint': False})
    with pytest.raises(ValueError, match='event 0/1'):
        cohort_km_matching(data.assign(event=2), spec(), draws=1)


def test_frontier_figure_handles_both_missing_and_estimable_matches():
    from opera.visualization.paper_figures import figure_evaluation_km_matching
    import matplotlib.pyplot as plt
    for data in [sample(), sample().assign(time_days=20, event=0)]:
        matching, curve = cohort_km_matching(data, spec(), draws=2, minimum_group=1, minimum_at_risk=1)
        fig = figure_evaluation_km_matching(curve, matching)
        fig.canvas.draw()
        assert len(fig.axes) == 2
        plt.close(fig)
