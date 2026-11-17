import numpy as np
import pandas as pd
import pytest
from opera.evaluation.headline_holm import HEADLINE_FAMILY, headline_holm


def summaries():
    return pd.DataFrame([dict(contrast=c, split='held_out', seed=42, metric='auc',
        horizon_days=0, stratum='all_cells', aggregation='equal_cell_not_population_or_endpoint_group_macro',
        p_value=p) for c, p in zip(HEADLINE_FAMILY, [.001, .01, .02, .03, .04, .1, .2, .9])])


def test_holm_matches_known_stepdown_and_excludes_secondary_analyses():
    rows=summaries()
    other=rows.assign(seed=43, p_value=0)
    result=headline_holm(pd.concat([rows, other]), splits=['held_out'])
    np.testing.assert_allclose(result.p_value_holm, [.008,.07,.12,.15,.16,.3,.4,.9])
    assert result.reject_holm.tolist() == [True] + [False] * 7
    assert result.family_complete.all()


def test_missing_national_does_not_shrink_family():
    result=headline_holm(summaries().query("contrast != 'E1'"), splits=['held_out'])
    assert len(result)==8
    assert not result.family_complete.any()
    assert result.iloc[0].p_value_holm == pytest.approx(.008)
    assert np.isnan(result.set_index('contrast').loc['E1','p_value_holm'])


def test_bayesian_sign_tails_are_not_adjusted_as_frequentist_tests():
    result=headline_holm(summaries(), splits=['held_out'], bootstrap_method='bayesian')
    assert result.p_value_holm.isna().all()
    assert not result.reject_holm.any()


def test_zero_simulated_tail_has_explicit_finite_resolution_floor():
    rows=summaries().assign(n_bootstrap_replicates=1000)
    rows.loc[0,'p_value']=0
    result=headline_holm(rows, splits=['held_out'])
    assert result.iloc[0].p_value_raw==0
    assert result.iloc[0].p_value_for_holm==pytest.approx(2/1001)
    assert result.iloc[0].p_value_holm==pytest.approx(16/1001)
    with pytest.raises(ValueError,match='Duplicate'):
        headline_holm(pd.concat([rows, rows]), splits=['held_out'])
