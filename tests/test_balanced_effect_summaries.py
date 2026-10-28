import numpy as np
import pandas as pd
from opera.evaluation.secondary_analysis import effect_summaries
from opera.run.resummarise_paired_analysis import rebuild


def inputs():
    rows = pd.DataFrame([dict(split='tuning', seed=42, contrast='R1', metric=metric,
        cohort=d, outcome=o, horizon_days=h, gain=g)
        for metric in ['auc', 'brier', 'ipa']
        for d, o, h, g in [('A', 'one', 30, 0.), ('A', 'one', 365, .2),
                            ('A', 'two', 365, 0.), ('B', 'one', 365, .3)]])
    draws = rows.drop(columns='gain').assign(gain_draws=[
        value + np.linspace(-.01, .01, 20) for value in rows.gain])
    metadata = rows[['cohort', 'outcome', 'horizon_days']].drop_duplicates()
    return rows, draws, metadata


def test_balanced_mean_uses_same_roster_and_shared_draws():
    rows, draws, metadata = inputs()
    for scheme, expected in [('equal_cell', .125), ('disease_balanced', .175)]:
        _, result = effect_summaries(rows, draws, metadata, aggregation=scheme)
        pooled = result.loc[(result.horizon_days == 0) & (result.stratum == 'all_cells')]
        assert len(pooled) == 3
        np.testing.assert_allclose(pooled.gain, expected)
        np.testing.assert_allclose(pooled.lower, expected - .0095)
        np.testing.assert_allclose(pooled.upper, expected + .0095)
        assert set(pooled.n_cells) == {4}
        assert set(pooled.n_diseases) == {2}


def test_undefined_draws_do_not_renormalize_task_weights():
    rows, draws, metadata = inputs()
    # All task draws undefined except one replicate: cannot obtain a finite CI
    # by silently changing the task mixture in the other replicates.
    draws.at[0, 'gain_draws'] = np.array([0.] + [np.nan] * 19)
    _, result = effect_summaries(rows, draws, metadata, aggregation='disease_balanced')
    part = result.loc[(result.metric == 'auc') & (result.horizon_days == 0)
                      & (result.stratum == 'all_cells')].iloc[0]
    assert np.isclose(part.gain, .175)
    assert np.isnan(part.lower)
    assert part.minimum_finite_cells_per_draw == 3


def test_rebuild_saved_analysis_without_rescoring(tmp_path):
    rows, draws, _ = inputs()
    source = tmp_path / 'source'
    source.mkdir()
    rows.to_csv(source / 'contrasts.csv', index=False)
    draws.drop(columns='gain_draws').to_csv(source / 'paired_bootstrap_index.csv', index=False)
    np.savez_compressed(source / 'paired_bootstrap_draws_INTERNAL.npz', gains=np.stack(draws.gain_draws))
    result = rebuild(source, tmp_path / 'summaries')
    assert set(result) == {'effect_summaries.csv', 'effect_summaries_disease_balanced.csv'}
    output = pd.read_csv(tmp_path / 'summaries/paired_mean_comparison.csv')
    assert set(output.weighting) == {'Equal cell', 'Disease balanced'}
    assert set(output.metric) == {'auc', 'brier', 'ipa'}
