"""The faster restriction comparison preserves the original patient bootstrap."""
import numpy as np
import pandas as pd
import pytest

from opera.evaluation.metrics import compute_competing_risk_metrics_at_horizon
from opera.run import compare_input_restrictions as comparison


def reference_auc(frame, column, horizon):
    return compute_competing_risk_metrics_at_horizon(
        frame.time_days.to_numpy(float), frame.event.to_numpy(int),
        frame[column].to_numpy(float), horizon)["cif_auc"]


def predictions():
    rng = np.random.default_rng(42)
    n = 44
    parts = []
    for outcome in ("a", "b"):
        parts.append(pd.DataFrame(dict(
            subject_id=np.arange(n), cohort=np.where(np.arange(n) % 2, "D1", "D2"),
            outcome=outcome, time_days=rng.choice([0., 10., 30., 60., 100.], n),
            event=rng.choice([0, 1, 2], n), cif_60d=rng.choice([.1, .3, .7, .9], n))))
    return pd.concat(parts, ignore_index=True)


def test_all_restriction_estimates_and_intervals_match_original(monkeypatch):
    full = predictions()
    restricted = {"gap7": full.assign(cif_60d=full.cif_60d.to_numpy()[::-1]),
                  "gap28": full.assign(cif_60d=np.roll(full.cif_60d.to_numpy(), 7))}
    messages = []
    options = dict(horizon=60, draws=24, seed=9, domains={"a": "A", "b": "B"})
    actual = comparison.compare(full, restricted, progress=messages.append, **options)
    monkeypatch.setattr(comparison, "_auc", reference_auc)
    expected = comparison.compare(full, restricted, engine='expanded', **options)
    for fast, slow in zip(actual, expected):
        keys = ['restriction', 'outcome'] if 'outcome' in slow else ['restriction', 'group']
        pd.testing.assert_frame_equal(fast[slow.columns].sort_values(keys).reset_index(drop=True),
                                     slow.sort_values(keys).reset_index(drop=True),
                                     check_exact=False, rtol=1e-12, atol=1e-12)
    assert any("Restriction 2/2" in message for message in messages)
    assert any("outcome 2/2" in message for message in messages)
    assert messages[-1].endswith("24/24 draws complete")


@pytest.mark.parametrize("event", [0, 1, 2])
def test_auc_without_both_classes_remains_nan(event):
    frame = predictions().iloc[:4].assign(event=event, time_days=30.)
    assert np.isnan(comparison._auc(frame, "cif_60d", 60))


def test_auc_empty_resample_remains_nan():
    assert np.isnan(comparison._auc(predictions().iloc[:0], "cif_60d", 60))


def test_multi_horizon_reuses_exact_patient_draws_and_clinical_family_means():
    full = predictions().assign(cif_30d=lambda d: d.cif_60d / 2)
    restricted = {'gap7': full.assign(cif_60d=lambda d: d.cif_60d[::-1].to_numpy(),
                                     cif_30d=lambda d: np.roll(d.cif_30d, 3))}
    multi = comparison.compare(full, restricted, horizons=[30, 60], draws=24, domains={'a': 'A', 'b': 'B'})
    assert set(multi[1].group) == {'All outcomes', 'A', 'B'}
    for h in [30, 60]:
        single = comparison.compare(full, restricted, horizon=h, draws=24, domains={'a': 'A', 'b': 'B'})
        for a, b in zip(multi, single):
            pd.testing.assert_frame_equal(a.loc[a.horizon_days.eq(h)].reset_index(drop=True), b)


def test_bootstrap_absent_time_groups_and_risk_ties_match_expanded_rows():
    times = np.array([0., 10., 10., 30., 30., 50., 70.])
    events = np.array([0, 1, 2, 0, 1, 0, 1])
    risk = np.array([.1, .1, .2, .2, .7, .9, .9])
    counts = np.array([[1, 1, 1, 1, 1, 1, 1], [0, 0, 0, 1, 1, 2, 0], [3, 0, 0, 0, 0, 0, 0]])
    for horizon in [10, 30, 60]:
        weights = comparison._bootstrap_ipcw(times, events, counts, horizon)
        actual = comparison._rank_auc(risk, *weights)
        for i, draw in enumerate(counts):
            sample = np.repeat(np.arange(len(times)), draw)
            expected = comparison.fast_cif_auc(times[sample], events[sample], risk[sample], horizon)[0]
            np.testing.assert_allclose(actual[i], expected, equal_nan=True)


@pytest.mark.parametrize('change', ['label', 'time', 'missing_horizon', 'split'])
def test_restrictions_cannot_compare_different_labels_or_splits(change):
    full = predictions().assign(split='tuning')
    other = full.copy()
    if change == 'label':
        other.loc[0, 'event'] = (other.loc[0, 'event'] + 1) % 3
    elif change == 'time':
        other.loc[0, 'time_days'] += 1
    elif change == 'missing_horizon':
        other = other.drop(columns='cif_60d')
    else:
        other['split'] = 'held_out'
    with pytest.raises(ValueError):
        comparison.compare(full, {'gap': other}, horizon=60, draws=10)


def test_output_preserves_single_horizon_consumers_and_makes_paginated_recipe(tmp_path):
    full = predictions().assign(cif_365d=lambda d: d.cif_60d)
    tables = comparison.compare(full, {'Last 7 days hidden': full}, horizons=[60, 365], draws=12)
    comparison.write_outputs(*tables, tmp_path)
    assert set(pd.read_csv(tmp_path / 'input_restriction_summary.csv').horizon_days) == {365}
    assert set(pd.read_csv(tmp_path / 'input_restriction_summary_by_horizon.csv').horizon_days) == {60, 365}
    recipe = comparison.figure_recipe(tmp_path / 'input_restriction_summary_by_horizon.csv', [30, 60, 90, 180, 365, 730])
    assert len(recipe['figures']) == 4
    assert [p['options']['horizons'] for p in recipe['figures'][1:]] == [[30, 60], [90, 180], [365, 730]]


def test_completed_workspace_refresh_reuses_exports_and_only_calls_renderer(tmp_path, monkeypatch):
    import json
    from opera.run import refresh_recent_record_gaps as refresh
    workspace = tmp_path / 'analysis'
    workspace.mkdir()
    full = predictions().assign(split='tuning', cif_30d=lambda d: d.cif_60d / 2)
    source = tmp_path / 'full.parquet'
    full.to_parquet(source)
    (workspace / 'completion.json').write_text(json.dumps(dict(inputs=dict(fit=dict(predictions=str(source))))))
    for days in [7, 14, 21, 28]:
        path = workspace / f'input_restrictions/gap_{days}d'
        path.mkdir(parents=True)
        full.to_parquet(path / 'direct_cr_predictions.parquet')
    before = {str(p): p.read_bytes() for p in workspace.rglob('*') if p.is_file()}
    commands = []
    monkeypatch.setattr(refresh.subprocess, 'run', lambda command, **kwargs: commands.append(command))
    output = tmp_path / 'refreshed'
    monkeypatch.setattr('sys.argv', ['refresh', '--workspace', str(workspace), '--output-dir', str(output),
                                   '--horizons', '30', '60', '--draws', '12'])
    refresh.main()
    assert len(commands) == 1 and 'opera.run.render_paper_results' in commands[0]
    assert before == {str(p): p.read_bytes() for p in workspace.rglob('*') if p.is_file()}
    assert (output / 'recent_record_review.zip').is_file()
    assert set(pd.read_csv(output / 'input_restriction_auc_by_horizon.csv').horizon_days) == {30, 60}
