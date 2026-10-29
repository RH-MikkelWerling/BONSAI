import numpy as np
import pandas as pd
import pytest

from opera.evaluation.metrics import compute_competing_risk_metrics_at_horizon
from opera.run.audit_tuning_router import reconcile_metrics


def test_explicit_tabular_reconciliation_includes_censored_risk_set_and_detects_errors(tmp_path):
    metadata = tmp_path / 'metadata.parquet'
    pd.DataFrame({'subject_id': ['a', 'b', 'c', 'd']}).to_parquet(metadata)
    # Case, horizon survivor, competing death and early administrative censor.
    frame = pd.DataFrame({
        'subject_id': ['a', 'b', 'c', 'd'], 'cohort': 'A', 'outcome': 'event',
        'horizon_days': 30, 'split': 'tuning', 'seed': 42, 'data_contract': 'v1',
        'time_days': [10., 30., 15., 5.], 'event': [1, 0, 2, 0],
        'label': [1, 0, 0, 0], 'known': [True, True, True, False],
        'probability': [.8, .2, .3, 0.],
        'prediction_absent_zero_weight': [False, False, False, True],
    })
    predictions = tmp_path / 'tabular.parquet'
    frame.to_parquet(predictions)
    native = compute_competing_risk_metrics_at_horizon(
        frame.time_days.to_numpy(), frame.event.to_numpy(),
        frame.probability.to_numpy(), 30)
    analysis = tmp_path / 'prediction_analysis'
    analysis.mkdir()
    saved = pd.DataFrame([
        {'split': 'tuning', 'seed': 42, 'cohort': 'A', 'outcome': 'event',
         'horizon_days': 30, 'model': 'X', 'metric': metric,
         'estimate': native['cif_' + metric] + (.1 if metric == 'brier' else 0.)}
        for metric in ('auc', 'brier')
    ])
    saved.to_csv(analysis / 'estimates.csv', index=False)
    config = {'patient_metadata': str(metadata), 'estimand': 'cif_ipcw',
              'artifacts': [{'path': str(predictions), 'model': 'X', 'split': 'tuning'}]}
    assert reconcile_metrics(config, tmp_path) == []  # Neural-only default.
    report, = reconcile_metrics(config, tmp_path, models=['X'])
    assert report['recomputed_metrics'] == 2
    assert report['missing_saved_estimates'] == 0
    assert report['finite_status_mismatches'] == 0
    assert report['native_vs_analysis_finite_status_mismatches'] == 0
    assert report['native_vs_analysis_max_abs_error'] < 1e-12
    assert np.isclose(report['recomputed_vs_saved_max_abs_error'], .1)
    with pytest.raises(ValueError, match='no tuning artifacts'):
        reconcile_metrics(config, tmp_path, models=['X', 'XP'])

    # OPERA normally exists only as an analysis alias. Resolve the selected
    # physical model without overriding the long table's declared model.
    frame.assign(model='R_L').to_parquet(predictions)
    saved.assign(model='OPERA').to_csv(analysis / 'estimates.csv', index=False)
    config['opera_model'] = 'R_L'
    config['artifacts'][0]['model'] = 'R_L'
    alias, = reconcile_metrics(config, tmp_path, models=['OPERA'])
    assert alias['model'] == 'OPERA'
    assert alias['source_model'] == 'R_L'
    assert alias['missing_saved_estimates'] == 0
    assert alias['finite_status_mismatches'] == 0
    assert np.isclose(alias['recomputed_vs_saved_max_abs_error'], .1)
