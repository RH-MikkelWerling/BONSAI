import json
import sys

import numpy as np
import pandas as pd
import pytest
import yaml

from opera.evaluation.secondary_analysis import metric_values, normalize_predictions
from opera.run.analyse_locked_predictions import read_artifact
from opera.run.evaluate_whole_cohort_outcomes import evaluate, main, population_diagnostics


def predictions():
    rows = []
    # Neither disease has 5 cases AND 5 controls. Pooling must retain both.
    # Disease identity alone discriminates across diseases, but not within them.
    for model in ('R_L', 'J_L'):
        for disease, events, risk in [('A', 3, .1), ('B', 27, .9)]:
            for patient in range(30):
                event = int(patient < events)
                rows.append(dict(model=model, cohort=disease, subject_id=f'{disease}{patient:02}',
                    outcome='event', horizon_days=365, split='tuning', seed=42,
                    data_contract='v1', event=event, time_days=100. if event else 500.,
                    probability=risk if model == 'R_L' else .5))
    return pd.DataFrame(rows)


def test_pooled_discrimination_retains_small_cells_and_exposes_case_mix():
    result = evaluate(predictions(), ['R_L', 'J_L'], replicates=20)
    auc = result['common_population_estimates'].query("metric == 'auc'").set_index('model')
    assert auc.loc['R_L', 'estimate'] == pytest.approx(.9)
    assert auc.loc['J_L', 'estimate'] == pytest.approx(.5)
    assert (auc.n == 60).all()
    gain = result['paired_contrasts'].query("metric == 'auc'").iloc[0]
    assert gain.gain == pytest.approx(.4)
    assert np.isfinite(gain.lower) and np.isfinite(gain.upper)
    assert (result['model_coverage'].available_diseases == 2).all()


def test_model_coverage_explicitly_reports_common_population_exclusions():
    frame = predictions()
    frame = frame.loc[~(frame.model.eq('J_L') & frame.subject_id.eq('A00'))]
    result = evaluate(frame, ['R_L', 'J_L'], replicates=20)
    coverage = result['model_coverage'].set_index('model')
    assert coverage.loc['R_L', 'available_n'] == 60
    assert coverage.loc['J_L', 'available_n'] == 59
    assert (coverage.common_n == 59).all()
    assert coverage.loc['R_L', 'excluded_for_model_parity_n'] == 1
    assert (result['common_population_estimates'].n == 59).all()


def test_absent_model_outcome_remains_visible_in_coverage():
    frame = predictions()
    extra = frame.loc[frame.model.eq('R_L')].assign(outcome='missing_x_fit')
    result = evaluate(pd.concat([frame, extra]), ['R_L', 'J_L'], replicates=20)
    coverage = result['model_coverage'].query("outcome == 'missing_x_fit'").set_index('model')
    assert coverage.loc['J_L', 'available_n'] == 0
    assert coverage.loc['J_L', 'status'] == 'missing_model_task'
    assert coverage.loc['R_L', 'common_n'] == 0
    assert coverage.loc['R_L', 'status'] == 'no_common_population'
    assert coverage.loc['J_L', 'missing_diseases'] == 'A|B'
    assert not result['common_population_estimates'].outcome.eq('missing_x_fit').any()


def test_pooled_auc_can_reverse_without_changing_any_within_disease_ranking():
    frame = predictions()
    # Each disease has perfect ordering. Disease B's scores are put below A's
    # by a disease-specific monotone shift, damaging only cross-disease pairs.
    frame['probability'] = np.where(frame.event.eq(1), .8, .2)
    shifted = frame.model.eq('J_L') & frame.cohort.eq('B')
    frame.loc[shifted, 'probability'] *= .1
    normalized, _ = normalize_predictions(frame, 'cif_ipcw')
    diseases, pairs = population_diagnostics(normalized, 'common')
    assert np.allclose(diseases.within_disease_auc, 1.)
    pairs = pairs.set_index('model')
    assert pairs.loc['R_L', 'pooled_auc'] == pytest.approx(1.)
    assert pairs.loc['J_L', 'pooled_auc'] == pytest.approx(.19)
    assert pairs.loc['J_L', 'between_disease_pair_fraction'] == pytest.approx(.82)


def test_auc_decomposition_matches_brute_force_with_censoring_ties_and_competing_death():
    frame = predictions()
    frame.loc[frame.subject_id.isin(['A03', 'A04', 'A05']), 'time_days'] = 10.
    frame.loc[frame.subject_id.eq('B28'), ['event', 'time_days']] = [2, 20.]
    normalized, _ = normalize_predictions(frame, 'cif_ipcw')
    _, pairs = population_diagnostics(normalized, 'common')
    for model, cell in normalized.groupby('model'):
        row = pairs.loc[pairs.model.eq(model)].iloc[0]
        cases = cell.loc[cell.label.eq(1) & cell.ipcw_weight.gt(0)]
        controls = cell.loc[cell.label.eq(0) & cell.ipcw_weight.gt(0)]
        weight = cases.ipcw_weight.to_numpy()[:, None] * controls.ipcw_weight.to_numpy()
        comparison = cases.probability.to_numpy()[:, None] - controls.probability.to_numpy()
        concordance = (comparison > 0) + .5 * (comparison == 0)
        same = cases.cohort.to_numpy()[:, None] == controls.cohort.to_numpy()
        assert row.pooled_auc == pytest.approx(np.sum(weight * concordance) / weight.sum())
        assert row.within_disease_pair_auc == pytest.approx(np.sum(weight[same] * concordance[same]) / weight[same].sum())
        assert row.between_disease_pair_auc == pytest.approx(np.sum(weight[~same] * concordance[~same]) / weight[~same].sum())
        assert row.pooled_auc == pytest.approx(row.within_disease_pair_fraction * row.within_disease_pair_auc
            + row.between_disease_pair_fraction * row.between_disease_pair_auc)


def test_zero_weight_placeholder_predictions_cannot_change_auc_or_probability_diagnostics():
    frame = predictions()
    unknown = frame.subject_id.eq('A29')
    frame.loc[unknown, 'time_days'] = 10.
    frame['prediction_absent_zero_weight'] = unknown
    frame.loc[unknown, 'probability'] = 0.
    normalized, _ = normalize_predictions(frame, 'cif_ipcw')
    diseases, pairs = population_diagnostics(normalized, 'common')
    normalized.loc[unknown, 'probability'] = 1.
    diseases_changed, pairs_changed = population_diagnostics(normalized, 'common')
    pd.testing.assert_frame_equal(diseases, diseases_changed)
    pd.testing.assert_frame_equal(pairs, pairs_changed)


def test_native_sequence_and_horizon_truncated_x_exports_give_identical_pooled_metrics(tmp_path):
    native = predictions().query("model == 'R_L'").drop(columns='model').copy()
    native.loc[native.subject_id.isin(['A03', 'A04']), 'time_days'] = 10.
    native.loc[native.subject_id.eq('B28'), ['event', 'time_days']] = [2, 20.]
    tabular = native.copy()
    tabular['event'] = np.where(tabular.time_days.gt(365), 0, tabular.event)
    tabular['time_days'] = tabular.time_days.clip(upper=365)
    tabular['prediction_absent_zero_weight'] = tabular.event.eq(0) & tabular.time_days.lt(365)
    tabular.loc[tabular.prediction_absent_zero_weight, 'probability'] = 0.
    tabular_path, native_path = tmp_path / 'x.parquet', tmp_path / 'sequence.parquet'
    tabular.to_parquet(tabular_path)
    native.drop(columns='horizon_days').rename(columns={'probability': 'cif_365d'}).to_parquet(native_path)
    x, _ = read_artifact(dict(path=tabular_path, model='X', format='long'), estimand='cif_ipcw')
    j, _ = read_artifact(dict(path=native_path, model='J_L', format='direct_cif'), estimand='cif_ipcw')
    result = evaluate(pd.concat([x, j], ignore_index=True), ['X', 'J_L'], replicates=20)
    estimates = result['common_population_estimates'].pivot(index='metric', columns='model', values='estimate')
    np.testing.assert_allclose(estimates.X, estimates.J_L, equal_nan=True)
    contrasts = result['paired_contrasts']
    np.testing.assert_allclose(contrasts.gain, 0., atol=1e-14)
    np.testing.assert_allclose(contrasts.lower, 0., atol=1e-14)
    np.testing.assert_allclose(contrasts.upper, 0., atol=1e-14)


@pytest.mark.parametrize('field,value,match', [
    ('cohort', 'wrong', 'one fine disease'),
    ('data_contract', 'wrong', 'parity'),
    ('time_days', 150., 'Follow-up parity'),
    ('event', 2, 'parity'),
])
def test_inconsistent_patient_identity_or_follow_up_fails(field, value, match):
    frame = predictions()
    frame.loc[frame.model.eq('J_L') & frame.subject_id.eq('A00'), field] = value
    with pytest.raises(ValueError, match=match):
        evaluate(frame, ['R_L', 'J_L'], replicates=20)


def test_censoring_weights_remain_disease_conditional_after_pooling():
    frame = predictions()
    frame.loc[frame.subject_id.isin(['A03', 'A04', 'A05']), 'time_days'] = 10.
    normalized, _ = normalize_predictions(frame, 'cif_ipcw')
    model = normalized.loc[normalized.model.eq('R_L')]
    expected = metric_values(model.label.to_numpy(), model.probability.to_numpy(),
                             ipcw=model.ipcw_weight.to_numpy())['brier']
    pooled, _ = normalize_predictions(frame.assign(cohort='All'), 'cif_ipcw')
    model = pooled.loc[pooled.model.eq('R_L')]
    different_estimand = metric_values(model.label.to_numpy(), model.probability.to_numpy(),
                                       ipcw=model.ipcw_weight.to_numpy())['brier']
    assert not np.isclose(expected, different_estimand)
    result = evaluate(frame, ['R_L', 'J_L'], replicates=20)
    for key in ('available_population_estimates', 'common_population_estimates'):
        brier = result[key].query("model == 'R_L' and metric == 'brier'").iloc[0]
        assert brier.estimate == pytest.approx(expected)


def test_cli_tuning_does_not_open_held_out_artifacts(tmp_path, monkeypatch):
    frame = predictions()
    path = tmp_path / 'predictions.parquet'
    frame.to_parquet(path)
    metadata = tmp_path / 'metadata.parquet'
    pd.DataFrame({'subject_id': frame.subject_id.unique()}).to_parquet(metadata)
    config = tmp_path / 'analysis.yaml'
    config.write_text(yaml.safe_dump(dict(estimand='cif_ipcw', opera_model='R_L',
        patient_metadata=str(metadata), artifacts=[
            dict(path=str(path), split='tuning', seed=42),
            dict(path=str(tmp_path / 'DO_NOT_OPEN.parquet'), model='R_L', split='held_out', seed=42),
        ])))
    # Inventories use one declaration per model, even when source files contain
    # several models. Give each model its own file as production inventories do.
    config_data = yaml.safe_load(config.read_text())
    config_data['artifacts'] = config_data['artifacts'][1:]
    for model in ('R_L', 'J_L'):
        path = tmp_path / f'{model}.parquet'
        frame.loc[frame.model.eq(model)].to_parquet(path)
        config_data['artifacts'].append(dict(path=str(path), model=model, split='tuning', seed=42))
    config.write_text(yaml.safe_dump(config_data))
    output = tmp_path / 'output'
    monkeypatch.setattr(sys, 'argv', ['whole-cohort', '--analysis-config', str(config),
        '--output-dir', str(output), '--models', 'OPERA', 'J_L', '--replicates', '20'])
    main()
    manifest = json.loads((output / 'manifest.json').read_text())
    assert manifest['models'] == ['R_L', 'J_L']
    assert manifest['split'] == 'tuning'
    assert not any('DO_NOT_OPEN' in key for key in manifest['input_sha256'])
