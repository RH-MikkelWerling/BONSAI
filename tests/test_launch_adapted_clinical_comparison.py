import argparse
import hashlib
import json

import pandas as pd
import yaml

from opera.run.launch_adapted_clinical_comparison import compact, prepare, paired_auc


def test_paired_auc_reuses_draws_preserves_ties_and_handles_censored_rows():
    import numpy as np
    labels = np.array([0, 0, 1, 1, 0])
    score = np.array([0., .5, .5, 1., 1.])
    ipcw = np.array([1., 1., 1., 1., 0.])
    draws = np.random.default_rng(2026).exponential(size=(100, 5))
    same = paired_auc(labels, score, score, ipcw, draws)
    assert same['gain'] == same['lower'] == same['upper'] == 0
    assert same['model_auc'] == .875
    better = paired_auc(labels, np.array([.1, .2, .8, .9, .9]), score, ipcw, draws)
    assert better['gain'] > 0
    assert better['valid_draws'] == 100


def test_compaction_keeps_only_registered_outcome_populations_and_native_fields(tmp_path):
    source = tmp_path / 'source.parquet'
    pd.DataFrame({'subject_id': ['1', '2', '3', '1'],
                  'outcome': ['OS', 'OS', 'OS', 'other'],
                  'time_days': [40.] * 4, 'event': [1] * 4,
                  'cif_365d': [.2] * 4, 'unused_payload': ['drop'] * 4}).to_parquet(source)
    metadata = pd.DataFrame({'subject_id': ['1', '2', '3'], 'cohort_fine': ['A', 'A', 'B'],
                             'index_group': ['low', None, 'high']})
    specs = [{'outcome': 'OS', 'cohorts': ['A'], 'group_column': 'index_group'}]
    target = tmp_path / 'compact.parquet'
    assert compact(source, target, specs, metadata) == 1
    frame = pd.read_parquet(target)
    assert list(frame.subject_id) == ['1']
    assert 'unused_payload' not in frame
    assert {'time_days', 'event', 'cif_365d'} <= set(frame)


def test_prepare_binds_adapted_fit_and_preserves_registered_recipe_without_execution(tmp_path):
    national = tmp_path / 'national'
    bundle = national / 'adapted_fits'; bundle.mkdir(parents=True)
    config = bundle / 'R.yaml'; config.write_text('seed: 42\n')
    (bundle / 'bundle.json').write_text(json.dumps({'runs': [{
        'name': 'seed42/EA/R', 'config_path': 'R.yaml',
        'config_sha256': hashlib.sha256(config.read_bytes()).hexdigest(), 'run_directory': 'fit'}]}))
    run = national / 'neural_runs/fit'; run.mkdir(parents=True)
    (run / 'best.ckpt').write_bytes(b'fixture')
    for split, label in [('tuning', 'tuning'), ('held_out', 'test')]:
        prediction = national / 'predictions' / split; prediction.mkdir(parents=True)
        (prediction / 'direct_cr_predictions.parquet').write_bytes(b'not read during preparation')
        (prediction / 'evaluation_complete.json').write_text('{}')
        analysis = national / f'extended_{label}_comparison'; analysis.mkdir()
        (analysis / 'analysis.yaml').write_text(yaml.safe_dump({'artifacts': [{
            'path': str(prediction / 'direct_cr_predictions.parquet'), 'model': 'R_EA',
            'seed': 42, 'split': split, 'format': 'direct_cif', 'data_contract': 'registered'}]}))
    refreshed = tmp_path / 'refresh'; refreshed.mkdir()
    recipe = {'scores': [{'id': 'IPI', 'model': 'J_OA_OWN_L', 'threshold_split': 'train',
                          'cutoff_rule': 'cohort_size_matched'}], 'native_endpoints': {}}
    (refreshed / 'clinical_scores.yaml').write_text(yaml.safe_dump(recipe))
    (refreshed / 'refresh_report.json').write_text('{}')
    output = tmp_path / 'output'
    prepare(argparse.Namespace(national_root=national, refreshed_root=refreshed,
                               output_dir=output, model='R_EA', training_predictions=None, execute=False))
    prepared = yaml.safe_load((output / 'clinical_scores.yaml').read_text())
    assert prepared['scores'][0] == {**recipe['scores'][0], 'model': 'R_EA'}
    assert yaml.safe_load((refreshed / 'clinical_scores.yaml').read_text()) == recipe
    analysis = yaml.safe_load((output / 'analysis.yaml').read_text())
    assert analysis['clinical_threshold_artifacts'][0]['split'] == 'train'
    assert {a['split'] for a in analysis['artifacts']} == {'tuning', 'held_out'}
    assert not (output / 'training_INTERNAL').exists()
