import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest
import yaml
from scripts import set_test_env as handoff


def test_selected_fit_uses_explicit_encoder_and_seed(monkeypatch):
    monkeypatch.setattr(handoff, 'canonical_models', lambda row, config, bundle: [config['alias']])
    rows = [(dict(seed=seed), Path(f'{alias}_{seed}.yaml'), {'alias': alias})
            for alias in ['J_OA_OWN_L', 'R_E', 'R_EA'] for seed in [42,43]]
    for alias in ['J_OA_OWN_L', 'R_E', 'R_EA']:
        row, path = handoff.selected_fit({}, rows, alias)
        assert row['seed']==42
        assert str(path)==f'{alias}_42.yaml'
    with pytest.raises(ValueError, match='found 0'):
        handoff.selected_fit({}, rows, 'unknown')


def test_clinical_handoff_retargets_existing_local_and_national_model_keys(tmp_path):
    path=tmp_path/'clinical.yaml'
    path.write_text('''all_indices:
  template:
    model: J_OA_OWN_L
    cutoff_rule: cohort_size_matched
scores:
  - id: test
    model: J_OA_OWN_L
    cutoff_rule: cohort_size_matched
native_endpoints:
  IPI: {outcome: overall_survival}
''')
    handoff.set_clinical_model(path, 'test', 'R_EA')
    config=yaml.safe_load(path.read_text())
    assert set(handoff.clinical_models(config).values())=={'R_EA'}
    assert config['native_endpoints']=={'IPI': {'outcome':'overall_survival'}}
    handoff.set_clinical_model(path, 'test2', 'R_E')
    assert set(handoff.clinical_models(yaml.safe_load(path.read_text())).values())=={'R_E'}


def test_environment_handoff_wires_explicit_clinical_workspace(tmp_path, monkeypatch):
    from dotenv import dotenv_values
    data = tmp_path / 'data'; run = data / 'neural_runs/fit'; run.mkdir(parents=True)
    (run / 'best.ckpt').write_bytes(b'fixture')
    metadata = tmp_path / 'metadata.parquet'; metadata.touch()
    training = tmp_path / 'train.parquet'; training.touch()
    source = tmp_path / 'source.yaml'
    source.write_text(yaml.safe_dump(dict(patient_metadata=str(metadata),
        clinical_threshold_artifacts=[dict(model='J_OA_OWN_L', seed=42,
                                          split='train', path=str(training))])))
    recipe = tmp_path / 'clinical.yaml'
    recipe.write_text('all_indices:\n  template:\n    model: J_OA_OWN_L\n    cutoff_rule: cohort_size_matched\nscores: []\n')
    env = tmp_path / '.env'; env.write_text('')
    monkeypatch.setattr(handoff, 'ROOT', data)
    monkeypatch.setattr(handoff, 'bundle_runs', lambda path: ({}, []))
    monkeypatch.setattr(handoff, 'selected_fit', lambda *args: (
        dict(name='fit', run_directory='fit'), tmp_path / 'config.yaml'))
    monkeypatch.setattr(sys, 'argv', ['set_test_env.py', '--env-file', str(env),
        '--split', 'held_out', '--clinical-config', str(recipe),
        '--clinical-source-analysis', str(source)])
    handoff.main()
    values = dotenv_values(env)
    assert values['OPERA_PAPER_CLINICAL_SCORE_CONFIG'] == str(recipe)
    assert values['OPERA_PAPER_PATIENT_METADATA'] == str(metadata)
    assert values['OPERA_PAPER_CLINICAL_SOURCE_ANALYSIS'] == str(source)
    assert values['OPERA_PAPER_INVENTORY_SPLITS'] == 'tuning,held_out'


@pytest.mark.parametrize('failure', ['process', 'incomplete_report'])
def test_failed_evaluation_cannot_leave_success_marker(tmp_path, failure):
    scripts=tmp_path/'scripts'; scripts.mkdir()
    shutil.copy('scripts/train_then_evaluate.sh', scripts/'train_then_evaluate.sh')
    stub = 'exit 1\n' if failure=='process' else '''mkdir -p data/paper_256_v3/predictions
cat > data/paper_256_v3/predictions/queue_tuning_shard0.json <<'EOF'
[{"name":"fit","paper_models":["J_L"],"status":"missing_training"}]
EOF
'''
    (scripts/'run_paper_experiments.sh').write_text('#!/usr/bin/env bash\n'+stub)
    result=subprocess.run(['bash',str(scripts/'train_then_evaluate.sh'),'0','0','1','failure_test'],
        cwd=tmp_path,env={**os.environ,'OPERA_PYTHON':sys.executable,'EVALUATE_ONLY':'1'},
        capture_output=True,text=True,timeout=20)
    assert result.returncode!=0
    markers=tmp_path/'data/paper_256_v3/predictions/.overnight/failure_test'
    assert not (markers/'shard_0_of_1.done').exists()
    assert (markers/'shard_0_of_1.failed').exists()
    assert not (tmp_path/'data/paper_256_v3/analysis/failure_test/pipeline_actions_complete').exists()


def test_evaluation_only_shard_uses_explicit_release_fit_list(tmp_path):
    scripts = tmp_path / 'scripts'; scripts.mkdir()
    shutil.copy('scripts/train_then_evaluate.sh', scripts / 'train_then_evaluate.sh')
    (scripts / 'run_paper_experiments.sh').write_text('''#!/usr/bin/env bash
printf '%s\\n' "$@" > launch_args.txt
mkdir -p data/paper_256_v3/predictions
echo '[{"name":"chosen","paper_models":["J_L"],"status":"evaluated"}]' > data/paper_256_v3/predictions/queue_held_out_shard1.json
''')
    names = tmp_path / 'fits.txt'; names.write_text('seed42/L/chosen\nseed43/L/repeat\n')
    result = subprocess.run(['bash', str(scripts / 'train_then_evaluate.sh'), '0', '1', '2', 'scope'],
        env={**os.environ, 'OPERA_PYTHON': sys.executable, 'EVALUATE_ONLY': '1',
             'SPLIT': 'held_out', 'EVALUATION_FIT_NAMES': str(names)},
        capture_output=True, text=True, timeout=20)
    assert result.returncode == 0, result.stderr
    args = (tmp_path / 'launch_args.txt').read_text().splitlines()
    assert args.count('--fit-name') == 2
    assert 'seed42/L/chosen' in args and 'seed43/L/repeat' in args
    assert 'worker' not in args
    assert (tmp_path / 'data/paper_256_v3/predictions/.overnight/scope/shard_1_of_2.done').is_file()


def test_local_launch_carries_clinical_training_artifacts_into_test_analysis(tmp_path):
    scripts = tmp_path / 'scripts'; scripts.mkdir(); (scripts / '__init__.py').touch()
    for name in ['train_then_evaluate.sh', 'set_test_env.py']:
        shutil.copy(Path('scripts') / name, scripts / name)
    (scripts / 'run_paper_experiments.sh').write_text('''#!/usr/bin/env bash
mkdir -p data/paper_256_v3/predictions
echo '[{"name":"fit","paper_models":["J_L"],"status":"evaluated"}]' > data/paper_256_v3/predictions/queue_held_out_shard0.json
''')
    package = tmp_path / 'opera/run'; package.mkdir(parents=True)
    (package.parent / '__init__.py').touch(); (package / '__init__.py').touch()
    (package / 'paper_prediction_queue.py').write_text('''from pathlib import Path
def canonical_models(row, config, bundle): return ['J_OA_OWN_L']
def bundle_runs(path): return {}, [(dict(seed=42, name='fit', run_directory='local'), Path('config.yaml'), {})]
''')
    (package / 'paper_experiments.py').write_text('''import sys, yaml
from pathlib import Path
from dotenv import dotenv_values
action = sys.argv[1]
env = dotenv_values(sys.argv[sys.argv.index('--env-file') + 1])
root = Path(env['OPERA_PAPER_ANALYSIS_ROOT'])
with (root / 'actions.txt').open('a') as f: f.write(action + '\\n')
config = root / 'analysis_inventory/analysis.yaml'
if action == 'inventory':
    config.parent.mkdir(parents=True)
    config.write_text(yaml.safe_dump(dict(artifacts=[])))
if action == 'analyse':
    import csv
    path = root / 'prediction_analysis/effect_summaries.csv'
    path.parent.mkdir()
    with path.open('w') as f:
        w = csv.DictWriter(f, fieldnames=['split','seed','metric','horizon_days','stratum','aggregation','gain','contrast'])
        w.writeheader()
        for code in ['P1','P2','P5','T2','O1','O2','router_borrow_AB','O2_MATCHED','OFF_J','OAOWN_OFF','O_CORRECTIONS_MATCHED']:
            w.writerow(dict(split='held_out', seed=42, metric='auc', horizon_days=0,
                stratum='all_cells', aggregation='equal_cell_not_population_or_endpoint_group_macro', gain=0.01, contrast=code))
if action == 'clinical-scores':
    c = yaml.safe_load(config.read_text())
    assert c['clinical_threshold_artifacts'][0]['split'] == 'train'
    assert c['bootstrap_method'] == 'multiplier' and c['bootstrap_seed'] == 2026
    assert env['OPERA_PAPER_INVENTORY_SPLITS'] == 'tuning,held_out'
''')
    source = tmp_path / 'clinical_source.yaml'
    source.write_text(yaml.safe_dump(dict(clinical_threshold_artifacts=[
        dict(model='J_OA_OWN_L', split='train', path='/registered/train.parquet')],
        patient_metadata_column_map={'raw_id': 'subject_id'})))
    (tmp_path / '.env').write_text(f'OPERA_PAPER_CLINICAL_SOURCE_ANALYSIS={source}\n')
    result = subprocess.run(['bash', str(scripts / 'train_then_evaluate.sh'), '0', '0', '1', 'release'],
        env={**os.environ, 'PYTHONPATH': str(tmp_path), 'OPERA_PYTHON': sys.executable,
             'EVALUATE_ONLY': '1', 'SPLIT': 'held_out'}, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
    root = tmp_path / 'data/paper_256_v3/analysis/release'
    actions = (root / 'actions.txt').read_text().splitlines()
    assert actions.index('clinical-scores') < actions.index('paper-numbers')
    assert json.loads((root / 'release_record.json').read_text())['split'] == 'held_out'
    assert (root / 'pipeline_actions_complete').is_file()
