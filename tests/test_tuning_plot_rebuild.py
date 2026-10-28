import json
from pathlib import Path
from types import SimpleNamespace
import yaml
from opera.run import run_tuning_plot_rebuild as runner


def test_isolated_rebuild_continues_and_resumes(tmp_path, monkeypatch):
    old = tmp_path / 'existing'
    (old / 'source_removal/tuning').mkdir(parents=True)
    cached = old / 'source_removal/tuning/borrowing_map.csv'
    cached.write_text('real,aggregate\n1,2\n')
    (old / 'attributions').mkdir()
    (old / 'attributions/raw_patients.csv').write_text('do not copy')
    template = tmp_path / 'template.yaml'
    template.write_text('bootstrap_method: bayesian\n')
    env = tmp_path / 'base.env'
    env.write_text(f'OPERA_PAPER_ANALYSIS_ROOT={old}\nOPERA_PAPER_ANALYSIS_TEMPLATE={template}\n')
    out = tmp_path / 'overnight'
    calls = []
    def execute(command, **kwargs):
        action = command[3]
        calls.append(action)
        return SimpleNamespace(returncode=1 if action == 'clinical-scores' else 0)
    monkeypatch.setattr(runner.subprocess, 'run', execute)
    argv = ['rebuild', '--env-file', str(env), '--output-dir', str(out)]
    monkeypatch.setattr('sys.argv', argv)
    runner.main()
    report = json.loads((out / 'overnight_run.json').read_text())
    assert report['status'] == 'finished_with_gaps'
    assert report['steps']['figures']['status'] == 'complete'
    assert report['steps']['clinical_scores']['status'] == 'failed'
    assert yaml.safe_load((out / 'analysis_template.yaml').read_text()) == {
        'bootstrap_method': 'multiplier', 'performance_min_cases': 5, 'performance_min_controls': 5}
    assert cached.read_text() == 'real,aggregate\n1,2\n'
    assert not (out / 'attributions/raw_patients.csv').exists()
    assert not {'train', 'evaluate', 'attributions', 'input-restrictions'} & set(calls)
    calls.clear()
    monkeypatch.setattr('sys.argv', argv + ['--resume'])
    runner.main()
    assert calls == ['clinical-scores']


def test_no_overwrite_of_original(tmp_path, monkeypatch):
    env = tmp_path / 'base.env'
    env.write_text(f'OPERA_PAPER_ANALYSIS_ROOT={tmp_path}\n')
    monkeypatch.setattr('sys.argv', ['rebuild', '--env-file', str(env), '--output-dir', str(tmp_path / 'nested')])
    import pytest
    with pytest.raises(SystemExit):
        runner.main()


def test_bootstrap_preflight_rejects_incompatible_module(monkeypatch):
    import pytest
    from opera.evaluation import secondary_analysis
    monkeypatch.setattr(secondary_analysis, 'patient_bootstrap', lambda frame, replicates, seed: {})
    with pytest.raises(RuntimeError, match='Sync opera/evaluation/secondary_analysis.py'):
        runner.check_bootstrap_api()
