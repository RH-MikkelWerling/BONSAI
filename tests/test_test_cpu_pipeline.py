from pathlib import Path
import importlib.util
import pytest
from opera.run.test_cpu_pipeline import validate_scoring, atomic_json, check_local_contrasts

spec = importlib.util.spec_from_file_location('pause_test_coordinator',
    Path(__file__).resolve().parents[1] / 'scripts/pause_test_coordinator.py')
pause = importlib.util.module_from_spec(spec); spec.loader.exec_module(pause)


def test_pause_identifies_only_exact_shard_zero_shell():
    assert pause.coordinator(['bash','scripts/train_then_evaluate.sh','0','0','4','run'], 'run')
    assert not pause.coordinator(['bash','scripts/train_then_evaluate.sh','0','1','4','run'], 'run')
    assert not pause.coordinator(['python','scripts/train_then_evaluate.sh','0','0','4','run'], 'run')
    assert not pause.coordinator(['bash','scripts/train_then_evaluate.sh','0','0','4','other'], 'run')


def test_descendants_do_not_include_unrelated_gpu_jobs():
    snapshot = {1:dict(ppid=0), 2:dict(ppid=1), 3:dict(ppid=2), 4:dict(ppid=0)}
    assert pause.descendants(snapshot, 1) == {2,3}


def test_handoff_requires_exact_complete_scoring_shard():
    validate_scoring([dict(name='a',status='evaluated'),dict(name='b',status='not_estimable')], ['a','b'])
    for rows in [[dict(name='a',status='evaluated')],
                 [dict(name='a',status='training_incomplete'),dict(name='b',status='evaluated')],
                 [dict(name='a',status='evaluated'),dict(name='a',status='evaluated')]]:
        with pytest.raises(ValueError): validate_scoring(rows, ['a','b'])


def test_cpu_markers_are_atomic_json(tmp_path):
    import json
    p = tmp_path/'status.json'; atomic_json(p, dict(ok=True))
    assert json.loads(p.read_text()) == {'ok':True}
    assert not p.with_suffix('.tmp').exists()


def test_required_contrasts_cannot_be_replaced_by_tuning_results(tmp_path):
    p = tmp_path/'summaries.csv'
    p.write_text('split,seed,metric,horizon_days,stratum,aggregation,gain,contrast\n'
                 'tuning,42,auc,0,all_cells,equal_cell_not_population_or_endpoint_group_macro,.1,P1\n')
    with pytest.raises(ValueError, match='Missing required local contrasts'):
        check_local_contrasts(p)


def test_two_cpu_roles_overlap_and_join_before_paper_numbers(tmp_path):
    import json, os, shutil, subprocess, sys, time, yaml
    package = tmp_path/'opera/run'; package.mkdir(parents=True)
    (package.parent/'__init__.py').touch(); (package/'__init__.py').touch()
    shutil.copy(Path(__file__).resolve().parents[1]/'opera/run/test_cpu_pipeline.py', package/'test_cpu_pipeline.py')
    (package/'paper_prediction_queue.py').write_text('def bundle_runs(path): return {}, []\n')
    scripts = tmp_path/'scripts'; scripts.mkdir(); (scripts/'__init__.py').touch()
    (scripts/'set_test_env.py').write_text("from pathlib import Path\ndef selected_fit(*args): return dict(run_directory='fit'), Path('cfg.yaml')\n")
    (package/'paper_experiments.py').write_text('''import sys, csv, time, yaml
from pathlib import Path
from dotenv import dotenv_values
action = sys.argv[1]
env = dotenv_values(sys.argv[sys.argv.index('--env-file')+1])
root = Path(env['OPERA_PAPER_ANALYSIS_ROOT'])
config = root/'analysis_inventory/analysis.yaml'
if action == 'inventory':
    config.parent.mkdir(); config.write_text('artifacts: []\\n')
elif action == 'analyse':
    (root/'numerical_work_started').touch(); time.sleep(2)
    path = root/'prediction_analysis/effect_summaries.csv'; path.parent.mkdir()
    with path.open('w') as f:
        w=csv.DictWriter(f, fieldnames=['split','seed','metric','horizon_days','stratum','aggregation','gain','contrast'])
        w.writeheader()
        for code in ['P1','P2','P5','T2','O1','O2','router_borrow_AB','O2_MATCHED','OFF_J','OAOWN_OFF','O_CORRECTIONS_MATCHED']:
            w.writerow(dict(split='held_out',seed=42,metric='auc',horizon_days=0,stratum='all_cells',aggregation='equal_cell_not_population_or_endpoint_group_macro',gain=.1,contrast=code))
    (root/'numerical_work_finished').touch()
elif action == 'clinical-scores':
    c=yaml.safe_load(config.read_text())
    assert c['bootstrap_seed']==2026 and c['clinical_threshold_artifacts']
    assert (root/'numerical_work_started').exists() and not (root/'numerical_work_finished').exists()
    (root/'clinical_overlapped').touch()
elif action == 'paper-numbers':
    assert (root/'clinical_overlapped').exists() and (root/'numerical_work_finished').exists()
''')
    data=tmp_path/'data'; marks=data/'predictions/.overnight/tag'; marks.mkdir(parents=True)
    (marks/'cpu_handoff.json').write_text('{}')
    (marks/'shard_1_of_2.done').touch()
    for i,name in enumerate(['a','b']):
        (data/f'predictions/queue_held_out_shard{i}.json').write_text(json.dumps([dict(name=name,status='evaluated')]))
    names=tmp_path/'fits.txt'; names.write_text('a\nb\n')
    source=tmp_path/'source.yaml'
    source.write_text(yaml.safe_dump(dict(patient_metadata='/prepared/metadata',clinical_threshold_artifacts=[dict(split='train')])))
    recipe=tmp_path/'clinical.yaml'; recipe.write_text('scores: []\n')
    (tmp_path/'.env').write_text(f'OPERA_PAPER_OPERA_MODEL=J_OA_OWN_L\nOPERA_PAPER_PLOT_SPLIT=held_out\nOPERA_PAPER_CLINICAL_SOURCE_ANALYSIS={source}\nOPERA_PAPER_CLINICAL_SCORE_CONFIG={recipe}\n')
    command=[sys.executable,'-m','opera.run.test_cpu_pipeline','--tag','tag','--paper-root',str(data),'--shards','2','--fit-names',str(names),'--wait-hours','.02']
    env={**os.environ,'PYTHONPATH':str(tmp_path)}
    numerical=subprocess.Popen([*command,'--role','numerical'],cwd=tmp_path,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
    clinical=None
    try:
        deadline=time.monotonic()+20
        root=data/'analysis/tag'
        while not (root/'numerical_work_started').exists():
            if numerical.poll() is not None or time.monotonic()>deadline:
                raise AssertionError(numerical.communicate(timeout=5)[0])
            time.sleep(.02)
        clinical=subprocess.Popen([*command,'--role','clinical'],cwd=tmp_path,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
        output=clinical.communicate(timeout=30)[0]; assert clinical.returncode==0,output
        output=numerical.communicate(timeout=40)[0]; assert numerical.returncode==0,output
        assert (root/'pipeline_actions_complete').exists()
        assert json.loads((marks/'cpu_handoff.json').read_text())['scoring_complete']
    finally:
        for process in [numerical,clinical]:
            if process is not None and process.poll() is None:
                process.kill(); process.wait()


@pytest.mark.parametrize('distributed', [False, True])
def test_resume_reuses_inventory_and_completed_clinical_work(tmp_path, monkeypatch, distributed):
    import sys
    from opera.run import test_cpu_pipeline as pipeline
    root = tmp_path/'analysis/tag'
    (root/'analysis_inventory').mkdir(parents=True)
    (root/'analysis_inventory/analysis.yaml').write_text('artifacts: []\n')
    (root/'cpu_logs').mkdir()
    (root/'overnight.env').write_text('')
    marks = tmp_path/'predictions/.overnight/tag'; marks.mkdir(parents=True)
    (marks/'cpu_inventory_ready').touch()
    (marks/'cpu_clinical_done').touch()
    (marks/'cpu_numerical.failed').write_text('{"error":"memory"}')
    actions = []
    def fake_run(command, **kw):
        actions.append(command[4])
        if distributed:
            assert kw['env']['OPERA_ANALYSIS_CHECKPOINT_DIR'] == str((root/'_cpu_checkpoints_INTERNAL').resolve())
    monkeypatch.setattr(pipeline.subprocess, 'run', fake_run)
    monkeypatch.setattr(pipeline, 'check_local_contrasts', lambda path: None)
    monkeypatch.setattr(sys, 'argv', ['pipeline', '--tag', 'tag', '--role', 'numerical',
        '--paper-root', str(tmp_path), '--resume-analysis', *(['--distributed'] if distributed else [])])
    pipeline.main()
    assert actions == ['analyse','plot-tables','task-structure','paper-numbers',
                       'figure-readiness','init-figures','figures']
    assert (root/'pipeline_actions_complete').exists()
    assert not (marks/'cpu_numerical.failed').exists()
    assert len(list(marks.glob('cpu_numerical.failed.previous_*'))) == 1
