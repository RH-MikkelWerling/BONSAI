"""Numerical parity, interruption recovery and shared job claims."""
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time

import numpy as np
import pandas as pd
import pytest

from opera.evaluation import distributed_analysis as da
from opera.evaluation import secondary_analysis as sa


def predictions():
    rng = np.random.default_rng(123)
    parts = []
    for cohort in ('CLL', 'MM'):
        for outcome in range(12):
            times = rng.integers(1, 500, 20).astype(float)
            events = rng.choice([0, 1, 2], 20)
            for model in ('A', 'B'):
                for horizon in (90, 365):
                    parts.append(pd.DataFrame(dict(model=model, split='held_out', seed=42,
                        cohort=cohort, outcome=f'o{outcome}', horizon_days=horizon,
                        subject_id=[f'{cohort}_{i}' for i in range(20)],
                        time_days=times, event=events, probability=rng.uniform(.05, .95, 20),
                        data_contract='c')))
    return sa.normalize_predictions(pd.concat(parts, ignore_index=True), 'cif_ipcw')[0]


def assert_tables_same(left, right):
    assert left.keys() == right.keys()
    for name in left:
        pd.testing.assert_frame_equal(left[name], right[name])


def run_worker_threads(queue, stop, errors):
    while not stop.is_set():
        try:
            if not da.perform_one(queue):
                time.sleep(.02)
        except Exception as error:
            errors.append(error)
            return


def test_distributed_batches_equal_serial_and_reuse_saved_work(tmp_path, monkeypatch):
    frame = predictions()
    draws = sa.patient_bootstrap(frame, 17, 2026, method='multiplier')
    options = dict(replicates=17, draws=draws, min_cases=1, min_controls=1)
    serial = sa.paired_analysis(frame, {'gain':'gain(A,B)'}, **options)
    queue = tmp_path/'queue'; queue.mkdir()
    stop = threading.Event(); errors = []
    workers = [threading.Thread(target=run_worker_threads, args=(queue,stop,errors)) for _ in range(2)]
    for worker in workers: worker.start()
    try:
        distributed = sa.paired_analysis(frame, {'gain':'gain(A,B)'}, checkpoint_dir=queue, **options)
    finally:
        stop.set()
        for worker in workers: worker.join(10)
    assert not errors
    assert_tables_same(serial, distributed)
    files = {p.name:p.stat().st_mtime_ns for p in queue.glob('*.result_INTERNAL.pkl')}
    assert len(files) == 3  # 48 cells; 20 per durable batch
    monkeypatch.setattr(sa, '_paired_cells', lambda *a,**kw: pytest.fail('Completed work recomputed'))
    resumed = sa.paired_analysis(frame, {'gain':'gain(A,B)'}, checkpoint_dir=queue, **options)
    assert_tables_same(serial, resumed)
    assert files == {p.name:p.stat().st_mtime_ns for p in queue.glob('*.result_INTERNAL.pkl')}
    with pytest.raises(ValueError, match='options changed'):
        sa.paired_analysis(frame, {'changed':'gain(A,B)'}, checkpoint_dir=queue, **options)


def test_interrupted_publication_and_dead_worker_are_recoverable(tmp_path):
    frame = predictions()
    key, cell = next(iter(frame.groupby(sa.CELL, sort=True)))
    draws = sa.patient_bootstrap(frame, 3, 2026, method='multiplier')
    batch = [(key,cell,{key[:3]:draws[key[:3]]})]
    options = dict(parsed={'gain':{'A':1,'B':-1}},patient_gain_contrasts=[],min_cases=0,min_controls=0)
    da.publish_job(tmp_path, 0, batch, options)
    lock_path = tmp_path/'batch_00000.lock'
    child = subprocess.Popen([sys.executable,'-c',
        'import fcntl,sys,time; f=open(sys.argv[1],"a"); fcntl.flock(f,fcntl.LOCK_EX); print("locked",flush=True); time.sleep(60)',
        str(lock_path)],stdout=subprocess.PIPE,text=True)
    try:
        assert child.stdout.readline().strip() == 'locked'
        assert not da.perform_one(tmp_path)
    finally:
        child.kill();child.wait()
    assert da.perform_one(tmp_path)
    first = (tmp_path/'batch_00000.result_INTERNAL.pkl').stat().st_mtime_ns
    # Simulate a coordinator interrupted before publishing the remaining jobs.
    parts = list(da.distributed_parts(tmp_path, iter([batch]), 1, options))
    assert len(parts) == 1
    assert first == (tmp_path/'batch_00000.result_INTERNAL.pkl').stat().st_mtime_ns
    assert not da.perform_one(tmp_path)


def test_prepared_snapshot_roundtrip_and_signature_invalidation(tmp_path):
    frame = predictions().iloc[:100].copy()
    # Noncontiguous indices and float64 weights must survive parquet partitions.
    frame.index = np.arange(len(frame))*3
    context = dict(draws=sa.patient_bootstrap(frame, 5, 2026, method='multiplier'), audit={'n':100})
    da.save_prepared(tmp_path/'snapshot', frame, context, chunk_rows=19)
    restored, saved = da.load_prepared(tmp_path/'snapshot')
    pd.testing.assert_frame_equal(frame, restored)
    assert saved['audit'] == context['audit']
    for key in context['draws']:
        assert saved['draws'][key][0] == context['draws'][key][0]
        np.testing.assert_array_equal(saved['draws'][key][1], context['draws'][key][1])
    artifact=tmp_path/'artifact';artifact.write_text('original')
    config={'artifacts':[{'path':str(artifact)}], 'bootstrap_seed':2026}
    original=da.signature(config)
    assert original == da.signature(config)
    assert original != da.signature({**config,'bootstrap_seed':99})
    artifact.write_text('new contents')
    assert original != da.signature(config)


def test_atomic_publication_and_stage_reuse(tmp_path):
    path=tmp_path/'result'
    def interrupted(handle):
        handle.write(b'incomplete')
        raise RuntimeError('interrupted')
    with pytest.raises(RuntimeError): da.atomic_write(path,interrupted)
    assert not path.exists()
    assert not list(tmp_path.glob('*.tmp'))
    assert da.cached_call(tmp_path,'summary',lambda: {'gain':.1}) == {'gain':.1}
    assert da.cached_call(tmp_path,'summary',lambda: pytest.fail('Stage repeated')) == {'gain':.1}


def test_full_runner_recovers_with_two_worker_processes_without_preparing_again(tmp_path, monkeypatch):
    from opera.run import analyse_locked_predictions as runner
    from test_secondary_analysis import predictions as small_predictions
    small_predictions().to_csv(tmp_path/'p.csv',index=False)
    pd.DataFrame([dict(cohort='CLL',outcome=o,horizon_days=365,
        n_events_train=12,n_patients_train=120,endpoint_group='Infection',
        outcome_family='t',loss_endpoint=o) for o in ('a','b')]).to_csv(tmp_path/'cells.csv',index=False)
    config=dict(estimand='binary_known_status_unweighted',
        artifacts=[{'path':str(tmp_path/'p.csv')}],
        contrast_manifest='opera/configs/manifests/final_contrasts_20261014.yaml',
        cell_metadata=str(tmp_path/'cells.csv'),bootstrap_replicates=7)
    monkeypatch.setenv('OPERA_ANALYSIS_WORKERS','1')
    monkeypatch.setenv('OPERA_SUMMARY_WORKERS','1')
    serial=tmp_path/'serial'; runner.run(config,serial)
    checkpoints=tmp_path/'_cpu_checkpoints_INTERNAL'
    monkeypatch.setenv('OPERA_ANALYSIS_CHECKPOINT_DIR',str(checkpoints))
    command=[sys.executable,'-u','-m','opera.run.analysis_cpu_worker',
             '--checkpoint-dir',str(checkpoints),'--wait-hours','.1','--poll-seconds','.05']
    repo=Path(__file__).resolve().parents[1]
    environment={**os.environ,'PYTHONPATH':str(repo)}
    logs=[(tmp_path/f'worker{i}.log').open('w') for i in range(2)]
    workers=[subprocess.Popen(command,cwd=repo,env=environment,stdout=log,stderr=subprocess.STDOUT)
             for log in logs]
    original=runner.panel_estimates
    monkeypatch.setattr(runner,'panel_estimates',lambda *a,**kw: (_ for _ in ()).throw(RuntimeError('coordinator interrupted')))
    try:
        with pytest.raises(RuntimeError,match='coordinator interrupted'):
            runner.run(config,tmp_path/'distributed')
        assert list(checkpoints.glob('*/paired/primary/*.result_INTERNAL.pkl'))
        monkeypatch.setattr(runner,'panel_estimates',original)
        monkeypatch.setattr(runner,'prepare_analysis_predictions',lambda *a: pytest.fail('Raw prediction preparation repeated'))
        runner.run(config,tmp_path/'distributed')
        for name in ('estimates','contrasts','effect_summaries','effect_summaries_disease_balanced',
                     'calibration','calibration_summary','strategy_selection'):
            pd.testing.assert_frame_equal(pd.read_csv(serial/f'{name}.csv'),
                                          pd.read_csv(tmp_path/'distributed'/f'{name}.csv'))
        np.testing.assert_array_equal(np.load(serial/'paired_bootstrap_draws_INTERNAL.npz')['gains'],
            np.load(tmp_path/'distributed/paired_bootstrap_draws_INTERNAL.npz')['gains'])
    finally:
        for worker in workers:
            worker.terminate()
        for worker in workers:
            worker.wait(timeout=10)
        for log in logs: log.close()


def test_workers_refuse_incompatible_code_and_failures_are_explicit(tmp_path, monkeypatch):
    frame=predictions()
    key,cell=next(iter(frame.groupby(sa.CELL,sort=True)))
    draws=sa.patient_bootstrap(frame,3,2026,method='multiplier')
    batch=[(key,cell,{key[:3]:draws[key[:3]]})]
    options=dict(parsed={'gain':{'A':1,'B':-1}},patient_gain_contrasts=[],min_cases=0,min_controls=0)
    da.publish_job(tmp_path,0,batch,options)
    da.save_json(tmp_path/'queue.json',dict(total=1,options=options,implementation={'wrong':'version'}))
    with pytest.raises(RuntimeError,match='differs from coordinator'): da.perform_one(tmp_path)
    da.save_json(tmp_path/'queue.json',dict(total=1,options=options,implementation=da.worker_identity()))
    monkeypatch.setattr(sa,'_paired_cells',lambda *a,**kw: (_ for _ in ()).throw(MemoryError('test failure')))
    with pytest.raises(MemoryError): da.perform_one(tmp_path)
    assert not (tmp_path/'batch_00000.result_INTERNAL.pkl').exists()
    assert 'test failure' in (tmp_path/'batch_00000.failed.json').read_text()
    with pytest.raises(RuntimeError,match='CPU batch failed'):
        list(da.distributed_parts(tmp_path,iter([batch]),1,options))


@pytest.mark.parametrize('aggregation',['equal_cell','disease_balanced'])
@pytest.mark.parametrize('stable',[False,True])
def test_distributed_effect_summaries_preserve_draws_and_weighting(tmp_path, aggregation, stable):
    frame=predictions()
    draws=sa.patient_bootstrap(frame,25,2026,method='multiplier')
    paired=sa.paired_analysis(frame,{'ab':'gain(A,B)','ba':'gain(B,A)'},draws=draws,replicates=25)
    # Exercise fixed-roster missing draws as well as stable-cell exclusion.
    paired['bootstrap_draws'].iloc[0].gain_draws[0] = np.nan
    metadata=frame[['cohort','outcome','horizon_days']].drop_duplicates().assign(
        loss_endpoint=lambda x:x.outcome.str.replace(r'\d','',regex=True),
        outcome_family='laboratory',support_bin='low')
    kwargs=dict(strata=['cohort','support_bin'],stable_cells_only=stable,aggregation=aggregation,
                _summary_workers=1)
    original=sa.effect_summaries(paired['contrasts'],paired['bootstrap_draws'],metadata,**kwargs)
    queue=tmp_path/'effect';queue.mkdir()
    stop=threading.Event();errors=[]
    thread=threading.Thread(target=run_worker_threads,args=(queue,stop,errors));thread.start()
    try:
        distributed=sa.effect_summaries(paired['contrasts'],paired['bootstrap_draws'],metadata,
                                        checkpoint_dir=queue,**kwargs)
    finally:
        stop.set();thread.join(10)
    assert not errors
    pd.testing.assert_frame_equal(original[0],distributed[0])
    keys=['stratum','split','seed','contrast','metric','horizon_days','stratum_value']
    pd.testing.assert_frame_equal(original[1].sort_values(keys).reset_index(drop=True),
                                 distributed[1].sort_values(keys).reset_index(drop=True))


def test_distributed_calibration_and_operating_points_equal_serial(tmp_path):
    from opera.evaluation.operating_points import operating_points, DEFAULT_RULES
    # Add tuning patients with distinct IDs; retain full patient draw sets.
    held=predictions()
    tuning=held.copy();tuning['split']='tuning';tuning['subject_id']='t_'+tuning.subject_id
    frame=pd.concat([held,tuning],ignore_index=True)
    draws=sa.patient_bootstrap(frame,19,2026,method='multiplier')
    serial_cal=(sa.reliability_tables(frame),sa.calibration_summary(frame))
    pairs=[('A','B'),('B','A')]
    serial_op=[operating_points(frame,draws,rule=rule,pairs=pairs,minimum_events=1) for rule in DEFAULT_RULES]
    stop=threading.Event();errors=[]
    queues=[tmp_path/'calibration',tmp_path/'operating']
    for queue in queues:queue.mkdir()
    threads=[threading.Thread(target=run_worker_threads,args=(q,stop,errors)) for q in queues]
    for thread in threads:thread.start()
    try:
        distributed_cal=da.distributed_calibration(frame,queues[0])
        distributed_op=da.distributed_operating_points(frame,draws,queues[1],rules=DEFAULT_RULES,
            pairs=pairs,minimum_events=1)
    finally:
        stop.set()
        for thread in threads:thread.join(10)
    assert not errors
    for left,right in zip(serial_cal,distributed_cal):pd.testing.assert_frame_equal(left,right)
    for left,right in zip(serial_op,distributed_op):
        for a,b in zip(left,right):pd.testing.assert_frame_equal(a,b)


def test_distributed_patient_subgroups_equal_serial(tmp_path):
    frame=predictions()
    draws=sa.patient_bootstrap(frame,23,2026,method='multiplier')
    result=sa.paired_analysis(frame,{'ab':'gain(A,B)','ba':'gain(B,A)'},draws=draws,replicates=23)
    gains=result['patient_gains']
    metadata=pd.DataFrame({'subject_id':frame.subject_id.unique()})
    metadata['age']=np.where(np.arange(len(metadata)) % 2,70,40)
    metadata['region']=np.where(np.arange(len(metadata)) % 3,'east','west')
    specifications={'age':{'provenance':'pre_index','bins':[0,50,100]},
                    'region':{'provenance':'pre_index'}}
    serial=(sa.subgroup_patient_gains(gains,metadata,specifications,draws=draws),
            sa.panel_subgroup_gains(gains,metadata,specifications,draws=draws))
    queue=tmp_path/'subgroups';queue.mkdir()
    stop=threading.Event();errors=[]
    thread=threading.Thread(target=run_worker_threads,args=(queue,stop,errors));thread.start()
    try:
        distributed=da.distributed_patient_subgroups(gains,metadata,specifications,draws,queue)
    finally:
        stop.set();thread.join(10)
    assert not errors
    for a,b in zip(serial,distributed):pd.testing.assert_frame_equal(a,b)
    assert len(list((queue/'shared_draws').glob('*.pkl'))) == len(draws)


def test_index_filtered_sensitivity_matches_materialized_subset():
    frame=predictions()
    ids=set(frame.subject_id.unique()[::2])
    models={'A','B'}
    draws=sa.patient_bootstrap(frame,11,2026,method='multiplier')
    contrasts={'ab':'gain(A,B)'}
    reference=sa.paired_analysis(frame.loc[frame.subject_id.isin(ids)&frame.model.isin(models)],
        contrasts,draws=draws,replicates=11,patient_gain_contrasts=[])
    actual=sa.paired_analysis(frame,contrasts,draws=draws,replicates=11,
        patient_gain_contrasts=[],subject_ids=ids,model_names=models)
    for name in ('estimates','contrasts','contrast_coverage'):
        pd.testing.assert_frame_equal(reference[name],actual[name])
    for name,column in [('bootstrap_draws','gain_draws'),('model_bootstrap_draws','estimate_draws')]:
        a=reference[name];b=actual[name]
        pd.testing.assert_frame_equal(a.drop(columns=column),b.drop(columns=column))
        for left,right in zip(a[column],b[column]):
            np.testing.assert_array_equal(left,right)


def test_bounded_grouping_and_identity_counts_match_pandas():
    frame=predictions().sample(frac=1,random_state=13).reset_index(drop=True)
    ids=set(frame.subject_id.unique()[::2])
    expected=frame.groupby(sa.CELL,sort=True).indices
    actual=sa.bounded_cell_indices(frame,chunk_rows=17)
    assert actual.keys()==expected.keys()
    for key in actual:
        np.testing.assert_array_equal(actual[key],expected[key])
    reference=frame[[*sa.CELL,'subject_id']].drop_duplicates()
    before=reference.groupby(sa.CELL).size().rename('n_known_original')
    after=reference.loc[reference.subject_id.isin(ids)].groupby(sa.CELL).size().rename('n_known_retained')
    wanted=before.to_frame().join(after).fillna({'n_known_retained':0}).reset_index()
    counts=sa.bounded_identity_counts(frame,ids,chunk_rows=17)
    pd.testing.assert_frame_equal(counts,wanted,check_dtype=False)
