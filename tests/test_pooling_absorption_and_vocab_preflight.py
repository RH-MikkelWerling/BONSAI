import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from opera.analysis.pooling_absorption import run as pooling_run, pooling_quantities
from opera.diagnostics.vocabulary_population_preflight import coverage, run as audit_run, timing_estimates


def test_pooling_closure_uses_paired_replicates():
    arms={e:dict(pooled='D_'+e,joint='J_'+e,specialist='S_'+e) for e in ['L','EA']}
    values={m:(i/20,np.array([i/20,i/15])) for i,m in enumerate(['D_L','J_L','S_L','D_EA','J_EA','S_EA'])}
    q=pooling_quantities(values,arms)
    a,b=q['L','pooled','gap'],q['EA','pooled','gap']
    assert q['EA','pooled','closure'][0]==pytest.approx(a[0]-b[0])
    np.testing.assert_allclose(q['EA','pooled','closure'][1],a[1]-b[1])


def test_pooling_missing_specialists_are_blocked_not_replaced(tmp_path):
    from opera.evaluation.distributed_analysis import save_prepared
    rng=np.random.default_rng(9);rows=[]
    for outcome in ['a','b','c','d']:
        for model in ['D_L','J_L','S_L','J_E','J_EA']:
            for sid in range(20):
                rows.append(dict(split='tuning',seed=42,cohort='d',outcome=outcome,horizon_days=30,
                                 model=model,subject_id=sid,label=sid%2,probability=rng.uniform(),ipcw_weight=1.))
    prepared=tmp_path/'prepared'
    save_prepared(prepared,pd.DataFrame(rows),dict(draws={('tuning',42,'d'):(dict(zip(range(20),range(20))),rng.exponential(size=(8,20)))},
        performance_support=dict(min_cases=1,min_controls=1),hashes={}))
    meta=tmp_path/'meta.csv'
    pd.DataFrame(dict(cohort=['d']*4,outcome=['a','b','c','d'],horizon_days=[30]*4,n_events_train=[5,10,20,40])).to_csv(meta,index=False)
    out=tmp_path/'out';pooling_run(prepared,out,cell_metadata=meta)
    blocked=pd.read_csv(out/'blocked.csv')
    assert set(blocked.encoder)=={'E','EA'}
    assert 'S_EA' in blocked.loc[blocked.encoder=='EA','missing_models'].iloc[0]
    summaries=pd.read_csv(out/'pooling_summaries.csv')
    assert set(summaries.encoder)=={'L'}
    assert set(summaries.quantity)=={'gap'}
    assert set(summaries[summaries.stratum=='overall'].n_cells)=={4}
    assert 'events_quartile' in set(summaries.stratum)
    assert (out/'pooling_gap.pdf').exists()
    with pytest.raises(ValueError,match='Held-out'):
        pooling_run(prepared,tmp_path/'heldout',split='held_out')


def test_coverage_is_event_weighted_and_never_guesses_code_parents():
    d,s=coverage({'DX//long':8,'DX//short':2,'[UNK]':1},{'DX//short':0,'[UNK]':1})
    assert s['direct_known_event_share']==pytest.approx(2/11)
    assert not d.set_index('code').loc['DX//long','mapped_known']
    _,mapped=coverage({'DX//long':8,'DX//short':2},{'DX//short':0},{'DX//long':'DX//short'})
    assert mapped['mapped_known_event_share']==1


def test_pooling_uses_one_roster_for_all_encoders_and_scopes(tmp_path):
    from opera.evaluation.distributed_analysis import save_prepared
    rows=[]
    for outcome in ['complete','mismatched']:
        for encoder in ['L','E','EA']:
            for role in ['D','J','S']:
                model=f'{role}_{encoder}'
                for sid in range(20):
                    if outcome=='mismatched' and model=='S_EA' and sid==0:
                        continue
                    rows.append(dict(split='tuning',seed=42,cohort='d',outcome=outcome,horizon_days=30,
                        model=model,subject_id=sid,label=sid%2,probability=(sid%2)*.5+sid/100,ipcw_weight=1.))
    prepared=tmp_path/'prepared'
    save_prepared(prepared,pd.DataFrame(rows),dict(draws={('tuning',42,'d'):(dict(zip(range(20),range(20))),np.ones((8,20)))},
        performance_support=dict(min_cases=1,min_controls=1),hashes={}))
    out=tmp_path/'out';pooling_run(prepared,out)
    cells=pd.read_csv(out/'pooling_cells.csv')
    assert set(cells.outcome)=={'complete'}
    assert set(cells.encoder)=={'L','E','EA'}
    assert len(cells[cells.quantity=='closure'])==4
    assert np.allclose(cells.gain,0)
    assert set(pd.read_csv(out/'roster_exclusions.csv').reason)=={'different_patient_roster'}


def test_preflight_cutoff_static_and_timing(tmp_path):
    records=tmp_path/'records.parquet'
    pd.DataFrame(dict(code=['A','B','A','STATIC'],time=pd.to_datetime(['2021-01-01','2021-12-31','2022-01-01',None]))).to_parquet(records)
    vocab=tmp_path/'vocab.json';vocab.write_text(json.dumps({'[UNK]':0,'A':1,'STATIC':2}))
    measurement=tmp_path/'timing.log';measurement.write_text('fixture measured timing')
    reference=dict(measurement_file=str(measurement),reference_seconds=3600,reference_optimizer_steps=100,
        target_optimizer_steps=200,reference_vocab_size=100,target_vocab_size=200,gpu='fixture',
        same_architecture_batch_sequence_and_gpu=True)
    times=timing_estimates({'pretraining':reference,'joint':reference})
    assert times.iloc[0].step_scaled_gpu_hours==2
    assert times.iloc[0].vocab_linear_sensitivity_gpu_hours==4
    assert times.iloc[1].vocab_linear_sensitivity_gpu_hours==2
    assert times.iloc[2].status=='blocked_missing_timing_reference'
    result=audit_run([records],vocab,tmp_path/'out')
    assert result['coverage']['events']==2
    assert result['coverage']['direct_known_event_share']==.5
    assert result['coverage']['excluded_post_cutoff_rows']==1
    assert result['training_launched'] is False
    assert result['approval_required'] is True
