import numpy as np
import pandas as pd
import pytest
import torch
from pandas.testing import assert_frame_equal

from opera.analysis.representation_absorption import ARMS, cell_values, patient_half, quantities, rank_correlation
from opera.diagnostics.correction_replay import correction_ratios, corrected_representation


def fixture_cell():
    rows=[]
    ids=list(range(40))
    rng=np.random.default_rng(7)
    for pair in ARMS.values():
        for model in pair:
            for sid in ids:
                rows.append(dict(model=model,subject_id=sid,label=sid%2,
                                 probability=float(rng.uniform(.05,.95)),ipcw_weight=1.))
    return pd.DataFrame(rows),(dict(zip(ids,ids)),np.random.default_rng(4).exponential(size=(12,len(ids))))


def test_common_roster_and_exact_paired_arithmetic():
    frame,draws=fixture_cell()
    original=frame.copy(deep=True)
    values,status=cell_values(frame,draws,min_cases=1,min_controls=1)
    assert status=='supported'
    assert {'full','half0','half1'} <= values.keys()
    q=quantities(values['full'],'EA')
    assert q['closure'][0]==pytest.approx(q['gap_local'][0]-q['gap'][0])
    np.testing.assert_allclose(q['closure'][1],q['gap_local'][1]-q['gap'][1])
    assert_frame_equal(frame,original)
    frame=frame.drop(frame.index[0])
    assert cell_values(frame,draws,min_cases=1,min_controls=1)[1]=='different_patient_roster'


def test_disjoint_halves_and_constant_correlation():
    assert {patient_half(i) for i in range(40)}=={0,1}
    assert all(patient_half(i)==patient_half(i) for i in range(40))
    assert np.isnan(rank_correlation(np.ones(5),np.arange(5)))
    assert rank_correlation(np.arange(5),np.arange(5))==pytest.approx(1)


def test_correction_norms_and_zeroing_include_global():
    representation=torch.tensor([[[3.,4.],[0.,0.]]])
    terms={'global':torch.ones_like(representation),'own_disease':2*torch.ones_like(representation)}
    result=correction_ratios(representation,terms,.5)
    assert result['global'][0,0]==pytest.approx(np.sqrt(2)*.5/5)
    assert torch.isnan(result['sum'][0,1])
    assert torch.equal(corrected_representation(representation,terms,.5,'all'),representation)
    torch.testing.assert_close(corrected_representation(representation,terms,.5,'own_disease'),representation+.5)
    torch.testing.assert_close(corrected_representation(representation,terms,.5,''),representation+1.5)


def test_snapshot_pipeline_keeps_common_cells_and_reports_missing_arm(tmp_path):
    from opera.evaluation.distributed_analysis import save_prepared
    from opera.analysis.representation_absorption import run
    frame,weights=fixture_cell()
    parts=[]
    for outcome in ['a','b','c']:
        cell=frame.assign(split='tuning',seed=42,cohort='d',outcome=outcome,horizon_days=30)
        parts.append(cell)
    parts.append(parts[0].assign(outcome='missing').loc[lambda d:d.model.ne('R_EA')])
    all_rows=pd.concat(parts,ignore_index=True)
    prepared=tmp_path/'prepared'
    save_prepared(prepared,all_rows,dict(draws={('tuning',42,'d'):weights},
        performance_support={'min_cases':1,'min_controls':1},hashes={}),chunk_rows=130)
    metadata=tmp_path/'metadata.csv'
    pd.DataFrame(dict(cohort=['d']*3,outcome=['a','b','c'],horizon_days=[30]*3,
                      n_events_train=[10,20,40],outcome_family=['f']*3)).to_csv(metadata,index=False)
    output=tmp_path/'diagnostics'
    result=run(prepared,output,cell_metadata=metadata)
    assert result['n_cells']==3
    assert pd.read_csv(output/'cell_effects.csv').shape[0]==6
    assert 'missing_arm' in set(pd.read_csv(output/'roster_exclusions.csv').reason)
    assert (output/'absorption_EA.pdf').exists()
    assert (output/'closure_vs_level_EA.pdf').exists()
    assert len(pd.read_csv(output/'absorption_correlations.csv'))>0
    with pytest.raises(FileExistsError):
        run(prepared,output)
    with pytest.raises(ValueError,match='held-out'):
        run(prepared,tmp_path/'heldout',split='held_out')


@pytest.mark.parametrize('with_ipcw',[False,True])
def test_auc_only_matches_main_metric_with_ties(with_ipcw):
    from opera.analysis.representation_absorption import bootstrap_auc
    from opera.evaluation.secondary_analysis import bootstrap_metric_values
    y=np.array([0,1,1,0,1,0.])
    p=np.array([.1,.5,.5,.8,.9,.9])
    weights=np.random.default_rng(2).exponential(size=(140,6))
    weights[0]=0
    ipcw=np.array([1,2,0,1,3,1.]) if with_ipcw else None
    np.testing.assert_allclose(bootstrap_auc(y,p,weights,ipcw),
        bootstrap_metric_values(y,p,weights,ipcw=ipcw)['auc'],equal_nan=True)


def test_scarcity_contract_rejects_unmatched_subsets(tmp_path):
    import json
    import yaml
    from opera.analysis.representation_absorption import verify_scarcity_contracts
    config={'training':{'train_patient_fraction':.1,'train_fraction_seed':42},'seed':42,'dapt_ckpt':'fixed'}
    a,b=tmp_path/'joint.yaml',tmp_path/'opera.yaml'
    a.write_text(yaml.safe_dump(config));b.write_text(yaml.safe_dump(config))
    plan=tmp_path/'contracts.json'
    plan.write_text(json.dumps({'10':{origin:{'joint_config':str(a),'opera_config':str(b)} for origin in ['L','EA']}}))
    assert verify_scarcity_contracts(plan)[0]=={10}
    config['training']['train_fraction_seed']=7
    b.write_text(yaml.safe_dump(config))
    with pytest.raises(ValueError,match='patient-subset'):
        verify_scarcity_contracts(plan)
