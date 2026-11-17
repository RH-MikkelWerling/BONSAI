import pandas as pd
from pandas.testing import assert_frame_equal
import pytest
from opera.evaluation.coordinator_reporting import annotate_gate,spool_rarity,spool_table
from opera.evaluation.secondary_analysis import annotate_router_gate
from opera.run.analyse_locked_predictions import rarity_inputs


@pytest.mark.parametrize('empty',[False,True])
def test_gate_matches_original_without_changing_input(empty):
    table=pd.DataFrame(dict(split=['held_out']*4,seed=[42]*4,metric=['auc']*4,
        horizon_days=[0,0,30,30],contrast=['R1','P1','R1','P1'],gain=[.1]*4),index=[4,4,2,1])
    before=table.copy(deep=True)
    gate=pd.DataFrame(dict(split=['held_out'],seed=[42],metric=['auc'],horizon_days=[0],status=['passed']))
    if empty:
        gate=gate.iloc[:0]
    assert_frame_equal(annotate_gate(table,gate,gated=['R1']),annotate_router_gate(table,gate,gated=['R1']))
    assert_frame_equal(table,before)


def test_reporting_spools_preserve_tables(tmp_path):
    table=pd.DataFrame(dict(metric=['auc','other','brier'],cohort=['a']*3,outcome=['o']*3,
        horizon_days=[30]*3,contrast=['P1']*3,gain=[.1,.2,.3],n_cases=[2]*3,n_controls=[5]*3))
    stored=spool_rarity(table,tmp_path/'rarity.csv',rarity_inputs,chunk_rows=1)
    stored.to_csv(tmp_path/'final.csv',index=False)
    assert_frame_equal(pd.read_csv(tmp_path/'final.csv'),rarity_inputs(table).reset_index(drop=True))
    assert len(stored)==2
    stored=spool_table(table,tmp_path/'table.csv')
    stored.to_csv(tmp_path/'copied.csv')
    assert_frame_equal(pd.read_csv(tmp_path/'copied.csv'),table)


@pytest.mark.parametrize('include_split',[False,True])
def test_coverage_matches_full_table_algorithm(include_split):
    from opera.evaluation.coordinator_reporting import coverage_bounded
    from opera.evaluation.secondary_analysis import coverage_inventory
    frame=pd.DataFrame(dict(model=['A','A','A','B','B','B'],seed=[42]*6,
        cohort=['c']*6,outcome=['o']*6,horizon_days=[30]*6,
        split=['tuning','held_out','held_out','held_out','held_out','tuning'],
        label=[1,1,0,0,1,1],ipcw_weight=[1,1,0,1,2,1],payload=['unused']*6))
    roster=frame[['model','seed','cohort','outcome','horizon_days','split']].drop_duplicates().assign(status='available')
    if not include_split:
        roster=roster.drop(columns='split').drop_duplicates()
    assert_frame_equal(coverage_bounded(frame,roster,chunk_rows=2),coverage_inventory(frame,roster))


def test_subgroup_dispatch_matches_original_payloads_and_results(tmp_path,monkeypatch):
    from opera.evaluation import distributed_analysis as da
    from opera.evaluation.coordinator_reporting import patient_subgroups_bounded
    from opera.evaluation.secondary_analysis import subgroup_patient_gains,panel_subgroup_gains
    gains=pd.DataFrame([dict(split='held_out',seed=42,cohort='c',outcome='o',horizon_days=30,
                             contrast=contrast,subject_id=sid,brier_gain=.1*sid,log_loss_gain=.2*sid)
                        for contrast in ['b','a'] for sid in range(4)])
    metadata=pd.DataFrame({'subject_id':range(4),'sex':['f','m','f','m']})
    specifications={'sex':{'provenance':'pre_index'}}
    draws={('held_out',42,'c'):({i:i for i in range(4)},__import__('numpy').ones((5,4)))}
    def immediate(directory,batches,total,options):
        count=0
        for path,meta,spec,refs in batches:
            part=da.load_pickle(path);resolved={key:da.load_pickle(value) for key,value in refs.items()}
            count+=1
            yield subgroup_patient_gains(part,meta,spec,draws=resolved),panel_subgroup_gains(part,meta,spec,draws=resolved)
        assert count==total
    monkeypatch.setattr(da,'distributed_parts',immediate)
    original=da.distributed_patient_subgroups(gains,metadata,specifications,draws,tmp_path/'old')
    bounded=patient_subgroups_bounded(gains,metadata,specifications,draws,tmp_path/'new')
    assert_frame_equal(original[0],bounded[0]);assert_frame_equal(original[1],bounded[1])
