import pandas as pd
import pytest
import yaml
from opera.run.refresh_clinical_table1 import refresh, CLINICAL
from opera.run.export_table1 import build_tables
from opera.run.compute_prognostic_scores import DEFAULT_COLUMNS, NCCN_EXTRANODAL_SITES, compute_scores


def test_final_population_and_count_explanation(tmp_path):
    lyfo=pd.DataFrame({'patientid':['a','b'],'age_diagnosis':[65,75],
        'LDH_diagnosis':[200,200],'AA_stage_diagnosis':[2,2],'PS_diagnosis':[0,0],
        'n_extranodal_regions_diagnosis':[2,0]})
    for column in NCCN_EXTRANODAL_SITES:
        lyfo[column]='No'
    lyfo.to_parquet(tmp_path/'lyfo.parquet',index=False)
    old_scores=compute_scores(lyfo,DEFAULT_COLUMNS)
    base=pd.DataFrame({'subject_id':['a','b'],'split':['train',None],
                       'cohort_fine':['DLBCL','DLBCL'],'age':[65,75]})
    old=base.merge(old_scores,on='subject_id')
    old['PS']='0'
    old['performance_status']='0–1'
    old.to_parquet(tmp_path/'old.parquet',index=False)
    source=tmp_path/'analysis.yaml'
    source.write_text(yaml.safe_dump({'patient_metadata':str(tmp_path/'old.parquet')}))
    new=base.merge(compute_scores(lyfo,{**DEFAULT_COLUMNS,'extranodal_sites':'n_extranodal_regions_diagnosis'}),on='subject_id')
    new['PS']='0'
    new['performance_status']='0–1'
    prepared=tmp_path/'prepared';prepared.mkdir()
    new.to_parquet(prepared/'patient_metadata.parquet',index=False)
    original=tmp_path/'table.csv'
    original_table=build_tables(old.loc[old.split.notna()])
    original_table.to_csv(original,index=False)
    result,report=refresh(original,prepared,tmp_path/'lyfo.parquet',source,tmp_path/'out')
    assert report['final_patients']==1
    assert report['excluded_without_registered_split']==1
    assert report['ipi_changed_scores_explained_by_count_substitution']==1
    result=pd.read_csv(tmp_path/'out/table1_values_long.csv')
    pd.testing.assert_frame_equal(result.loc[~result.variable.isin(CLINICAL)].reset_index(drop=True),
        pd.read_csv(original).loc[~original_table.variable.isin(CLINICAL)].reset_index(drop=True))
    bad=pd.read_csv(original);bad.loc[bad.variable.eq('patients'),'count']=5
    bad.to_csv(tmp_path/'bad.csv',index=False)
    with pytest.raises(ValueError,match='counts differ'):
        refresh(tmp_path/'bad.csv',prepared,tmp_path/'lyfo.parquet',source,tmp_path/'badout')
    quality=tmp_path/'quality.parquet'
    pd.DataFrame({'subject_id':['a'],'PS':[3]}).to_parquet(quality,index=False)
    updated,quality_report=refresh(original,prepared,tmp_path/'lyfo.parquet',source,
                                   tmp_path/'quality_out',quality)
    assert quality_report['performance_status']['changed_existing_ps']==1
    ps=updated.loc[updated.variable.eq('performance_status')]
    assert (ps.category.eq('2–4') & ps['count'].eq(1)).any()
    updated=pd.read_csv(tmp_path/'quality_out/table1_values_long.csv')
    stable=~updated.variable.isin(CLINICAL|{'PS','ps','performance_status'})
    original_frame=pd.read_csv(original)
    pd.testing.assert_frame_equal(updated.loc[stable].reset_index(drop=True),
        original_frame.loc[~original_frame.variable.isin(CLINICAL|{'PS','ps','performance_status'})].reset_index(drop=True))
    pd.DataFrame({'subject_id':['b'],'PS':[3]}).to_parquet(quality,index=False)
    with pytest.raises(ValueError,match='outside the frozen'):
        refresh(original,prepared,tmp_path/'lyfo.parquet',source,tmp_path/'outside',quality)
    pd.DataFrame({'subject_id':['a'],'PS':[99]}).to_parquet(quality,index=False)
    with pytest.raises(ValueError,match='ECOG'):
        refresh(original,prepared,tmp_path/'lyfo.parquet',source,tmp_path/'invalid_ps',quality)
