import json
import pandas as pd
import pytest
import yaml
from opera.run.prepare_clinical_registry_refresh import prepare


def test_refresh_keeps_multiple_indices_and_population(tmp_path):
    base = pd.DataFrame({'subject_id':['a','b'], 'cohort_fine':['DLBCL','CLL'],
                         'age_at_index':[65,75], 'split':['held_out','train'],
                         'history_event_count':[123,456], 'index':['old','old'], 'IPI':['low','low']})
    lyfo = pd.DataFrame({'patientid':['a','b'], 'age_diagnosis':[65,75],
        'LDH_diagnosis':[200,200], 'AA_stage_diagnosis':[2,2], 'PS_diagnosis':[0,0],
        'n_extranodal_regions_diagnosis':[2,0]})
    from opera.run.compute_prognostic_scores import NCCN_EXTRANODAL_SITES
    for column in NCCN_EXTRANODAL_SITES:
        lyfo[column] = ['No','No']
    indices = pd.DataFrame({'patientid':['a','a','b'], 'index':['R-IPI','MIPI','CLL-IPI'],
                           'ipi':['low','high','low'], 'disease':['DLBCL','DLBCL','CLL']})
    for name, frame in [('base',base),('lyfo',lyfo),('indices',indices)]:
        frame.to_parquet(tmp_path / f'{name}.parquet', index=False)
    config = {'patient_metadata': str(tmp_path/'base.parquet'), 'artifacts':[{'path':'fixed'}],
              'clinical_threshold_artifacts':[{'path':'fixed_train'}]}
    (tmp_path/'analysis.yaml').write_text(yaml.safe_dump(config))
    recipe = 'opera/configs/clinical_score_panels.release_20261109.yaml'
    from opera.run.export_table1 import build_tables
    original_table = build_tables(base)
    original_table.to_csv(tmp_path/'table1.csv', index=False)
    refreshed, result, report = prepare(tmp_path/'analysis.yaml', tmp_path/'lyfo.parquet',
        tmp_path/'indices.parquet', recipe, tmp_path/'refresh', tmp_path/'table1.csv')
    table = pd.read_csv(tmp_path/'refresh/table1_values_long.csv')
    assert table.loc[table.variable.eq('patients'), 'count'].tolist() == original_table.loc[original_table.variable.eq('patients'), 'count'].tolist()
    assert refreshed.subject_id.tolist() == ['a','b']
    assert refreshed.ipi_computed_score.tolist() == [2,1]  # age plus actual count, not seven-site sum
    assert refreshed.loc[0, report['index_columns']['R-IPI']] == 'low'
    assert refreshed.loc[0, report['index_columns']['MIPI']] == 'high'
    assert report['multi_index_patients'] == 1
    assert len(refreshed) == 2
    saved = yaml.safe_load((tmp_path/'refresh/analysis.yaml').read_text())
    assert saved['artifacts'] == config['artifacts']
    assert saved['clinical_threshold_artifacts'] == config['clinical_threshold_artifacts']
    assert all('index_column' not in spec for spec in result['scores'])
    assert any(spec['cohorts'] == ['DLBCL'] and spec.get('score_column') == 'nccn_ipi_score'
               for spec in result['scores'])
    from opera.run.render_clinical_score_panels import expand_index_scores
    assert expand_index_scores(result, refreshed) == result['scores']
    with pytest.raises(FileExistsError):
        prepare(tmp_path/'analysis.yaml', tmp_path/'lyfo.parquet', tmp_path/'indices.parquet',
                recipe, tmp_path/'refresh')
