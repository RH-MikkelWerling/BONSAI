import gzip
import json
from pathlib import Path
from unittest.mock import patch
import pandas as pd
import pytest
from scripts.review_extended_results import read_summary
from scripts.package_extended_results import package
from opera.evaluation.distributed_analysis import save_pickle


def summary():
    return pd.DataFrame([dict(contrast='EXT_ADAPTATION',split='held_out',seed=42,metric='auc',
        stratum='all_cells',horizon_days=0,gain=.01,lower=0,upper=.02,n_cells=100)])


def test_reads_completed_balanced_worker_summaries_without_stage_tuple(tmp_path):
    base=tmp_path/'checkpoints';base.mkdir()
    (base/'active.json').write_text(json.dumps({'generation':'generation'}))
    queue=base/'generation/tasks/disease_balanced';queue.mkdir(parents=True)
    (queue/'queue.json').write_text(json.dumps({'total':1,'options':{'kind':'effect_summary','aggregation':'disease_balanced'}}))
    with pytest.raises(RuntimeError,match='not complete'):
        read_summary(tmp_path,'disease_balanced')
    save_pickle(queue/'batch_00000.result_INTERNAL.pkl',summary())
    result,_=read_summary(tmp_path,'disease_balanced')
    assert result.gain.iloc[0]==.01


def test_bundle_excludes_patient_outputs_and_preserves_aggregate_tables(tmp_path):
    project=tmp_path/'project';results=project/'results';results.mkdir(parents=True)
    summary().to_csv(results/'effect_summaries.csv',index=False)
    (results/'analysis_manifest.json').write_text(json.dumps({'config':{'opera_model':'J_OA_OWN_L'}}))
    pd.DataFrame({'subject_id':[1],'value':[2]}).to_csv(results/'unexpected.csv',index=False)
    pd.DataFrame({'subject_id':[1]}).to_csv(results/'patient_gains_INTERNAL.csv',index=False)
    output=tmp_path/'bundle'
    def render(command,check):
        directory=Path(command[command.index('--output-dir')+1]);directory.mkdir()
        (directory/'figure.pdf').write_bytes(b'figure')
    with patch('scripts.package_extended_results.subprocess.run',side_effect=render):
        archive=package(project,output)
    assert archive.is_file()
    assert not (output/'tables/unexpected.csv.gz').exists()
    assert not (output/'tables/patient_gains_INTERNAL.csv.gz').exists()
    with gzip.open(output/'tables/effect_summaries.csv.gz','rt') as handle:
        assert pd.read_csv(handle).gain.iloc[0]==.01
    manifest=json.loads((output/'bundle_manifest.json').read_text())
    assert manifest['remaining']['label_efficiency']
    assert package(project,output)==archive
