import json
import yaml
import pytest
from opera.run.prepare_national_tuning_analysis import prepare, MODELS, BASELINES


def test_tuning_only_and_four_national_arms(tmp_path):
    source=tmp_path/'source.yaml'
    artifacts=[dict(path=f'{m}_{split}.parquet',model=m,split=split,seed=42,data_contract='fixed')
               for m in BASELINES for split in ['tuning','held_out']]
    source.write_text(yaml.safe_dump(dict(artifacts=artifacts,estimand='cif_ipcw',
        bootstrap_replicates=1000,bootstrap_seed=2026,bootstrap_method='multiplier',
        analysis_horizons_days=[30,60,90,180,365,730])))
    root=tmp_path/'national';(root/'direct_fits').mkdir(parents=True)
    rows=[]
    for mode in MODELS:
        run=f'E/seed42/{mode}'
        rows.append(dict(name=f'seed42/E/{mode}',origin='E',seed=42,run_directory=run))
        directory=root/'predictions'/run/'tuning';directory.mkdir(parents=True)
        (directory/'evaluation_complete.json').write_text('{}')
        (directory/'direct_cr_predictions.parquet').touch()
    (root/'direct_fits/bundle.json').write_text(json.dumps({'runs':rows}))
    config=prepare(source,root,tmp_path/'output')
    assert all(a['split']=='tuning' for a in config['artifacts'])
    assert {a['model'] for a in config['artifacts']}==BASELINES|set(MODELS.values())
    assert config['bootstrap_replicates']==1000
    assert config['bootstrap_method']=='multiplier'
    assert config['opera_model']=='J_OA_OWN_L'
    assert config['extra_contrasts']['NATIONAL_OPERA_XGB']=='gain(R_E,X)'
    (root/'predictions/E/seed42/OFF/tuning/evaluation_complete.json').unlink()
    with pytest.raises(ValueError,match='Incomplete'):
        prepare(source,root,tmp_path/'incomplete')
