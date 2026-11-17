import json
import yaml
from opera.run import prepare_national_fit_handoff as handoff


def test_matched_national_queues(tmp_path, monkeypatch):
    local=tmp_path/'local'
    local.mkdir()
    rows=[dict(name=f'seed42/L/{s}',scientific_hash=m) for m,s in handoff.SOURCES.items()]
    (local/'bundle.json').write_text(json.dumps({'runs':rows}))
    def load(row, directory):
        config={'paths':{'vocabulary':'old'},'cohorts':{'A':{'data_dir':'fixed_outcomes','subject_data_paths':['old']}},
            'training':{'epochs':30,'marker':row['name']},'supervision_factorial':{'regime':'J'}}
        return None,config,config
    monkeypatch.setattr(handoff,'load_run_config',load)
    root=tmp_path/'national'
    handoff.prepare(local,root,tmp_path/'sequences',tmp_path/'direct.ckpt',tmp_path/'adapted.ckpt',0.1)
    direct=json.loads((root/'direct_fits/bundle.json').read_text())
    pending=json.loads((root/'adapted_fits_pending/bundle.json').read_text())
    assert len(direct['runs'])==4
    assert len(pending['runs'])==9
    for name, bundle in [('direct_fits',direct),('adapted_fits_pending',pending)]:
        for row in bundle['runs']:
            config=yaml.safe_load((root/name/row['config_path']).read_text())
            assert config['cohorts']['A']['data_dir']=='fixed_outcomes'
            assert config['training']['epochs']==30
            assert config['training']['encoder_lr_multiplier']==0.1
            assert config['paths']['vocabulary']==str(tmp_path/'sequences/vocabulary.pt')
            assert config['dapt_ckpt']==str(tmp_path/('direct.ckpt' if row['origin']=='E' else 'adapted.ckpt'))
    fractions=[yaml.safe_load((root/'adapted_fits_pending'/r['config_path']).read_text())['training']['train_patient_fraction']
               for r in pending['runs'] if 'frac' in r['name']]
    assert fractions==[0.1,0.25,0.5]
