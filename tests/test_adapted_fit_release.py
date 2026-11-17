import json
import yaml
from opera.run import prepare_adapted_fit_release as release
from opera.run.paper_prediction_queue import canonical_models


def test_fraction_aliases_and_audit(tmp_path,monkeypatch):
    source=tmp_path/'adapted_fits_pending';source.mkdir()
    names=['J','R','AB','OFF','NODIS','NOOUT','J_frac10','J_frac25','J_frac50']
    rows=[dict(name=f'seed42/EA/{n}',origin='EA',seed=42,phase='external',condition={}) for n in names]
    (source/'bundle.json').write_text(json.dumps({'runs':rows}))
    def load(row,path):
        mode=row['name'].rsplit('/',1)[-1]
        config={'dapt_ckpt':'adapted','paths':{'vocabulary':'vocab'},
                'supervision_factorial':{'regime':'J'},'competing_risk':{'readout_mode':'shared'},
                'training':{}}
        if 'frac' in mode:
            config['training']['train_patient_fraction']=int(mode.split('frac')[1])/100
        return None,config,config
    monkeypatch.setattr(release,'load_run_config',load)
    calls=[]
    monkeypatch.setattr(release,'audit_encoder',lambda *a,**k: calls.append(k) or {'status':'passed'})
    monkeypatch.setattr(release,'audit_dapt_to_opera',lambda *a: {})
    release.prepare(tmp_path)
    report=json.loads((tmp_path/'adapted_fits/bundle.json').read_text())
    assert report['phase_counts']=={'external':6,'data_efficiency':3}
    assert calls[0]['minimum_dapt_epochs']==10
    for row in report['runs']:
        if row['phase']=='data_efficiency':
            config=yaml.safe_load((tmp_path/'adapted_fits'/row['config_path']).read_text())
            assert canonical_models(row,config,report)==['J_EA_F'+row['name'].split('frac')[1]]
