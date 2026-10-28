import hashlib
import json
import pandas as pd
import yaml
from opera.run import prepare_attribution_refresh as refresh


def test_prepares_selected_fit_tuning_ig_without_execution(tmp_path, monkeypatch):
    root = tmp_path / 'run'; root.mkdir()
    config = root / 'fit.yaml'; config.write_text('selected')
    checkpoint = root / 'best.ckpt'; checkpoint.write_bytes(b'locked')
    predictions = root / 'predictions.parquet'
    pd.DataFrame(dict(outcome=['overall_survival'])).to_parquet(predictions)
    fit = dict(config=str(config),checkpoint=str(checkpoint),predictions=str(predictions),
        config_sha256=hashlib.sha256(config.read_bytes()).hexdigest(),
        checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest())
    (root / 'completion.json').write_text(json.dumps(dict(inputs=dict(fit=fit))))
    (root / 'overnight.env').write_text('OPERA_PAPER_ROOT=/original\n')
    output = tmp_path / 'ig'
    monkeypatch.setattr('sys.argv',['refresh','--analysis-root',str(root),'--output-dir',str(output)])
    refresh.main()
    cfg=yaml.safe_load((output / 'attribution.yaml').read_text())
    assert cfg['run']['split']=='tuning'
    assert cfg['run']['check_outcomes']==['overall_survival']
    manifest=json.loads((output / 'refresh_manifest.json').read_text())
    assert manifest['fit']==fit
    assert [c[3] for c in manifest['commands']]==['attributions','attribution-summary']
