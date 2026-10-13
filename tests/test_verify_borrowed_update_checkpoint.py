import hashlib
import inspect
import json
import sys

import torch
import yaml

from bonsai.functional.checkpointing import MODEL_INIT_CONFIG_KEY
from opera.modules.networks.source_anchored_readout import SourceAnchoredReadout
from opera.run.verify_borrowed_update_checkpoint import main


def test_saved_checkpoint_verification_checks_mode_and_borrowed_gradient_flow(tmp_path, monkeypatch, capsys):
    risk = {'readout_mode': 'unanchored_adapters', 'adapter_source_mode': 'both',
            'fine_names': ['A', 'B'], 'interval_boundaries_days': [30, 365],
            'no_competing_outcomes': [], 'adapter_query_mode': 'interacting'}
    readout = SourceAnchoredReadout(8, ['A', 'B'], ['x', 'y'], anchored=False, query_mode='interacting')
    readout.set_training_support([20] * 4, [10] * 4, [20] * 4)
    head = torch.nn.Linear(8, 12)
    state = {'model.source_adapter_readout.' + key: value for key, value in readout.state_dict().items()}
    state.update({'model.competing_risk_head.weight': head.weight.detach(),
                  'model.competing_risk_head.bias': head.bias.detach()})
    checkpoint = {'state_dict': state, 'epoch': 3,
                  'hyper_parameters': {MODEL_INIT_CONFIG_KEY: {'competing_risk_config': risk, 'cross_outcome_config': {}}}}
    (tmp_path / 'hydra').mkdir()
    (tmp_path / 'hydra/config.yaml').write_text(yaml.safe_dump(
        {'competing_risk': risk, 'outcomes': {'x': {}, 'y': {}}, 'cross_outcome': {}}))
    torch.save(checkpoint, tmp_path / 'best.ckpt')
    digest = hashlib.sha256(open(inspect.getfile(SourceAnchoredReadout), 'rb').read()).hexdigest()
    (tmp_path / 'router_code_provenance_INTERNAL.json').write_text(json.dumps({'loaded_class_file_sha256': digest}))
    monkeypatch.setattr(sys, 'argv', ['verify', '--run', str(tmp_path)])
    main()
    result = json.loads(capsys.readouterr().out)
    assert result['checkpoint_readout_mode'] == 'unanchored_adapters'
    assert result['borrowed_connectivity_verified'] is True
