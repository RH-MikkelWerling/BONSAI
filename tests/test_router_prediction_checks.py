import numpy as np
import pandas as pd
import pytest
import torch

from opera.diagnostics.router_prediction_checks import compare_saved_predictions, component_cifs
from test_router_axis_separation import _model


def test_replay_detects_risk_and_followup_changes():
    frame = pd.DataFrame({'subject_id': [1, 2], 'cohort': ['A', 'B'], 'outcome': ['x', 'x'],
                          'time_days': [10., 20.], 'event': [1, 0], 'cif_30d': [.2, .4]})
    assert compare_saved_predictions(frame, frame.iloc[:1])['passed']
    changed = frame.copy()
    changed.loc[0, 'cif_30d'] += .02
    assert not compare_saved_predictions(frame, changed)['passed']
    changed.loc[0, 'event'] = 0
    with pytest.raises(ValueError, match='follow-up'):
        compare_saved_predictions(frame, changed)


def test_zero_borrowing_preserves_full_when_borrowed_terms_are_zero():
    model = _model('both', adapter_global=True).eval()
    patient = torch.randn(2, 16)
    fine = torch.tensor([0, 1])
    # Default zero-initialized adapter ups make all corrections zero.
    result = component_cifs(model, patient, fine, [30, 365])
    torch.testing.assert_close(result['encoder_only'], result['no_borrowing_zero'])
    with torch.no_grad():
        model.source_adapter_readout.up.normal_(std=.3)
    result = component_cifs(model, patient, fine, [30, 365])
    assert not torch.allclose(result['encoder_only'], result['no_borrowing_zero'])
    assert torch.isfinite(result['no_borrowing_zero']).all()


def test_zero_borrowing_matches_explicit_source_zeroing_without_gate_changes():
    from opera.modules.networks.competing_risk import piecewise_exponential_cif
    model = _model('both', adapter_global=True).eval()
    patient, fine = torch.randn(1, 16), torch.tensor([0])
    readout = model.source_adapter_readout
    with torch.no_grad():
        readout.up.normal_(std=.3)
        readout.global_up.normal_(std=.3)
        expected = component_cifs(model, patient, fine, [30, 365])['no_borrowing_zero']
        weights = readout.route_weights(patient, fine).clone()
        # For disease A, target x: disease B and outcome y are borrowed.
        readout.up[[1, 3]] = 0
        torch.testing.assert_close(weights, readout.route_weights(patient, fine))
        hazards, _ = model.competing_risk_log_hazards(patient, fine, return_routes=True)
        objective = model.competing_risk_loss
        actual = piecewise_exponential_cif(hazards, objective.boundaries * objective.time_scale_days,
                                          [30, 365], time_scale_days=objective.time_scale_days)
        torch.testing.assert_close(expected[:, 0], actual[:, 0])


def test_verification_command_runs_both_checkpoints_into_fresh_outputs(tmp_path, monkeypatch):
    import json
    import yaml
    from opera.run import verify_router_predictions as verify
    inventory = tmp_path / 'analysis_inventory'
    inventory.mkdir()
    metadata = tmp_path / 'metadata.parquet'
    pd.DataFrame({'subject_id': [1, 2], 'cohort_fine': ['A', 'B']}).to_parquet(metadata)
    (inventory / 'analysis.yaml').write_text(yaml.safe_dump({'patient_metadata': str(metadata)}))
    env = tmp_path / '.env'
    env.write_text(f'OPERA_PAPER_ROOT={tmp_path}\nOPERA_PAPER_ANALYSIS_ROOT={tmp_path}\n')
    rows = []
    frame = pd.DataFrame({'subject_id': [1, 2], 'cohort': ['A', 'B'], 'outcome': ['x', 'x'],
                          'time_days': [10., 20.], 'event': [1, 0], 'cif_30d': [.2, .4]})
    for model in ['J_L', 'R_L']:
        path = tmp_path / f'{model}.parquet'
        frame.to_parquet(path)
        rows.append(dict(model=model, split='tuning', predictions=str(path),
                         config_path=str(tmp_path / 'frozen.yaml'), checkpoint_path=str(tmp_path / f'{model}.ckpt'),
                         **{key: True for key in ['bundle_config_hash_matches',
                            'evaluation_matches_selected_checkpoint', 'checkpoint_hash_matches', 'prediction_hash_matches']}))
    monkeypatch.setattr(verify, 'audit', lambda *a, **kw: {'artifacts': rows})
    monkeypatch.setattr(verify, 'reconcile_metrics', lambda *a: [])
    commands = []

    def replay(command, **kwargs):
        commands.append(command)
        target = verify.Path(command[command.index('--output-dir') + 1])
        target.mkdir()
        saved = frame.copy()
        for component in ['encoder_only', 'no_borrowing_zero']:
            saved[f'audit_{component}_30d'] = saved.cif_30d - .01
        saved.to_parquet(target / 'direct_cr_predictions.parquet')

    monkeypatch.setattr(verify.subprocess, 'run', replay)
    output = tmp_path / 'fresh'
    monkeypatch.setattr(verify.sys, 'argv', ['verify', '--env-file', str(env), '--output-dir', str(output)])
    verify.main()
    report = json.loads((output / 'report.json').read_text())
    assert report['status'] == 'replay_passed'
    assert len(commands) == 2
    assert '--audit-components' in commands[1]
    assert commands[1][commands[1].index('--checkpoint') + 1] == str(tmp_path / 'R_L.ckpt')
    assert report['router_components']['no_borrowing_zero']['30']['maximum'] > 0
    with pytest.raises(FileExistsError):
        verify.main()
