import json

import pytest

from opera.run.clinical_figure_provenance import (
    NATIVE_FOUR, render_receipt, verify_clinical_render, training_artifact_with_contract,
)


def _report(tmp_path):
    (tmp_path / 'clinical_score.pdf').write_bytes(b'new clinical render')
    spec = dict(id='nccn_ipi', model='R_L', score_label='NCCN-IPI',
                score_column='nccn_ipi_score', group_order=NATIVE_FOUR,
                threshold_split='train', manuscript_figures=True)
    receipt = render_receipt(tmp_path, [spec], 'tuning')
    (tmp_path / 'report_manifest.json').write_text(json.dumps(receipt))
    return receipt


def test_accepts_current_training_based_native_render_and_detects_changed_bytes(tmp_path):
    _report(tmp_path)
    assert verify_clinical_render(tmp_path, expected_model='R_L')['valid']
    (tmp_path / 'clinical_score.pdf').write_bytes(b'old figure copied over the new one')
    result = verify_clinical_render(tmp_path, expected_model='R_L')
    assert not result['valid']
    assert any('content changed' in error for error in result['errors'])


@pytest.mark.parametrize('change,expected', [
    ('old_renderer', 'source changed'), ('tuning_threshold', 'training data'),
    ('three_groups', 'four native groups'), ('wrong_model', 'differs from selected'),
    ('extra_figure', 'output list'),
])
def test_rejects_stale_or_wrong_clinical_contract(tmp_path, change, expected):
    receipt = _report(tmp_path)
    if change == 'old_renderer':
        receipt['renderer_source_sha256'] = {}
    elif change == 'tuning_threshold':
        receipt['score_definitions'][0]['threshold_split'] = 'tuning'
    elif change == 'three_groups':
        receipt['score_definitions'][0]['group_order'] = ['Low', 'Intermediate', 'High']
    elif change == 'wrong_model':
        receipt['score_definitions'][0]['model'] = 'J_L'
    else:
        (tmp_path / 'old_copy.png').write_bytes(b'old')
    (tmp_path / 'report_manifest.json').write_text(json.dumps(receipt))
    result = verify_clinical_render(tmp_path, expected_model='R_L')
    assert not result['valid']
    assert any(expected in error for error in result['errors'])


def test_cpu_refresh_requires_exported_training_predictions_before_writing(tmp_path, monkeypatch):
    import yaml
    from opera.run import refresh_clinical_figures as refresh
    workspace = tmp_path / 'workspace'
    (workspace / 'analysis_inventory').mkdir(parents=True)
    (workspace / 'analysis_inventory/analysis.yaml').write_text(yaml.safe_dump(
        dict(opera_model='R_L', artifacts=[])))
    output = tmp_path / 'new'
    monkeypatch.setattr('sys.argv', ['refresh', '--workspace', str(workspace), '--output-dir', str(output)])
    with pytest.raises(ValueError, match='No existing primary-model training predictions'):
        refresh.main()
    assert not output.exists()


def test_cpu_refresh_rebuilds_native_groups_and_reuses_configured_training_export(tmp_path, monkeypatch):
    import numpy as np
    import pandas as pd
    import yaml
    from types import SimpleNamespace
    from opera.run import refresh_clinical_figures as refresh
    workspace = tmp_path / 'workspace'
    (workspace / 'analysis_inventory').mkdir(parents=True)
    meta = workspace / 'metadata.parquet'
    pd.DataFrame(dict(subject_id=range(60), cohort_fine='DLBCL', age_at_index=65,
                      IPI='Intermediate', index='NCCN-IPI',
                      nccn_ipi_score=[0, 2, 4, 7] * 15,
                      nccn_ipi_group='Intermediate')).to_parquet(meta)
    prediction = workspace / 'tuning.parquet'
    pd.DataFrame(dict(subject_id=range(60), outcome='overall_survival')).to_parquet(prediction)
    training = workspace / 'training.parquet'
    pd.DataFrame(dict(subject_id=range(60, 120), outcome='overall_survival')).to_parquet(training)
    np.savez(workspace / 'patient_embeddings.npz', embeddings=np.ones((60, 2)))
    analysis = dict(patient_metadata=str(meta), opera_model='R_L', artifacts=[
        dict(model='R_L', split='tuning', seed=42, path=str(prediction), data_contract='release-test')],
        clinical_threshold_artifacts=[dict(model='R_L', split='train', seed=42, path=str(training))])
    (workspace / 'analysis_inventory/analysis.yaml').write_text(yaml.safe_dump(analysis))
    output = tmp_path / 'new'
    commands = []
    def execute(command, **kwargs):
        commands.append(command)
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(refresh.subprocess, 'run', execute)
    monkeypatch.setattr('sys.argv', ['refresh', '--workspace', str(workspace), '--output-dir', str(output)])
    refresh.main()
    assert len(commands) == 1 and commands[0][2] == 'opera.run.render_clinical_score_panels'
    updated = yaml.safe_load((output / 'analysis_inventory/analysis.yaml').read_text())
    assert updated['clinical_threshold_artifacts'][0]['path'] == str(training)
    assert updated['clinical_threshold_artifacts'][0]['data_contract'] == 'release-test'
    recipe = yaml.safe_load((output / 'configs/clinical_score_panels.tuning.yaml').read_text())
    main, = [s for s in recipe['scores'] if s.get('manuscript_figures')]
    assert main['group_order'] == NATIVE_FOUR and main['threshold_split'] == 'train'
    assert set(pd.read_parquet(output / 'patient_metadata.parquet').nccn_ipi_group) == set(NATIVE_FOUR)


def test_training_contract_uses_matching_model_and_seed_and_rejects_conflicts():
    training = dict(model='R_L', seed=42, split='train', path='unchanged.parquet')
    analysis = dict(artifacts=[
        dict(model='R_L', seed=42, split='tuning', data_contract='registered-release'),
        dict(model='J_L', seed=42, split='tuning', data_contract='other-model')])
    spec = training_artifact_with_contract(training, analysis)
    assert spec['data_contract'] == 'registered-release'
    assert 'data_contract' not in training
    with pytest.raises(ValueError, match='declarations differ'):
        training_artifact_with_contract(dict(training, data_contract='wrong-release'), analysis)
    with pytest.raises(ValueError, match='one registered tuning data_contract'):
        training_artifact_with_contract(dict(training, seed=43), analysis)
    analysis['artifacts'].append(dict(model='R_L', seed=42, split='tuning', data_contract='conflict'))
    with pytest.raises(ValueError, match='one registered tuning data_contract'):
        training_artifact_with_contract(training, analysis)
