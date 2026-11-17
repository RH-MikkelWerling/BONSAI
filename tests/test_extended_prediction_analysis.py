import json
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml

from opera.run.prepare_extended_prediction_analysis import prepare


def test_preserves_primary_and_includes_registry(tmp_path):
    source = tmp_path / 'source.yaml'
    source.write_text(yaml.safe_dump(dict(
        artifacts=[dict(model=m, split='held_out', seed=42, data_contract='frozen')
                   for m in ['J_L', 'J_OA_OWN_L', 'S_L', 'X', 'XP', 'XR']],
        opera_model='J_OA_OWN_L', contrast_manifest='original.yaml')))
    root = tmp_path / 'national'
    def rows(bundle):
        origin = 'E' if Path(bundle).name == 'direct_fits' else 'EA'
        result = []
        for name in ['J', 'R', 'AB', 'OFF']:
            directory = root / 'predictions' / origin / name / 'held_out'
            directory.mkdir(parents=True)
            (directory / 'evaluation_complete.json').write_text('{}')
            (directory / 'direct_cr_predictions.parquet').touch()
            result.append((dict(seed=42, phase='external', origin=origin,
                                name='seed42/' + name, run_directory=f'{origin}/{name}'),
                           None, {'competing_risk': {'readout_mode': 'shared' if name == 'J' else 'own'}}))
        return {}, result
    with patch('opera.run.prepare_extended_prediction_analysis.bundle_runs', side_effect=rows):
        config = prepare(source, root, tmp_path / 'output', 'held_out')
    assert config['opera_model'] == 'J_OA_OWN_L'
    assert config['extra_contrasts']['EXT_R_EA_VS_XP'] == 'gain(R_EA,XP)'
    assert config['extra_contrasts']['EXT_R_E_VS_XR'] == 'gain(R_E,XR)'
    assert config['extra_contrasts']['EXT_R_EA_VS_S_L'] == 'gain(R_EA,S_L)'
    assert len(config['artifacts']) == 14
    assert json.loads((tmp_path / 'output/preparation_report.json').read_text())['registry_augmented_present']
    with pytest.raises(FileExistsError):
        prepare(source, root, tmp_path / 'output', 'held_out')


def test_pending_ablations_do_not_block_but_candidates_do(tmp_path):
    source = tmp_path / 'source.yaml'
    source.write_text(yaml.safe_dump(dict(
        artifacts=[dict(model=m, split='tuning', seed=42, data_contract='frozen')
                   for m in ['J_L', 'J_OA_OWN_L', 'X', 'XP']],
        opera_model='J_OA_OWN_L')))
    root = tmp_path / 'national'
    def rows(bundle):
        origin = 'E' if Path(bundle).name == 'direct_fits' else 'EA'
        result = []
        for name in ['R', 'NODIS', 'J']:
            directory = root / 'predictions' / origin / name / 'tuning'
            if name == 'R':
                directory.mkdir(parents=True, exist_ok=True)
                (directory / 'evaluation_complete.json').write_text('{}')
                (directory / 'direct_cr_predictions.parquet').touch()
            result.append((dict(seed=42, phase='external', origin=origin,
                                name='seed42/' + name, run_directory=f'{origin}/{name}'),
                           None, {'competing_risk': {'readout_mode': 'shared' if name == 'J' else 'own'}}))
        return {}, result
    with patch('opera.run.prepare_extended_prediction_analysis.bundle_runs', side_effect=rows):
        config = prepare(source, root, tmp_path / 'output', 'tuning')
    assert 'EXT_ADAPTATION' in config['extra_contrasts']
    assert 'EXT_ARCH_EA' not in config['extra_contrasts']
    report = json.loads((tmp_path / 'output/preparation_report.json').read_text())
    assert len(report['pending_evaluations']) == 4
    def missing_candidate(bundle):
        metadata, result = rows(bundle)
        return metadata, [r for r in result if r[0]['origin'] != 'EA' or r[0]['name'] != 'seed42/R']
    with patch('opera.run.prepare_extended_prediction_analysis.bundle_runs', side_effect=missing_candidate):
        with pytest.raises(ValueError, match='Missing selection candidate'):
            prepare(source, root, tmp_path / 'blocked', 'tuning')
    assert not (tmp_path / 'blocked').exists()
