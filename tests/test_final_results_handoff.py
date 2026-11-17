import hashlib
import importlib.util
import json
from pathlib import Path
from zipfile import ZipFile

import pandas as pd
import pytest
import yaml


def test_final_ig_preparation_binds_adapted_fit_and_reuses_same_recipe(tmp_path, monkeypatch):
    import bonsai.functional.checkpointing as checkpointing
    from opera.run.run_final_interpretation import prepare
    root = tmp_path / 'national'; bundle = root / 'adapted_fits'; bundle.mkdir(parents=True)
    config = bundle / 'R.yaml'; config.write_text('seed: 42\n')
    (bundle / 'bundle.json').write_text(json.dumps({'runs': [{
        'name': 'seed42/EA/R', 'config_path': 'R.yaml', 'run_directory': 'EA/fit',
        'config_sha256': hashlib.sha256(config.read_bytes()).hexdigest()}]}))
    fit = root / 'neural_runs/EA/fit'; fit.mkdir(parents=True)
    (fit / 'best.ckpt').write_bytes(b'frozen fit')
    monkeypatch.setattr(checkpointing, 'verified_training_selection',
                        lambda path: {'best_checkpoint_sha256': 'selected-checkpoint'})
    predictions = root / 'predictions/EA/fit/held_out'; predictions.mkdir(parents=True)
    pd.DataFrame({'outcome': ['overall_survival', 'neutropenia_g3plus']}).to_parquet(
        predictions / 'direct_cr_predictions.parquet')
    base_env = tmp_path / 'base.env'; base_env.write_text('OPERA_PAPER_ROOT=/old/local\nSECRET=do-not-export\n')
    output = root / 'interpretation'
    first = prepare(root, output, base_env)
    recipe = yaml.safe_load((output / 'attribution.yaml').read_text())
    assert first['fit']['model'] == 'R_EA'
    assert recipe['run']['split'] == 'held_out'
    assert recipe['run']['outcomes'] == 'all'
    assert recipe['run']['check_outcomes'] == ['overall_survival', 'neutropenia_g3plus']
    assert 'do-not-export' not in (output / 'interpretation.env').read_text()
    assert prepare(root, output, base_env) == first
    config.write_text('seed: 43\n')
    with pytest.raises(ValueError, match='config changed'):
        prepare(root, output, base_env)


def test_shipper_exports_existing_matching_tables_and_no_patient_inputs(tmp_path):
    spec = importlib.util.spec_from_file_location('ship', Path('scripts/ship_paper_results.py'))
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    source = tmp_path / 'data/paper_256_v3_national_20261111/clinical_20261115/R_EA/clinical_results'
    source.mkdir(parents=True)
    (source / 'report_manifest.json').write_text('{}')
    (source / 'evaluation_km_matched_groups.csv').write_text('end,additional_n\nlow,100\n')
    (source / 'evaluation_km_frontiers.csv').write_text('end,model_n\nlow,200\n')
    (source / 'patients_INTERNAL.parquet').write_bytes(b'private')
    (source / 'patients.csv').write_text('subject_id,model_risk\nprivate,.1\n')
    output = tmp_path / 'handoff'
    module.ship(tmp_path, output)
    with ZipFile(output / 'clinical/R_EA.zip') as clinical:
        assert 'evaluation_km_matched_groups.csv' in clinical.namelist()
        assert not any('patients' in name for name in clinical.namelist())
    report = json.loads((output / 'manifest.json').read_text())
    assert 'clinical_R_EA' in report['included']
    assert any('interpretation' in name.lower() or 'integrated gradients' in name.lower()
               for name in report['missing'])
    assert output.with_suffix('.zip').exists()


def test_fraction_preparation_keeps_available_registered_contrasts_and_reports_local_gaps(tmp_path):
    from opera.run.prepare_fraction_results import prepare
    root = tmp_path / 'national'; root.mkdir()
    local = tmp_path / 'local/bundles/stage16'; local.mkdir(parents=True)
    (local / 'bundle.json').write_text(json.dumps({'runs': []}))
    direct = root / 'direct_fits'; direct.mkdir()
    (direct / 'bundle.json').write_text(json.dumps({'provenance': {'frozen_local_bundle': str(local)}}))
    adapted = root / 'adapted_fits'; adapted.mkdir()
    rows = []
    for percent in (10, 25, 50):
        config = adapted / f'F{percent}.yaml'
        config.write_text(yaml.safe_dump({'training': {'train_patient_fraction': percent / 100}}))
        rows.append({'name': f'seed42/EA/J_frac{percent}', 'origin': 'EA', 'phase': 'data_efficiency',
                     'seed': 42, 'config_path': config.name, 'run_directory': f'F{percent}',
                     'config_sha256': hashlib.sha256(config.read_bytes()).hexdigest()})
        predictions = root / f'predictions/F{percent}/held_out'; predictions.mkdir(parents=True)
        (predictions / 'evaluation_complete.json').write_text(json.dumps({'inputs': {
            'config_sha256': rows[-1]['config_sha256'], 'split': 'held_out'}}))
        (predictions / 'direct_cr_predictions.parquet').write_bytes(b'not loaded by preparer')
    (adapted / 'bundle.json').write_text(json.dumps({'runs': rows}))
    main = root / 'extended_test_comparison'; main.mkdir()
    original_manifest = Path('opera/configs/manifests/final_contrasts_20261014.yaml').resolve()
    base = {'contrast_manifest': str(original_manifest), 'estimand': 'cif_ipcw',
            'bootstrap_replicates': 1000, 'analysis_horizons_days': [30, 60, 90, 180, 365, 730],
            'artifacts': [{'model': m, 'split': 'held_out', 'seed': 42, 'data_contract': 'frozen'}
                          for m in ['J_L', 'J_EA', 'S_L', 'XP']]}
    (main / 'analysis.yaml').write_text(yaml.safe_dump(base))
    output = root / 'fraction_results'
    result = prepare(root, output)
    assert 'XP' not in {a['model'] for a in result['artifacts']}
    manifest = yaml.safe_load((output / 'focused_contrasts.yaml').read_text())
    comparisons = {k for v in manifest['essential_contrasts'].values() for k in v}
    assert {'FD10', 'FD25', 'FD50', 'LSE10', 'LSE25', 'LSE50'} <= comparisons
    assert 'N10' not in comparisons
    report = json.loads((output / 'preparation_report.json').read_text())
    assert report['missing_contrasts']['N10'] == ['J_L_F10']
    assert result['bootstrap_replicates'] == 1000
    assert result['analysis_horizons_days'] == [30, 60, 90, 180, 365, 730]
    assert result['strategy_candidates']  # Empty candidates break the existing strategy collector.
