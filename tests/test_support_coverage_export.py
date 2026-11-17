import importlib.util
import json
from pathlib import Path
from zipfile import ZipFile

import pytest
import yaml


@pytest.mark.parametrize('same_baseline', [True, False])
def test_coverage_fallback_requires_matching_frozen_baselines_and_metadata(tmp_path, same_baseline):
    spec = importlib.util.spec_from_file_location('support', Path('scripts/package_support_coverage_inputs.py'))
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    root = tmp_path / 'data/paper_256_v3/analysis/local_test'
    (root / 'analysis_inventory').mkdir(parents=True)
    metadata = tmp_path / 'cell_metadata.csv'
    metadata.write_text('cohort,outcome,n_events_train\nDLBCL,OS,100\n')
    config = {'estimand': 'cif_ipcw', 'cell_metadata': str(metadata),
              'artifacts': [{'model': 'J_L', 'path': 'frozen-joint.parquet', 'split': 'held_out', 'seed': 42}]}
    (root / 'analysis_inventory/analysis.yaml').write_text(yaml.safe_dump(config))
    extended = tmp_path / 'data/paper_256_v3_national_20261111/extended_test_comparison'
    (extended / 'results').mkdir(parents=True)
    (extended / 'results/analysis_manifest.json').write_text('{}')
    if not same_baseline:
        config['artifacts'][0]['path'] = 'different-joint.parquet'
    (extended / 'analysis.yaml').write_text(yaml.safe_dump(config))
    for name in ('contrast_coverage.csv', 'model_coverage.csv'):
        (extended / 'results' / name).write_text('cohort,outcome,status\nDLBCL,OS,paired\n')
    output = tmp_path / 'support.zip'
    module.package(root, output)
    with ZipFile(output) as archive:
        manifest = json.loads(archive.read('manifest.json'))
        assert manifest['extended_coverage_fallback_validated'] is same_baseline
        assert ('aggregates/model_coverage.csv' in archive.namelist()) is same_baseline
        assert 'cell_metadata.csv' in archive.namelist()
        assert 'plot_tables/support_curve.csv' in manifest['missing']
