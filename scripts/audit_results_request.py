"""Read-only inventory for the 15 November request; never loads predictions.

Run from the repository root. JSON goes to stdout; redirect it outside any
running analysis directory. Existence is deliberately not called verification.
"""
import argparse
import hashlib
import json
from pathlib import Path

import yaml


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return yaml.safe_load(path.read_text()) or {}


def inventory(root):
    def resolve(value):
        path = Path(value)
        return path if path.is_absolute() else root / path

    report = {'scope': 'Existence and small-config inventory; no prediction or checkpoint verification',
              'clinical': [], 'fits': [], 'aggregate_files': [], 'errors': []}
    refresh = root / 'data/clinical_registry_refresh'
    for config in sorted(refresh.glob('*/*/analysis.yaml')):
        try:
            prepared = config.parent
            analysis = read(config)
            manifest_path = prepared / 'clinical_results/report_manifest.json'
            manifest = read(manifest_path) if manifest_path.exists() else {}
            checks = {'analysis_config_sha': config,
                      'score_config_sha': prepared / 'clinical_scores.yaml',
                      'metadata_sha': resolve(analysis['patient_metadata'])}
            sources = read(prepared / 'refresh_report.json').get('sources', {}) if (prepared / 'refresh_report.json').exists() else {}
            report['clinical'].append({
                'prepared': str(prepared), 'render_manifest_exists': manifest_path.exists(),
                'render_hash_matches': {key: path.exists() and manifest.get(key) == digest(path)
                                        for key, path in checks.items()},
                'recorded_source_hash_matches_current_file': {
                    name: resolve(name).exists() and digest(resolve(name)) == sha
                    for name, sha in sources.items()},
                'threshold_artifact_count': len(analysis.get('clinical_threshold_artifacts', []) or []),
                'score_models': sorted(set(str(s.get('model')) for s in
                    read(prepared / 'clinical_scores.yaml').get('scores', []) if isinstance(s, dict))),
                'note': 'Source matches do not prove extract recency. Inspect score inventory for rendered models and endpoints.'})
        except Exception as error:
            report['errors'].append({'path': str(config), 'error': str(error)})

    local = root / 'data/paper_256_v3'
    national = root / 'data/paper_256_v3_national_20261111'
    bundles = list((local / 'bundles').glob('*/bundle.json'))
    bundles += [national / name / 'bundle.json' for name in ('direct_fits', 'adapted_fits')]
    for bundle_path in sorted(set(bundles)):
        if not bundle_path.exists():
            continue
        try:
            base = national if bundle_path.is_relative_to(national) else local
            for row in read(bundle_path).get('runs', []):
                phase = row.get('phase')
                if phase not in ('data_efficiency', 'pretraining_control', 'external', 'supervision'):
                    continue
                run = base / 'neural_runs' / row['run_directory']
                config_path = bundle_path.parent / row['config_path']
                config = read(config_path)
                training = config.get('training', {})
                report['fits'].append({
                    'bundle': str(bundle_path), 'name': row.get('name'), 'phase': phase,
                    'origin': row.get('origin'), 'seed': row.get('seed'),
                    'fraction': training.get('train_patient_fraction', 1),
                    'config_hash_matches': digest(config_path) == row.get('config_sha256'),
                    'run_directory': str(run), 'checkpoint_exists': (run / 'best.ckpt').exists(),
                    'not_estimable': (run / 'not_estimable.json').exists(),
                    'evaluation_marker_exists': {split: (base / 'predictions' / row['run_directory'] / split / 'evaluation_complete.json').exists()
                                                 for split in ('tuning', 'held_out')},
                    'note': 'Markers are not a fresh verification of selection or prediction hashes.'})
        except Exception as error:
            report['errors'].append({'path': str(bundle_path), 'error': str(error)})

    patterns = ['data/paper_256_v3_national_20261111/extended_*_comparison/results*/*',
                'data/paper_256_v3/analysis/local_test_20261109_four_allocations/prediction_analysis*/*',
                'data/clinical_registry_refresh/*/*/clinical_results/*',
                'data/diagnostics/*/*/*', 'data/diagnostics/*/*']
    names = {'effect_summaries.csv', 'panel_estimates.csv', 'analysis_manifest.json',
             'report_manifest.json', 'pipeline_complete.json', 'score_inventory.csv',
             'calibration_summary.csv', 'source_removal_sensitivity.csv',
             'preparation_report.json'}
    found = set()
    for pattern in patterns:
        for path in root.glob(pattern):
            if path.name in names and path.is_file():
                found.add(path)
    report['aggregate_files'] = [{'path': str(p), 'bytes': p.stat().st_size}
                                 for p in sorted(found)]
    report['limitations'] = ['Known standard locations only; custom output directories need separate inspection.',
                             'No new training, evaluation, queue claims, or analysis locks are performed.',
                             'Additional seeds and fractions require verified training selection before evaluation.']
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo-root', type=Path, default=Path.cwd())
    args = parser.parse_args()
    print(json.dumps(inventory(args.repo_root.resolve()), indent=2))
