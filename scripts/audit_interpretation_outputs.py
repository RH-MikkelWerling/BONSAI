"""Inventory existing IG, embedding and clinical plotting inputs without inference.

Only paths, sizes, recipe references and selected provenance fields are reported.
Never reads patient arrays, prediction tables or environment credentials.
"""
import argparse
import csv
import json
import os
from collections import deque
from pathlib import Path

import yaml


PRUNE = {'neural_runs', 'bundles', 'checkpoints', '_cpu_checkpoints_INTERNAL',
         'external_dapt', 'external_direct', 'sequences', '.git', '__pycache__'}
MARKERS = {'attribution_run.json', 'summary_manifest.json', 'refresh_manifest.json',
           'review_manifest.json', 'report_manifest.json', 'completion.json',
           'pipeline_complete.json', 'figure_manifest.json', 'analysis_provenance.json',
           'analysis_metadata.json'}
PROVENANCE = {'model', 'model_alias', 'opera_model', 'checkpoint', 'checkpoint_sha256',
              'config', 'config_sha256', 'config_name', 'split', 'status', 'horizon_days',
              'attribution_run_sha256', 'source', 'method', 'embedding_dim', 'n_subjects'}
ENV_KEYS = {'OPERA_PAPER_OPERA_MODEL', 'OPERA_PAPER_ATTRIBUTION_CHECKPOINT',
            'OPERA_PAPER_ATTRIBUTION_CONFIG', 'OPERA_PAPER_ATTRIBUTION_CONFIG_NAME',
            'OPERA_PAPER_ATTRIBUTION_CONFIG_DIR', 'OPERA_PAPER_EMBEDDING_CONFIG',
            'OPERA_PAPER_ATTRIBUTION_SUMMARY_DIR', 'OPERA_PAPER_CLINICAL_SCORE_CONFIG',
            'OPERA_PAPER_PLOT_SPLIT'}


def identity(value):
    """Select provenance recursively, without copying patient or tensor payloads."""
    if not isinstance(value, dict):
        return {}
    result = {}
    for key, item in value.items():
        if key in PROVENANCE and isinstance(item, (str, int, float, bool, type(None))):
            result[key] = item
        elif isinstance(item, dict):
            nested = identity(item)
            if nested:
                result[key] = nested
    return result


def audit(repo, extra_roots=(), max_directories=40000):
    recipes = [repo / 'opera/configs' / name for name in
               ('paper_results_figures.full.example.yaml', 'attribution_figures.example.yaml')]
    required_names = set()
    plates = []
    for path in recipes:
        for plate in yaml.safe_load(path.read_text())['figures']:
            if any(term in plate['function'] for term in
                   ('attribution', 'embedding', 'patient_map', 'patient_disease',
                    'calibration_map', 'risk_group', 'clinical_index', 'recent_record', 'correction_arrow')):
                references = {**plate.get('tables', {}), **plate.get('json_inputs', {})}
                required_names.update(Path(v).name for v in references.values())
                plates.append({'id': plate['id'], 'function': plate['function'],
                               'recipe': str(path), 'inputs': references})
    roots = [repo / 'data' / name for name in
             ('paper_256_v3', 'paper_256_v3_national_20261111',
              'clinical_registry_refresh', 'diagnostics')]
    roots += [Path(p).resolve() for p in extra_roots]
    queue = deque((p, 0) for p in roots if p.is_dir())
    report = {'scope': 'Read-only plotting and interpretation inventory; existence is not validation',
              'roots': [str(p) for p in roots], 'registered_figures': plates,
              'files': [], 'environments': [], 'errors': [], 'scan_truncated': False}
    visited = set()
    while queue:
        directory, depth = queue.popleft()
        if directory in visited:
            continue
        if len(visited) >= max_directories:
            report['scan_truncated'] = True
            break
        visited.add(directory)
        try:
            with os.scandir(directory) as entries:
                for entry in entries:
                    path = Path(entry.path)
                    if entry.is_symlink():
                        continue
                    if entry.is_dir(follow_symlinks=False):
                        if depth < 8 and entry.name not in PRUNE and not entry.name.startswith('.'):
                            queue.append((path, depth + 1))
                        continue
                    name = entry.name
                    if name == 'overnight.env':
                        values = {}
                        for line in path.read_text().splitlines():
                            key, sep, value = line.partition('=')
                            if sep and key in ENV_KEYS:
                                try:
                                    value = json.loads(value)
                                except ValueError:
                                    value = value.strip("\"'")
                                values[key] = value
                        report['environments'].append({'path': str(path), 'selected_values': values})
                        continue
                    interesting = (name in required_names or name in MARKERS or
                        name.startswith('attribution_') and name.endswith(('.csv', '.json')) or
                        name.endswith('.metadata.json') or
                        'embedding' in name and name.endswith(('.npz', '.yaml', '.json')) or
                        name in {'clinical_scores.yaml', 'attribution.yaml', 'input_restriction_summary.csv'} or
                        name.endswith(('.pdf', '.png', '.plot.json')) and
                        (any(term in name for term in
                            ('embed', 'patient_map', 'patient_umap', 'patient_pca', 'patient_tsne',
                             'attribution', 'risk_group', 'clinical', 'calibration_map')) or
                         any(term in str(path.parent).lower() for term in ('embedding', 'atlas'))))
                    if not interesting:
                        continue
                    row = {'path': str(path), 'bytes': entry.stat().st_size}
                    if name in MARKERS or name.endswith('.metadata.json'):
                        if row['bytes'] <= 2 * 1024 * 1024:
                            row['provenance'] = identity(json.loads(path.read_text()))
                        else:
                            row['note'] = 'Large manifest not opened'
                    if name.endswith('.csv'):
                        with path.open(newline='') as handle:
                            row['columns'] = next(csv.reader(handle), [])
                    if name.endswith('.yaml') and row['bytes'] < 1024 * 1024:
                        config = yaml.safe_load(path.read_text()) or {}
                        row['provenance'] = identity(config)
                        if isinstance(config.get('scores'), list):
                            row['clinical_models'] = sorted({str(s.get('model')) for s in config['scores']})
                        if isinstance(config.get('stages'), dict):
                            row['stages'] = config['stages']
                    report['files'].append(row)
        except (OSError, ValueError, yaml.YAMLError) as error:
            report['errors'].append({'path': str(directory), 'error': str(error)})
    report['directories_scanned'] = len(visited)
    report['required_filename_matches'] = {name: [r['path'] for r in report['files']
                                                   if Path(r['path']).name == name]
                                           for name in sorted(required_names)}
    report['notes'] = ['No training, IG, inference, table loading or figure rendering was launched.',
        'Model identity must be confirmed from checkpoint/config provenance; old OPERA aliases may mean cohort OPERA.',
        'Patient arrays and coordinates stay on the server. Use prepare_offline_figure_bundle for portable figure states.',
        'Scan depth is eight; pruned or custom workspaces need --extra-root. A missing match is not proof that a run never existed.']
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo-root', type=Path, default=Path.cwd())
    parser.add_argument('--extra-root', type=Path, action='append', default=[])
    args = parser.parse_args()
    print(json.dumps(audit(args.repo_root.resolve(), args.extra_root), indent=2))
