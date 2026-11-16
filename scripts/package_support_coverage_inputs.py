"""Export saved support/coverage plotting inputs; never fits a curve or reads patients."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import yaml


def package(root, output):
    config = root / 'analysis_inventory/analysis.yaml'
    if not config.exists():
        config = root / 'analysis.yaml'
    analysis = yaml.load(config.read_text(), Loader=getattr(yaml, 'CSafeLoader', yaml.SafeLoader))
    metadata = Path(analysis['cell_metadata'])
    # The original coordinator may have only progress files while the separately
    # completed extended analysis has already published these same baselines.
    data_root = next((p for p in root.parents if p.name == 'data'), None)
    fallback = (data_root / 'paper_256_v3_national_20261111/extended_test_comparison'
                if data_root is not None else None)
    fallback_valid = False
    if fallback is not None and (fallback / 'results/analysis_manifest.json').is_file():
        extended = yaml.load((fallback / 'analysis.yaml').read_text(),
                             Loader=getattr(yaml, 'CSafeLoader', yaml.SafeLoader))
        def baselines(config):
            return {(a['model'], a['path']) for a in config.get('artifacts', [])
                    if a.get('model') in {'J_L', 'S_L'} and a.get('split') == 'held_out'
                    and int(a.get('seed', 42)) == 42}
        fallback_valid = (extended.get('estimand') == analysis.get('estimand')
                          and Path(extended['cell_metadata']).resolve() == metadata.resolve()
                          and bool(baselines(analysis)) and baselines(extended) == baselines(analysis))
    required = {'cell_metadata.csv': metadata}
    for name in ('coverage.csv', 'training_support.csv', 'paired_cells.csv', 'support_curve.csv'):
        required['plot_tables/' + name] = root / 'figure_inputs/plot_tables' / name
    optional = {}
    for family in ('pooling', 'tabular'):
        optional[f'bayesian/{family}/auc/support_spline/posterior_curve.csv'] = (
            root / 'bayesian' / family / 'auc/support_spline/posterior_curve.csv')
    for name in ('contrast_coverage.csv', 'model_coverage.csv'):
        path = root / 'prediction_analysis' / name
        if not path.is_file() and fallback_valid:
            path = fallback / 'results' / name
        required['aggregates/' + name] = path
    files = {name: path for name, path in required.items() if path.is_file()}
    files.update({name: path for name, path in optional.items() if path.is_file()})
    for path in (root / 'bayesian_recipes').glob('**/*.yaml'):
        files['bayesian_recipes/' + str(path.relative_to(root / 'bayesian_recipes'))] = path
    report = {'scope': 'Existing saved outputs only; no new support trend or inference fitted',
              'source_analysis_root': str(root),
              'extended_coverage_fallback_validated': fallback_valid,
              'current_curve_plan': 'Existing descriptive smoother; posterior files are optional historical outputs',
              'optional_unavailable': [name for name, path in optional.items() if not path.is_file()],
              'missing': [name for name, path in required.items() if not path.is_file()], 'files': {}}
    for name, path in files.items():
        if path.suffix == '.csv':
            with path.open(newline='') as handle:
                columns = next(csv.reader(handle), [])
            if {c.lower() for c in columns} & {'subject_id', 'patient_id', 'patientid', 'cpr', 'pnr'}:
                raise ValueError(f'Patient identifier in supposed aggregate: {path}')
        h = hashlib.sha256()
        with path.open('rb') as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b''):
                h.update(chunk)
        report['files'][name] = {'bytes': path.stat().st_size, 'sha256': h.hexdigest(), 'source': str(path)}
    output.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(output, 'x', compression=ZIP_DEFLATED, compresslevel=1) as archive:
        for name, path in files.items():
            archive.write(path, name)
        archive.writestr('manifest.json', json.dumps(report, indent=2))
    print(json.dumps({'download': str(output), 'included': list(files), 'missing': report['missing']}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--analysis-root', type=Path, default=Path(
        'data/paper_256_v3/analysis/local_test_20261109_four_allocations'))
    parser.add_argument('--output', type=Path, default=Path('support_coverage_inputs_20261115.zip'))
    args = parser.parse_args()
    package(args.analysis_root.resolve(), args.output.resolve())
