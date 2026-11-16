"""Build and export aggregate plotting inputs from existing completed analyses.

No fitting, prediction, bootstrap analysis, or queue work. Private source views
stay on the server; only identifier-free plot tables and validated presentation
states enter the download archive.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import traceback
from zipfile import ZipFile, ZIP_DEFLATED

import yaml


def digest(path):
    value = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            value.update(chunk)
    return value.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--repo-root', type=Path, default=Path.cwd())
    args = parser.parse_args()
    repo = args.repo_root.resolve()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    from opera.run.build_paper_plot_tables import build
    from opera.run.prepare_offline_figure_bundle import copy_presentation
    from opera.run.package_figure_review import PATIENT_COLUMNS
    national = repo / 'data/paper_256_v3_national_20261111'
    projects = {
        'local': repo / 'data/paper_256_v3/analysis/local_test_20261109_four_allocations',
        'extended': national / 'extended_test_comparison',
        'fractions': national / 'fraction_results_20261116',
    }
    reports = {}
    selected = []
    for name, project in projects.items():
        report = reports[name] = {'project': str(project)}
        analysis = next((p for p in [project / 'prediction_analysis', project / 'results']
                         if (p / 'analysis_manifest.json').is_file()), None)
        configs = [project / 'analysis_inventory/analysis.yaml', project / 'analysis.yaml']
        config = next((p for p in configs if p.is_file()), None)
        if analysis is None or config is None:
            report.update(status='missing', reason='Completed analysis manifest or source config absent')
            continue
        spec = yaml.safe_load(config.read_text())
        metadata = Path(spec['cell_metadata'])
        if not metadata.is_absolute():
            metadata = repo / metadata
        view = output / '_SERVER_INTERNAL_views' / name
        view.mkdir(parents=True)
        (view / 'prediction_analysis').symlink_to(analysis, target_is_directory=True)
        atlas = None
        # Explicitly retain the manuscript's cohort-joint atlas identity.
        for artifact in spec.get('artifacts', []):
            if artifact.get('model') == 'J_L' and artifact.get('split') == 'held_out':
                candidate = Path(artifact['path']).parent / 'direct_cr_pooled_metrics.csv'
                if candidate.is_file():
                    atlas = candidate
                    break
        tables = output / name / 'plot_tables'
        try:
            manifest = build(view, metadata, tables, seed='42', atlas_metrics=atlas,
                             choices=repo / 'opera/configs/paper_figure_choices.yaml')
            report.update(status='built', tables=manifest['tables'], atlas_model='J_L' if atlas else None)
        except Exception as exc:
            report.update(status='builder_error', error=str(exc), traceback=traceback.format_exc())
        if not tables.exists():
            continue
        for path in sorted(tables.iterdir()):
            if path.suffix not in {'.csv', '.json'} or path.is_symlink():
                continue
            if path.suffix == '.csv':
                with path.open(newline='') as handle:
                    header = next(csv.reader(handle), [])
                if {c.strip().lower() for c in header} & (PATIENT_COLUMNS | {'patientid', 'subject_ids'}):
                    report.setdefault('excluded', []).append(path.name)
                    continue
            # These are outputs of the canonical aggregate builder, not arbitrary
            # JSON or patient coordinates discovered elsewhere in the workspace.
            selected.append(path)
        # Preserve paired effects for local iterations beyond the currently
        # registered converters (including the newer external-arm contrasts).
        import shutil
        patterns = ['contrasts.csv', 'contrasts_all_horizons.csv', 'panel_estimates.csv',
                    'effect_summaries*.csv', 'patient_subgroup_panels.csv',
                    'stratified*.csv', 'strategy_selection*.csv']
        for pattern in patterns:
            for path in sorted(analysis.glob(pattern)):
                with path.open(newline='') as handle:
                    header = next(csv.reader(handle), [])
                if {c.strip().lower() for c in header} & (PATIENT_COLUMNS | {'patientid', 'subject_ids'}):
                    report.setdefault('excluded', []).append(path.name)
                    continue
                target = output / name / 'aggregates' / path.name
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(path, target)
                selected.append(target)
    clinical = {}
    for model in ['R_EA', 'J_EA', 'R_AB_EA']:
        source = national / 'clinical_20261115' / model / 'clinical_results'
        states = list(source.rglob('*.plot.json')) if source.exists() else []
        clinical[model] = {'source': str(source), 'states': len(states)}
        for path in states:
            target = output / 'clinical_presentations' / model / path.relative_to(source)
            copy_presentation(path, target)
    for path in (output / 'clinical_presentations').rglob('*'):
        if path.is_file():
            selected.append(path)
    report = {'scope': 'Existing analyses only; no inference or model fitting',
              'projects': reports, 'clinical_presentations': clinical,
              'note': 'Missing upstream tables are explicit. This export does not create missing '
                      'seeds, component ablations, embeddings, IG or new paired contrasts.'}
    report['files'] = {str(p.relative_to(output)): digest(p)
                       for p in selected}
    receipt = output / 'export_report.json'
    receipt.write_text(json.dumps(report, indent=2))
    archive = output.with_suffix('.zip')
    with ZipFile(archive, 'x', compression=ZIP_DEFLATED, compresslevel=1) as z:
        for path in [*selected, receipt]:
            z.write(path, path.relative_to(output))
    print(json.dumps({'download': str(archive), 'projects': {
        k: {'status': v['status'], 'error': v.get('error')} for k, v in reports.items()},
        'clinical_presentations': clinical}, indent=2))


if __name__ == '__main__':
    main()
