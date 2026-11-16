"""Collect existing aggregate results and native artwork into a dated download.

Does not train, evaluate, render, read patient arrays or interrupt workers.
Missing result families remain explicit in the manifest and README.
"""
import argparse
import csv
import hashlib
import json
import importlib.util
from pathlib import Path
import shutil
from zipfile import ZIP_DEFLATED, ZipFile


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def aggregate_csv(path):
    with path.open(newline='') as handle:
        columns = next(csv.reader(handle), [])
    return not ({c.lower() for c in columns} & {'subject_id', 'subject_ids', 'patient_id',
                                               'patientid', 'person_id', 'cpr', 'pnr'})


def ship(repo, output):
    from opera.run.refresh_clinical_figures import package_clinical
    from opera.run.package_figure_review import REPORT_TABLES
    national = repo / 'data/paper_256_v3_national_20261111'
    output.mkdir(parents=True, exist_ok=False)
    report = {'scope': 'Existing aggregate results and artwork; no new calculations',
              'included': {}, 'missing': [], 'excluded': [], 'files': {}}
    def copy(path, relative):
        if path.is_symlink() or not path.is_file() or '_INTERNAL' in path.name:
            return False
        if path.suffix == '.csv' and not aggregate_csv(path):
            report['excluded'].append(str(path)); return False
        target = output / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        return True
    clinical = [('R_EA', national / 'clinical_20261115/R_EA/clinical_results'),
                ('J_EA', national / 'clinical_20261115/J_EA/clinical_results'),
                ('R_AB_EA', national / 'clinical_20261115/R_AB_EA/clinical_results'),
                ('J_OA_OWN_L', repo / 'data/clinical_registry_refresh/20261111/prepared_clinical/clinical_results')]
    for model, source in clinical:
        if not (source / 'report_manifest.json').exists():
            report['missing'].append(f'clinical {model}: no final render manifest at {source}'); continue
        target = output / 'clinical' / f'{model}.zip'; target.parent.mkdir(exist_ok=True)
        package_clinical(source, target)
        discrimination = source.parent / 'clinical_discrimination.csv'
        if discrimination.exists():
            if not aggregate_csv(discrimination):
                raise ValueError('Unexpected patient identifiers in clinical discrimination table')
            with ZipFile(target, 'a') as archive:
                archive.write(discrimination, 'clinical_discrimination.csv')
        copy(source.parent / 'launch_manifest.json', f'clinical/{model}_launch_manifest.json')
        report['included'][f'clinical_{model}'] = str(target.relative_to(output))
    table1_base = repo / 'data/clinical_registry_refresh/20261111'
    table1 = table1_base / 'table1_quality_refresh_20261116/table1_review.zip'
    if not table1.is_file():
        table1 = table1_base / 'table1_refresh_20261116/table1_review.zip'
    if copy(table1, 'table1/table1_review.zip'):
        report['included']['table1'] = 'table1/table1_review.zip'
    else:
        report['missing'].append('Table 1 refresh due 16 November: run scripts/refresh_table1_today.sh')
    report['missing'].append('Full Table 1 quality-registry refresh: score-row wrapper preserves existing multi-registry performance status')
    for family, source in [('embeddings', national / 'encoder_embedding_review_20261115/embedding_figures.zip')]:
        if copy(source, f'{family}/{source.name}'):
            report['included'][family] = f'{family}/{source.name}'
        else:
            report['missing'].append(f'{family}: expected export not yet present at {source}')
    # Construct this lightweight export directly; no separate packaging command is needed.
    support_root = repo / 'data/paper_256_v3/analysis/local_test_20261109_four_allocations'
    if (support_root / 'analysis_inventory/analysis.yaml').is_file():
        spec = importlib.util.spec_from_file_location('support_export', repo / 'scripts/package_support_coverage_inputs.py')
        module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
        destination = output / 'support_coverage/support_coverage_inputs.zip'
        module.package(support_root, destination)
        report['included']['support_coverage'] = str(destination.relative_to(output))
    else:
        report['missing'].append('Support/coverage export: original analysis inventory absent')
    fractions = national / 'fraction_results_20261116'
    if (fractions / 'results/analysis_manifest.json').exists():
        for name in ('effect_summaries.csv', 'effect_summaries_disease_balanced.csv', 'panel_estimates.csv',
                     'estimates.csv', 'contrast_coverage.csv', 'model_coverage.csv', 'analysis_manifest.json'):
            copy(fractions / 'results' / name, 'label_efficiency/' + name)
        copy(fractions / 'preparation_report.json', 'label_efficiency/preparation_report.json')
        report['included']['label_efficiency'] = 'label_efficiency'
    else:
        report['missing'].append('Focused fraction/control aggregate analysis not yet complete')
    for split in ('tuning', 'held_out'):
        name = f'fraction_evaluation_{split}_20261116.json'
        copy(national / name, 'label_efficiency/' + name)
    ig = national / 'final_interpretation_20261116'
    if (ig / 'interpretation_manifest.json').exists():
        copy(ig / 'interpretation_manifest.json', 'interpretation/interpretation_manifest.json')
        for folder in ('attributions/summary_365d', 'input_restrictions'):
            for name in REPORT_TABLES:
                copy(ig / folder / name, 'interpretation/' + folder + '/' + name)
        for path in (ig / 'attributions/summary_365d/figures').glob('*'):
            if path.suffix in {'.pdf', '.png'} or path.name == 'figure_manifest.json':
                copy(path, 'interpretation/figures/' + path.name)
        state = json.loads((ig / 'interpretation_manifest.json').read_text()).get('status')
        report['included']['interpretation'] = {'directory': 'interpretation', 'status': state}
        if state != 'complete':
            report['missing'].append(f'Final-model interpretation is partial: {state}')
    else:
        report['missing'].append('Final R_EA integrated gradients and restrictions not yet packaged')
    diagnostic = repo / 'data/diagnostics/representation_absorption_20261112/tuning_core'
    diagnostic_names = {'closure_summaries.csv', 'absorption_correlations.csv', 'risk_agreement_summary.csv',
                        'scarcity_summaries.csv', 'scarcity_missing_arms.csv', 'seed_results.csv', 'seed_spread.csv',
                        'manifest.json', 'report.md', 'pipeline_complete.json'}
    if (diagnostic / 'pipeline_complete.json').exists():
        for name in diagnostic_names:
            copy(diagnostic / name, 'exploratory_representation/' + name)
        for path in diagnostic.glob('*.pdf'):
            copy(path, 'exploratory_representation/' + path.name)
        for name in ('inference_ablation_summaries.csv', 'correction_norm_overall.csv',
                     'correction_norm_intervals.csv', 'blocked.csv', 'manifest.json'):
            copy(diagnostic / 'correction_reliance' / name, 'exploratory_representation/correction_reliance/' + name)
        report['included']['representation_diagnostics'] = 'exploratory_representation'
    else:
        report['missing'].append('Exploratory representation core completion marker absent')
    report['missing'] += ['XP clinical training/full-population export still needs its separate recovery implementation',
                          'This exporter does not evaluate or aggregate missing fraction arms',
                          'Direct R_EA versus S_L paired interval is not fabricated from other intervals',
                          'Vocabulary-controlled pretraining remains unlaunched pending explicit approval']
    for path in output.rglob('*'):
        if path.is_file():
            report['files'][str(path.relative_to(output))] = {'bytes': path.stat().st_size, 'sha256': digest(path)}
    (output / 'manifest.json').write_text(json.dumps(report, indent=2))
    (output / 'README.md').write_text('# Paper results handoff\n\n'
        'Existing outputs only. Clinical ZIPs are rebuilt to include the observed-risk matching tables. '
        'No patient prediction tables, embeddings, raw IG or environment files are exported.\n\n'
        '## Included\n\n' + '\n'.join(f'- {name}: {value}' for name, value in report['included'].items())
        + '\n\n## Missing or incomplete\n\n' + '\n'.join('- ' + x for x in report['missing']) + '\n')
    archive = output.with_suffix('.zip')
    if archive.exists():
        raise FileExistsError(archive)
    with ZipFile(archive, 'x', compression=ZIP_DEFLATED, compresslevel=1) as target:
        for path in output.rglob('*'):
            if path.is_file():
                target.write(path, path.relative_to(output))
    print(json.dumps({'download': str(archive), 'included': report['included'], 'missing': report['missing']}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo-root', type=Path, default=Path.cwd())
    parser.add_argument('--output-dir', type=Path, default=Path('data/paper_results_export_20261116'))
    args = parser.parse_args()
    ship(args.repo_root.resolve(), args.output_dir.resolve())
