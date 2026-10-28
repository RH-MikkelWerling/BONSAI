"""Render current aggregate inputs and collect figure families into one review folder.

CPU only. Missing analyses are reported, never synthesized or silently borrowed
from another workspace. Patient-level tables and INTERNAL files are not copied.
"""
from __future__ import annotations
import copy
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from datetime import datetime
from zoneinfo import ZoneInfo
import yaml
from opera.run.paper_experiments import load_environment


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--analysis-root', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--restriction-dir', type=Path, help='Explicit completed input_restrictions directory')
    parser.add_argument('--clinical-dir', type=Path, help='Explicit refreshed clinical_scores directory')
    parser.add_argument('--attribution-dir', type=Path, help='Explicit completed IG directory with summary_365d')
    args = parser.parse_args()
    root = args.analysis_root.resolve()
    output = (args.output_dir or root / ('figure_review_' + datetime.now(
        ZoneInfo('Europe/Copenhagen')).strftime('%Y%m%d_%H%M%S'))).resolve()
    output.mkdir(parents=True, exist_ok=False)
    env = {**os.environ, **load_environment(root / 'overnight.env')}
    env.update(OPERA_PAPER_ANALYSIS_ROOT=str(root), OPERA_PAPER_PLOT_SPLIT='tuning',
        OPERA_PAPER_PLOT_TABLE_ROOT=str(root / 'figure_inputs/plot_tables'),
        OPERA_PAPER_ATTRIBUTION_SUMMARY_DIR=str(root / 'attributions/summary_365d'),
        OPERA_PAPER_INPUT_RESTRICTION_DIR=str(root / 'input_restrictions'))
    attribution = args.attribution_dir.resolve() if args.attribution_dir else root / 'attributions'
    env['OPERA_PAPER_ATTRIBUTION_SUMMARY_DIR'] = str(attribution / 'summary_365d')
    restrictions = args.restriction_dir.resolve() if args.restriction_dir else root / 'input_restrictions'
    env['OPERA_PAPER_INPUT_RESTRICTION_DIR'] = str(restrictions)
    repo = Path(__file__).resolve().parents[2]
    report = dict(analysis_root=str(root), attribution_dir=str(attribution), steps={}, copied={}, missing=[],
        notes=['Rendered from this workspace only. Existing clinical plots retain their source provenance.',
               'Attribution summaries may be reused analyses; inspect their provenance before publication.',
               'No inference, training, or integrated-gradient computation was launched.'])
    def save():
        (output / 'review_manifest.json').write_text(json.dumps(report, indent=2))
    def render(name, recipe):
        recipe_path = output / (name + '.yaml')
        recipe_path.write_text(yaml.safe_dump(recipe, sort_keys=False))
        # Expand variables explicitly; relative template assets are resolved below.
        text = recipe_path.read_text()
        for key, value in env.items():
            text = text.replace('${' + key + '}', value)
        recipe_path.write_text(text)
        command = [sys.executable, '-m', 'opera.run.render_paper_results', '--config', str(recipe_path),
                   '--output-dir', str(output / name), '--execute', '--available-only']
        with (output / (name + '.log')).open('w') as log:
            result = subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT)
        report['steps'][name] = dict(exit_code=result.returncode)
        save()
    recipe = yaml.safe_load((repo / 'opera/configs/paper_results_figures.full.example.yaml').read_text())
    for plate in recipe['figures']:
        plate.setdefault('options', {})['mark'] = 'TUNING SET — not test results'
        for section in ('tables','json_inputs'):
            for key, source in plate.get(section, {}).items():
                if not source.startswith('${OPERA_PAPER_ANALYSIS_ROOT}/'):
                    plate[section][key] = str(root / 'figure_inputs/plot_tables' / Path(source).name)
    if args.clinical_dir:
        for plate in recipe['figures']:
            for key, source in plate.get('tables', {}).items():
                if '/clinical_scores/' in source:
                    plate['tables'][key] = str(args.clinical_dir.resolve() / source.split('/clinical_scores/',1)[1])
    render('main', recipe)
    zero = copy.deepcopy(recipe)
    zero['required_figures'] = []
    zero['figures'] = [p for p in zero['figures'] if p['id'] in {'fig_borrowing_map','supp_borrowing_detail'}]
    for plate in zero['figures']:
        plate['id'] += '_zero'
        plate['options']['mark'] = 'ZERO-ABLATION SENSITIVITY — tuning set'
        for key, source in plate['tables'].items():
            plate['tables'][key] = source.replace('/source_removal/${OPERA_PAPER_PLOT_SPLIT}/', '/source_removal/${OPERA_PAPER_PLOT_SPLIT}/zero_ablation/')
    render('borrowing_zero', zero)
    recipe = yaml.safe_load((repo / 'opera/configs/attribution_figures.example.yaml').read_text())
    for plate in recipe['figures']:
        plate.setdefault('options', {})['mark'] = 'TUNING SET — not test results'
        for key, source in plate.get('tables', {}).items():
            if not source.startswith('${'):
                plate['tables'][key] = str((repo / 'opera/configs' / source).resolve())
    render('attributions', recipe)
    render('recent_records', dict(figures=[dict(id='supp_recent_record_gaps',
        function='figure_recent_record_gaps', tables=dict(restrictions=str(restrictions / 'input_restriction_summary.csv')),
        options=dict(mark='TUNING SET — not test results'))]))
    clinical = args.clinical_dir.resolve() if args.clinical_dir else root / 'clinical_scores'
    report['clinical_dir'] = str(clinical)
    for suffix in ('*.pdf','*.png'):
        for source in sorted(clinical.rglob(suffix)):
            target = output / 'clinical_scores' / source.relative_to(clinical)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source,target)
            report['copied'][str(target.relative_to(output))] = dict(source=str(source), sha256=sha(source))
    for relative in ('completion.json','figure_results_audit.json','clinical_scores/score_inventory.csv',
                     'clinical_scores/headline_scores.csv','clinical_scores/classification_at_thresholds.csv',
                     'clinical_scores/report_manifest.json','configs/config_audit.json',
                     'source_removal/tuning/analysis_provenance.json',
                     'source_removal/tuning/borrowing_map.csv',
                     'source_removal/tuning/borrowing_levels.csv',
                     'input_restrictions/input_restriction_summary.csv'):
        source = (clinical / relative.split('/',1)[1] if relative.startswith('clinical_scores/') else
                  restrictions / relative.split('/',1)[1] if relative.startswith('input_restrictions/') else root / relative)
        if source.is_file():
            target = output / 'reports' / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source,target)
    for name, relative in dict(borrowing='source_removal/tuning/borrowing_map.csv',
        integrated_gradients='attributions/summary_365d/attribution_variance_sources.csv',
        recent_record_removal='input_restrictions/input_restriction_summary.csv',
        clinical_scores='clinical_scores/score_inventory.csv',
        classification_metrics='clinical_scores/classification_at_thresholds.csv').items():
        source = (attribution / 'summary_365d/attribution_variance_sources.csv' if name == 'integrated_gradients'
                  else clinical / relative.split('/',1)[1] if relative.startswith('clinical_scores/')
                  else restrictions / relative.split('/',1)[1] if relative.startswith('input_restrictions/') else root / relative)
        if not source.is_file():
            report['missing'].append(dict(family=name, input=relative))
    report['figure_manifests'] = {}
    for path in output.rglob('figure_manifest.json'):
        report['figure_manifests'][str(path.relative_to(output))] = json.loads(path.read_text())
    report['status'] = 'review_ready' if (report['figure_manifests'] and
        all(m.get('complete', False) for m in report['figure_manifests'].values())) and not report['missing'] and all(
        s['exit_code'] == 0 for s in report['steps'].values()) else 'partial_review'
    save()
    print(json.dumps(dict(status=report['status'], output_dir=str(output), missing=report['missing']),indent=2))


if __name__ == '__main__':
    main()
