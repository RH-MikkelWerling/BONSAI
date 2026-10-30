"""Fresh CPU tuning figure review using the author's 30 October revisions.

Reuses fixed predictions, encoder states and completed analyses. Does not
train, run the encoder, repeat source removals or open held-out data. All work
is in a new directory. Return only paper_review.zip, never this whole workspace.
"""
from __future__ import annotations

import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import yaml

from opera.run.paper_experiments import load_environment


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--analysis-root', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--mechanism-dir', type=Path)
    parser.add_argument('--whole-cohort-dir', type=Path)
    parser.add_argument('--restriction-dir', type=Path)
    parser.add_argument('--attribution-dir', type=Path)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--refresh-record-gaps', action='store_true',
                        help='Recompute horizon/family summaries from saved gap predictions on CPU')
    args = parser.parse_args()
    if args.threads < 1:
        parser.error('--threads must be positive')
    if args.refresh_record_gaps and args.restriction_dir:
        parser.error('Use either --restriction-dir or --refresh-record-gaps')
    root, output = args.analysis_root.resolve(), args.output_dir.resolve()
    if output == root or root in output.parents:
        raise ValueError('Use a fresh sibling directory outside the running analysis workspace')
    analysis = yaml.safe_load((root / 'analysis_inventory/analysis.yaml').read_text())
    if not analysis.get('artifacts') or any(a.get('split') != 'tuning' for a in analysis['artifacts']):
        raise ValueError('This refresh accepts the tuning-only analysis inventory')
    output.mkdir(parents=True, exist_ok=False)
    logs = output / 'logs'
    logs.mkdir()
    workspace = output / 'workspace'
    workspace.mkdir()
    for name in ('analysis_inventory', 'figure_inputs'):
        shutil.copytree(root / name, workspace / name)
    for name in ('completion.json', 'overnight.env'):
        shutil.copy2(root / name, workspace / name)
    # Existing analyses are read through links. No writer targets these paths.
    for name in ('source_removal', 'prediction_analysis', 'attributions', 'input_restrictions'):
        if (root / name).exists():
            (workspace / name).symlink_to(root / name, target_is_directory=True)
    env = {**os.environ, 'CUDA_VISIBLE_DEVICES': '', 'OPERA_ANALYSIS_WORKERS': str(args.threads),
           'OPENBLAS_NUM_THREADS': '1', 'OMP_NUM_THREADS': str(args.threads)}
    report = dict(started=datetime.now().astimezone().isoformat(), source=str(root),
                  split='tuning', steps={}, status='running',
                  note='New derived figures. Original predictions, models and analyses remain unchanged.')
    def save():
        (output / 'refresh_report.json').write_text(json.dumps(report, indent=2))
    def run(name, module, *arguments):
        command = [sys.executable, '-u', '-m', 'opera.run.' + module, *map(str, arguments)]
        print(f'{datetime.now().astimezone().isoformat()} {name}', flush=True)
        save()
        with (logs / (name + '.log')).open('w') as handle:
            result = subprocess.run(command, env=env, stdout=handle, stderr=subprocess.STDOUT)
        report['steps'][name] = dict(exit_code=result.returncode, command=command)
        save()
        return result.returncode == 0

    clinical = output / 'clinical_refresh'
    run('clinical', 'refresh_clinical_figures', '--workspace', root, '--output-dir', clinical,
        '--grouping-mode', 'cohort_size_matched')
    settings = load_environment(root / 'overnight.env')
    table_args = ['--analysis-root', root / 'prediction_analysis',
                  '--cell-metadata', analysis.get('cell_metadata') or settings['OPERA_PAPER_CELL_METADATA'],
                  '--output-dir', workspace / 'figure_inputs/plot_tables', '--split', 'tuning', '--seed', '42']
    if settings.get('OPERA_PAPER_ATLAS_METRICS'):
        table_args += ['--atlas-metrics', settings['OPERA_PAPER_ATLAS_METRICS']]
    run('plot_tables', 'build_paper_plot_tables', *table_args)

    embedding_path = clinical / 'configs/embedding_tables.tuning.yaml'
    if embedding_path.is_file():
        embedding = yaml.safe_load(embedding_path.read_text())
        vectors = output / 'map_vectors'
        success = run('map_vectors', 'refresh_routed_map_vectors', '--analysis-root', root,
            '--output-dir', vectors, '--threads', args.threads,
            '--outcomes', *embedding['calibration_maps']['outcomes'])
        if success:
            embedding['calibration_maps']['routed_vectors'] = json.loads((vectors / 'map_vectors.json').read_text())
            revised_embedding = output / 'embedding_tables.tuning.yaml'
            revised_embedding.write_text(yaml.safe_dump(embedding, sort_keys=False))
            run('embedding_tables', 'build_paper_embedding_tables', '--config', revised_embedding,
                '--patient-metadata', clinical / 'patient_metadata.parquet',
                '--output-dir', workspace / 'figure_inputs/plot_tables', '--rehearse-on-tuning')

    restrictions = args.restriction_dir
    if args.refresh_record_gaps:
        refreshed = output / 'record_gap_refresh'
        if run('record_gap_summaries', 'refresh_recent_record_gaps',
               '--workspace', root, '--output-dir', refreshed):
            restrictions = refreshed

    mechanism = None
    if args.mechanism_dir:
        mechanism = output / 'mechanism'
        mechanism.mkdir()
        for name in ('report.json', 'effect_summaries.csv', 'joint_path_summaries.csv'):
            source = args.mechanism_dir / name
            if source.is_file():
                shutil.copy2(source, mechanism / name)
        run('mechanism_figures', 'audit_frozen_router_mechanism', '--render-only', '--output-dir', mechanism)
    review_args = ['--analysis-root', workspace, '--output-dir', output / 'review',
                   '--split', 'tuning']
    clinical_dir = clinical / 'clinical_scores'
    if (clinical_dir / 'report_manifest.json').is_file():
        review_args += ['--clinical-dir', clinical_dir]
    for option, value in [('--restriction-dir', restrictions),
                          ('--attribution-dir', args.attribution_dir),
                          ('--whole-cohort-dir', args.whole_cohort_dir),
                          ('--mechanism-dir', mechanism)]:
        if value:
            review_args += [option, Path(value).resolve()]
    run('assemble', 'assemble_figure_review', *review_args)
    if (output / 'review/review_manifest.json').is_file():
        run('package', 'package_figure_review', '--review-dir', output / 'review',
            '--output', output / 'paper_review.zip', '--analysis-root', workspace)
    review_manifest = output / 'review/review_manifest.json'
    ready = (review_manifest.is_file() and
             json.loads(review_manifest.read_text()).get('status') == 'review_ready')
    report['status'] = ('review_ready' if ready and all(
        s['exit_code'] == 0 for s in report['steps'].values()) else 'finished_with_gaps')
    report['finished'] = datetime.now().astimezone().isoformat()
    save()
    print(f"{report['status']}: {output / 'paper_review.zip'}\n"
          'Transfer only the review ZIP. Workspace and INTERNAL arrays stay on the server.', flush=True)


if __name__ == '__main__':
    main()
