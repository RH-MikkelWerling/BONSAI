"""Rebuild clinical reports on CPU from an existing workspace's predictions.

Use complete_local_figures --clinical-only first if training predictions have
not been exported. This command never trains or runs model inference.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import subprocess
import sys
import zipfile

import yaml

from opera.run.prepare_available_plot_configs import build
from opera.run.prepare_clinical_figure_metadata import prepare
from opera.run.clinical_figure_provenance import training_artifact_with_contract


def package_clinical(directory, destination):
    """Only figures and named aggregate reports; never the workspace metadata."""
    from opera.run.package_figure_review import PATIENT_COLUMNS
    allowed = {'score_inventory.csv', 'headline_scores.csv', 'classification_at_thresholds.csv',
        'classification_at_anchors.csv', 'cutoff_diagnostics.csv', 'score_population_flow.csv',
        'stratification_diagnostics.csv', 'risk_group_point_estimates.csv', 'score_availability.csv',
        'size_matched_survival_comparisons.csv', 'km_matched_low_groups.csv',
    'clinical_operating_point_counts.csv',
        'report_manifest.json', 'layout_report.json', 'supplement_clinical_scores.tex'}
    selected = []
    for path in sorted(directory.rglob('*')):
        if not path.is_file() or path.is_symlink() or 'INTERNAL' in path.name.upper():
            continue
        if path.name not in allowed and path.suffix.lower() not in {'.pdf', '.png'}:
            continue
        if path.suffix == '.csv':
            with path.open(newline='') as handle:
                header = next(csv.reader(handle), [])
            if PATIENT_COLUMNS.intersection(name.strip().lower() for name in header):
                raise ValueError(f'Patient-level columns in an aggregate clinical report: {path.name}')
        selected.append(path)
    with zipfile.ZipFile(destination, 'x', zipfile.ZIP_DEFLATED) as archive:
        for path in selected:
            archive.write(path, path.relative_to(directory))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace', type=Path, required=True,
                        help='Existing completed clinical workspace, not its clinical_scores subdirectory')
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--grouping-mode', choices=['current', 'cohort_size_matched'], default='current',
                        help='Explicit new benchmark mode; current preserves the existing recipe')
    args = parser.parse_args()
    workspace = args.workspace.resolve()
    analysis = yaml.safe_load((workspace / 'analysis_inventory/analysis.yaml').read_text())
    model = analysis.get('opera_model', 'R_L')
    training = [a for a in [*analysis['artifacts'], *analysis.get('clinical_threshold_artifacts', [])]
                if a.get('model') == model and a.get('split') == 'train']
    if not training or any(not Path(a['path']).is_file() for a in training):
        raise ValueError('No existing primary-model training predictions. Run '
                         'opera.run.complete_local_figures --clinical-only in a GPU allocation '
                         'with --analysis-root pointing to the completed analysis and a fresh --output-dir.')
    for key in ('artifacts', 'clinical_threshold_artifacts'):
        if key in analysis:
            analysis[key] = [training_artifact_with_contract(a, analysis)
                             if a.get('split') == 'train' else a for a in analysis[key]]
    output = args.output_dir.resolve()
    if output == workspace or workspace in output.parents:
        raise ValueError('Use a fresh sibling output directory')
    output.mkdir(parents=True, exist_ok=False)
    prepare(analysis['patient_metadata'], [], output / 'patient_metadata.parquet')
    analysis['patient_metadata'] = str(output / 'patient_metadata.parquet')
    (output / 'analysis_inventory').mkdir()
    config_path = output / 'analysis_inventory/analysis.yaml'
    config_path.write_text(yaml.safe_dump(analysis, sort_keys=False))
    build(analysis, output / 'configs')
    if args.grouping_mode == 'cohort_size_matched':
        score_path = output / 'configs/clinical_score_panels.tuning.yaml'
        score_config = yaml.safe_load(score_path.read_text())
        new_settings = dict(cutoff_rule='cohort_size_matched', comparison_bootstrap=1000,
            lock_provenance='Rehearsal candidate: baseline-only cohort ranks for groups; separate development KM and specificity anchors')
        for spec in score_config.get('scores', []):
            spec.update(new_settings)
        if score_config.get('all_indices'):
            score_config['all_indices']['template'].update(new_settings)
        score_path.write_text(yaml.safe_dump(score_config, sort_keys=False))
    command = [sys.executable, '-m', 'opera.run.render_clinical_score_panels',
               '--analysis-config', str(config_path),
               '--score-config', str(output / 'configs/clinical_score_panels.tuning.yaml'),
               '--output-dir', str(output / 'clinical_scores'), '--split', 'tuning', '--execute']
    env = {**os.environ, 'OPERA_PAPER_FIGURE_MARK': 'TUNING SET — not test results'}
    subprocess.run(command, env=env, check=True)
    clinical_dir = output / 'clinical_scores'
    review_zip = output / 'clinical_scores_review.zip'
    if clinical_dir.is_dir():
        package_clinical(clinical_dir, review_zip)
    print(json.dumps(dict(workspace=str(workspace), clinical_dir=str(output / 'clinical_scores'),
                          review_zip=str(review_zip) if review_zip.is_file() else None,
                          training_predictions_reused=[a['path'] for a in training],
                          grouping_mode=args.grouping_mode,
                          note='CPU report refresh only; no inference or training'), indent=2))


if __name__ == '__main__':
    main()
