"""Rebuild clinical reports on CPU from an existing workspace's predictions.

Use complete_local_figures --clinical-only first if training predictions have
not been exported. This command never trains or runs model inference.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

import yaml

from opera.run.prepare_available_plot_configs import build
from opera.run.prepare_clinical_figure_metadata import prepare


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace', type=Path, required=True,
                        help='Existing completed clinical workspace, not its clinical_scores subdirectory')
    parser.add_argument('--output-dir', type=Path, required=True)
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
    command = [sys.executable, '-m', 'opera.run.render_clinical_score_panels',
               '--analysis-config', str(config_path),
               '--score-config', str(output / 'configs/clinical_score_panels.tuning.yaml'),
               '--output-dir', str(output / 'clinical_scores'), '--split', 'tuning', '--execute']
    env = {**os.environ, 'OPERA_PAPER_FIGURE_MARK': 'TUNING SET — not test results'}
    subprocess.run(command, env=env, check=True)
    print(json.dumps(dict(workspace=str(workspace), clinical_dir=str(output / 'clinical_scores'),
                          training_predictions_reused=[a['path'] for a in training],
                          note='CPU report refresh only; no inference or training'), indent=2))


if __name__ == '__main__':
    main()
