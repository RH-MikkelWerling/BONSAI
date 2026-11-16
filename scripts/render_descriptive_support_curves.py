"""Render the existing descriptive support smoother in the native pooling plate.

Uses downloaded aggregate cells only. Cell-bootstrap bands describe the smoother;
they are not posterior bands or shared-patient bootstrap trend intervals.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

import pandas as pd
import yaml


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--registered-dir', type=Path, default=Path('scratch/registered_figures_20261114'))
    parser.add_argument('--output-dir', type=Path, default=Path('scratch/descriptive_support_20261116'))
    parser.add_argument('--install-paper', action='store_true')
    args = parser.parse_args()
    source = args.registered_dir / 'plot_tables/paired_cells.csv'
    if args.output_dir.exists():
        raise FileExistsError('Use a fresh descriptive-curve output directory')
    from opera.run.build_paper_plot_tables import descriptive_support_curve
    cells = pd.read_csv(source)
    # Fixed existing recipe: no smoothing choices selected by inspecting results.
    selected = ('D-S', 'O-S', 'J-S')
    curve = descriptive_support_curve(cells, contrasts=selected, bandwidth=.35,
                                       n_boot=200, points=60, seed=2026)
    if set(curve.contrast) != set(selected):
        raise ValueError('Missing a pooling contrast; do not silently render a partial curve set')
    output = args.output_dir.resolve(); (output / 'plot_tables').mkdir(parents=True)
    shutil.copy2(source, output / 'plot_tables/paired_cells.csv')
    curve.to_csv(output / 'plot_tables/support_curve.csv', index=False)
    recipe = {'figures': [{'id': 'fig_pooling', 'function': 'figure_pooling',
                           'tables': {'cells': 'plot_tables/paired_cells.csv',
                                      'curve': 'plot_tables/support_curve.csv'}}]}
    path = output / 'figures.yaml'
    path.write_text(yaml.safe_dump(recipe, sort_keys=False))
    manifest = {'source_cells': str(source.resolve()),
                'source_cells_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
                'source_registered_manifest_sha256': hashlib.sha256(
                    (args.registered_dir / 'input_manifest.json').read_bytes()).hexdigest(),
                'analysis': 'Existing descriptive Gaussian-kernel smoother',
                'support': 'One-year training-event count; gains pooled over supported horizons',
                'bandwidth_log10_events': .35, 'draws': 200, 'grid_points': 60, 'seed': 2026,
                'uncertainty': 'Pointwise percentile band from resampling disease-outcome cells',
                'not_included': ['Shared-patient dependence between cells', 'Model refitting/selection uncertainty'],
                'primary_effect_estimates_changed': False, 'bayesian_fit_launched': False,
                'new_patient_inference_launched': False}
    (output / 'method_manifest.json').write_text(json.dumps(manifest, indent=2))
    (output / 'README.md').write_text('# Descriptive support curves\n\n'
        'The existing Gaussian smoother is applied to downloaded held-out aggregate '
        'cells. Shading is a pointwise cell-bootstrap band, not a posterior or '
        'paired-patient bootstrap trend interval. It does not account for shared '
        'patients/outcomes across cells. Main paired effect estimates and their '
        'intervals are unchanged. No GPU, new predictions or Bayesian fit.\n')
    subprocess.run([sys.executable, '-m', 'opera.run.render_paper_results', '--config', str(path),
                    '--output-dir', str(output / 'rendered'), '--execute'], check=True)
    if args.install_paper:
        destination = Path('paper/figures/results')
        for suffix in ('pdf', 'png'):
            shutil.copy2(output / f'rendered/fig_pooling.{suffix}', destination / f'fig_pooling.{suffix}')
        shutil.copy2(output / 'method_manifest.json', destination / 'descriptive_support_20261116.json')
    print(output / 'rendered/fig_pooling.pdf')


if __name__ == '__main__':
    main()
