"""Extend completed recent-record exports across horizons, without GPU inference."""
from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import subprocess
import sys
import zipfile

import yaml

from opera.run.compare_input_restrictions import read_predictions, compare, write_outputs, figure_recipe
from opera.visualization.paper_taxonomy import outcome_endpoint_groups


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace', type=Path, required=True, help='Existing completion workspace containing input_restrictions/')
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--horizons', nargs='+', type=int, default=[30, 60, 90, 180, 365, 730])
    parser.add_argument('--draws', type=int, default=1000)
    parser.add_argument('--seed', type=int, default=2026)
    args = parser.parse_args()
    workspace, output = args.workspace.resolve(), args.output_dir.resolve()
    if output == workspace or workspace in output.parents or output.exists():
        raise ValueError('Use a fresh sibling output directory')
    fit = json.loads((workspace / 'completion.json').read_text())['inputs']['fit']
    full = read_predictions(fit['predictions'])
    if not full.split.eq('tuning').all():
        raise ValueError('This rehearsal refresh reads tuning predictions only')
    paths = {f'Last {d} days hidden': workspace / f'input_restrictions/gap_{d}d' for d in (7, 14, 21, 28)}
    restricted = {name: read_predictions(path) for name, path in paths.items()}
    print('Comparing saved predictions only; no encoder inference or model fitting.', flush=True)
    table, summary = compare(full, restricted, horizons=args.horizons, draws=args.draws, seed=args.seed,
        domains=outcome_endpoint_groups(), progress=lambda message: print(f'{datetime.now().isoformat(timespec="seconds")} {message}', flush=True))
    write_outputs(table, summary, output)
    recipe = output / 'recent_record_gaps.yaml'
    recipe.write_text(yaml.safe_dump(figure_recipe(output / 'input_restriction_summary_by_horizon.csv',
        args.horizons, 'TUNING SET — not test results')))
    subprocess.run([sys.executable, '-m', 'opera.run.render_paper_results', '--config', str(recipe),
                    '--output-dir', str(output / 'figures'), '--execute'], check=True)
    (output / 'refresh.json').write_text(json.dumps(dict(workspace=str(workspace),
        full_predictions=fit['predictions'], restricted={name: str(path) for name, path in paths.items()},
        horizons=args.horizons, draws=args.draws, seed=args.seed,
        weighting='equal outcome means; diseases pooled within each outcome',
        note='Macro bootstrap uses a fixed point-estimable outcome roster per horizon; no individual patient data exported'), indent=2) + '\n')
    with zipfile.ZipFile(output / 'recent_record_review.zip', 'w', zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(output.glob('*.csv')):
            archive.write(path, path.name)
        archive.write(output / 'refresh.json', 'refresh.json')
        for path in sorted((output / 'figures').glob('*')):
            if path.suffix in {'.pdf', '.png'}:
                archive.write(path, 'figures/' + path.name)
    print(f'Complete: {output / "recent_record_review.zip"}', flush=True)


if __name__ == '__main__':
    main()
