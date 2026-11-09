"""Re-render selected figures from an extracted review ZIP, using local aggregates.

Never trains, reads server paths, or substitutes missing data. New outputs only.
Example: python scripts/render_offline_review.py --bundle scratch/figure_audit_20261106
  --output-dir scratch/figure_audit_20261106/iterations/pass01
  --figures supp_specialisation supp_clinical_index_summary
"""
import argparse
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import yaml

REPO = Path(__file__).resolve().parents[1]


def prepare(bundle, ids):
    recipe = yaml.safe_load((REPO / 'opera/configs/paper_results_figures.full.example.yaml').read_text())
    lookup = {p['id']: p for p in recipe['figures']}
    unknown = set(ids) - set(lookup)
    if unknown:
        raise ValueError(f'No aggregate figure recipe for {sorted(unknown)}')
    tables = bundle / 'aggregates/figure_inputs/plot_tables'
    plates, capability = [], []
    for name in ids:
        plate = copy.deepcopy(lookup[name])
        missing = []
        for section in ('tables', 'json_inputs'):
            for key, source in plate.get(section, {}).items():
                if '/source_removal/' in source:
                    tail = source.split('/source_removal/', 1)[1].replace('${OPERA_PAPER_PLOT_SPLIT}', 'tuning')
                    path = bundle / 'aggregates/source_removal' / tail
                elif '/clinical_scores/' in source:
                    path = bundle / 'review/reports/clinical_scores' / source.split('/clinical_scores/', 1)[1]
                else:
                    path = tables / Path(source).name
                # Some derived clinical tables live in the paper-number reports.
                if not path.is_file():
                    other = bundle / 'review/reports/paper_numbers' / path.name
                    if other.is_file():
                        path = other
                plate[section][key] = str(path.resolve())
                if not path.is_file():
                    missing.append(f'{key}: {path.name}')
        capability.append(dict(id=name, offline=not missing, missing=missing))
        if not missing:
            # Primary identity is declared by the clinical manifest in this ZIP.
            manifest = bundle / 'review/clinical_scores/report_manifest.json'
            if name == 'fig_headline' and manifest.is_file():
                identities = {s['model'] for s in json.loads(manifest.read_text())['score_definitions']}
                if len(identities) != 1:
                    raise ValueError('Clinical model identity is ambiguous')
                plate.setdefault('options', {})['clinical_model'] = identities.pop()
            plates.append(plate)
    return dict(figures=plates), capability


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--figures', nargs='+', required=True)
    parser.add_argument('--preview', action='store_true', help='List exact available/missing inputs only')
    args = parser.parse_args()
    bundle = args.bundle.resolve()
    output = args.output_dir.resolve()
    if output.exists():
        raise ValueError('Use a new iteration directory; source figures stay unchanged')
    recipe, capability = prepare(bundle, args.figures)
    if args.preview:
        print(json.dumps(capability, indent=2))
        return
    if any(not c['offline'] for c in capability):
        raise ValueError('Missing local inputs: ' + json.dumps(capability))
    output.mkdir(parents=True)
    (output / 'offline_inputs.json').write_text(json.dumps(capability, indent=2))
    env = {**os.environ, 'CUDA_VISIBLE_DEVICES': '', 'MPLBACKEND': 'Agg',
           'OPERA_PAPER_FIGURE_MARK': 'TUNING REHEARSAL — local render'}
    # Separate processes bound memory when several figures share large tables.
    for plate in recipe['figures']:
        path = output / (plate['id'] + '.yaml')
        path.write_text(yaml.safe_dump(dict(figures=[plate]), sort_keys=False))
        subprocess.run([sys.executable, '-m', 'opera.run.render_paper_results',
            '--config', str(path), '--output-dir', str(output / plate['id']), '--execute'],
            cwd=REPO, env=env, check=True)
    print(f'Local renders: {output}')


if __name__ == '__main__':
    main()
