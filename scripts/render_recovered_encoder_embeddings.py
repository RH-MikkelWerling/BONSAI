"""Render existing fine-tuned L/E/EA embeddings with the native comparison plot.

CPU only, no prediction-table reads. Outputs patient coordinates under a server-only
directory; the download ZIP uses the existing offline presentation packager.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import yaml


def select_stages(inventory):
    files = {f['path']: f for f in inventory['files']}
    wanted = {
        'joint': [('Local joint', '/predictions/L/', 'J_all_diseases_all_outcomes'),
                  ('National joint', '/predictions/E/', 'J'),
                  ('Adapted joint', '/predictions/EA/', 'J')],
        'opera': [('Local OPERA', '/predictions/L/', 'fix_J_OA_OWN'),
                  ('National OPERA', '/predictions/E/', 'R'),
                  ('Adapted OPERA', '/predictions/EA/', 'R')]}
    result = {}
    for family, definitions in wanted.items():
        result[family] = {}
        for label, origin, config in definitions:
            stages = []
            for split in ('tuning', 'held_out'):
                matches = [f for p, f in files.items()
                           if p.endswith('/patient_embeddings.metadata.json') and origin in p
                           and '/seed42/' in p and f.get('provenance', {}).get('config_name') == config
                           and f['provenance'].get('split') == split]
                if len(matches) != 1:
                    raise ValueError(f'Need one {label}/{split} export; found {len(matches)}')
                source = matches[0]['path'].replace('.metadata.json', '.npz')
                if source not in files or not files[source]['bytes']:
                    raise ValueError(f'Missing or empty recorded embedding export: {source}')
                stages.append(source)
            result[family][label] = stages
    return result


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inventory', type=Path, required=True)
    parser.add_argument('--patient-metadata', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    inventory = json.loads(args.inventory.read_text())
    stages = select_stages(inventory)
    if not args.execute:
        print(json.dumps({'stages': stages, 'action': 'CPU projection, native plots and portable ZIP'}, indent=2))
        return
    os.environ.update(CUDA_VISIBLE_DEVICES='', MPLBACKEND='Agg', OPERA_PAPER_EXPORT_PLOT_STATE='1',
                      OPERA_PAPER_FIGURE_MARK='',
                      OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', NUMBA_NUM_THREADS='1')
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    private = output / 'server_only'; private.mkdir()
    review = output / 'review'; review.mkdir()
    # Delayed imports: preview does not initialise torch or plotting libraries.
    from opera.run.build_paper_embedding_tables import embedding_comparison
    from opera.run.analyse_locked_predictions import read_table
    metadata = read_table(args.patient_metadata)
    group_map = yaml.safe_load(Path('opera/configs/paper_embedding_tables.example.yaml').read_text()).get('ipi_group_map')
    manifest = {'evaluation_split': 'held_out', 'analysis_root': str(output),
                'notes': ['Fine-tuned encoders only; no frozen-pretraining representations are implied.',
                          'Projection fitted on tuning patients and applied to held-out patients.',
                          'Separate maps show geometry, not evidence of improved discrimination.'],
                'steps': {}, 'source_hashes': {str(args.patient_metadata): sha(args.patient_metadata),
                                               str(args.inventory): sha(args.inventory)}}
    recipe = {'figures': []}
    for family, family_stages in stages.items():
        for paths in family_stages.values():
            for source in paths:
                path = Path(source)
                receipt = json.loads(path.with_suffix('.metadata.json').read_text())
                expected = next(f['provenance'] for f in inventory['files']
                                if f['path'] == str(path.with_suffix('.metadata.json')))
                if any(receipt.get(key) != expected.get(key) for key in ('checkpoint', 'config_name', 'split')):
                    raise ValueError(f'Embedding provenance changed: {path}')
                manifest['source_hashes'][source] = sha(path)
        coords, _ = embedding_comparison(family_stages, metadata, method='umap', group_map=group_map)
        table = private / f'{family}_matched_patient_embeddings.csv'
        coords.to_csv(table, index=False)
        recipe['figures'].append({'id': f'{family}_encoder_comparison',
                                 'function': 'figure_embedding_comparison',
                                 'tables': {'coords': str(table)},
                                 'options': {'model_order': list(family_stages)}})
        manifest['steps'][family] = {'patients': int(coords.patient_id.nunique()),
                                     'stages': family_stages,
                                     'projection': sorted(coords.projection.unique())}
    recipe_path = review / 'embeddings.yaml'
    recipe_path.write_text(yaml.safe_dump(recipe, sort_keys=False))
    (review / 'review_manifest.json').write_text(json.dumps(manifest, indent=2))
    subprocess.run([sys.executable, '-m', 'opera.run.render_paper_results', '--config', str(recipe_path),
                    '--output-dir', str(review / 'embeddings'), '--execute'], check=True)
    manifest['status'] = 'complete_for_requested_embedding_comparisons'
    (review / 'review_manifest.json').write_text(json.dumps(manifest, indent=2))
    subprocess.run([sys.executable, '-m', 'opera.run.prepare_offline_figure_bundle',
                    '--review-dir', str(review), '--analysis-root', str(output),
                    '--output-dir', str(output / 'portable'), '--output-zip', str(output / 'embedding_figures.zip'),
                    '--skip-clinical-refresh'], check=True)
    print(f'Download only: {output / "embedding_figures.zip"}', flush=True)


if __name__ == '__main__':
    main()
