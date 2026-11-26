"""Refresh aggregate risk maps from saved coordinates; no UMAP, fitting or GPU.

Writes a separate export and leaves all existing interpretation caches untouched.
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED

import pandas as pd

from opera.run.interpretation_worker import claim, digest, write_json


def pack(directory, target, files):
    temporary = target.with_suffix('.writing')
    with ZipFile(temporary, 'w', ZIP_DEFLATED) as archive:
        for path in files:
            archive.write(path, path.relative_to(directory))
    temporary.replace(target)


def run(story, output, all_outcomes=True, render=False):
    import pyarrow.parquet as pq
    from opera.run.finish_story_inputs import _survival
    from opera.run.export_outcome_embedding_grids import outcome_grids
    from opera.visualization.encoder_embedding_comparison import map_grids
    story, output = story.resolve(), output.resolve()
    status = json.loads((story / 'story_status.json').read_text())['embeddings']
    if status['status'] != 'complete' or status.get('projection_version') != 'held_out_descriptive_v1':
        raise ValueError('Need completed held-out-fitted maps')
    identity = {'story_status_sha256': digest(story / 'story_status.json'),
                'implementation_sha256': {str(path): digest(path) for path in [
                    Path(__file__).resolve(), Path(__file__).resolve().parents[1] / 'visualization/kernel_risk_support.py',
                    Path(__file__).resolve().parents[1] / 'visualization/encoder_embedding_comparison.py',
                    Path(__file__).resolve().with_name('export_outcome_embedding_grids.py'),
                    Path(__file__).resolve().parents[1] / 'visualization/censoring_aware_embeddings.py']},
                'method': 'ipcw_kernel_kish_ess_v1', 'min_effective_support': 25,
                'bandwidth': .009, 'grid': 800, 'all_outcomes': all_outcomes, 'inputs': {}}
    for family in ('joint', 'opera'):
        paths = [story / f'server_only/{family}_encoder_coordinates.csv',
                 story / f'ready/tables/embedding_grids_{family}_meta.csv']
        paths += [Path(p) for p in status[family]['prediction_sources'].values()]
        for path in paths:
            identity['inputs'][str(path)] = digest(path)
    output.mkdir(parents=True, exist_ok=True)
    with claim(output / '.refresh.lock') as owned:
        if not owned:
            raise RuntimeError('Another risk-support refresh owns this output')
        intent = output / 'intent_INTERNAL.json'
        if intent.exists() and json.loads(intent.read_text()) != identity:
            raise ValueError('Inputs/code changed: use a fresh output directory')
        write_json(intent, identity)
        source = Path(status['opera']['prediction_sources']['Adapted OPERA'])
        survival = _survival(source)
        survival['patient_id'] = survival.patient_id.astype(str)
        tables = output / 'tables'
        tables.mkdir(exist_ok=True)
        diagnostics = []
        for family in ('joint', 'opera'):
            coords = pd.read_csv(story / f'server_only/{family}_encoder_coordinates.csv', dtype={'patient_id': str})
            old_meta = pd.read_csv(story / f'ready/tables/embedding_grids_{family}_meta.csv')
            frame = old_meta.iloc[0][['x0','x1','y0','y1']].to_numpy(float)
            gridpath = tables / f'embedding_grids_{family}.csv.gz'
            metapath = tables / f'embedding_grids_{family}_meta.csv'
            complete = tables / f'embedding_grids_{family}.complete.json'
            if not complete.exists():
                cells, meta = map_grids(coords, survival, grid=800, bandwidth=.009,
                                       frame=frame, risk_support='ipcw_ess', export_support=True)
                temporary = gridpath.with_suffix('.writing')
                cells.to_csv(temporary, index=False, compression='gzip')
                temporary.replace(gridpath)
                temporary = metapath.with_suffix('.writing')
                meta.to_csv(temporary, index=False)
                temporary.replace(metapath)
                write_json(complete, {'grid_sha256':digest(gridpath), 'meta_sha256':digest(metapath)})
            else:
                saved = json.loads(complete.read_text())
                if digest(gridpath) != saved['grid_sha256'] or digest(metapath) != saved['meta_sha256']:
                    raise ValueError('Cached supported encoder grid changed')
                meta = pd.read_csv(metapath)
            diagnostics.append(meta.assign(family=family, outcome='overall_survival'))
            print(f'{family}: encoder maps refreshed with effective support', flush=True)
            if not all_outcomes:
                continue
            label, model = ('Adapted joint', 'J_EA') if family == 'joint' else ('Adapted OPERA', 'R_EA')
            prediction = Path(status[family]['prediction_sources'][label])
            selected = coords.loc[coords.model.eq(label)]
            reference, = meta.loc[meta.model.eq(label)].to_dict('records')
            outcomes = set()
            for batch in pq.ParquetFile(prediction).iter_batches(columns=['outcome']):
                outcomes.update(batch.column(0).to_pylist())
            from opera.visualization.paper_taxonomy import outcome_endpoint_groups
            from opera.visualization.outcome_names import outcome_display_name
            domains = outcome_endpoint_groups()
            destination = output / f'outcome_grids_{family}'
            destination.mkdir(exist_ok=True)
            rows, files = [], []
            for i, outcome in enumerate(sorted(outcomes)):
                path = destination / f'outcome_{i:03d}.csv.gz'
                receipt = path.with_suffix('.json')
                if receipt.exists():
                    row = json.loads(receipt.read_text())
                    if digest(path) != row['grid_sha256']:
                        raise ValueError('Saved supported outcome grid changed')
                else:
                    p = pq.read_table(prediction, columns=['subject_id','time_days','event','cif_365d'],
                                      filters=[('outcome','=',outcome)]).to_pandas()
                    cells, meta_out = outcome_grids(selected, p, reference, risk_support='ipcw_ess')
                    temporary = path.with_suffix('.writing')
                    cells.to_csv(temporary, index=False, compression='gzip')
                    temporary.replace(path)
                    row = meta_out.iloc[0].to_dict()
                    row.update(outcome=outcome, label=outcome_display_name(outcome),
                               domain=domains.get(outcome,'Other'), file=path.name, grid_sha256=digest(path))
                    write_json(receipt, row)
                rows.append(row)
                files.append(path)
                print(f'{family}: {i+1}/{len(outcomes)} {outcome}: {row["status"]}', flush=True)
            pd.DataFrame(rows).to_csv(destination / 'outcome_grid_meta.csv', index=False)
            write_json(destination / 'manifest.json', {
                'status':'complete','model':model,'outcomes':len(rows), 'supported':sum(r['status']=='complete' for r in rows),
                'risk_support_method':'ipcw_kernel_kish_ess_v1', 'projection_version':'held_out_descriptive_v1',
                'note':'Same saved coordinates and predictions; no refitting; unsupported surfaces remain explicit'})
            pack(destination, destination / 'download_outcome_grids.zip',
                 files + [destination / 'outcome_grid_meta.csv', destination / 'manifest.json'])
        pd.concat(diagnostics, ignore_index=True).to_csv(tables / 'risk_support_diagnostics.csv', index=False)
        render_receipt = output / 'render_complete_INTERNAL.json'
        if render:
            repo = Path(__file__).resolve().parents[2]
            subprocess.run([sys.executable, str(repo / 'scripts/render_encoder_maps_from_grids.py'),
                            str(tables), str(output / 'figures')], cwd=repo, check=True)
            if all_outcomes:
                for family in ('joint', 'opera'):
                    subprocess.run([sys.executable, str(repo / 'scripts/render_outcome_gallery.py'),
                                    str(output / f'outcome_grids_{family}'),
                                    str(output / f'figures/outcomes_{family}')], cwd=repo, check=True)
            write_json(render_receipt, {str(p.relative_to(output)):digest(p)
                                       for p in (output/'figures').rglob('*')
                                       if p.is_file() and p.suffix in {'.pdf','.png'}})
        rendered_files = []
        if render_receipt.exists():
            for name, expected in json.loads(render_receipt.read_text()).items():
                path = output / name
                if digest(path) != expected:
                    raise ValueError('Saved rendered risk-support figure changed')
                rendered_files.append(path)
        write_json(output / 'manifest.json', {
            'status':'complete','method':'ipcw_kernel_kish_ess_v1', 'all_outcomes':all_outcomes,
            'rendered':bool(rendered_files),
            'note':'Exploratory display-support amendment; same encoders, coordinates, endpoints and predictions'})
        files = [p for p in tables.glob('*') if p.name.endswith(('.csv', '.csv.gz'))]
        files += list(output.glob('outcome_grids_*/*.zip')) + [output / 'manifest.json']
        files += rendered_files
        archive = output / 'download_risk_support.zip'
        pack(output, archive, files)
        print('DOWNLOAD:', archive, flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--story-root', type=Path, default=Path('data/cutoff_recovery_20261121/exports/story_held_out_descriptive_v1'))
    p.add_argument('--output-dir', type=Path, default=Path('data/cutoff_recovery_20261121/exports/risk_support_ess_v1'))
    p.add_argument('--encoder-only', action='store_true', help='Quick encoder-map check; use a distinct output before an all-outcome refresh')
    p.add_argument('--render', action='store_true', help='Also render the existing encoder plates and outcome galleries on the server')
    a = p.parse_args()
    run(a.story_root, a.output_dir, all_outcomes=not a.encoder_only, render=a.render)


if __name__ == '__main__':
    main()
