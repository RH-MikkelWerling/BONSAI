"""Export all-outcome aggregate maps on an adapted joint/OPERA story projection.

Reuses map_grids, its smoothing and suppression. No UMAP, inference or training.
Only named aggregate products enter the ZIP; coordinates and predictions stay server-side.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED

for name in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS']:
    os.environ.setdefault(name, '1')
import numpy as np
import pandas as pd
import pyarrow.parquet as pq


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def outcome_grids(coords, predictions, reference_meta, *, horizon=365, bandwidth=.009):
    from opera.visualization.encoder_embedding_comparison import map_grids
    p = predictions.copy()
    p['patient_id'] = p.subject_id.astype(str)
    if p.patient_id.duplicated().any():
        raise ValueError('Duplicate patient prediction for outcome')
    p = p.set_index('patient_id')
    c = coords.loc[coords.patient_id.isin(p.index)].copy()
    if c.empty:
        raise ValueError('No patients with outcome predictions on the reference map')
    c['predicted'] = p.loc[c.patient_id, f'cif_{horizon}d'].to_numpy(float)
    if not np.isfinite(c.predicted).all() or not c.predicted.between(0, 1).all():
        raise ValueError('Invalid outcome risks')
    s = p.loc[c.patient_id, ['time_days', 'event']].reset_index()
    if not np.isfinite(s[['time_days', 'event']].to_numpy(float)).all() or (s.time_days < 0).any() or not s.event.isin([0, 1, 2]).all():
        raise ValueError('Invalid outcome follow-up')
    r = reference_meta
    cells, meta = map_grids(c, s, grid=int(r['grid']), horizon=horizon, bandwidth=bandwidth,
                           frame=[r[k] for k in ['x0', 'x1', 'y0', 'y1']])
    # Density/index layers are not needed for the outcome plates. Retain count
    # for the identical renderer's common background/support mask.
    cells = cells.loc[cells.layer.isin(['count', 'predicted', 'mortality'])].copy()
    meta['horizon_days'] = horizon
    meta['prediction_patients'] = len(p)
    meta['prediction_patients_on_reference_map'] = len(c)
    meta['status'] = 'complete' if {'predicted', 'mortality'} <= set(cells.layer) else 'unsupported_risk_surface'
    return cells, meta


def run(story, output, *, family='opera'):
    if family not in {'opera', 'joint'}:
        raise ValueError('Choose opera or joint family')
    label, model = ('Adapted OPERA', 'R_EA') if family == 'opera' else ('Adapted joint', 'J_EA')
    story, output = Path(story), Path(output)
    status = json.loads((story / 'story_status.json').read_text())
    if status['embeddings']['status'] != 'complete':
        raise ValueError('Complete final story embeddings first')
    source = Path(status['embeddings'][family]['prediction_sources'][label])
    coordpath = story / f'server_only/{family}_encoder_coordinates.csv'
    metapath = story / f'ready/tables/embedding_grids_{family}_meta.csv'
    coords = pd.read_csv(coordpath, dtype={'patient_id': str})
    coords = coords.loc[coords.model.eq(label)].copy()
    if coords.patient_id.duplicated().any() or coords.empty:
        raise ValueError(f'Need unique {label} reference coordinates')
    refs = pd.read_csv(metapath)
    reference, = refs.loc[refs.model.eq(label)].to_dict('records')
    if int(reference['grid']) != 800:
        raise ValueError('Need final 800-grid story export')
    outcomes = set()
    for batch in pq.ParquetFile(source).iter_batches(columns=['outcome'], batch_size=50000):
        outcomes.update(batch.column(0).to_pylist())
    identity = dict(family=family, predictions_sha256=sha(source), coordinates_sha256=sha(coordpath),
                    reference_meta_sha256=sha(metapath), implementation_sha256=sha(__file__),
                    grid_implementation_sha256=sha(Path(__file__).resolve().parents[1] / 'visualization/encoder_embedding_comparison.py'),
                    horizon_days=365, bandwidth=.009, min_count=5, min_support=25)
    output.mkdir(parents=True, exist_ok=True)
    import fcntl
    with (output / '.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        intent = output / 'intent_INTERNAL.json'
        if intent.exists() and json.loads(intent.read_text()) != identity:
            raise ValueError('Inputs/code changed: use a fresh output directory')
        intent.write_text(json.dumps(identity, indent=2))
        from opera.visualization.paper_taxonomy import outcome_endpoint_groups
        from opera.visualization.outcome_names import outcome_display_name
        domains = outcome_endpoint_groups()
        rows = []
        files = []
        for i, outcome in enumerate(sorted(outcomes)):
            stem = f'outcome_{i:03d}'
            gridpath = output / f'{stem}.csv.gz'
            receipt = output / f'{stem}.json'
            if receipt.exists():
                row = json.loads(receipt.read_text())
                if not gridpath.exists() or sha(gridpath) != row['grid_sha256']:
                    raise ValueError('Cached aggregate grid changed')
            else:
                p = pq.read_table(source, columns=['subject_id', 'time_days', 'event', 'cif_365d'],
                                  filters=[('outcome', '=', outcome)]).to_pandas()
                cells, meta = outcome_grids(coords, p, reference)
                row = meta.iloc[0].to_dict()
                row.update(outcome=outcome, label=outcome_display_name(outcome),
                           domain=domains.get(outcome, 'Other'), file=gridpath.name)
                temp = gridpath.with_suffix('.tmp')
                cells.to_csv(temp, index=False, compression='gzip')
                temp.replace(gridpath)
                row['grid_sha256'] = sha(gridpath)
                receipt.write_text(json.dumps(row, indent=2))
            rows.append(row)
            files.append(gridpath)
            print(f'Outcome grids {i + 1}/{len(outcomes)}: {outcome}: {row["status"]}', flush=True)
        pd.DataFrame(rows).to_csv(output / 'outcome_grid_meta.csv', index=False)
        manifest = dict(status='complete', model=model, exploratory_display=True,
                        same_projection_as=f'story {label}', **identity,
                        outcomes=len(rows), supported=sum(r['status'] == 'complete' for r in rows),
                        note='IPCW display surfaces; fixed common frame, smoothing and suppression from story maps; no patient exports')
        (output / 'manifest.json').write_text(json.dumps(manifest, indent=2))
        archive_name = 'download_outcome_grids_joint.zip' if family == 'joint' else 'download_outcome_grids.zip'
        with ZipFile(output / archive_name, 'w', ZIP_DEFLATED) as archive:
            for path in files + [output / 'outcome_grid_meta.csv', output / 'manifest.json']:
                archive.write(path, path.name)
        print(f'Download: {output / archive_name}', flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--story-root', type=Path, default=Path('data/story_closure_20261116'))
    p.add_argument('--family', choices=['opera', 'joint'], default='opera')
    p.add_argument('--output-dir', type=Path)
    a = p.parse_args()
    output = a.output_dir or Path('data/outcome_embedding_grids_joint_20261119' if a.family == 'joint'
                                  else 'data/outcome_embedding_grids_20261119')
    run(a.story_root, output, family=a.family)


if __name__ == '__main__':
    main()
