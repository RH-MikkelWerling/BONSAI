"""Add prespecified per-patient national-token coverage without changing source metadata.

Secure-server computation only: output patient metadata remains INTERNAL. Uses
existing tokenised sequences and frozen vocabulary, never fits or predicts.
"""
import argparse
import gc
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
import yaml


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def merge_coverage(metadata, coverage):
    left, right = metadata.copy(), coverage.copy()
    left['subject_id'], right['subject_id'] = left.subject_id.astype(str), right.subject_id.astype(str)
    if left.subject_id.duplicated().any() or right.subject_id.duplicated().any():
        raise ValueError('Metadata/coverage must have unique patient IDs')
    if set(left.subject_id) - set(right.subject_id):
        raise ValueError('National sequences do not cover every metadata patient; verify sequence paths')
    values = right.national_vocabulary_coverage.dropna()
    if not np.isfinite(values).all() or not values.between(0, 1).all():
        raise ValueError('Coverage must be a fraction or missing for empty clinical history')
    return left.drop(columns='national_vocabulary_coverage', errors='ignore').merge(
        right, on='subject_id', how='left', validate='one_to_one', sort=False)


def run(analysis, bundle_path, output):
    import torch
    from opera.functional.outcomes import _pandas_abspos
    from opera.run.analyse_locked_predictions import read_table
    from opera.run.build_paper_patient_metadata import national_coverage
    from opera.modules.datamodules.OutcomeFinetuneDataModule import load_subject_pool
    analysis, bundle_path, output = map(Path, [analysis, bundle_path, output])
    if output.exists():
        raise FileExistsError('Use a fresh coverage output directory')
    config = yaml.safe_load(analysis.read_text())
    metadata_path = Path(config['patient_metadata'])
    metadata = read_table(metadata_path)
    if not {'subject_id', 'index_date'} <= set(metadata):
        raise ValueError('Source patient metadata needs actual index_date; do not guess dates')
    bundle = json.loads((bundle_path / 'bundle.json').read_text())
    row, = [r for r in bundle['runs'] if r['name'] == 'seed42/EA/R']
    fitconfig_path = bundle_path / row['config_path']
    if sha(fitconfig_path) != row['config_sha256']:
        raise ValueError('Frozen adapted R config hash changed')
    fitconfig = yaml.safe_load(fitconfig_path.read_text())
    vocabulary_path = Path(fitconfig['paths']['vocabulary'])
    paths = sorted({str(p) for c in fitconfig['cohorts'].values() for p in c.get('subject_data_paths', [])})
    if not paths or any('${' in p for p in paths + [str(vocabulary_path)]):
        raise ValueError('Need resolved national subject_data_paths and vocabulary from the frozen fit')
    vocabulary = torch.load(vocabulary_path, map_location='cpu', weights_only=False)
    if '[UNK]' not in vocabulary:
        raise ValueError('National vocabulary has no declared [UNK] token')
    dates = pd.to_datetime(metadata.index_date, errors='raise')
    if dates.isna().any() or metadata.subject_id.isna().any():
        raise ValueError('Index dates/IDs must be complete')
    index = dict(zip(metadata.subject_id.astype(str), _pandas_abspos(dates)))
    pieces = []
    for i, path in enumerate(paths):
        pool = load_subject_pool([path])
        def subjects():
            for s in pool:
                key = str(s['subject_id'])
                if key not in index:
                    continue
                times = np.asarray(s['abspos'], float)
                if not np.isfinite(times).all() or (np.diff(times) < 0).any():
                    raise ValueError('National sequence times must be finite and ordered')
                yield dict(subject_id=key, abspos=s['abspos'], code=s['code'])
        pieces.append(national_coverage(subjects(), index, vocabulary))
        del pool
        gc.collect()
        print(f'National coverage shard {i + 1}/{len(paths)}', flush=True)
    merged = merge_coverage(metadata, pd.concat(pieces, ignore_index=True))
    if not merged.national_vocabulary_coverage.notna().any():
        raise ValueError('No patients have evaluable clinical history')
    output.mkdir(parents=True)
    merged.to_parquet(output / 'patient_metadata_INTERNAL.parquet', index=False)
    values = merged.national_vocabulary_coverage
    groups = pd.cut(values, bins=[0, .8, .95, 1.], include_lowest=True).astype('string').fillna('No clinical history')
    counts = groups.value_counts().rename_axis('coverage_group').reset_index(name='n_patients')
    counts.to_csv(output / 'coverage_counts.csv', index=False)
    report = dict(status='complete', source_metadata_sha256=sha(metadata_path),
        output_metadata_sha256=sha(output / 'patient_metadata_INTERNAL.parquet'),
        frozen_config_sha256=sha(fitconfig_path), vocabulary_sha256=sha(vocabulary_path),
        sequence_sources=[dict(path=p, bytes=Path(p).stat().st_size, mtime_ns=Path(p).stat().st_mtime_ns) for p in paths],
        patients=len(merged), with_clinical_history=int(values.notna().sum()),
        definition='Fraction of national-tokenised clinical pre-index tokens that are not [UNK]; background/special tokens excluded',
        patient_metadata='patient_metadata_INTERNAL.parquet',
        note='No fits, predictions, outcomes, split labels or original metadata changed; patient output stays server-only')
    (output / 'coverage_manifest.json').write_text(json.dumps(report, indent=2))
    print(f'Prepared server-only metadata: {output / "patient_metadata_INTERNAL.parquet"}')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-analysis', type=Path, default=Path('data/paper_256_v3/analysis/local_test_20261109_four_allocations/analysis_inventory/analysis.yaml'))
    p.add_argument('--bundle-dir', type=Path, default=Path('data/paper_256_v3_national_20261111/adapted_fits'))
    p.add_argument('--output-dir', type=Path, default=Path('data/national_coverage_20261119'))
    a = p.parse_args()
    run(a.source_analysis, a.bundle_dir, a.output_dir)


if __name__ == '__main__':
    main()
