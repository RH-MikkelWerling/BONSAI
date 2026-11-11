"""Compare local/national laboratory bin coverage and replay a bounded national sample.

All outputs are aggregate; no patient IDs or individual measurements are written.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd
import polars as pl
import pyarrow.parquet as pq

BIN = re.compile(r'//BIN_(\d+)$')


def summarize(root):
    files = sorted((Path(root) / 'data').rglob('*.parquet'))
    if not files:
        raise ValueError(f'No data shards under {root}/data')
    codes = pl.scan_parquet([str(p) for p in files]).group_by('code').agg(pl.len().alias('events')).collect().to_pandas()
    labs = codes.loc[codes.code.str.startswith('LABTEST//')].copy()
    labs['binned'] = labs.code.str.contains(r'//BIN_\d+$')
    labs['base_code'] = labs.code.str.replace(r'//BIN_\d+$', '', regex=True)
    labs['binned_events'] = labs.events.where(labs.binned, 0)
    return files, labs.groupby('base_code')[['events', 'binned_events']].sum()


def replay(files, metadata, per_shard, diagnostics=None, *, check_suffix=True):
    if metadata.code.duplicated().any():
        raise ValueError('External numeric metadata has duplicate codes')
    transforms = metadata.set_index('code').to_dict('index')
    checked = mismatches = suffix_mismatches = unknown = near_edge = 0
    mismatch_counts = Counter()
    for path in files:
        used = 0
        schema = pq.ParquetFile(path).schema_arrow.names
        required = ['code', 'numeric_value', 'numeric_value_bin']
        if not set(required) <= set(schema):
            raise ValueError(f'Cannot replay numeric annotation: missing columns in {path.name}')
        for batch in pq.ParquetFile(path).iter_batches(batch_size=65536, columns=required):
            frame = batch.to_pandas()
            frame = frame.loc[frame.code.str.startswith('LABTEST//') & frame.numeric_value.notna()]
            for row in frame.head(per_shard-used).itertuples(index=False):
                match = BIN.search(row.code)
                code = BIN.sub('', row.code)
                transform = transforms.get(code)
                expected = None
                edge_distance = None
                if transform is None:
                    unknown += 1
                else:
                    low = transform.get('numeric/bin_1')
                    high = transform.get('numeric/bin_99')
                    hard_low = transform.get('numeric/hard_minimum')
                    hard_high = transform.get('numeric/hard_maximum')
                    valid = np.isfinite(row.numeric_value) and pd.notna(low) and (pd.isna(hard_low) or row.numeric_value >= hard_low) and (pd.isna(hard_high) or row.numeric_value <= hard_high)
                    edges = transform.get('numeric/bin_edges')
                    if valid and edges is not None:
                        normalized = 0.0 if high <= low else (np.clip(row.numeric_value, low, high)-low)/(high-low)
                        edge_array = np.asarray(edges, dtype=float)
                        expected = int(np.count_nonzero(edge_array <= normalized))
                        edge_distance = float(np.min(np.abs(edge_array-normalized))) if len(edge_array) else None
                actual = None if pd.isna(row.numeric_value_bin) else int(row.numeric_value_bin)
                mismatches += actual != expected
                if actual != expected:
                    close = edge_distance is not None and edge_distance <= 1e-6
                    near_edge += int(close)
                    mismatch_counts[(code, str(expected), str(actual), close)] += 1
                if check_suffix:
                    suffix_mismatches += (int(match.group(1)) if match else None) != actual
                checked += 1
                used += 1
            if used >= per_shard:
                break
    if diagnostics is not None:
        diagnostics.extend(dict(code=k[0], expected_bin=k[1], actual_bin=k[2],
            near_normalized_edge=k[3], rows=n) for k, n in mismatch_counts.most_common())
    return dict(mismatches_near_bin_edge=near_edge, sampled_numeric_labs=checked, external_transform_mismatches=int(mismatches),
                bin_suffix_mismatches=int(suffix_mismatches), sampled_without_external_transform=unknown)


def intermediate_replays(root, metadata, per_shard):
    results = {}
    for stage in ('annotate_numeric_values', 'join_numeric_bins', 'join_lab_text_values', 'finalize_MEDS_data'):
        directory = Path(root) / stage
        files = sorted(p for p in directory.rglob('*.parquet')
                       if {'code', 'numeric_value', 'numeric_value_bin'} <=
                       set(pq.ParquetFile(p).schema_arrow.names)) if directory.exists() else []
        if not files:
            results[stage] = {'status': 'intermediate_not_available'}
            continue
        diagnostics = []
        result = replay(files, metadata, per_shard, diagnostics,
                        check_suffix=stage != 'annotate_numeric_values')
        result['numeric_value_dtypes'] = sorted({str(pq.ParquetFile(p).schema_arrow.field('numeric_value').type) for p in files})
        result['shards'] = len(files)
        result['status'] = 'sample_replayed'
        results[stage] = result
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--national-meds', type=Path, required=True)
    parser.add_argument('--local-meds', type=Path)
    parser.add_argument('--intermediate-only', action='store_true',
                        help='Replay retained conversion stages and final data, without rescanning local coverage')
    parser.add_argument('--bundle', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--sample-per-shard', type=int, default=10000)
    args = parser.parse_args()
    if args.sample_per_shard <= 0:
        parser.error('sample-per-shard must be positive')
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    if args.intermediate_only:
        metadata = pl.read_json(args.bundle/'preprocessing/numeric_metadata.json').to_pandas()
        results = intermediate_replays(args.national_meds, metadata, args.sample_per_shard)
        final_files = sorted((args.national_meds/'data').rglob('*.parquet'))
        if not final_files:
            raise ValueError('Final data shards are absent')
        results['final_data'] = replay(final_files, metadata, args.sample_per_shard)
        results['final_data']['numeric_value_dtypes'] = sorted({str(pq.ParquetFile(p).schema_arrow.field('numeric_value').type) for p in final_files})
        results['scope'] = 'Bounded first numeric lab rows per shard; stage samples are not event-matched. Tests supplied transforms, not national training implementation equivalence.'
        args.output_dir.mkdir(parents=True)
        (args.output_dir/'intermediate_report.json').write_text(json.dumps(results, indent=2))
        print(json.dumps(results, indent=2))
        annotated = results['annotate_numeric_values']
        if annotated.get('sampled_numeric_labs', 0) == 0 or annotated.get('external_transform_mismatches', 0):
            raise SystemExit('Pre-finalization replay absent or mismatched; do not approve yet')
        return
    if args.local_meds is None:
        parser.error('--local-meds is required unless --intermediate-only is used')
    files, national = summarize(args.national_meds)
    _, local = summarize(args.local_meds)
    comparison = national.add_suffix('_national').join(local.add_suffix('_local'), how='outer').fillna(0)
    comparison['newly_binned_concept'] = (comparison.binned_events_national > 0) & (comparison.binned_events_local == 0)
    metadata_path = args.bundle/'preprocessing/numeric_metadata.json'
    metadata = pl.read_json(metadata_path).to_pandas()
    diagnostics = []
    report = replay(files, metadata, args.sample_per_shard, diagnostics)
    report.update(national_lab_events=int(national.events.sum()),
        national_binned_lab_events=int(national.binned_events.sum()),
        local_lab_events=int(local.events.sum()), local_binned_lab_events=int(local.binned_events.sum()),
        newly_binned_concepts=int(comparison.newly_binned_concept.sum()),
        numeric_metadata_sha256=hashlib.sha256(metadata_path.read_bytes()).hexdigest(),
        sampling='First numeric laboratory rows per shard, bounded deterministic sample; not a complete replay or matched-event comparison')
    args.output_dir.mkdir(parents=True)
    comparison.to_csv(args.output_dir/'lab_binning_comparison.csv')
    pd.DataFrame(diagnostics, columns=['code','expected_bin','actual_bin','near_normalized_edge','rows']).to_csv(args.output_dir/'bin_mismatch_summary.csv', index=False)
    (args.output_dir/'report.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    if not report['sampled_numeric_labs'] or report['external_transform_mismatches'] or report['bin_suffix_mismatches']:
        raise SystemExit('Numeric replay failed; inspect before tokenization')


if __name__ == '__main__':
    main()
