"""Rebuild paired means from saved cell estimates/draws, without prediction scoring."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from opera.evaluation.secondary_analysis import effect_summaries
from opera.run.analyse_locked_predictions import file_hash


def rebuild(analysis_dir, output_dir):
    analysis_dir, output_dir = Path(analysis_dir), Path(output_dir)
    if output_dir.exists():
        raise FileExistsError('Use a fresh summary output directory')
    manifest_path = analysis_dir / 'analysis_manifest.json'
    manifest = json.loads(manifest_path.read_text()) if manifest_path.is_file() else {}
    config = manifest.get('config', {})
    contrasts = pd.read_csv(analysis_dir / 'contrasts.csv')
    index = pd.read_csv(analysis_dir / 'paired_bootstrap_index.csv')
    with np.load(analysis_dir / 'paired_bootstrap_draws_INTERNAL.npz') as saved:
        gains = saved['gains']
    if len(index) != len(gains):
        raise ValueError('Bootstrap index and draw archive row counts differ')
    index['gain_draws'] = list(gains)
    keys = ['cohort', 'outcome', 'horizon_days']
    strata = config.get('cell_summary_strata') or list(dict.fromkeys([
        'support_bin', 'outcome_family', 'endpoint_group', 'cohort',
        *config.get('cell_groups', {})]))
    columns = [c for c in strata if c in contrasts and c not in keys]
    metadata = contrasts[keys + columns].drop_duplicates()
    raw = contrasts.drop(columns=columns)
    results = {}
    for scheme, filename in [('equal_cell', 'effect_summaries.csv'),
                             ('disease_balanced', 'effect_summaries_disease_balanced.csv')]:
        _, results[filename] = effect_summaries(raw, index, metadata,
                                               strata=strata, aggregation=scheme)
    output_dir.mkdir(parents=True)
    for filename, data in results.items():
        data.to_csv(output_dir / filename, index=False)
    comparison = pd.concat([data.assign(weighting=('Equal cell' if name == 'effect_summaries.csv'
                                                  else 'Disease balanced'))
                            for name, data in results.items()], ignore_index=True)
    comparison['interval'] = [f'{low:.3f} to {high:.3f}' if np.isfinite([low, high]).all() else None
                              for low, high in zip(comparison.lower, comparison.upper)]
    comparison.loc[(comparison.stratum == 'all_cells') & comparison.metric.isin(['auc', 'brier', 'ipa'])].to_csv(
        output_dir / 'paired_mean_comparison.csv', index=False)
    versions = {}
    for name in ['contrasts.csv', 'paired_bootstrap_index.csv', 'paired_bootstrap_draws_INTERNAL.npz']:
        path = analysis_dir / name
        stat = path.stat()
        versions[name] = dict(path=str(path.resolve()), bytes=stat.st_size, mtime_ns=stat.st_mtime_ns)
    (output_dir / 'summary_manifest.json').write_text(json.dumps(dict(
        source_analysis=str(analysis_dir.resolve()), inputs=versions,
        analysis_manifest_sha256=file_hash(manifest_path) if manifest_path.is_file() else None,
        bootstrap_method=manifest.get('bootstrap_method', 'see source analysis manifest'),
        source_code_sha256=file_hash(Path(__file__).parents[1] / 'evaluation/secondary_analysis.py'),
        note='Existing supported cells and paired draws reused; no scoring, resampling or roster changes. Disease-balanced weights: disease, outcome, horizon; endpoint-family weighting is not applied.'), indent=2))
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--analysis-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    rebuild(args.analysis_dir, args.output_dir)
    print(args.output_dir.resolve())


if __name__ == '__main__':
    main()
