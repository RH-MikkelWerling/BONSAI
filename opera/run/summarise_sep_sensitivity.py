"""Compare SEP sensitivity arms on paired, supported tuning cells only."""
import argparse
import json
from pathlib import Path
import pandas as pd
from opera.run.analyse_locked_predictions import read_artifact, read_table
from opera.evaluation.secondary_analysis import patient_bootstrap, paired_analysis, effect_summaries


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--patient-metadata', type=Path)
    args = parser.parse_args()
    manifest = json.loads((args.root / 'sensitivity.json').read_text())
    metadata_path = args.patient_metadata or manifest.get('patient_metadata')
    if not metadata_path:
        parser.error('Supply --patient-metadata with subject_id and cohort_fine')
    metadata = read_table(metadata_path)
    frames = []
    for arm in ('exclude_sep', 'include_sep'):
        for seed in manifest['seeds']:
            path = args.root / arm / f'joint_s{seed}/tuning_predictions/direct_cr_predictions.parquet'
            frame, _ = read_artifact(dict(path=str(path), format='direct_cif', model=arm,
                                          seed=seed, split='tuning', cohort_from_metadata=True,
                                          data_contract='sep_sensitivity_matched_local'),
                                     metadata, 'cif_ipcw')
            frames.append(frame)
    frame = pd.concat(frames, ignore_index=True)
    draws = patient_bootstrap(frame, 1000, 2026, method='multiplier')
    result = paired_analysis(frame, {'include_vs_exclude': 'gain(include_sep,exclude_sep)'},
                             draws=draws, replicates=1000, min_cases=5, min_controls=5,
                             patient_gain_contrasts=[])
    cells = result['contrasts'][['cohort', 'outcome', 'horizon_days']].drop_duplicates()
    output = args.root / 'paired_tuning_analysis'
    output.mkdir(exist_ok=False)
    result['contrasts'].to_csv(output / 'paired_cells.csv', index=False)
    summaries = []
    for aggregation in ('equal_cell', 'disease_balanced'):
        _, summary = effect_summaries(result['contrasts'], result['bootstrap_draws'], cells,
                                      aggregation=aggregation)
        summary['aggregation'] = aggregation
        summaries.append(summary)
    pd.concat(summaries).to_csv(output / 'paired_means.csv', index=False)
    print(f'Saved {output}; intervals condition on each fine-tuning seed and one pretraining seed.')


if __name__ == '__main__':
    main()
