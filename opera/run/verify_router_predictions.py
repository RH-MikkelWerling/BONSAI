"""Fresh tuning-only checkpoint replay; report.json contains aggregates only."""
import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from dotenv import dotenv_values

from opera.diagnostics.router_prediction_checks import compare_saved_predictions
from opera.run.audit_tuning_router import audit, digest, reconcile_metrics


def differences(left, right):
    delta = np.abs(np.asarray(left, float) - np.asarray(right, float))
    return dict(zip(['minimum', 'median', 'p90', 'p99', 'maximum'],
                    map(float, np.quantile(delta, [0, .5, .9, .99, 1]))))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--env-file', type=Path, default=Path('.env.rehearsal'))
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--patients-per-cohort', type=int, default=2)
    args = parser.parse_args()
    if args.patients_per_cohort < 1:
        parser.error('--patients-per-cohort must be positive')
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    env = dotenv_values(args.env_file)
    root = Path(env['OPERA_PAPER_ROOT']).resolve()
    analysis = Path(env['OPERA_PAPER_ANALYSIS_ROOT']).resolve()
    config = yaml.safe_load((analysis / 'analysis_inventory/analysis.yaml').read_text())
    report = {'status': 'running', 'env_file': str(args.env_file.resolve()),
              'note': 'Replay is a deterministic small sample, not a performance estimate.',
              'source_sha256': {str(Path(p).relative_to(Path(__file__).resolve().parents[2])): digest(Path(p))
                                for p in [__file__,
                                          str(Path(__file__).with_name('evaluate_direct_competing_risk.py')),
                                          str(Path(__file__).parents[1] / 'diagnostics/router_prediction_checks.py')]}}

    def save():
        (output / 'report.json').write_text(json.dumps(report, indent=2, default=str))

    save()
    try:
        report['provenance'] = audit(root, analysis, verify=True, route_diagnostics=True)
        report['metric_reconciliation'] = reconcile_metrics(config, analysis)
        save()
        selected = {}
        frames = {}
        for model in ['J_L', 'R_L']:
            rows = [r for r in report['provenance']['artifacts']
                    if r['model'] == model and r['split'] == 'tuning']
            if len(rows) != 1:
                raise ValueError(f'Expected exactly one tuning artifact for {model}')
            row = rows[0]
            for check in ['bundle_config_hash_matches', 'evaluation_matches_selected_checkpoint',
                          'checkpoint_hash_matches', 'prediction_hash_matches']:
                if row.get(check) is not True:
                    raise ValueError(f'{model} failed provenance check: {check}')
            selected[model] = row
            frames[model] = pd.read_parquet(row['predictions'])
            frames[model]['subject_id'] = frames[model].subject_id.astype(str)
        keys = ['subject_id', 'cohort', 'outcome']
        joined = frames['J_L'].merge(frames['R_L'], on=keys, suffixes=('_joint', '_router'),
                                     how='outer', validate='one_to_one', indicator=True)
        if not joined['_merge'].eq('both').all():
            raise ValueError('Joint/router saved files have different patient/outcome coverage')
        for field in ['time_days', 'event']:
            if not np.array_equal(joined[field + '_joint'], joined[field + '_router']):
                raise ValueError('Joint/router follow-up differs: ' + field)
        horizons = sorted(int(c[4:-1]) for c in frames['R_L'] if c.startswith('cif_') and c.endswith('d'))
        report['saved_joint_vs_router'] = {
            str(h): differences(joined[f'cif_{h}d_joint'], joined[f'cif_{h}d_router']) for h in horizons}
        # Deterministic, label-blind selection from every exported cohort.
        from opera.run.analyse_locked_predictions import read_table
        metadata = read_table(config['patient_metadata'])
        metadata['subject_id'] = metadata.subject_id.astype(str)
        patients = frames['R_L'][['subject_id']].drop_duplicates().merge(
            metadata[['subject_id', 'cohort_fine']], on='subject_id', validate='one_to_one')
        if len(patients) != frames['R_L'].subject_id.nunique():
            raise ValueError('Patient metadata does not cover saved predictions')
        patients = patients.sort_values(['cohort_fine', 'subject_id'])
        ids = sorted(set(patients.groupby('cohort_fine').head(args.patients_per_cohort).subject_id))
        internal = output / 'selected_subjects_INTERNAL.json'
        internal.write_text(json.dumps(ids))
        report['replay_patients'] = len(ids)
        report['replays'] = {}
        save()
        for model, row in selected.items():
            child = output / model
            config_path = Path(row['config_path']).resolve()
            checkpoint = Path(row['checkpoint_path']).resolve()
            command = [sys.executable, '-m', 'opera.run.evaluate_direct_competing_risk',
                       '--checkpoint', str(checkpoint), '--config-dir', str(config_path.parent),
                       '--config-name', config_path.stem, '--split', 'tuning',
                       '--horizons', ','.join(map(str, horizons)), '--batch-size', '1',
                       '--num-workers', '0', '--device', args.device,
                       '--audit-subjects', str(internal), '--output-dir', str(child)]
            if model == 'R_L':
                command.append('--audit-components')
            with (output / f'{model}.log').open('w') as log:
                subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
            replay = pd.read_parquet(child / 'direct_cr_predictions.parquet')
            if set(replay.subject_id.astype(str)) != set(ids):
                raise ValueError(f'{model} replay omitted selected patients')
            report['replays'][model] = compare_saved_predictions(frames[model], replay)
            if model == 'R_L':
                report['router_components'] = {
                    component: {str(h): differences(replay[f'cif_{h}d'], replay[f'audit_{component}_{h}d'])
                                for h in horizons}
                    for component in ['encoder_only', 'no_borrowing_zero']}
                report['component_interpretation'] = (
                    'Encoder-only uses the router checkpoint encoder/head, not the separately trained joint model. '
                    'No-borrowing zero removes borrowed source terms at unchanged gates and preserves own/global '
                    'corrections; this diagnostic is not the primary tuning-mean source-removal analysis.')
            save()
        report['status'] = 'replay_passed' if all(r['passed'] for r in report['replays'].values()) else 'replay_failed'
    except Exception as error:
        report['status'] = 'error'
        report['error'] = str(error)
        save()
        raise
    save()
    print(f"{report['status']}: {output / 'report.json'}")
    if report['status'] != 'replay_passed':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
