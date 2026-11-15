"""Fresh, single-model clinical comparison using the refreshed registered recipe.

Reuses tuning/test predictions, exports training predictions from an existing
fit if necessary, and runs the existing clinical renderer. Models run separately
so their full prediction panels never coexist in the renderer's memory.
"""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import yaml


MODELS = {'R_EA': 'R', 'J_EA': 'J', 'R_AB_EA': 'AB', 'R_PF_OFF_EA': 'OFF'}


def read(path):
    return yaml.load(Path(path).read_text(), Loader=getattr(yaml, 'CSafeLoader', yaml.SafeLoader))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def compact(source, destination, specs, metadata):
    """Stream only score-eligible patients/outcomes and required native CIF fields."""
    import pandas as pd
    import pyarrow as pa
    import pyarrow.parquet as pq
    populations = {}
    for spec in specs:
        available = metadata.cohort_fine.isin(spec['cohorts']) & metadata[spec['group_column']].notna()
        populations.setdefault(spec['outcome'], set()).update(metadata.loc[available, 'subject_id'].astype(str))
    source_file = pq.ParquetFile(source)
    columns = [c for c in source_file.schema_arrow.names
               if c in {'subject_id', 'outcome', 'time_days', 'event'}
               or c.startswith('cif_') and c.endswith('d')]
    if not {'subject_id', 'outcome', 'time_days', 'event'} <= set(columns):
        raise ValueError('Need native neural prediction fields; this is not a tabular exporter')
    writer = None
    rows = 0
    try:
        for batch in source_file.iter_batches(batch_size=8192, columns=columns):
            frame = batch.to_pandas()
            keep = pd.Series(False, index=frame.index)
            ids = frame.subject_id.astype(str)
            for outcome, patients in populations.items():
                keep |= frame.outcome.eq(outcome) & ids.isin(patients)
            if not keep.any():
                continue
            table = pa.Table.from_pandas(frame.loc[keep], preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(destination, table.schema)
            writer.write_table(table)
            rows += len(table)
    finally:
        if writer is not None:
            writer.close()
    if not rows:
        raise ValueError(f'No score-eligible predictions in {source}')
    return rows


def paired_auc(labels, risk, score, ipcw, multiplicities):
    import numpy as np
    from opera.evaluation.secondary_analysis import metric_values, bootstrap_metric_values
    model_auc = metric_values(labels, risk, ipcw=ipcw)['auc']
    index_auc = metric_values(labels, score, ipcw=ipcw)['auc']
    model_draws = bootstrap_metric_values(labels, risk, multiplicities, ipcw=ipcw)['auc']
    index_draws = bootstrap_metric_values(labels, score, multiplicities, ipcw=ipcw)['auc']
    differences = model_draws - index_draws
    finite = differences[np.isfinite(differences)]
    lower, upper = np.quantile(finite, [.025, .975]) if len(finite) else (float('nan'), float('nan'))
    model_interval = np.nanquantile(model_draws, [.025, .975])
    index_interval = np.nanquantile(index_draws, [.025, .975])
    return {'model_auc': model_auc, 'model_lower': float(model_interval[0]), 'model_upper': float(model_interval[1]),
            'index_auc': index_auc, 'index_lower': float(index_interval[0]), 'index_upper': float(index_interval[1]),
            'gain': model_auc - index_auc,
            'lower': float(lower), 'upper': float(upper), 'valid_draws': len(finite)}


def discrimination(analysis, specs, metadata, output):
    """Paired index AUCs using the existing CIF weights and bootstrap estimators."""
    import numpy as np
    import pandas as pd
    from opera.run.analyse_locked_predictions import read_artifact
    from opera.run.render_clinical_score_panels import score_patients, clinical_weights, followup_horizon
    from opera.evaluation.secondary_analysis import patient_bootstrap
    artifact, = [a for a in analysis['artifacts'] if a['split'] == 'held_out']
    predictions, _ = read_artifact(artifact, metadata, 'cif_ipcw')
    rows = []
    for original in specs:
        spec = dict(original)
        if spec.get('endpoint_role') == 'native':
            held, _ = score_patients(predictions, metadata,
                                     {**spec, 'horizon_days': max(spec['horizons_days'])}, 'held_out')
            spec['horizon_days'], _ = followup_horizon(held, spec)
        patients, _ = score_patients(predictions, metadata, spec, 'held_out')
        data = clinical_weights(patients, spec['horizon_days'])
        cases = int(((data.label == 1) & (data.ipcw_weight > 0)).sum())
        controls = int(((data.label == 0) & (data.ipcw_weight > 0)).sum())
        row = {'score_id': spec['id'], 'model': spec['model'], 'split': 'held_out',
               'outcome': spec['outcome'], 'horizon_days': spec['horizon_days'],
               'n_patients': len(data), 'n_cases': cases, 'n_controls': controls,
               'estimand': 'cif_ipcw', 'index_representation': 'numeric' if spec.get('score_column') else 'ordinal_categories',
               'bootstrap': 'cohort_stratified_patient_multiplier_1000_seed2026',
               'censoring_weights': 'estimated_once_on_score_available_population'}
        if min(cases, controls) < 5:
            rows.append({**row, 'status': 'insufficient_cases_controls'})
            continue
        draws = patient_bootstrap(data, 1000, 2026, method='multiplier')
        weights = np.empty((1000, len(data)))
        for cohort, part in data.groupby('cohort', sort=True):
            ids, counts = draws[('held_out', spec['seed'], cohort)]
            weights[:, part.index.to_numpy()] = counts[:, [ids[s] for s in part.subject_id]]
        # Any strictly increasing scaling preserves the index's AUC, including ties.
        score = data.ipi_score.to_numpy(float)
        span = np.ptp(score)
        score = (score - score.min()) / span if span else np.zeros(len(score))
        values = paired_auc(data.label.to_numpy(), data.model_risk.to_numpy(), score,
                            data.ipcw_weight.to_numpy(), weights)
        rows.append({**row, **values, 'status': 'estimated'})
    pd.DataFrame(rows).to_csv(output, index=False)


def prepare(args):
    root = args.national_root.resolve()
    refreshed = args.refreshed_root.resolve()
    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError('Use a fresh clinical workspace; existing outputs are retained')
    model = args.model
    bundle = root / 'adapted_fits'
    rows = read(bundle / 'bundle.json')['runs']
    row, = [r for r in rows if r['name'] == f'seed42/EA/{MODELS[model]}']
    config = bundle / row['config_path']
    if sha(config) != row['config_sha256']:
        raise ValueError('Frozen fit configuration changed')
    checkpoint = root / 'neural_runs' / row['run_directory'] / 'best.ckpt'
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    recipe = read(refreshed / 'clinical_scores.yaml')
    if recipe.get('all_indices') or not recipe.get('scores'):
        raise ValueError('Use the expanded refreshed clinical recipe, not the unexpanded template')
    for spec in recipe['scores']:
        spec['model'] = model
        if (spec.get('decision_curve') or {}).get('comparators'):
            raise ValueError('Register decision-curve comparators separately; this job loads one model')
    metadata_path = refreshed / 'patient_metadata.parquet'
    declarations = []
    for split in ('tuning', 'held_out'):
        analysis = read(root / f'extended_{"test" if split == "held_out" else "tuning"}_comparison/analysis.yaml')
        matches = [a for a in analysis['artifacts'] if a['model'] == model and a['split'] == split and int(a['seed']) == 42]
        if len(matches) != 1 or matches[0].get('format') != 'direct_cif':
            raise ValueError(f'Need one registered native artifact for {model}/{split}')
        artifact = matches[0]
        marker = Path(artifact['path']).parent / 'evaluation_complete.json'
        if not Path(artifact['path']).is_file() or not marker.is_file():
            raise ValueError(f'Completed prediction export missing: {artifact["path"]}')
        declarations.append(copy.deepcopy(artifact))
    if len({a['data_contract'] for a in declarations}) != 1:
        raise ValueError('Clinical tuning and held-out data contracts differ')
    output.mkdir(parents=True)
    training = args.training_predictions or output / 'training_INTERNAL/direct_cr_predictions.parquet'
    train = {**declarations[0], 'split': 'train', 'path': str(training)}
    all_artifacts = [*declarations, train]
    analysis = {'opera_model': model, 'patient_metadata': str(metadata_path),
                'patient_metadata_column_map': {}, 'artifacts': declarations,
                'clinical_threshold_artifacts': [train]}
    (output / 'clinical_scores.yaml').write_text(yaml.safe_dump(recipe, sort_keys=False))
    command = [sys.executable, '-u', '-m', 'opera.run.evaluate_direct_competing_risk',
               '--checkpoint', str(checkpoint), '--config-dir', str(config.parent),
               '--config-name', config.stem, '--split', 'train', '--batch-size', '1',
               '--num-workers', '0', '--output-dir', str(training.parent)]
    render = [sys.executable, '-u', '-m', 'opera.run.render_clinical_score_panels',
              '--analysis-config', str(output / 'analysis.yaml'), '--score-config',
              str(output / 'clinical_scores.yaml'), '--split', 'held_out',
              '--output-dir', str(output / 'clinical_results'), '--execute']
    plan = {'model': model, 'status': 'prepared', 'training_command': command,
            'render_command': render, 'metadata': str(metadata_path),
            'sources': {str(p): sha(p) for p in [config, refreshed / 'clinical_scores.yaml',
                                                refreshed / 'refresh_report.json']},
            'scope': 'Existing clinical risk-group analyses plus paired model-index AUC with shared patient multiplier draws.',
            'xp': 'Separate tabular export/refit path needed for training-derived thresholds; no XP refit is launched.'}
    plan_path = output / 'launch_manifest.json'
    plan_path.write_text(json.dumps(plan, indent=2))
    (output / 'analysis.yaml').write_text(yaml.safe_dump(analysis, sort_keys=False))
    print(json.dumps(plan, indent=2), flush=True)
    if not args.execute:
        return
    # Confirm the registered checkpoint is still the completed selected fit.
    from bonsai.functional.checkpointing import verified_training_selection
    selection = verified_training_selection(checkpoint.parent)
    for artifact in declarations:
        marker = read(Path(artifact['path']).parent / 'evaluation_complete.json')
        if (marker.get('inputs', {}).get('checkpoint_sha256') != selection['best_checkpoint_sha256']
                or marker.get('inputs', {}).get('config_sha256') != row['config_sha256']
                or marker.get('prediction_sha256') != sha(artifact['path'])):
            raise ValueError('Clinical prediction export does not match selected checkpoint')
    plan['status'] = 'running'; plan_path.write_text(json.dumps(plan, indent=2))
    try:
        if args.training_predictions is None:
            subprocess.run(command, check=True)
            receipt = {'checkpoint_sha256': selection['best_checkpoint_sha256'],
                       'config_sha256': row['config_sha256'], 'split': 'train',
                       'prediction_sha256': sha(training)}
            (training.parent / 'training_export_receipt.json').write_text(json.dumps(receipt, indent=2))
        else:
            receipt = read(training.parent / 'training_export_receipt.json')
            if (receipt.get('checkpoint_sha256') != selection['best_checkpoint_sha256']
                    or receipt.get('config_sha256') != row['config_sha256']
                    or receipt.get('split') != 'train' or receipt.get('prediction_sha256') != sha(training)):
                raise ValueError('Reused training export must have a matching training_export_receipt.json')
        from opera.run.analyse_locked_predictions import read_table
        metadata = read_table(metadata_path)
        metadata['subject_id'] = metadata.subject_id.astype(str)
        private = output / 'clinical_predictions_INTERNAL'; private.mkdir()
        plan['compacted_rows'] = {}
        for artifact in all_artifacts:
            destination = private / f'{artifact["split"]}.parquet'
            plan['compacted_rows'][artifact['split']] = compact(artifact['path'], destination, recipe['scores'], metadata)
            artifact['path'] = str(destination)
        (output / 'analysis.yaml').write_text(yaml.safe_dump(analysis, sort_keys=False))
        discrimination(analysis, recipe['scores'], metadata, output / 'clinical_discrimination.csv')
        environment = {**os.environ, 'OPERA_ANALYSIS_WORKERS': '1', 'OPERA_SUMMARY_WORKERS': '1',
                       'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1', 'MPLBACKEND': 'Agg',
                       'OPERA_PAPER_FIGURE_MARK': '', 'OPERA_PAPER_EXPORT_PLOT_STATE': '1',
                       'CUDA_VISIBLE_DEVICES': ''}
        rendered = subprocess.run(render, check=False, env=environment)
        plan['renderer_returncode'] = rendered.returncode
        if rendered.returncode and not (output / 'clinical_results/report_manifest.json').is_file():
            raise subprocess.CalledProcessError(rendered.returncode, render)
        from opera.run.refresh_clinical_figures import package_clinical
        package_clinical(output / 'clinical_results', output / f'{model}_clinical_figures.zip')
        from zipfile import ZipFile
        with ZipFile(output / f'{model}_clinical_figures.zip', 'a') as archive:
            archive.write(output / 'clinical_discrimination.csv', 'clinical_discrimination.csv')
        plan['status'] = 'complete' if rendered.returncode == 0 else 'complete_with_gaps'
    except BaseException as error:
        plan.update(status='failed', error=str(error))
        raise
    finally:
        plan_path.write_text(json.dumps(plan, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', choices=MODELS, required=True)
    parser.add_argument('--national-root', type=Path, default=Path('data/paper_256_v3_national_20261111'))
    parser.add_argument('--refreshed-root', type=Path, default=Path('data/clinical_registry_refresh/20261111/prepared_clinical'))
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--training-predictions', type=Path, help='Reuse an existing export from this exact fit; never a tuning/test export')
    parser.add_argument('--execute', action='store_true')
    prepare(parser.parse_args())
