"""Read-only router/figure provenance audit; prints aggregates, never patient IDs.

python -m opera.run.audit_tuning_router --env-file .env.rehearsal
Use the exact env file used by the latest figure run. --verify-artifacts also
hashes potentially large prediction/checkpoint files. No inference is run.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd
import yaml
from dotenv import dotenv_values


def digest(path):
    result = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            result.update(chunk)
    return result.hexdigest()


def reconcile_metrics(config, analysis):
    """Recompute cell metrics from the same CIFs; compare both implementations
    and saved estimates without fitting or bootstrapping anything."""
    import numpy as np

    from opera.evaluation.metrics import compute_competing_risk_metrics_at_horizon
    from opera.evaluation.secondary_analysis import metric_values, normalize_predictions
    from opera.run.analyse_locked_predictions import read_artifact, read_table

    metadata = read_table(config['patient_metadata'])
    metadata['subject_id'] = metadata.subject_id.astype(str)
    saved_path = analysis / 'prediction_analysis' / 'estimates.csv'
    saved = pd.read_csv(saved_path) if saved_path.is_file() else None
    wanted = {'J_L', 'R_L', 'R_AB_L', 'R_ORIG_L', config.get('opera_model')}
    reports = []
    for spec in config['artifacts']:
        if spec['model'] not in wanted or spec.get('split') != 'tuning':
            continue
        if spec.get('format') != 'direct_cif':
            continue
        frame, _ = read_artifact(spec, metadata, config['estimand'])
        if config.get('analysis_horizons_days'):
            frame = frame.loc[frame.horizon_days.isin(config['analysis_horizons_days'])]
        keys = ['split', 'seed', 'cohort', 'outcome', 'horizon_days']
        metrics, errors, status_mismatches = [], [], 0
        for identity, cell in frame.groupby(keys):
            ipcw = cell.ipcw_weight.to_numpy() if 'ipcw_weight' in cell else None
            values = metric_values(cell.label.to_numpy(), cell.probability.to_numpy(), ipcw=ipcw)
            # CIF IPCW and known-status are distinct estimands; compare the two
            # IPCW implementations only under the former.
            if config['estimand'] == 'cif_ipcw':
                native = compute_competing_risk_metrics_at_horizon(
                    cell.time_days.to_numpy(), cell.event.to_numpy(),
                    cell.probability.to_numpy(), identity[-1])
                for metric in ('auc', 'brier'):
                    status_mismatches += int(np.isfinite(values[metric]) != np.isfinite(native['cif_' + metric]))
                    if np.isfinite([values[metric], native['cif_' + metric]]).all():
                        errors.append(abs(values[metric] - native['cif_' + metric]))
            for metric in ('auc', 'brier'):
                metrics.append({**dict(zip(keys, identity)), 'model': spec['model'],
                                'metric': metric, 'recomputed': values[metric]})
        recomputed = pd.DataFrame(metrics)
        report = {'model': spec['model'], 'path': spec['path'],
                  'estimand': config['estimand'], 'recomputed_metrics': len(metrics),
                  'native_vs_analysis_max_abs_error': max(errors) if errors else None,
                  'native_vs_analysis_finite_status_mismatches': status_mismatches}
        if saved is not None and len(recomputed):
            paired = recomputed.merge(saved, on=[*keys, 'model', 'metric'],
                                      how='left', validate='one_to_one', indicator=True)
            present = paired['_merge'].eq('both')
            report['missing_saved_estimates'] = int((~present).sum())
            report['finite_status_mismatches'] = int((present & (
                np.isfinite(paired.recomputed) != np.isfinite(paired.estimate))).sum())
            finite = paired.loc[present & np.isfinite(paired.recomputed) & np.isfinite(paired.estimate)]
            report['recomputed_vs_saved_max_abs_error'] = (
                float((finite.recomputed - finite.estimate).abs().max()) if len(finite) else None)
        pooled_recomputed = None
        if config['estimand'] == 'cif_ipcw':
            pooled_frame, _ = normalize_predictions(frame.assign(cohort='ALL'), 'cif_ipcw')
            pooled_rows = []
            for (outcome, horizon), cell in pooled_frame.groupby(['outcome', 'horizon_days']):
                values = metric_values(cell.label, cell.probability, ipcw=cell.ipcw_weight)
                pooled_rows.append({'outcome': outcome, 'horizon_days': horizon,
                                    'recomputed_auc': values['auc']})
            pooled_recomputed = pd.DataFrame(pooled_rows)
            report['native_patient_pooled_recomputed_auc_mean'] = (
                float(pooled_recomputed.recomputed_auc.mean()) if len(pooled_recomputed) else None)
        native_pooled = Path(spec['path']).parent / 'direct_cr_pooled_metrics.csv'
        if native_pooled.is_file():
            pooled = pd.read_csv(native_pooled)
            pooled = pooled.loc[pooled.horizon_days.isin(frame.horizon_days.unique())]
            report['native_pooled_horizons'] = sorted(map(int, pooled.horizon_days.unique()))
            report['native_patient_pooled_auc_mean'] = float(pooled.cif_auc.mean())
            if pooled_recomputed is not None and len(pooled_recomputed):
                checked = pooled_recomputed.merge(pooled, on=['outcome', 'horizon_days'],
                                                    how='left', validate='one_to_one')
                finite = checked.loc[np.isfinite(checked.recomputed_auc) & np.isfinite(checked.cif_auc)]
                report['native_pooled_saved_max_abs_error'] = (
                    float((finite.recomputed_auc - finite.cif_auc).abs().max()) if len(finite) else None)
            report['native_patient_pooled_description'] = (
                'patients pooled across diseases per outcome, then finite outcome/horizon AUCs averaged; '
                'not the fine-disease cell estimator')
        reports.append(report)
    return reports


def gain_distribution(cells):
    """Expose exact ranking ties and sparse-cell composition without rounding gains."""
    import numpy as np
    result = []
    for label, subset in [
        ('all_finite', cells),
        ('supported_min5', cells.loc[(cells.n_cases >= 5) & (cells.n_controls >= 5)]),
        ('sparse_below5', cells.loc[(cells.n_cases < 5) | (cells.n_controls < 5)]),
    ]:
        gain = subset.gain.to_numpy(float)
        gain = gain[np.isfinite(gain)]
        result.append({'population': label, 'finite_cells': len(gain),
                       'improved': int((gain > 0).sum()), 'worsened': int((gain < 0).sum()),
                       'exactly_zero': int((gain == 0).sum()),
                       'nonzero_within_1e_minus12': int(((gain != 0) & (np.abs(gain) <= 1e-12)).sum()),
                       'median': float(np.median(gain)) if len(gain) else None,
                       'mean': float(np.mean(gain)) if len(gain) else None,
                       'minimum_cases': int(subset.n_cases.min()) if len(subset) else None,
                       'minimum_controls': int(subset.n_controls.min()) if len(subset) else None})
    return result


def audit(root, analysis, *, verify=False, route_diagnostics=False):
    config_path = analysis / 'analysis_inventory' / 'analysis.yaml'
    config = yaml.safe_load(config_path.read_text())
    report = {'analysis_config': str(config_path),
              'opera_model': config.get('opera_model'),
              'route_model': config.get('route_model'),
              'route_weights': config.get('route_weights'),
              'artifacts': [], 'auc_panels': [],
              'contrast_definitions': {'R1': 'R_L - J_L',
                                       'router_borrow_AB': 'R_AB_L - RBH_AB_L',
                                       'router_both_AB': 'R_AB_L - N_A_L'}}
    configs = {}
    for path in sorted((root / 'bundles').glob('stage*/bundle.json')):
        for row in json.loads(path.read_text())['runs']:
            configs.setdefault(row['config_sha256'], (row, path.parent / row['config_path']))
    wanted = {'R_L', 'R_ORIG_L', 'R_AB_L', 'RBH_AB_L', 'N_A_L', 'J_L',
              'R_AB_IPW_L', 'J_IPW_L'}
    for spec in config.get('artifacts', []):
        if spec['model'] not in wanted:
            continue
        predictions = Path(spec['path'])
        marker = predictions.parent / 'evaluation_complete.json'
        item = {'model': spec['model'], 'predictions': str(predictions),
                'split': spec.get('split'), 'completion_marker': marker.is_file()}
        if marker.is_file():
            saved = json.loads(marker.read_text())
            signature = saved['inputs']
            item['evaluation_inputs'] = signature
            if signature.get('config_sha256') in configs:
                row, path = configs[signature['config_sha256']]
                resolved = yaml.safe_load(path.read_text())
                item.update(fit_name=row['name'], config_path=str(path),
                            bundle_config_hash_matches=digest(path) == row['config_sha256'],
                            pooling=resolved.get('model', {}).get('pooling'),
                            readout={key: value for key, value in resolved.get('competing_risk', {}).items()
                                     if key in {'readout_mode', 'adapter_source_mode',
                                                'adapter_global', 'adapter_global_adapter',
                                                'adapter_router_warmup_epochs',
                                                'adapter_logit_scale', 'adapter_own_outside',
                                                'adapter_bottleneck', 'adapter_query_mode',
                                                'adapter_query_hidden_dim', 'adapter_home_floor',
                                                'adapter_shared_logit_init', 'adapter_global_correction'}})
                run = root / 'neural_runs' / row['run_directory']
                training_marker = run / 'training_complete.json'
                item['training_completion_marker'] = training_marker.is_file()
                if training_marker.is_file():
                    selection = json.loads(training_marker.read_text()).get('selection', {})
                    item['training_selection'] = selection
                    item['evaluation_matches_selected_checkpoint'] = (
                        selection.get('best_checkpoint_sha256') == signature.get('checkpoint_sha256'))
                checkpoint = run / 'best.ckpt'
                item['checkpoint_path'] = str(checkpoint)
                if verify:
                    item['checkpoint_hash_matches'] = digest(checkpoint) == signature['checkpoint_sha256']
            if verify:
                item['prediction_hash_matches'] = digest(predictions) == saved['prediction_sha256']
        report['artifacts'].append(item)
    borrowing_table = analysis / 'source_removal' / 'tuning' / 'borrowing_map.csv'
    if borrowing_table.is_file():
        pairs = pd.read_csv(borrowing_table)
        pairs = pairs.loc[(pairs.target != pairs.source) & (pairs.endpoint_group == 'All outcomes')]
        report['borrowing_scale_diagnostics'] = {
            'units': 'AUC points, already multiplied by 100 upstream',
            'off_diagonal_pairs': len(pairs),
            'median_abs_shrunken_effect': float(pairs.effect.abs().median()),
            'maximum_abs_shrunken_effect': float(pairs.effect.abs().max()),
            **({'median_abs_raw_effect': float(pairs.raw_effect.abs().median()),
                'maximum_abs_raw_effect': float(pairs.raw_effect.abs().max())}
               if 'raw_effect' in pairs else {}),
            'note': 'Pairwise heatmaps use shrunken effects; total dependence panels are unshrunk.'}
    if config.get('route_weights'):
        directory = Path(config['route_weights']).parent
        report['route_predictions_directory'] = str(directory)
        report['route_directory_matches_primary'] = any(
            a['model'] == 'R_L' and Path(a['predictions']).parent == directory
            for a in report['artifacts'])
    effects = analysis / 'prediction_analysis' / 'effect_summaries.csv'
    if effects.is_file():
        frame = pd.read_csv(effects)
        rows = frame.loc[(frame.split == 'tuning') & (frame.metric == 'auc')
                         & (frame.stratum == 'all_cells')
                         & frame.contrast.isin(['R1', 'R1_orig', 'R_orig', 'C6',
                                               'IPW_R', 'IPW_J', 'router_borrow_AB',
                                               'router_own_AB', 'router_both_AB'])].copy()
        report['auc_panels'] = rows.to_dict('records')
        report['auc_unit'] = 'raw AUC; multiply gains/intervals by 100 for AUC points'
    contrasts_path = analysis / 'prediction_analysis' / 'contrasts.csv'
    if contrasts_path.is_file():
        contrasts = pd.read_csv(contrasts_path)
        selected = contrasts.loc[(contrasts.split == 'tuning') & (contrasts.metric == 'auc')
                                 & contrasts.contrast.isin(['R1', 'R1_orig', 'router_borrow_AB',
                                                           'router_both_AB'])]
        comparisons = []
        for (contrast, seed), cells in selected.groupby(['contrast', 'seed']):
            for horizon in [0, *sorted(cells.horizon_days.unique())]:
                part = cells if horizon == 0 else cells.loc[cells.horizon_days == horizon]
                finite = part.loc[part.gain.notna()]
                supported = finite.loc[(finite.n_cases >= 5) & (finite.n_controls >= 5)]
                weighted = None
                if len(supported):
                    weights = 1 / (supported.cohort.nunique()
                                   * supported.groupby('cohort').outcome.transform('nunique')
                                   * supported.groupby(['cohort', 'outcome']).horizon_days.transform('nunique'))
                    weighted = float((supported.gain * weights).sum())
                comparisons.append({'contrast': contrast, 'seed': int(seed),
                                    'horizon_days': int(horizon), 'finite_cells': len(finite),
                                    'cell_median': float(finite.gain.median()),
                                    'equal_cell_mean': float(finite.gain.mean()),
                                    'decision_weighted_mean_min5': weighted,
                                    'decision_supported_cells': len(supported),
                                    'gain_distribution': gain_distribution(finite)})
        report['point_summary_comparison_raw_auc'] = comparisons
    if route_diagnostics:
        from opera.run.router_decision import routing_diagnostics, routing_health
        diagnostics = {}
        for item in report['artifacts']:
            if item['model'] not in {'R_L', 'R_ORIG_L', 'R_AB_L'}:
                continue
            directory = Path(item['predictions']).parent
            key = str(directory)
            if key in diagnostics:
                continue
            path = directory / 'source_route_weights.parquet'
            if not path.is_file():
                diagnostics[key] = {'status': 'missing_route_export'}
                continue
            try:
                result = routing_diagnostics(path)
                diagnostics[key] = {k: result[k] for k in ('rows', 'overall', 'top_sources')}
                support = directory / 'source_route_support.json'
                if support.is_file():
                    health = routing_health(path, support)
                    diagnostics[key]['support_weighted_health'] = {
                        k: health[k] for k in ('healthy', 'criteria')}
            except (ValueError, KeyError, IndexError) as error:
                diagnostics[key] = {'status': 'diagnostic_failed', 'error': str(error)}
        report['routing_diagnostics'] = diagnostics
    removal_manifest = analysis / 'source_removal' / 'tuning' / 'source_removal.json'
    if removal_manifest.is_file():
        report['borrowing_figure_provenance'] = json.loads(removal_manifest.read_text())
    plot = analysis / 'figure_inputs' / 'plot_tables' 
    coords, outcomes = plot / 'gallery_coordinates.csv', plot / 'embedding_gallery.parquet'
    if coords.is_file() and outcomes.is_file():
        # Mirror the production renderer's CSV and Parquet readers.
        left = pd.read_csv(coords, dtype={'subject_id': str}).subject_id
        right = pd.read_parquet(outcomes, columns=['subject_id']).subject_id
        report['embedding_join'] = {'csv_id_dtype': str(left.dtype),
                                   'parquet_id_dtype': str(right.dtype),
                                   'matched_rows_before_normalization': int(right.isin(left).sum()),
                                   'matched_rows_after_normalization': int(right.astype(str).isin(left).sum())}
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--env-file', type=Path, default=Path('.env.rehearsal'))
    parser.add_argument('--analysis-root', type=Path)
    parser.add_argument('--verify-artifacts', action='store_true')
    parser.add_argument('--route-diagnostics', action='store_true')
    parser.add_argument('--recompute-metrics', action='store_true',
                        help='CPU: recompute native and paper cell AUC/Brier from saved tuning CIFs')
    args = parser.parse_args()
    env = dotenv_values(args.env_file)
    root = Path(env['OPERA_PAPER_ROOT']).resolve()
    analysis = args.analysis_root or Path(env['OPERA_PAPER_ANALYSIS_ROOT'])
    report = audit(root, analysis.resolve(), verify=args.verify_artifacts,
                   route_diagnostics=args.route_diagnostics)
    if args.recompute_metrics:
        config = yaml.safe_load((analysis / 'analysis_inventory' / 'analysis.yaml').read_text())
        report['metric_reconciliation'] = reconcile_metrics(config, analysis)
    report["active_env_file"] = str(args.env_file.resolve())
    report["configured_routed_model"] = env.get("OPERA_PAPER_ROUTED_MODEL")
    report["configured_opera_model"] = env.get("OPERA_PAPER_OPERA_MODEL")
    alignment = {}
    clinical_path = env.get('OPERA_PAPER_CLINICAL_SCORE_CONFIG')
    if clinical_path and Path(clinical_path).is_file():
        clinical = yaml.safe_load(Path(clinical_path).read_text())
        requested = {s.get('model') for s in clinical.get('scores', []) if s.get('model')}
        template_model = clinical.get('all_indices', {}).get('template', {}).get('model')
        if template_model:
            requested.add(template_model)
        alignment['clinical_requested_models'] = sorted(requested)
        alignment['clinical_config'] = clinical_path
        alignment['clinical_requests_selected_opera'] = report.get('opera_model') in requested
    atlas = env.get('OPERA_PAPER_ATLAS_METRICS')
    if atlas:
        primary = [a for a in report['artifacts'] if a['model'] == report.get('opera_model')
                   and a.get('split') == 'tuning']
        alignment['atlas_metrics'] = atlas
        alignment['atlas_matches_selected_opera_directory'] = any(
            Path(a['predictions']).parent.resolve() == Path(atlas).parent.resolve() for a in primary)
    report['figure_model_alignment'] = alignment
    print(json.dumps(report, indent=2, default=str))


if __name__ == '__main__':
    main()
