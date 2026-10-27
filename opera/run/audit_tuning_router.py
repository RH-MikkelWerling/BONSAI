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
                checkpoint = root / 'neural_runs' / row['run_directory'] / 'best.ckpt'
                if verify:
                    item['checkpoint_hash_matches'] = digest(checkpoint) == signature['checkpoint_sha256']
            if verify:
                item['prediction_hash_matches'] = digest(predictions) == saved['prediction_sha256']
        report['artifacts'].append(item)
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
                                    'decision_supported_cells': len(supported)})
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
    args = parser.parse_args()
    env = dotenv_values(args.env_file)
    root = Path(env['OPERA_PAPER_ROOT']).resolve()
    analysis = args.analysis_root or Path(env['OPERA_PAPER_ANALYSIS_ROOT'])
    report = audit(root, analysis.resolve(), verify=args.verify_artifacts,
                   route_diagnostics=args.route_diagnostics)
    report["active_env_file"] = str(args.env_file.resolve())
    report["configured_routed_model"] = env.get("OPERA_PAPER_ROUTED_MODEL")
    report["configured_opera_model"] = env.get("OPERA_PAPER_OPERA_MODEL")
    print(json.dumps(report, indent=2, default=str))


if __name__ == '__main__':
    main()
