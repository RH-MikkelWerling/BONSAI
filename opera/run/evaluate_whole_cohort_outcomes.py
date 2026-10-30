"""Whole-cohort outcome evaluation from fixed predictions; CPU only.

Keep disease-conditional censoring weights and outcomes/horizons separate.
Report each model's available population and an explicit common model roster.
This neither trains models nor changes the disease-specific primary analysis.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from opera.evaluation.secondary_analysis import (
    PATIENT, METRICS, _interval, bootstrap_metric_values, calibration_summary,
    metric_values, normalize_predictions, paired_analysis, patient_bootstrap,
)
from opera.run.analyse_locked_predictions import file_hash, read_artifact, read_table


TASK = ['split', 'seed', 'outcome', 'horizon_days']


def _auc_pair_mass(part):
    """Weighted concordant and total case/control pair mass, ties half credited."""
    cases = part.loc[part.label.eq(1) & part.ipcw_weight.gt(0)]
    controls = part.loc[part.label.eq(0) & part.ipcw_weight.gt(0)].sort_values('probability')
    total = float(cases.ipcw_weight.sum() * controls.ipcw_weight.sum())
    if not total:
        return 0., 0.
    risk = controls.probability.to_numpy(float)
    cumulative = np.r_[0., controls.ipcw_weight.to_numpy(float).cumsum()]
    left = np.searchsorted(risk, cases.probability.to_numpy(float), side='left')
    right = np.searchsorted(risk, cases.probability.to_numpy(float), side='right')
    concordant = np.dot(cases.ipcw_weight, (cumulative[left] + cumulative[right]) / 2)
    return float(concordant), total


def population_diagnostics(frame, population):
    """Explain pooled AUC without changing predictions, weights or the estimand.

    Pooled AUC ranks every case against every control, including patients from
    different diseases. This exact point-estimate decomposition is descriptive,
    not an additional hypothesis test or a replacement for the paired analysis.
    """
    diseases, decomposition = [], []
    for key, part in frame.groupby(['model', *TASK], sort=True):
        identity = dict(population=population, **dict(zip(['model', *TASK], key)))
        within_concordant = within_mass = 0.
        for disease, cell in part.groupby('cohort', sort=True):
            known = cell.loc[cell.ipcw_weight.gt(0)]
            cases = known.label.eq(1)
            concordant, mass = _auc_pair_mass(cell)
            within_concordant += concordant
            within_mass += mass
            diseases.append(dict(identity, cohort=disease, n=len(cell),
                n_cases=int(cases.sum()), n_controls=int((~cases).sum()),
                meets_fine_cell_support=bool(cases.sum() >= 5 and (~cases).sum() >= 5),
                n_zero_weight=int(cell.ipcw_weight.eq(0).sum()),
                n_placeholder_predictions=int(cell.get('prediction_absent_zero_weight',
                    pd.Series(False, index=cell.index)).eq(True).sum()),
                n_unique_known_predictions=known.probability.nunique(),
                known_probability_min=known.probability.min(),
                known_probability_mean=known.probability.mean(),
                known_probability_max=known.probability.max(),
                case_weight=float(known.loc[cases, 'ipcw_weight'].sum()),
                control_weight=float(known.loc[~cases, 'ipcw_weight'].sum()),
                within_disease_auc=concordant / mass if mass else np.nan))
        total_concordant, total_mass = _auc_pair_mass(part)
        between_mass = max(0., total_mass - within_mass)
        between_concordant = total_concordant - within_concordant
        decomposition.append(dict(identity, n=len(part), n_diseases=part.cohort.nunique(),
            pooled_auc=total_concordant / total_mass if total_mass else np.nan,
            within_disease_pair_fraction=within_mass / total_mass if total_mass else np.nan,
            between_disease_pair_fraction=between_mass / total_mass if total_mass else np.nan,
            within_disease_pair_auc=within_concordant / within_mass if within_mass else np.nan,
            between_disease_pair_auc=between_concordant / between_mass if between_mass else np.nan,
            weighted_case_control_pair_mass=total_mass))
    return pd.DataFrame(diseases), pd.DataFrame(decomposition)


def evaluate(frame, models, *, replicates=1000, seed=2026):
    """Pool patients after estimating censoring within their fine diseases."""
    if len(models) < 2 or len(set(models)) != len(models):
        raise ValueError('Supply at least two distinct fixed model identities')
    frame = frame.loc[frame.model.isin(models)].copy()
    if set(frame.model) != set(models):
        raise ValueError('A requested whole-cohort model has no registered predictions')
    frame, _ = normalize_predictions(frame, 'cif_ipcw')
    # Normalization also rejects conflicting fine disease identities before
    # any intersection can conceal a mislabelled patient.
    available = frame.copy()
    available['cohort'] = 'All lymphoid cancers'
    draws = patient_bootstrap(available, replicates, seed, method='multiplier')
    absolute, coverage = [], []
    for key, part in available.groupby(['model', 'split', 'seed', 'outcome', 'horizon_days'], sort=True):
        model, split, training_seed, outcome, horizon = key
        part = part.sort_values('subject_id')
        y, p, w = part.label.to_numpy(), part.probability.to_numpy(), part.ipcw_weight.to_numpy()
        mapping, counts = draws[(split, training_seed, 'All lymphoid cancers')]
        bootstrap = counts[:, [mapping[sid] for sid in part.subject_id]]
        n_cases = int(((w > 0) & (y == 1)).sum())
        n_controls = int(((w > 0) & (y == 0)).sum())
        eligible = n_cases >= 5 and n_controls >= 5
        point = metric_values(y, p, ipcw=w) if eligible else dict.fromkeys(METRICS, np.nan)
        sampled = bootstrap_metric_values(y, p, bootstrap, ipcw=w) if eligible else {
            metric: np.full(replicates, np.nan) for metric in METRICS}
        identity = dict(model=model, split=split, seed=training_seed, outcome=outcome,
                        horizon_days=horizon, n=len(part), n_cases=n_cases, n_controls=n_controls)
        for metric in METRICS:
            absolute.append(dict(identity, metric=metric, estimate=point[metric],
                                 **_interval(sampled[metric])))
    present = frame.groupby(PATIENT, sort=False).model.nunique()
    common_keys = present.loc[present == len(models)].index
    common = frame.loc[pd.MultiIndex.from_frame(frame[PATIENT]).isin(common_keys)].copy()
    if common.empty:
        raise ValueError('No common patient/outcome/horizon roster across the requested models')
    # Include absent model/task combinations too. Otherwise an outcome with no
    # shared roster would disappear from the old coverage report altogether.
    common_groups = {key: part for key, part in common.groupby(TASK, sort=True)}
    for key, task in frame.groupby(TASK, sort=True):
        shared = common_groups.get(key, task.iloc[:0])
        union_n = task.subject_id.nunique()
        for model in models:
            part, kept = task.loc[task.model.eq(model)], shared.loc[shared.model.eq(model)]
            coverage.append(dict(model=model, **dict(zip(TASK, key)),
                available_n=len(part), common_n=len(kept), union_n=union_n,
                available_fraction_of_union=len(part) / union_n,
                common_fraction_of_union=len(kept) / union_n,
                excluded_for_model_parity_n=len(part)-len(kept),
                available_diseases=part.cohort.nunique(), common_diseases=kept.cohort.nunique(),
                missing_diseases='|'.join(sorted(set(task.cohort) - set(part.cohort))),
                status='missing_model_task' if part.empty else 'no_common_population' if kept.empty
                       else 'complete_population' if len(kept) == union_n else 'partial_population'))
    # Re-estimate censoring on the shared evaluation roster, retaining fine
    # disease strata. Collapsing cohort labels earlier would change this estimand.
    common, _ = normalize_predictions(common, 'cif_ipcw')
    disease_reports, pair_reports = [], []
    for population, roster in [('available', frame), ('common', common)]:
        disease_report, pair_report = population_diagnostics(roster, population)
        disease_reports.append(disease_report)
        pair_reports.append(pair_report)
    common['cohort'] = 'All lymphoid cancers'
    contrasts = {f'{models[0]}_vs_{reference}': f'gain({models[0]},{reference})'
                 for reference in models[1:]}
    shared_draws = patient_bootstrap(common, replicates, seed, method='multiplier')
    paired = paired_analysis(common, contrasts, draws=shared_draws, replicates=replicates,
                             patient_gain_contrasts=[], min_cases=5, min_controls=5)
    return dict(available_population_estimates=pd.DataFrame(absolute),
                common_population_estimates=paired['estimates'],
                paired_contrasts=paired['contrasts'], model_coverage=pd.DataFrame(coverage),
                contrast_coverage=paired['contrast_coverage'],
                common_population_calibration=calibration_summary(common),
                population_by_disease=pd.concat(disease_reports, ignore_index=True),
                auc_pair_composition=pd.concat(pair_reports, ignore_index=True))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--analysis-config', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--models', nargs='+', default=['OPERA', 'J_L', 'X', 'XP'])
    parser.add_argument('--split', choices=['tuning', 'held_out'], default='tuning')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--horizons', type=int, nargs='+', default=[365])
    parser.add_argument('--replicates', type=int, default=1000)
    args = parser.parse_args()
    if args.replicates < 20:
        parser.error('Use at least 20 paired bootstrap draws')
    analysis = yaml.safe_load(args.analysis_config.read_text())
    if analysis.get('estimand') != 'cif_ipcw':
        raise ValueError('Whole-cohort analysis requires a locked cif_ipcw inventory')
    models = [analysis.get('opera_model', 'R_L') if name == 'OPERA' else name for name in args.models]
    metadata = read_table(analysis['patient_metadata']).rename(
        columns=analysis.get('patient_metadata_column_map', {}))
    metadata.subject_id = metadata.subject_id.astype(str)
    artifacts = [a for a in analysis['artifacts'] if a['model'] in models
                 and a.get('split') == args.split and int(a.get('seed', 42)) == args.seed]
    frames = [read_artifact(a, metadata, 'cif_ipcw')[0] for a in artifacts]
    if not frames:
        raise ValueError('No registered predictions for the requested models/split/seed')
    frame = pd.concat(frames, ignore_index=True)
    frame = frame.loc[frame.horizon_days.isin(args.horizons)]
    for model in models:
        missing = set(args.horizons) - set(frame.loc[frame.model.eq(model), 'horizon_days'])
        if missing:
            raise ValueError(f'{model} has no predictions at requested horizons {sorted(missing)}')
    result = evaluate(frame, models, replicates=args.replicates)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    for name, table in result.items():
        table.to_csv(args.output_dir / f'{name}.csv', index=False)
    (args.output_dir / 'manifest.json').write_text(json.dumps(dict(
        split=args.split, models=models, seed=args.seed, horizons=args.horizons,
        replicates=args.replicates, bootstrap='paired patient multiplier', bootstrap_seed=2026,
        estimand='cif_ipcw with fine-disease-conditional censoring weights',
        model_roles={'X': 'Portfolio of separately fitted disease/outcome/horizon XGBoost models',
                     'XP': 'Disease-pooled XGBoost model per outcome/horizon'},
        uncertainty='Paired patient multiplier draws conditional on fitted models and estimated IPCW',
        config_sha256=file_hash(args.analysis_config),
        metadata_sha256=file_hash(analysis['patient_metadata']),
        evaluator_sha256=file_hash(__file__),
        input_sha256={a['path']: file_hash(a['path']) for a in artifacts},
        interpretation='Pooled discrimination includes between-disease risk differences. '
                       'Common-model estimates use the explicitly reported shared roster. '
                       'X is a portfolio, not a model fitted on the whole cohort. '
                       'No per-disease performance-support filter is applied before pooling. '
                       'AUC pair composition is a point-estimate identity, not causal attribution.'), indent=2))
    print(f'Whole-cohort estimates and coverage: {args.output_dir}')


if __name__ == '__main__':
    main()
