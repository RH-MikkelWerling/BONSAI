"""Exploratory fixed-budget risk stratification from an immutable prepared snapshot.

No training, threshold optimisation, model selection or main-analysis mutations.
IPCW weights and paired patient draws come from the existing analysis. Fractional
boundary ties define an expected random allocation within a tied score group.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
from zipfile import ZipFile, ZIP_DEFLATED

for _name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_name, '1')

import numpy as np
import pandas as pd

from opera.evaluation.distributed_analysis import load_pickle, save_pickle
from opera.evaluation.secondary_analysis import CELL

MODELS = ['R_EA', 'J_EA', 'J_OA_OWN_L', 'XP', 'X', 'XR']
METRICS = ('event_risk', 'relative_risk', 'event_capture')


def divide(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    return np.divide(a, b, out=np.full(np.broadcast_shapes(a.shape, b.shape), np.nan), where=b > 0)


def interval(values, minimum_valid=.8):
    values = np.asarray(values, float)
    finite = values[np.isfinite(values)]
    if not len(values) or len(finite) / len(values) < minimum_valid:
        return np.nan, np.nan, len(finite), 'insufficient_draws'
    low, high = np.quantile(finite, [.025, .975])
    return float(low), float(high), len(finite), 'estimated'


def tail_statistics(probability, labels, ipcw, counts, *, share=.1):
    """Statistics for each row of bootstrap multiplicities (ones for point estimates).

    Re-selects the quantile in every replicate using all risk-set patients,
    including those censored early. IPCW-normalised (Hajek) group incidence is
    bounded in [0,1]; event capture uses the weighted event totals. Predictions
    themselves are not recalibrated. The censoring model stays fixed.
    """
    p, y, w, c = (np.asarray(v, float) for v in (probability, labels, ipcw, counts))
    if c.ndim == 1:
        c = c[None, :]
    if p.ndim != 1 or not len(p) or y.shape != p.shape or w.shape != p.shape or c.shape[1] != len(p):
        raise ValueError('Unaligned risk-set arrays')
    if not 0 < share <= .5 or not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError('Invalid share or probabilities')
    if not np.isin(y, [0, 1]).all() or not np.isfinite(w).all() or (w < 0).any():
        raise ValueError('Invalid labels/IPCW')
    if not np.isfinite(c).all() or (c < 0).any():
        raise ValueError('Invalid patient multiplicities')
    order = np.argsort(p, kind='stable')
    starts = np.r_[0, np.flatnonzero(np.diff(p[order])) + 1]
    def grouped(values):
        return np.add.reduceat(c[:, order] * np.asarray(values)[order], starts, axis=1)
    mass = grouped(np.ones(len(p)))
    observed = grouped(w)
    events = grouped(w * y)
    predicted = grouped(p)
    raw_events = grouped(y * (w > 0))
    known = grouped((w > 0).astype(float))
    total, all_events = mass.sum(axis=1), events.sum(axis=1)
    population_risk = divide(all_events, observed.sum(axis=1))
    result = {}
    for tail, reverse in [('low', False), ('high', True)]:
        m, o, e, pr, re, kn = [v[:, ::-1] if reverse else v for v in
                               [mass, observed, events, predicted, raw_events, known]]
        before = np.cumsum(m, axis=1) - m
        fraction = np.clip(divide(share * total[:, None] - before, m), 0, 1)
        fraction = np.nan_to_num(fraction, nan=0.)
        selected = (m * fraction).sum(axis=1)
        event_total = (e * fraction).sum(axis=1)
        weight_total = (o * fraction).sum(axis=1)
        risk = divide(event_total, weight_total)
        result[tail] = {
            'event_risk': risk,
            'relative_risk': divide(risk, population_risk),
            'event_capture': divide(event_total, all_events),
            'population_risk': population_risk,
            'selected_expected_n': selected,
            'selected_share': divide(selected, total),
            'predicted_risk': divide((pr * fraction).sum(axis=1), selected),
            'observed_events_fractional': (re * fraction).sum(axis=1),
            'known_status_fractional': (kn * fraction).sum(axis=1),
        }
    return result


def evaluate_model(frame, draw, share, batch_size=32):
    frame = frame.sort_values('subject_id').copy()
    if frame.subject_id.duplicated().any():
        raise ValueError('Duplicate patient in a cell/model')
    p, y, w = [frame[k].to_numpy(float) for k in ['probability', 'label', 'ipcw_weight']]
    if ('prediction_absent_zero_weight' in frame
            and frame.prediction_absent_zero_weight.fillna(False).any()):
        raise ValueError('missing_real_risk_for_early_censored_patients')
    mapping, counts = draw
    mapping = {str(k): v for k, v in mapping.items()}
    indices = [mapping[str(s)] for s in frame.subject_id]
    point = tail_statistics(p, y, w, np.ones(len(p)), share=share)
    parts = {tail: {metric: [] for metric in METRICS} for tail in ['low', 'high']}
    for start in range(0, len(counts), batch_size):
        c = counts[start:start + batch_size, indices]
        values = tail_statistics(p, y, w, c, share=share)
        for tail in parts:
            for metric in METRICS:
                parts[tail][metric].append(values[tail][metric])
    return ({tail: {k: float(v[0]) for k, v in values.items()} for tail, values in point.items()},
            {tail: {k: np.concatenate(v) for k, v in metrics.items()} for tail, metrics in parts.items()})


def bucket(prepared, output, models, split, buckets):
    import pyarrow.parquet as pq
    target = output / 'buckets_INTERNAL'
    if (target / 'complete.json').is_file():
        return target
    if target.exists():
        shutil.rmtree(target)  # Only this module's incomplete private bucketing.
    target.mkdir()
    manifest = json.loads((prepared / 'prepared.json').read_text())
    required = CELL + ['subject_id', 'model', 'label', 'probability', 'ipcw_weight']
    for number, name in enumerate(manifest['parts']):
        path = prepared / name
        if Path(name).is_absolute() or '..' in Path(name).parts:
            raise ValueError('Unsafe prepared partition path')
        file = pq.ParquetFile(path)
        if not set(required) <= set(file.schema.names):
            raise ValueError('Need a full-risk-set cif_ipcw prepared snapshot')
        columns = required + [k for k in ['prediction_absent_zero_weight'] if k in file.schema.names]
        for i, batch in enumerate(file.iter_batches(batch_size=50000, columns=columns)):
            frame = batch.to_pandas()
            frame = frame.loc[frame.model.isin(models) & frame.split.eq(split)].copy()
            if frame.empty:
                continue
            frame['subject_id'] = frame.subject_id.astype(str)
            codes = pd.util.hash_pandas_object(frame[CELL], index=False).to_numpy() % buckets
            for code in np.unique(codes):
                frame.loc[codes == code].to_parquet(target / f'{int(code):04d}_{number:06d}_{i:06d}.parquet', index=False)
        print(f'Fixed-budget bucketing {number + 1}/{len(manifest["parts"])}', flush=True)
    (target / 'complete.json').write_text(json.dumps({'complete': True}))
    return target


def add(accumulator, key, point, draws):
    if not np.isfinite(point) or not np.isfinite(draws).all():
        return
    if key not in accumulator:
        accumulator[key] = [0., 0, np.zeros(len(draws))]
    accumulator[key][0] += point
    accumulator[key][1] += 1
    accumulator[key][2] += draws


def process_bucket(paths, context, models, primary, share):
    frame = pd.concat([pd.read_parquet(p) for p in paths], ignore_index=True)
    cells, pairs, coverage, acc = [], [], [], {}
    support = context['performance_support']
    for key, cell in frame.groupby(CELL, sort=True):
        identity = dict(zip(CELL, key))
        groups = {model: part.sort_values('subject_id') for model, part in cell.groupby('model')}
        outcomes = {}
        for model, part in groups.items():
            known = part.ipcw_weight.gt(0)
            cases, controls = int((part.label.eq(1) & known).sum()), int((part.label.eq(0) & known).sum())
            if min(cases - support['min_cases'], controls - support['min_controls']) < 0:
                coverage.append({**identity, 'model': model, 'reason': 'insufficient_cell_support'})
                continue
            try:
                point, draws = evaluate_model(part, context['draws'][key[:3]], share)
            except ValueError as error:
                if str(error) != 'missing_real_risk_for_early_censored_patients':
                    raise
                coverage.append({**identity, 'model': model, 'reason': str(error)})
                continue
            outcomes[model] = point, draws
        common = set(outcomes) == set(models)
        reason = 'missing_or_unsupported_model'
        if common:
            ref = groups[models[0]]
            common = all(np.array_equal(g.subject_id.to_numpy(), ref.subject_id.to_numpy()) for g in groups.values())
            reason = 'different_patient_roster'
            if common:
                for g in groups.values():
                    for column in ['label', 'ipcw_weight']:
                        if not np.array_equal(g[column].to_numpy(), ref[column].to_numpy()):
                            raise ValueError(f'Matched models disagree on {column}: {identity}')
        if not common:
            if reason == 'missing_or_unsupported_model':
                reason += ':' + ','.join(sorted(set(models) - set(outcomes)))
            coverage.append({**identity, 'model': 'ALL', 'reason': reason})
        stable = common and all(np.isfinite(point[tail][metric]) and np.isfinite(draws[tail][metric]).all()
                                for point, draws in outcomes.values() for tail in ['low', 'high'] for metric in METRICS)
        if common and not stable:
            coverage.append({**identity, 'model': 'ALL', 'reason': 'nonfinite_tail_or_bootstrap_excluded_from_common_summary'})
        for model, (point, draws) in outcomes.items():
            for tail in ['low', 'high']:
                for metric in METRICS:
                    low, high, valid, status = interval(draws[tail][metric])
                    cells.append({**identity, 'model': model, 'tail': tail, 'metric': metric,
                        'estimate': point[tail][metric], 'lower': low, 'upper': high,
                        'valid_draws': valid, 'interval_status': status,
                        'n_patients': len(groups[model]), 'common_roster': common,
                        'common_summary_included': stable, **{k: v for k, v in point[tail].items() if k not in METRICS},
                        'small_tail': point[tail]['selected_expected_n'] < 20,
                        'zero_observed_tail_events': point[tail]['observed_events_fractional'] == 0})
                    if stable:
                        add(acc, ('model', model, *key[:2], key[2], tail, metric), point[tail][metric], draws[tail][metric])
        if not common:
            continue
        a_point, a_draws = outcomes[primary]
        for model in models:
            if model == primary:
                continue
            b_point, b_draws = outcomes[model]
            for tail in ['low', 'high']:
                for metric in METRICS:
                    point = a_point[tail][metric] - b_point[tail][metric]
                    draws = a_draws[tail][metric] - b_draws[tail][metric]
                    low, high, valid, status = interval(draws)
                    name = f'{primary}_minus_{model}'
                    pairs.append({**identity, 'contrast': name, 'tail': tail, 'metric': metric,
                                  'estimate': point, 'lower': low, 'upper': high,
                                  'valid_draws': valid, 'interval_status': status,
                                  'common_summary_included': stable})
                    if stable:
                        add(acc, ('contrast', name, *key[:2], key[2], tail, metric), point, draws)
    return cells, pairs, coverage, acc


def summarize(acc):
    grouped = defaultdict(list)
    for (kind, name, split, seed, cohort, tail, metric), values in acc.items():
        grouped[(kind, name, split, seed, tail, metric)].append((cohort, *values))
    result = []
    for (kind, name, split, seed, tail, metric), parts in grouped.items():
        n = sum(p[2] for p in parts)
        for weighting in ['equal_cell', 'disease_balanced']:
            if weighting == 'equal_cell':
                point = sum(p[1] for p in parts) / n
                draws = sum(p[3] for p in parts) / n
            else:
                point = np.mean([p[1] / p[2] for p in parts])
                draws = np.mean([p[3] / p[2] for p in parts], axis=0)
            low, high, valid, status = interval(draws)
            result.append(dict(kind=kind, name=name, split=split, seed=seed, tail=tail, metric=metric,
                               aggregation=weighting, estimate=point, lower=low, upper=high,
                               n_cells=n, n_diseases=len(parts), valid_draws=valid, interval_status=status))
    return pd.DataFrame(result)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_prepared(prepared=None, checkpoint_dir=None):
    if prepared is not None:
        return Path(prepared).resolve()
    root = Path(checkpoint_dir)
    generation = json.loads((root / 'active.json').read_text())['generation']
    if Path(generation).name != generation:
        raise ValueError('Unsafe checkpoint generation')
    return (root / generation / 'prepared').resolve()


def run(prepared, output, *, models=None, primary='R_EA', split='held_out', share=.1, buckets=256):
    import fcntl
    prepared, output = Path(prepared), Path(output)
    models = list(models or MODELS)
    if len(set(models)) != len(models) or primary not in models or not 0 < share <= .5 or buckets < 1:
        raise ValueError('Invalid models, primary, share or bucket count')
    manifest = json.loads((prepared / 'prepared.json').read_text())
    identity = dict(prepared=str(prepared.resolve()), prepared_sha256=sha(prepared / 'prepared.json'),
                    context_sha256=sha(prepared / 'context_INTERNAL.pkl'), models=models,
                    primary=primary, split=split, share=share, buckets=buckets,
                    implementation_sha256=sha(__file__))
    output.mkdir(parents=True, exist_ok=True)
    with (output / '.run.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        intent = output / 'intent_INTERNAL.json'
        if intent.exists() and json.loads(intent.read_text()) != identity:
            raise ValueError('Inputs/settings changed; use a fresh output directory')
        intent.write_text(json.dumps(identity, indent=2))
        if (output / 'manifest.json').is_file():
            package(output)
            print(f'Already complete: {output}'); return
        context = load_pickle(prepared / 'context_INTERNAL.pkl')
        context.pop('metadata', None)
        if context.get('combined_audit', {}).get('estimand') != 'cif_ipcw':
            raise ValueError('A confirmed full-risk-set cif_ipcw snapshot is required')
        if context.get('bootstrap_method', 'multinomial') not in ['multinomial', 'multiplier']:
            raise ValueError('Use the existing frequentist paired patient draws')
        directory = bucket(prepared, output, models, split, buckets)
        checkpoint = output / 'batches_INTERNAL'; checkpoint.mkdir(exist_ok=True)
        groups = sorted({p.name.split('_')[0] for p in directory.glob('*.parquet')})
        combined = {}
        tables = {name: [] for name in ['cell_metrics', 'paired_differences', 'coverage']}
        handles = {name: (output / (name + '.csv')).open('w', newline='') for name in tables}
        writers = {}
        try:
            for i, code in enumerate(groups):
                result = checkpoint / f'{code}.result_INTERNAL.pkl'
                if not result.is_file():
                    save_pickle(result, process_bucket(sorted(directory.glob(code + '_*.parquet')),
                                                       context, models, primary, share))
                cells, pairs, coverage, acc = load_pickle(result)
                for name, rows in zip(tables, [cells, pairs, coverage]):
                    if not rows:
                        continue
                    if name not in writers:
                        writers[name] = csv.DictWriter(handles[name], fieldnames=list(rows[0]))
                        writers[name].writeheader()
                    writers[name].writerows(rows)
                for key, (point, count, draws) in acc.items():
                    if key not in combined:
                        combined[key] = [point, count, draws.copy()]
                    else:
                        combined[key][0] += point; combined[key][1] += count; combined[key][2] += draws
                print(f'Fixed-budget batches {i + 1}/{len(groups)}', flush=True)
        finally:
            for name in tables:
                if name not in writers:
                    fields = CELL + (['model', 'reason'] if name == 'coverage' else
                                     ['contrast', 'tail', 'metric', 'estimate', 'lower', 'upper'])
                    csv.writer(handles[name]).writerow(fields)
            for handle in handles.values(): handle.close()
        summary = summarize(combined)
        summary.to_csv(output / 'summary.csv', index=False)
        if summary.empty:
            raise ValueError('No common, supported, finite-bootstrap roster; inspect coverage.csv')
        result = dict(exploratory=True, status='complete', **identity,
                      bootstrap_method=context.get('bootstrap_method', 'multinomial'),
                      replicates=context['replicates'], input_hashes=context.get('hashes', {}),
                      performance_support=context['performance_support'],
                      tie_policy='Fractional expected selection within boundary score ties; reselected per draw',
                      risk_estimator='Hajek ratio using fixed main-analysis IPCW weights',
                      summary_roster='Common models/patients/cells, finite in every paired draw for all metrics/tails',
                      uncertainty='Patient resampling conditional on fitted models and fixed censoring weights; no multiplicity claims')
        (output / 'report.md').write_text(
            '# Fixed-budget exploratory analysis\n\n'
            'Highest and lowest ' + str(share * 100) + '% in every supported cell.\n\n'
            'Cell risks and event capture are descriptive; percentile groups use predictions only. '
            'Boundary ties are fractional expected random allocations, not uniquely identified patients. '
            'Early-censored patients remain in the ranking population. Group risks are IPCW-normalised '
            'ratios; censoring weights are not refitted within tails or bootstrap draws. '
            'Relative risks are not prevalence-independent measures of task difficulty. '
            'Small tails and zero observed tail events are flagged; a zero-event degenerate bootstrap '
            'interval does not establish safety. Point risks are also retained outside the common '
            'comparison roster. Summary estimates use a fixed common finite-bootstrap roster. '
            'No p-values, model selection, training or analysis-lock changes.\n')
        plot_summary(summary, output, share)
        (output / 'manifest.json').write_text(json.dumps(result, indent=2))
        package(output)
    return result


def plot_summary(summary, output, share):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from opera.visualization.style import setup_style
    setup_style()
    labels = {'R_EA': 'Adapted OPERA', 'J_EA': 'Adapted joint',
              'J_OA_OWN_L': 'Cohort OPERA', 'XP': 'Pooled XGBoost',
              'X': 'Task-specific XGBoost', 'XR': 'Registry-augmented XGBoost'}
    data = summary.loc[summary.kind.eq('model')]
    for (split, seed), frame in data.groupby(['split', 'seed']):
        models = list(frame.name.drop_duplicates())
        fig, axes = plt.subplots(2, 2, figsize=(10, max(5, .65 * len(models) + 1)),
                                 sharey=True, layout='constrained')
        for row, metric in enumerate(['relative_risk', 'event_capture']):
            for column, tail in enumerate(['high', 'low']):
                ax = axes[row, column]
                ax.axvline(1 if metric == 'relative_risk' else share, color='#A0A0A0', ls='--', lw=.8)
                for i, model in enumerate(models):
                    for weighting, offset, colour in [('equal_cell', -.12, '#2D3A8C'),
                                                        ('disease_balanced', .12, '#B86A55')]:
                        part = frame.loc[frame.name.eq(model) & frame.metric.eq(metric)
                                         & frame['tail'].eq(tail) & frame.aggregation.eq(weighting)]
                        if part.empty:
                            continue
                        r = part.iloc[0]
                        ax.plot([r.lower, r.upper], [i + offset] * 2, color=colour, lw=1)
                        ax.scatter(r.estimate, i + offset, color=colour, s=20,
                                   label=weighting.replace('_', ' ') if i == 0 else None)
                ax.set_yticks(range(len(models)), [labels.get(m, m) for m in models])
                ax.invert_yaxis() if not ax.yaxis_inverted() else None
                ax.set_title(f'{"Highest" if tail == "high" else "Lowest"}-risk {share:.0%}', loc='left')
                ax.set_xlabel('Mean task-specific risk / task incidence (95% CI)' if metric == 'relative_risk'
                              else 'Mean fraction of events captured (95% CI)')
                ax.grid(axis='x', alpha=.2)
        axes[0, 0].legend(fontsize=8)
        fig.suptitle(f'Exploratory fixed-budget stratification — {split}, seed {seed}')
        for suffix in ['pdf', 'png']:
            fig.savefig(output / f'fixed_budget_{split}_seed{int(seed)}.{suffix}', dpi=180)
        plt.close(fig)


def package(output):
    names = ['cell_metrics.csv', 'paired_differences.csv', 'summary.csv', 'coverage.csv', 'manifest.json', 'report.md']
    names += [p.name for p in sorted(output.glob('fixed_budget_*_seed*.pdf'))]
    names += [p.name for p in sorted(output.glob('fixed_budget_*_seed*.png'))]
    with ZipFile(output / 'fixed_budget_review.zip', 'w', ZIP_DEFLATED) as archive:
        for name in names:
            path = output / name
            if path.suffix == '.csv' and path.stat().st_size:
                with path.open(newline='') as handle:
                    columns = next(csv.reader(handle), [])
                if {c.lower() for c in columns} & {'subject_id', 'subject_ids', 'patient_id', 'cpr', 'pnr'}:
                    raise ValueError('Patient columns in aggregate output')
            archive.write(path, name)
    print(f'Download: {output / "fixed_budget_review.zip"}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--prepared-dir', type=Path)
    source.add_argument('--checkpoint-dir', type=Path)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--models', nargs='+', default=MODELS)
    parser.add_argument('--primary', default='R_EA', help='Declared comparison reference, not selected here')
    parser.add_argument('--split', choices=['tuning', 'held_out'], default='held_out')
    parser.add_argument('--share', type=float, default=.1)
    parser.add_argument('--buckets', type=int, default=256)
    args = parser.parse_args()
    run(resolve_prepared(args.prepared_dir, args.checkpoint_dir), args.output_dir,
        models=args.models, primary=args.primary, split=args.split, share=args.share, buckets=args.buckets)


if __name__ == '__main__':
    main()
