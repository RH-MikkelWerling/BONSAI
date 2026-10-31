"""Tuning-only component/gate interventions using saved encoder vectors on CPU.

No encoder pass, retraining, source-freshness guard or held-out evaluation.
The selected head must reconstruct saved probabilities before interpretation.
Writes aggregate reports and an allowlisted review ZIP, never patient arrays.
"""
from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
import zipfile

import numpy as np
import pandas as pd
import torch
import yaml

from opera.diagnostics.router_mechanism import (
    COMPONENTS, VARIANTS, FrozenRouterHead, TaskReferences, borrowed_mixture_health,
    variant_terms,
)
from opera.evaluation.secondary_analysis import (
    effect_summaries, paired_analysis, patient_bootstrap,
)
from opera.run.analyse_locked_predictions import file_hash, read_artifact, read_table


def checkpoint_for_predictions(predictions):
    """predictions/<origin>/seed<k>/<hash>/tuning/<file> -> neural_runs/.../best.ckpt."""
    path = Path(predictions).resolve()
    run, data = path.parent.parent, path.parents[5] if len(path.parents) > 5 else None
    if path.parent.name != 'tuning' or data is None or path.parents[4].name != 'predictions':
        raise ValueError(f'Unexpected prediction layout, pass --checkpoint: {path}')
    return data / 'neural_runs' / run.relative_to(path.parents[4]) / 'best.ckpt'


def selected_inputs(root, seed, model=None, checkpoint=None):
    """Select only the declared tuning artifact, never discover test files.

    Without ``model``: the analysis's declared OPERA model and the fit recorded
    in completion.json. With ``model`` (e.g. R_PP_L; lock amendment 2026-10-30):
    that alias's declared tuning artifact and its checkpoint, whose identity the
    saved encoder vectors must confirm."""
    analysis = yaml.safe_load((root / 'analysis_inventory/analysis.yaml').read_text())
    if analysis.get('estimand') != 'cif_ipcw':
        raise ValueError('This audit requires the paper CIF IPCW estimand')
    declared = model or analysis.get('opera_model', 'R_L')
    candidates = [a for a in analysis['artifacts'] if a['model'] == declared
                  and a.get('split') == 'tuning' and int(a.get('seed', 42)) == seed]
    if len(candidates) != 1 or candidates[0].get('format') != 'direct_cif':
        raise ValueError(f'Need exactly one declared native tuning artifact for {declared}')
    if model is not None:
        path = Path(checkpoint) if checkpoint else checkpoint_for_predictions(candidates[0]['path'])
        if not path.is_file():
            raise FileNotFoundError(f'No checkpoint for {model}: {path}')
        fit = dict(predictions=candidates[0]['path'], checkpoint=str(path),
                   checkpoint_sha256=file_hash(path), model=model)
        return analysis, candidates[0], fit
    fit = json.loads((root / 'completion.json').read_text())['inputs']['fit']
    if Path(candidates[0]['path']).resolve() != Path(fit['predictions']).resolve():
        raise ValueError('Primary artifact and recorded fit identify different predictions')
    if file_hash(fit['checkpoint']) != fit['checkpoint_sha256']:
        raise ValueError('Selected checkpoint content differs from its recorded fit')
    return analysis, candidates[0], fit


def load_vectors(root, fit, override=None):
    """Align by subject ID, not by file order; require tuning provenance."""
    path = Path(override) if override else Path(fit['predictions']).parent / 'patient_embeddings.npz'
    if not path.is_file() and override is None:
        paths = sorted((root / 'routed_vectors').glob('*/routed_patient_vectors_INTERNAL.npz'))
        if not paths:
            raise FileNotFoundError('No saved pooled vectors. Supply --vectors for the selected tuning fit.')
        path = paths[0]
    marker_path = (path.parent / 'patient_embeddings.metadata.json'
                   if path.name == 'patient_embeddings.npz' else path.with_suffix('.json'))
    marker = json.loads(marker_path.read_text())
    if marker.get('split') != 'tuning':
        raise ValueError('Encoder vectors must be tuning-only')
    if Path(marker['checkpoint']).resolve() != Path(fit['checkpoint']).resolve():
        raise ValueError('Encoder vectors identify a different checkpoint')
    with np.load(path, allow_pickle=False) as saved:
        if 'splits' in saved and not np.all(saved['splits'].astype(str) == 'tuning'):
            raise ValueError('Encoder vector rows must be tuning-only')
        ids = saved['subject_ids'].astype(str)
        values = saved['embeddings' if 'embeddings' in saved else 'shared'].astype(np.float32)
    if (len(set(ids)) != len(ids) or values.ndim != 2 or len(values) != len(ids)
            or not np.isfinite(values).all()):
        raise ValueError('Invalid or duplicate pooled encoder vectors')
    return ids, values, path, marker_path


def prepare_population(analysis, specification, head, ids, values, *, outcomes, horizons):
    metadata = read_table(analysis['patient_metadata'])
    metadata['subject_id'] = metadata.subject_id.astype(str)
    if metadata.subject_id.duplicated().any():
        raise ValueError('Duplicate patients in metadata')
    baseline, _ = read_artifact(specification, metadata, 'cif_ipcw')
    if not baseline.split.eq('tuning').all():
        raise ValueError('Baseline predictions must be tuning-only')
    if outcomes == ['all']:
        outcomes = list(head.outcomes)
    if not outcomes or len(set(outcomes)) != len(outcomes) or set(outcomes) - set(head.outcomes):
        raise ValueError('Unknown, empty or duplicate target outcomes')
    absent = set(outcomes) - set(baseline.outcome)
    if absent:
        raise ValueError(f'Requested outcomes missing from saved predictions: {sorted(absent)}')
    if horizons is None:
        horizons = sorted(analysis.get('analysis_horizons_days') or baseline.horizon_days.unique())
    if not horizons or len(set(horizons)) != len(horizons) or any(h <= 0 for h in horizons):
        raise ValueError('Need distinct positive horizons')
    if set(horizons) - set(baseline.horizon_days):
        raise ValueError('Requested horizon absent from saved predictions')
    baseline = baseline.loc[baseline.outcome.isin(outcomes) & baseline.horizon_days.isin(horizons)].copy()
    baseline = baseline.sort_values(['subject_id', 'outcome', 'horizon_days']).reset_index(drop=True)
    patients = sorted(baseline.subject_id.unique())
    position = pd.Index(ids).get_indexer(patients)
    if (position < 0).any():
        raise ValueError(f'Saved vectors lack {int((position < 0).sum())} required tuning patients')
    fine_by_id = metadata.set_index('subject_id').cohort_fine
    membership = fine_by_id.reindex(patients)
    fine_names = list(head.readout.fine_names)
    fine = pd.Index(fine_names).get_indexer(membership)
    if (fine < 0).any():
        raise ValueError('Missing or unknown fine-disease identity in audit population')
    if not baseline.cohort.eq(baseline.subject_id.map(fine_by_id)).all():
        raise ValueError('Paper evaluation cohorts must match fine-disease metadata')
    target_position = {o: i for i, o in enumerate(outcomes)}
    subject_position = {s: i for i, s in enumerate(patients)}
    eligible = np.zeros((len(patients), len(outcomes)), dtype=bool)
    pair = baseline[['subject_id', 'outcome']].drop_duplicates()
    eligible[[subject_position[s] for s in pair.subject_id],
             [target_position[o] for o in pair.outcome]] = True
    row_index = (baseline.subject_id.map(subject_position).to_numpy(int),
                 baseline.outcome.map(target_position).to_numpy(int),
                 baseline.horizon_days.map({h: i for i, h in enumerate(horizons)}).to_numpy(int))
    return (baseline, torch.from_numpy(values[position]), torch.tensor(fine),
            torch.from_numpy(eligible), outcomes, [int(h) for h in horizons], row_index)


def infer_interventions(head, patient, fine, eligible, indices, horizons, *, batch_size=128):
    reference = TaskReferences(head, len(indices))
    algebra_error = 0.
    with torch.inference_mode():
        for start in range(0, len(patient), batch_size):
            stop = start + batch_size
            x, f = patient[start:stop], fine[start:stop]
            terms, gates = head.terms(x, f, indices)
            # Independent production readout, not the intervention reconstruction.
            adapted = head.readout(x, f)[:, indices]
            original = torch.einsum('boh,okh->bok', adapted, head.weight[indices]) + head.bias[indices][None]
            algebra_error = max(algebra_error, float((sum(terms.values()) - original).abs().max()))
            if not torch.allclose(sum(terms.values()), original, atol=2e-5, rtol=2e-5):
                raise ValueError('Component terms do not reconstruct the production readout/head')
            reference.add(f, eligible[start:stop], terms, gates, head.borrowed_mask(f, indices))
        reference.finalize()
        predictions = {name: np.empty((len(patient), len(indices), len(horizons)), np.float32)
                       for name in ('FULL', *VARIANTS)}
        for start in range(0, len(patient), batch_size):
            stop = min(start + batch_size, len(patient))
            for name, flat in variant_terms(head, patient[start:stop], fine[start:stop], indices, reference):
                predictions[name][start:stop] = head.cif(flat, indices, horizons).numpy()
    return predictions, reference, algebra_error


def head_effect_table(head, reference, outcomes):
    rows = []
    intervals = len(head.boundaries) + 1
    for fine, disease in enumerate(head.readout.fine_names):
        for o, outcome in enumerate(outcomes):
            n = int(reference.count[fine, o])
            if not n:
                continue
            for component in COMPONENTS:
                signed = reference.terms[component][fine, o].reshape(2, intervals)
                absolute = reference.absolute_terms[component][fine, o].reshape(2, intervals)
                for cause, label in enumerate(('target_event', 'competing_event')):
                    for interval in range(intervals):
                        rows.append(dict(cohort=disease, outcome=outcome, component=component,
                            cause=label, interval=interval, n_patients=n,
                            cause_active=not (label == 'competing_event' and outcome in
                                {name for i, name in enumerate(head.outcomes) if head.no_competing[i]}),
                            mean_head_change=float(signed[cause, interval]),
                            mean_absolute_head_change=float(absolute[cause, interval])))
    return pd.DataFrame(rows)


def risk_change_table(baseline, candidate, variant):
    delta = candidate.probability.to_numpy() - baseline.probability.to_numpy()
    frame = baseline[['cohort', 'outcome', 'horizon_days']].assign(delta=delta)
    event = baseline.event.to_numpy()
    time = baseline.time_days.to_numpy()
    horizon = baseline.horizon_days.to_numpy()
    status = np.where((event == 1) & (time <= horizon), 'event',
        np.where((event == 2) & (time <= horizon), 'competing_death',
        np.where(time >= horizon, 'event_free', 'censored')))
    frame = pd.concat([frame.assign(status='all'), frame.assign(status=status)], ignore_index=True)
    rows = []
    keys = ['cohort', 'outcome', 'horizon_days', 'status']
    for key, group in frame.groupby(keys):
        values = group.delta.to_numpy()
        rows.append(dict(zip(keys, key)) | dict(
            variant=variant, n_patients=len(values), mean_risk_change=float(values.mean()),
            direction='intervention_minus_full',
            median_absolute_risk_change=float(np.median(abs(values))),
            p90_absolute_risk_change=float(np.quantile(abs(values), .9))))
    return pd.DataFrame(rows)


# Separately trained ladders. The partial-pooling one (lock amendment
# 2026-10-30) splits R_PP's difference from R_AB into the structural change
# (R_PP_FROZEN - R_AB) and learned routing (R_PP - R_PP_FROZEN).
LADDERS = {'R_AB': ['J_L', 'N_A_L', 'RBH_AB_L', 'R_AB_L'],
           'R_PP': ['J_L', 'R_AB_L', 'R_PP_FROZEN_L', 'R_PP_L']}


def ladder_for(model):
    if model and model.startswith('R_PT'):
        # Task-routed candidates: frozen even-mix control, then the candidate.
        return ['J_L', 'R_AB_L', 'R_PP_FROZEN_L', model]
    return LADDERS['R_PP'] if model and model.startswith('R_PP') else LADDERS['R_AB']


def gate_tables(head, reference, outcomes, output):
    """Where the borrowed weight goes (lock amendment 2026-10-30): the mean
    gate of every borrowed source for each disease x target task, and its
    disease-level average. Unlabelled tuning means; descriptive, not reliance."""
    r = head.readout
    sources = [*(f'disease:{n}' for n in r.fine_names), *(f'outcome:{n}' for n in r.outcome_names)]
    rows = []
    for d, disease in enumerate(r.fine_names):
        for o, outcome in enumerate(outcomes):
            n = float(reference.count[d, o])
            if n <= 0:
                continue
            gates = reference.gates[d, o, 1:]
            own = {d, r.n_disease + r.outcome_names.index(outcome)}
            borrowed = sum(float(gates[s]) for s in range(r.n_sources) if s not in own)
            for s, source in enumerate(sources):
                if s in own:
                    continue
                weight = float(gates[s])
                rows.append(dict(disease=disease, outcome=outcome, source=source,
                                 bank=source.split(':', 1)[0], n_patients=int(n),
                                 mean_gate=weight,
                                 share_of_borrowed=weight / borrowed if borrowed > 0 else float('nan')))
    tasks = pd.DataFrame(rows)
    if tasks.empty:
        return dict(status='not_available'), tasks
    tasks.to_csv(output / 'borrowed_weights_by_task.csv', index=False)
    by_disease = (tasks.groupby(['disease', 'source', 'bank'], as_index=False)
                  .share_of_borrowed.mean().sort_values(['disease', 'share_of_borrowed'],
                                                        ascending=[True, False]))
    by_disease.to_csv(output / 'borrowed_weights_by_disease.csv', index=False)
    top = by_disease.groupby('disease').head(3)
    return dict(status='complete', tasks=len(tasks.groupby(['disease', 'outcome'])),
                top_sources_by_disease={d: list(zip(g.source, g.share_of_borrowed.round(4)))
                                        for d, g in top.groupby('disease')}), tasks


# Prespecified route definitions (lock amendment 2026-10-30, before any R_PP
# result): a borrowed source is "recognisable" when it shares the target's group
# at one of these levels, taken from the paper cell metadata.
ROUTE_LEVELS = {'lineage': ('disease', 'cohort_group'),
                'endpoint_family': ('outcome', 'loss_endpoint'),
                'endpoint_group': ('outcome', 'endpoint_group')}


def route_concentration(tasks, cell_metadata, *, permutations=2000, seed=2026):
    """Within-group share of borrowed weight against an even mix, per level.

    For each target task, the observed share is the borrowed weight (within the
    level's bank) on sources in the target's own group; the expected share is
    the in-group fraction of available sources. Tasks without an available
    group partner are excluded. The permutation null shuffles group labels
    among sources, keeping group sizes.
    """
    rng = np.random.default_rng(seed)
    meta = cell_metadata.drop_duplicates(['cohort', 'outcome'])
    groups = {'disease': meta.drop_duplicates('cohort').set_index('cohort'),
              'outcome': meta.drop_duplicates('outcome').set_index('outcome')}
    results = {}
    for level, (bank, column) in ROUTE_LEVELS.items():
        if column not in groups[bank]:
            results[level] = dict(status='not_available', reason=f'No {column} in cell metadata')
            continue
        label = groups[bank][column].astype(str).to_dict()
        part = tasks.loc[tasks.bank == bank].copy()
        part['name'] = part.source.str.split(':', n=1).str[1]
        part = part.loc[(part.mean_gate > 1e-12) & part.name.isin(label)]
        if part.empty:
            results[level] = dict(status='not_estimable', reason=f'No available {bank} sources')
            continue
        part = part.loc[part[bank].isin(label)]
        # Targets and sources are the same set (diseases, or outcomes): one
        # permutation relabels both, so a target's own exclusion is preserved.
        names = sorted(set(part.name) | set(part[bank]))
        codes = {name: i for i, name in enumerate(names)}
        source_label = np.array([label[name] for name in names])
        source_index = part.name.map(codes).to_numpy()
        target_index = part[bank].map(codes).to_numpy()
        task_index = pd.MultiIndex.from_frame(part[['disease', 'outcome']])
        task, task_keys = task_index.factorize()
        gate = part.mean_gate.to_numpy(float)
        total = np.bincount(task, gate)
        available = np.bincount(task)

        def shares(assigned, *, with_keys=False):
            inside = assigned[source_index] == assigned[target_index]
            in_count = np.bincount(task, inside, minlength=len(total))
            keep = in_count > 0  # tasks with an available group partner
            observed = np.bincount(task, gate * inside, minlength=len(total))[keep] / total[keep]
            expected = in_count[keep] / available[keep]
            if with_keys:
                return observed, expected, task_keys[keep]
            return pd.Series(observed), pd.Series(expected)

        observed, expected = shares(source_label)
        if observed.empty:
            results[level] = dict(status='not_estimable', reason='No task has an available group partner')
            continue
        statistic = float(observed.mean())
        null = np.array([shares(rng.permutation(source_label))[0].mean() for _ in range(permutations)])
        # Descriptive breakdown by the target's own group: where any concentration sits.
        obs, exp, keys = shares(source_label, with_keys=True)
        target_group = [label[key[0] if bank == 'disease' else key[1]] for key in keys]
        by_group = (pd.DataFrame(dict(group=target_group, observed=obs, expected=exp))
                    .groupby('group').agg(n_tasks=('observed', 'size'), observed=('observed', 'mean'),
                                          expected=('expected', 'mean')))
        by_group['enrichment'] = by_group.observed / by_group.expected
        results[level] = dict(
            by_target_group={g: {k: (int(v) if k == 'n_tasks' else round(float(v), 4))
                                 for k, v in row.items()} for g, row in by_group.iterrows()},
            status='complete', grouping=column, n_tasks=int(len(observed)),
            observed_in_group_share=statistic, expected_in_group_share=float(expected.mean()),
            enrichment=statistic / float(expected.mean()) if float(expected.mean()) > 0 else float('nan'),
            permutation_p=float((1 + (null >= statistic).sum()) / (1 + permutations)),
            permutations=permutations)
    return results


def trained_control_ladder(root, seed, horizons, outcomes, output, models=None):
    """Descriptive existing-fit comparisons, kept separate from interventions."""
    source = root / 'prediction_analysis/estimates.csv'
    if not source.is_file():
        return dict(status='not_available', reason='No existing per-model estimate table')
    models = list(models or LADDERS['R_AB'])
    columns = ['split', 'seed', 'cohort', 'outcome', 'horizon_days', 'model', 'metric', 'estimate']
    parts = []
    for chunk in pd.read_csv(source, usecols=columns, chunksize=100_000):
        parts.append(chunk.loc[(chunk.split == 'tuning') & (chunk.seed == seed)
            & chunk.model.isin(models) & (chunk.metric == 'auc')
            & chunk.horizon_days.isin(horizons) & chunk.outcome.isin(outcomes)])
    frame = pd.concat(parts, ignore_index=True)
    if set(models) - set(frame.model):
        return dict(status='not_available', reason='Missing matched local trained controls',
                    missing_models=sorted(set(models) - set(frame.model)))
    keys = ['cohort', 'outcome', 'horizon_days']
    wide = frame.pivot(index=keys, columns='model', values='estimate').reindex(columns=models)
    rows, coverage = [], []
    for horizon in [0, *horizons]:
        panel = wide if horizon == 0 else wide.loc[wide.index.get_level_values('horizon_days') == horizon]
        common = panel.dropna()
        coverage.append(dict(horizon_days=horizon, n_common_cell_horizons=len(common),
            n_any_supported_cell_horizons=int(panel.notna().any(axis=1).sum())))
        for model in models:
            rows.append(dict(model=model, horizon_days=horizon, n_cell_horizons=len(common),
                mean_auc=float(common[model].mean()),
                mean_auc_difference_vs_joint=float((common[model] - common.J_L).mean()),
                interpretation='Descriptive separately trained fits, on an explicitly common roster'))
    pd.DataFrame(rows).to_csv(output / 'trained_control_ladder.csv', index=False)
    return dict(status='available', source=str(source), coverage=coverage,
                uncertainty='Descriptive point estimates; no new trained-control interval calculated')


def compare_joint_paths(analysis, baseline, predictions, row_index, draws, metadata, output, *, replicates):
    """Separate the trained router's encoder/head path from its added terms."""
    seed = int(baseline.seed.iloc[0])
    specs = [a for a in analysis['artifacts'] if a['model'] == 'J_L'
             and a.get('split') == 'tuning' and int(a.get('seed', 42)) == seed]
    if not specs:
        return dict(status='not_available', reason='No declared local joint tuning artifact')
    if len(specs) != 1:
        raise ValueError('Need exactly one declared local joint tuning artifact')
    patient_metadata = read_table(analysis['patient_metadata'])
    patient_metadata['subject_id'] = patient_metadata.subject_id.astype(str)
    joint, _ = read_artifact(specs[0], patient_metadata, 'cif_ipcw')
    joint = joint.loc[joint.outcome.isin(baseline.outcome.unique())
                      & joint.horizon_days.isin(baseline.horizon_days.unique())].copy()
    keys = ['subject_id', 'cohort', 'outcome', 'horizon_days']
    check = baseline[keys].merge(joint[keys], on=keys, how='outer', validate='one_to_one', indicator=True)
    if not check['_merge'].eq('both').all():
        raise ValueError('Joint/router mechanism comparison has different patient/target coverage')
    joint['model'] = 'JOINT'
    frame = pd.concat([joint, baseline, *[
        baseline.assign(model=name, probability=predictions[name][row_index])
        for name in ('ENCODER_HEAD_ONLY', 'GLOBAL_ONLY')]], ignore_index=True)
    contrasts = {name + '_VS_JOINT': f'gain({name},JOINT)'
                 for name in ('FULL', 'ENCODER_HEAD_ONLY', 'GLOBAL_ONLY')}
    tables = paired_analysis(frame, contrasts, replicates=replicates, draws=draws,
        patient_gain_contrasts=set(), min_cases=analysis.get('performance_min_cases', 5),
        min_controls=analysis.get('performance_min_controls', 5))
    summaries = []
    for aggregation in ('equal_cell', 'disease_balanced'):
        _, summary = effect_summaries(tables['contrasts'], tables['bootstrap_draws'], metadata,
                                     aggregation=aggregation)
        summaries.append(summary.assign(weighting=aggregation))
    for name in ('contrasts', 'estimates', 'contrast_coverage'):
        tables[name].to_csv(output / ('joint_path_' + name + '.csv'), index=False)
    pd.concat(summaries, ignore_index=True).to_csv(output / 'joint_path_summaries.csv', index=False)
    return dict(status='available', joint_model='J_L', joint_artifact=specs[0]['path'],
        contrasts=contrasts, interpretation='Encoder/head and global-only paths use the router checkpoint. '
            'JOINT uses the separate J_L fit. Paired patient uncertainty uses the same draws as interventions.')


COMPONENT_DISPLAY = [
    ('GLOBAL', 'Global correction', '#737D8C'),
    ('OWN', 'Own disease / outcome', '#B18B4D'),
    ('BORROWED_DISEASE', 'Other diseases', '#3E8A87'),
    ('BORROWED_OUTCOME', 'Other outcomes', '#9460A5'),
    ('ALL_BORROWED', 'All borrowed sources', '#2D3A8C'),
]


def _overview_rows(frame, metric, key):
    if frame is None or frame.empty:
        return pd.DataFrame(columns=[key, 'gain', 'lower', 'upper']).set_index(key)
    chosen = frame.loc[(frame.stratum == 'all_cells') & (frame.horizon_days == 0)
                       & (frame.metric == metric)]
    if 'weighting' in chosen:
        chosen = chosen.loc[chosen.weighting == 'equal_cell']
    if chosen.duplicated(key).any():
        raise ValueError(f'Duplicate overview rows for {key}/{metric}')
    return chosen.set_index(key)


def _mechanism_point(ax, table, key, y, color, *, hollow=False, marker=None):
    if key not in table.index or not np.isfinite(table.loc[key, 'gain']):
        return False
    row = table.loc[key]
    if np.isfinite([row.lower, row.upper]).all():
        ax.plot([row.lower, row.upper], [y, y], color=color, lw=1.15,
                alpha=.65 if hollow else .95, zorder=2)
    ax.plot(row.gain, y, marker=marker or ('o' if hollow else 'D'), ls='',
            ms=4.5, mfc='white' if hollow else color, mec=color, mew=1.0, zorder=3)
    return True


def _mechanism_axis(ax, labels, xlabel, *, ticks=3):
    from matplotlib.ticker import MaxNLocator
    from opera.visualization.style import PALETTE, despine
    ax.axvline(0, color=PALETTE['ink_muted'], lw=.75, zorder=1)
    ax.set_yticks(range(len(labels)), labels, fontsize=8)
    ax.set_ylim(len(labels) - .45, -.55)
    ax.set_xlabel(xlabel, fontsize=8, labelpad=7)
    ax.tick_params(axis='y', length=0, pad=8)
    ax.tick_params(axis='x', labelsize=7.5)
    ax.xaxis.set_major_locator(MaxNLocator(ticks))
    ax.ticklabel_format(axis='x', style='sci', scilimits=(-4, 4), useMathText=True, useOffset=False)
    ax.margins(x=.18)
    if max(abs(v) for v in ax.get_xlim()) < 1e-12:
        # Exact/no-rank-change effects can differ by machine roundoff. Do not
        # turn 1e-17 AUC into an apparently substantial forest plot.
        ax.set_xlim(-.001, .001)
    despine(ax, 'x')
    ax.spines['left'].set_visible(False)


def figure_router_mechanism(summary, joint, *, metric='auc', mark='TUNING SET — not test results'):
    """Three questions, with matched comparisons and explicit replacements."""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from opera.visualization.style import PALETTE, TITLE_SIZE
    unit = {'auc': 'AUC', 'ipa': 'IPA'}[metric]
    data = _overview_rows(summary, metric, 'variant')
    paths = _overview_rows(joint, metric, 'contrast')
    fig = plt.figure(figsize=(8.4, 6.2))
    a = fig.add_axes([.27, .69, .70, .20])
    b = fig.add_axes([.225, .19, .255, .30])
    c = fig.add_axes([.755, .19, .215, .30])
    fig.text(.07, .965, 'A  Where does the gain over joint appear?', fontsize=TITLE_SIZE, va='top')
    fig.text(.07, .918, 'Paths through the frozen OPERA checkpoint, compared with the separately trained joint model',
             fontsize=7.5, color=PALETTE['ink_secondary'], va='top')
    path_rows = [('ENCODER_HEAD_ONLY_VS_JOINT', 'Encoder/head\nwithout corrections', '#A0A9B8'),
                 ('GLOBAL_ONLY_VS_JOINT', 'With global correction', '#737D8C'),
                 ('FULL_VS_JOINT', 'Full OPERA', PALETTE['opera'])]
    a.axhspan(1.63, 2.37, color=PALETTE['opera'], alpha=.045, lw=0, zorder=0)
    for y, (key, _, color) in enumerate(path_rows):
        if not _mechanism_point(a, paths, key, y, color):
            a.text(.5, y, 'Comparison unavailable', transform=a.get_yaxis_transform(),
                   ha='center', va='center', fontsize=7, color=PALETTE['ink_muted'])
    _mechanism_axis(a, [r[1] for r in path_rows], f'{unit} gain over joint (mean, 95% CI)', ticks=4)
    a.get_yticklabels()[-1].set_color(PALETTE['opera'])
    a.get_yticklabels()[-1].set_fontweight('bold')

    fig.text(.07, .57, 'B  Which corrections help?', fontsize=TITLE_SIZE, va='top')
    legend = [Line2D([], [], ls='', marker='D', ms=4, color=PALETTE['ink_secondary'], label='Mean replacement'),
              Line2D([], [], ls='', marker='o', ms=4, color=PALETTE['ink_secondary'], mfc='white', label='Zero replacement')]
    fig.legend(handles=legend, loc='upper left', bbox_to_anchor=(.066, .531), ncol=2,
               fontsize=7, frameon=False, handletextpad=.4, columnspacing=1.1, borderaxespad=0)
    for y, (key, _, color) in enumerate(COMPONENT_DISPLAY):
        _mechanism_point(b, data, 'MEAN_' + key, y-.12, color)
        _mechanism_point(b, data, 'ZERO_' + key, y+.12, color, hollow=True)
    _mechanism_axis(b, [r[1] for r in COMPONENT_DISPLAY], f'{unit} gain: full − altered\n(mean, 95% CI)')

    fig.text(.565, .57, 'C  Does individual routing help?', fontsize=TITLE_SIZE, va='top')
    fig.text(.565, .52, 'Adapter outputs still vary between patients', fontsize=7.5,
             color=PALETTE['ink_secondary'], va='top')
    for y, key in enumerate(('TASK_MEAN_GATES', 'TASK_MEAN_BORROWED_COMPOSITION')):
        _mechanism_point(c, data, key, y, PALETTE['opera'])
    _mechanism_axis(c, ['Fix all gate\nweights', 'Fix borrowed-source\nproportions'],
                    f'{unit} gain: full − altered\n(mean, 95% CI)')
    c.set_ylim(1.7, -.7)
    # Both panels measure full minus intervention. A common scale makes the
    # magnitude of gating effects comparable with the correction effects.
    limit = max(abs(v) for ax in (b, c) for v in ax.get_xlim())
    b.set_xlim(-limit, limit)
    c.set_xlim(-limit, limit)
    fig.text(.07, .075, 'B–C: positive values favour full OPERA. Mean replacement retains the average correction.\n'
             'C: fixing source proportions preserves each patient’s total borrowed weight.',
             fontsize=7, color=PALETTE['ink_secondary'], va='top', linespacing=1.5)
    fig.text(.97, .018, mark, fontsize=6.5, color=PALETTE['ink_muted'], ha='right', va='bottom')
    return fig


def figure_router_mechanism_sensitivity(summary, *, mark='TUNING SET — not test results'):
    """Complete results in nine readable rows instead of fourteen code names."""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from opera.visualization.style import PALETTE, TITLE_SIZE
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 5.9), sharey=True,
        gridspec_kw={'left': .255, 'right': .98, 'bottom': .19, 'top': .85, 'wspace': .32})
    extra = [('ENCODER_HEAD_ONLY', 'Encoder/head only', '#A0A9B8'),
             ('GLOBAL_ONLY', 'Global path only', '#737D8C'),
             ('TASK_MEAN_GATES', 'Fixed gates', PALETTE['opera']),
             ('TASK_MEAN_BORROWED_COMPOSITION', 'Fixed source proportions', PALETTE['opera'])]
    labels = [r[1] for r in COMPONENT_DISPLAY] + [r[1] for r in extra]
    for ax, metric, title in zip(axes, ('auc', 'ipa'), ('A  Discrimination', 'B  Prediction accuracy')):
        data = _overview_rows(summary, metric, 'variant')
        for y, (key, _, color) in enumerate(COMPONENT_DISPLAY):
            _mechanism_point(ax, data, 'MEAN_'+key, y-.11, color)
            _mechanism_point(ax, data, 'ZERO_'+key, y+.11, color, hollow=True)
        for y, (key, _, color) in enumerate(extra, start=5):
            _mechanism_point(ax, data, key, y, color)
        unit = {'auc': 'AUC gain', 'brier': 'Brier improvement', 'ipa': 'IPA gain'}[metric]
        _mechanism_axis(ax, labels, f'{unit}\n(mean, 95% CI)')
        for boundary in (4.5, 6.5):
            ax.axhline(boundary, color=PALETTE['grid'], lw=.65, zorder=0)
        ax.set_title(title, loc='left', fontsize=TITLE_SIZE)
    fig.legend(handles=[Line2D([], [], ls='', marker='D', color=PALETTE['ink_secondary'], ms=4, label='Mean replacement / fixed path'),
                        Line2D([], [], ls='', marker='o', color=PALETTE['ink_secondary'], mfc='white', ms=4, label='Zero replacement')],
               loc='upper left', bbox_to_anchor=(.06, .98), ncol=2, frameon=False, fontsize=7.5)
    fig.text(.06, .07, 'All contrasts are full OPERA minus the altered prediction. Positive values favour full OPERA.\n'
             'Same frozen checkpoint and matched patients. Paired patient-bootstrap intervals.', fontsize=7,
             color=PALETTE['ink_secondary'], va='top', linespacing=1.5)
    fig.text(.98, .018, mark, fontsize=6.5, color=PALETTE['ink_muted'], ha='right', va='bottom')
    return fig


def render_report(summary, output, *, mark='TUNING SET — not test results'):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from opera.visualization.style import save_fig, setup_style
    setup_style()
    joint_path = output / 'joint_path_summaries.csv'
    joint_summary = pd.read_csv(joint_path) if joint_path.is_file() else None
    figures = [('router_mechanism', figure_router_mechanism(summary, joint_summary, metric='auc', mark=mark)),
               ('router_mechanism_probability', figure_router_mechanism(summary, joint_summary, metric='ipa', mark=mark)),
               ('router_mechanism_sensitivity', figure_router_mechanism_sensitivity(summary, mark=mark))]
    for name, fig in figures:
        save_fig(fig, str(output / (name + '.png')))
        plt.close(fig)
    data = summary.loc[(summary.stratum == 'all_cells') & (summary.horizon_days == 0)]
    if 'weighting' in data:
        data = data.loc[data.weighting == 'equal_cell']
    table = data.loc[data.metric.isin(['auc', 'brier', 'ipa']),
                     ['variant', 'metric', 'gain', 'lower', 'upper', 'n_cells']]
    ladder = output / 'trained_control_ladder.csv'
    ladder_html = ('<h2>Separately trained control ladder</h2><p>Descriptive common-roster means. '
        'These are different fits, not the fixed-checkpoint interventions above.</p>'
        + pd.read_csv(ladder).to_html(index=False, float_format=lambda v: f'{v:.5f}')) if ladder.is_file() else ''
    joint_path = output / 'joint_path_summaries.csv'
    joint_html = ''
    if joint_path.is_file():
        joint = pd.read_csv(joint_path)
        joint = joint.loc[(joint.weighting == 'equal_cell') & (joint.stratum == 'all_cells')
            & (joint.horizon_days == 0) & joint.metric.isin(['auc', 'brier', 'ipa']),
            ['contrast', 'metric', 'gain', 'lower', 'upper', 'n_cells']]
        joint_html = ('<h2>Where the advantage over joint appears</h2><p>Matched comparison with J_L. '
            'Encoder/head and global-only paths belong to the trained router checkpoint. '
            'Positive gains favour the router path over joint.</p>'
            + joint.to_html(index=False, float_format=lambda v: f'{v:.5f}'))
    (output / 'index.html').write_text('<!doctype html><meta charset="utf-8"><title>Frozen router audit</title>'
        '<style>body{font:16px system-ui;color:#233343;max-width:1100px;margin:40px auto;padding:0 24px}'
        'img{width:100%;height:auto}h1,h2{font-weight:500}h2{margin-top:44px}p{max-width:850px;line-height:1.6}'
        'td,th{padding:7px;text-align:left;border-bottom:1px solid #e8edf1}table{border-collapse:collapse;font-size:13px}'
        'details{overflow:auto;margin:24px 0}summary{cursor:pointer;padding:12px 0;color:#2d3a8c}</style>'
        '<h1>What does OPERA add to joint learning?</h1><p>' + html.escape(mark) + '</p>'
        '<p>The first panel locates the gain over joint learning. The next two test which corrections '
        'help and whether patients benefit from individual allocation. All altered paths retain the fitted parameters.</p>'
        '<img src="router_mechanism.png" alt="Where the gain appears, useful corrections and patient-specific gates">'
        '<h2>Does the same pattern hold for probability accuracy?</h2>'
        '<p>The same contrasts shown as IPA differences. Higher IPA means lower Brier error relative to the reference prediction.</p>'
        '<img src="router_mechanism_probability.png" alt="The three mechanism questions evaluated with IPA">'
        '<h2>All interventions and sensitivity checks</h2>'
        '<p>Filled diamonds show mean replacement. Open circles show zero replacement. '
        'The remaining rows use the indicated fixed path or gates.</p>'
        '<img src="router_mechanism_sensitivity.png" alt="AUC and IPA sensitivity results. Raw Brier differences are retained in the tables.">'
        '<details><summary>Numerical estimates and matched comparisons</summary>'
        + table.to_html(index=False, float_format=lambda v: f'{v:.5f}')
        + joint_html + ladder_html + '</details><details><summary>Definitions and interpretation</summary>'
        '<p>Reference means and fitted parameters are held fixed in the patient bootstrap. '
        'These interventions test reliance of this fit. They do not estimate the performance of a retrained static model.</p><ul>' + ''.join(
            f'<li><b>{html.escape(k)}</b>: {html.escape(v)}</li>' for k, v in VARIANTS.items()) + '</ul></details>')
    figure_names = [name + suffix for name, _ in figures for suffix in ('.pdf', '.png')]
    (output / 'figure_render.json').write_text(json.dumps(dict(
        source_sha256=file_hash(__file__), inputs={p.name: file_hash(p) for p in
            [output / 'effect_summaries.csv', joint_path] if p.is_file()},
        figure_sha256={name: file_hash(output / name) for name in figure_names}, mark=mark), indent=2)+'\n')


def package_review(output):
    included = ['report.json', 'component_head_effects.csv', 'estimates.csv', 'contrasts.csv',
                'contrast_coverage.csv', 'effect_summaries.csv', 'risk_changes.csv',
                'index.html', 'figure_render.json']
    included += [name + suffix for name in ('router_mechanism', 'router_mechanism_probability',
        'router_mechanism_sensitivity') for suffix in ('.pdf', '.png')]
    for name in ('trained_control_ladder.csv', 'borrowed_weights_by_task.csv',
                 'borrowed_weights_by_disease.csv'):
        if (output / name).is_file():
            included.append(name)
    included += [p.name for p in sorted(output.glob('joint_path_*.csv'))]
    with zipfile.ZipFile(output / 'router_mechanism_review.zip', 'w', zipfile.ZIP_DEFLATED) as bundle:
        for name in included:
            bundle.write(output / name, arcname=name)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--analysis-root', type=Path)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--render-only', action='store_true',
                        help='Refresh figures/HTML/ZIP from an existing audit without recomputing metrics')
    parser.add_argument('--vectors', type=Path, help='Optional selected-fit tuning pooled-vector NPZ')
    parser.add_argument('--outcomes', nargs='+', default=['all'])
    parser.add_argument('--horizons', nargs='+', type=int)
    parser.add_argument('--seed', type=int, default=42, help='Selected training seed')
    parser.add_argument('--model', help='Audit this declared tuning alias (e.g. R_PP_L) instead of '
                                        'the analysis OPERA model')
    parser.add_argument('--checkpoint', type=Path, help='With --model: checkpoint, if not in the '
                                                        'standard neural_runs layout')
    parser.add_argument('--bootstrap', type=int, default=400)
    parser.add_argument('--bootstrap-seed', type=int, default=2026)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--threads', type=int, default=8)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if args.render_only:
        summary = pd.read_csv(output / 'effect_summaries.csv')
        render_report(summary, output)
        package_review(output)
        print(f'Rendered existing results: {output / "router_mechanism_review.zip"}', flush=True)
        return
    if args.analysis_root is None:
        parser.error('--analysis-root is required unless --render-only is used')
    if args.bootstrap < 50 or args.batch_size < 1 or args.threads < 1:
        parser.error('Need bootstrap >=50, positive batch size and CPU thread count')
    if output.exists():
        raise FileExistsError('Use a fresh mechanism-audit output directory')
    root = args.analysis_root.resolve()
    analysis, spec, fit = selected_inputs(root, args.seed, args.model, args.checkpoint)
    torch.set_num_threads(args.threads)
    head = FrozenRouterHead.from_checkpoint(Path(fit['checkpoint'])).eval()
    ids, vectors, vector_path, marker_path = load_vectors(root, fit, args.vectors)
    baseline, patient, fine, eligible, outcomes, horizons, row_index = prepare_population(
        analysis, spec, head, ids, vectors, outcomes=args.outcomes, horizons=args.horizons)
    if patient.shape[1] != head.readout.hidden_size:
        raise ValueError('Saved encoder-vector width differs from the selected readout')
    output.mkdir(parents=True)
    report = dict(status='running', split='tuning', training_seed=args.seed,
        audited_model=args.model or analysis.get('opera_model', 'R_L'),
        checkpoint=fit['checkpoint'], checkpoint_sha256=fit['checkpoint_sha256'],
        vectors=str(vector_path), vector_marker=str(marker_path), n_patients=len(patient),
        n_prediction_rows=len(baseline), outcomes=outcomes, horizons_days=horizons,
        variants=VARIANTS, bootstrap_method='multiplier', bootstrap_replicates=args.bootstrap,
        bootstrap_seed=args.bootstrap_seed,
        interpretation='Frozen-checkpoint reliance, not retrained or causal model-part comparisons. '
            'Within-task reference means use unlabelled eligible tuning patients. '
            'Patient intervals condition on fitted parameters and those reference means.',
        source_files={name: file_hash(Path(__file__).resolve().parents[2] / name)
            for name in ('opera/diagnostics/router_mechanism.py', 'opera/run/audit_frozen_router_mechanism.py')})
    def save():
        (output / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    save()
    try:
        # Lock amendment 2026-10-30: two-sided borrowed-mixture health for every
        # router, and the learned pooling quantities for the partial-pooling one.
        health_indices = torch.tensor([head.outcomes.index(o) for o in outcomes])
        report['borrowed_mixture_health'] = borrowed_mixture_health(
            head, patient, fine, eligible, health_indices)
        if hasattr(head.readout, 'pooling_summary'):
            report['partial_pooling'] = head.readout.pooling_summary()
        save()
        print(f'Reconstructing fixed readout for {len(patient):,} tuning patients and {len(outcomes)} outcomes', flush=True)
        indices = torch.tensor([head.outcomes.index(o) for o in outcomes])
        predictions, reference, error = infer_interventions(
            head, patient, fine, eligible, indices, horizons, batch_size=args.batch_size)
        replay = predictions['FULL'][row_index]
        original = baseline.probability.to_numpy(float)
        close = np.isclose(replay, original, atol=1e-5, rtol=1e-4)
        report['borrowed_weights'], tasks = gate_tables(head, reference, outcomes, output)
        if not tasks.empty and analysis.get('cell_metadata'):
            report['route_concentration'] = route_concentration(
                tasks, read_table(analysis['cell_metadata']))
        report['reconstruction'] = dict(passed=bool(close.all()), n_rows=len(close),
            mismatched_rows=int((~close).sum()), max_probability_error=float(abs(replay-original).max()),
            max_head_algebra_error=error, atol=1e-5, rtol=1e-4)
        save()
        if not close.all():
            raise ValueError('Saved vectors/head do not reconstruct selected predictions; inspect report.json')
        head_effect_table(head, reference, outcomes).to_csv(output / 'component_head_effects.csv', index=False)
        torch.save(dict(outcomes=outcomes, fine_names=list(head.readout.fine_names),
            checkpoint_sha256=fit['checkpoint_sha256'], counts=reference.count,
            gates=reference.gates, borrowed_composition=reference.composition,
            component_head_means=reference.terms, split='tuning'), output / 'references_INTERNAL.pt')
        baseline['model'] = 'FULL'
        baseline['probability'] = replay
        draws = patient_bootstrap(baseline, args.bootstrap, args.bootstrap_seed, method='multiplier')
        metadata = read_table(analysis['cell_metadata']) if analysis.get('cell_metadata') else (
            baseline[['cohort', 'outcome', 'horizon_days']].drop_duplicates())
        print('Paired joint comparison: full, router encoder/head, router global-only path', flush=True)
        report['joint_path_comparison'] = compare_joint_paths(
            analysis, baseline, predictions, row_index, draws, metadata, output, replicates=args.bootstrap)
        save()
        all_summary, risk_tables = [], []
        for i, variant in enumerate(VARIANTS):
            print(f'{i+1}/{len(VARIANTS)} paired metrics: {variant}', flush=True)
            candidate = baseline.assign(model=variant, probability=predictions.pop(variant)[row_index])
            risk_tables.append(risk_change_table(baseline, candidate, variant))
            contrast = 'FULL_MINUS_' + variant
            tables = paired_analysis(pd.concat([baseline, candidate], ignore_index=True),
                {contrast: f'gain(FULL,{variant})'}, replicates=args.bootstrap, draws=draws,
                patient_gain_contrasts=set(),
                min_cases=analysis.get('performance_min_cases', 5),
                min_controls=analysis.get('performance_min_controls', 5))
            for aggregation in ('equal_cell', 'disease_balanced'):
                _, summary = effect_summaries(tables['contrasts'], tables['bootstrap_draws'],
                    metadata, aggregation=aggregation)
                all_summary.append(summary.assign(variant=variant, weighting=aggregation))
            for name in ('estimates', 'contrasts', 'contrast_coverage'):
                path = output / (name + '.csv')
                table = tables[name].assign(intervention=variant)
                table.to_csv(path, mode='w' if i == 0 else 'a', header=i == 0, index=False)
            report['completed_interventions'] = list(VARIANTS)[:i+1]
            save()
        summary = pd.concat(all_summary, ignore_index=True)
        summary.to_csv(output / 'effect_summaries.csv', index=False)
        pd.concat(risk_tables, ignore_index=True).to_csv(output / 'risk_changes.csv', index=False)
        report['trained_controls'] = trained_control_ladder(
            root, args.seed, horizons, outcomes, output, ladder_for(args.model))
        render_report(summary.loc[summary.weighting == 'equal_cell'], output)
        report['status'] = 'complete'
        report['support'] = dict(min_cases=analysis.get('performance_min_cases', 5),
                                min_controls=analysis.get('performance_min_controls', 5))
        save()
        # Explicit aggregate allowlist. No IDs, pooled vectors, patient scores,
        # bootstrap patient weights or reference tensors enter the review ZIP.
        package_review(output)
        print(f'Complete: {output / "router_mechanism_review.zip"}', flush=True)
    except Exception as exc:
        report.update(status='failed', error=str(exc))
        save()
        raise


if __name__ == '__main__':
    main()
