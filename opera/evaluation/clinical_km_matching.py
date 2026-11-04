"""Descriptive evaluation-cohort KM matching, distinct from frozen cutoffs.

Each end is a separate binary selection, not a partition into risk groups.
All tied scores stay together. No survival tolerance or monotonic smoothing is
applied: every supported empirical boundary is searched. Bootstrap samples
re-estimate the clinical target and repeat that entire search.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from opera.evaluation.clinical_risk_groups import km_survival


def _prefix_curve(risk, time, event, horizon, *, high=False):
    order = np.argsort(-risk if high else risk, kind='stable')
    r, t, e = risk[order], time[order], event[order]
    event_times = np.unique(t[(e == 1) & (t <= horizon)])
    survival = np.ones(len(r))
    previous_risk = np.zeros(len(event_times))
    previous_events = np.zeros(len(event_times))
    for start in range(0, len(r), 128):
        stop = min(start + 128, len(r))
        y = (t[start:stop, None] >= event_times).cumsum(0) + previous_risk
        d = ((t[start:stop, None] == event_times) & (e[start:stop, None] == 1)).cumsum(0) + previous_events
        survival[start:stop] = np.prod(1 - np.divide(d, y, out=np.zeros_like(d, float), where=y > 0), axis=1)
        if len(event_times):
            previous_risk, previous_events = y[-1], d[-1]
    ends = np.r_[np.flatnonzero(r[:-1] != r[1:]) + 1, len(r)]
    return pd.DataFrame(dict(model_n=ends, selected_fraction=ends / len(r),
        cutoff=r[ends - 1], model_survival=survival[ends - 1],
        model_at_risk_horizon_n=np.cumsum(t >= horizon)[ends - 1]))


def _point(patients, spec, minimum_group, minimum_at_risk):
    risk = patients.model_risk.to_numpy(float)
    time = patients.time_days.to_numpy(float)
    event = patients.event.to_numpy()
    if (not len(risk) or not np.isfinite(risk).all() or not np.isfinite(time).all()
            or (time < 0).any() or not np.isin(event, [0, 1]).all()):
        raise ValueError('KM matching requires finite aligned risks/follow-up and death-inclusive event 0/1')
    horizon = spec['horizon_days']
    rows, curves = [], []
    for end, name in [('low', spec['group_order'][0]), ('high', spec['group_order'][-1])]:
        native = patients.ipi_group.eq(name).to_numpy()
        target = km_survival(time[native], event[native], horizon)
        curve = _prefix_curve(risk, time, event, horizon, high=end == 'high')
        curve['supported'] = (curve.model_n >= minimum_group) & (curve.model_at_risk_horizon_n >= minimum_at_risk)
        curve['meets_target'] = (curve.model_survival >= target if end == 'low' else curve.model_survival <= target)
        curve['end'] = end
        curve['index_survival'] = target
        curve['index_n'] = int(native.sum())
        native_supported = native.sum() >= minimum_group and ((time >= horizon) & native).sum() >= minimum_at_risk
        feasible = curve.loc[curve.supported & curve.meets_target] if native_supported else curve.iloc[:0]
        row = dict(end=end, index_group=name, index_n=int(native.sum()), index_survival=target,
            index_at_risk_horizon_n=int(((time >= horizon) & native).sum()),
            n_patients=len(patients), feasible_boundaries=len(feasible),
            status='estimable' if len(feasible) else 'insufficient_native_followup' if not native_supported else 'no_supported_match')
        if len(feasible):
            chosen = feasible.iloc[-1]
            row.update(model_n=int(chosen.model_n), model_survival=float(chosen.model_survival),
                cutoff=float(chosen.cutoff), model_at_risk_horizon_n=int(chosen.model_at_risk_horizon_n),
                additional_n=int(chosen.model_n) - int(native.sum()),
                additional_per_1000=1000 * (chosen.model_n - native.sum()) / len(patients))
        rows.append(row)
        curves.append(curve)
    return rows, pd.concat(curves, ignore_index=True)


def cohort_km_matching(patients, spec, *, draws=200, seed=2026,
                       minimum_group=20, minimum_at_risk=10, progress=None):
    """Largest low/high risk prefixes matching observed index KM at one horizon.

    Low: S_model >= S_index. High: S_model <= S_index. Zero tolerance. Risk
    boundaries include their entire tie block (<= low cutoff, >= high cutoff).
    At least 80% of requested replicates must match for a percentile interval.
    Valid-draw counts are always exported; intervals can be unstable near an
    empirical crossing and do not remove cutoff-selection optimism.
    """
    if not spec.get('death_in_endpoint'):
        raise ValueError('Use CIF matching, not KM matching, for competing-risk endpoints')
    if (not isinstance(draws, int) or draws < 1 or minimum_group < 1
            or minimum_at_risk < 1 or spec['horizon_days'] <= 0):
        raise ValueError('Positive draws, horizon and support settings required')
    rows, curve = _point(patients, spec, minimum_group, minimum_at_risk)
    samples = {'low': [], 'high': []}
    rng = np.random.default_rng(seed)
    for replicate in range(draws):
        sampled = patients.iloc[rng.integers(0, len(patients), len(patients))]
        estimates, _ = _point(sampled, spec, minimum_group, minimum_at_risk)
        for estimate in estimates:
            samples[estimate['end']].append(estimate)
        if progress and (replicate + 1) % max(1, draws // 4) == 0:
            progress(f'{replicate + 1}/{draws} paired evaluation-KM matching draws complete')
    for row in rows:
        sample = pd.DataFrame(samples[row['end']])
        valid = sample.status.eq('estimable')
        row.update(score_id=spec['id'], horizon_days=spec['horizon_days'],
            analysis_role='descriptive_evaluation_cohort_operating_point',
            grouping_uses_future_outcomes=True, grouping_rule='largest_tie_preserving_prefix',
            match_tolerance=0., cutoff_inclusive=True,
            minimum_group=minimum_group, minimum_at_risk=minimum_at_risk,
            bootstrap_draws=draws, bootstrap_seed=seed, valid_matching_draws=int(valid.sum()),
            matching_draw_fraction=float(valid.mean()),
            bootstrap_rule='paired_patients_reestimate_index_KM_and_reselect_model_prefix',
            interval_caution='Pointwise percentile intervals; unstable crossings and zero-event degeneracy possible; no deployment or non-inferiority claim')
        for field in ['additional_n', 'additional_per_1000', 'model_n', 'model_survival', 'index_survival']:
            # Index survival remains estimable in draws even when a model match is absent.
            eligible = sample[field].dropna() if field in sample else pd.Series(dtype=float)
            if field != 'index_survival':
                eligible = sample.loc[valid, field].dropna() if field in sample else eligible
            enough = (len(eligible) >= .8 * draws and
                      (row['status'] == 'estimable' or field == 'index_survival'))
            lo, hi = np.quantile(eligible, [.025, .975]) if enough else [np.nan, np.nan]
            row[field + '_lower'], row[field + '_upper'] = float(lo), float(hi)
        row['lower'], row['upper'] = row['additional_per_1000_lower'], row['additional_per_1000_upper']
    curve['score_id'] = spec['id']
    curve['horizon_days'] = spec['horizon_days']
    return pd.DataFrame(rows), curve
