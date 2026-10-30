"""What each part of the record adds: AUC with restricted inputs vs the full input.

Compares ``direct_cr_predictions.parquet`` from the full evaluation with
evaluations of the same checkpoint on restricted inputs
(``evaluate_direct_competing_risk --content-reference``, ``--hide-prefixes``,
``--keep-last-days``). For every outcome, the held-out AUC at ``horizon``
(pooled over diseases) is compared with paired patient-bootstrap intervals,
using the same resamples for every outcome and restriction so that macro
averages and domain averages have valid intervals. The model is not
retrained: this measures what the fitted model relies on.

    python -m opera.run.compare_input_restrictions \\
        --full runs/eval_full \\
        --restricted "Timeline only=runs/eval_timeline" \\
        --restricted "Without laboratory results=runs/eval_no_labs" \\
        --restricted "Last year only=runs/eval_last365" \\
        --output-dir analysis/input_restrictions
"""

from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path

import numpy as np
import pandas as pd

from opera.run.compare_approved_neural_tuning_predictions import fast_cif_auc

KEY = ["subject_id", "cohort", "outcome"]


def read_predictions(path):
    path = Path(path)
    if path.is_dir():
        path = path / "direct_cr_predictions.parquet"
    return pd.read_parquet(path)


def _auc(frame, column, horizon):
    # Re-estimate censoring for each resample, as in the original evaluator.
    # Weighted ranks avoid materialising every case/control pair or computing
    # the unused Brier score. An empty resample has no estimable AUC.
    if frame.empty:
        return float("nan")
    return fast_cif_auc(
        frame.time_days.to_numpy(float), frame.event.to_numpy(int),
        frame[column].to_numpy(float), horizon)[0]


def _compare_expanded(full, restricted, *, horizon=365, draws=200, seed=2026, domains=None,
            progress=None):
    """Per-outcome and macro AUC differences (restricted minus full)."""
    column = f"cif_{int(horizon)}d"
    patients = full.subject_id.unique()
    rng = np.random.default_rng(seed)
    counts = rng.multinomial(len(patients), np.full(len(patients), 1 / len(patients)),
                             size=draws)
    position = pd.Series(np.arange(len(patients)), index=patients)
    rows, replicate_rows = [], []
    for restriction_number, (name, frame) in enumerate(restricted.items(), 1):
        merged = full[KEY + ["time_days", "event", column]].merge(
            frame[KEY + [column]], on=KEY, suffixes=("_full", "_restricted"),
            validate="one_to_one")
        if len(merged) != len(full):
            raise ValueError(f"{name}: restricted predictions do not cover the full rows")
        groups_by_outcome = merged.groupby("outcome")
        for outcome_number, (outcome, part) in enumerate(groups_by_outcome, 1):
            label = (f"Restriction {restriction_number}/{len(restricted)} ({name}), "
                     f"outcome {outcome_number}/{len(groups_by_outcome)} ({outcome})")
            if progress:
                progress(f"{label}: starting {draws} paired bootstrap draws")
            full_auc = _auc(part, f"{column}_full", horizon)
            restricted_auc = _auc(part, f"{column}_restricted", horizon)
            index = position[part.subject_id].to_numpy()
            deltas = []
            for r in range(draws):
                repeat = counts[r, index]
                sample = part.loc[np.repeat(part.index.to_numpy(), repeat)]
                deltas.append(_auc(sample, f"{column}_restricted", horizon)
                              - _auc(sample, f"{column}_full", horizon))
                if progress and ((r + 1) % max(1, draws // 4) == 0 or r + 1 == draws):
                    progress(f"{label}: {r + 1}/{draws} draws complete")
            deltas = np.asarray(deltas, float)
            replicate_rows.append(pd.Series(deltas, name=(name, outcome)))
            rows.append({"restriction": name, "outcome": outcome, "auc_full": full_auc,
                         "auc_restricted": restricted_auc,
                         "delta": restricted_auc - full_auc,
                         "lower": np.nanquantile(deltas, 0.025),
                         "upper": np.nanquantile(deltas, 0.975),
                         "n_rows": len(part)})
    table = pd.DataFrame(rows)
    replicates = pd.DataFrame(replicate_rows)
    summary = []
    groups = {"All outcomes": None}
    if domains:
        groups.update({d: [o for o, g in domains.items() if g == d]
                       for d in dict.fromkeys(domains.values())})
    for name in restricted:
        for label, members in groups.items():
            part = table.loc[(table.restriction == name)
                             & (table.outcome.isin(members) if members else True)]
            if part.empty:
                continue
            reps = replicates.loc[[(name, o) for o in part.outcome]].mean(0)
            summary.append({"restriction": name, "group": label,
                            "n_outcomes": len(part), "mean_delta": part.delta.mean(),
                            "lower": np.nanquantile(reps, 0.025),
                            "upper": np.nanquantile(reps, 0.975),
                            "mean_auc_full": part.auc_full.mean(),
                            "mean_auc_restricted": part.auc_restricted.mean()})
    return table, pd.DataFrame(summary)


def _bootstrap_ipcw(times, events, counts, horizon):
    """The original re-estimated censoring KM, without expanding duplicate rows."""
    order = np.argsort(times, kind='stable')
    unique, inverse, sizes = np.unique(times, return_inverse=True, return_counts=True)
    starts = np.r_[0, np.cumsum(sizes)[:-1]]
    # reduceat needs values ordered by time; unique inverse refers to raw rows.
    mass = np.add.reduceat(counts[:, order], starts, axis=1)
    admin = np.add.reduceat(counts[:, order] * (events[order] == 0),
                           starts, axis=1)
    at_risk = mass.sum(1)[:, None] - (np.cumsum(mass, axis=1) - mass)
    fraction = np.divide(admin, at_risk, out=np.zeros_like(admin, dtype=float), where=at_risk > 0)
    after = np.cumprod(1 - fraction, axis=1)
    before = np.maximum(np.c_[np.ones(len(counts)), after[:, :-1]], 1e-6)
    h_index = np.searchsorted(unique, horizon, side='left') - 1
    g_horizon = np.ones(len(counts)) if h_index < 0 else np.maximum(after[:, h_index], 1e-6)
    case = (times <= horizon) & (events == 1)
    competing = (times <= horizon) & (events == 2)
    control = competing | (times > horizon) | ((times >= horizon) & (events == 0))
    g_time = before[:, inverse]
    cases = counts * case / g_time
    controls = counts * control / np.where(competing[None], g_time, g_horizon[:, None])
    return cases, controls


def _rank_auc(risk, cases, controls):
    """Weighted tied-rank AUC for every bootstrap replicate at once."""
    order = np.argsort(risk, kind='stable')
    _, starts = np.unique(risk[order], return_index=True)
    case_mass = np.add.reduceat(cases[:, order], starts, axis=1)
    control_mass = np.add.reduceat(controls[:, order], starts, axis=1)
    lower = np.cumsum(control_mass, axis=1) - control_mass
    numerator = np.sum(case_mass * (lower + .5 * control_mass), axis=1)
    denominator = case_mass.sum(1) * control_mass.sum(1)
    return np.divide(numerator, denominator, out=np.full(len(cases), np.nan), where=denominator > 0)


def compare(full, restricted, *, horizon=365, horizons=None, draws=200, seed=2026,
            domains=None, progress=None, engine='vectorized'):
    """Paired restriction effects across horizons, with one shared patient draw.

    Macro means keep a fixed, point-estimable outcome roster per horizon.
    Bootstrap means are undefined if any member is undefined in that draw.
    """
    horizons = list(horizons) if horizons is not None else [horizon]
    if (not horizons or len(set(horizons)) != len(horizons)
            or any(h <= 0 or int(h) != h for h in horizons) or draws < 1 or not restricted):
        raise ValueError('Need distinct positive integer horizons, draws and restrictions')
    if engine == 'expanded':
        tables = [_compare_expanded(full, restricted, horizon=h, draws=draws, seed=seed,
                                   domains=domains, progress=progress) for h in horizons]
        return tuple(pd.concat([t[i].assign(horizon_days=int(h)) for h, t in zip(horizons, tables)], ignore_index=True)
                     for i in (0, 1))
    if engine != 'vectorized':
        raise ValueError('Unknown restriction comparison engine')
    columns = [f'cif_{int(h)}d' for h in horizons]
    required = {*KEY, 'time_days', 'event', *columns}
    if not required <= set(full) or full.empty or full.duplicated(['subject_id', 'outcome']).any():
        raise ValueError('Need one full prediction per patient/outcome and every requested horizon')
    if (not np.isfinite(full[['time_days', 'event', *columns]].to_numpy(float)).all()
            or not full.event.isin([0, 1, 2]).all() or (full.time_days < 0).any()
            or ('split' in full and full.split.nunique() != 1)):
        raise ValueError('Invalid full follow-up, risks or mixed evaluation splits')
    aligned = {}
    for name, frame in restricted.items():
        if not required <= set(frame) or frame.duplicated(KEY).any():
            raise ValueError(f'{name}: missing horizon/follow-up fields or duplicate predictions')
        merged = full[KEY + ['time_days', 'event']].merge(frame, on=KEY, suffixes=('_full', ''), validate='one_to_one')
        if (len(merged) != len(full) or len(frame) != len(full)
                or not np.array_equal(merged.time_days_full, merged.time_days)
                or not np.array_equal(merged.event_full, merged.event)
                or not np.isfinite(merged[columns].to_numpy(float)).all()
                or ('split' in full and ('split' not in merged or not merged.split.eq(full['split'].iloc[0]).all()))):
            raise ValueError(f'{name}: restricted predictions do not cover identical patients, labels and split')
        aligned[name] = merged[columns].to_numpy(float)
    patients = full.subject_id.unique()
    counts = np.random.default_rng(seed).multinomial(len(patients), np.full(len(patients), 1 / len(patients)), size=draws)
    position = pd.Series(np.arange(len(patients)), index=patients)
    rows, replicate = [], {}
    groups = list(full.groupby('outcome', sort=True).indices.items())
    for h_index, h in enumerate(horizons):
        for o_index, (outcome, index) in enumerate(groups, 1):
            part = full.iloc[index]
            times, events = part.time_days.to_numpy(float), part.event.to_numpy(int)
            local_counts = counts[:, position[part.subject_id].to_numpy()]
            # First row is the point estimate. Batching bounds temporary memory.
            all_counts = np.vstack([np.ones(len(part), dtype=int), local_counts])
            aucs = {name: [] for name in ['__full__', *restricted]}
            scores = {'__full__': part[columns[h_index]].to_numpy(float),
                      **{name: values[index, h_index] for name, values in aligned.items()}}
            for start in range(0, draws + 1, 128):
                cases, controls = _bootstrap_ipcw(times, events, all_counts[start:start + 128], h)
                for name, risk in scores.items():
                    aucs[name].append(_rank_auc(risk, cases, controls))
            aucs = {name: np.concatenate(values) for name, values in aucs.items()}
            for r_index, name in enumerate(restricted, 1):
                delta = aucs[name] - aucs['__full__']
                replicate[(int(h), name, outcome)] = delta[1:]
                valid = np.isfinite(delta[1:])
                ci = np.quantile(delta[1:][valid], [.025, .975]) if valid.mean() >= .8 else [np.nan, np.nan]
                rows.append(dict(restriction=name, outcome=outcome, horizon_days=int(h),
                    clinical_family=domains.get(outcome, 'Unmapped outcomes') if domains else None,
                    auc_full=aucs['__full__'][0], auc_restricted=aucs[name][0], delta=delta[0],
                    lower=ci[0], upper=ci[1], n_rows=len(part), valid_draws=int(valid.sum()),
                    n_cases=int(((events == 1) & (times <= h)).sum()),
                    n_controls=int(((events > 1) | (times >= h)).sum() - ((events == 1) & (times == h)).sum())))
                if progress:
                    progress(f'Horizon {int(h)}d, Restriction {r_index}/{len(restricted)} ({name}), '
                             f'outcome {o_index}/{len(groups)} ({outcome}): {draws}/{draws} draws complete')
    table = pd.DataFrame(rows)
    summary = []
    memberships = {'All outcomes': None}
    if domains:
        memberships.update({name: [o for o, d in domains.items() if d == name] for name in dict.fromkeys(domains.values())})
        unmapped = sorted(set(full.outcome) - set(domains))
        if unmapped:
            memberships['Unmapped outcomes'] = unmapped
    for h in horizons:
        for name in restricted:
            base = table.loc[table.horizon_days.eq(h) & table.restriction.eq(name)]
            for family, members in memberships.items():
                part = base.loc[base.outcome.isin(members)] if members is not None else base
                good = part.loc[np.isfinite(part.delta)]
                if part.empty:
                    continue
                values = np.stack([replicate[(int(h), name, o)] for o in good.outcome]) if len(good) else np.full((1, draws), np.nan)
                means = values.mean(0)
                valid = np.isfinite(means)
                ci = np.quantile(means[valid], [.025, .975]) if valid.mean() >= .8 else [np.nan, np.nan]
                summary.append(dict(restriction=name, group=family, horizon_days=int(h),
                    n_outcomes=len(good), n_requested_outcomes=len(part), valid_draws=int(valid.sum()),
                    mean_delta=good.delta.mean(), lower=ci[0], upper=ci[1],
                    mean_auc_full=good.auc_full.mean(), mean_auc_restricted=good.auc_restricted.mean()))
    return table, pd.DataFrame(summary)


def write_outputs(table, summary, output):
    """Preserve one-year consumers instead of silently averaging all horizons."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    table.to_csv(output / 'input_restriction_auc_by_horizon.csv', index=False)
    summary.to_csv(output / 'input_restriction_summary_by_horizon.csv', index=False)
    selected = 365 if 365 in set(summary.horizon_days) else int(summary.horizon_days.min())
    table.loc[table.horizon_days.eq(selected)].to_csv(output / 'input_restriction_auc.csv', index=False)
    summary.loc[summary.horizon_days.eq(selected)].to_csv(output / 'input_restriction_summary.csv', index=False)


def figure_recipe(summary_path, horizons, mark=None):
    """One overview plus readable two-horizon clinical-family pages."""
    table = {'restrictions': str(Path(summary_path).resolve())}
    figures = [dict(id='supp_recent_record_gaps', function='figure_recent_record_gap_horizons',
                    tables=table, options=dict(mark=mark))]
    for page, start in enumerate(range(0, len(horizons), 2), 1):
        figures.append(dict(id=f'supp_recent_record_gaps_families_{page}',
            function='figure_recent_record_gaps', tables=table,
            options=dict(horizons=list(horizons[start:start + 2]), mark=mark)))
    return {'figures': figures}


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--full", type=Path, required=True)
    parser.add_argument("--restricted", action="append", required=True,
                        help="label=path to a restricted evaluation (repeatable)")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--horizon", type=int, help='Single horizon (legacy shorthand)')
    group.add_argument('--horizons', nargs='+', type=int, help='Horizons sharing the same patient bootstrap')
    parser.add_argument("--draws", type=int, default=200)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    from opera.visualization.paper_taxonomy import outcome_endpoint_groups

    restricted = {}
    for item in args.restricted:
        label, _, path = item.partition("=")
        if not path:
            raise ValueError("--restricted takes label=path")
        restricted[label] = read_predictions(path)
    table, summary = compare(read_predictions(args.full), restricted, horizons=args.horizons or [args.horizon or 365],
                             draws=args.draws, seed=args.seed,
                             domains=outcome_endpoint_groups(),
                             progress=lambda message: print(
                                 f"{datetime.now().isoformat(timespec='seconds')} {message}",
                                 flush=True))
    write_outputs(table, summary, args.output_dir)
    print(json.dumps(summary.loc[summary.group == "All outcomes"].to_dict("records"),
                     indent=2, default=float))


if __name__ == "__main__":
    main()
