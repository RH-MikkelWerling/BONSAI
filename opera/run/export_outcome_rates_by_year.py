"""Frozen-label yearly incidence audit; aggregates only, no model inference.

The optional source specification maps actual raw source columns explicitly;
no diagnosis position or LPR version is inferred from dates.
"""
import argparse
import hashlib
import json
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED
import numpy as np
import pandas as pd
import yaml


def canonical_patient_ids(values):
    """Exact integer ID keys, matching frozen binarisation's integer namespace."""
    from decimal import Decimal, InvalidOperation
    import numbers
    def one(value):
        if pd.isna(value) or isinstance(value, (bool, np.bool_)):
            raise ValueError('Missing or boolean patient ID')
        if isinstance(value, numbers.Real) and not isinstance(value, numbers.Integral):
            if not np.isfinite(value) or abs(value) >= 2**53:
                raise ValueError('Floating patient ID exceeds exact-integer safety range')
        try:
            number=Decimal(str(value))
        except InvalidOperation:
            raise ValueError('Frozen patient IDs must be exact integers') from None
        if not number.is_finite() or number != number.to_integral_value():
            raise ValueError('Non-integral patient ID')
        return str(int(number))
    return values.map(one)


def incidence(time, event, horizon=365):
    """Aalen-Johansen incidence and delta-method logit 95% interval, tied events.

    Propagates the survival/incidence covariance using multinomial cause hazards.
    Death is cause 2, administrative censoring 0; all ties share the risk set.
    """
    t, e = np.asarray(time, float), np.asarray(event, int)
    if not np.isfinite(t).all() or (t < 0).any() or not np.isin(e, [0, 1, 2]).all():
        raise ValueError('Invalid frozen survival labels')
    if len(t) == 0: return np.nan, np.nan, np.nan
    s, f, v = 1., 0., np.zeros((2, 2))
    for day in np.unique(t[(e != 0) & (t <= horizon)]):
        n = (t >= day).sum()
        h = np.array([((t == day) & (e == 1)).sum(), ((t == day) & (e == 2)).sum()]) / n
        a = np.array([[1-h.sum(), 0], [h[0], 1]])
        b = np.array([[-s, -s], [s, 0]])
        v = a @ v @ a.T + b @ ((np.diag(h)-np.outer(h, h))/n) @ b.T
        f += s*h[0]; s *= 1-h.sum()
    if t.max() < horizon and s > 0:
        return np.nan, np.nan, np.nan  # No nonparametric support to the requested horizon.
    if 0 < f < 1:
        se = np.sqrt(max(0., v[1, 1])) / (f*(1-f))
        z = np.log(f/(1-f)) + np.array([-1, 1])*1.959963984540054*se
        lower, upper = 1/(1+np.exp(-np.clip(z, -700, 700)))
    else: lower = upper = f
    return f, lower, upper


def summarize(frame, group_columns):
    rows = []
    for key, group in frame.groupby(group_columns, dropna=False, sort=True):
        if not isinstance(key, tuple): key = (key,)
        n = len(group); events = int(((group.event == 1) & (group.time_days <= 365)).sum())
        row = dict(zip(group_columns, key))
        # Suppress the whole row's measures, preventing CI/denominator leakage.
        row['suppressed'] = n < 5 or events < 5
        if row['suppressed']:
            row.update(n_patients=None, events_365d=None, censored_before_365d=None,
                       competing_deaths_365d=None, incidence_365d=None, lower=None, upper=None)
        else:
            f, lo, hi = incidence(group.time_days, group.event)
            censored = int(((group.event == 0) & (group.time_days < 365)).sum())
            deaths = int(((group.event == 2) & (group.time_days <= 365)).sum())
            # Small secondary counts are also withheld.
            row.update(n_patients=n, events_365d=events,
                       censored_before_365d=censored if censored == 0 or censored >= 5 else None,
                       competing_deaths_365d=deaths if deaths == 0 or deaths >= 5 else None,
                       incidence_365d=f, lower=lo, upper=hi)
        rows.append(row)
    return pd.DataFrame(rows)


def frozen_outcomes(bundle_path, metadata_path, outcome_names=None):
    from bonsai.functional.outcomes import binarize_outcomes
    from opera.functional.outcomes import filter_registry_eligible_outcomes, resolve_registry_start_date
    from opera.functional.outcomes import filter_outcome_eligibility
    from opera.modules.datamodules.MultiCohortContrastiveDataModule import _outcome_path, _competing_path, _eligibility_path, _read_population
    from opera.run.finish_figure_inputs import sha
    bundle_path = Path(bundle_path)
    bundle = json.loads((bundle_path / 'bundle.json').read_text())
    row, = [r for r in bundle['runs'] if r['name'] == 'seed42/EA/J']
    config_path = bundle_path / row['config_path']
    if sha(config_path) != row['config_sha256']: raise ValueError('Frozen fit config changed')
    config = yaml.safe_load(config_path.read_text())
    meta = pd.read_parquet(metadata_path)
    meta['subject_id'] = canonical_patient_ids(meta.subject_id)
    if meta.subject_id.duplicated().any(): raise ValueError('Duplicate metadata patients')
    meta = meta.loc[meta.split.isin(['train', 'tuning', 'held_out'])].set_index('subject_id')
    outcomes = config['outcomes']
    for name, ocfg in outcomes.items():
        if outcome_names is not None and name not in outcome_names: continue
        parts = []
        audit = []
        for cohort, ccfg in config['cohorts'].items():
            if name in ccfg.get('exclude_outcomes', []): continue
            data_dir = ccfg['data_dir']
            path = _outcome_path(data_dir, ocfg)
            frame = pd.read_parquet(path)
            population = _read_population(ccfg, data_dir)
            # Parquet outcomes and CSV population files can store the same IDs
            # as text vs integers. Compare canonical text, retaining original
            # frame types for the frozen eligibility/binarisation machinery.
            source_n = len(frame)
            frame = frame.loc[canonical_patient_ids(frame.subject_id).isin(canonical_patient_ids(population.subject_id))].copy()
            membership_n = len(frame)
            frame = filter_outcome_eligibility(frame, _eligibility_path(data_dir, ocfg),
                       cohort=cohort, outcome_name=name, eligibility_scope=config.get('eligibility_scope', 'ascertainment'))
            frame = filter_registry_eligible_outcomes(frame, resolve_registry_start_date(ccfg, ocfg))
            frame = frame.loc[canonical_patient_ids(frame.subject_id).isin(meta.index)].copy()
            competing_path = _competing_path(data_dir, ocfg)
            competing = pd.read_parquet(competing_path) if competing_path else None
            records = binarize_outcomes(frame, ocfg['n_hours_start_include'], ocfg.get('n_hours_end_include'),
                                       require_min_followup=False, competing_event_df=competing)
            dates = frame.assign(subject_id=canonical_patient_ids(frame.subject_id)).set_index('subject_id').index_date
            audit.append(dict(cohort=cohort,source_rows=source_n,population_rows=len(population),
                              membership_rows=membership_n,eligible_metadata_rows=len(frame),records=len(records),
                              source_id_dtype=str(frame.subject_id.dtype),population_id_dtype=str(population.subject_id.dtype)))
            part = pd.DataFrame([dict(subject_id=str(k), time_days=v['time_days'], event=v['event']) for k,v in records.items()])
            if part.empty: continue
            part['index_year'] = pd.to_datetime(part.subject_id.map(dates), errors='raise').dt.year
            part['region'] = part.subject_id.map(meta['predominant_region']).fillna('Not recorded')
            parts.append(part)
        if not parts:
            raise ValueError(f'No frozen outcome records: {name}; cohort row audit: {json.dumps(audit)}')
        frame = pd.concat(parts, ignore_index=True)
        if frame.subject_id.duplicated().any(): raise ValueError(f'Overlapping cohort patients: {name}')
        frame['outcome'] = name
        yield frame


def source_audit(spec_path, output):
    spec = yaml.safe_load(Path(spec_path).read_text())
    ends = []
    for name, entry in spec['sources'].items():
        path = Path(entry['path'])
        frame = pd.read_parquet(path, columns=[entry['date_column']])
        dates = pd.to_datetime(frame[entry['date_column']], errors='raise')
        ends.append(dict(source=name, last_event_date=dates.max().isoformat() if dates.notna().any() else None))
    pd.DataFrame(ends).to_csv(output / 'source_end_dates.csv', index=False)
    entry = spec['sepsis']
    columns = [entry[k] for k in ['date_column','code_column','position_column','contact_version_column']]
    frame = pd.read_parquet(entry['path'], columns=list(set(columns)))
    code = frame[entry['code_column']].astype(str).str.upper().str.replace('.', '', regex=False)
    frame = frame.loc[code.str.startswith(tuple(entry['code_prefixes']))].copy()
    frame['event_year'] = pd.to_datetime(frame[entry['date_column']], errors='raise').dt.year
    frame['diagnosis_position'] = frame[entry['position_column']].map(entry['position_map'])
    frame['contact_version'] = frame[entry['contact_version_column']].map(entry['contact_version_map'])
    if frame[['diagnosis_position','contact_version','event_year']].isna().any().any():
        raise ValueError('Unmapped sepsis source values; supply explicit mappings')
    counts = frame.groupby(['event_year','diagnosis_position','contact_version']).size().rename('n_records').reset_index()
    counts['suppressed'] = counts.n_records < 5
    counts.loc[counts.suppressed, 'n_records'] = np.nan
    counts.to_csv(output / 'sepsis_event_calendar_year.csv', index=False)


def run(bundle, metadata, output, source_spec=None):
    from opera.run.finish_figure_inputs import sha
    output = Path(output)
    if output.exists(): raise FileExistsError('Use a fresh output directory')
    output.mkdir(parents=True)
    yearly, regional = [], []
    for frame in frozen_outcomes(bundle, metadata):
        yearly.append(summarize(frame, ['outcome','index_year']))
        if any(s in frame.outcome.iloc[0] for s in ['sepsis','serious_infection','antibiotic','antimicrobial']):
            regional.append(summarize(frame, ['outcome','index_year','region']))
        print(f'Yearly audit: {frame.outcome.iloc[0]}', flush=True)
    if len(yearly) != 69:
        raise ValueError(f'Expected 69 frozen outcomes, found {len(yearly)}; inspect selected config')
    pd.concat(yearly).to_csv(output / 'outcome_rates_by_index_year.csv', index=False)
    if regional: pd.concat(regional).to_csv(output / 'infection_rates_by_region_year.csv', index=False)
    gaps = []
    if source_spec: source_audit(source_spec, output)
    else: gaps.append('Raw source date and sepsis diagnosis-position/LPR-version mappings not supplied')
    report = dict(status='complete_with_gaps' if gaps else 'complete', gaps=gaps,
                  estimand='365-day Aalen-Johansen cumulative incidence; death competing; early censoring retained',
                  uncertainty='Delta-method survival/incidence covariance; logit 95% interval',
                  metadata_sha256=sha(metadata), source_bundle_sha256=sha(Path(bundle)/'bundle.json'),
                  region_definition='Pre-index predominant-contact region from frozen patient metadata; not event hospital region',
                  outcomes=len(yearly), scope='Frozen labels; no fitting, inference or analysis-lock changes')
    (output / 'manifest.json').write_text(json.dumps(report, indent=2))
    target = output / 'download_outcome_rates_by_year.zip'
    with ZipFile(target, 'x', ZIP_DEFLATED) as z:
        for path in sorted(output.glob('*.csv')): z.write(path, path.name)
        z.write(output/'manifest.json','manifest.json')
    print(f'Download: {target}')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bundle', type=Path, default=Path('data/paper_256_v3_national_20261111/adapted_fits'))
    p.add_argument('--metadata', type=Path, default=Path('data/clinical_registry_refresh/20261111/prepared_clinical/patient_metadata.parquet'))
    p.add_argument('--output-dir', type=Path, default=Path('data/outcome_rates_by_year_20261120'))
    p.add_argument('--source-spec', type=Path)
    a=p.parse_args();run(a.bundle,a.metadata,a.output_dir,a.source_spec)

if __name__ == '__main__': main()
