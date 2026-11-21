"""Opt-in recovery checks on the datasets actually constructed for training."""
import json
import os
from datetime import datetime
from pathlib import Path

import torch

from bonsai.functional.censoring import censor_subject
from bonsai.functional.features import compute_abspos


def audit_dataset(dataset):
    import pandas as pd
    from opera.run.export_outcome_rates_by_year import canonical_patient_ids
    boundary = float(compute_abspos(datetime(2022, 1, 1)))
    if dataset.cutoff_date != boundary:
        raise ValueError('Recovery requires the exclusive 2022-01-01 cutoff')
    if dataset.abspos_subject_jitter_years:
        raise ValueError('Recovery audit requires unjittered event timestamps')
    patients = set()
    retained = removed = 0
    latest = None
    keys = canonical_patient_ids(pd.Series([raw['subject_id'] for raw in dataset.subjects]))
    for raw, key in zip(dataset.subjects, keys):
        if key in patients:
            raise ValueError('Duplicate pretraining patient')
        patients.add(key)
        times = raw['abspos']
        if not torch.isfinite(times).all() or (times[1:] < times[:-1]).any():
            raise ValueError('Pretraining source timestamps must be finite and ordered')
        actual = censor_subject(raw, dataset.cutoff_date, inclusive=False)['abspos']
        if (actual >= boundary).any():
            raise ValueError('Post-cutoff event survived censoring')
        retained += len(actual)
        removed += len(times) - len(actual)
        if len(actual):
            maximum = float(actual.max())
            latest = maximum if latest is None else max(latest, maximum)
    if not retained or not len(dataset):
        raise ValueError('Empty recovery dataset after cutoff')
    # Every actual window is also checked at access time, including AR targets.
    dataset.required_exclusive_cutoff = boundary
    return dict(patients=len(patients), windows=len(dataset), retained_tokens=retained,
                removed_tokens=removed, latest_retained_abspos=latest,
                post_cutoff_tokens=0, every_window_runtime_guard=True), patients


def audit_datamodule(module, output):
    train, train_ids = audit_dataset(module.train_dataset)
    validation, val_ids = audit_dataset(module.val_dataset)
    if train_ids & val_ids:
        raise ValueError('Pretraining train/validation patient overlap')
    if hasattr(module, 'population'):
        import pandas as pd
        from opera.run.export_outcome_rates_by_year import canonical_patient_ids
        expected = canonical_patient_ids(pd.Series(module.population['subject_id'].to_list()))
        if expected.duplicated().any() or set(expected) != train_ids | val_ids:
            raise ValueError('Actual pretraining patients differ from the configured population')
    report = dict(status='passed', exclusive_cutoff='2022-01-01',
                  training=train, validation=validation,
                  scope='Population-filtered actual datasets; no patient identifiers exported')
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f'.{path.name}.{os.getpid()}.writing')
    temporary.write_text(json.dumps(report, indent=2) + '\n')
    temporary.replace(path)
    print('PRETRAINING CUTOFF AUDIT PASSED:', path, flush=True)
    return report
