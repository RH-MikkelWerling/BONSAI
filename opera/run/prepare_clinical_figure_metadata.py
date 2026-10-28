"""Add real computed IPI/NCCN-IPI fields without rebuilding patient histories."""
from pathlib import Path
import numpy as np
import pandas as pd
from opera.run.analyse_locked_predictions import read_table

SCORES = [('ipi_computed', 5, [1, 2, 3]), ('nccn_ipi', 8, [1, 3, 5])]


def prepare(base, extra_paths, output):
    data = read_table(base).copy().reset_index(drop=True)
    data['subject_id'] = data.subject_id.astype(str)
    if data.subject_id.duplicated().any():
        raise ValueError('Base metadata has duplicate patients')
    for path in dict.fromkeys(map(str, extra_paths)):
        extra = read_table(path).copy()
        extra['subject_id'] = extra.subject_id.astype(str)
        if extra.subject_id.duplicated().any():
            raise ValueError(f'Clinical score source has duplicate patients: {path}')
        fields = [c for c in extra if c.startswith(('ipi_computed_', 'nccn_ipi_'))]
        if not fields:
            continue
        extra = extra.set_index('subject_id').reindex(data.subject_id)
        for field in fields:
            value = extra[field].reset_index(drop=True)
            if field in data:
                both = data[field].notna() & value.notna()
                left, right = data.loc[both, field], value[both]
                if field.endswith('_score'):
                    agree = np.isclose(pd.to_numeric(left), pd.to_numeric(right))
                else:
                    from opera.run.prepare_available_plot_configs import NATIVE_GROUP_MAP as GROUP_MAP
                    agree = left.astype(str).str.lower().str.strip().map(GROUP_MAP).eq(
                        right.astype(str).str.lower().str.strip().map(GROUP_MAP))
                if not np.all(agree):
                    raise ValueError(f'Conflicting clinical values: {field}')
                data[field] = data[field].fillna(value)
            else:
                data[field] = value
    report = {}
    for prefix, maximum, cuts in SCORES:
        score, group = prefix + '_score', prefix + '_group'
        if score in data:
            values = pd.to_numeric(data[score], errors='raise')
            observed = values.dropna()
            if not (observed.between(0, maximum) & observed.eq(np.floor(observed))).all():
                raise ValueError(f'Invalid numeric score: {score}')
            from opera.run.prepare_available_plot_configs import FOUR_GROUP_ORDER
            labels = pd.cut(values, [-np.inf, *cuts, np.inf], labels=FOUR_GROUP_ORDER).astype('string')
            # Numeric totals recover the native strata even if an earlier export
            # collapsed its intermediate groups. Preserve source labels for audit.
            if group in data:
                data[group + '_source'] = data[group]
                data[group] = labels
            else:
                data[group] = labels
        missing = sorted({score, group} - set(data))
        complete = int(data[[score, group]].notna().all(axis=1).sum()) if not missing else 0
        report[prefix] = dict(missing_columns=missing, n_complete=complete,
                             status='available' if complete else 'missing_or_empty')
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    data.to_parquet(output, index=False)
    return report
