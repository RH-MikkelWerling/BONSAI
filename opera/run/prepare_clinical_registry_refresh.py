"""Versioned clinical-only registry refresh; never modifies the source analysis."""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
from pathlib import Path

import pandas as pd
import yaml

from opera.run.compute_prognostic_scores import DEFAULT_COLUMNS, compute_scores
from opera.run.flatten_ipi_records import flatten, nccn_risk
from opera.run.render_clinical_score_panels import expand_index_scores


def prepare(analysis_path, lyfo_path, indices_path, recipe_path, output, table1_values=None):
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(output)
    analysis = yaml.safe_load(Path(analysis_path).read_text())
    base = pd.read_parquet(analysis['patient_metadata']).rename(
        columns=analysis.get('patient_metadata_column_map') or {})
    base['subject_id'] = base.subject_id.astype(str)
    if base.subject_id.duplicated().any():
        raise ValueError('Base metadata must have one row per patient')
    lyfo = pd.read_parquet(lyfo_path)
    indices = pd.read_parquet(indices_path)
    for frame in (lyfo, indices):
        frame['patientid'] = frame.patientid.astype(str)
        if frame.patientid.isna().any() or frame.patientid.eq('nan').any():
            raise ValueError('Missing registry patient ID')
    if lyfo.patientid.duplicated().any():
        raise ValueError('LYFO must have one row per patient')
    if indices.duplicated(['patientid', 'index']).any():
        raise ValueError('Repeated patient/index pairs must be resolved explicitly')
    mapping = {**DEFAULT_COLUMNS, 'extranodal_sites': 'n_extranodal_regions_diagnosis'}
    scores = compute_scores(lyfo, mapping)
    scores.subject_id = scores.subject_id.astype(str)
    # Reuse established one-row display resolution; preserve every index below
    # in independent columns so multi-index patients enter every comparison.
    grouped = indices.groupby('patientid', sort=False)
    records = {sid: part.rename(columns={'patientid': 'subject_id', 'ipi': 'IPI',
                                         'disease': 'Disease'}).to_dict('records')
               for sid, part in grouped}
    ps = lyfo.set_index('patientid').PS_diagnosis.to_dict()
    raw = []
    for sid in base.subject_id:
        entries = copy.deepcopy(records.get(sid, []))
        if not entries and pd.notna(ps.get(sid)):
            entries = [{}]
        for row in entries:
            row['PS'] = float(ps[sid]) if pd.notna(ps.get(sid)) else None
            for key, value in list(row.items()):
                if pd.isna(value):
                    row[key] = None
        raw.append(json.dumps(entries))
    refreshed = base.copy()
    refreshed['ipi_records_json'] = raw
    display = flatten(refreshed)
    fields = [c for c in display if c != 'subject_id'] + list(scores.columns[1:])
    refreshed = refreshed.drop(columns=[c for c in fields if c in refreshed])
    refreshed = refreshed.merge(display, on='subject_id', how='left', validate='one_to_one')
    refreshed = refreshed.merge(scores, on='subject_id', how='left', validate='one_to_one')
    refreshed['nccn_ipi_risk'] = nccn_risk(refreshed.nccn_ipi_group, refreshed.cohort_fine)
    refreshed['risk_source'] = refreshed['index'].where(refreshed.risk_category.ne('Not recorded'))
    use = refreshed.nccn_ipi_risk.notna() & refreshed.nccn_ipi_risk.ne('Not recorded')
    refreshed.loc[use, 'risk_category'] = refreshed.loc[use, 'nccn_ipi_risk']
    refreshed.loc[use, 'risk_source'] = 'NCCN-IPI (computed)'
    recipe = yaml.safe_load(Path(recipe_path).read_text())
    # Generate the existing registered/native endpoint specs from all records,
    # then substitute independent per-index columns in one-row metadata.
    long = refreshed.drop(columns=['index', 'IPI'], errors='ignore').merge(
        indices.rename(columns={'patientid': 'subject_id', 'ipi': 'IPI'}),
        on='subject_id', how='left', validate='one_to_many')
    specs = expand_index_scores(recipe, long)
    index_columns = {}
    for value, part in indices.groupby('index', sort=True):
        column = 'registry_group_' + hashlib.sha256(str(value).encode()).hexdigest()[:12]
        index_columns[str(value)] = column
        refreshed[column] = refreshed.subject_id.map(part.set_index('patientid').ipi)
    for spec in specs:
        if spec.get('index_column'):
            values = spec['index_values']
            if len(values) != 1:
                raise ValueError('Expected one recorded index per comparison')
            spec['group_column'] = index_columns[str(values[0])]
            spec.pop('index_column')
            spec.pop('index_values')
        elif spec.get('group_column') in refreshed and not spec.get('cohorts'):
            # expand_index_scores marks absent computed columns unavailable on
            # the long registry table; restore the original registered scope.
            original = next((s for s in recipe.get('scores', []) if s['id'] == spec['id']), None)
            if original:
                spec['cohorts'] = original['cohorts']
    recipe.pop('all_indices', None)
    recipe['scores'] = specs
    # Already expanded native endpoints; prevent a second expansion on rendering.
    recipe['native_endpoints'] = {}
    recipe['index_endpoint'] = 'shared'
    report = {'amendment': 'Registry coverage refresh after clinical held-out results were viewed; models and evaluation definitions unchanged',
              'patients': len(base), 'lyfo_rows': len(lyfo), 'recorded_index_rows': len(indices),
              'multi_index_patients': int((indices.groupby('patientid').size() > 1).sum()),
              'index_columns': index_columns, 'fields': {}}
    for column in ['IPI', 'index', *scores.columns[1:], *index_columns.values()]:
        old = base[column] if column in base else pd.Series(pd.NA, index=base.index)
        new = refreshed[column]
        both = old.notna() & new.notna()
        report['fields'][column] = {'old_available': int(old.notna().sum()),
            'new_available': int(new.notna().sum()),
            'changed_overlapping': int((old.astype('string').ne(new.astype('string')) & both).sum())}
    for column in base:
        if column not in {*fields, 'ipi_records_json', 'nccn_ipi_risk', 'risk_source'}:
            pd.testing.assert_series_equal(base[column].reset_index(drop=True),
                                           refreshed[column].reset_index(drop=True))
    updated_table1 = None
    if table1_values is not None:
        from opera.run.export_table1 import build_tables
        previous = pd.read_csv(table1_values)
        population_rows = previous.loc[previous.variable.eq('patients') & previous.cohort_level.eq('all')]
        expected = {'Overall': len(refreshed), **refreshed.split.value_counts().to_dict()}
        for _, row in population_rows.iterrows():
            if int(row['count']) != expected[row['split']]:
                raise ValueError('Table 1 population differs from refreshed metadata; refusing replacement')
        if not set(expected) <= set(population_rows.split):
            raise ValueError('Table 1 lacks population counts for reconciliation')
        clinical_variables = {'index', 'IPI', 'ipi', 'ipi_computed_score', 'ipi_computed_group',
            'nccn_ipi_score', 'nccn_ipi_group', 'nccn_ipi_risk', 'risk_category',
            'performance_status', 'prognostic_score_by_system'}
        fresh = build_tables(refreshed)
        updated_table1 = pd.concat([previous.loc[~previous.variable.isin(clinical_variables)],
            fresh.loc[fresh.variable.isin(clinical_variables)]], ignore_index=True)
    output.mkdir(parents=True)
    if updated_table1 is not None:
        updated_table1.to_csv(output / 'table1_values_long.csv', index=False)
    refreshed.to_parquet(output / 'patient_metadata.parquet', index=False)
    scores.to_parquet(output / 'computed_scores.parquet', index=False)
    indices.to_parquet(output / 'recorded_indices.parquet', index=False)
    analysis['patient_metadata'] = str(output / 'patient_metadata.parquet')
    analysis['patient_metadata_column_map'] = {}
    (output / 'analysis.yaml').write_text(yaml.safe_dump(analysis, sort_keys=False))
    (output / 'clinical_scores.yaml').write_text(yaml.safe_dump(recipe, sort_keys=False))
    report['sources'] = {str(Path(p).resolve()): hashlib.sha256(Path(p).read_bytes()).hexdigest()
                         for p in [analysis_path, lyfo_path, indices_path, recipe_path]}
    (output / 'refresh_report.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    return refreshed, recipe, report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['analysis-config', 'lyfo', 'indices', 'score-config', 'output-dir']:
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--table1-values', type=Path, help='Existing internal Table 1 long CSV; replace clinical rows only')
    args = parser.parse_args()
    prepare(args.analysis_config, args.lyfo, args.indices, args.score_config, args.output_dir, args.table1_values)


if __name__ == '__main__':
    main()
