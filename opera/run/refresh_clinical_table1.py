"""Replace clinical rows in the exact original Table 1; audit the IPI count change."""
import argparse
import hashlib
import json
from pathlib import Path
import pandas as pd
import yaml
from opera.run.compute_prognostic_scores import DEFAULT_COLUMNS, compute_scores
from opera.run.export_table1 import build_tables

CLINICAL = {'index','IPI','ipi','ipi_computed_score','ipi_computed_group',
            'nccn_ipi_score','nccn_ipi_group','nccn_ipi_risk','risk_category',
            'prognostic_score_by_system'}


def refresh(original, prepared, lyfo_path, source_analysis, output, quality_metadata=None):
    output=Path(output)
    if output.exists():
        raise FileExistsError(output)
    previous=pd.read_csv(original)
    metadata=pd.read_parquet(Path(prepared)/'patient_metadata.parquet')
    selected=metadata.loc[metadata.split.isin(['train','tuning','held_out'])].copy()
    if selected.subject_id.duplicated().any():
        raise ValueError('Duplicate final-cohort IDs')
    replaced = set(CLINICAL)
    quality_report = None
    quality_changes = None
    if quality_metadata is not None:
        from opera.run.flatten_ipi_records import performance_group
        quality_path = Path(quality_metadata)
        quality = (pd.read_parquet(quality_path) if quality_path.suffix == '.parquet'
                   else pd.read_csv(quality_path, dtype={'subject_id': str}))
        if not {'subject_id', 'PS'} <= set(quality) or quality.subject_id.isna().any():
            raise ValueError('Quality metadata needs nonmissing subject_id and raw ECOG PS')
        quality['subject_id'] = quality.subject_id.astype(str)
        selected['subject_id'] = selected.subject_id.astype(str)
        if quality.subject_id.duplicated().any():
            raise ValueError('Resolve duplicate quality-registry patient records before Table 1 refresh')
        if not set(quality.subject_id) <= set(selected.subject_id):
            raise ValueError('Quality metadata contains patients outside the frozen final cohort')
        quality['PS'] = pd.to_numeric(quality.PS, errors='raise')
        if not quality.PS.dropna().isin([0, 1, 2, 3, 4]).all():
            raise ValueError('Quality-registry PS must use explicitly mapped ECOG values 0–4')
        conflict_flags = quality.get('quality_registry_ps_conflict', pd.Series(False, index=quality.index)).fillna(False)
        if not conflict_flags.isin([True, False, 0, 1]).all():
            raise ValueError('Quality-registry PS conflict flags must be boolean')
        conflict_flags = conflict_flags.astype(bool)
        if (conflict_flags & quality.PS.notna()).any():
            raise ValueError('Conflicting quality-registry PS must be explicitly missing')
        quality['quality_registry_ps_conflict'] = conflict_flags
        if not quality.PS.notna().any() and not conflict_flags.any():
            raise ValueError('No nonmissing quality-registry PS supplied')
        supplied = selected.subject_id.map(quality.set_index('subject_id').PS)
        old_ps = pd.to_numeric(selected.get('PS', pd.Series(index=selected.index, dtype=float)), errors='coerce')
        covered = supplied.notna()
        conflicted = selected.subject_id.map(quality.set_index('subject_id').quality_registry_ps_conflict).fillna(False).astype(bool)
        for column in ('PS', 'ps', 'performance_status'):
            if column in selected:
                selected[column] = selected[column].astype(object)
        selected.loc[covered, 'PS'] = supplied.loc[covered]
        selected.loc[covered, 'performance_status'] = supplied.loc[covered].map(performance_group)
        if 'ps' in selected:
            selected.loc[covered, 'ps'] = supplied.loc[covered]
        # An explicit unresolved conflict must not silently fall back to an old
        # PS from a potentially different episode. Keep the patient, clear PS only.
        for column in ('PS', 'ps', 'performance_status'):
            if column in selected:
                selected.loc[conflicted, column] = pd.NA
        # Display unknown PS as an explicit category, including populations
        # where every PS is missing. Raw PS remains missing, never imputed.
        selected['performance_status'] = selected['performance_status'].fillna('Not recorded')
        replaced |= {'PS', 'ps', 'performance_status'}
        changes = selected[['cohort_fine', 'split']].assign(
            refreshed_ps_available=covered.astype(int),
            newly_observed_ps=(covered & old_ps.isna()).astype(int),
            changed_existing_ps=(covered & old_ps.notna() & old_ps.ne(supplied)).astype(int))
        changes['unresolved_ps_conflict'] = conflicted.astype(int)
        quality_changes = changes.groupby(['cohort_fine', 'split'], as_index=False).sum()
        source_manifest = quality_path.parent / 'quality_registry_manifest.json'
        source_notes = json.loads(source_manifest.read_text()) if source_manifest.exists() else {}
        quality_report = {'input_sha256': hashlib.sha256(quality_path.read_bytes()).hexdigest(),
                          'patients_with_refreshed_ps': int(covered.sum()),
                          'newly_observed_ps': int((covered & old_ps.isna()).sum()),
                          'changed_existing_ps': int((covered & old_ps.notna() & old_ps.ne(supplied)).sum()),
                          'patients_with_unresolved_ps_conflict': int(conflicted.sum()),
                          'unresolved_conflict_policy': 'Patient retained; PS missing; no fallback to previous PS',
                          'all_required_ps_sources_declared_loaded': source_notes.get('all_required_sources_loaded', False),
                          'source_names': list(source_notes.get('sources', {})),
                          'uncovered_patients': 'Original multi-registry values retained',
                          'scope': 'Clinical scores and PS refreshed; other Table 1 characteristics retained'}
    fresh=build_tables(selected)
    # Reconcile every available cohort/split count, not just the overall total.
    keys=['cohort_level','cohort','split']
    old_counts=previous.loc[previous.variable.eq('patients'),keys+['count']]
    new_counts=fresh.loc[fresh.variable.eq('patients'),keys+['count']]
    check=old_counts.merge(new_counts,on=keys,how='outer',suffixes=('_old','_new'),validate='one_to_one')
    if check[['count_old','count_new']].isna().any().any() or not check.count_old.eq(check.count_new).all():
        raise ValueError('Original Table 1 and final metadata cohort/split counts differ')
    result=pd.concat([previous.loc[~previous.variable.isin(replaced)],
                      fresh.loc[fresh.variable.isin(replaced)]],ignore_index=True)
    lyfo=pd.read_parquet(lyfo_path)
    counterfactual=compute_scores(lyfo,DEFAULT_COLUMNS)
    corrected=compute_scores(lyfo,{**DEFAULT_COLUMNS,'extranodal_sites':'n_extranodal_regions_diagnosis'})
    source=yaml.safe_load(Path(source_analysis).read_text())
    old=pd.read_parquet(source['patient_metadata']).rename(columns=source.get('patient_metadata_column_map') or {})
    for frame in (old,counterfactual,corrected,selected):
        frame['subject_id']=frame.subject_id.astype(str)
    ids=selected[['subject_id']]
    audit=ids.merge(old[['subject_id','ipi_computed_score']],on='subject_id',how='left',validate='one_to_one')
    audit=audit.rename(columns={'ipi_computed_score':'old'})
    audit=audit.merge(counterfactual[['subject_id','ipi_computed_score']].rename(columns={'ipi_computed_score':'seven_site'}),on='subject_id',how='left',validate='one_to_one')
    audit=audit.merge(corrected[['subject_id','ipi_computed_score']].rename(columns={'ipi_computed_score':'total_count'}),on='subject_id',how='left',validate='one_to_one')
    paired=audit.dropna(subset=['old','seven_site','total_count'])
    changed=paired.old.ne(paired.total_count)
    explained=paired.old.eq(paired.seven_site)
    report={'metadata_patients':len(metadata),'final_patients':len(selected),
        'excluded_without_registered_split':len(metadata)-len(selected),
        'split_counts':selected.split.value_counts().to_dict(),
        'all_table1_cohort_split_counts_match':True,
        'performance_status':quality_report or 'Original Table 1 rows retained pending all refreshed registry sources',
        'ipi_complete_for_component_audit':len(paired),
        'ipi_changed_scores_in_audit':int(changed.sum()),
        'ipi_changed_scores_explained_by_count_substitution':int((changed & explained).sum()),
        'ipi_changed_scores_with_other_component_or_source_changes':int((changed & ~explained).sum()),
        'original_table_sha256':hashlib.sha256(Path(original).read_bytes()).hexdigest(),
        'scope':'Component explanation is conditional on complete old and refreshed counterfactual scores; no patient identifiers exported.'}
    output.mkdir(parents=True)
    result.to_csv(output/'table1_values_long.csv',index=False)
    (output/'table1_refresh_report.json').write_text(json.dumps(report,indent=2))
    if quality_changes is not None:
        quality_changes.to_csv(output/'quality_ps_change_counts.csv',index=False)
    pd.crosstab(paired.old,paired.total_count).to_csv(output/'ipi_score_transition_counts.csv')
    print(json.dumps(report,indent=2))
    return result,report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ['original','prepared-dir','lyfo','source-analysis','output-dir']:
        parser.add_argument('--'+key,type=Path,required=True)
    parser.add_argument('--quality-metadata', type=Path,
                        help='Reviewed one-row-per-patient raw ECOG PS export from the quality-registry template')
    a=parser.parse_args()
    refresh(a.original,a.prepared_dir,a.lyfo,a.source_analysis,a.output_dir,a.quality_metadata)


if __name__=='__main__':
    main()
