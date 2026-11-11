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
            'performance_status','prognostic_score_by_system'}


def refresh(original, prepared, lyfo_path, source_analysis, output):
    output=Path(output)
    if output.exists():
        raise FileExistsError(output)
    previous=pd.read_csv(original)
    metadata=pd.read_parquet(Path(prepared)/'patient_metadata.parquet')
    selected=metadata.loc[metadata.split.isin(['train','tuning','held_out'])].copy()
    if selected.subject_id.duplicated().any():
        raise ValueError('Duplicate final-cohort IDs')
    fresh=build_tables(selected)
    # Reconcile every available cohort/split count, not just the overall total.
    keys=['cohort_level','cohort','split']
    old_counts=previous.loc[previous.variable.eq('patients'),keys+['count']]
    new_counts=fresh.loc[fresh.variable.eq('patients'),keys+['count']]
    check=old_counts.merge(new_counts,on=keys,how='outer',suffixes=('_old','_new'),validate='one_to_one')
    if check[['count_old','count_new']].isna().any().any() or not check.count_old.eq(check.count_new).all():
        raise ValueError('Original Table 1 and final metadata cohort/split counts differ')
    result=pd.concat([previous.loc[~previous.variable.isin(CLINICAL)],
                      fresh.loc[fresh.variable.isin(CLINICAL)]],ignore_index=True)
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
        'ipi_complete_for_component_audit':len(paired),
        'ipi_changed_scores_in_audit':int(changed.sum()),
        'ipi_changed_scores_explained_by_count_substitution':int((changed & explained).sum()),
        'ipi_changed_scores_with_other_component_or_source_changes':int((changed & ~explained).sum()),
        'original_table_sha256':hashlib.sha256(Path(original).read_bytes()).hexdigest(),
        'scope':'Component explanation is conditional on complete old and refreshed counterfactual scores; no patient identifiers exported.'}
    output.mkdir(parents=True)
    result.to_csv(output/'table1_values_long.csv',index=False)
    (output/'table1_refresh_report.json').write_text(json.dumps(report,indent=2))
    pd.crosstab(paired.old,paired.total_count).to_csv(output/'ipi_score_transition_counts.csv')
    print(json.dumps(report,indent=2))
    return result,report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ['original','prepared-dir','lyfo','source-analysis','output-dir']:
        parser.add_argument('--'+key,type=Path,required=True)
    a=parser.parse_args()
    refresh(a.original,a.prepared_dir,a.lyfo,a.source_analysis,a.output_dir)


if __name__=='__main__':
    main()
