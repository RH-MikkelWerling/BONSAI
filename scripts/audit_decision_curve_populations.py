"""Read-only aggregate audit of DCA versus the final clinical populations."""
import argparse
import json
from pathlib import Path
import pandas as pd
import pyarrow.parquet as pq
import yaml
from opera.run.export_decision_curves import clinical_reference,selected_predictions
from opera.run.export_outcome_rates_by_year import canonical_patient_ids
from opera.run.render_clinical_score_panels import score_patients,expand_index_scores
from opera.analysis.fixed_budget_risk import resolve_prepared
from opera.run.finish_figure_inputs import sha


def metadata(path):
    data=pd.read_parquet(path);data['subject_id']=canonical_patient_ids(data.subject_id)
    if data.subject_id.duplicated().any():raise ValueError('Duplicate metadata IDs')
    return data


def counts(predictions,meta,spec):
    configured={**spec,'model':'R_EA','seed':42}
    held,flow=score_patients(predictions,meta,configured,'held_out')
    return held,dict(n_patients=len(held),flow=flow)


def audit(a):
    meta=metadata(a.metadata);recipe=yaml.safe_load(a.score_config.read_text())
    reference,reference_path=clinical_reference(a.clinical_results)
    specs=recipe['scores'];expanded={s['id']:s for s in expand_index_scores(recipe,meta)}
    pairs={(s['outcome'],int(s['horizon_days'])) for s in specs}
    predictions=selected_predictions(resolve_prepared(checkpoint_dir=a.checkpoint_dir),pairs)
    predictions=predictions.loc[predictions.model.eq('R_EA')].copy()
    report=dict(metadata_path=str(a.metadata),metadata_sha256=sha(a.metadata),reference_path=str(reference_path),populations=[])
    manifest_path=a.clinical_results/'report_manifest.json'
    if manifest_path.is_file():
        saved=json.loads(manifest_path.read_text());report['final_render_metadata_sha256']=saved.get('metadata_sha')
        report['metadata_matches_final_render']=saved.get('metadata_sha')==sha(a.metadata)
    clinical=None;clinical_meta=None
    if a.clinical_analysis.is_file():
        clinical=yaml.safe_load(a.clinical_analysis.read_text());clinical_meta=metadata(clinical['patient_metadata'])
        report['clinical_analysis_metadata_path']=clinical['patient_metadata']
        report['clinical_analysis_metadata_sha256']=sha(clinical['patient_metadata'])
    for spec in specs:
        selected=predictions.loc[predictions.outcome.eq(spec['outcome']) & predictions.horizon_days.eq(spec['horizon_days'])].copy()
        held,result=counts(selected,meta,spec)
        ref=reference.loc[reference.score_id.eq(spec['id'])]
        row=dict(score_id=spec['id'],reference_n=int(ref.n_patients.iloc[0]),current_prepared=result,
                 raw_cohorts=spec['cohorts'],expanded_cohorts=expanded[spec['id']]['cohorts'])
        revised=selected.copy();revised['cohort']=revised.subject_id.map(meta.set_index('subject_id').cohort_fine)
        row['prepared_vs_metadata_cohort_disagreements']=int(selected.cohort.ne(revised.cohort).sum())
        _,row['using_metadata_cohorts']=counts(revised,meta,expanded[spec['id']])
        if clinical is not None:
            declared=[x for x in clinical['artifacts'] if x.get('split')=='held_out' and x.get('model')=='R_EA']
            if len(declared)==1 and declared[0].get('format')=='direct_cif':
                path=Path(declared[0]['path']);column=f"cif_{int(spec['horizon_days'])}d"
                table=pq.read_table(path,columns=['subject_id','outcome','time_days','event',column],filters=[('outcome','=',spec['outcome'])]).to_pandas()
                table['subject_id']=canonical_patient_ids(table.subject_id)
                table['cohort']=table.subject_id.map(clinical_meta.set_index('subject_id').cohort_fine)
                table=table.assign(model='R_EA',seed=42,split='held_out',horizon_days=int(spec['horizon_days']))
                table=table.rename(columns={column:'probability'})
                clinical_specs={s['id']:s for s in expand_index_scores(recipe,clinical_meta)}
                original,original_counts=counts(table,clinical_meta,clinical_specs[spec['id']])
                row['actual_clinical_source']=original_counts
                row['clinical_patients_without_prepared_predictions']=len(set(original.subject_id)-set(selected.subject_id))
                row['prepared_only_score_patients']=len(set(held.subject_id)-set(original.subject_id))
        report['populations'].append(row)
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint-dir',type=Path,default=Path('data/figure_closure_20261119/comparisons_core/checkpoints'))
    p.add_argument('--metadata',type=Path,default=Path('data/clinical_registry_refresh/20261111/prepared_clinical/patient_metadata.parquet'))
    p.add_argument('--score-config',type=Path,default=Path('data/story_closure_20261116/server_only/score_config_flipi2_os.yaml'))
    p.add_argument('--clinical-results',type=Path,default=Path('data/story_closure_20261116/server_only/clinical_scores'))
    p.add_argument('--clinical-analysis',type=Path,default=Path('data/paper_256_v3_national_20261111/clinical_20261115/R_EA/analysis.yaml'))
    p.add_argument('--output',type=Path,default=Path('logs/decision_curve_population_audit_20261120.json'))
    a=p.parse_args();report=audit(a)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(report,indent=2))
    for row in report['populations']:
        print(row['score_id'],'reference=',row['reference_n'],'prepared=',row['current_prepared']['n_patients'],
              'metadata_cohort=',row['using_metadata_cohorts']['n_patients'],
              'clinical_source=',row.get('actual_clinical_source',{}).get('n_patients'),flush=True)
    print('Download aggregate audit:',a.output)

if __name__=='__main__':main()
