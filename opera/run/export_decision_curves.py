"""Checklist K: saved-prediction KM decision curves and training-mapped indices.

No recalibration or training. Missing tabular full-risk-set predictions are
reported as unavailable, never filled or removed from the reference population.
"""
import argparse
import json
from pathlib import Path
from zipfile import ZipFile,ZIP_DEFLATED
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import yaml
from opera.evaluation.km_decision_curves import evaluate,index_mapping

MODELS=['R_EA','J_EA','X','XP']


def selected_predictions(prepared,pairs):
    prepared=Path(prepared);manifest=json.loads((prepared/'prepared.json').read_text())
    pieces=[]
    for name in manifest['parts']:
        if Path(name).is_absolute() or '..' in Path(name).parts:raise ValueError('Unsafe partition path')
        file=pq.ParquetFile(prepared/name)
        columns=['subject_id','model','seed','split','cohort','outcome','horizon_days','probability','time_days','event']
        if not set(columns)<=set(file.schema.names):raise ValueError('Prepared snapshot lacks native survival fields')
        optional=['prediction_absent_zero_weight'] if 'prediction_absent_zero_weight' in file.schema.names else []
        for batch in file.iter_batches(batch_size=50000,columns=columns+optional):
            frame=batch.to_pandas()
            keep=frame.model.isin(MODELS)&frame.split.eq('held_out')&frame.seed.eq(42)
            keep &= pd.MultiIndex.from_frame(frame[['outcome','horizon_days']]).isin(pairs)
            part=frame.loc[keep].copy()
            if optional:part=part.loc[~part.prediction_absent_zero_weight.astype('boolean').fillna(False).to_numpy(dtype=bool)]
            if not part.empty: pieces.append(part)
    if not pieces:raise ValueError('No requested held-out predictions')
    frame=pd.concat(pieces,ignore_index=True);frame['subject_id']=frame.subject_id.astype(str)
    keys=['model','subject_id','outcome','horizon_days']
    if frame.duplicated(keys).any():raise ValueError('Duplicate patient/model/outcome/horizon predictions')
    return frame


def aligned_risks(base,predictions):
    risks={};gaps=[];ids=base.subject_id.astype(str)
    for model in MODELS:
        part=predictions.loc[predictions.model.eq(model)].set_index('subject_id')
        absent=set(ids)-set(part.index)
        if absent:
            gaps.append(dict(strategy=model,status='unavailable_full_population',missing_patients=len(absent)));continue
        selected=part.loc[ids]
        if not np.allclose(selected.time_days.to_numpy(float),base.time_days.to_numpy(float),rtol=0,atol=1e-8) or not np.array_equal(selected.event.to_numpy(int),base.event.to_numpy(int)):
            raise ValueError(f'{model}: outcome labels differ on paired population')
        risk=selected.probability.to_numpy(float)
        if not np.isfinite(risk).all() or ((risk<0)|(risk>1)).any():
            gaps.append(dict(strategy=model,status='unavailable_invalid_risk'));continue
        risks[model]=risk
    return risks,gaps


def clinical_reference(directory):
    """Accept the server-native discrimination table or the local assembled table."""
    directory=Path(directory)
    candidates=[directory] if directory.is_file() else [directory/'matched_risk_and_auc.csv', directory/'clinical_discrimination.csv']
    for path in candidates:
        if not path.is_file():continue
        frame=pd.read_csv(path)
        required={'score_id','outcome','horizon_days','n_patients'}
        if not required<=set(frame):raise ValueError(f'Clinical reference missing columns {required-set(frame)}: {path}')
        if frame.score_id.duplicated().any():raise ValueError(f'Duplicate score populations: {path}')
        return frame,path
    raise FileNotFoundError(f'No clinical population reference in {directory}; expected clinical_discrimination.csv or matched_risk_and_auc.csv. Pass --clinical-results with the completed clinical directory or CSV.')


def run(prepared,metadata_path,score_config,clinical_results,output,*,n_boot=1000):
    from opera.run.render_clinical_score_panels import score_patients
    from opera.run.export_outcome_rates_by_year import frozen_outcomes
    from opera.run.finish_figure_inputs import sha
    prepared,output=Path(prepared),Path(output)
    metadata=pd.read_parquet(metadata_path);metadata['subject_id']=metadata.subject_id.astype(str)
    if metadata.subject_id.duplicated().any():raise ValueError('Duplicate metadata patients')
    recipe=yaml.safe_load(Path(score_config).read_text())
    # Use the final story recipe, including FLIPI2 OS, not the old proxy recipe.
    specs=recipe['scores']
    reference,reference_path=clinical_reference(clinical_results)
    if not set(reference.score_id)=={s['id'] for s in specs}:raise ValueError('Clinical recipe and final figure roster differ')
    for spec in specs:
        row=reference.loc[reference.score_id.eq(spec['id'])]
        if len(row)!=1 or row.outcome.iloc[0]!=spec['outcome'] or int(row.horizon_days.iloc[0])!=int(spec['horizon_days']):
            raise ValueError(f'Endpoint/horizon differs from final clinical plate: {spec["id"]}')
    pairs={('overall_survival',365),('treatment_failure',730)}|{(s['outcome'],int(s['horizon_days'])) for s in specs}
    if {p[0] for p in pairs}-{'overall_survival','treatment_failure'}:
        raise ValueError('New endpoint requires competing-risk review; KM only for death-inclusive endpoints')
    predictions=selected_predictions(prepared,pairs)
    training={}
    for frame in frozen_outcomes('data/paper_256_v3_national_20261111/adapted_fits',metadata_path,{p[0] for p in pairs}):
        ids=set(metadata.loc[metadata.split.eq('train'),'subject_id'])
        frame=frame.loc[frame.subject_id.isin(ids)].copy()
        frame['cohort']=frame.subject_id.map(metadata.set_index('subject_id').cohort_fine)
        training[frame.outcome.iloc[0]]=frame
    parts=json.loads((prepared/'prepared.json').read_text())['parts']
    part_facts={name:[(prepared/name).stat().st_size,(prepared/name).stat().st_mtime_ns] for name in parts}
    identity=dict(prepared_partition_stats=part_facts,implementation_sha256=sha(__file__),
                  km_implementation_sha256=sha(Path(__file__).resolve().parents[1]/'evaluation/km_decision_curves.py'),
                  prepared_sha256=sha(prepared/'prepared.json'),metadata_sha256=sha(metadata_path),
                  score_config_sha256=sha(score_config),clinical_reference_path=str(reference_path),clinical_reference_sha256=sha(reference_path),
                  n_boot=n_boot,seed=2026,method='KM within selected group, refit per paired patient resample; training-only index mapping')
    output.mkdir(parents=True,exist_ok=True)
    intent=output/'intent.json'
    if intent.exists() and json.loads(intent.read_text())!=identity:raise ValueError('Inputs changed; use a fresh DCA output root')
    intent.write_text(json.dumps(identity,indent=2))
    jobs=[dict(id='all_held_out_os',outcome='overall_survival',horizon_days=365),
          dict(id='all_held_out_treatment_failure',outcome='treatment_failure',horizon_days=730),*specs]
    all_curves=[];all_diffs=[];all_mapping=[];gaps=[]
    for spec in jobs:
        population=spec['id'];endpoint=spec['outcome'];horizon=int(spec['horizon_days'])
        subset=predictions.loc[predictions.outcome.eq(endpoint)&predictions.horizon_days.eq(horizon)]
        primary=subset.loc[subset.model.eq('R_EA')].copy()
        if 'group_column' in spec:
            # Same eligibility, date rules, category normalisation and cohort roster as the clinical plate.
            s={**spec,'model':'R_EA','seed':42}
            held,_=score_patients(primary,metadata,s,'held_out')
            if len(held)!=int(reference.loc[reference.score_id.eq(population),'n_patients'].iloc[0]):
                raise ValueError(f'Clinical population count differs from final plate: {population}')
            base=primary.set_index('subject_id').loc[held.subject_id].reset_index()
        else:
            expected=set(metadata.loc[metadata.split.eq('held_out'),'subject_id'])
            if set(primary.subject_id)!=expected:
                gaps.append(dict(population=population,strategy='R_EA',status='incomplete_all_patient_reference',missing_patients=len(expected-set(primary.subject_id))));continue
            base=primary
        if base.event.eq(2).any():raise ValueError('Competing event found in supposedly death-inclusive endpoint')
        events=int(((base.event==1)&(base.time_days<=horizon)).sum())
        if len(base)<5 or events<5:
            gaps.append(dict(population=population,status='suppressed_small_population_or_events'));continue
        risks,missing=aligned_risks(base,subset)
        gaps.extend([dict(population=population,**entry) for entry in missing])
        if 'R_EA' not in risks:raise ValueError('Primary full-population risk unavailable')
        if 'group_column' in spec:
            train=training[endpoint].copy()
            train=train.assign(model='R_EA',seed=42,split='train',horizon_days=horizon,probability=0.)
            scored,_=score_patients(train,metadata,s,'train')
            scored['index_value']=scored.ipi_score.astype(str) if spec.get('score_column') else scored.ipi_group.astype(str)
            held['index_value']=held.ipi_score.astype(str) if spec.get('score_column') else held.ipi_group.astype(str)
            try:
                mapping=index_mapping(scored,horizon)
                values=held.index_value.map(mapping.set_index('index_value').risk)
                if values.isna().any():raise ValueError('Held-out index category absent from training')
                risks['clinical_index']=values.to_numpy(float)
                mapping.insert(0,'population',population)
                # Hide small training-category cells; retain mapping internally for fixed index strategy.
                suppressed=(mapping.training_n<5)|(mapping.training_events<5)
                mapping['suppressed']=suppressed
                mapping.loc[suppressed,['training_n','training_events','risk']]=np.nan
                all_mapping.append(mapping)
            except ValueError as error:gaps.append(dict(population=population,strategy='clinical_index',status='unsupported_training_mapping',reason=str(error)))
        thresholds=np.arange(1,61)/100 if endpoint=='overall_survival' else np.arange(5,81)/100
        cache=output/f'{population}_INTERNAL.json'
        if cache.exists():
            saved=json.loads(cache.read_text());curves=pd.DataFrame(saved['curves']);diffs=pd.DataFrame(saved['differences'])
        else:
            curves,diffs=evaluate(base,risks,horizon,thresholds,n_boot=n_boot,seed=2026)
            cache.write_text(json.dumps(dict(curves=curves.to_dict('records'),differences=diffs.to_dict('records'))))
        if curves.net_benefit.isna().any():
            gaps.append(dict(population=population,status='some_threshold_groups_lack_horizon_support',unsupported_rows=int(curves.net_benefit.isna().sum())))
        for frame in [curves,diffs]:
            frame.insert(0,'population',population);frame.insert(1,'endpoint',endpoint);frame.insert(2,'horizon_days',horizon)
        all_curves.append(curves);all_diffs.append(diffs)
        print(f'Decision curves complete: {population}; {len(base)} patients; strategies {list(risks)}',flush=True)
    curves=pd.concat(all_curves,ignore_index=True) if all_curves else pd.DataFrame()
    differences=pd.concat(all_diffs,ignore_index=True) if all_diffs else pd.DataFrame()
    mappings=pd.concat(all_mapping,ignore_index=True) if all_mapping else pd.DataFrame(columns=['population','index_value','risk'])
    curves.to_csv(output/'decision_curves.csv',index=False);differences.to_csv(output/'decision_curve_differences.csv',index=False)
    mappings.to_csv(output/'index_risk_mapping.csv',index=False)
    if not differences.empty:
        chosen=((differences.endpoint.eq('overall_survival')&differences.threshold.round(2).isin([.1,.2,.3]))|
                (differences.endpoint.eq('treatment_failure')&differences.threshold.round(2).isin([.2,.4])))
        differences.loc[chosen].to_csv(output/'net_benefit_per_100.csv',index=False)
    report=dict(requested_models=MODELS,bootstrap_replicates=n_boot,bootstrap_seed=2026,
                status='complete_with_gaps' if gaps else 'complete',identity=identity,gaps=gaps,
                scope='Exploratory saved-prediction KM decision curves; no training/recalibration/primary selection',
                interpretation='Net true-positive equivalents per100, not an observed count of correctly classified patients',
                mapping_uncertainty='Training category risks fixed; intervals condition on that mapping',
                interval='Pointwise paired percentile 95% CI; at least80%finite replicates required; no simultaneous band')
    (output/'manifest.json').write_text(json.dumps(report,indent=2))
    target=output/'download_decision_curves.zip'
    with ZipFile(target,'w',ZIP_DEFLATED) as z:
        for name in ['decision_curves.csv','decision_curve_differences.csv','index_risk_mapping.csv','net_benefit_per_100.csv','manifest.json']:
            path=output/name
            if path.is_file():z.write(path,name)
    print(f'Download: {target}')


def main():
    from opera.analysis.fixed_budget_risk import resolve_prepared
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint-dir',type=Path,default=Path('data/figure_closure_20261119/comparisons_core/checkpoints'))
    p.add_argument('--metadata',type=Path,default=Path('data/clinical_registry_refresh/20261111/prepared_clinical/patient_metadata.parquet'))
    p.add_argument('--score-config',type=Path,default=Path('data/story_closure_20261116/server_only/score_config_flipi2_os.yaml'))
    p.add_argument('--clinical-results',type=Path,default=Path('data/story_closure_20261116/server_only/clinical_scores'))
    p.add_argument('--output-dir',type=Path,default=Path('data/decision_curves_20261120'))
    a=p.parse_args()
    import fcntl
    a.output_dir.mkdir(parents=True,exist_ok=True)
    with (a.output_dir/'.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        run(resolve_prepared(checkpoint_dir=a.checkpoint_dir),a.metadata,a.score_config,a.clinical_results,a.output_dir)

if __name__=='__main__':main()
