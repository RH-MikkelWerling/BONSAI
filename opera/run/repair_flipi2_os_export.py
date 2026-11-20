"""Repair FLIPI2 OS compaction from existing full predictions; no inference/fits."""
import argparse
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import yaml
from opera.run.finish_figure_inputs import sha,immutable

SCORE='flipi2_all_subtypes_native'


def run(workspace,story,national,output):
    import pandas as pd
    from opera.run.analyse_locked_predictions import read_table
    from opera.run.launch_adapted_clinical_comparison import compact,discrimination
    from opera.run.render_clinical_score_panels import expand_index_scores,score_patients
    from bonsai.functional.checkpointing import verified_training_selection
    from opera.run.refresh_clinical_figures import package_clinical
    workspace,story,national,output=map(lambda p:Path(p).resolve(),[workspace,story,national,output])
    original=yaml.safe_load((workspace/'analysis.yaml').read_text())
    recipe=yaml.safe_load((story/'server_only/score_config_flipi2_os.yaml').read_text())
    meta=read_table(original['patient_metadata']);meta.subject_id=meta.subject_id.astype(str)
    meta=meta.rename(columns=original.get('patient_metadata_column_map') or {})
    spec,=[s for s in expand_index_scores(recipe,meta) if s['id']==SCORE]
    if spec['outcome']!='overall_survival' or int(spec['horizon_days'])!=365:raise ValueError('Need final one-year OS FLIPI2 recipe')
    recipe={**copy.deepcopy(recipe),'scores':[spec],'all_indices':None,'native_endpoints':{}}
    plan=json.loads((workspace/'launch_manifest.json').read_text())
    command=plan['training_command'];training=Path(command[command.index('--output-dir')+1])/'direct_cr_predictions.parquet'
    if not training.is_file():raise FileNotFoundError(f'Existing full training predictions needed, no inference launched: {training}')
    bundle=json.loads((national/'adapted_fits/bundle.json').read_text());row,=[r for r in bundle['runs'] if r['name']=='seed42/EA/R']
    selection=verified_training_selection(national/'neural_runs'/row['run_directory'])
    training_receipt=json.loads((training.parent/'training_export_receipt.json').read_text())
    if (training_receipt['checkpoint_sha256']!=selection['best_checkpoint_sha256'] or
        training_receipt['config_sha256']!=row['config_sha256'] or training_receipt.get('split')!='train' or training_receipt['prediction_sha256']!=sha(training)):
        raise ValueError('Full training export does not match the selected R_EA fit')
    artifacts=[]
    for split,folder in [('tuning','extended_tuning_comparison'),('held_out','extended_test_comparison')]:
        source=yaml.safe_load((national/folder/'analysis.yaml').read_text())
        artifact,=[a for a in source['artifacts'] if a['model']=='R_EA' and a['split']==split and int(a['seed'])==42]
        marker=json.loads((Path(artifact['path']).parent/'evaluation_complete.json').read_text())
        if marker['prediction_sha256']!=sha(artifact['path']) or marker['inputs']['checkpoint_sha256']!=selection['best_checkpoint_sha256'] or marker['inputs']['config_sha256']!=row['config_sha256']:
            raise ValueError(f'Full {split} prediction receipt changed')
        artifacts.append(copy.deepcopy(artifact))
    train=copy.deepcopy(original['clinical_threshold_artifacts'][0]);train.update(model='R_EA',split='train',path=str(training))
    intent=dict(scope='Repair endpoint-specific compaction; FLIPI2 OS only; existing full predictions',
        score=spec,metadata_sha256=sha(original['patient_metadata']),
        source_prediction_sha256={a['split']:sha(a['path']) for a in [*artifacts,train]})
    output.mkdir(parents=True,exist_ok=True);immutable(output/'intent.yaml',intent)
    private=output/'predictions_INTERNAL';private.mkdir(exist_ok=True)
    for artifact in [*artifacts,train]:
        target=private/f'{artifact["split"]}.parquet'
        if not target.exists():
            temp=target.with_suffix('.tmp');compact(artifact['path'],temp,[spec],meta);temp.replace(target)
        artifact['path']=str(target)
    analysis={**original,'artifacts':artifacts,'clinical_threshold_artifacts':[train]}
    immutable(output/'analysis.yaml',analysis);immutable(output/'clinical_scores.yaml',recipe)
    results=output/'clinical_results'
    if not (results/'report_manifest.json').is_file():
        if results.exists():
            import time
            results.rename(output/f'clinical_results_incomplete_{time.time_ns()}_INTERNAL')
        subprocess.run([sys.executable,'-u','-m','opera.run.render_clinical_score_panels',
            '--analysis-config',str(output/'analysis.yaml'),'--score-config',str(output/'clinical_scores.yaml'),
            '--split','held_out','--output-dir',str(results),'--execute'],check=True,
            env={**os.environ,'CUDA_VISIBLE_DEVICES':'','OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1',
                 'OPERA_ANALYSIS_WORKERS':'1','OPERA_SUMMARY_WORKERS':'1','MPLBACKEND':'Agg',
                 'OPERA_THESIS_STYLE':'1','OPERA_PAPER_FIGURE_MARK':'','OPERA_PAPER_EXPORT_PLOT_STATE':'1'})
    discrimination(analysis,[spec],meta,results/'clinical_discrimination.csv')
    target=output/'download_flipi2_os_repair.zip'
    if target.exists():target.unlink()
    package_clinical(results,target)
    print(f'Download: {target}',flush=True)
    print(f'DCA reference override: {results/"clinical_discrimination.csv"}',flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--workspace',type=Path,default=Path('data/paper_256_v3_national_20261111/clinical_20261115/R_EA'))
    p.add_argument('--story-root',type=Path,default=Path('data/story_closure_20261116'))
    p.add_argument('--national-root',type=Path,default=Path('data/paper_256_v3_national_20261111'))
    p.add_argument('--output-dir',type=Path,default=Path('data/flipi2_os_repair_20261120'))
    a=p.parse_args()
    import fcntl
    a.output_dir.mkdir(parents=True,exist_ok=True)
    with (a.output_dir/'.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        run(a.workspace,a.story_root,a.national_root,a.output_dir)

if __name__=='__main__':main()
