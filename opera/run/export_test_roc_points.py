"""Export exploratory, IPCW test-ROC summaries from existing prepared predictions.

Separate from tuning-selected operating points; no fitting or analysis-lock edits.
Shared patient bootstrap draws preserve paired comparisons and re-estimate each ROC.
"""
import argparse
import json
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED

import numpy as np
import pandas as pd

from opera.analysis.fixed_budget_risk import bucket, resolve_prepared, interval, add
from opera.evaluation.distributed_analysis import load_pickle, save_pickle
from opera.evaluation.secondary_analysis import CELL
from opera.evaluation.test_roc_points import sensitivities
from opera.run.interpretation_worker import claim, digest, write_json


def process(paths, context, models, targets):
    frame = pd.concat([pd.read_parquet(p) for p in paths], ignore_index=True)
    points, differences, coverage, accumulator = [], [], [], {}
    for key, cell in frame.groupby(CELL, sort=True):
        identity = dict(zip(CELL, key))
        groups = {name: group.loc[group.ipcw_weight.gt(0)].sort_values('subject_id')
                  for name, group in cell.groupby('model')}
        if set(groups) != set(models):
            coverage.append(identity | {'status':'missing_model'})
            continue
        reference = groups[models[0]]
        if any('prediction_absent_zero_weight' in group and
               group.prediction_absent_zero_weight.fillna(False).any() for group in groups.values()):
            coverage.append(identity | {'status':'missing_real_observable_prediction'})
            continue
        if reference.subject_id.duplicated().any() or any(
                not reference[['subject_id','label','ipcw_weight']].reset_index(drop=True).equals(
                    group[['subject_id','label','ipcw_weight']].reset_index(drop=True)) for group in groups.values()):
            coverage.append(identity | {'status':'unmatched_observable_population'})
            continue
        y = reference.label.to_numpy(int)
        minimum = context['performance_support']
        if (y.sum() < minimum['min_cases'] or (1-y).sum() < minimum['min_controls']):
            coverage.append(identity | {'status':'insufficient_cases_or_controls'})
            continue
        index, draws = context['draws'][key[:3]]
        index = {str(subject):i for subject,i in index.items()}
        w = reference.ipcw_weight.to_numpy(float)
        samples = draws[:, [index[str(s)] for s in reference.subject_id]] * w
        outputs = {}
        for model in models:
            p = groups[model].probability.to_numpy(float)
            point = sensitivities(p,y,w,targets)[0]
            boot = sensitivities(p,y,samples,targets)
            outputs[model] = point,boot
            for j,target in enumerate(targets):
                lo,hi,valid,status = interval(boot[:,j])
                points.append(identity | {'model':model,'specificity':target,'sensitivity':point[j],
                                           'lower':lo,'upper':hi,'n_observable':len(y),
                                           'n_cases':int(y.sum()),'n_controls':int((1-y).sum()),
                                           'valid_draws':valid,'interval_status':status})
        stable = all(np.isfinite(point).all() and np.isfinite(boot).all() for point,boot in outputs.values())
        coverage.append(identity | {'status':'estimated' if stable else 'unstable_bootstrap', 'n_observable':len(y)})
        for a,model_a in enumerate(models):
            for model_b in models[a+1:]:
                delta = outputs[model_a][0]-outputs[model_b][0]
                draws_delta = outputs[model_a][1]-outputs[model_b][1]
                for j,target in enumerate(targets):
                    lo,hi,valid,status = interval(draws_delta[:,j])
                    differences.append(identity | {'model_a':model_a,'model_b':model_b,
                        'specificity':target,'gain':delta[j],'lower':lo,'upper':hi,
                        'valid_draws':valid,'interval_status':status})
                    if stable:
                        add(accumulator,('difference',model_a,model_b,target),delta[j],draws_delta[:,j])
        if stable:
            for model,(point,boot) in outputs.items():
                for j,target in enumerate(targets):
                    add(accumulator,('sensitivity',model,'',target),point[j],boot[:,j])
    return points,differences,coverage,accumulator


def run(prepared, output, models, targets):
    output.mkdir(parents=True,exist_ok=True)
    with claim(output/'.lock') as owned:
        if not owned:
            raise RuntimeError('Another exporter owns this output')
        identity={'prepared':str(prepared),'prepared_sha256':digest(prepared/'prepared.json'),
                  'context_sha256':digest(prepared/'context_INTERNAL.pkl'), 'models':models,'specificities':targets,
                  'implementation_sha256':{str(p):digest(p) for p in [Path(__file__),Path(__file__).parents[1]/'evaluation/test_roc_points.py']}}
        intent=output/'intent_INTERNAL.json'
        if intent.exists() and json.loads(intent.read_text()) != identity:
            raise ValueError('Inputs/code changed; use a fresh output directory')
        write_json(intent,identity)
        context=load_pickle(prepared/'context_INTERNAL.pkl');context.pop('metadata',None)
        if context.get('combined_audit',{}).get('estimand') != 'cif_ipcw':
            raise ValueError('Need the existing full-risk-set IPCW prepared snapshot')
        directory=bucket(prepared,output,models,'held_out',256)
        batches=output/'batches_INTERNAL';batches.mkdir(exist_ok=True)
        codes=sorted({p.name.split('_')[0] for p in directory.glob('*.parquet')})
        tables=[[],[],[]];combined={}
        for i,code in enumerate(codes):
            path=batches/f'{code}.result_INTERNAL.pkl'
            if not path.exists():
                save_pickle(path,process(sorted(directory.glob(code+'_*.parquet')),context,models,targets))
            parts=load_pickle(path)
            for destination,rows in zip(tables,parts[:3]):destination.extend(rows)
            for key,(value,n,draws) in parts[3].items():
                if key not in combined:combined[key]=[0.,0,np.zeros(len(draws))]
                combined[key][0]+=value;combined[key][1]+=n;combined[key][2]+=draws
            print(f'Test ROC batches {i+1}/{len(codes)}',flush=True)
        names=['cell_sensitivity.csv','paired_differences.csv','coverage.csv']
        for name,rows in zip(names,tables):
            table=pd.DataFrame(rows)
            for count in ('n_observable','n_cases','n_controls'):
                if count in table:table.loc[table[count].between(1,4),count]=np.nan
            table.to_csv(output/name,index=False)
        rows=[]
        for (kind,a,b,target),(value,n,draws) in combined.items():
            lo,hi,valid,status=interval(draws/n)
            rows.append(dict(kind=kind,model_a=a,model_b=b,specificity=target,estimate=value/n,
                             lower=lo,upper=hi,n_cells=n,valid_draws=valid,interval_status=status,
                             aggregation='equal_cell_common_models_stable_all_draws'))
        pd.DataFrame(rows).to_csv(output/'summary.csv',index=False)
        write_json(output/'manifest.json',{'status':'complete','scope':'Exploratory test ROC operating points; no fixed deployment cutoff',
            'specificities':targets,'models':models,'weighting':'Existing IPCW weights; censoring estimates held fixed',
            'bootstrap':'Existing paired patient draws; ROC boundary re-estimated each draw',
            'ties':'Linear ROC interpolation; randomised boundary interpretation',
            'coverage':'Common observable population across all requested models; panel means use cells stable in every draw',
            'support':context['performance_support'],'prepared_identity':identity})
        archive=output/'download_test_roc_points.zip'
        with ZipFile(archive.with_suffix('.writing'),'w',ZIP_DEFLATED) as z:
            for name in names+['summary.csv','manifest.json']:z.write(output/name,name)
        archive.with_suffix('.writing').replace(archive)
        print('DOWNLOAD:',archive,flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint-dir',type=Path,default=Path('data/cutoff_recovery_20261121/analyses/core_damage/checkpoints'))
    p.add_argument('--output-dir',type=Path,default=Path('data/cutoff_recovery_20261121/exports/test_roc_points'))
    p.add_argument('--models',nargs='+',default=['R_EA','J_EA','X','XP'])
    p.add_argument('--specificities',nargs='+',type=float,default=[.75,.8,.9,.95])
    a=p.parse_args()
    run(resolve_prepared(checkpoint_dir=a.checkpoint_dir),a.output_dir.resolve(),a.models,a.specificities)


if __name__=='__main__':main()
