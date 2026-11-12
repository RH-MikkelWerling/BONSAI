"""Paired correction-replay effects using fixed main-analysis IPCW and draws."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from opera.analysis.representation_absorption import ARMS, interval, bootstrap_auc
from opera.evaluation.distributed_analysis import load_pickle
from opera.evaluation.secondary_analysis import CELL, metric_values, bootstrap_metric_values


def native_long(path,metadata,seed):
    frame=pd.read_parquet(path)
    frame['subject_id']=frame.subject_id.astype(str)
    cohorts=metadata[['subject_id','cohort_fine']].copy()
    cohorts['subject_id']=cohorts.subject_id.astype(str)
    frame=frame.drop(columns=['cohort','cohort_fine'],errors='ignore').merge(cohorts,on='subject_id',validate='many_to_one')
    frame=frame.rename(columns={'cohort_fine':'cohort'})
    columns=[c for c in frame if c.startswith('cif_') and c.endswith('d')]
    result=frame.melt(id_vars=['subject_id','cohort','outcome','time_days','event','split'],
                      value_vars=columns,var_name='horizon_days',value_name='probability')
    result['horizon_days']=result.horizon_days.str.extract(r'(\d+)')[0].astype(int)
    result['seed']=seed
    return result


def run(prepared,plan_path,replay_root,output,split='tuning',common_roster=None):
    prepared,replay_root,output=map(Path,(prepared,replay_root,output))
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    context=load_pickle(prepared/'context_INTERNAL.pkl')
    metadata=context['metadata']
    plan=json.loads(Path(plan_path).read_text())
    manifest=json.loads((prepared/'prepared.json').read_text())
    effects,summaries,blockers,norms=[],[],[],[]
    for job in plan['jobs']:
        root=replay_root/f"{job['encoder']}_seed{job['seed']}"
        full_path=root/'full/direct_cr_predictions.parquet'
        if not full_path.exists():
            blockers.append({'run_id':job['run_id'],'reason':'full replay missing'})
            continue
        full=native_long(full_path,metadata,job['seed'])
        norm_path=root/'full/correction_norm_summary.csv'
        if norm_path.exists():
            norms.append(pd.read_csv(norm_path))
        for mode in job.get('modes',['full','all','own_disease','own_outcome','global']):
            if mode=='full':
                continue
            ablated_path=root/mode/'direct_cr_predictions.parquet'
            if not ablated_path.exists():
                blockers.append({'run_id':job['run_id'],'mode':mode,'reason':'ablated replay missing'})
                continue
            zero=native_long(ablated_path,metadata,job['seed'])
            paired=full.merge(zero,on=CELL+['subject_id'],suffixes=('_full','_zero'),validate='one_to_one',how='outer',indicator=True)
            if not paired['_merge'].eq('both').all():
                raise ValueError('Full and zeroed replay patient rosters differ')
            for column in ['event','time_days']:
                if not np.array_equal(paired[column+'_full'],paired[column+'_zero'],equal_nan=True):
                    raise ValueError('Replay endpoint values changed')
            selected=ARMS[job['encoder']][1]
            for part in manifest['parts']:
                reference=pd.read_parquet(prepared/part)
                reference=reference.loc[reference.model.eq(selected)&reference.seed.eq(job['seed'])&reference.split.eq(split)]
                if reference.empty:
                    continue
                # Prepared parts can split cells: collect narrow reference rows first.
                destination=output/f".reference_{job['encoder']}_{job['seed']}_{part}"
                reference[CELL+['subject_id','label','ipcw_weight','probability']].to_parquet(destination,index=False)
            paths=list(output.glob(f".reference_{job['encoder']}_{job['seed']}_*.parquet"))
            if not paths:
                blockers.append({'run_id':job['run_id'],'reason':'matching prepared arm missing'})
                continue
            reference=pd.concat([pd.read_parquet(p) for p in paths],ignore_index=True)
            for p in paths:
                p.unlink()
            if common_roster is not None:
                roster=pd.read_csv(common_roster)[CELL].drop_duplicates()
                reference=reference.merge(roster,on=CELL,how='inner',validate='many_to_one')
            reference['subject_id']=reference.subject_id.astype(str)
            matched=reference.merge(paired,on=CELL+['subject_id'],validate='one_to_one',how='left')
            if not np.allclose(matched.probability,matched.probability_full,rtol=1e-5,atol=1e-6,equal_nan=False):
                raise ValueError('Full replay does not reproduce the main OPERA predictions')
            if matched.probability_full.isna().any() or matched.probability_zero.isna().any():
                raise ValueError('Replay missing patients from main analysis risk set')
            records=[]
            for key,cell in matched.groupby(CELL):
                y=cell.label.to_numpy(float); ipcw=cell.ipcw_weight.to_numpy(float)
                known=ipcw>0
                if ((y[known]==1).sum()<context['performance_support']['min_cases']
                        or (y[known]==0).sum()<context['performance_support']['min_controls']):
                    continue
                mapping,weights=context['draws'][key[:3]]
                # Main bootstrap maps can use integer IDs; resolve by their string forms.
                mapping={str(k):v for k,v in mapping.items()}
                w=weights[:,[mapping[s] for s in cell.subject_id]]
                a=cell.probability_full.to_numpy(float); b=cell.probability_zero.to_numpy(float)
                gain=metric_values(y,a,ipcw=ipcw)['auc']-metric_values(y,b,ipcw=ipcw)['auc']
                draws=bootstrap_auc(y,a,w,ipcw=ipcw)-bootstrap_auc(y,b,w,ipcw=ipcw)
                lo,hi=interval(draws)
                row={**dict(zip(CELL,key)), 'encoder':job['encoder'],'mode':mode,'run_id':job['run_id'],
                     'gain':gain,'lower':lo,'upper':hi,'mean_abs_delta_cif':float(np.abs(a-b).mean()),'n':len(cell)}
                effects.append(row);records.append((row,draws))
            for stratum in ['overall','cohort','outcome']:
                groups={}
                for row,draw in records:
                    groups.setdefault('all' if stratum=='overall' else row[stratum],[]).append((row,draw))
                for value,items in groups.items():
                    stable=[item for item in items if np.isfinite(item[1]).all()]
                    lo,hi=interval(np.mean([v for _,v in stable],axis=0)) if stable else [np.nan,np.nan]
                    summaries.append(dict(encoder=job['encoder'],seed=job['seed'],mode=mode,stratum=stratum,value=value,
                        gain=np.mean([r['gain'] for r,_ in stable]) if stable else np.nan,lower=lo,upper=hi,
                        n_cells=len(stable),dropped_nonfinite=len(items)-len(stable)))
    pd.DataFrame(effects).to_csv(output/'inference_ablation_cells.csv',index=False)
    pd.DataFrame(summaries,columns=['encoder','seed','mode','stratum','value','gain','lower','upper','n_cells','dropped_nonfinite']).to_csv(output/'inference_ablation_summaries.csv',index=False)
    pd.DataFrame(blockers).to_csv(output/'blocked.csv',index=False)
    if norms:
        all_norms=pd.concat(norms,ignore_index=True)
        norm_intervals=[]
        for job in plan['jobs']:
            path=replay_root/f"{job['encoder']}_seed{job['seed']}"/'full/correction_norms_INTERNAL.parquet'
            if not path.exists():
                continue
            raw=pd.read_parquet(path)
            for (cohort,outcome,term),g in raw.groupby(['cohort','outcome','term']):
                key=(split,job['seed'],cohort)
                if key not in context['draws']:
                    continue
                mapping,counts=context['draws'][key]
                mapping={str(k):v for k,v in mapping.items()}
                g=g.loc[g.subject_id.astype(str).isin(mapping)&np.isfinite(g.relative_change)]
                if g.empty:
                    continue
                w=counts[:,[mapping[str(s)] for s in g.subject_id]]
                denominator=w.sum(axis=1)
                averages=np.divide(w@g.relative_change.to_numpy(),denominator,
                    out=np.full(len(w),np.nan),where=denominator>0)
                lo,hi=interval(averages)
                norm_intervals.append(dict(encoder=job['encoder'],seed=job['seed'],cohort=cohort,
                    outcome=outcome,term=term,mean=g.relative_change.mean(),lower=lo,upper=hi,n=len(g)))
        pd.DataFrame(norm_intervals).to_csv(output/'correction_norm_intervals.csv',index=False)
        all_norms.to_csv(output/'correction_norms.csv',index=False)
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        all_norms['weighted_sum']=all_norms['mean']*all_norms['count']
        summary=all_norms.groupby(['encoder','seed','term'])[['weighted_sum','count']].sum()
        summary['relative_change']=summary.weighted_sum/summary['count']
        summary.reset_index().to_csv(output/'correction_norm_overall.csv',index=False)
        fig,ax=plt.subplots()
        for term,g in summary.reset_index().groupby('term'):
            g=g.groupby('encoder').relative_change.mean().reindex(['L','E','EA'])
            ax.plot(g.index,g.values,marker='o',label=term)
        ax.set_ylabel('Mean relative representation change');ax.legend()
        fig.savefig(output/'correction_size.pdf',bbox_inches='tight');plt.close(fig)
    (output/'manifest.json').write_text(json.dumps(dict(exploratory=True,split=split,
        plan=json.loads(Path(plan_path).read_text()),bootstrap='Exact main-analysis patient weights',
        interpretation='Inference reliance, not training without corrections',blocked=blockers),indent=2))
