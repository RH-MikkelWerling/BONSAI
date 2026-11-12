"""Bounded reporting helpers; kept separate from live CPU worker implementations."""
from pathlib import Path
import uuid

import numpy as np
import pandas as pd

from opera.evaluation.identity_result_recovery import CSVResultTable


def annotate_gate(table, gate, *, gated):
    """Same gate merge without copying every payload column into a merged block."""
    if table.empty:
        return table
    keys=['split','seed','metric','horizon_days']
    if 'router_null_gate' in table:
        raise ValueError('Gate already annotated')
    if gate.empty:
        status=np.full(len(table),np.nan,dtype=object)
    else:
        narrow=pd.DataFrame({c:table[c].array for c in keys},copy=False)
        lookup=gate[keys+['status']].rename(columns={'status':'router_null_gate'})
        status=narrow.merge(lookup,on=keys,how='left',validate='many_to_one',sort=False).router_null_gate.to_numpy()
    scope=table.contrast.isin(gated).to_numpy()
    status[scope & pd.isna(status)]='missing_null_router_contrast'
    status[~scope]=np.nan
    result=pd.DataFrame({c:table[c].array for c in table},copy=False)
    result['router_null_gate']=pd.Series(status,dtype=object) if gate.empty else status
    return result


def spool_table(table,path):
    """Write a reporting-only table now and retain only its path and row count."""
    path=Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name('.'+path.name+'.'+uuid.uuid4().hex)
    try:
        table.to_csv(temporary,index=False,chunksize=50_000)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
    return CSVResultTable(path,len(table))


def spool_rarity(annotated,path,builder,chunk_rows=100_000):
    """Never construct the full expanded rarity table in coordinator memory."""
    path=Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name('.'+path.name+'.'+uuid.uuid4().hex)
    count=0
    try:
        if not len(annotated):
            builder(annotated).to_csv(temporary,index=False)
        for start in range(0,len(annotated),chunk_rows):
            part=builder(annotated.iloc[start:start+chunk_rows],chunk_rows=chunk_rows)
            part.to_csv(temporary,index=False,mode='a',header=start==0,chunksize=50_000)
            count+=len(part)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
    return CSVResultTable(path,count)


def coverage_bounded(frame,roster,chunk_rows=100_000):
    """Exact coverage counts without copying the whole prediction payload."""
    from collections import defaultdict
    required={'model','seed','cohort','outcome','horizon_days','status'}
    if 'split' in roster:
        required.add('split')
    keys=sorted(required-{'status'})
    if required-set(roster) or roster.duplicated(keys).any():
        raise ValueError('Coverage roster requires unique model/seed/task and status')
    numeric={'seed','horizon_days'}
    def keyed(table):
        result=table.copy()
        for key in keys:
            result[key]=pd.to_numeric(result[key]).astype(float) if key in numeric else result[key].astype(str)
        return result
    counts=defaultdict(lambda:[0,0.])
    for start in range(0,len(frame),chunk_rows):
        stop=min(start+chunk_rows,len(frame))
        mask=np.ones(stop-start,dtype=bool)
        if 'split' not in roster:
            mask &= frame['split'].iloc[start:stop].eq('held_out').to_numpy()
        if 'ipcw_weight' in frame:
            mask &= frame['ipcw_weight'].iloc[start:stop].gt(0).to_numpy()
        narrow=pd.DataFrame({key:frame[key].iloc[start:stop].to_numpy()[mask] for key in [*keys,'label']})
        if narrow.empty:
            continue
        grouped=keyed(narrow).groupby(keys,sort=False).label.agg(['size','sum'])
        for identity,row in grouped.iterrows():
            value=counts[identity]
            value[0]+=int(row['size']);value[1]+=float(row['sum'])
    rows=[dict(zip(keys,key))|dict(n_known_predictions=n,n_cases=cases) for key,(n,cases) in counts.items()]
    totals=pd.DataFrame(rows,columns=[*keys,'n_known_predictions','n_cases'])
    if totals.empty:
        n=np.zeros(len(roster),dtype=int);cases=n.copy()
    else:
        merged=keyed(roster[keys]).merge(totals,on=keys,how='left',validate='many_to_one')
        n=merged.n_known_predictions.fillna(0).astype(int).to_numpy()
        cases=merged.n_cases.fillna(0).astype(int).to_numpy()
    result=roster.reset_index(drop=True).copy()
    result['n_known_predictions']=n;result['n_cases']=cases;result['n_controls']=n-cases
    result['prediction_status']=np.where(n==0,'absent',np.where((cases==0)|(cases==n),'single_class','two_class'))
    result['auc_estimable']=(cases>0)&(cases<n)
    return result


def patient_subgroups_bounded(gains,metadata,specifications,draws,directory):
    """Publish the unchanged subgroup jobs without pandas sorting all gain rows."""
    import hashlib
    from opera.evaluation import distributed_analysis as da
    from opera.evaluation.secondary_analysis import CELL,_GAIN_METRICS
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    refs=da._draw_references(directory,draws)
    contrasts=sorted(gains.contrast.unique())
    def payloads():
        for code in contrasts:
            path=(directory/f'gain_{hashlib.sha256(str(code).encode()).hexdigest()}_INTERNAL.pkl').resolve()
            if not path.exists():
                positions=np.flatnonzero(gains.contrast.eq(code).to_numpy())
                part=pd.DataFrame({column:gains[column].iloc[positions].array for column in gains},
                                  index=gains.index.take(positions),copy=False)
                da.save_pickle(path,part)
                del part,positions
            for column,spec in specifications.items():
                yield str(path),metadata[['subject_id',column]],{column:spec},refs
    results=list(da.distributed_parts(directory,payloads(),len(contrasts)*len(specifications),
                                     dict(kind='patient_subgroups',specifications=specifications)))
    combined=[]
    for index in [0,1]:
        table=pd.concat([part[index] for part in results],ignore_index=True)
        if not table.empty:
            table['_covariate_order']=table.covariate.map({c:i for i,c in enumerate(specifications)})
            table['_metric_order']=table.metric.map({m:i for i,m in enumerate(_GAIN_METRICS)})
            keys=(CELL+['contrast','subgroup']) if index==0 else ['split','seed','horizon_days','contrast','subgroup']
            table=table.sort_values(['_covariate_order',*keys,'_metric_order'],kind='stable').drop(
                columns=['_covariate_order','_metric_order']).reset_index(drop=True)
        combined.append(table)
    return tuple(combined)
