"""Exploratory survival decision curves, KM within each threshold-selected group.

Separate from the existing IPCW/tuning-mapped clinical decision-curve code:
this implements checklist K's training-mapped index and KM estimand explicitly.
"""
import numpy as np
import pandas as pd


def km_risk(time, event, horizon, weights=None):
    t,e=np.asarray(time,float),np.asarray(event,int)
    if not np.isfinite(t).all() or (t<0).any() or not np.isin(e,[0,1]).all():
        raise ValueError('KM decision curves require finite times and binary event/censoring; no competing deaths')
    w=np.ones((1,len(t))) if weights is None else np.atleast_2d(np.asarray(weights,float))
    if w.shape[1]!=len(t) or not np.isfinite(w).all() or (w<0).any():raise ValueError('Invalid patient multiplicities')
    if not len(t):return np.full(w.shape[0],np.nan)
    order=np.argsort(t,kind='stable');t,e,w=t[order],e[order],w[:,order]
    starts=np.r_[0,np.flatnonzero(np.diff(t))+1]
    mass=np.add.reduceat(w,starts,axis=1)
    deaths=np.add.reduceat(w*(e==1),starts,axis=1)
    risk=np.cumsum(mass[:,::-1],axis=1)[:,::-1]
    hazard=np.divide(deaths,risk,out=np.zeros_like(deaths),where=risk>0)
    by=t[starts]<=horizon
    survival=np.prod(1-hazard[:,by],axis=1)
    supported=(w[:,t>=horizon].sum(axis=1)>0) | (survival<=1e-14)
    return np.where(supported,1-survival,np.nan)


def net_benefit(time,event,risk,horizon,thresholds,weights=None):
    t,e,p=np.asarray(time,float),np.asarray(event,int),np.asarray(risk,float)
    if len(t)!=len(p) or not np.isfinite(p).all() or ((p<0)|(p>1)).any():raise ValueError('Missing/invalid risk; never substitute zero for unavailable prediction')
    w=np.ones((1,len(t))) if weights is None else np.atleast_2d(np.asarray(weights,float))
    total=w.sum(axis=1)
    values=[]
    for pt in thresholds:
        if not 0<pt<1:raise ValueError('Threshold outside (0,1)')
        hit=p>=pt; selected=w[:,hit].sum(axis=1)
        fraction=np.divide(selected,total,out=np.zeros_like(selected),where=total>0)
        observed=km_risk(t[hit],e[hit],horizon,w[:,hit]) if hit.any() else np.zeros(len(w))
        observed=np.where(selected==0,0,observed)
        values.append(fraction*(observed-(1-observed)*pt/(1-pt)))
    return np.array(values).T


def index_mapping(training,horizon):
    rows=[]
    for category,group in training.groupby('index_value',dropna=False):
        if pd.isna(category):raise ValueError('Missing index training category')
        value=km_risk(group.time_days,group.event,horizon)[0]
        if not np.isfinite(value):raise ValueError(f'Index training category unsupported at horizon: {category}')
        rows.append(dict(index_value=category,training_n=len(group),training_events=int(((group.event==1)&(group.time_days<=horizon)).sum()),risk=float(value)))
    return pd.DataFrame(rows)


def evaluate(frame,risks,horizon,thresholds,*,n_boot=1000,seed=2026,batch=32):
    """Same patient bootstrap for every strategy, refitting KM in each draw."""
    n=len(frame); t,e=frame.time_days.to_numpy(float),frame.event.to_numpy(int)
    if not n:raise ValueError('Empty population')
    prevalence=km_risk(t,e,horizon)[0]
    strategies={**risks,'treat_all':np.ones(n),'treat_none':np.zeros(n)}
    point={s:net_benefit(t,e,p,horizon,thresholds)[0] for s,p in strategies.items()}
    draws={s:[] for s in strategies};rng=np.random.default_rng(seed)
    for start in range(0,n_boot,batch):
        weights=rng.multinomial(n,np.full(n,1/n),size=min(batch,n_boot-start))
        for s,p in strategies.items():draws[s].append(net_benefit(t,e,p,horizon,thresholds,weights))
    draws={s:np.concatenate(v) for s,v in draws.items()}
    curves=[];differences=[]
    events=int(((e==1)&(t<=horizon)).sum())
    for strategy,values in point.items():
        for j,pt in enumerate(thresholds):
            curves.append(dict(strategy=strategy,threshold=float(pt),net_benefit=values[j],
                standardised_net_benefit=values[j]/prevalence if prevalence>0 else np.nan,
                net_true_positives_per_100=100*values[j],n=n,events=events,prevalence=prevalence))
    for strategy in strategies:
        if strategy=='R_EA':continue
        for j,pt in enumerate(thresholds):
            delta=draws['R_EA'][:,j]-draws[strategy][:,j];finite=np.isfinite(delta);valid=int(finite.sum())
            lo,hi=np.quantile(delta[finite],[.025,.975]) if valid>=.8*n_boot else [np.nan,np.nan]
            gain=point['R_EA'][j]-point[strategy][j]
            differences.append(dict(comparator=strategy,threshold=float(pt),gain=gain,lower=lo,upper=hi,
                net_true_positives_per_100=100*gain,lower_per_100=100*lo,upper_per_100=100*hi,
                n_boot_valid=valid,n_boot_requested=n_boot,status='estimated' if valid>=.8*n_boot else 'insufficient_bootstrap_support'))
    return pd.DataFrame(curves),pd.DataFrame(differences)
