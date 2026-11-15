"""Exploratory summary previews; these are not the registered manuscript layouts.

For manuscript plates use prepare_registered_figures_from_download.py followed
by opera.run.render_paper_results, which invokes the existing figure functions.
"""
import argparse
import hashlib
import json
from pathlib import Path
import textwrap

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.ticker import MaxNLocator, FormatStrFormatter


COLORS=['#355f8d','#b18b4d','#3e8a87']
DOMAINS={
    'Acute care and other events':'Acute care',
    'Calcium and magnesium':'Calcium / magnesium',
    'Cardiovascular and thrombotic events':'Cardiovascular / thrombosis',
    'Disease course and later malignancy':'Disease course / malignancy',
    'Erythroid findings and transfusion':'Erythroid / transfusion',
    'Infection and antimicrobial use':'Infection / antimicrobials',
    'Liver injury and coagulation':'Liver / coagulation',
    'Lymphocyte and eosinophil counts':'Lymphocytes / eosinophils',
    'Metabolic and cell-turnover markers':'Metabolic / cell turnover',
    'Neutrophil, white-cell and platelet findings':'Neutrophils / WBC / platelets',
    'Renal injury and support':'Kidney',
    'Sodium and potassium':'Sodium / potassium'}


def select(frame,contrast,*,stratum='all_cells',value=None,horizon=0):
    d=frame[(frame.split=='held_out')&(frame.seed==42)&(frame.metric=='auc')&
            (frame.contrast==contrast)&(frame.stratum==stratum)&(frame.horizon_days==horizon)]
    if value is not None:
        d=d[d.stratum_value.astype(str)==str(value)]
    return d


def one(frame,code,**kwargs):
    d=select(frame,code,**kwargs)
    if len(d)!=1:
        raise ValueError(f'Expected one saved paired summary: {code}, {kwargs}; got {len(d)}')
    return d.iloc[0]


def forest(ax,frame,items,title,color=COLORS[0]):
    for y,(label,code) in enumerate(items):
        r=one(frame,code)
        ax.hlines(y,r.lower,r.upper,color=color,lw=2)
        ax.plot(r.gain,y,'o',color=color,ms=5)
    ax.set_yticks(range(len(items)),[label for label,_ in items])
    ax.invert_yaxis();ax.axvline(0,color='#8b9299',lw=.8)
    ax.set_title(title,loc='left',fontweight='bold',fontsize=10)
    ax.set_xlabel('Mean AUC difference (paired 95% interval)')
    ax.xaxis.set_major_locator(MaxNLocator(4))
    ax.xaxis.set_major_formatter(FormatStrFormatter('%.3f'))
    ax.spines[['top','right']].set_visible(False)


def save(fig,output,name):
    fig.savefig(output/(name+'.pdf'),bbox_inches='tight')
    fig.savefig(output/(name+'.png'),dpi=220,bbox_inches='tight')
    plt.close(fig)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bundle',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--cell-cache',type=Path,help='Previously derived pooling_heatmap_cells.csv; parent source manifest must match')
    a=p.parse_args()
    if a.output_dir.exists():
        raise FileExistsError(a.output_dir)
    manifest=json.loads((a.bundle/'bundle_manifest.json').read_text())
    needed=['effect_summaries.csv.gz','effect_summaries_disease_balanced.csv.gz',
            'contrasts.csv.gz','contrasts_all_horizons.csv.gz']
    for name in needed:
        with (a.bundle/'tables'/name).open('rb') as f:
            actual=hashlib.file_digest(f,'sha256').hexdigest()
        if actual!=manifest['files']['tables/'+name]:
            raise ValueError(f'Source checksum mismatch: {name}')
    effects=pd.read_csv(a.bundle/'tables/effect_summaries.csv.gz',low_memory=False)
    balanced=pd.read_csv(a.bundle/'tables/effect_summaries_disease_balanced.csv.gz',low_memory=False)
    if set(effects.split)!={'held_out'} or set(effects.seed)!={42}:
        raise ValueError('Expected held-out seed 42')
    a.output_dir.mkdir(parents=True)
    plt.rcParams.update({'font.size':8,'axes.labelsize':8,'xtick.labelsize':7,
                         'ytick.labelsize':7,'pdf.fonttype':42})
    if a.cell_cache:
        cached_manifest=json.loads((a.cell_cache.parent/'manifest.json').read_text())
        if cached_manifest['sources']!={name:manifest['files']['tables/'+name] for name in needed}:
            raise ValueError('Derived cell cache comes from different source tables')
        cells=pd.read_csv(a.cell_cache)
        if cells.duplicated(['cohort','outcome']).any() or set(cells.split)!={'held_out'} or set(cells.seed)!={42}:
            raise ValueError('Invalid derived cell cache roster')
    else:
        # Recover only aggregate metadata; stream the large cell table.
        parts=[]
        cols=['split','seed','cohort','outcome','horizon_days','metric','endpoint_group','n_events_train','n_patients_train']
        for d in pd.read_csv(a.bundle/'tables/contrasts.csv.gz',usecols=cols,chunksize=100_000):
            d=d[(d.split=='held_out')&(d.seed==42)&(d.metric=='auc')&(d.horizon_days==365)]
            parts.append(d[['cohort','outcome','endpoint_group','n_events_train','n_patients_train']].drop_duplicates())
        meta=pd.concat(parts,ignore_index=True).drop_duplicates()
        if meta.duplicated(['cohort','outcome']).any():
            raise ValueError('Ambiguous saved one-year cell metadata')
        cells=pd.read_csv(a.bundle/'tables/contrasts_all_horizons.csv.gz',
                          usecols=['split','seed','cohort','outcome','contrast','metric','gain'])
        cells=cells[(cells.split=='held_out')&(cells.seed==42)&(cells.metric=='auc')&
                    (cells.contrast=='P5')&np.isfinite(cells.gain)]
        cells=cells.merge(meta,on=['cohort','outcome'],how='left',validate='many_to_one')
    if cells.endpoint_group.isna().any():
        raise ValueError('Some pooled cells lack domain metadata; do not silently drop them')
    cells.to_csv(a.output_dir/'pooling_heatmap_cells.csv',index=False)
    # Complete main pooling plate: all intervals use saved patient-bootstrap summaries.
    fig=plt.figure(figsize=(7.2,8.5))
    gs=fig.add_gridspec(2,2,height_ratios=[1.45,1],width_ratios=[1.15,1],
                        left=.09,right=.98,bottom=.11,top=.93,hspace=.55,wspace=.65)
    ax=fig.add_subplot(gs[0,0]);bx=fig.add_subplot(gs[0,1])
    cohort_order=cells.groupby('cohort').n_patients_train.max().sort_values().index.tolist()
    groups=sorted(cells.endpoint_group.unique())
    matrix=cells.pivot_table(index='cohort',columns='endpoint_group',values='gain',aggfunc='median').reindex(index=cohort_order,columns=groups)
    vmax=max(.01,float(np.nanmax(np.abs(matrix.to_numpy()))))
    cmap=plt.get_cmap('RdBu').copy();cmap.set_bad('#e8ecf1')
    im=ax.imshow(matrix,cmap=cmap,norm=TwoSlopeNorm(vcenter=0,vmin=-vmax,vmax=vmax),aspect='auto')
    ax.set_yticks(range(len(cohort_order)),cohort_order)
    ax.set_xticks(range(len(groups)),[DOMAINS.get(g,g).replace(' / ','\n') for g in groups],rotation=60,ha='right',fontsize=6)
    ax.set_title('A  Joint − specialist by disease and domain',loc='left',fontweight='bold',fontsize=10)
    fig.colorbar(im,ax=ax,fraction=.035,pad=.02,label='Median horizon-pooled AUC difference')
    for offset,code,label,color in zip([-.22,0,.22],['P1','P2','P5'],
            ['Pool diseases','Pool outcomes','Joint'],COLORS):
        d=select(effects,code,stratum='cohort').set_index('stratum_value')
        for y,cohort in enumerate(cohort_order):
            if cohort not in d.index:
                continue
            r=d.loc[cohort]
            bx.hlines(y+offset,r.lower,r.upper,color=color,lw=.65,alpha=.65)
            bx.plot(r.gain,y+offset,'o',color=color,ms=2.5,label=label if y==0 else None)
    bx.set_ylim(len(cohort_order)-.5,-.5);bx.set_yticks(range(len(cohort_order)),cohort_order)
    bx.axvline(0,color='grey',lw=.7);bx.set_xlabel('Mean AUC difference (paired 95% interval)')
    bx.set_title('B  Which supervision is shared?',loc='left',fontweight='bold',fontsize=10)
    bx.legend(frameon=False,fontsize=7,loc='upper center',bbox_to_anchor=(.5,-.13),ncol=3)
    cx=fig.add_subplot(gs[1,0]);dx=fig.add_subplot(gs[1,1])
    support=[('0–19','(-1.001, 19.0]'),('20–99','(19.0, 99.0]'),
             ('100–499','(99.0, 499.0]'),('500+','(499.0, inf]')]
    for x,(label,value) in enumerate(support):
        r=one(effects,'P5',stratum='support_bin',value=value)
        cx.vlines(x,r.lower,r.upper,color=COLORS[0],lw=2);cx.plot(x,r.gain,'o',color=COLORS[0])
    cx.set_xticks(range(4),[s[0] for s in support]);cx.axhline(0,color='grey',lw=.7)
    cx.set(xlabel='Training events (prespecified cell-support strata)',ylabel='Joint − specialist AUC')
    cx.set_title('C  Gain by training support',loc='left',fontweight='bold',fontsize=10)
    domain=select(effects,'P5',stratum='endpoint_group').sort_values('gain')
    for y,r in enumerate(domain.itertuples()):
        dx.hlines(y,r.lower,r.upper,color=COLORS[0],lw=1.4);dx.plot(r.gain,y,'o',color=COLORS[0],ms=3.5)
    dx.set_yticks(range(len(domain)),[DOMAINS.get(str(x),str(x)) for x in domain.stratum_value],fontsize=6)
    dx.axvline(0,color='grey',lw=.7);dx.set_xlabel('Mean AUC difference (paired 95% interval)')
    dx.set_title('D  Gains across clinical domains',loc='left',fontweight='bold',fontsize=10)
    fig.suptitle('Learning across diseases and outcomes: held-out results',fontsize=13,fontweight='bold',y=.985)
    save(fig,a.output_dir,'fig_pooling')
    # Matched architecture comparisons, with both pretraining stages visible.
    fig,axes=plt.subplots(2,2,figsize=(7.2,5.4),layout='constrained')
    forest(axes[0,0],effects,[('Corrections − joint','OFF_J'),('OPERA − corrections','OAOWN_OFF'),
        ('Routing − corrections','AB_OFF')],'A  Cohort encoder')
    forest(axes[0,1],effects,[('Corrections − joint','OFF_J_EA'),('OPERA − corrections','OAOWN_OFF_EA'),
        ('Routing − corrections','AB_OFF_EA')],'B  Adapted national encoder',COLORS[2])
    items=[('Cohort','EXT_ARCH_L'),('National','EXT_ARCH_E'),('Adapted','EXT_ARCH_EA')]
    forest(axes[1,0],effects,items,'C  OPERA − joint: equal-cell')
    forest(axes[1,1],balanced,items,'D  OPERA − joint: disease-balanced',COLORS[2])
    for ax in axes[0]:
        ax.set_xlim(-.008,.014)
    for ax in axes[1]:
        ax.set_xlim(-.008,.020)
    fig.suptitle('The value of architectural specialisation changes with pretraining',fontsize=12,fontweight='bold')
    save(fig,a.output_dir,'fig_gain_sources')
    # Horizon effects are already saved, with no interpolation or new resampling.
    fig,axes=plt.subplots(1,3,figsize=(10,3.3),sharey=True,layout='constrained')
    for ax,code,label,color in zip(axes,['P1','P2','P5'],['Pool diseases','Pool outcomes','Joint'],COLORS):
        d=effects[(effects.contrast==code)&(effects.metric=='auc')&(effects.stratum=='all_cells')&
                  effects.horizon_days.gt(0)].sort_values('horizon_days')
        if d.horizon_days.duplicated().any() or len(d)!=6:
            raise ValueError('Expected six unique horizon summaries')
        for x,r in enumerate(d.itertuples()):
            ax.vlines(x,r.lower,r.upper,color=color,lw=1.5);ax.plot(x,r.gain,'o',color=color)
        ax.set_xticks(range(6),d.horizon_days.astype(int).astype(str));ax.axhline(0,color='grey',lw=.7)
        ax.set(title=label,xlabel='Prediction horizon (days)')
    axes[0].set_ylabel('AUC gain over specialist (paired 95% interval)')
    fig.suptitle('Held-out gains across prediction horizons',fontweight='bold')
    save(fig,a.output_dir,'pooling_horizon_effects')
    effects.to_csv(a.output_dir/'saved_effect_summaries.csv.gz',index=False,compression='gzip')
    balanced.to_csv(a.output_dir/'saved_disease_balanced_effects.csv.gz',index=False,compression='gzip')
    (a.output_dir/'manifest.json').write_text(json.dumps(dict(split='held_out',seed=42,
        sources={name:manifest['files']['tables/'+name] for name in needed},
        implementation_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        derived_cell_sha256=hashlib.sha256((a.output_dir/'pooling_heatmap_cells.csv').read_bytes()).hexdigest(),
        reused_cell_cache_sha256=hashlib.sha256(a.cell_cache.read_bytes()).hexdigest() if a.cell_cache else None,
        newly_estimated_patient_intervals=False,
        limitations=['Heatmap shows descriptive medians of existing horizon-pooled cells.',
                     'Supported cell rosters differ between contrasts and horizons.',
                     'Support strata are saved summaries, not a fitted hierarchical curve.',
                     'External fraction-arm, clinical, route and attribution plots require separate outputs.']),indent=2))
    print(a.output_dir)


if __name__=='__main__':
    main()
