"""Exploratory analysis 5: paired pooling gaps, without fitting missing arms."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from opera.analysis.representation_absorption import bucket_predictions, cell_values, interval
from opera.evaluation.distributed_analysis import load_pickle
from opera.evaluation.secondary_analysis import CELL

# D pools diseases; O pools outcomes and is deliberately not substituted here.
DEFAULT_ARMS = {e: {'pooled': f'D_{e}', 'joint': f'J_{e}', 'specialist': f'S_{e}'}
                for e in ['L', 'E', 'EA']}


def pooling_quantities(values, arms):
    result = {}
    for encoder, roles in arms.items():
        for scope in ['pooled', 'joint']:
            a, b = values[roles[scope]], values[roles['specialist']]
            result[encoder, scope, 'gap'] = (a[0]-b[0], a[1]-b[1])
    if 'L' in arms:
        for encoder in arms:
            if encoder == 'L':
                continue
            for scope in ['pooled', 'joint']:
                a, b = result['L', scope, 'gap'], result[encoder, scope, 'gap']
                result[encoder, scope, 'closure'] = (a[0]-b[0], a[1]-b[1])
    return result


def run(prepared, output, split='tuning', cell_metadata=None, arms=None, held_out_ready=False):
    if split == 'held_out' and not held_out_ready:
        raise ValueError('Held-out requires --held-out-ready after main evaluation completes')
    prepared, output = Path(prepared), Path(output)
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    arms = DEFAULT_ARMS if arms is None else arms
    if 'L' not in arms or any(set(r) != {'pooled','joint','specialist'} for r in arms.values()):
        raise ValueError('Require local arms and explicit pooled/joint/specialist roles')
    wanted = {m for roles in arms.values() for m in roles.values()}
    context = load_pickle(prepared/'context_INTERNAL.pkl')
    directory = bucket_predictions(prepared, output, split, wanted=wanted)
    # Determine availability per seed before fixing a common cell roster.
    available = {}
    for path in directory.glob('*.parquet'):
        d = pd.read_parquet(path, columns=['seed','model'])
        for seed,g in d.groupby('seed'):
            available.setdefault(int(seed),set()).update(g.model)
    complete, blocked = {}, []
    for seed, models in sorted(available.items()):
        complete[seed] = {}
        for encoder, roles in arms.items():
            missing = set(roles.values())-models
            if missing:
                blocked.append(dict(seed=seed,encoder=encoder,reason='missing_existing_arms',
                                    missing_models=','.join(sorted(missing))))
            else:
                complete[seed][encoder] = roles
    rows, vectors, exclusions = [], {}, []
    for bucket in sorted({p.name.split('_')[0] for p in directory.glob('*.parquet')}):
        d = pd.concat([pd.read_parquet(p) for p in sorted(directory.glob(bucket+'_*.parquet'))],ignore_index=True)
        for key,cell in d.groupby(CELL,sort=True):
            identity = dict(zip(CELL,key))
            selected = complete[int(identity['seed'])]
            if 'L' not in selected:
                exclusions.append({**identity,'reason':'local_reference_arms_missing'})
                continue
            pairs = {e+'_'+scope:(roles[scope],roles['specialist'])
                     for e,roles in selected.items() for scope in ['pooled','joint']}
            values,status = cell_values(cell,context['draws'][key[:3]],arms=pairs,
                                       include_halves=False,**context['performance_support'])
            if values is None:
                exclusions.append({**identity,'reason':status});continue
            quantities = pooling_quantities(values['full'],selected)
            # Identical supported, finite-draw cell roster across scopes and encoders.
            if not all(np.isfinite(v[0]) and np.isfinite(v[1]).all() for v in quantities.values()):
                exclusions.append({**identity,'reason':'nonfinite_paired_bootstrap'});continue
            for (encoder,scope,quantity),(point,draws) in quantities.items():
                index=len(rows);vectors[index]=draws
                rows.append({**identity,'encoder':encoder,'scope':scope,'quantity':quantity,'gain':point,'draw_index':index})
    cells=pd.DataFrame(rows)
    if cell_metadata and not cells.empty:
        path=Path(cell_metadata)
        meta=pd.read_parquet(path) if path.suffix=='.parquet' else pd.read_csv(path)
        keys=['cohort','outcome','horizon_days']
        if meta.duplicated(keys).any():
            raise ValueError('Duplicate cell metadata keys')
        if 'n_events_train' in meta:
            cells=cells.merge(meta[keys+['n_events_train']],on=keys,how='left',validate='many_to_one',sort=False)
    summaries=[]
    if not cells.empty:
        for seed,g in cells.groupby('seed'):
            base=g[['cohort','outcome','horizon_days']+(['n_events_train'] if 'n_events_train' in g else [])].drop_duplicates()
            if 'n_events_train' in base:
                base['events_quartile']=pd.qcut(base.n_events_train,4,labels=False,duplicates='drop')
                cells.loc[g.index,'events_quartile']=g.merge(base,on=['cohort','outcome','horizon_days','n_events_train'],how='left',validate='many_to_one').events_quartile.to_numpy()
        for identity,g in cells.groupby(['seed','encoder','scope','quantity']):
            strata=[('overall','all',g)]
            if 'events_quartile' in g:
                strata += [('events_quartile',str(q),part) for q,part in g.groupby('events_quartile')]
            for stratum,value,part in strata:
                draws=np.mean([vectors[int(i)] for i in part.draw_index],axis=0)
                lo,hi=interval(draws)
                summaries.append(dict(zip(['seed','encoder','scope','quantity'],identity)) |
                    dict(stratum=stratum,value=value,gain=part.gain.mean(),lower=lo,upper=hi,n_cells=len(part)))
    summary=pd.DataFrame(summaries)
    cells.to_csv(output/'pooling_cells.csv',index=False)
    summary.to_csv(output/'pooling_summaries.csv',index=False)
    pd.DataFrame(blocked,columns=['seed','encoder','reason','missing_models']).to_csv(output/'blocked.csv',index=False)
    pd.DataFrame(exclusions).to_csv(output/'roster_exclusions.csv',index=False)
    if not summary.empty:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        seeds=sorted(summary.seed.unique())
        fig,axes=plt.subplots(1,len(seeds),squeeze=False,figsize=(5*len(seeds),4))
        for ax,seed in zip(axes[0],seeds):
            for scope in ['pooled','joint']:
                part=summary[(summary.seed==seed)&(summary.scope==scope)&(summary.quantity=='gap')&(summary.stratum=='overall')]
                for _,r in part.iterrows():
                    x=['L','E','EA'].index(r.encoder)+(-.05 if scope=='pooled' else .05)
                    ax.vlines(x,r.lower,r.upper);ax.plot(x,r.gain,'o',label=scope if r.encoder=='L' else None)
            ax.set(xticks=[0,1,2],xticklabels=['Local','National','Adapted'],ylabel='Pooled − specialist AUC',title=f'{split}, seed {seed}')
            ax.axhline(0,color='grey',lw=.7);ax.legend()
        fig.tight_layout();fig.savefig(output/'pooling_gap.pdf');fig.savefig(output/'pooling_gap.png',dpi=180);plt.close(fig)
    manifest=dict(exploratory=True,analysis=5,split=split,arms=arms,available_arms=complete,
                  main_bootstrap_reused=True,aggregation='equal-cell',input_hashes=context.get('hashes',{}),
                  implementation_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  metadata_sha256=hashlib.sha256(Path(cell_metadata).read_bytes()).hexdigest() if cell_metadata else None,
                  missing_event_metadata='n_events_train' not in cells,
                  limitation='Missing encoder specialists cannot be replaced by local specialists; no fits launched.')
    source_recipe=prepared/'source_analysis.yaml'
    if source_recipe.exists():
        import yaml
        config=yaml.safe_load(source_recipe.read_text())
        manifest['source_analysis_sha256']=hashlib.sha256(source_recipe.read_bytes()).hexdigest()
        manifest['source_artifacts']=[a for a in config.get('artifacts',[]) if a.get('model') in wanted]
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2,default=str))
    (output/'report.md').write_text('# Exploratory disease-pooling analysis\n\n'
        'Verdict: no mechanistic conclusion from pooling gaps alone.\n\n'
        +('```text\n'+summary.to_string(index=False)+'\n```' if not summary.empty else 'No common supported cells.')
        +'\n\nSee blocked.csv: unavailable specialist arms are not trained or replaced. '
        'Training-event quartiles require supplied n_events_train metadata. Seeds remain separate.\n')
    return manifest


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--prepared-dir',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--split',choices=['tuning','held_out'],default='tuning')
    p.add_argument('--held-out-ready',action='store_true')
    p.add_argument('--cell-metadata',type=Path)
    p.add_argument('--arms',type=Path,help='JSON encoder -> pooled/joint/specialist aliases, verified against source recipe')
    a=p.parse_args()
    run(a.prepared_dir,a.output_dir,a.split,a.cell_metadata,
        json.loads(a.arms.read_text()) if a.arms else None,a.held_out_ready)


if __name__=='__main__':
    main()
