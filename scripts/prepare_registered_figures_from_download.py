"""Translate downloaded aggregates for the existing registered figure renderers.

Does not design new panels, fit a support curve or modify analysis results.
"""
import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd
import yaml

from opera.run.build_paper_plot_tables import paired_cells, gain_sources_cells, CONTRAST_CODES, GAIN_SOURCE_STEPS
from opera.visualization.paper_taxonomy import to_display


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bundle',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,required=True)
    a=p.parse_args()
    if a.output_dir.exists():
        raise FileExistsError(a.output_dir)
    source_manifest=json.loads((a.bundle/'bundle_manifest.json').read_text())
    files=['contrasts.csv.gz','contrasts_all_horizons.csv.gz','effect_summaries.csv.gz']
    hashes={}
    for name in files:
        path=a.bundle/'tables'/name
        with path.open('rb') as handle:
            hashes[name]=hashlib.file_digest(handle,'sha256').hexdigest()
        if hashes[name]!=source_manifest['files']['tables/'+name]:
            raise ValueError(f'Checksum mismatch: {name}')
    wanted=set(CONTRAST_CODES)|set(GAIN_SOURCE_STEPS)
    raw=[];metadata=[]
    for d in pd.read_csv(a.bundle/'tables/contrasts.csv.gz',chunksize=100_000,low_memory=False):
        d=d[(d.split=='held_out')&(d.seed==42)&(d.metric=='auc')&(d.horizon_days==365)]
        if d.empty:
            continue
        metadata.append(d[['cohort','outcome','horizon_days','cohort_group','endpoint_group',
                           'n_events_train','n_patients_train']].drop_duplicates())
        raw.append(d.loc[d.contrast.isin(wanted)])
    metadata=pd.concat(metadata,ignore_index=True).drop_duplicates()
    if metadata.duplicated(['cohort','outcome','horizon_days']).any():
        raise ValueError('Conflicting cell metadata')
    meta_all=pd.concat([metadata,metadata.assign(horizon_days=0)],ignore_index=True)
    pooled=pd.read_csv(a.bundle/'tables/contrasts_all_horizons.csv.gz',low_memory=False)
    pooled=pooled[(pooled.split=='held_out')&(pooled.seed==42)&(pooled.metric=='auc')&pooled.contrast.isin(wanted)]
    # Match the canonical converter's one-year-support proxy for pooled cells.
    pooled=pooled.merge(meta_all,on=['cohort','outcome','horizon_days'],how='left',validate='many_to_one')
    cells=pd.concat([*raw,pooled],ignore_index=True)
    effects=pd.read_csv(a.bundle/'tables/effect_summaries.csv.gz',low_memory=False)
    effects=effects[(effects.split=='held_out')&(effects.seed==42)&(effects.metric=='auc')]
    paired=paired_cells(cells,meta_all,seed=42)
    selected=effects[effects.contrast.isin(CONTRAST_CODES)].copy()
    selected['contrast']=selected.contrast.map(CONTRAST_CODES)
    for stratum,prefix,keys in [
        ('all_cells','panel',['contrast','metric','horizon_days']),
        ('endpoint_group','group',['contrast','metric','horizon_days','endpoint_group']),
        ('cohort','disease',['contrast','metric','horizon_days','cohort'])]:
        part=selected[selected.stratum==stratum].copy()
        if stratum=='endpoint_group':
            part['endpoint_group']=part.stratum_value
        elif stratum=='cohort':
            part['cohort']=part.stratum_value
            part=to_display(part)
        part=part.rename(columns={c:f'{prefix}_{suffix}' for c,suffix in
            [('gain','mean'),('lower','lower'),('upper','upper')]})
        columns=[f'{prefix}_{suffix}' for suffix in ['mean','lower','upper']]
        paired=paired.merge(part[keys+columns],on=keys,how='left',validate='many_to_one')
    gain=gain_sources_cells(cells,effects,meta_all,seed=42)
    a.output_dir.mkdir(parents=True)
    tables=a.output_dir/'plot_tables';tables.mkdir()
    paired.to_csv(tables/'paired_cells.csv',index=False)
    gain.to_csv(tables/'gain_sources_cells.csv',index=False)
    metadata.to_csv(tables/'cell_metadata.csv',index=False)
    recipe=dict(required_figures=['fig_pooling','fig_gain_sources'],figures=[
        dict(id='fig_pooling',function='figure_pooling',
             tables=dict(cells='plot_tables/paired_cells.csv')),
        dict(id='fig_gain_sources',function='figure_gain_sources',
             tables=dict(cells='plot_tables/gain_sources_cells.csv'))])
    (a.output_dir/'figures.yaml').write_text(yaml.safe_dump(recipe,sort_keys=False))
    missing=['Pooling panel C: hierarchical posterior/support curve absent from download; '
             'the registered renderer can show the cell scatter, but no replacement curve is fitted.']
    absent=set(GAIN_SOURCE_STEPS)-set(cells.contrast)
    if absent:
        missing.append('Gain-source rows absent from download: '+', '.join(sorted(absent)))
    (a.output_dir/'input_manifest.json').write_text(json.dumps(dict(
        split='held_out',seed=42,source_hashes=hashes,
        implementation_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        registered_functions=['opera.visualization.paper_figures.figure_pooling',
                              'opera.visualization.opera_design_figures.figure_gain_sources'],
        missing_inputs=missing,new_support_model_fitted=False),indent=2))
    (a.output_dir/'README.md').write_text('# Registered figure layouts, held-out inputs\n\n'
        'Uses the existing manuscript rendering functions, not the ad hoc summary layouts.\n\n'
        +'\n'.join('- '+m for m in missing)+'\n')
    print(a.output_dir)


if __name__=='__main__':
    main()
