"""Optional raw pretrained encoder maps; inference only, aggregate-only ZIP."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
from zipfile import ZipFile, ZIP_DEFLATED
import pandas as pd
import yaml


def run(root, story, metadata, execute=False):
    from opera.run.finish_figure_inputs import sha, immutable
    root,story,metadata=map(lambda x:Path(x).resolve(),[root,story,metadata])
    stages={}
    national=Path('data/paper_256_v3_national_20261111').resolve()
    sources=[('Cohort pretrained',Path('data/paper_256_v3/bundles/stage1').resolve(),'seed42/L/J_all_diseases_all_outcomes'),
             ('National pretrained',national/'direct_fits','seed42/E/J'),
             ('Adapted pretrained',national/'adapted_fits','seed42/EA/J')]
    root.mkdir(parents=True,exist_ok=True)
    private=root/'server_only';private.mkdir(exist_ok=True)
    meta=pd.read_parquet(metadata);meta['subject_id']=meta.subject_id.astype(str)
    meta=meta.loc[meta.split.isin(['tuning','held_out'])].copy()
    if meta.subject_id.duplicated().any():raise ValueError('Duplicate metadata patients')
    index=private/'prediction_origins_INTERNAL.parquet'
    if index.exists():
        pd.testing.assert_frame_equal(pd.read_parquet(index),meta.reset_index(drop=True))
    else:meta.reset_index(drop=True).to_parquet(index,index=False)
    identity=dict(metadata_sha256=sha(metadata),stages={})
    commands=[]
    for i,(label,bundlepath,name) in enumerate(sources):
        bundle=json.loads((bundlepath/'bundle.json').read_text());row,=[r for r in bundle['runs'] if r['name']==name]
        path=bundlepath/row['config_path']
        if sha(path)!=row['config_sha256']:raise ValueError('Frozen source configuration changed')
        cfg=yaml.safe_load(path.read_text())
        checkpoint=Path(cfg['dapt_ckpt']);vocab=Path(cfg['paths']['vocabulary'])
        parents={Path(p).parent for c in cfg['cohorts'].values() for p in c.get('subject_data_paths',[])}
        if len(parents)!=1:raise ValueError(f'{label}: extractor needs one physical sequence directory; found {parents}')
        parent,=parents
        for split in ['train','tuning']:
            if not (parent/f'subject_data_{split}.pt').is_file():raise FileNotFoundError(parent/f'subject_data_{split}.pt')
        output=private/f'encoder_{i}.npz';stages[label]=str(output)
        identity['stages'][label]=dict(config_sha256=sha(path),checkpoint_sha256=sha(checkpoint),vocabulary_sha256=sha(vocab))
        commands.append([sys.executable,'-u','-m','opera.run.extract_patient_embeddings',
            '--checkpoint',str(checkpoint),'--vocabulary',str(vocab),'--subject-data-dir',str(parent),
            '--index-table',str(index),'--output',str(output),'--batch-size','1','--num-workers','0',
            '--pooling','cls_last','--attention-backend','sdpa'])
    immutable(root/'intent.yaml',identity)
    print(json.dumps({'commands':commands,'scope':'Optional inference only; no training; NPZ stays server-side'},indent=2),flush=True)
    if not execute:return
    for (label,path),cmd in zip(stages.items(),commands):
        if Path(path).is_file() and Path(path).with_suffix('.metadata.json').is_file():
            receipt=json.loads(Path(path).with_suffix('.metadata.json').read_text())['embedding_extraction']
            if receipt['checkpoint_hash']!=identity['stages'][label]['checkpoint_sha256'] or receipt['index_table_hash']!=sha(index):
                raise ValueError('Existing embedding identity differs')
        else:subprocess.run(cmd,check=True)
    import umap  # Require UMAP; no silent PCA substitute for the story projection.
    from opera.run.build_paper_embedding_tables import embedding_comparison
    from opera.visualization.encoder_embedding_comparison import map_grids
    coords,_=embedding_comparison(stages,meta,method='umap')
    expected=set(pd.read_csv(story/'server_only/joint_encoder_coordinates.csv',dtype={'patient_id':str}).patient_id)
    if set(coords.patient_id)!=expected:raise ValueError('Pretrained maps do not cover exactly the story held-out roster')
    # Only density/disease layers are exported. Dummy follow-up is never reported.
    survival=pd.DataFrame({'patient_id':sorted(expected),'time_days':366.,'event':0})
    cells,gridmeta=map_grids(coords,survival,grid=800,bandwidth=.009,min_count=5,min_support=25)
    cells=cells.loc[cells.layer.isin(['count','disease'])]
    cells.to_csv(root/'pretrained_encoder_grids.csv.gz',index=False)
    gridmeta.to_csv(root/'grid_meta.csv',index=False)
    (root/'manifest.json').write_text(json.dumps(dict(status='complete',scope='Raw pretrained encoder inference; no outcome training',
        identities=identity,projection_fit_split='tuning',display_split='held_out',patients=len(expected),
        layers=['count','disease'],grid=800,bandwidth=.009,min_count=5),indent=2))
    target=root/'download_pretrained_encoder_grids.zip'
    with ZipFile(target,'w',ZIP_DEFLATED) as z:
        for name in ['pretrained_encoder_grids.csv.gz','grid_meta.csv','manifest.json']:z.write(root/name,name)
    print(f'Download: {target}')

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-dir',type=Path,default=Path('data/pretrained_encoder_grids_20261120'))
    p.add_argument('--story-root',type=Path,default=Path('data/story_closure_20261116'))
    p.add_argument('--metadata',type=Path,default=Path('data/clinical_registry_refresh/20261111/prepared_clinical/patient_metadata.parquet'))
    p.add_argument('--execute',action='store_true')
    a=p.parse_args();run(a.output_dir,a.story_root,a.metadata,a.execute)
