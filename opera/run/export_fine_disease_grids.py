"""Fine-disease density layers on the existing adapted-joint story map."""
import argparse
import json
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED
import pandas as pd
import pyarrow.parquet as pq


def run(story, metadata, output):
    from opera.visualization.encoder_embedding_comparison import map_grids
    from opera.run.export_outcome_embedding_grids import sha
    story, output = Path(story), Path(output)
    if output.exists(): raise FileExistsError('Use fresh output directory')
    status=json.loads((story/'story_status.json').read_text())
    if status['embeddings']['status']!='complete':raise ValueError('Story embeddings incomplete')
    coords_path=story/'server_only/joint_encoder_coordinates.csv'
    coords=pd.read_csv(coords_path,dtype={'patient_id':str})
    coords=coords.loc[coords.model.eq('Adapted joint')].copy()
    meta=pd.read_parquet(metadata);meta['subject_id']=meta.subject_id.astype(str)
    if meta.subject_id.duplicated().any():raise ValueError('Duplicate metadata IDs')
    meta=meta.set_index('subject_id')
    if not set(coords.patient_id)<=set(meta.index):raise ValueError('Missing map-patient metadata')
    coords['disease']=coords.patient_id.map(meta.cohort_fine)
    if coords.disease.isna().any():raise ValueError('Missing fine disease')
    counts=coords.disease.value_counts()
    allowed=set(counts[counts>=25].index)
    refs=pd.read_csv(story/'ready/tables/embedding_grids_joint_meta.csv')
    ref,=refs.loc[refs.model.eq('Adapted joint')].to_dict('records')
    source=status['embeddings']['joint']['prediction_sources']['Adapted joint']
    survival=pq.read_table(source,columns=['subject_id','time_days','event'],filters=[('outcome','=','overall_survival')]).to_pandas().rename(columns={'subject_id':'patient_id'})
    survival.patient_id=survival.patient_id.astype(str)
    cells,gridmeta=map_grids(coords,survival,grid=800,bandwidth=.009,min_count=5,min_support=25,
                            frame=[ref[k] for k in ['x0','x1','y0','y1']])
    cells=cells.loc[cells.layer.eq('count') | (cells.layer.eq('disease') & cells.key.isin(allowed))]
    output.mkdir(parents=True)
    cells.to_csv(output/'disease_fine_grids.csv.gz',index=False)
    gridmeta.to_csv(output/'grid_meta.csv',index=False)
    rows=[]
    for fine,n in counts.items():
        groups=meta.loc[coords.loc[coords.disease.eq(fine),'patient_id'],'cohort_grouped'].unique()
        if len(groups)!=1:raise ValueError('Fine disease has ambiguous lineage')
        rows.append(dict(cohort_fine=fine,cohort_grouped=groups[0],n_patients=int(n),included=fine in allowed))
    pd.DataFrame(rows).to_csv(output/'disease_meta.csv',index=False)
    report=dict(status='complete',model='J_EA',min_support=25,min_count=5,grid=800,bandwidth=.009,
                coordinates_sha256=sha(coords_path),metadata_sha256=sha(metadata),
                excluded_subtypes=[r for r in rows if not r['included']])
    (output/'manifest.json').write_text(json.dumps(report,indent=2))
    target=output/'download_disease_fine_grids.zip'
    with ZipFile(target,'x',ZIP_DEFLATED) as z:
        for name in ['disease_fine_grids.csv.gz','grid_meta.csv','disease_meta.csv','manifest.json']:z.write(output/name,name)
    print(f'Download: {target}')

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--story-root',type=Path,default=Path('data/story_closure_20261116'))
    p.add_argument('--metadata',type=Path,default=Path('data/clinical_registry_refresh/20261111/prepared_clinical/patient_metadata.parquet'))
    p.add_argument('--output-dir',type=Path,default=Path('data/disease_fine_grids_20261120'))
    a=p.parse_args();run(a.story_root,a.metadata,a.output_dir)
