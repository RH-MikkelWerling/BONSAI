"""Allowlisted aggregate-only review bundle for the adapted20 control."""
import argparse
import csv
import json
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED


def package(root):
    from bonsai.functional.checkpointing import verified_training_selection
    root = Path(root)
    bundle = json.loads((root/'joint/bundle/bundle.json').read_text())
    row, = bundle['runs']
    selection = verified_training_selection(root/'joint/neural_runs'/row['run_directory'])
    report = dict(status='complete', variant='J_EA_epoch20', training_selection=selection,
                  scientific_changes=['dapt_ckpt'], scope='Exploratory; no primary aliases changed')
    (root/'joint/review_summary.json').write_text(json.dumps(report, indent=2))
    # Validation curves are aggregate scalars; no model/patient objects in ZIP.
    curves=[]
    candidates = [('adapted10',Path('data/paper_256_v3_national_20261111/external_dapt/version_0/metrics.csv'),0),
                  ('cohort20',Path('data/models/daly_care_t100_joined_bins_with_sor_final_v2/pretrain_scaled_time2vec_hidden256/version_0/metrics.csv'),0),
                  ('adapted20',root/'pretraining/version_0/metrics.csv',json.loads((root/'preparation.json').read_text())['epoch_offset'])]
    import pandas as pd
    missing=[]
    for variant,path,offset in candidates:
        if not path.is_file(): missing.append(str(path));continue
        table=pd.read_csv(path)
        columns=[c for c in table if c in ['epoch','step','val/loss','train/loss','lr-AdamW']]
        part=table[columns].copy();part['variant']=variant
        if 'epoch' in part:part['completed_epoch']=part.epoch+1+offset
        curves.append(part)
    if curves:pd.concat(curves,ignore_index=True).to_csv(root/'pretraining_curves.csv',index=False)
    report['missing_curve_sources']=missing
    (root/'review_summary.json').write_text(json.dumps(report,indent=2))
    target=root/'download_adapted20.zip'
    with ZipFile(target,'w',ZIP_DEFLATED) as z:
        for name in ['preparation.json','review_summary.json','pretraining_curves.csv']:
            path=root/name
            if path.is_file():z.write(path,name)
        for split in ['tuning','held_out']:
            folder=root/'joint/predictions'/row['run_directory']/split
            for name in ['direct_cr_metrics.csv','direct_cr_pooled_metrics.csv','direct_cr_family_metrics.csv','direct_cr_summary.json']:
                path=folder/name
                if path.suffix=='.csv':
                    with path.open(newline='') as f:
                        if {'subject_id','patient_id','cpr','pnr'} & {c.lower() for c in next(csv.reader(f))}:
                            raise ValueError('Patient identifiers in aggregate file')
                z.write(path,f'{split}/{name}')
    print(f'Download: {target}')

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,default=Path('data/adapted20_control_20261120'))
    package(p.parse_args().root)
