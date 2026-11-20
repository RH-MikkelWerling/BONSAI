"""Allowlisted aggregate bundle for both vocabulary-control checkpoints and fits."""
import argparse
import json
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED
import pandas as pd
from opera.run.prepare_epoch20_joint_finetune import checkpoint_record


def package(root):
    from bonsai.functional.checkpointing import verified_training_selection
    root=Path(root)
    receipt=json.loads((root/'preparation.json').read_text())
    metrics=root/'pretraining/version_0/metrics.csv'
    table=pd.read_csv(metrics)
    allowed=['epoch','completed_epoch','step','val/loss','val/code_loss','val/value_bin_loss','val/value_regression_loss','train/loss','audit/tokens_this_epoch',
             'audit/examples_this_epoch','audit/optimizer_steps_this_epoch','audit/tokens_cumulative']
    table[[c for c in allowed if c in table]].to_csv(root/'pretraining_curves.csv',index=False)
    report=dict(status='complete',scope='Exploratory vocabulary/lab-binning control; no primary aliases changed',
                checkpoints={},selection={},source_audit=receipt['audit'],
                finetuning_changes=receipt['finetuning_changes'])
    files=[('preparation.json',root/'preparation.json'),('pretraining_curves.csv',root/'pretraining_curves.csv')]
    for epoch in [10,20]:
        checkpoint=root/'pretraining/version_0'/('epoch_010.ckpt' if epoch==10 else 'last.ckpt')
        report['checkpoints'][str(epoch)]=checkpoint_record(checkpoint)
        bundle=json.loads((root/f'joint{epoch}/bundle/bundle.json').read_text());row,=bundle['runs']
        report['selection'][str(epoch)]=verified_training_selection(root/f'joint{epoch}/neural_runs'/row['run_directory'])
        for split in ['tuning','held_out']:
            folder=root/f'joint{epoch}/predictions'/row['run_directory']/split
            for name in ['direct_cr_metrics.csv','direct_cr_pooled_metrics.csv','direct_cr_family_metrics.csv','direct_cr_summary.json']:
                path=folder/name
                if path.suffix=='.csv':
                    columns=pd.read_csv(path,nrows=0).columns
                    if {'subject_id','patient_id','cpr','pnr'}&set(columns.str.lower()):raise ValueError('Patient data in aggregate table')
                if not path.is_file():raise FileNotFoundError(path)
                files.append((f'NV{epoch}/{split}/{name}',path))
    results=root/'matched/comparisons_vocab_control/results'
    if not (results/'analysis_manifest.json').is_file():raise ValueError('Matched vocabulary-control analysis not complete')
    for name in ['effect_summaries.csv','effect_summaries_disease_balanced.csv','panel_estimates.csv','contrasts.csv','contrast_coverage.csv']:
        path=results/name
        if not path.is_file():raise FileNotFoundError(path)
        if {'subject_id','patient_id','cpr','pnr'}&set(pd.read_csv(path,nrows=0).columns.str.lower()):raise ValueError('Patient data in matched aggregate')
        files.append((f'matched/{name}',path))
    manifest=root/'manifest.json';manifest.write_text(json.dumps(report,indent=2));files.append(('manifest.json',manifest))
    target=root/'download_vocab_control.zip'
    with ZipFile(target,'w',ZIP_DEFLATED) as z:
        for name,path in files:z.write(path,name)
    print(f'Download: {target}')

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,default=Path('data/vocabulary_control_20261120'))
    package(p.parse_args().root)
