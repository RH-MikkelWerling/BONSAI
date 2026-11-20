"""Read-only pretraining cutoff provenance and aggregate sequence-date audit."""
import argparse
from datetime import datetime
import gc
import json
from pathlib import Path
import numpy as np
import pandas as pd
import yaml


def sequence_dates(pool,allowed,boundary):
    from opera.run.export_outcome_rates_by_year import canonical_patient_ids
    patients=tokens_after=patients_after=0;latest=None
    for subject in pool:
        key=canonical_patient_ids(pd.Series([subject['subject_id']])).iloc[0]
        if key not in allowed:continue
        times=np.asarray(subject['abspos'],dtype=float)
        if not np.isfinite(times).all():raise ValueError('Non-finite source event times')
        patients+=1;count=int((times>=boundary).sum());tokens_after+=count;patients_after+=count>0
        if len(times):latest=float(times.max()) if latest is None else max(latest,float(times.max()))
    return dict(population_patients_in_shard=patients,tokens_on_or_after_2022=tokens_after,
                patients_with_tokens_on_or_after_2022=patients_after,
                latest_stored_date=(pd.Timestamp('1970-01-01')+pd.Timedelta(hours=latest)).isoformat() if latest is not None else None)


def audit(root,scan):
    from omegaconf import OmegaConf
    if not OmegaConf.has_resolver('version'):OmegaConf.register_new_resolver('version',lambda:0)
    paths=[p for p in [root/'hydra/config.yaml',root/'.hydra/config.yaml'] if p.is_file()]
    if not paths:raise FileNotFoundError(f'No saved Hydra configuration at {root}')
    result=dict(root=str(root),saved_configs=[])
    sidecar=root/'version_0/checkpoint_metadata.json'
    if sidecar.is_file():
        saved=json.loads(sidecar.read_text());notes=saved.get('checkpoint_metadata',{})
        result['checkpoint_notes']={
            'training_stage':notes.get('training_stage'),
            'pretraining_window_policy':notes.get('pretraining_window_policy'),
            'completed_epochs':saved.get('extra',{}).get('completed_epochs'),
            'explicit_cutoff_fields':{k:v for k,v in notes.items() if 'cutoff' in k}}
    resolved=[]
    for path in paths:
        cfg=OmegaConf.to_container(OmegaConf.load(path),resolve=True)
        overrides=path.parent/'overrides.yaml'
        override_values=yaml.safe_load(overrides.read_text()) if overrides.is_file() else []
        result['saved_configs'].append(dict(path=str(path),cutoff_date=cfg['training'].get('cutoff_date'),
            dataset=cfg.get('dataset'),cutoff_overrides=[str(v) for v in (override_values or []) if 'cutoff' in str(v)],
            population=cfg['paths']['population'],train_split=cfg['paths']['train_split'],val_split=cfg['paths']['val_split']))
        resolved.append(cfg)
    if scan:
        import torch
        from bonsai.functional.features import compute_abspos
        from opera.run.export_outcome_rates_by_year import canonical_patient_ids
        cfg=resolved[0];allowed=set(canonical_patient_ids(pd.read_csv(cfg['paths']['population']).subject_id))
        boundary=float(compute_abspos(datetime(2022,1,1)))
        result['sequence_dates']={}
        for split in ['train_split','val_split']:
            print(f'Auditing stored times: {root.name}/{split}',flush=True)
            pool=torch.load(cfg['paths'][split],map_location='cpu',weights_only=False)
            result['sequence_dates'][split]=sequence_dates(pool,allowed,boundary)
            del pool;gc.collect()
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--local-root',type=Path,default=Path('data/models/daly_care_t100_joined_bins_with_sor_final_v2/pretrain_scaled_time2vec_hidden256'))
    p.add_argument('--adapted-root',type=Path,default=Path('data/paper_256_v3_national_20261111/external_dapt'))
    p.add_argument('--scan-sequences',action='store_true')
    p.add_argument('--output',type=Path,default=Path('logs/pretraining_cutoff_audit_20261120.json'))
    a=p.parse_args()
    if a.scan_sequences:
        import subprocess,sys
        subprocess.run([sys.executable,'scripts/check_results_resources.py','--minimum-free-gib','96','--require-no-coordinator'],check=True)
    report=dict(scope='Read-only aggregate audit; stored timestamps are not proof of sampled training windows',
        local=audit(a.local_root,a.scan_sequences),adapted=audit(a.adapted_root,a.scan_sequences))
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))
    print('Download:',a.output)

if __name__=='__main__':main()
