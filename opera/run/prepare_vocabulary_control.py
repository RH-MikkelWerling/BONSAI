"""Isolated national-vocabulary control: strict source audit, no training launch."""
import argparse
import copy
import gc
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
import yaml
from opera.run.export_outcome_rates_by_year import canonical_patient_ids
from opera.run.finish_figure_inputs import sha,immutable


def frozen_config(bundle, name):
    bundle=Path(bundle);report=json.loads((bundle/'bundle.json').read_text())
    row,=[r for r in report['runs'] if r['name']==name]
    path=bundle/row['config_path']
    if sha(path)!=row['config_sha256']:raise ValueError('Frozen reference configuration changed')
    return yaml.safe_load(path.read_text()),path,row


def clone_pretraining(local, national, architecture):
    cfg=copy.deepcopy(local)
    cutoff=cfg['training'].get('cutoff_date')
    if not isinstance(cutoff,dict) or pd.Timestamp(**cutoff)!=pd.Timestamp('2022-01-01'):
        raise ValueError('Require explicit original pre-2022 cutoff at 2022-01-01')
    if cfg['training'].get('numeric_value_control','observed')!=national['training'].get('numeric_value_control','observed'):
        raise ValueError('National tokenized source has a different numeric-value policy')
    for key in ['max_seqlen','hidden_size','num_layers','num_attention_heads','bias','dropout','attention_dropout',
                'causal','value_bin_vocab_size','value_embedding_mode','abspos_encoding','rope_interleaved']:
        if key in architecture:cfg['model'][key]=architecture[key]
    cfg.pop('continuation',None)
    cfg['paths'].update(train_split=national['paths']['train_split'],val_split=national['paths']['val_split'],
                        vocab=national['paths']['vocab'],initialization_ckpt=None,initialization_vocab=None,
                        initialization_use_source_architecture=False,ckpt_path=None)
    cfg['training'].update(epochs=20,checkpoint_milestone_epochs=[10],auto_resume=True,early_stopping_patience=None,export_epoch_counters=True)
    cfg['seed']=42
    return cfg


def clone_joint(local, national, checkpoint):
    from opera.run.prepare_epoch20_joint_finetune import clone_joint_config
    cfg=clone_joint_config(local,checkpoint)
    if set(local['cohorts'])!=set(national['cohorts']):raise ValueError('Cohort definitions differ between local and national data releases')
    cfg['paths']['vocabulary']=national['paths']['vocabulary']
    for name in cfg['cohorts']:
        paths=national['cohorts'][name].get('subject_data_paths')
        if not paths:raise ValueError('Missing explicit national-tokenized subject paths')
        cfg['cohorts'][name]['subject_data_paths']=copy.deepcopy(paths)
    # Preserve LR multiplier, head/readout, labels, populations, selection, stopping.
    return cfg


def sequence_inventory(path, population, cutoff, vocabulary_size):
    import torch
    from bonsai.functional.censoring import censor_subject
    pool=torch.load(path,map_location='cpu',weights_only=False)
    rows={};seen=set();eligible_tokens=0
    for raw in pool:
        key=canonical_patient_ids(pd.Series([raw['subject_id']])).iloc[0]
        if key not in population:continue
        if key in rows:raise ValueError('Duplicate subject in physical pretraining shard')
        subject=censor_subject(raw,cutoff,inclusive=False)
        times=np.asarray(subject['abspos'],dtype=np.float64)
        codes=np.asarray(subject['code'])
        if not np.isfinite(times).all() or (times>=cutoff).any() or (np.diff(times)<0).any():raise ValueError('Invalid or post-cutoff pretraining times')
        if not np.issubdtype(codes.dtype,np.integer) or (codes<0).any() or (codes>=vocabulary_size).any():
            raise ValueError('Token IDs do not match the specified vocabulary')
        rows[key]=dict(tokens=len(codes),time_sha256=hashlib.sha256(np.sort(times).tobytes()).hexdigest())
        seen.update(int(x) for x in np.unique(codes));eligible_tokens+=len(codes)
    del pool;gc.collect()
    return rows,seen,eligible_tokens


def audit_sources(local,national,expected_patients):
    import torch
    from bonsai.functional.features import compute_abspos
    from datetime import datetime
    local_ids=canonical_patient_ids(pd.read_csv(local['paths']['population']).subject_id)
    national_ids=canonical_patient_ids(pd.read_csv(national['paths']['population']).subject_id)
    if local_ids.duplicated().any() or national_ids.duplicated().any():raise ValueError('Duplicate population IDs')
    if set(local_ids)!=set(national_ids) or len(local_ids)!=expected_patients:
        raise ValueError(f'Population mismatch: local={len(local_ids)}, national={len(national_ids)}, expected={expected_patients}')
    if local['training'].get('cutoff_date')!=national['training'].get('cutoff_date'):
        raise ValueError('Pretraining cutoffs differ')
    cutoff=float(compute_abspos(datetime(**local['training']['cutoff_date'])))
    vocabularies=[torch.load(cfg['paths']['vocab'],map_location='cpu',weights_only=False) for cfg in [local,national]]
    for vocab in vocabularies:
        if set(vocab.values())!=set(range(len(vocab))):raise ValueError('Vocabulary IDs must be contiguous')
    inventory={};national_seen=set();all_seen=set();union=set();parts={}
    for key in ['train_split','val_split']:
        old,_,_=sequence_inventory(local['paths'][key],set(local_ids),cutoff,len(vocabularies[0]))
        new,seen,tokens=sequence_inventory(national['paths'][key],set(local_ids),cutoff,len(vocabularies[1]))
        if set(old)!=set(new):raise ValueError(f'Patient split mismatch: {key}')
        changed=sum(old[k]['time_sha256']!=new[k]['time_sha256'] for k in old)
        if changed:raise ValueError(f'{changed} patients have different pre-cutoff event-time sequences in {key}; retokenize the original cohort records before training this control')
        if union & set(new):raise ValueError('Physical training/validation patient overlap')
        union.update(new);all_seen.update(seen)
        if key=='train_split':national_seen=seen
        parts[key]=dict(patients=len(new),eligible_tokens=tokens,matched_event_time_records=True)
    if union!=set(local_ids):raise ValueError('Pretraining shards do not cover the full intended population')
    clinical=[int(i) for token,i in vocabularies[1].items() if not str(token).startswith('[')]
    return dict(status='verified',population_patients=len(union),cutoff_date=local['training']['cutoff_date'],
                splits=parts,national_vocabulary_size=len(vocabularies[1]),
                national_tokens_unseen_in_training=len(vocabularies[1])-len(national_seen),
                national_token_share_unseen_in_training=(len(vocabularies[1])-len(national_seen))/len(vocabularies[1]),
                national_token_share_unseen_in_all_cohort_records=(len(vocabularies[1])-len(all_seen))/len(vocabularies[1]),
                token_scope='Stored pre-cutoff cohort tokens; runtime-inserted masking/special tokens are not counted as seen',
                national_clinical_tokens=len(clinical),
                clinical_tokens_unseen_in_training=sum(i not in national_seen for i in clinical),
                clinical_token_share_unseen_in_training=sum(i not in national_seen for i in clinical)/len(clinical),
                clinical_token_share_unseen_in_all_cohort_records=sum(i not in all_seen for i in clinical)/len(clinical),
                source_provenance='National tokenized shards/vocabulary are the exact original DAPT inputs; same cohort IDs, physical split, cutoff and event-time sequences verified')


def prepare(root,local_bundle,national_root,expected_patients=78668):
    import torch
    from bonsai.functional.checkpointing import get_saved_encoder_config
    root=Path(root).resolve();national_root=Path(national_root).resolve()
    joint,jointpath,_=frozen_config(local_bundle,'seed42/L/J_all_diseases_all_outcomes')
    n_joint,npath,_=frozen_config(national_root/'adapted_fits','seed42/EA/J')
    checkpoint=Path(joint['dapt_ckpt'])
    run=checkpoint.parent.parent
    candidates=[run/'hydra/config.yaml',run/'.hydra/config.yaml']
    source=next((p for p in candidates if p.is_file()),None)
    if source is None:raise FileNotFoundError('Missing original cohort pretraining Hydra config')
    national_source=national_root/'external_dapt/hydra/config.yaml'
    from omegaconf import OmegaConf
    if not OmegaConf.has_resolver('version'):OmegaConf.register_new_resolver('version',lambda:0)
    local=OmegaConf.to_container(OmegaConf.load(source),resolve=True)
    national=OmegaConf.to_container(OmegaConf.load(national_source),resolve=True)
    data=torch.load(checkpoint,map_location='cpu',mmap=True,weights_only=False)
    architecture=get_saved_encoder_config(data['hyper_parameters']);del data;gc.collect()
    cfg=clone_pretraining(local,national,architecture)
    if sha(national['paths']['vocab'])!=sha(n_joint['paths']['vocabulary']):raise ValueError('National pretraining and fine-tuning vocabularies differ')
    paths=[source,national_source,jointpath,npath,checkpoint]+[Path(c['paths'][k]) for c in [local,national] for k in ['train_split','val_split','vocab','population']]
    identity={str(p.resolve()):sha(p) for p in paths}
    root.mkdir(parents=True,exist_ok=True)
    receipt=root/'preparation.json'
    if receipt.exists():
        report=json.loads(receipt.read_text())
        if report['inputs']!=identity:raise ValueError('Vocabulary-control inputs changed; use a fresh root')
        immutable(root/'pretraining.yaml',cfg)
        return report
    audit=audit_sources(local,national,expected_patients)
    report=dict(status='prepared_not_launched',scope='Approved exploratory from-scratch national-vocabulary control',
        inputs=identity,source_joint_config=str(jointpath.resolve()),national_joint_config=str(npath.resolve()),
        audit=audit,checkpoint_epochs=[10,20],optimizer='Original cohort optimizer and schedule; no transfer weights',
        pretraining_changes=['national vocabulary and its corresponding laboratory-binned tokenized records','twenty epochs; checkpoint ten'],
        finetuning_changes=['dapt_ckpt','paths.vocabulary','cohorts.*.subject_data_paths'],
        finetuning_settings='Original cohort J_L training/head/selection settings unchanged; native labels/populations retained')
    immutable(root/'pretraining.yaml',cfg)
    receipt.write_text(json.dumps(report,indent=2));return report


def joint_stage(root,epoch):
    from opera.run.prepare_epoch20_joint_finetune import checkpoint_record
    from opera.run.prepare_paper_run_bundle import write_bundle
    root=Path(root).resolve();report=json.loads((root/'preparation.json').read_text())
    localpath=Path(report['source_joint_config']);nationalpath=Path(report['national_joint_config'])
    for path in [localpath,nationalpath]:
        if sha(path)!=report['inputs'][str(path)]:raise ValueError('Source joint configuration changed')
    checkpoint=root/'pretraining/version_0'/('epoch_010.ckpt' if epoch==10 else 'last.ckpt')
    record=checkpoint_record(checkpoint)
    if record['completed_epochs']!=epoch:raise ValueError('Vocabulary-control checkpoint epoch mismatch')
    cfg=clone_joint(yaml.safe_load(localpath.read_text()),yaml.safe_load(nationalpath.read_text()),checkpoint)
    destination=root/f'joint{epoch}/bundle'
    if destination.exists():
        bundle=json.loads((destination/'bundle.json').read_text());row,=bundle['runs']
        if yaml.safe_load((destination/row['config_path']).read_text())!=cfg:raise ValueError('Existing joint bundle changed')
        return
    write_bundle([dict(name='seed42/L/J_all_diseases_all_outcomes',phase='supervision',origin='L',seed=42,condition={},config=cfg)],
        destination,provenance=dict(variant=f'J_NV{epoch}',checkpoint=record,source_audit=report['audit'],
            scientific_changes=report['finetuning_changes'],scope='Exploratory; no original aliases changed'))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-root',type=Path,default=Path('data/vocabulary_control_20261120'))
    p.add_argument('--local-bundle',type=Path,default=Path('data/paper_256_v3/bundles/stage1'))
    p.add_argument('--national-root',type=Path,default=Path('data/paper_256_v3_national_20261111'))
    p.add_argument('--stage',choices=['prepare','joint10','joint20'],default='prepare')
    a=p.parse_args()
    if a.stage=='prepare':print(json.dumps(prepare(a.output_root,a.local_bundle,a.national_root),indent=2))
    else:joint_stage(a.output_root,10 if a.stage=='joint10' else 20)

if __name__=='__main__':main()
