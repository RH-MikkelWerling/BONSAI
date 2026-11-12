"""Exploratory OPERA correction replay using the existing locked evaluator.

A JSON plan supplies jobs with encoder, seed, checkpoint, run_id and evaluator
arguments. Full and inference-zeroed passes use unchanged checkpoints. This
module never trains models or modifies the shared prediction queue.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import pandas as pd
import torch

from opera.modules.networks.attention_own_corrections import AttentionOwnCorrections


def correction_ratios(representation, terms, scale):
    denominator=torch.linalg.vector_norm(representation.float(),dim=-1)
    ratios={}
    for name, term in {**terms,'sum':sum(terms.values())}.items():
        numerator=torch.linalg.vector_norm(scale*term.float(),dim=-1)
        ratios[name]=torch.where(denominator>0,numerator/denominator,torch.nan)
    return ratios


def corrected_representation(representation,terms,scale,zero):
    retained=[term for name,term in terms.items() if zero!='all' and name!=zero]
    return representation if not retained else representation+scale*sum(retained).to(representation.dtype)


def digest_file(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda:handle.read(1024*1024),b''):
            digest.update(chunk)
    return digest.hexdigest()


def verified_replay(output,job,mode,split,checkpoint_hash):
    output=Path(output)
    marker=output/'diagnostic_provenance.json'
    if not output.exists():
        return False
    if not marker.exists():
        raise FileExistsError(f'Incomplete replay retained for inspection: {output}')
    saved=json.loads(marker.read_text())
    if (saved['job']!=job or saved['mode']!=mode or saved['split']!=split
            or saved['checkpoint_sha256']!=checkpoint_hash):
        raise ValueError('Replay inputs changed; use a fresh diagnostic output root')
    for name,digest in saved['outputs'].items():
        if not (output/name).is_file() or digest_file(output/name)!=digest:
            raise ValueError(f'Replay output changed: {output/name}')
    print(f'Reused verified diagnostic replay: {output}',flush=True)
    return True


def replay(job,mode,output,split):
    from opera.run import evaluate_direct_competing_risk as evaluator
    import pyarrow as pa
    import pyarrow.parquet as pq
    started=time.monotonic()
    checkpoint_hash=digest_file(job['checkpoint'])
    if job.get('checkpoint_sha256',checkpoint_hash)!=checkpoint_hash:
        raise ValueError('Checkpoint differs from verified replay plan')
    if verified_replay(output,job,mode,split,checkpoint_hash):
        return
    forbidden={'--split','--output-dir','--checkpoint'}
    if any(a.split('=')[0] in forbidden for a in job['evaluator_args']):
        raise ValueError('Plan evaluator_args must not override checkpoint, split or output')
    original_load=evaluator._load_model
    original_forward=AttentionOwnCorrections.forward
    current={}
    writers=[]
    def load(*args,**kwargs):
        model=original_load(*args,**kwargs)
        if getattr(model,'attention_own_corrections',None) is None:
            raise ValueError('Replay requires the own-corrections OPERA architecture')
        get=model.get_objective_representations
        def tracked(batch):
            current['subjects']=batch['subject_id'].detach().cpu().numpy()
            return get(batch)
        model.get_objective_representations=tracked
        return model
    def forward(module,representation,fine_index):
        terms=module.components(representation,fine_index)
        if mode!='full' and mode!='all' and mode not in terms:
            raise ValueError(f'Requested absent correction term: {mode}')
        if mode=='full':
            output.mkdir(parents=True,exist_ok=True)
            ratios=correction_ratios(representation,terms,module.residual_scale)
            records=[]
            for term,ratio in ratios.items():
                values=ratio.detach().cpu().numpy()
                for patient,subject in enumerate(current['subjects']):
                    disease=int(fine_index[patient].item())
                    for index,outcome in enumerate(module.outcome_names):
                        records.append(dict(subject_id=str(subject),cohort=module.fine_names[disease]
                            if 0<=disease<len(module.fine_names) else 'unknown', outcome=outcome,
                            encoder=job['encoder'],seed=job['seed'],term=term,relative_change=float(values[patient,index])))
            table=pa.Table.from_pandas(pd.DataFrame(records),preserve_index=False)
            if not writers:
                writers.append(pq.ParquetWriter(output/'correction_norms_INTERNAL.parquet',table.schema))
            writers[0].write_table(table)
        return corrected_representation(representation,terms,module.residual_scale,
                                        '' if mode=='full' else mode)
    evaluator._load_model=load
    AttentionOwnCorrections.forward=forward
    old_argv=sys.argv
    try:
        sys.argv=['correction_replay','--checkpoint',job['checkpoint'],'--split',split,
                  '--output-dir',str(output),*job['evaluator_args']]
        evaluator.main()
    finally:
        for writer in writers:
            writer.close()
        sys.argv=old_argv
        evaluator._load_model=original_load
        AttentionOwnCorrections.forward=original_forward
    if mode=='full':
        norms=pd.read_parquet(output/'correction_norms_INTERNAL.parquet')
        norms.groupby(['encoder','seed','cohort','outcome','term']).relative_change.agg(
            ['mean','median','count']).reset_index().to_csv(output/'correction_norm_summary.csv',index=False)
    files=['direct_cr_predictions.parquet']
    if mode=='full':
        files += ['correction_norms_INTERNAL.parquet','correction_norm_summary.csv']
    from opera.evaluation.distributed_analysis import save_json
    save_json(output/'diagnostic_provenance.json',dict(exploratory=True,job=job,
        mode=mode,split=split,checkpoint_sha256=checkpoint_hash,elapsed_seconds=time.monotonic()-started,
        outputs={name:digest_file(output/name) for name in files},
        interpretation='Inference reliance, not a trained ablation'))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--split',choices=['tuning','held_out'],default='tuning')
    parser.add_argument('--held-out-ready',action='store_true')
    parser.add_argument('--job-index',type=int)
    parser.add_argument('--mode')
    parser.add_argument('--modes',nargs='+',help='Subset of planned modes; e.g. full all for a fast first pass')
    args=parser.parse_args()
    if args.split=='held_out' and not args.held_out_ready:
        parser.error('Wait for main held-out evaluation, then pass --held-out-ready')
    plan=json.loads(args.plan.read_text())
    if args.job_index is not None:
        replay(plan['jobs'][args.job_index],args.mode,args.output_dir,args.split)
        return
    for index,job in enumerate(plan['jobs']):
        planned=job.get('modes',['full','all','own_disease','own_outcome','global'])
        modes=args.modes or planned
        if not set(modes)<=set(planned):
            parser.error('Requested modes are absent from replay plan')
        for mode in modes:
            command=[sys.executable,'-u','-m',__name__ if __name__!='__main__' else
                     'opera.diagnostics.correction_replay','--plan',str(args.plan),
                     '--job-index',str(index),'--mode',mode,'--split',args.split,
                     '--output-dir',str(args.output_dir/f"{job['encoder']}_seed{job['seed']}"/mode)]
            if args.held_out_ready:
                command.append('--held-out-ready')
            subprocess.run(command,check=True)


if __name__=='__main__':
    main()
