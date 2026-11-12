"""Prepare exact completed OPERA checkpoint replay jobs; never launches GPUs."""
import argparse
import json
from pathlib import Path
import re

from bonsai.functional.checkpointing import verified_training_selection
from opera.run.paper_prediction_queue import bundle_runs, canonical_models


def prepare(national_root,output,local_bundle=None,local_runs=None):
    root=Path(national_root).resolve();output=Path(output).resolve()
    if output.exists():
        raise FileExistsError(output)
    direct=json.loads((root/'direct_fits/bundle.json').read_text())
    if local_bundle is None:
        local_bundle=direct['provenance']['frozen_local_bundle']
    local_bundle=Path(local_bundle).resolve()
    local_runs=Path(local_runs or 'data/paper_256_v3/neural_runs').resolve()
    jobs,blocked=[],[]
    for origin,bundle_dir,runs in [('L',local_bundle,local_runs),
                                  ('E',root/'direct_fits',root/'neural_runs'),
                                  ('EA',root/'adapted_fits',root/'neural_runs')]:
        bundle,rows=bundle_runs(bundle_dir)
        for row,path,config in rows:
            models=canonical_models(row,config,bundle)
            expected=(any(re.fullmatch(r'J_OA_OWN(?:_s\d+)?_L',m) for m in models)
                      if origin=='L' else f'R_{origin}' in models)
            if not expected:
                continue
            run=runs/row['run_directory']
            try:
                selection=verified_training_selection(run)
            except (ValueError,FileNotFoundError) as error:
                blocked.append(dict(origin=origin,run_id=row['name'],reason=str(error)))
                continue
            jobs.append(dict(encoder=origin,seed=int(config['seed']),run_id=row['name'],
                checkpoint=str(run/'best.ckpt'),checkpoint_sha256=selection['best_checkpoint_sha256'],
                scientific_hash=row['scientific_hash'],
                evaluator_args=['--config-dir',str(path.parent),'--config-name',path.stem,
                                '--batch-size','1','--num-workers','2','--device','cuda'],
                modes=['full','all','own_disease','own_outcome','global']))
    counts={origin:sum(j['encoder']==origin for j in jobs) for origin in ['L','E','EA']}
    if any(n==0 for n in counts.values()):
        raise ValueError(f'Missing completed OPERA checkpoint(s): {counts}; blockers: {blocked}')
    identities=[(j['encoder'],j['seed']) for j in jobs]
    if len(identities)!=len(set(identities)):
        raise ValueError('Duplicate encoder/seed OPERA candidates; review rather than choose automatically')
    output.parent.mkdir(parents=True,exist_ok=True)
    plan=dict(jobs=jobs,blocked=blocked,training_launched=False,
              inference_passes=sum(len(j['modes']) for j in jobs),
              priority='Run only on a GPU free from held-out evaluation; no scarce-label training approved')
    output.write_text(json.dumps(plan,indent=2))
    print(json.dumps(dict(plan=str(output),jobs=len(jobs),counts=counts,
                         inference_passes=plan['inference_passes'],blocked=blocked),indent=2))
    return plan


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--national-root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--local-bundle',type=Path)
    p.add_argument('--local-runs',type=Path)
    args=p.parse_args();prepare(args.national_root,args.output,args.local_bundle,args.local_runs)


if __name__=='__main__':
    main()
