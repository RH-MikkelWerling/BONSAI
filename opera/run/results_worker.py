"""Serve several analysis queues, one isolated batch at a time per allocation.

Order checkpoint directories by priority. Batch subprocesses release all memory
on exit. Main distributed worker implementations and result formats are unchanged.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def pending_queues(base):
    base=Path(base)
    if any((base.parent/name).exists() for name in
           ['pipeline_actions_complete','results/analysis_manifest.json','prediction_analysis/analysis_manifest.json']):
        return []
    active=base/'active.json'
    if not active.exists():
        return []
    generation=json.loads(active.read_text())['generation']
    if Path(generation).name!=generation:
        raise ValueError('Invalid checkpoint generation')
    root=base/generation
    pending=[]
    for queue in sorted([*root.glob('paired/*/queue.json'),*root.glob('tasks/*/queue.json')]):
        if (queue.parent/'collected.json').exists():
            continue
        if list(queue.parent.glob('batch_*.failed.json')):
            raise RuntimeError(f'Failed queue requires inspection: {queue.parent}')
        for ready in queue.parent.glob('batch_*.ready.json'):
            result=ready.with_name(ready.name.replace('.ready.json','.result_INTERNAL.pkl'))
            if not result.exists():
                pending.append(queue.parent)
                break
    return pending


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint-dir',type=Path,action='append',default=[])
    p.add_argument('--wait-hours',type=float,default=48)
    p.add_argument('--one-queue',type=Path,help=argparse.SUPPRESS)
    args=p.parse_args()
    for name in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:
        os.environ[name]='1'
    os.environ['CUDA_VISIBLE_DEVICES']=''
    if args.one_queue:
        from opera.evaluation.distributed_analysis import perform_one
        sys.exit(0 if perform_one(args.one_queue) else 3)
    if not args.checkpoint_dir or args.wait_hours<=0:
        p.error('Supply checkpoint directories and a positive wait duration')
    # One multi-queue worker per host/repository. Never holds this lock in children.
    import fcntl
    import socket
    lockdir=Path('logs');lockdir.mkdir(exist_ok=True)
    with (lockdir/f'results_worker_{socket.gethostname()}.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        deadline=time.monotonic()+args.wait_hours*3600
        last_notice=0
        while time.monotonic()<deadline:
            did_work=False
            for base in args.checkpoint_dir:
                for directory in pending_queues(base):
                    result=subprocess.run([sys.executable,'-u','-m','opera.run.results_worker',
                                           '--one-queue',str(directory)],close_fds=True)
                    if result.returncode not in [0,3]:
                        raise RuntimeError(f'Batch process exited {result.returncode}: {directory}; completed results retained')
                    if result.returncode==0:
                        print(f'Completed one batch; memory released: {directory}',flush=True)
                        did_work=True
                        break
                if did_work:
                    break
            if not did_work:
                if time.monotonic()-last_notice>=60:
                    print('Waiting for claimable work across configured queues',flush=True)
                    last_notice=time.monotonic()
                time.sleep(10)


if __name__=='__main__':
    main()
