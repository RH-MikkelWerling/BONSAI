"""Run durable paired-analysis jobs on any CPU node with shared secure storage."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import time

# Set before numpy/pandas imports; one worker should use one CPU core.
for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[name] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = ''

from opera.evaluation.distributed_analysis import perform_one


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint-dir', type=Path, required=True)
    parser.add_argument('--wait-hours', type=float, default=48)
    parser.add_argument('--poll-seconds', type=float, default=10)
    parser.add_argument('--status', action='store_true', help='Print saved queue progress and exit')
    args = parser.parse_args()
    if min(args.wait_hours, args.poll_seconds) <= 0:
        parser.error('Wait time and polling interval must be positive')
    if args.status:
        active = args.checkpoint_dir / 'active.json'
        if not active.exists():
            print('Coordinator has not published a checkpoint generation yet')
            return
        generation = json.loads(active.read_text())['generation']
        if Path(generation).name != generation:
            raise ValueError('Invalid checkpoint generation')
        root = args.checkpoint_dir / generation
        print(f'Prepared inputs saved: {(root / "prepared/prepared.json").exists()}')
        for queue in sorted([*root.glob('paired/*/queue.json'), *root.glob('tasks/*/queue.json')]):
            total = json.loads(queue.read_text())['total']
            ready = len(list(queue.parent.glob('batch_*.ready.json')))
            done = len(list(queue.parent.glob('batch_*.result_INTERNAL.pkl')))
            failed = len(list(queue.parent.glob('batch_*.failed.json')))
            print(f'{queue.parent.name}: {done}/{total} complete; {ready} published; {failed} failed')
        return
    deadline = time.monotonic() + args.wait_hours * 3600
    completed, last_notice = 0, 0
    print(f'CPU worker PID {os.getpid()}; waiting for {args.checkpoint_dir}', flush=True)
    while time.monotonic() < deadline:
        if (args.checkpoint_dir.parent / 'pipeline_actions_complete').exists():
            print(f'Pipeline complete; worker completed {completed} batches', flush=True)
            return
        active = args.checkpoint_dir / 'active.json'
        queues = []
        if active.exists():
            generation = json.loads(active.read_text())['generation']
            root = args.checkpoint_dir / generation
            if Path(generation).name != generation:
                raise ValueError('Invalid checkpoint generation')
            queues = sorted([*root.glob('paired/*/queue.json'), *root.glob('tasks/*/queue.json')])
            pending_queues = [q for q in queues if not (q.parent / 'collected.json').exists()]
            for queue in pending_queues:
                failures = list(queue.parent.glob('batch_*.failed.json'))
                if failures:
                    raise RuntimeError(f'Batch failed; inspect {failures[0]} before retrying')
            did_work = False
            for queue in pending_queues:
                if perform_one(queue.parent):
                    completed += 1
                    print(f'CPU worker completed {completed} batches; last queue {queue.parent.name}', flush=True)
                    did_work = True
                    break
            if did_work:
                continue
        if time.monotonic() - last_notice >= 60:
            done = sum(len(list(q.parent.glob('batch_*.result_INTERNAL.pkl'))) for q in queues)
            print(f'Waiting; {done} batch results saved across {len(queues)} queues', flush=True)
            last_notice = time.monotonic()
        time.sleep(args.poll_seconds)
    raise TimeoutError('Worker wait limit reached; completed results are saved and reusable')


if __name__ == '__main__':
    main()
