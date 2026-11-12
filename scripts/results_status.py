"""Lightweight status of both test projects; no prediction reads or rehashing."""
import json
from pathlib import Path


def snapshot(root,checkpoint_name,results_name):
    root=Path(root)
    print('\nPROJECT',root,flush=True)
    final=root/results_name/'analysis_manifest.json'
    progress=root/(results_name+'.progress')/'effect_summaries.csv'
    print('Final analysis complete:',final.is_file())
    print('Main effects readable:',progress if progress.exists() else 'not yet')
    base=root/checkpoint_name
    active=base/'active.json'
    if not active.exists():
        print('No checkpoint generation published');return
    name=json.loads(active.read_text())['generation']
    if Path(name).name!=name:
        raise ValueError('Invalid checkpoint generation')
    generation=base/name
    for queue in sorted([*generation.glob('paired/*/queue.json'),*generation.glob('tasks/*/queue.json')]):
        total=json.loads(queue.read_text())['total']
        done=len(list(queue.parent.glob('batch_*.result_INTERNAL.pkl')))
        failed=len(list(queue.parent.glob('batch_*.failed.json')))
        print(f'{queue.parent.relative_to(generation)}: {done}/{total}; failed={failed}; collected={(queue.parent/"collected.json").exists()}')


if __name__=='__main__':
    snapshot('data/paper_256_v3_national_20261111/extended_test_comparison','checkpoints','results')
    snapshot('data/paper_256_v3/analysis/local_test_20261109_four_allocations','_cpu_checkpoints_INTERNAL','prediction_analysis')
