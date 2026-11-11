"""Reuse completed queues across the reviewed columnwise-grouping memory fix.

Stop coordinator and CPU workers before syncing the fix and running this tool.
Old generation stays intact; payload files are hard-linked, queue descriptors
are atomically replaced in the new generation. Never delete the old generation:
trusted pickle payloads may retain references to its internal draw files.
"""
import argparse
import copy
import hashlib
import json
import shutil
from pathlib import Path
import uuid
import yaml
from opera.evaluation import distributed_analysis as da

OLD_HASH = '39f75fe2b67a2e7b38d77d25699f30028dfe620b63282a497ac856297b058abf'


def previous_signature(config, identity):
    paths = {str(a['path']) for a in config.get('artifacts', [])}
    for key in ('patient_metadata','cell_metadata','contrast_manifest','coverage_roster','route_weights','compute_inventory'):
        if config.get(key):
            paths.add(str(config[key]))
    repo=Path(__file__).resolve().parents[2]
    code={relative:hashlib.sha256((repo/relative).read_bytes()).hexdigest()
          for relative in ('opera/evaluation/secondary_analysis.py','opera/evaluation/distributed_analysis.py',
                           'opera/run/analyse_locked_predictions.py','opera/evaluation/operating_points.py')}
    code['opera/evaluation/distributed_analysis.py']=OLD_HASH
    inputs={str(Path(p).resolve()):[Path(p).stat().st_size,Path(p).stat().st_mtime_ns] for p in sorted(paths)}
    payload=dict(config=config,inputs=inputs,implementation=code,environment=identity,version=1)
    return hashlib.sha256(json.dumps(payload,sort_keys=True,default=str).encode()).hexdigest()


def migrate(config, base, execute=False):
    base=Path(base)
    active=json.loads((base/'active.json').read_text())['generation']
    if Path(active).name != active:
        raise ValueError('Invalid generation')
    old=base/active
    identity=da.worker_identity()
    previous=copy.deepcopy(identity)
    previous['implementation']['distributed_analysis.py']=OLD_HASH
    if previous_signature(config,previous) != active:
        raise ValueError('Old generation does not match this exact recipe, inputs, environment and reviewed implementation; refusing reuse')
    if not (old/'prepared/prepared.json').is_file():
        raise ValueError('Prepared snapshot missing')
    queues=list(old.glob('paired/*/queue.json'))+list(old.glob('tasks/*/queue.json'))
    empty=[]
    for queue in queues:
        descriptor=json.loads(queue.read_text())
        if descriptor['implementation'] != previous:
            raise ValueError(f'Unexpected queue identity: {queue}')
        total=descriptor['total']
        done=len(list(queue.parent.glob('batch_*.result_INTERNAL.pkl')))
        ready=len(list(queue.parent.glob('batch_*.ready.json')))
        if list(queue.parent.glob('batch_*.failed.json')):
            raise ValueError(f'Failed queue requires inspection: {queue}')
        if done==0 and ready==0:
            empty.append(queue.parent.relative_to(old))
        elif done != total or not (queue.parent/'collected.json').is_file():
            raise ValueError(f'Only fully collected or completely unpublished queues are eligible: {queue}')
    target=base/da.signature(config)
    if target.exists():
        raise FileExistsError(target)
    print(f'Validated {len(queues)-len(empty)} completed queues; dropping {len(empty)} unpublished queues')
    print('Source:',old,'\nDestination:',target)
    if not execute:
        print('Preview only; pass --execute after stopping CPU workers')
        return target
    temporary=base/('.migration_'+uuid.uuid4().hex)
    try:
        shutil.copytree(old,temporary,copy_function=__import__('os').link)
        for relative in empty:
            shutil.rmtree(temporary/relative)
        for queue in [*temporary.glob('paired/*/queue.json'),*temporary.glob('tasks/*/queue.json')]:
            descriptor=json.loads(queue.read_text())
            descriptor['implementation']=identity
            da.save_json(queue,descriptor)
        da.save_json(temporary/'memory_fix_migration.json',dict(source_generation=active,
            new_generation=target.name,change='Columnwise group extraction; completed numerical outputs reused unchanged',
            old_distributed_hash=OLD_HASH,new_identity=identity,retain_source_generation=True))
        temporary.rename(target)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    da.save_json(base/'active.json',{'generation':target.name})
    print('Migration complete; retain original generation and restart CPU workers/coordinator.')
    return target


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--checkpoint-dir',type=Path,required=True)
    parser.add_argument('--execute',action='store_true')
    args=parser.parse_args()
    migrate(yaml.safe_load(args.config.read_text()),args.checkpoint_dir,args.execute)


if __name__=='__main__':
    main()
