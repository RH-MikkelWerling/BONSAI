"""Read-only Linux allocation headroom check before a large coordinator launch."""
import argparse
import json
from pathlib import Path
import subprocess


def available_bytes():
    mem={}
    for line in Path('/proc/meminfo').read_text().splitlines():
        key,value=line.split(':',1)
        mem[key]=int(value.strip().split()[0])*1024
    available=mem['MemAvailable']
    # cgroup-v2 allocation limits can be smaller than host RAM.
    relative=''
    for line in Path('/proc/self/cgroup').read_text().splitlines():
        if line.startswith('0::'):
            relative=line.split('::',1)[1].lstrip('/')
    root=Path('/sys/fs/cgroup')/relative
    for directory in [root,*root.parents,Path('/sys/fs/cgroup')]:
        if not str(directory).startswith('/sys/fs/cgroup'):
            continue
        limit=directory/'memory.max';current=directory/'memory.current'
        if limit.is_file() and current.is_file():
            text=limit.read_text().strip()
            if text!='max':
                available=min(available,max(0,int(text)-int(current.read_text())))
    return available


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--minimum-free-gib',type=float,default=96)
    p.add_argument('--require-no-coordinator',action='store_true')
    p.add_argument('--require-free-gpu',action='store_true')
    args=p.parse_args()
    if not Path('/proc/meminfo').exists():
        raise SystemExit('Run resource preflight on the Linux cluster allocation')
    available=available_bytes()/1024**3
    print(json.dumps(dict(available_gib=round(available,1),minimum_free_gib=args.minimum_free_gib,
                         note='Launch headroom check; not a measured peak-RAM guarantee'),indent=2))
    if available<args.minimum_free_gib:
        raise SystemExit('Insufficient current allocation headroom; do not launch this coordinator here')
    if args.require_free_gpu:
        active=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).strip()
        if active:
            raise SystemExit('GPU compute processes are active; do not displace evaluation or training')
    if args.require_no_coordinator:
        output=subprocess.check_output(['ps','-u',str(__import__('os').getuid()),'-o','pid=,args='],text=True)
        for line in output.splitlines():
            if any(name in line for name in ['-m opera.run.analyse_locked_predictions','-m opera.run.test_cpu_pipeline']):
                raise SystemExit('A coordinator is already active on this allocation; use a different machine')


if __name__=='__main__':
    main()
