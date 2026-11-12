"""Publish local allocation headroom to shared storage, or read all reports."""
import argparse
import datetime as dt
import json
import os
from pathlib import Path
import socket
import subprocess
import time

from check_results_resources import available_bytes


def command(args):
    try:
        result = subprocess.run(args, capture_output=True, text=True, timeout=10)
        return result.stdout.strip() if result.returncode == 0 else None
    except (OSError, subprocess.TimeoutExpired):
        return None


def snapshot():
    processes = command(['ps', '-u', str(os.getuid()), '-o', 'pid=,rss=,args=']) or ''
    jobs = [line.strip() for line in processes.splitlines()
            if any(token in line for token in (
                '-m opera.run.', '-m opera.analysis.', '-m opera.diagnostics.',
                'run_representation_diagnostics.sh'))]
    coordinator = any(token in line for line in jobs for token in (
        '-m opera.run.analyse_locked_predictions', '-m opera.run.test_cpu_pipeline'))
    worker = any(token in line for line in jobs for token in (
        '-m opera.run.analysis_cpu_worker', '-m opera.run.results_worker'))
    gpu_processes = command(['nvidia-smi', '--query-compute-apps=pid,used_memory',
                             '--format=csv,noheader,nounits'])
    free = available_bytes() / 1024 ** 3
    if coordinator:
        recommendation = 'Keep coordinator here; reserve headroom and leave running'
    elif jobs:
        recommendation = ('Existing worker active; keep it serving queues' if worker else
                          'Other OPERA work active; inspect before adding work')
    elif gpu_processes == '' and free >= 48:
        recommendation = 'Candidate for dedicated replay + CPU diagnostics (48 GiB check)'
    elif free >= 24:
        recommendation = 'Candidate for one CPU worker; batch peak RAM remains unmeasured'
    else:
        recommendation = 'Low headroom; do not add heavy work'
    return dict(host=socket.gethostname(), timestamp=time.time(),
                allocation_id=os.environ.get('SLURM_JOB_ID') or os.environ.get('PBS_JOBID'),
                available_gib=round(free, 1), cpu_affinity=len(os.sched_getaffinity(0)),
                load=os.getloadavg(), jobs=jobs,
                gpu_inventory=command(['nvidia-smi', '--query-gpu=index,name,memory.free,utilization.gpu',
                                       '--format=csv,noheader,nounits']),
                gpu_processes=gpu_processes, recommendation=recommendation)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, default=Path('data/allocation_reports'))
    parser.add_argument('--watch', action='store_true', help='Publish this allocation every 60 seconds')
    parser.add_argument('--show', action='store_true', help='Read shared reports; no remote process access')
    args = parser.parse_args()
    if args.show:
        for path in sorted(args.directory.glob('*.json')):
            report = json.loads(path.read_text())
            age = time.time() - report['timestamp']
            print(f"\n{report['host']} allocation={report['allocation_id']} "
                  f"age={age:.0f}s {'STALE' if age > 180 else 'fresh'} "
                  f"available={report['available_gib']} GiB CPUs={report['cpu_affinity']}")
            print(report['recommendation'] if age <= 180 else 'Refresh report before placing work')
            print('GPU:', report['gpu_inventory'] or 'unavailable')
            print('GPU processes:', repr(report['gpu_processes']))
            print('\n'.join(report['jobs']) or 'No matching OPERA jobs')
        return
    if not Path('/proc/meminfo').exists():
        parser.error('Publish reports on Linux allocations; --show also works locally')
    args.directory.mkdir(parents=True, exist_ok=True)
    target = args.directory / f'{socket.gethostname()}_{os.environ.get("SLURM_JOB_ID", os.environ.get("PBS_JOBID", "host"))}.json'
    while True:
        report = snapshot()
        temporary = target.with_suffix(f'.{os.getpid()}.tmp')
        temporary.write_text(json.dumps(report, indent=2))
        temporary.replace(target)
        print(dt.datetime.now(dt.timezone.utc).isoformat(), target, flush=True)
        if not args.watch:
            break
        time.sleep(60)


if __name__ == '__main__':
    main()
