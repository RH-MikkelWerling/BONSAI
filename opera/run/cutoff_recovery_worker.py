"""Join the shared recovery queue: train then evaluate, one isolated fit per child."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
from types import SimpleNamespace

import yaml

from opera.run.finish_figure_inputs import sha


def read(path):
    return yaml.load(Path(path).read_text(), Loader=getattr(yaml, 'CSafeLoader', yaml.SafeLoader))


def fingerprint(path):
    path = Path(path)
    stat = path.stat()
    return dict(path=str(path), bytes=stat.st_size, mtime_ns=stat.st_mtime_ns)


def completed(root, digest, job):
    marker = root/'completed'/f'{digest}.json'
    if not marker.is_file():
        return False
    saved = json.loads(marker.read_text())
    if saved['config_sha256'] != job['row']['config_sha256']:
        raise ValueError('Completed recovery job belongs to a different config')
    for record in saved['files']:
        if fingerprint(record['path']) != record:
            raise ValueError(f'Completed recovery output changed: {record["path"]}')
    return True


def encoder_ready(config):
    path = Path(config['dapt_ckpt'])
    if not path.is_file() or not (path.parent/'cutoff_audit.json').is_file():
        return False
    predecessor = config.get('source_opera_ckpt')
    if predecessor and (not Path(predecessor).is_file() or not (Path(predecessor).parent/'training_complete.json').is_file()):
        return False
    if path.name == 'last.ckpt':
        sidecar = path.parent/'checkpoint_metadata.json'
        if not sidecar.is_file():
            return False
        saved = read(sidecar)
        epochs = saved.get('completed_epochs', saved.get('extra', {}).get('completed_epochs', 0))
        return int(epochs) >= config['cutoff_recovery']['expected_epochs']
    return True  # Recovery milestones are atomically published after saving.


def one_job(root, digest):
    from bonsai.functional.checkpointing import verified_training_selection
    from opera.run.run_paper_queue import audit_encoder, load_run_config, train_run, NotEstimable
    from opera.run.paper_prediction_queue import evaluate_row
    plan = read(root/'queue_plan.yaml')
    if sha(Path(plan['bundle'])/'bundle.json') != plan['bundle_sha256']:
        raise ValueError('Recovery bundle changed')
    job = read(root/'jobs'/f'{digest}.yaml')
    row = job['row']
    path, raw, resolved = load_run_config(row, Path(plan['bundle']))
    run = root/'neural_runs'/row['run_directory']
    run.mkdir(parents=True, exist_ok=True)
    records, evaluations = [path], []
    if (run/'not_estimable.json').is_file():
        status = 'not_estimable'
        records.append(run/'not_estimable.json')
    else:
        if not (run/'training_complete.json').is_file():
            audit = audit_encoder(resolved, row['origin'])
            try:
                train_run(row, path, run, audit)
            except NotEstimable:
                return one_job(root, digest)
        selection = verified_training_selection(run)
        for split in job['splits']:
            args = SimpleNamespace(runs_dir=root/'neural_runs', output_dir=root/'predictions',
                split=split, execute=True, export_routes=False,
                horizons='30,60,90,180,365,730', batch_size=1, num_workers=0, device='auto')
            result = evaluate_row(row, path, raw, args, aliases=job['models'])
            if result['status'] not in {'evaluated', 'reused_verified_evaluation'}:
                raise ValueError(f'Recovery evaluation not complete: {result}')
            evaluations.append(result)
            directory = Path(result['prediction_directory'])
            records.extend([directory/'evaluation_complete.json', directory/'direct_cr_predictions.parquet'])
            if raw['cutoff_recovery'].get('export_embeddings'):
                records.extend([directory/'patient_embeddings.npz', directory/'patient_embeddings.metadata.json'])
        for split in job.get('removal_splits', []):
            args.split = split
            result = evaluate_row(row, path, raw, args, aliases=job['models'], borrowing_design=True)
            if result['status'] not in {'evaluated', 'reused_verified_evaluation'}:
                raise ValueError(f'Source-removal export incomplete: {result}')
            directory = Path(result['prediction_directory'])
            records.extend([directory/'evaluation_complete.json', directory/'direct_cr_predictions.parquet'])
            records.extend(sorted((directory/'source_removal').rglob('*.parquet')))
            records.append(directory/'source_removal/design.json')
        records.extend([run/'training_complete.json', run/selection['best_checkpoint']])
        status = 'evaluated'
    destination = root/'completed'
    destination.mkdir(exist_ok=True)
    report = dict(status=status, config_sha256=row['config_sha256'], models=job['models'],
                  files=[fingerprint(p) for p in records], evaluations=evaluations)
    temporary = destination/f'.{digest}.{os.getpid()}.tmp'
    temporary.write_text(json.dumps(report, indent=2))
    temporary.replace(destination/f'{digest}.json')
    return report


def interpretation_ready(root, plan):
    required = [(digest, job) for digest, job in plan['jobs'].items()
                if 'R_EA' in job['models'] and job['row']['seed'] == 42]
    return len(required) == 1 and completed(root, *required[0])


def try_interpretation(root, plan, worker):
    from opera.run.paper_worker import Claim
    output = root/'final_interpretation'
    manifest = output/'interpretation_manifest.json'
    if manifest.is_file() and read(manifest).get('status') == 'complete':
        return 'complete'
    if not interpretation_ready(root, plan): return 'not_ready'
    for label, item in plan.get('export_inputs', {}).items():
        if sha(Path(item['path'])) != item['sha256']:
            raise ValueError(f'Frozen interpretation/export input changed: {label}')
    task = root/'interpretation_task'
    task.mkdir(exist_ok=True)
    failure = task/'failure.json'
    if failure.is_file() and read(failure).get('attempts', 0) >= 2:
        return 'failed_needs_inspection'
    claim = Claim(task, worker, 60)
    if not claim.acquire(): return 'claimed_elsewhere'
    try:
        with (task/'job.log').open('a') as log:
            subprocess.run([sys.executable, '-u', '-m', 'opera.run.cutoff_recovery_model_exports',
                '--root', str(root)], check=True, stdout=log, stderr=subprocess.STDOUT)
            subprocess.run([sys.executable, '-u', '-m', 'opera.run.run_final_interpretation',
                '--national-root', str(root), '--output-dir', str(output), '--execute'],
                check=True, stdout=log, stderr=subprocess.STDOUT)
        if not manifest.is_file() or read(manifest).get('status') != 'complete':
            raise RuntimeError('Interpretation did not publish a complete manifest')
        failure.unlink(missing_ok=True)
        return 'complete'
    except Exception as error:
        previous = read(failure).get('attempts', 0) if failure.is_file() else 0
        failure.write_text(json.dumps(dict(attempts=previous+1, error=str(error), log=str(task/'job.log')), indent=2))
        raise
    finally:
        claim.release()


def work(root, *, wait_hours=72, poll_seconds=30, max_jobs=None, with_interpretation=False):
    from opera.run.paper_worker import Claim, record_attempt, record_failure, now
    plan = read(root/'queue_plan.yaml')
    if sha(Path(plan['bundle'])/'bundle.json') != plan['bundle_sha256']:
        raise ValueError('Recovery bundle changed')
    jobs = sorted(plan['jobs'].items(), key=lambda pair: (pair[1]['priority'], pair[1]['row']['seed'], pair[0]))
    configs = {digest: read(Path(plan['bundle'])/job['row']['config_path']) for digest, job in jobs}
    worker = f'{socket.gethostname()}-{os.getpid()}'
    deadline = time.monotonic() + wait_hours*3600
    finished = failures = 0
    while time.monotonic() < deadline:
        counts, claimed = Counter(), False
        for digest, job in jobs:
            if completed(root, digest, job):
                counts['completed'] += 1
                continue
            run = root/'neural_runs'/job['row']['run_directory']
            failed = run/'worker_failed.json'
            if failed.exists() and read(failed).get('attempts', 0) >= 2:
                counts['failed_needs_inspection'] += 1
                continue
            if not encoder_ready(configs[digest]):
                counts['waiting_for_encoder'] += 1
                continue
            run.mkdir(parents=True, exist_ok=True)
            claim = Claim(run, worker, 60)
            if not claim.acquire():
                counts['claimed_or_stale'] += 1
                continue
            claimed = True
            started, status = now(), 'failed'
            try:
                # Another worker may have completed it between our scan and claim.
                if not completed(root, digest, job):
                    print(f'CLAIMED priority={job["priority"]} models={job["models"]} fit={digest}', flush=True)
                    with (run/'recovery_job.log').open('a') as log:
                        subprocess.run([sys.executable, '-u', '-m', 'opera.run.cutoff_recovery_worker',
                                        'one-job', '--root', str(root), '--job', digest],
                                       stdout=log, stderr=subprocess.STDOUT, check=True)
                    if not completed(root, digest, job):
                        raise RuntimeError('Job exited without verified completion receipt')
                status = 'evaluated'
                finished += 1
                failures = 0
                failed.unlink(missing_ok=True)
                print(f'COMPLETE {digest}; training/evaluation memory released', flush=True)
            except Exception as error:
                record_failure(run, worker, error)
                failures += 1
                print(f'FAILED {digest}: {error!r}; inspect {run / "recovery_job.log"}', flush=True)
                if failures >= 3:
                    raise RuntimeError('Three consecutive failures; inspect allocation/configuration') from error
            finally:
                record_attempt(run, worker, started, status)
                claim.release()
            break
        if max_jobs and finished >= max_jobs:
            return
        if not claimed:
            print(json.dumps(dict(worker=worker, counts=counts)), flush=True)
            if with_interpretation:
                interpretation = try_interpretation(root, plan, worker)
                print('Interpretation:', interpretation, flush=True)
            if counts['completed'] == len(jobs):
                if with_interpretation and interpretation == 'failed_needs_inspection':
                    raise RuntimeError(f'Fits complete; interpretation needs inspection: {root/"interpretation_task/failure.json"}')
                if with_interpretation and interpretation == 'claimed_elsewhere':
                    print('GPU fits complete; another worker owns interpretation. This worker can exit.', flush=True)
                    return
                if with_interpretation and interpretation != 'complete':
                    time.sleep(poll_seconds)
                    continue
                print('RECOVERY GPU QUEUE COMPLETE', flush=True)
                return
            if counts['failed_needs_inspection'] and not (counts['waiting_for_encoder'] or counts['claimed_or_stale']):
                raise RuntimeError('Queue has failed jobs requiring inspection')
            time.sleep(poll_seconds)
    raise TimeoutError('Worker walltime budget reached; completed jobs and partial checkpoints retained')


def status(root):
    plan = read(root/'queue_plan.yaml')
    counts = Counter()
    for digest, job in plan['jobs'].items():
        run = root/'neural_runs'/job['row']['run_directory']
        if completed(root, digest, job): state = 'complete'
        elif (run/'.worker_claim.json').exists(): state = 'claimed_check_heartbeat'
        elif (run/'worker_failed.json').exists(): state = 'failed_or_retryable'
        else: state = 'pending_or_waiting_for_encoder'
        counts[state] += 1
    print(json.dumps(dict(counts=counts, summary=read(root/'queue_summary.json')), indent=2))


def clear_stale(root, *, execute=False, confirmed_stopped=False, stale_minutes=30):
    if execute and not confirmed_stopped:
        raise ValueError('Confirm the owning allocations and all their processes stopped before clearing claims')
    plan = read(root/'queue_plan.yaml')
    paths = [root/'neural_runs'/job['row']['run_directory']/'.worker_claim.json' for job in plan['jobs'].values()]
    paths.append(root/'interpretation_task/.worker_claim.json')
    stale = []
    for path in paths:
        if not path.is_file(): continue
        claim = read(path)
        age = (datetime.now(timezone.utc)-datetime.fromisoformat(claim['heartbeat_at'])).total_seconds()/60
        if age <= stale_minutes: continue
        stale.append(dict(path=str(path), age_minutes=age, host=claim.get('host'), worker=claim.get('worker')))
        if execute: path.unlink()
    print(json.dumps(dict(stale_claims=stale, cleared=execute), indent=2))
    return stale


def main():
    from dotenv import load_dotenv
    load_dotenv()
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', choices=['work', 'one-job', 'status', 'clear-stale'])
    p.add_argument('--root', type=Path, default=Path('data/cutoff_recovery_20261121'))
    p.add_argument('--job')
    p.add_argument('--wait-hours', type=float, default=72)
    p.add_argument('--poll-seconds', type=float, default=30)
    p.add_argument('--max-jobs', type=int)
    p.add_argument('--with-interpretation', action='store_true', help='One idle worker runs fresh R_EA interpretation after fit work is claimed')
    p.add_argument('--execute', action='store_true', help='clear-stale: actually clear claims')
    p.add_argument('--confirm-allocations-stopped', action='store_true')
    a = p.parse_args()
    if a.command == 'one-job':
        if not a.job: p.error('one-job requires --job')
        one_job(a.root.resolve(), a.job)
    elif a.command == 'status': status(a.root.resolve())
    elif a.command == 'clear-stale':
        clear_stale(a.root.resolve(), execute=a.execute, confirmed_stopped=a.confirm_allocations_stopped)
    else:
        if a.wait_hours <= 0 or not 0 < a.poll_seconds <= 60: p.error('Invalid wait/poll duration')
        work(a.root.resolve(), wait_hours=a.wait_hours, poll_seconds=a.poll_seconds, max_jobs=a.max_jobs,
             with_interpretation=a.with_interpretation)


if __name__ == '__main__':
    main()
