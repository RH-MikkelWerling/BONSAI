"""Shared-filesystem IG helpers; keep the scientific attribution engine unchanged.

Workers only publish the engine's normal per-patient caches. Their disposable
tables stay in workers_INTERNAL; the ordinary serial resume verifies and collects
the caches once, after the queue drains. No patient data leave this workspace.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import time
import uuid


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, default=str))
    os.replace(temporary, path)


@contextmanager
def claim(path, *, shared=False):
    """Kernel-owned claims survive long jobs but release on exit/walltime.

    Never delete lock files: unlinking an active inode would allow two owners.
    Requires working cross-node flock on the shared filesystem (see runbook).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a') as handle:
        try:
            fcntl.flock(handle, (fcntl.LOCK_SH if shared else fcntl.LOCK_EX) | fcntl.LOCK_NB)
        except BlockingIOError:
            yield False
            return
        try:
            yield handle
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def progress(directory):
    directory = Path(directory)
    run = directory / 'attribution_run.json'
    planned = json.loads(run.read_text()).get('planned_patients') if run.exists() else None
    completed = sum(1 for _ in (directory / 'patient_cache_INTERNAL').glob('patient-*/complete.json'))
    return {'planned_patients': planned, 'completed_patients': completed,
            'complete': planned is not None and completed == planned}


def ready(directory):
    manifest = directory.parent / 'interpretation_manifest.json'
    try:
        pilot_complete = (manifest.is_file() and json.loads(manifest.read_text())
                          .get('steps', {}).get('pilot', {}).get('status') == 'complete')
    except json.JSONDecodeError:
        return False  # Older coordinators wrote this receipt in place.
    return pilot_complete and all((directory / name).is_file() for name in (
        'attribution_intent.json', 'attribution_run.json',
        'baseline_INTERNAL.pt', 'baseline_sha256.txt'))


class PatientQueue:
    """Adapter at the existing cache boundary, not at the IG calculation."""
    def __init__(self, directory, cached_explanation, worker_dir):
        self.directory = Path(directory)
        self.cached_explanation = cached_explanation
        self.worker_dir = Path(worker_dir)
        self.completed = 0
        self.visited = 0

    def __call__(self, ignored_directory, rank, meta, compute):
        self.visited += 1
        cache = self.directory / 'patient_cache_INTERNAL' / f'patient-{rank:06d}'
        empty = {'tokens': [], 'patients': [], 'faithfulness': [], 'sanity': []}
        with claim(self.directory / 'patient_claims_INTERNAL' / f'{rank:06d}.lock') as owned:
            if not owned:
                return empty
            marker = cache / 'complete.json'
            if marker.exists():
                if json.loads(marker.read_text())['meta'] != meta:
                    raise ValueError('Cached attribution patient roster changed')
                # Final collection hash-verifies every cache using the original engine.
                return empty
            self.cached_explanation(self.directory, rank, meta, compute)
            self.completed += 1
            write_json(self.worker_dir / 'status.json', {
                'status': 'running', 'completed_here': self.completed,
                'visited': self.visited, 'updated_unix': time.time(),
            })
            if self.completed == 1 or self.completed % 10 == 0:
                print(f'IG helper completed {self.completed} patients; shared {progress(self.directory)}', flush=True)
        # Do not accumulate patient tables in each helper's RAM or overwrite final tables.
        return empty


def run_ig_pass(directory):
    """One model load, dynamic patient claims across every joining allocation."""
    from opera.run import explain_direct_competing_risk as engine
    intent_path = directory / 'attribution_intent.json'
    intent = json.loads(intent_path.read_text())
    queue_identity = {'intent_sha256': digest(intent_path),
                      'queue_implementation_sha256': digest(Path(__file__))}
    # Serialize only this tiny initialization; patient work uses separate locks.
    with (directory / '.parallel-identity.lock').open('a') as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        path = directory / 'parallel_identity_INTERNAL.json'
        if path.exists() and json.loads(path.read_text()) != queue_identity:
            raise ValueError('Parallel queue code/intent changed; stop helpers before reviewing the migration')
        if not path.exists():
            write_json(path, queue_identity)
    worker_dir = directory / 'workers_INTERNAL' / f'{socket.gethostname()}-{os.getpid()}-{uuid.uuid4().hex[:8]}'
    worker_dir.mkdir(parents=True)
    for name in ('attribution_intent.json', 'baseline_INTERNAL.pt', 'baseline_sha256.txt'):
        shutil.copyfile(directory / name, worker_dir / name)
    if digest(worker_dir / 'baseline_INTERNAL.pt') != (worker_dir / 'baseline_sha256.txt').read_text().strip():
        raise ValueError('Cached baseline changed')
    # Recover defaults and required path conversions from the ORIGINAL parser.
    settings = intent['settings']
    args = engine.build_parser().parse_args([
        '--checkpoint', settings['checkpoint'], '--config-name', settings['config_name'],
        '--output-dir', str(worker_dir)])
    unknown = set(settings) - set(vars(args))
    if unknown:
        raise ValueError(f'Unknown frozen IG arguments: {sorted(unknown)}')
    vars(args).update(settings)
    for name in ('checkpoint', 'config_dir'):
        setattr(args, name, Path(getattr(args, name)))
    args.output_dir, args.resume, args.max_new_patients = worker_dir, True, None
    # Match the paper's established execution budget; this is excluded from intent.
    args.token_budget = 16384
    original = engine.cached_explanation
    queue = PatientQueue(directory, original, worker_dir)
    engine.cached_explanation = queue
    write_json(worker_dir / 'worker_provenance.json', {
        'intent_sha256': digest(intent_path), 'baseline_sha256': digest(directory / 'baseline_INTERNAL.pt'),
        'queue_implementation_sha256': digest(Path(__file__)),
        'scope': 'Disposable helper outputs; final tables require ordinary serial collection',
    })
    try:
        # _run still verifies the full frozen intent, including original engine code,
        # checkpoint, config, vocabulary and input hashes before computing anything.
        engine._run(args)
        expected = progress(directory)['planned_patients']
        if queue.visited != expected:
            raise ValueError(f'Patient plan changed: visited {queue.visited}, expected {expected}')
        write_json(worker_dir / 'status.json', {
            'status': 'pass_complete', 'completed_here': queue.completed,
            'visited': queue.visited, 'updated_unix': time.time(),
        })
    except BaseException as error:
        write_json(worker_dir / 'status.json', {'status': 'failed', 'error': str(error)})
        raise
    finally:
        engine.cached_explanation = original


def restriction_plan(workspace):
    """Freeze the existing dispatcher's independent evaluation commands once."""
    from opera.run.paper_experiments import build_commands, load_environment
    environment = load_environment(workspace / 'interpretation.env')
    args = argparse.Namespace(action='input-restrictions', phase=None, stage=1,
                              shard_count=1, shard_index=0)
    plan_path = workspace / 'restriction_queue_INTERNAL.json'
    identity = {'env_sha256': digest(workspace / 'interpretation.env'),
                'recipe_sha256': digest(workspace / 'attribution.yaml'),
                'intent_sha256': digest(workspace / 'attributions/attribution_intent.json')}
    with claim(workspace / '.restriction-plan.lock') as owned:
        if not owned:
            return None, environment
        if plan_path.exists():
            plan = json.loads(plan_path.read_text())
            if plan['identity'] != identity:
                raise ValueError('Input-restriction queue inputs changed')
        else:
            commands = build_commands(args, environment)
            evaluations = []
            for cmd in commands:
                if 'opera.run.evaluate_direct_competing_risk' in cmd:
                    out = Path(cmd[cmd.index('--output-dir') + 1])
                    evaluations.append({'command': cmd, 'output': str(out)})
            plan = {'identity': identity, 'evaluations': evaluations}
            write_json(plan_path, plan)
    return plan, environment


def run_restrictions(workspace, gate=None):
    plan, environment = restriction_plan(workspace)
    if plan is None:
        return
    locks = workspace / 'restriction_claims_INTERNAL'
    for task in plan['evaluations']:
        out = Path(task['output'])
        key = hashlib.sha256(str(out).encode()).hexdigest()
        with claim(locks / (key + '.lock')) as owned:
            if not owned or restriction_complete(workspace, task):
                continue
            print(f'Input-restriction helper: {out.name}', flush=True)
            log = workspace / 'logs' / f'restriction_{out.name}.log'
            log.parent.mkdir(exist_ok=True)
            with log.open('a') as handle:
                subprocess.run(task['command'], env=environment, stdout=handle,
                               stderr=subprocess.STDOUT, check=True,
                               pass_fds=tuple(h.fileno() for h in (owned, gate) if h is not None))
            prediction = out / 'direct_cr_predictions.parquet'
            write_json(locks / (key + '.complete.json'), {
                'command': task['command'], 'prediction_sha256': digest(prediction)})


def restriction_complete(workspace, task):
    out = Path(task['output'])
    key = hashlib.sha256(str(out).encode()).hexdigest()
    receipt = workspace / 'restriction_claims_INTERNAL' / (key + '.complete.json')
    if not receipt.exists():
        return False
    record = json.loads(receipt.read_text())
    if record['command'] != task['command'] or record['prediction_sha256'] != digest(out / 'direct_cr_predictions.parquet'):
        raise ValueError('Completed input-restriction evaluation changed')
    return True


def wait_for_ig(directory, deadline):
    while time.monotonic() < deadline:
        state = progress(directory)
        print(json.dumps(state), flush=True)
        if state['complete']:
            # Wait for helper model teardown too, before serial collection takes EX.
            with claim(directory / '.attribution.lock') as owned:
                if owned:
                    return
        time.sleep(15)
    raise TimeoutError('IG queue is unfinished; completed patient caches are preserved')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['work', 'wait', 'status', 'probe-hold', 'probe-check'])
    parser.add_argument('--workspace', type=Path, required=True)
    parser.add_argument('--wait-hours', type=float, default=72)
    parser.add_argument('--ig-only', action='store_true', help='Do not help with input restrictions')
    args = parser.parse_args()
    workspace = args.workspace.resolve()
    directory = workspace / 'attributions'
    if args.action in {'probe-hold', 'probe-check'}:
        with claim(workspace / '.cross-node-lock-probe') as owned:
            if args.action == 'probe-hold':
                if not owned:
                    raise RuntimeError('Probe already held elsewhere')
                print('Probe held for 60 seconds; run probe-check on another allocation NOW', flush=True)
                time.sleep(60)
            elif owned:
                raise RuntimeError('Probe NOT blocked: hold it on another allocation before checking')
            else:
                print('PASS: the other allocation blocks this shared-filesystem lock', flush=True)
        return
    if args.action == 'status':
        report = {'ig': progress(directory)}
        plan_path = workspace / 'restriction_queue_INTERNAL.json'
        if plan_path.exists():
            tasks = json.loads(plan_path.read_text())['evaluations']
            report['restrictions'] = {'planned': len(tasks),
                                     'complete': sum(restriction_complete(workspace, task) for task in tasks)}
        report['workers'] = [json.loads(path.read_text()) for path in
                             (directory / 'workers_INTERNAL').glob('*/status.json')]
        report['scope'] = 'Aggregate receipts only; worker receipts are not proof of live processes'
        print(json.dumps(report, indent=2))
        return
    deadline = time.monotonic() + args.wait_hours * 3600
    if args.action == 'wait':
        wait_for_ig(directory, deadline)
        # After IG collection the regular restriction step owns this gate exclusively.
        # Drain helper evaluations now; all missing ones are handled by the dispatcher.
        while time.monotonic() < deadline:
            with claim(workspace / '.restriction-helpers.lock') as owned:
                if owned:
                    path = workspace / 'restriction_queue_INTERNAL.json'
                    if not path.exists():
                        return  # --ig-only helpers: ordinary dispatcher handles restrictions.
                    tasks = json.loads(path.read_text())['evaluations']
                    complete = sum(restriction_complete(workspace, task) for task in tasks)
                    print(f'Input-restriction queue: {complete}/{len(tasks)} complete', flush=True)
                    if complete == len(tasks):
                        return
            time.sleep(15)
        raise TimeoutError('Restriction helpers have not finished')
    else:
        while time.monotonic() < deadline:
            if ready(directory):
                # Serial pilot/full IG owns EX. Helpers can join only after it exits.
                with claim(directory / '.attribution.lock', shared=True) as owned:
                    if owned:
                        if not progress(directory)['complete']:
                            run_ig_pass(directory)
                        break
            print('Waiting for the serial pilot to finish and publish its baseline/plan', flush=True)
            time.sleep(15)
        else:
            raise TimeoutError('No finished pilot available; no IG settings were invented')
        if not args.ig_only:
            import gc
            gc.collect()
            if 'torch' in sys.modules:
                sys.modules['torch'].cuda.empty_cache()
            with claim(workspace / '.restriction-helpers.lock', shared=True) as owned:
                if owned:
                    run_restrictions(workspace, gate=owned)
        print('Helper pass finished; restart helpers if an allocation was interrupted', flush=True)


if __name__ == '__main__':
    main()
