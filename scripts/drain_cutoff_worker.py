"""On this allocation only: finish the current fit, then reload the GPU worker.

Pauses the queue parent, leaving its existing job child running. Keeps that
parent's claims alive until its children finish, then retires the old parent and
launches the updated worker. No checkpoint, fit or evaluation is deleted.
"""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import socket
import time


def command(pid):
    return [x.decode() for x in Path(f'/proc/{pid}/cmdline').read_bytes().split(b'\0') if x]


def process_state(pid):
    try:
        for line in Path(f'/proc/{pid}/status').read_text().splitlines():
            if line.startswith('State:'):
                return line.split()[1]
    except FileNotFoundError:
        return None


def live(pid):
    return process_state(pid) not in {None, 'Z', 'X'}


def worker_root(args, cwd):
    try:
        module = args.index('-m')
        if args[module+1:module+3] != ['opera.run.cutoff_recovery_worker', 'work']:
            return None
        value = args[args.index('--root')+1] if '--root' in args else 'data/cutoff_recovery_20261121'
        path = Path(value)
        return (path if path.is_absolute() else Path(cwd)/path).resolve()
    except (ValueError, IndexError):
        return None


def workers(root):
    found = []
    for directory in Path('/proc').iterdir():
        if not directory.name.isdigit():
            continue
        try:
            if directory.stat().st_uid != os.getuid():
                continue
            pid = int(directory.name)
            if worker_root(command(pid), (directory/'cwd').resolve()) == root and live(pid):
                found.append(pid)
        except (OSError, ProcessLookupError):
            continue
    return found


def children(pid):
    path = Path(f'/proc/{pid}/task/{pid}/children')
    return [int(x) for x in path.read_text().split()] if path.exists() else []


def owned_claims(root, pid):
    patterns = ('neural_runs/**/.worker_claim.json', 'epoch20_priority/**/.worker_claim.json',
                'interpretation_task/.worker_claim.json')
    claims = []
    for pattern in patterns:
        for path in root.glob(pattern):
            data = json.loads(path.read_text())
            if data.get('pid') == pid and data.get('host') == socket.gethostname():
                claims.append((path, data['token']))
    return claims


def refresh_claim(path, token):
    data = json.loads(path.read_text())
    if data['token'] != token:
        raise RuntimeError(f'Claim ownership changed: {path}')
    data['heartbeat_at'] = datetime.now(timezone.utc).isoformat()
    temporary = path.with_name(f'.drain_{os.getpid()}.writing')
    temporary.write_text(json.dumps(data))
    temporary.replace(path)


def validate_worker(root, pid):
    if Path(f'/proc/{pid}').stat().st_uid != os.getuid():
        raise ValueError('Worker belongs to another user')
    if worker_root(command(pid), Path(f'/proc/{pid}/cwd').resolve()) != root:
        raise ValueError('PID is not a recovery queue worker for this root')


def drain(root, pid, poll_seconds=10):
    validate_worker(root, pid)
    paused = retired = False
    claims = []
    try:
        os.kill(pid, signal.SIGSTOP)
        paused = True
        deadline = time.monotonic()+5
        while live(pid) and process_state(pid) not in {'T', 't'}:
            if time.monotonic() > deadline:
                raise RuntimeError('Worker did not pause')
            time.sleep(.05)
        if not live(pid):
            raise RuntimeError('Worker exited before handover')
        # Once stopped, this parent cannot claim or spawn another fit. Existing
        # one-job/interpretation children can still finish all their subprocesses.
        active = children(pid)
        claims = owned_claims(root, pid)
        print(f'DRAINING worker {pid}; children={active}; claims={len(claims)}', flush=True)
        for child in active:
            try:
                print(f'Continuing child {child}: {" ".join(command(child))}', flush=True)
            except FileNotFoundError:
                pass
        while any(live(child) for child in active):
            if not live(pid):
                raise RuntimeError('Paused queue parent disappeared; inspect the allocation')
            for path, token in claims:
                refresh_claim(path, token)
            print('Current job still running; next fit is held until handover.', flush=True)
            time.sleep(poll_seconds)
        # Termination is pending before CONT, so the old loop cannot claim a
        # new job. Do not signal its former children or their training groups.
        os.kill(pid, signal.SIGTERM)
        os.kill(pid, signal.SIGCONT)
        deadline = time.monotonic()+10
        while live(pid):
            if time.monotonic() > deadline:
                raise RuntimeError('Old worker did not terminate; refusing to start a second worker')
            time.sleep(.1)
        retired = True
        for path, token in claims:
            if path.exists() and json.loads(path.read_text()).get('token') == token:
                path.unlink()
                print(f'Released stopped worker claim: {path}', flush=True)
        print('Old worker retired; current job outputs retained.', flush=True)
    finally:
        if paused and not retired and live(pid):
            os.kill(pid, signal.SIGCONT)
            print('Handover interrupted; resumed original worker.', flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, default=Path('data/cutoff_recovery_20261121'))
    p.add_argument('--pid', type=int)
    p.add_argument('--execute', action='store_true')
    a = p.parse_args()
    if not Path('/proc/self').exists():
        p.error('Run inside the Linux worker allocation')
    root = a.root.resolve()
    repo = Path(__file__).resolve().parents[1]
    os.chdir(repo)
    found = workers(root)
    print(json.dumps(dict(host=socket.gethostname(), worker_pids=found, execute=a.execute)), flush=True)
    if a.pid is not None:
        if a.pid not in found:
            p.error('Requested PID is not one of this allocation\'s matching workers')
        pid = a.pid
    elif len(found) != 1:
        p.error('Expected one worker on this allocation; use --pid after inspecting the listed processes')
    else:
        pid = found[0]
    if not a.execute:
        return
    def interrupted(signum, frame):
        raise KeyboardInterrupt(f'Handover received signal {signum}')
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    drain(root, pid)
    os.environ['OPERA_RECOVERY_ROOT'] = str(root)
    print('Launching updated worker; prepared epoch20 checks take priority.', flush=True)
    os.execv('/bin/bash', ['bash', str(repo/'scripts/run_cutoff_recovery.sh'), 'worker'])


if __name__ == '__main__':
    main()
