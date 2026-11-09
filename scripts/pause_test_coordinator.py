"""Pause only the shard-zero shell; GPU scoring children continue unaffected."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import signal
import socket
import time


def coordinator(argv, tag):
    return (len(argv) >= 6 and Path(argv[0]).name == 'bash'
            and Path(argv[1]).name == 'train_then_evaluate.sh'
            and argv[3] == '0' and argv[5] == tag)


def processes():
    if not Path('/proc').is_dir():
        raise RuntimeError('Run this on the Linux allocation hosting shard zero')
    result = {}
    for path in Path('/proc').iterdir():
        if not path.name.isdigit():
            continue
        try:
            if path.stat().st_uid != os.getuid():
                continue
            argv = (path / 'cmdline').read_bytes().decode().strip('\0').split('\0')
            status = dict(line.split(':', 1) for line in (path / 'status').read_text().splitlines() if ':' in line)
            stat = (path / 'stat').read_text().rsplit(')', 1)[1].split()
            result[int(path.name)] = dict(argv=argv, ppid=int(status['PPid']),
                start_ticks=stat[19], state=status['State'].strip()[0])
        except (OSError, ValueError, UnicodeError):
            continue
    return result


def descendants(snapshot, pid):
    found = {pid}
    while True:
        new = {p for p, row in snapshot.items() if row['ppid'] in found}
        if new <= found:
            return found - {pid}
        found |= new


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tag', required=True)
    parser.add_argument('--paper-root', type=Path, default=Path('data/paper_256_v3'))
    parser.add_argument('--finish', action='store_true',
        help='After scoring handoff completes, terminate the paused shell (run on its original allocation)')
    args = parser.parse_args()
    if Path(args.tag).name != args.tag:
        raise ValueError('Tag must be a single directory name')
    snapshot = processes()
    directory = args.paper_root / 'predictions/.overnight' / args.tag
    if args.finish:
        record = json.loads((directory / 'cpu_handoff.json').read_text())
        if not record.get('scoring_complete'):
            raise RuntimeError('Scoring handoff is not complete; keep this allocation alive')
        if record['hostname'] != socket.gethostname():
            raise RuntimeError('Run --finish on the original shard-zero allocation')
        pid = record['pid']
        if pid not in snapshot:
            print('Original coordinator has already exited'); return
        if snapshot[pid]['start_ticks'] != record['start_ticks']:
            raise RuntimeError('PID has been reused; no signal sent')
        live = [p for p in descendants(snapshot, pid) if snapshot[p]['state'] != 'Z']
        if live:
            raise RuntimeError(f'Scoring children are still exiting: {live}; retry shortly')
        os.kill(pid, signal.SIGTERM)
        try:
            os.kill(pid, signal.SIGCONT)
        except ProcessLookupError:
            pass
        print('Original paused shell terminated; CPU analysis runs independently'); return
    matches = [pid for pid, row in snapshot.items() if coordinator(row['argv'], args.tag)]
    if len(matches) != 1:
        raise RuntimeError(f'Expected one shard-zero shell on this allocation; found {matches}')
    pid, = matches
    previous = directory / 'cpu_handoff.json'
    if previous.exists():
        record = json.loads(previous.read_text())
        if (record['pid'] == pid and record['hostname'] == socket.gethostname()
                and record['start_ticks'] == snapshot[pid]['start_ticks']
                and snapshot[pid]['state'] in {'T', 't'}):
            print(f'Coordinator PID {pid} is already paused; no signals sent'); return
        raise RuntimeError('A handoff record already exists; inspect it before changing coordinator state')
    os.kill(pid, signal.SIGSTOP)
    try:
        after = processes()
        for _ in range(200):
            if pid not in after or after[pid]['state'] in {'T', 't'}:
                break
            time.sleep(.01)
            after = processes()
        if pid not in after or after[pid]['state'] not in {'T', 't'}:
            raise RuntimeError('Could not confirm stopped coordinator')
        active_analysis = [p for p in descendants(after, pid)
            if 'opera.run.analyse_locked_predictions' in after[p]['argv']]
        root = args.paper_root / 'analysis' / args.tag
        if active_analysis or root.exists():
            raise RuntimeError('CPU handoff is too late: analysis/setup already started; original coordinator resumed')
        directory.mkdir(parents=True, exist_ok=True)
        record = dict(pid=pid, hostname=socket.gethostname(),
            start_ticks=snapshot[pid]['start_ticks'], state='coordinator_paused',
            note='Only the shell was stopped. Its GPU scoring child processes were not signalled.')
        (directory / 'cpu_handoff.json').write_text(json.dumps(record, indent=2) + '\n')
    except BaseException:
        try:
            os.kill(pid, signal.SIGCONT)
        except ProcessLookupError:
            pass
        raise
    print(f'Paused coordinator PID {pid}; GPU scoring continues. Keep this allocation alive.')


if __name__ == '__main__':
    main()
