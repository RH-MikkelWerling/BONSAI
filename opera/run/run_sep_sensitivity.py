"""Run one prepared SEP sensitivity arm on one allocated GPU; no test evaluation."""
import argparse
import fcntl
import hashlib
import json
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--arm', choices=['exclude_sep', 'include_sep'], required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    manifest = json.loads((root / 'sensitivity.json').read_text())
    for name, digest in manifest['config_sha256'].items():
        if hashlib.sha256((root / 'configs' / name).read_bytes()).hexdigest() != digest:
            raise ValueError(f'Config changed after preparation: {name}; use a new sensitivity root')
    arm = root / args.arm
    arm.mkdir(exist_ok=True)
    # Prevent duplicate launchers from sharing checkpoints or GPUs for this arm.
    with (arm / 'run.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        jobs = [('bonsai.run.pretrain', f'pretrain_{args.arm}', arm / 'pretrain')]
        jobs += [('opera.run.contrastive_multicohort', f'joint_{args.arm}_s{seed}',
                  arm / f'joint_s{seed}') for seed in manifest['seeds']]
        for module, name, destination in jobs:
            # Training entrypoints resume their own last.ckpt. No primary outputs touched.
            destination.mkdir(parents=True, exist_ok=True)
            if module != 'bonsai.run.pretrain':
                subprocess.run([sys.executable, '-m', 'opera.run.preflight_direct_competing_risk',
                                '--config-dir', str(root / 'configs'), '--config-name', name,
                                '--require-existing-paths'], check=True)
            print(f'Running {name}; log: {destination / "sensitivity.log"}', flush=True)
            with (destination / 'sensitivity.log').open('a') as log:
                subprocess.run([sys.executable, '-u', '-m', module,
                                '--config-path', str(root / 'configs'), '--config-name', name,
                                f'hydra.run.dir={destination}'], stdout=log,
                               stderr=subprocess.STDOUT, check=True)
            if not (destination / 'last.ckpt').is_file():
                raise RuntimeError(f'Training did not produce {destination / "last.ckpt"}')
            if module != 'bonsai.run.pretrain':
                subprocess.run([sys.executable, '-m', 'opera.run.evaluate_direct_competing_risk',
                                '--checkpoint', str(destination / 'best.ckpt'),
                                '--config-dir', str(root / 'configs'), '--config-name', name,
                                '--split', 'tuning', '--device', 'cuda',
                                '--output-dir', str(destination / 'tuning_predictions')], check=True)
        print(f'{args.arm} training and tuning predictions complete.', flush=True)


if __name__ == '__main__':
    main()
