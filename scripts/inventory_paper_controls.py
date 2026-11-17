"""Bounded, read-only discovery of pretraining/control receipts; no model/data loads.

Reports candidate paths, not scientific verification. Never opens checkpoints,
patient records, token caches or prediction tables. Does not require rg.
"""
import argparse
import csv
import json
import os
from pathlib import Path

PRUNE = {'predictions', 'patient_cache_INTERNAL', 'token_attributions_INTERNAL',
         'prepared', 'replays', 'site-packages', '.git', '__pycache__'}
NAMES = {'hparams.yaml', 'config.yaml', 'metrics.csv', 'checkpoint_metadata.json',
         'training_selection.json', 'bundle.json', 'completion.json', 'report.json',
         'run.json', 'loss_check.csv', 'memorization_summary.csv',
         'memorization_excess.csv', 'summary.txt'}


def inspect(path):
    row = {'path': str(path), 'bytes': path.stat().st_size,
           'scope': 'candidate_only_not_verified'}
    if path.name == 'metrics.csv' and row['bytes'] <= 16 * 1024 * 1024:
        with path.open(newline='') as handle:
            records = csv.DictReader(handle)
            epochs, last_metrics = [], {}
            for item in records:
                if item.get('epoch'):
                    try:
                        epochs.append(float(item['epoch']))
                    except ValueError:
                        pass
                for key in ('val/loss', 'val/cr_cif_auc_macro', 'val/roc_auc_macro'):
                    if item.get(key):
                        last_metrics[key] = item[key]
        row.update(max_logged_epoch=max(epochs) if epochs else None,
                   last_logged_metrics=last_metrics,
                   note='Logged epoch may be zero-based; last metric is not tuning-best selection.')
    if path.suffix == '.ckpt':
        row['note'] = 'Checkpoint filename/size only; not loaded or hashed.'
    return row


def inventory(roots, limit):
    rows, errors, truncated = [], [], False
    for root in roots:
        if not root.is_dir():
            errors.append({'path': str(root), 'reason': 'directory_missing'})
            continue
        for directory, children, names in os.walk(root, followlinks=False):
            children[:] = sorted(c for c in children if c not in PRUNE
                                 and not c.startswith('.') and 'INTERNAL' not in c.upper())
            for name in sorted(names):
                if name not in NAMES and not name.endswith('.ckpt') and not name.endswith('_feature_missingness.csv'):
                    continue
                path = Path(directory) / name
                if path.is_symlink():
                    continue
                if len(rows) >= limit:
                    truncated = True
                    break
                try:
                    rows.append(inspect(path))
                except (OSError, csv.Error, UnicodeError) as error:
                    errors.append({'path': str(path), 'reason': str(error)})
            if truncated:
                break
        if truncated:
            break
    return {'scope': 'Read-only candidate inventory; no fitting, inference or checkpoint verification',
            'roots': [str(p) for p in roots], 'truncated': truncated,
            'candidates': rows, 'errors': errors,
            'review': ['Locate local 10/20-epoch pretraining AND matched downstream fits.',
                       'Identify random-init/frozen-encoder aliases from bundle/config receipts.',
                       'Identify frozen-head report checkpoint before reusing historical audit.',
                       'Locate memorisation and tabular missingness receipts.',
                       'Verify data release, vocabulary, seed and selection before comparison.']}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scan-root', type=Path, action='append')
    parser.add_argument('--max-matches', type=int, default=5000)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.max_matches < 1:
        parser.error('--max-matches must be positive')
    result = inventory(args.scan_root or [Path('data'), Path('logs')], args.max_matches)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as handle:
        json.dump(result, handle, indent=2)
    print(f'Candidate inventory: {args.output}; files={len(result["candidates"])}; truncated={result["truncated"]}')
