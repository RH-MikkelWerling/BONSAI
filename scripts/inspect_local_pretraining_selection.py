"""Read selected epoch scalars and saved joint/OPERA configs; no fitting/inference.

Loads only known locally generated pretraining checkpoints on CPU with memory
mapping. Does not export weights. No eager fallback if mmap is unsupported.
"""
import argparse
import gc
import hashlib
import json
from pathlib import Path

import yaml


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as handle:
        for part in iter(lambda: handle.read(1024 * 1024), b''):
            h.update(part)
    return h.hexdigest()


def checkpoint_scalars(path):
    import torch
    # These are the user's own generated checkpoints, not downloaded models.
    checkpoint = torch.load(path, map_location='cpu', mmap=True, weights_only=False)
    result = {'path': str(path), 'epoch_zero_based': checkpoint.get('epoch'),
              'global_step': checkpoint.get('global_step'), 'sha256': digest(path)}
    del checkpoint
    gc.collect()
    return result


def inspect(repo):
    directory = repo / ('data/models/daly_care_t100_joined_bins_with_sor_final_v2/'
                        'pretrain_scaled_time2vec_hidden256/version_0')
    report = {'scope': 'Read-only checkpoint scalar/config inspection; no weights exported, no fitting',
              'checkpoints': [], 'downstream_configs': [], 'errors': []}
    for name in ['epoch_010.ckpt', 'best.ckpt', 'last.ckpt']:
        path = directory / name
        try:
            report['checkpoints'].append(checkpoint_scalars(path))
        except Exception as error:
            report['errors'].append({'path': str(path), 'reason': str(error)})
    for release in ['paper_256_v2', 'paper_256_v3']:
        for bundle_path in sorted((repo / 'data' / release / 'bundles').glob('*/bundle.json')):
            try:
                bundle = json.loads(bundle_path.read_text())
                for row in bundle.get('runs', []):
                    name = row.get('name', '')
                    short = name.split('/')[-1]
                    if not name.startswith('seed42/L/') or not (
                        short == 'J_all_diseases_all_outcomes'
                        or short.startswith(('fix_J_OA_OWN', 'J_OA_OWN'))):
                        continue
                    path = bundle_path.parent / row['config_path']
                    config = yaml.safe_load(path.read_text())
                    training = config.get('training') or {}
                    report['downstream_configs'].append({
                        'bundle': str(bundle_path), 'name': name,
                        'run_directory': row['run_directory'], 'config': str(path),
                        'config_hash_matches': digest(path) == row.get('config_sha256'),
                        'upstream_checkpoint': config.get('dapt_ckpt'),
                        'encoder_initialization': config.get('encoder_initialization'),
                        'freeze_encoder': (config.get('model') or {}).get('freeze_encoder'),
                        'training': {k: training.get(k) for k in ['learning_rate', 'encoder_lr_multiplier',
                            'epochs', 'batch_size', 'accumulate_grad_batches', 'early_stopping_patience',
                            'eval_monitor_metric', 'train_patient_fraction']},
                        'checkpoint_exists': (repo / 'data' / release / 'neural_runs' /
                                              row['run_directory'] / 'best.ckpt').is_file()})
            except Exception as error:
                report['errors'].append({'path': str(bundle_path), 'reason': str(error)})
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo-root', type=Path, default=Path.cwd())
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = inspect(args.repo_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as handle:
        json.dump(result, handle, indent=2)
    print(json.dumps({'output': str(args.output),
                      'checkpoints': [{k: v for k, v in r.items() if k != 'sha256'} for r in result['checkpoints']],
                      'downstream_configs': len(result['downstream_configs']), 'errors': result['errors']}, indent=2))
