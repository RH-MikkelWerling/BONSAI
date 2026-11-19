"""Find saved downstream references to local pretraining; never fit/load models.

Also verifies resolved RI/FZ bundle configuration flags. Reads only bounded
metadata/configuration files, not checkpoints, predictions or patient data.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re

import yaml

SKIP = {'predictions', 'prepared', 'replays', 'site-packages', '__pycache__', '.git'}


def read(path):
    if path.stat().st_size > 16 * 1024 * 1024:
        raise ValueError('Metadata/config exceeds 16 MiB; skipped')
    value = path.read_text()
    return json.loads(value) if path.suffix == '.json' else yaml.safe_load(value)


def references(tree, prefix=''):
    if isinstance(tree, dict):
        for key, value in tree.items():
            yield from references(value, f'{prefix}.{key}' if prefix else str(key))
    elif isinstance(tree, list):
        for i, value in enumerate(tree):
            yield from references(value, f'{prefix}[{i}]')
    elif isinstance(tree, str) and '.ckpt' in tree:
        # Include earlier variants, but label their release instead of merging them.
        if ('pretrain_scaled_time2vec_hidden256' in tree
                or 'pretrain_harmonized_scaled_time2vec_len8192_20ep' in tree):
            name = Path(tree).name.lower()
            kind = ('epoch10' if re.search(r'epoch[_=-]?0?10(?:\D|$)', name)
                    else 'epoch20' if re.search(r'epoch[_=-]?0?20(?:\D|$)', name)
                    else 'last_epoch_unverified' if name == 'last.ckpt'
                    else 'best_epoch_unverified' if name == 'best.ckpt'
                    else 'epoch_unverified')
            yield {'field': prefix, 'checkpoint_reference': tree, 'reference_kind': kind,
                   'current_release': 'daly_care_t100_joined_bins_with_sor_final_v2' in tree}


def search(roots, max_files=30000):
    result = {'scope': 'Read-only provenance search; reference filenames do not verify checkpoint epochs',
              'references': [], 'controls': [], 'probe_candidates': [], 'head_audit_candidates': [],
              'errors': [], 'truncated': False, 'files_read': 0}
    def note_probe(path, data):
        text = yaml.safe_dump(data).lower()
        if any(token in text for token in ['linear_probe', 'linear-probe', 'head_only',
                                           'freeze_encoder: true', 'freeze_encoder=true']):
            result['probe_candidates'].append({
                'config_path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                'note': 'Candidate configuration only: verify frozen encoder AND linear head, fitted checkpoint and held-out selection/results separately.'})
    seen = set()
    for root in roots:
        if not root.is_dir():
            result['errors'].append({'path': str(root), 'reason': 'directory_missing'})
            continue
        for directory, children, names in os.walk(root, followlinks=False):
            children[:] = sorted(c for c in children if c not in SKIP
                                 and not c.startswith('.') and 'INTERNAL' not in c.upper()
                                 and not (Path(directory) / c).is_symlink())
            if any(token in str(directory).lower() for token in ['mechanism', 'linear_probe', 'linear-probe', 'head_only']):
                for name in sorted(set(names) & {'report.json', 'manifest.json', 'training_selection.json',
                                                'selection_receipt.json', 'router_mechanism_review.zip'}):
                    result['head_audit_candidates'].append({'path': str(Path(directory) / name),
                        'note': 'Filename candidate; checkpoint identity and completion not yet verified'})
            for name in sorted(set(names) & {'checkpoint_metadata.json', 'hparams.yaml', 'config.yaml', 'bundle.json'}):
                path = Path(directory) / name
                if path.is_symlink() or path.resolve() in seen:
                    continue
                if result['files_read'] >= max_files:
                    result['truncated'] = True
                    return result
                seen.add(path.resolve())
                result['files_read'] += 1
                try:
                    data = read(path)
                    note_probe(path, data)
                    for ref in references(data):
                        result['references'].append({'metadata_file': str(path), **ref})
                    if name == 'bundle.json' and isinstance(data, dict):
                        for row in data.get('runs', []):
                            if row.get('config_path'):
                                candidate = path.parent / row['config_path']
                                note_probe(candidate, read(candidate))
                            if row.get('origin') not in {'RI', 'FZ'}:
                                continue
                            config = path.parent / row['config_path']
                            cfg = read(config)
                            result['controls'].append({
                                'bundle': str(path), 'name': row.get('name'), 'origin': row['origin'],
                                'run_directory': row.get('run_directory'), 'config_path': str(config),
                                'config_hash_matches': hashlib.sha256(config.read_bytes()).hexdigest() == row.get('config_sha256'),
                                'encoder_initialization': cfg.get('encoder_initialization'),
                                'freeze_encoder': (cfg.get('model') or {}).get('freeze_encoder'),
                                'upstream_references': list(references(cfg)),
                                'note': 'Config flags only; runtime checkpoint/prediction verification is separate.'})
                except (OSError, ValueError, KeyError, UnicodeError, yaml.YAMLError) as error:
                    result['errors'].append({'path': str(path), 'reason': str(error)})
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scan-root', type=Path, action='append')
    parser.add_argument('--max-files', type=int, default=30000)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.max_files < 1:
        parser.error('--max-files must be positive')
    roots = args.scan_root or [Path('data/models'), Path('data/paper_256_v2'), Path('data/paper_256_v3')]
    result = search(roots, args.max_files)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as handle:
        json.dump(result, handle, indent=2)
    print(json.dumps({'output': str(args.output), 'references': len(result['references']),
                      'control_configs': len(result['controls']), 'errors': len(result['errors']),
                      'probe_candidates': len(result['probe_candidates']),
                      'head_audit_candidates': len(result['head_audit_candidates']),
                      'files_read': result['files_read'], 'truncated': result['truncated']}, indent=2))
