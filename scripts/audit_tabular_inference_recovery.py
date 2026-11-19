"""Inventory possible saved tabular estimators without loading models or patient data.

File presence is only a candidate: inference requires the exact fitted estimator,
preprocessor, feature order and selection provenance. Never refits or predicts.
"""
import argparse
import json
import os
from pathlib import Path


def inventory(roots, limit=300000):
    result = {'scope': 'Read-only filename/config inventory; no models or patient tables loaded',
              'missing_roots': [], 'candidate_serialized_files': [], 'metadata_examples': [],
              'selected_hyperparameters': [],
              'metadata_files': 0, 'files_examined': 0, 'truncated': False}
    for root in roots:
        root = Path(root)
        if not root.is_dir():
            result['missing_roots'].append(str(root)); continue
        for directory, children, names in os.walk(root, followlinks=False):
            children[:] = sorted(n for n in children if n not in
                {'.git', '__pycache__', 'analysis', 'predictions', 'patient_cache_INTERNAL',
                 'batches_INTERNAL', 'buckets_INTERNAL', 'prepared', 'replays'}
                and 'INTERNAL' not in n.upper())
            for name in sorted(names):
                result['files_examined'] += 1
                if result['files_examined'] > limit:
                    result['truncated'] = True
                    return result
                path = Path(directory) / name
                lower = name.lower()
                if lower.endswith('_metadata.json') and 'xgboost' in lower:
                    result['metadata_files'] += 1
                    try:
                        if path.stat().st_size < 2_000_000:
                            metadata = json.loads(path.read_text())
                            result['selected_hyperparameters'].append({'path': str(path), **{
                                k: metadata[k] for k in ['model_name', 'backend', 'tune_params',
                                    'n_features_used', 'fit_split', 'selection_split', 'evaluation_split'] if k in metadata}})
                    except (OSError, ValueError):
                        pass
                    if len(result['metadata_examples']) < 20:
                        entry = {'path': str(path)}
                        try:
                            if path.stat().st_size < 2_000_000:
                                data = json.loads(path.read_text())
                                entry['settings'] = {k: v for k, v in data.items() if k in
                                    {'model_name', 'backend', 'features', 'source_features', 'feature_profile',
                                     'feature_columns', 'feature_columns_used', 'tune_params', 'seed',
                                     'fit_split', 'selection_split', 'evaluation_split', 'label_estimand'}}
                        except (OSError, ValueError) as error:
                            entry['error'] = str(error)
                        result['metadata_examples'].append(entry)
                serialized = path.suffix.lower() in {'.pkl', '.pickle', '.joblib', '.ubj', '.bin', '.bst', '.model'}
                named_model_json = path.suffix.lower() == '.json' and any(
                    token in lower for token in ['booster', 'pipeline', 'preprocessor', 'estimator', 'xgboost_model'])
                if serialized or named_model_json:
                    try:
                        stat = path.stat()
                        result['candidate_serialized_files'].append({'path': str(path), 'bytes': stat.st_size})
                    except OSError:
                        pass
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--roots', nargs='+', type=Path, required=True)
    p.add_argument('--max-files', type=int, default=300000)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    result = inventory(args.roots, args.max_files)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as handle:
        json.dump(result, handle, indent=2)
    print(f'Inventory: {args.output}; truncated={result["truncated"]}; '
          f'candidate files={len(result["candidate_serialized_files"])}')


if __name__ == '__main__':
    main()
