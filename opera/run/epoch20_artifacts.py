"""Verified isolated epoch20 predictions for exploratory paired analyses."""
import copy
import json
from pathlib import Path


def verified_artifacts(root, reference, alias):
    from bonsai.functional.checkpointing import verified_training_selection
    from opera.run.finish_figure_inputs import sha
    root = Path(root)
    bundle = json.loads((root / 'bundle/bundle.json').read_text())
    row, = bundle['runs']
    selection = verified_training_selection(root / 'neural_runs' / row['run_directory'])
    expected_config = row['config_sha256']
    if sha(root / 'bundle' / row['config_path']) != expected_config:
        raise ValueError(f'{alias} frozen configuration changed')
    records = []
    for split in ['tuning', 'held_out']:
        folder = root / 'predictions' / row['run_directory'] / split
        path = folder / 'direct_cr_predictions.parquet'
        marker = json.loads((folder / 'evaluation_complete.json').read_text())
        if (marker['inputs']['config_sha256'] != expected_config or
            marker['inputs']['checkpoint_sha256'] != selection['best_checkpoint_sha256'] or
            marker['inputs']['split'] != split or marker['prediction_sha256'] != sha(path)):
            raise ValueError(f'Unverified {alias} evaluation: {split}')
        records.append({**copy.deepcopy(reference), 'model': alias, 'split': split,
                        'path': str(path.resolve()), 'format': 'direct_cif',
                        'cohort_from_metadata': True})
    return records
