"""Read-only checks of the 18 Nov story aggregate export before download."""
import argparse
from collections import Counter
import json
from pathlib import Path

import pandas as pd
import numpy as np


def check(root):
    status = json.loads((root / 'story_status.json').read_text())
    for step in ['embeddings', 'clinical']:
        if status.get(step, {}).get('status') != 'complete':
            raise ValueError(f'{step} is not complete')
    if status['clinical'].get('flipi2_endpoint') != 'overall_survival':
        raise ValueError('Unexpected clinical endpoint receipt')
    tables = root / 'ready/tables'
    report = {}
    for family in ['joint', 'opera']:
        meta = pd.read_csv(tables / f'embedding_grids_{family}_meta.csv')
        models = set(meta.model)
        if len(models) != 3 or meta.model.duplicated().any():
            raise ValueError(f'{family}: expected exactly three model columns')
        sources = status['embeddings'].get(family, {}).get('prediction_sources', {})
        if set(sources) != models or any(not str(v).endswith('direct_cr_predictions.parquet') for v in sources.values()):
            raise ValueError(f'{family}: incomplete prediction-source receipt; rerun embeddings with current code')
        counts = Counter()
        for chunk in pd.read_csv(tables / f'embedding_grids_{family}.csv.gz',
                                 usecols=['model', 'layer', 'value'], chunksize=100000):
            if not chunk.model.isin(models).all():
                raise ValueError(f'{family}: grid contains unknown model')
            if not np.isfinite(chunk.value.to_numpy(float)).all():
                raise ValueError(f'{family}: nonfinite exported grid values')
            counts.update(chunk.groupby(['model', 'layer']).size().to_dict())
        for model in models:
            for layer in ['count', 'disease', 'ipi', 'mortality', 'predicted']:
                if not counts[(model, layer)]:
                    raise ValueError(f'{family}/{model}: missing {layer} layer')
        if (tables / f'embedding_neighbourhood_{family}.csv').exists():
            raise ValueError(f'{family}: obsolete metrics still present')
        report[family] = {'models': sorted(models), 'grid_files_present': True,
                          'prediction_sources': sources,
                          'predicted_grid_rows': {model: counts[(model, 'predicted')] for model in sorted(models)},
                          'index_risk_groups': status['embeddings'][family].get('index_risk_groups')}
    return {'status': 'passed', 'families': report,
            'manual_checks_remaining': ['Compare logged risk-group counts with the matching Table 1 population.',
                                         'Inspect one clinical PNG for Pagella and IPI label.']}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('data/story_closure_20261116'))
    args = parser.parse_args()
    print(json.dumps(check(args.root), indent=2))
