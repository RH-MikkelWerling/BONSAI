"""Aggregate audit of outcome-specific readout vectors; patient arrays stay local."""
from __future__ import annotations

import argparse
from itertools import combinations
import json
from pathlib import Path

import numpy as np

from opera.run.analyse_locked_predictions import file_hash


def audit(root):
    paths = sorted(Path(root).glob('*/routed_patient_vectors_INTERNAL.npz'))
    if len(paths) < 2:
        raise ValueError('Need at least two per-outcome vector exports')
    reference_ids = reference_shared = checkpoint = split = None
    deltas, inputs = {}, []
    for path in paths:
        marker = json.loads(path.with_suffix('.json').read_text())
        outcome = marker['target_outcome']
        if outcome != path.parent.name or outcome in deltas:
            raise ValueError('Outcome marker/directory mismatch or duplicate outcome')
        with np.load(path, allow_pickle=False) as vectors:
            ids = vectors['subject_ids'].astype(str)
            shared = vectors['shared'].astype(np.float64)
            adapted = vectors['adapted'].astype(np.float64)
        if len(set(ids)) != len(ids):
            raise ValueError('Duplicate patient in vector export')
        if (shared.ndim != 2 or shared.shape != adapted.shape or len(shared) != len(ids)
                or len(ids) == 0 or not np.isfinite(shared).all() or not np.isfinite(adapted).all()):
            raise ValueError('Invalid vector shapes or non-finite vectors')
        order = np.argsort(ids)
        ids, shared, adapted = ids[order], shared[order], adapted[order]
        if reference_ids is None:
            reference_ids, reference_shared = ids, shared
            checkpoint, split = marker['checkpoint'], marker['split']
        elif (marker['checkpoint'] != checkpoint or marker['split'] != split
              or not np.array_equal(ids, reference_ids)
              or shared.shape != reference_shared.shape
              or not np.allclose(shared, reference_shared, rtol=1e-5, atol=1e-6)):
            raise ValueError('Vector exports differ in checkpoint/split/patient/shared representation')
        deltas[outcome] = adapted - shared
        inputs.append(dict(outcome=outcome, npz_sha256=file_hash(path),
                           marker_sha256=file_hash(path.with_suffix('.json'))))
    names = list(deltas)
    stacked = np.stack(list(deltas.values()))
    common = stacked.mean(axis=0)
    total_energy = float(np.square(stacked).sum())
    residual_energy = float(np.square(stacked - common).sum())
    pairs = []
    for left, right in combinations(names, 2):
        a, b = deltas[left], deltas[right]
        an, bn = np.linalg.norm(a, axis=1), np.linalg.norm(b, axis=1)
        valid = (an > 1e-12) & (bn > 1e-12)
        cosine = np.sum(a[valid] * b[valid], axis=1) / (an[valid] * bn[valid])
        pairs.append(dict(left=left, right=right, exactly_equal=bool(np.array_equal(a, b)),
            max_abs_delta_difference=float(np.max(np.abs(a-b))),
            median_delta_distance=float(np.median(np.linalg.norm(a-b, axis=1))),
            nonzero_pairs_n=int(valid.sum()),
            median_cosine=float(np.median(cosine)) if len(cosine) else None,
            cosine_p10=float(np.quantile(cosine, .1)) if len(cosine) else None,
            cosine_p90=float(np.quantile(cosine, .9)) if len(cosine) else None))
    return dict(split=split, checkpoint=checkpoint, n_patients=len(reference_ids),
        hidden_dimensions=reference_shared.shape[1], inputs=inputs,
        outcome_norms=[dict(outcome=name, median_norm=float(np.median(np.linalg.norm(d, axis=1))),
                           zero_fraction=float(np.mean(np.linalg.norm(d, axis=1) <= 1e-12)))
                       for name, d in deltas.items()],
        outcome_specific_energy_fraction=residual_energy/total_energy if total_energy else None,
        pairs=pairs,
        interpretation='Total readout correction includes global and weighted source terms. '
            'Outcome-specific energy subtracts the within-patient mean across exported outcomes. '
            'That common component does not isolate the global adapter. '
            'Similarity is a diagnostic, not evidence of predictive benefit.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--vectors-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Use a fresh output report')
    report = audit(args.vectors_root)
    report['evaluator_sha256'] = file_hash(__file__)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(f'Aggregate raw-vector audit: {args.output}')


if __name__ == '__main__':
    main()
