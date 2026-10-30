"""Reconstruct selected tuning readout vectors from saved encoder states on CPU.

No encoder pass, training or source-removal inference. Individual arrays stay
on the server and must not be added to the aggregate review ZIP.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from opera.diagnostics.router_mechanism import FrozenRouterHead
from opera.run.audit_frozen_router_mechanism import selected_inputs, load_vectors, prepare_population


def replay_vectors(head, patient, fine, indices, horizon, batch_size=128):
    adapted = np.empty((len(patient), len(indices), patient.shape[1]), np.float32)
    risks = np.empty((len(patient), len(indices)), np.float32)
    with torch.inference_mode():
        for start in range(0, len(patient), batch_size):
            stop = min(start + batch_size, len(patient))
            values = head.readout(patient[start:stop], fine[start:stop])[:, indices]
            adapted[start:stop] = values.numpy()
            flat = torch.einsum('boh,okh->bok', values, head.weight[indices]) + head.bias[indices][None]
            risks[start:stop] = head.cif(flat, indices, [horizon])[..., 0].numpy()
    return adapted, risks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--analysis-root', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--outcomes', nargs='+', required=True)
    parser.add_argument('--horizon', type=int, default=365)
    parser.add_argument('--threads', type=int, default=4)
    args = parser.parse_args()
    if args.threads < 1 or args.horizon < 1:
        parser.error('Need positive thread count and horizon')
    if args.output_dir.exists():
        raise FileExistsError('Use a fresh vector directory')
    torch.set_num_threads(args.threads)
    root = args.analysis_root.resolve()
    analysis, specification, fit = selected_inputs(root, 42)
    head = FrozenRouterHead.from_checkpoint(Path(fit['checkpoint'])).eval()
    ids, vectors, _, _ = load_vectors(root, fit)
    baseline, patient, fine, _, outcomes, _, row_index = prepare_population(
        analysis, specification, head, ids, vectors, outcomes=args.outcomes, horizons=[args.horizon])
    indices = torch.tensor([head.outcomes.index(name) for name in outcomes])
    adapted, risks = replay_vectors(head, patient, fine, indices, args.horizon)
    original = baseline.probability.to_numpy(float)
    replay = risks[row_index[:2]]
    if not np.allclose(replay, original, atol=1e-5, rtol=1e-4):
        raise ValueError('Reconstructed readout does not reproduce selected saved probabilities')
    ids = np.asarray(sorted(baseline.subject_id.unique()), dtype=str)
    args.output_dir.mkdir(parents=True)
    files = {}
    for index, outcome in enumerate(outcomes):
        folder = args.output_dir / outcome
        folder.mkdir()
        path = folder / 'routed_patient_vectors_INTERNAL.npz'
        np.savez_compressed(path, subject_ids=ids, shared=patient.numpy(), adapted=adapted[:, index])
        path.with_suffix('.json').write_text(json.dumps(dict(checkpoint=fit['checkpoint'],
            checkpoint_sha256=fit['checkpoint_sha256'], split='tuning', target_outcome=outcome,
            representation='shared pooled encoder and target-specific post-router readout',
            n_patients=len(ids), saved_probability_replay_passed=True), indent=2))
        files[outcome] = str(path.resolve())
    (args.output_dir / 'map_vectors.json').write_text(json.dumps(files, indent=2))
    print(f'Reconstructed {len(outcomes)} map outcomes for {len(ids):,} patients. '
          'INTERNAL vectors remain on the server.', flush=True)


if __name__ == '__main__':
    main()
