"""All-outcome correction geometry from saved tuning vectors, on CPU only.

This is independent of the running rehearsal. It exports aggregates only and
does not repeat encoder inference, metric bootstraps or predictive training.
"""
from __future__ import annotations

import argparse
from itertools import combinations
import json
from pathlib import Path
import zipfile

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import torch
from torch.nn import functional as F

from opera.diagnostics.router_mechanism import FrozenRouterHead
from opera.run.analyse_locked_predictions import file_hash
from opera.run.audit_frozen_router_mechanism import selected_inputs, load_vectors, prepare_population


def reconstruct(head, patient, fine, indices, horizon, batch_size=128):
    """Separate exact global/source-mixture vectors and replay the fitted head."""
    n, o, h = len(patient), len(indices), head.readout.hidden_size
    arrays = {name: np.empty((n, o, h), np.float32) for name in ('total', 'routed')}
    arrays['global'] = np.empty((n, h), np.float32)
    risk = np.empty((n, o), np.float32)
    error = 0.
    with torch.inference_mode():
        r = head.readout
        for start in range(0, n, batch_size):
            stop = min(start + batch_size, n)
            x, f = patient[start:stop], fine[start:stop]
            adapted, gates = r(x, f, return_weights=True)
            adapted, gates = adapted[:, indices], gates[:, indices]
            source = F.gelu(torch.einsum('bh,srh->bsr', x, r.down))
            source = torch.einsum('bsr,shr->bsh', source, r.up) * r.source_shrinkage[None, :, None]
            routed = r.residual_scale * torch.einsum('bos,bsh->boh', gates[..., 1:], source)
            general = (torch.zeros_like(x) if r.global_down is None else
                       r.residual_scale * F.linear(F.gelu(F.linear(x, r.global_down)), r.global_up))
            total = routed + general[:, None]
            replay = x[:, None] + total
            error = max(error, float((replay - adapted).abs().max()))
            if not torch.allclose(replay, adapted, atol=2e-5, rtol=2e-5):
                raise ValueError('Global/source decomposition does not reconstruct production readout')
            for name, value in [('total', total), ('routed', routed), ('global', general)]:
                arrays[name][start:stop] = value.numpy()
            flat = torch.einsum('boh,okh->bok', adapted, head.weight[indices]) + head.bias[indices][None]
            risk[start:stop] = head.cif(flat, indices, [horizon])[..., 0].numpy()
    return arrays, risk, error


def cosine_summary(a, b):
    norms = np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1)
    valid = norms > 1e-12
    values = np.sum(a[valid] * b[valid], axis=-1) / norms[valid]
    return dict(nonzero_pairs_n=int(valid.sum()),
                median_cosine=float(np.median(np.clip(values, -1, 1))) if len(values) else None)


def outcome_energy(values, eligible):
    """Within-patient outcome residual energy, not predictive importance."""
    values = np.asarray(values, np.float64)
    mass = eligible.sum(1)
    common = np.sum(values * eligible[..., None], axis=1) / np.maximum(mass[:, None], 1)
    total = float(np.sum(values ** 2 * eligible[..., None]))
    residual = float(np.sum((values - common[:, None]) ** 2 * eligible[..., None]))
    return dict(total_energy=total, outcome_specific_energy_fraction=residual / total if total else None,
                patients_with_two_or_more_outcomes=int((mass >= 2).sum()))


def summarise(arrays, risk, eligible, names, *, minimum_common=20):
    eligible = np.asarray(eligible, bool)
    if (len(names) < 2 or len(set(names)) != len(names) or risk.shape != eligible.shape
            or arrays['total'].shape[:2] != eligible.shape
            or arrays['routed'].shape != arrays['total'].shape
            or arrays['global'].shape != (len(risk), arrays['total'].shape[2])
            or len(names) != risk.shape[1]
            or minimum_common < 2 or not np.isfinite(risk).all()
            or any(not np.isfinite(value).all() for value in arrays.values())):
        raise ValueError('Need finite aligned global/routed/total vectors, risks and eligibility')
    if not np.allclose(arrays['total'], arrays['routed'] + arrays['global'][:, None], atol=2e-5, rtol=2e-5):
        raise ValueError('Total corrections must equal global plus routed mixture')
    outcomes, pairs = [], []
    for o, name in enumerate(names):
        keep = eligible[:, o]
        row = dict(outcome=name, n_patients=int(keep.sum()))
        for component in ('total', 'routed', 'global'):
            vectors = arrays[component][keep] if component == 'global' else arrays[component][keep, o]
            row[component + '_median_norm'] = float(np.median(np.linalg.norm(vectors, axis=-1))) if keep.any() else None
        row['global_routed_median_cosine'] = cosine_summary(arrays['global'][keep], arrays['routed'][keep, o])['median_cosine']
        outcomes.append(row)
    for left, right in combinations(range(len(names)), 2):
        keep = eligible[:, left] & eligible[:, right]
        row = dict(left=names[left], right=names[right], n_common=int(keep.sum()),
                   status='estimated' if keep.sum() >= minimum_common else 'insufficient_common_patients',
                   predicted_risk_spearman=None)
        if row['status'] == 'estimated':
            a, b = risk[keep, left], risk[keep, right]
            row['predicted_risk_spearman'] = (float(spearmanr(a, b).statistic)
                if np.ptp(a) > 0 and np.ptp(b) > 0 else None)
            for component in ('total', 'routed'):
                stats = cosine_summary(arrays[component][keep, left], arrays[component][keep, right])
                row.update({component + '_' + key: value for key, value in stats.items()})
        pairs.append(row)
    # These energies are not additive importance shares: the cross term can
    # be negative, so global_energy / total_energy can exceed one.
    global_energy = float(np.sum(arrays['global'].astype(float) ** 2 * eligible.sum(1)[:, None]))
    routed_energy = outcome_energy(arrays['routed'], eligible)
    total_energy = outcome_energy(arrays['total'], eligible)
    cross = float(2 * np.sum(arrays['global'][:, None].astype(float) * arrays['routed'] * eligible[..., None]))
    summary = dict(total=total_energy, routed_without_global=routed_energy,
                   global_energy=global_energy, global_routed_cross_term=cross,
                   energy_identity_error=abs(total_energy['total_energy'] - global_energy - routed_energy['total_energy'] - cross))
    return summary, pd.DataFrame(outcomes), pd.DataFrame(pairs)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--analysis-root', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--vectors', type=Path)
    parser.add_argument('--horizon', type=int, default=365)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--minimum-common', type=int, default=20)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    if min(args.threads, args.batch_size, args.horizon) < 1 or args.minimum_common < 2:
        parser.error('Need positive threads/batch size/horizon and minimum-common >=2')
    root, output = args.analysis_root.resolve(), args.output_dir.resolve()
    if output.exists():
        raise FileExistsError('Use a fresh geometry audit output directory')
    analysis, spec, fit = selected_inputs(root, args.seed)
    torch.set_num_threads(args.threads)
    head = FrozenRouterHead.from_checkpoint(Path(fit['checkpoint'])).eval()
    ids, vectors, vector_path, marker_path = load_vectors(root, fit, args.vectors)
    baseline, patient, fine, eligible, outcomes, _, row_index = prepare_population(
        analysis, spec, head, ids, vectors, outcomes=['all'], horizons=[args.horizon])
    print(f'Replaying {len(outcomes)} outcomes for {len(patient):,} tuning patients on CPU', flush=True)
    indices = torch.tensor([head.outcomes.index(name) for name in outcomes])
    arrays, risk, algebra_error = reconstruct(head, patient, fine, indices, args.horizon, args.batch_size)
    replay, original = risk[row_index[:2]], baseline.probability.to_numpy(float)
    close = np.isclose(replay, original, atol=1e-5, rtol=1e-4)
    output.mkdir(parents=True)
    report = dict(split='tuning', checkpoint=fit['checkpoint'], checkpoint_sha256=fit['checkpoint_sha256'],
                  source_sha256=file_hash(__file__), vector_input=str(vector_path), vector_marker=str(marker_path),
                  horizon_days=args.horizon, n_patients=len(patient), n_outcomes=len(outcomes),
                  reconstruction=dict(passed=bool(close.all()), mismatched_rows=int((~close).sum()),
                      max_probability_error=float(abs(replay - original).max()), max_vector_algebra_error=algebra_error),
                  interpretation='Geometry is not predictive importance. Routed means the full weighted own-and-borrowed source mixture, excluding the exact global adapter. '
                  'Risk Spearman compares model-predicted patient rankings, not observed outcomes or independently fitted embeddings. '
                  'Energy includes a global/routed cross term; norm or energy ratios are not additive performance shares.')
    if not close.all():
        report['status'] = 'failed_reconstruction'
        (output / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
        raise ValueError('Geometry replay does not match selected saved tuning probabilities')
    print('Saved probabilities reproduced. Comparing all eligible outcome pairs.', flush=True)
    energy, outcome_table, pairs = summarise(arrays, risk, eligible.numpy(), outcomes, minimum_common=args.minimum_common)
    report.update(status='complete', energy=energy)
    outcome_table.to_csv(output / 'outcome_geometry.csv', index=False)
    pairs.to_csv(output / 'outcome_pairs.csv', index=False)
    # Ranking is a transparent figure-selection aid; the full pair table is
    # retained. Constant risk predictions never count as a dissimilar pair.
    (pairs.dropna(subset=['predicted_risk_spearman'])
        .sort_values(['predicted_risk_spearman', 'left', 'right']).head(20)
        .to_csv(output / 'different_risk_patterns.csv', index=False))
    (output / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    with zipfile.ZipFile(output / 'router_geometry_review.zip', 'w', zipfile.ZIP_DEFLATED) as archive:
        for name in ('report.json', 'outcome_geometry.csv', 'outcome_pairs.csv', 'different_risk_patterns.csv'):
            archive.write(output / name, name)
    print(f'Complete: {output / "router_geometry_review.zip"}', flush=True)


if __name__ == '__main__':
    main()
