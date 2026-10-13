"""Verify saved unanchored settings and gradient connectivity using a CPU probe."""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
from pathlib import Path

import torch
import yaml

from bonsai.functional.checkpointing import MODEL_INIT_CONFIG_KEY
from opera.modules.networks.competing_risk import PiecewiseExponentialCompetingRiskLoss
from opera.modules.networks.source_anchored_readout import SourceAnchoredReadout


def gradient_probe(readout, head_weight, head_bias, risk, cross):
    """One synthetic target loss; checks connectivity, not predictive performance."""
    disease = int(readout.source_train_patients[:readout.n_disease].argmax())
    target = int(readout.source_train_events[readout.n_disease:].argmax())
    generator = torch.Generator().manual_seed(42)
    patient = torch.randn(4, readout.hidden_size, generator=generator)
    adapted = readout(patient, torch.full((4,), disease, dtype=torch.long))
    hazards = (torch.einsum('boh,okh->bok', adapted, head_weight) + head_bias[None]).reshape(
        4, readout.n_outcome, 2, -1)
    objective = PiecewiseExponentialCompetingRiskLoss(
        readout.outcome_names, risk['interval_boundaries_days'],
        no_competing_outcomes=risk.get('no_competing_outcomes', []),
        time_scale_days=float(risk.get('time_scale_days', 365.25)),
        objective='horizon_bce', cross_outcome_config=cross, weighter='uniform',
        smoothness_weight=float(risk.get('smoothness_weight', 0)),
    )
    name = readout.outcome_names[target]
    survival = {name: {'times': torch.tensor([10., 365., 365., 20.]),
                       'events': torch.tensor([1, 0, 0, 1])}}
    objective(hazards, survival)[0].backward()
    result = {}
    for bank, indices, home in (
        ('disease', range(readout.n_disease), disease),
        ('outcome', range(readout.n_disease, readout.n_sources), readout.n_disease + target),
    ):
        borrowed = [index for index in indices if index != home and readout.source_shrinkage[index] > 0]
        for parameter in ('down', 'up'):
            grad = getattr(readout, parameter).grad
            result[f'{bank}_{parameter}_home_gradient_rms'] = float(grad[home].square().mean().sqrt())
            result[f'{bank}_{parameter}_borrowed_gradient_rms'] = (
                float(grad[borrowed].square().mean().sqrt()) if borrowed else None)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--checkpoint', choices=('best', 'last'), default='best')
    args = parser.parse_args()
    config = yaml.safe_load((args.run / 'hydra/config.yaml').read_text(encoding='utf-8'))
    checkpoint = torch.load(args.run / f'{args.checkpoint}.ckpt', map_location='cpu', weights_only=False)
    init = checkpoint['hyper_parameters'].get(MODEL_INIT_CONFIG_KEY, {})
    risk = init.get('competing_risk_config', {})
    if risk.get('readout_mode') != 'unanchored_adapters':
        raise ValueError(f"Checkpoint did not save unanchored_adapters: {risk.get('readout_mode')}")
    if config['competing_risk'] != risk:
        raise ValueError('Saved config and checkpoint competing-risk settings differ; inspect before interpreting')
    if risk.get('adapter_source_mode', 'both') != 'both':
        raise ValueError('This verification expects both source banks')
    provenance = json.loads((args.run / 'router_code_provenance_INTERNAL.json').read_text(encoding='utf-8'))
    source_path = Path(inspect.getfile(SourceAnchoredReadout))
    if hashlib.sha256(source_path.read_bytes()).hexdigest() != provenance['loaded_class_file_sha256']:
        raise ValueError('Local readout source differs from the source recorded for this run')
    state = checkpoint['state_dict']
    source_state = {key.split('source_adapter_readout.', 1)[1]: value
                    for key, value in state.items() if 'source_adapter_readout.' in key}
    hidden_size = source_state['down'].shape[-1]
    names = sorted(config['outcomes'])
    def build(anchored):
        readout = SourceAnchoredReadout(
            hidden_size, risk['fine_names'], names, anchored=anchored,
            bottleneck=int(risk.get('adapter_bottleneck', 8)),
            residual_scale=float(risk.get('adapter_residual_scale', .1)),
            home_floor=float(risk.get('adapter_home_floor', .1)),
            query_mode=risk.get('adapter_query_mode', 'additive'),
            query_hidden_dim=int(risk.get('adapter_query_hidden_dim', 128)), source_mode='both',
        )
        readout.load_state_dict(source_state, strict=True)
        return readout
    weights = [value for key, value in state.items() if key.endswith('competing_risk_head.weight')]
    biases = [value for key, value in state.items() if key.endswith('competing_risk_head.bias')]
    if len(weights) != 1 or len(biases) != 1:
        raise ValueError('Cannot uniquely identify the saved linear risk head')
    weight = weights[0].float().reshape(len(names), -1, hidden_size)
    bias = biases[0].float().reshape(len(names), -1)
    cross = init.get('cross_outcome_config', config['cross_outcome'])
    live = gradient_probe(build(False), weight, bias, risk, cross)
    control = gradient_probe(build(True), weight, bias, risk, cross)
    passed = all(live[key] is not None and live[key] > 0
                 and (control[key] == 0 if 'borrowed_gradient' in key else control[key] > 0)
                 for key in live)
    print(json.dumps({'run': str(args.run), 'checkpoint_epoch': checkpoint.get('epoch'),
                      'checkpoint_readout_mode': risk['readout_mode'], 'anchored': False,
                      'probe': 'Synthetic CPU horizon-BCE connectivity test; no optimizer step or patient data',
                      'unanchored_gradients': live, 'anchored_negative_control': control,
                      'borrowed_connectivity_verified': passed}, indent=2))
    if not passed:
        raise RuntimeError('Borrowed connectivity probe failed; inspect the reported gradients')


if __name__ == '__main__':
    main()
