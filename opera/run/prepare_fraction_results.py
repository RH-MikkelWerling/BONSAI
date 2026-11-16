"""Prepare a focused six-horizon label-efficiency/control analysis from saved fits.

No training, encoder selection or change to the original numerical analysis.
Missing local fraction arms are reported rather than replaced by different fits.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path

import yaml


def read(path):
    return yaml.load(Path(path).read_text(), Loader=getattr(yaml, 'CSafeLoader', yaml.SafeLoader))


def prepare(root, output, split='held_out', local_bundle=None):
    from opera.run.analyse_locked_predictions import models_for
    root, output = Path(root).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError('Use a fresh focused-analysis directory')
    source = root / f'extended_{"test" if split == "held_out" else "tuning"}_comparison/analysis.yaml'
    config = read(source)
    artifacts = [copy.deepcopy(a) for a in config['artifacts'] if a['model'] in {'J_L', 'J_EA', 'S_L'}
                 and a['split'] == split and int(a['seed']) == 42]
    contracts = {a['data_contract'] for a in artifacts}
    if len(contracts) != 1 or not {'J_L', 'J_EA', 'S_L'} <= {a['model'] for a in artifacts}:
        raise ValueError('Need the registered local joint, adapted joint and specialist references')
    contract, = contracts
    direct = read(root / 'direct_fits/bundle.json')
    local_bundle = Path(local_bundle or direct['provenance']['frozen_local_bundle']).resolve()
    local_root = local_bundle.parent.parent
    pending, discovered = [], []
    for bundle in (root / 'adapted_fits', local_bundle):
        for row in read(bundle / 'bundle.json')['runs']:
            if row.get('phase') not in {'data_efficiency', 'pretraining_control'} or row.get('seed') != 42:
                continue
            origin = row['origin']
            frozen_path = bundle / row['config_path']
            frozen_hash = hashlib.sha256(frozen_path.read_bytes()).hexdigest()
            if frozen_hash != row['config_sha256']:
                raise ValueError(f'Frozen configuration changed: {frozen_path}')
            frozen = read(frozen_path)
            if row['phase'] == 'data_efficiency':
                fraction = frozen['training']['train_patient_fraction']
                model = f'J_{origin}_F{round(100 * fraction)}'
            else:
                model, fraction = f'J_{origin}', 1
            if origin not in {'L', 'EA', 'RI', 'FZ'}:
                continue
            base = root if bundle == root / 'adapted_fits' else local_root
            directory = base / 'predictions' / row['run_directory'] / split
            identity = {'model': model, 'fraction': fraction, 'seed': 42,
                        'name': row['name'], 'prediction_directory': str(directory),
                        'scientific_hash': row.get('scientific_hash'),
                        'config_path': str(frozen_path), 'config_sha256': frozen_hash,
                        'train_fraction_seed': frozen.get('training', {}).get('train_fraction_seed')}
            discovered.append(identity)
            if not (directory / 'evaluation_complete.json').is_file() or not (directory / 'direct_cr_predictions.parquet').is_file():
                pending.append(identity); continue
            marker = read(directory / 'evaluation_complete.json')
            if (marker.get('inputs', {}).get('config_sha256') != frozen_hash
                    or marker.get('inputs', {}).get('split') != split):
                raise ValueError(f'Evaluation receipt does not match frozen fraction/control fit: {directory}')
            artifacts.append({'model': model, 'seed': 42, 'split': split, 'format': 'direct_cif',
                              'cohort_from_metadata': True, 'data_contract': contract,
                              'path': str(directory / 'direct_cr_predictions.parquet'),
                              'include_horizons_days': [30, 60, 90, 180, 365, 730]})
    models = {a['model'] for a in artifacts}
    if not {f'J_EA_F{n}' for n in (10, 25, 50)} <= models:
        raise ValueError('Evaluate all three adapted fraction arms first; pending: ' + json.dumps(pending))
    manifest = read(config['contrast_manifest'])
    families = {k: manifest['essential_contrasts'][k] for k in
                ('data_efficiency', 'label_savings', 'pretraining_control')}
    missing = {}
    for family, comparisons in families.items():
        keep = {}
        for name, expression in comparisons.items():
            absent = models_for({name: expression}) - models
            if absent:
                missing[name] = sorted(absent)
            else:
                keep[name] = expression
        families[family] = keep
    manifest['essential_contrasts'] = families
    output.mkdir(parents=True)
    contrast_path = output / 'focused_contrasts.yaml'
    contrast_path.write_text(yaml.safe_dump(manifest, sort_keys=False))
    learning_models = sorted(m for m in models if m in {'J_L', 'J_EA'} or '_F' in m and m.startswith(('J_L_', 'J_EA_')))
    config.update(artifacts=artifacts, contrast_manifest=str(contrast_path), extra_contrasts={},
                  opera_model=None, patient_gain_contrasts=[], strategy_candidates=['S_L', 'D_L', 'O_L'],
                  panel_estimate_groups={'label_efficiency_common_roster': learning_models},
                  clinical_threshold_artifacts=[], route_weights=None, identity_date_sensitivity=False,
                  stratified_reanalysis={}, patient_groups={}, operating_points={}, design_search=False)
    (output / 'analysis.yaml').write_text(yaml.safe_dump(config, sort_keys=False))
    report = {'scope': 'Focused fraction/control analysis; no model fitting or primary model selection',
              'source': str(source), 'split': split, 'models': sorted(models), 'discovered_fits': discovered,
              'pending_evaluations': pending, 'missing_contrasts': missing,
              'label_savings': 'Report paired differences and intervals; crossing zero does not establish noninferiority.'}
    (output / 'preparation_report.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    return config


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--national-root', type=Path, default=Path('data/paper_256_v3_national_20261111'))
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--split', choices=('held_out', 'tuning'), default='held_out')
    parser.add_argument('--local-bundle', type=Path)
    args = parser.parse_args()
    prepare(args.national_root, args.output_dir, args.split, args.local_bundle)
