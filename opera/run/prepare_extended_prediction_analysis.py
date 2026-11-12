"""Prepare separate national comparisons, reusing frozen baseline predictions.

Does not select an encoder, alter the primary OPERA alias, or modify a running
analysis. Pending optional predictions are reported; selection candidates are required.
"""
import argparse
import hashlib
import json
from pathlib import Path

import yaml

from opera.run.paper_prediction_queue import bundle_runs, canonical_models


def prepare(source, root, output, split):
    source, root, output = map(lambda p: Path(p).resolve(), (source, root, output))
    if output.exists():
        raise FileExistsError(output)
    config = yaml.safe_load(source.read_text())
    baselines = {'J_L', 'J_OA_OWN_L', 'R_AB_L', 'R_PF_OFF_L',
                 'S_L', 'D_L', 'O_L', 'X', 'XP', 'XR', 'LR'}
    artifacts = [a for a in config['artifacts']
                 if a.get('split') == split and a.get('model') in baselines
                 and int(a.get('seed', 42)) == 42]
    present = {a['model'] for a in artifacts}
    required = {'J_L', 'J_OA_OWN_L', 'X', 'XP'}
    if required - present:
        raise ValueError(f'Missing baseline models: {sorted(required - present)}')
    contracts = {a['data_contract'] for a in artifacts}
    if len(contracts) != 1:
        raise ValueError('Expected one baseline data contract')
    missing = []
    national = set()
    for name in ('direct_fits', 'adapted_fits'):
        bundle, rows = bundle_runs(root / name)
        for row, _, resolved in rows:
            if row['seed'] != 42 or row['phase'] != 'external':
                continue
            directory = root / 'predictions' / row['run_directory'] / split
            prediction = directory / 'direct_cr_predictions.parquet'
            if not (directory / 'evaluation_complete.json').is_file() or not prediction.is_file():
                missing.append(dict(models=canonical_models(row, resolved, bundle),
                                    prediction_directory=str(directory)))
                continue
            for model in canonical_models(row, resolved, bundle):
                if model in national:
                    raise ValueError(f'Duplicate model: {model}')
                national.add(model)
                artifacts.append(dict(path=str(prediction), format='direct_cif',
                    model=model, seed=42, split=split,
                    data_contract=next(iter(contracts)), cohort_from_metadata=True,
                    include_horizons_days=config.get('analysis_horizons_days', [30,60,90,180,365,730])))
    missing_candidates = {'R_E', 'R_EA'} - national
    if missing_candidates:
        raise ValueError(f'Missing selection candidate evaluations: {sorted(missing_candidates)}')
    comparisons = dict(config.get('extra_contrasts') or {})
    for model in sorted(national | {'J_OA_OWN_L'}):
        for reference in sorted(present & {'X', 'XP', 'XR', 'J_L', 'J_OA_OWN_L'}):
            if model != reference:
                comparisons[f'EXT_{model}_VS_{reference}'] = f'gain({model},{reference})'
    comparisons['EXT_ADAPTATION'] = 'gain(R_EA,R_E)'
    comparisons['EXT_ARCH_L'] = 'gain(J_OA_OWN_L,J_L)'
    for origin in ('E', 'EA'):
        if f'J_{origin}' in national:
            comparisons[f'EXT_ARCH_{origin}'] = f'gain(R_{origin},J_{origin})'
        if f'R_PF_OFF_{origin}' in national:
            comparisons[f'EXT_READING_{origin}'] = f'gain(R_{origin},R_PF_OFF_{origin})'
            if f'R_AB_{origin}' in national:
                comparisons[f'EXT_ROUTING_{origin}'] = f'gain(R_AB_{origin},R_PF_OFF_{origin})'
    config.update(artifacts=artifacts, extra_contrasts=comparisons,
                  patient_gain_contrasts=list(comparisons), design_search=False,
                  route_weights=None, clinical_threshold_artifacts=[],
                  identity_date_sensitivity=False, stratified_reanalysis={},
                  patient_groups={}, operating_points={})
    output.mkdir(parents=True)
    (output / 'analysis.yaml').write_text(yaml.safe_dump(config, sort_keys=False))
    report = dict(split=split, models=sorted(present | national),
                  registry_augmented_present='XR' in present,
                  pending_evaluations=missing,
                  source_analysis=str(source), source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                  primary_alias=config.get('opera_model'),
                  scope='Fixed-arm comparisons; no primary encoder selection; clinical refresh remains separate')
    (output / 'preparation_report.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    return config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('source-analysis', 'national-root', 'output-dir'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--split', choices=('tuning', 'held_out'), required=True)
    args = parser.parse_args()
    prepare(args.source_analysis, args.national_root, args.output_dir, args.split)


if __name__ == '__main__':
    main()
