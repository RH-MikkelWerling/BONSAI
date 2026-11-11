"""Prepare direct and pending adapted fits from frozen local scientific configs."""
import argparse
import copy
import json
from pathlib import Path

from dotenv import dotenv_values
from opera.run.prepare_paper_run_bundle import external_sequences, write_bundle
from opera.run.run_paper_queue import load_run_config

SOURCES = {'J':'J_all_diseases_all_outcomes', 'R':'fix_J_OA_OWN',
           'AB':'fix_R_AB', 'OFF':'fix_R_PF_OFF',
           'NODIS':'fix_J_OA_OWN_NODIS', 'NOOUT':'fix_J_OA_OWN_NOOUT'}


def prepare(local_bundle, national_root, sequences, direct, adapted, encoder_lr):
    local_bundle = Path(local_bundle).resolve()
    manifest = json.loads((local_bundle/'bundle.json').read_text())
    rows = {r['name']: r for r in manifest['runs']}
    configs = {}
    for mode, source in SOURCES.items():
        name = f'seed42/L/{source}'
        if name not in rows:
            raise ValueError(f'Frozen bundle lacks {name}; supply the final local bundle')
        _, config, _ = load_run_config(rows[name], local_bundle)
        configs[mode] = config
    entries = []
    for origin, checkpoint, modes in [('E', direct, ['J','R','AB','OFF']),
                                     ('EA', adapted, list(SOURCES))]:
        for mode in modes:
            config = external_sequences(copy.deepcopy(configs[mode]), sequences)
            config['dapt_ckpt'] = str(Path(checkpoint).resolve())
            config['paper_run'] = {'origin':origin, 'seed':42}
            if origin == 'EA':
                config['paper_run']['direct_external_checkpoint'] = str(Path(direct).resolve())
            config.setdefault('training', {})['encoder_lr_multiplier'] = encoder_lr
            entries.append(dict(name=f'seed42/{origin}/{mode}', phase='external',
                origin=origin, seed=42, condition={}, config=config))
    joint = next(e for e in entries if e['name']=='seed42/EA/J')
    for percent in [10,25,50]:
        config = copy.deepcopy(joint['config'])
        config['training'].update(train_patient_fraction=percent/100, train_fraction_seed=42)
        entries.append(dict(name=f'seed42/EA/J_frac{percent}', phase='external', origin='EA',
                            seed=42, condition={}, config=config))
    provenance = {'frozen_local_bundle':str(local_bundle),
                  'source_scientific_hashes':{m:rows[f'seed42/L/{s}']['scientific_hash'] for m,s in SOURCES.items()},
                  'note':'Direct fits ready; adapted fits require completed audited DAPT. No evaluation launched.'}
    root = Path(national_root)
    for origin, name in [('E','direct_fits'),('EA','adapted_fits_pending')]:
        report = write_bundle([e for e in entries if e['origin']==origin], root/name,
                              provenance=provenance)
        print(name, len(report['runs']), 'fits', flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--env-file', type=Path, required=True)
    parser.add_argument('--local-bundle', type=Path)
    args=parser.parse_args()
    env=dotenv_values(args.env_file)
    # Original .env retains the local study root; national.env points elsewhere.
    local_env=dotenv_values('.env')
    root=Path(env['OPERA_PAPER_ROOT']).resolve()
    bundle=args.local_bundle
    if bundle is None:
        candidates=list((Path(local_env['OPERA_PAPER_ROOT'])/'bundles').glob('stage*/bundle.json'))
        candidates.sort(key=lambda p:int(p.parent.name.removeprefix('stage')), reverse=True)
        required={f'seed42/L/{s}' for s in SOURCES.values()}
        for path in candidates:
            data=json.loads(path.read_text())
            if required <= {r['name'] for r in data['runs']}:
                bundle=path.parent
                break
        if bundle is None:
            raise ValueError('Cannot locate final local bundle; supply --local-bundle')
    lr=env.get('OPERA_PAPER_EXTERNAL_ENCODER_LR')
    if not lr:
        raise ValueError('Set OPERA_PAPER_EXTERNAL_ENCODER_LR to the tuning-locked multiplier first')
    # OPERA and learned routing have separate aliases in the locked release.
    # ROUTED_MODEL=R_AB_L is the secondary comparator, not the primary OPERA.
    saved = Path('data/paper_256_v3/analysis/local_test_20261109_four_allocations/overnight.env')
    study_env = dotenv_values(saved) if saved.is_file() else local_env
    primary = study_env.get('OPERA_PAPER_OPERA_MODEL')
    if primary != 'J_OA_OWN_L':
        raise ValueError(f'This handoff requires OPERA_PAPER_OPERA_MODEL=J_OA_OWN_L; '
                         f'found {primary!r} in {saved if saved.is_file() else ".env"}')
    prepare(bundle, root, Path(env['OPERA_PAPER_EXTERNAL_SEQUENCE_DIR']),
            root/'external_direct/encoder.ckpt', root/'external_dapt/version_0/best.ckpt', float(lr))


if __name__=='__main__':
    main()
