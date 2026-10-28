"""Prepare an isolated, matched SEP-target pretraining and joint-finetuning pilot."""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import yaml
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf


def load_config(path):
    path = Path(path).resolve()
    OmegaConf.register_new_resolver('version', lambda: 'sep_sensitivity', replace=True)
    raw = OmegaConf.load(path)
    if 'defaults' in raw:
        with initialize_config_dir(config_dir=str(path.parent), version_base='1.2'):
            raw = compose(config_name=path.stem)
    # Saved Hydra output paths are runtime-specific, never scientific inputs.
    raw.pop('hydra', None)
    raw['run_id'] = 'sep_sensitivity'
    return OmegaConf.to_container(raw, resolve=True, throw_on_missing=True)


def prepare(pretrain, joint, output, seeds=(42, 43, 44)):
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError('Use distinct fine-tuning seeds')
    if not pretrain['paths']['dataset_class'].endswith('.ARPretrainDataset'):
        raise ValueError('This sensitivity requires code-only ARPretrainDataset')
    if set(pretrain['training'].get('ignore_target_tokens', [])) != {'[SEP]'}:
        raise ValueError('Reference pretraining must exclude exactly [SEP] targets')
    if pretrain['paths'].get('initialization_ckpt'):
        raise ValueError('Supply the original from-scratch pretraining config, not DAPT')
    risk = joint['competing_risk']
    if risk.get('readout_mode') != 'shared' or joint['model'].get('freeze_encoder', False):
        raise ValueError('Supply the trainable shared joint model, not a router')
    if joint.get('source_opera_ckpt'):
        raise ValueError('Joint sensitivity must not initialize from another fine-tuned model')
    if int(pretrain.get('hardware', {}).get('num_devices', 1)) != 1:
        raise ValueError('This launcher expects a single-GPU reference configuration')
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    configs = output / 'configs'
    configs.mkdir()
    def write(name, value):
        (configs / f'{name}.yaml').write_text(yaml.safe_dump(value, sort_keys=False))
    write('reference_pretrain', pretrain)
    write('reference_joint', joint)
    for arm, ignored in [('exclude_sep', ['[SEP]']), ('include_sep', [])]:
        config = copy.deepcopy(pretrain)
        config['paths']['ckpt_path'] = None
        config['training'].update(ignore_target_tokens=ignored, early_stopping_patience=None,
                                  auto_resume=True)
        # Final-epoch checkpoints impose the same budget; val/loss differs by arm.
        write(f'pretrain_{arm}', config)
        for seed in seeds:
            config = copy.deepcopy(joint)
            config['seed'] = seed
            config['dapt_ckpt'] = str(output / arm / 'pretrain/last.ckpt')
            config['paths']['resume_ckpt'] = None
            config['training']['auto_resume'] = True
            sampling = config['training'].get('batch_sampling', {})
            if isinstance(sampling, dict) and 'seed' in sampling:
                sampling['seed'] = seed
            write(f'joint_{arm}_s{seed}', config)
    manifest = dict(seeds=list(seeds), pretraining_seed=pretrain['seed'],
                    epochs=pretrain['training']['epochs'], checkpoint='last.ckpt',
                    purpose='Exploratory tuning-only SEP-target sensitivity; not a primary-model replacement',
                    config_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                   for p in configs.glob('*.yaml')})
    (output / 'sensitivity.json').write_text(json.dumps(manifest, indent=2))
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--env-file', type=Path, default=Path('.env.rehearsal'))
    parser.add_argument('--joint-config', type=Path, required=True)
    parser.add_argument('--pretrain-config', type=Path,
                        help='Actual saved/resolved production pretraining config')
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    from opera.run.paper_experiments import load_environment
    os.environ.update(load_environment(args.env_file))
    joint = load_config(args.joint_config)
    source = args.pretrain_config
    if source is None:
        checkpoint = Path(joint['dapt_ckpt'])
        candidates = [parent / folder / 'config.yaml' for parent in checkpoint.parents
                      for folder in ('hydra', '.hydra')]
        source = next((p for p in candidates if p.is_file()), None)
        if source is None:
            parser.error('Cannot find the original pretraining Hydra config; supply --pretrain-config explicitly')
    pretrain = load_config(source)
    for key in ('train_split', 'val_split', 'vocab'):
        if not Path(pretrain['paths'][key]).is_file():
            parser.error(f"Missing production pretraining input: paths.{key}")
    manifest = prepare(pretrain, joint, args.output_dir)
    manifest['patient_metadata'] = os.environ.get('OPERA_PAPER_PATIENT_METADATA')
    manifest['source_configs'] = {str(p.resolve()): hashlib.sha256(p.read_bytes()).hexdigest()
                                  for p in (source, args.joint_config)}
    (args.output_dir / 'sensitivity.json').write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    main()
