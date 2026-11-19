"""Prepare the explicitly requested matched epoch20 joint fine-tuning control.

Preparation only: source config and upstream checkpoints are verified, no model
training or evaluation launched by this module. Existing paper inputs stay frozen.
"""
import argparse
import copy
import json
from pathlib import Path
import yaml

EPOCH10_SHA = 'b0e0bf535ec4c4e928f67e7eee1fe89e04158df2a2aa2f89af0bf658cdded2cf'
EPOCH20_SHA = '6afe3c221e974cfe81dbccbc93e63c10627ea5bbecf950a1b351af8b0e0f89f0'


def clone_joint_config(source, checkpoint):
    if source.get('seed') != 42 or source.get('supervision_factorial', {}).get('regime') != 'J':
        raise ValueError('Expected the frozen seed42 full joint recipe')
    if source.get('model', {}).get('freeze_encoder', False) or source.get('encoder_initialization') == 'random':
        raise ValueError('This control fine-tunes pretrained encoder and head; no frozen/random initialization')
    if source.get('training', {}).get('train_patient_fraction') not in (None, 1, 1.):
        raise ValueError('Expected full-training-data joint reference')
    if source.get('paths', {}).get('resume_ckpt'):
        raise ValueError('Source has an explicit fine-tuning resume checkpoint; do not inherit it')
    result = copy.deepcopy(source)
    result['dapt_ckpt'] = str(Path(checkpoint).resolve())
    return result


def checkpoint_record(path):
    import gc
    import torch
    from opera.run.run_paper_queue import file_hash
    data = torch.load(path, map_location='cpu', mmap=True, weights_only=False)
    record = dict(path=str(Path(path).resolve()), sha256=file_hash(path),
                  completed_epochs=int(data.get('epoch', -1)) + 1, global_step=data.get('global_step'))
    del data
    gc.collect()
    return record


def reference_selection(run):
    from bonsai.functional.checkpointing import verified_training_selection
    return verified_training_selection(run)


def prepare(source_bundle, output, checkpoint):
    from opera.run.run_paper_queue import file_hash
    from opera.run.prepare_paper_run_bundle import write_bundle
    source_bundle, output, checkpoint = map(lambda p: Path(p).resolve(), [source_bundle, output, checkpoint])
    old = json.loads((source_bundle / 'bundle.json').read_text())
    row, = [r for r in old['runs'] if r['name'] == 'seed42/L/J_all_diseases_all_outcomes']
    path = source_bundle / row['config_path']
    if file_hash(path) != row['config_sha256']:
        raise ValueError('Frozen reference config hash changed')
    source = yaml.safe_load(path.read_text())
    first, last = checkpoint_record(Path(source['dapt_ckpt'])), checkpoint_record(checkpoint)
    if first['sha256'] != EPOCH10_SHA or first['completed_epochs'] != 10:
        raise ValueError('Reference encoder is not the verified current-release epoch10')
    if last['sha256'] != EPOCH20_SHA or last['completed_epochs'] != 20:
        raise ValueError('Proposed encoder is not the verified current-release epoch20')
    run = source_bundle.parent.parent / 'neural_runs' / row['run_directory']
    selection = reference_selection(run)
    new = clone_joint_config(source, checkpoint)
    identity = dict(scope='Explicitly requested exploratory epoch20 full joint fine-tune; no primary-model reselection',
        source_bundle=str(source_bundle), source_config=str(path), source_config_sha256=row['config_sha256'],
        reference_run=str(run), reference_selection=selection, encoder10=first, encoder20=last,
        allowed_scientific_changes=['dapt_ckpt'], output_root=str(output),
        settings=dict(seed=source['seed'], training=source.get('training', {})),
        inference_alias_warning='Isolated bundle uses canonical J_L internally; identify as J_L_epoch20 in comparisons, never overwrite or merge into the epoch10 alias')
    receipt = output / 'preparation.json'
    if receipt.exists():
        if json.loads(receipt.read_text()) != identity:
            raise ValueError('Prepared inputs/settings changed; use a fresh control output root')
        bundle = json.loads((output / 'bundle/bundle.json').read_text())
        for item in bundle['runs']:
            if file_hash(output / 'bundle' / item['config_path']) != item['config_sha256']:
                raise ValueError('Prepared epoch20 config changed')
        return identity
    if output.exists():
        raise FileExistsError('Use a fresh output root; no existing experiment will be overwritten')
    output.mkdir(parents=True)
    write_bundle([dict(name=row['name'], phase='supervision', origin='L', seed=42,
                       condition={}, config=new)], output / 'bundle', provenance=identity)
    receipt.write_text(json.dumps(identity, indent=2, default=str))
    return identity


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-bundle', type=Path, default=Path('data/paper_256_v3/bundles/stage1'))
    p.add_argument('--checkpoint', type=Path, default=Path('data/models/daly_care_t100_joined_bins_with_sor_final_v2/pretrain_scaled_time2vec_hidden256/version_0/last.ckpt'))
    p.add_argument('--output-root', type=Path, default=Path('data/epoch20_joint_control_20261119'))
    a = p.parse_args()
    report = prepare(a.source_bundle, a.output_root, a.checkpoint)
    print(json.dumps(report, indent=2), flush=True)
    print('Prepared only; no training launched', flush=True)


if __name__ == '__main__':
    main()
