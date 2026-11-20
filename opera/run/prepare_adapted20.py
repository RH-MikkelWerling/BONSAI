"""Prepare isolated, user-approved additional adaptation and matched joint fit."""
import argparse
import copy
import json
from pathlib import Path
import yaml


def continuation_config(source, model_config, checkpoint, *, optimizer_saved):
    cfg = copy.deepcopy(source)
    if int(cfg['training']['epochs']) != 10:
        raise ValueError('Original adaptation must have ten planned epochs')
    # The original resolved YAML can retain placeholder architecture fields:
    # use the actual saved encoder architecture before disabling transfer init.
    for key in ['max_seqlen', 'hidden_size', 'num_layers', 'num_attention_heads',
                'bias', 'dropout', 'attention_dropout', 'causal', 'value_bin_vocab_size',
                'value_embedding_mode', 'abspos_encoding', 'rope_interleaved']:
        if key in model_config:
            cfg['model'][key] = model_config[key]
    cfg['continuation'] = {'checkpoint': str(checkpoint), 'cumulative_epoch_offset': 0 if optimizer_saved else 10}
    paths = cfg['paths']
    if optimizer_saved:
        paths.update(initialization_ckpt=None, initialization_vocab=None,
                     initialization_use_source_architecture=False, ckpt_path=str(checkpoint))
        cfg['training'].update(epochs=20, checkpoint_milestone_epochs=[15], auto_resume=True,
                               early_stopping_patience=None)
        mode = 'resume optimizer and scheduler; original warm-up state retained; total epochs 20'
    else:
        paths.update(initialization_ckpt=str(checkpoint), initialization_vocab=paths['vocab'],
                     initialization_use_source_architecture=True, ckpt_path=None)
        cfg['training'].update(epochs=10, checkpoint_milestone_epochs=[5], auto_resume=True,
                               early_stopping_patience=None)
        mode = 'weights-only restart; original LR and warm-up; local epochs 1-10 represent cumulative 11-20'
    return cfg, mode


def prepare(root, national):
    import torch
    from bonsai.functional.checkpointing import get_saved_encoder_config
    from opera.run.finish_figure_inputs import sha, immutable
    root, national = Path(root).resolve(), Path(national).resolve()
    source = national / 'external_dapt/hydra/config.yaml'
    fit_bundle = national / 'adapted_fits'
    bundle = json.loads((fit_bundle / 'bundle.json').read_text())
    row, = [r for r in bundle['runs'] if r['name'] == 'seed42/EA/J']
    fitpath = fit_bundle / row['config_path']
    if sha(fitpath) != row['config_sha256']:
        raise ValueError('J_EA source config hash changed')
    fit = yaml.safe_load(fitpath.read_text())
    selected = Path(fit['dapt_ckpt'])
    last = selected.parent / 'last.ckpt'
    data = torch.load(last, map_location='cpu', mmap=True, weights_only=False)
    if int(data.get('epoch', -1)) != 9:
        raise ValueError('Need actual final adaptation epoch10 last.ckpt')
    optimizer_saved = bool(data.get('optimizer_states')) and bool(data.get('lr_schedulers'))
    model = get_saved_encoder_config(data['hyper_parameters'])
    if model.get('hidden_size') != 256 or model.get('vocab_size') != 64999:
        raise ValueError('Unexpected adapted encoder width/vocabulary; inspect source release')
    cfg, mode = continuation_config(yaml.safe_load(source.read_text()), model, last,
                                    optimizer_saved=optimizer_saved)
    cfg['continuation']['checkpoint_sha256'] = sha(last)
    root.mkdir(parents=True, exist_ok=True)
    receipt = dict(scope='Approved exploratory adaptation continuation; no primary alias changes',
                   source_config=str(source), source_config_sha256=sha(source),
                   source_fit_config=str(fitpath), source_fit_config_sha256=sha(fitpath),
                   selected_adaptation_checkpoint=str(selected), selected_adaptation_sha256=sha(selected),
                   final_adaptation_checkpoint=str(last), final_adaptation_sha256=sha(last),
                   completed_epochs=10, optimizer_state_restored=optimizer_saved,
                   schedule=mode, epoch_offset=0 if optimizer_saved else 10,
                   original_training=cfg['training'], joint_encoder_lr_multiplier=fit['training'].get('encoder_lr_multiplier'))
    immutable(root / 'adaptation.yaml', cfg)
    receipt_path = root / 'preparation.json'
    if receipt_path.exists() and json.loads(receipt_path.read_text()) != receipt:
        raise ValueError('Adaptation continuation inputs changed; use a fresh output root')
    receipt_path.write_text(json.dumps(receipt, indent=2))
    return receipt


def joint(root, national):
    from opera.run.prepare_epoch20_joint_finetune import clone_joint_config, checkpoint_record
    from opera.run.prepare_paper_run_bundle import write_bundle
    from opera.run.finish_figure_inputs import sha
    root, national = Path(root).resolve(), Path(national).resolve()
    receipt = json.loads((root / 'preparation.json').read_text())
    checkpoint = root / 'pretraining/version_0/last.ckpt'
    record = checkpoint_record(checkpoint)
    if record['completed_epochs'] + receipt['epoch_offset'] != 20:
        raise ValueError('Adaptation has not completed cumulative epoch20')
    path = Path(receipt['source_fit_config'])
    if sha(path) != receipt['source_fit_config_sha256']:
        raise ValueError('Original J_EA recipe changed')
    cfg = clone_joint_config(yaml.safe_load(path.read_text()), checkpoint)
    destination = root / 'joint/bundle'
    if destination.exists():
        saved = json.loads((destination / 'bundle.json').read_text())
        item, = saved['runs']
        if yaml.safe_load((destination / item['config_path']).read_text()) != cfg:
            raise ValueError('Existing adapted20 joint bundle differs')
        return
    write_bundle([dict(name='seed42/EA/J', phase='external', origin='EA', seed=42,
                       condition={}, config=cfg)], destination,
                 provenance=dict(scope='Exploratory J_EA_epoch20', allowed_scientific_changes=['dapt_ckpt'],
                                 reference_config_sha256=sha(path), encoder20=record))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-root', type=Path, default=Path('data/adapted20_control_20261120'))
    p.add_argument('--national-root', type=Path, default=Path('data/paper_256_v3_national_20261111'))
    p.add_argument('--stage', choices=['prepare', 'joint'], default='prepare')
    a = p.parse_args()
    if a.stage == 'joint': joint(a.output_root, a.national_root)
    else: print(json.dumps(prepare(a.output_root, a.national_root), indent=2), flush=True)

if __name__ == '__main__':
    main()
