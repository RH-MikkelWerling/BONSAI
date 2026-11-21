"""Prepare isolated cutoff-corrected encoders and replacement fit bundles. No launch."""
import argparse
import copy
import json
from pathlib import Path

import yaml

from opera.run.finish_figure_inputs import immutable, sha

NATIONAL_SHA = 'be4219b5213f8c5c07e39d1cfdae2da2d0e2574cc80aa84092f9b1b23fc3420d'
COHORT10_SHA = 'b0e0bf535ec4c4e928f67e7eee1fe89e04158df2a2aa2f89af0bf658cdded2cf'
COHORT20_SHA = '6afe3c221e974cfe81dbccbc93e63c10627ea5bbecf950a1b351af8b0e0f89f0'
DAPT10_SHA = '3807decd16f72f936359cac4e2f2aa13ddb4b890adcf7495755396f3cbb051c7'
CUTOFF = dict(year=2022, month=1, day=1)
ARCHITECTURE = ('max_seqlen', 'hidden_size', 'num_layers', 'num_attention_heads',
                'bias', 'dropout', 'attention_dropout', 'causal', 'value_bin_vocab_size',
                'value_embedding_mode', 'abspos_encoding', 'rope_interleaved')


def corrected_config(source, architecture, *, adapted):
    cfg = copy.deepcopy(source)
    cfg.pop('defaults', None)
    cfg.pop('hydra', None)
    cfg.pop('continuation', None)
    for key in ARCHITECTURE:
        if key in architecture:
            cfg['model'][key] = architecture[key]
    cfg['paths']['ckpt_path'] = None
    if not adapted:
        cfg['paths'].update(initialization_ckpt=None, initialization_vocab=None,
                            initialization_use_source_architecture=False)
    cfg['training'].update(cutoff_date=copy.deepcopy(CUTOFF), epochs=20,
                          checkpoint_milestone_epochs=[10],
                          auto_resume=True, early_stopping_patience=None,
                          recovery_cutoff_audit=True, export_epoch_counters=True)
    return cfg


def resolved(path):
    from dotenv import load_dotenv
    from omegaconf import OmegaConf
    load_dotenv()
    OmegaConf.register_new_resolver('version', lambda: 0, replace=True)
    cfg = OmegaConf.load(path)
    # Execution-only Hydra nodes can reference a live Hydra job. They are not
    # part of the frozen scientific recipe and must be dropped before resolve.
    cfg = OmegaConf.create({k: v for k, v in cfg.items_ex(resolve=False) if k != 'hydra'})
    return OmegaConf.to_container(cfg, resolve=True)


def prepare_encoders(root, local, national):
    import torch
    from bonsai.functional.checkpointing import get_saved_encoder_config
    receipts = {}
    for family, directory, adapted in [('cohort', local, False), ('adapted', national / 'external_dapt', True)]:
        source = directory / 'hydra/config.yaml'
        original_checkpoint = directory / 'version_0/last.ckpt'
        if sha(original_checkpoint) != (DAPT10_SHA if adapted else COHORT20_SHA):
            raise ValueError('Original encoder identity differs from the audited release')
        cfg = resolved(source)
        data = torch.load(original_checkpoint, map_location='cpu', mmap=True, weights_only=False)
        architecture = get_saved_encoder_config(data['hyper_parameters'])
        del data
        cfg = corrected_config(cfg, architecture, adapted=adapted)
        if adapted:
            initial = Path(cfg['paths'].get('initialization_ckpt') or '')
            if not initial.is_file() or sha(initial) != NATIONAL_SHA:
                raise ValueError('DAPT must initialize directly from the independently verified national encoder')
            if not cfg['paths'].get('initialization_vocab'):
                raise ValueError('Missing national source vocabulary')
        for field in ('train_split', 'val_split', 'population', 'vocab'):
            if not Path(cfg['paths'][field]).is_file():
                raise FileNotFoundError(cfg['paths'][field])
        destination = root / family
        destination.mkdir(parents=True, exist_ok=True)
        immutable(destination / 'config.yaml', cfg)
        receipts[family] = dict(source_config=str(source), source_config_sha256=sha(source),
                                architecture_checkpoint=str(original_checkpoint),
                                architecture_checkpoint_sha256=sha(original_checkpoint),
                                new_config_sha256=sha(destination / 'config.yaml'),
                                original_cutoff=resolved(source)['training'].get('cutoff_date'),
                                new_cutoff=CUTOFF,
                                initialization_checkpoint=cfg['paths'].get('initialization_ckpt'),
                                inputs={field: dict(path=cfg['paths'][field],
                                                    bytes=Path(cfg['paths'][field]).stat().st_size,
                                                    mtime_ns=Path(cfg['paths'][field]).stat().st_mtime_ns)
                                        for field in ('train_split', 'val_split', 'population', 'vocab')},
                                population_sha256=sha(Path(cfg['paths']['population'])),
                                vocabulary_sha256=sha(Path(cfg['paths']['vocab'])))
    immutable(root / 'encoder_preparation.yaml', receipts)
    return receipts


def replacement(config, root):
    if config.get('encoder_initialization') == 'random':
        if config.get('dapt_embedding_store') or config.get('model', {}).get('dapt_anchor_weight', 0):
            raise ValueError('Random control uses a pretrained representation dependency; classify it explicitly')
        return None, 'unaffected_random'
    checkpoint = Path(config.get('dapt_ckpt') or '')
    path = str(checkpoint)
    if 'pretrain_scaled_time2vec_hidden256' in path:
        if checkpoint.name not in ('epoch_010.ckpt', 'last.ckpt'):
            raise ValueError(f'Unclassified cohort checkpoint: {checkpoint}')
        family = 'cohort'
        name = checkpoint.name
    elif '/external_dapt/' in path:
        family, name = 'adapted', 'epoch_010.ckpt'
    else:
        return None, 'unaffected_or_unclassified'
    if config.get('source_opera_ckpt') or config.get('paths', {}).get('resume_ckpt'):
        raise ValueError('Supervised initialization/resume dependency needs an explicit corrected mapping')
    if config.get('dapt_embedding_store'):
        raise ValueError('Precomputed encoder embedding store must be regenerated for the corrected encoder')
    result = copy.deepcopy(config)
    result['dapt_ckpt'] = str((root / family / 'pretraining/version_0' / name).resolve())
    result['cutoff_recovery'] = dict(exclusive_cutoff='2022-01-01',
                                    expected_epochs=20 if family == 'cohort' and name == 'last.ckpt' else 10)
    if result.get('training', {}).get('auto_resume', True) is False:
        result['training']['auto_resume'] = True
        result['cutoff_recovery']['resume_policy_change'] = 'Enable own-directory last.ckpt resume; fresh-fit optimization unchanged'
    return result, family


def verify_affected_checkpoint(path, family):
    expected = DAPT10_SHA if family == 'adapted' else (COHORT10_SHA if path.name == 'epoch_010.ckpt' else COHORT20_SHA)
    if sha(path) != expected:
        raise ValueError(f'Unverified affected source encoder: {path}')


def prepare_bundle(source, output, root, phases=None):
    from opera.run.prepare_paper_run_bundle import write_bundle
    report = json.loads((source / 'bundle.json').read_text())
    entries, skipped, verified = [], [], set()
    for row in report['runs']:
        if phases and row['phase'] not in phases:
            skipped.append(dict(name=row['name'], reason='outside_requested_phases'))
            continue
        path = source / row['config_path']
        if sha(path) != row['config_sha256']:
            raise ValueError(f'Source config hash mismatch: {path}')
        original = yaml.safe_load(path.read_text())
        cfg, family = replacement(original, root)
        if cfg is None:
            skipped.append(dict(name=row['name'], reason=family))
            continue
        checkpoint = Path(original['dapt_ckpt'])
        if checkpoint not in verified:
            verify_affected_checkpoint(checkpoint, family)
            verified.add(checkpoint)
        entries.append({k: row[k] for k in ('name', 'phase', 'origin', 'seed', 'condition')} | dict(config=cfg))
    if not entries:
        raise ValueError('No affected fits in this source bundle')
    if output.exists():
        saved = json.loads((output / 'bundle.json').read_text())
        expected = {e['name']: e['config'] for e in entries}
        actual = {r['name']: yaml.safe_load((output / r['config_path']).read_text()) for r in saved['runs']}
        if actual != expected or saved['provenance']['source_bundle_sha256'] != sha(source / 'bundle.json'):
            raise ValueError('Recovery bundle inputs changed; use a fresh output')
        return saved
    return write_bundle(entries, output, selected_router=report.get('selected_router'),
                        provenance=dict(scope='Cutoff correction; original downstream settings retained',
                                        source_bundle=str(source), source_bundle_sha256=sha(source / 'bundle.json'),
                                        skipped=skipped, changed_fields=['dapt_ckpt', 'cutoff_recovery (provenance guard only)']))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage', choices=['encoders', 'bundle', 'inventory'])
    p.add_argument('--root', type=Path, default=Path('data/cutoff_recovery_20261121'))
    p.add_argument('--local-root', type=Path, default=Path('data/models/daly_care_t100_joined_bins_with_sor_final_v2/pretrain_scaled_time2vec_hidden256'))
    p.add_argument('--national-root', type=Path, default=Path('data/paper_256_v3_national_20261111'))
    p.add_argument('--source-bundle', type=Path)
    p.add_argument('--output-bundle', type=Path)
    p.add_argument('--phase', action='append', help='Bundle: copy only these original phases; repeat as needed')
    a = p.parse_args()
    root = a.root.resolve()
    if a.stage == 'encoders':
        result = prepare_encoders(root, a.local_root.resolve(), a.national_root.resolve())
    elif a.stage == 'inventory':
        result = []
        for directory in (Path('data/paper_256_v3'), a.national_root):
            if not directory.exists():
                raise FileNotFoundError(directory)
            for path in sorted(directory.rglob('bundle.json')):
                source = json.loads(path.read_text())
                counts = {}
                for row in source.get('runs', []):
                    phase = row['phase']
                    counts[phase] = counts.get(phase, 0) + 1
                if counts:
                    result.append(dict(source_bundle=str(path.parent), phase_counts=counts))
        root.mkdir(parents=True, exist_ok=True)
        (root / 'source_bundle_inventory.json').write_text(json.dumps(result, indent=2))
    else:
        if not a.source_bundle or not a.output_bundle:
            p.error('bundle requires --source-bundle and --output-bundle')
        if not a.output_bundle.resolve().is_relative_to(root):
            p.error('Recovery bundles must be inside the fresh recovery root')
        result = prepare_bundle(a.source_bundle.resolve(), a.output_bundle.resolve(), root, a.phase)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
