import copy
import pytest
import yaml
from opera.run.prepare_sep_sensitivity import prepare


def fixtures():
    pretrain = dict(seed=42, paths=dict(dataset_class='x.ARPretrainDataset'),
                    training=dict(ignore_target_tokens=['[SEP]'], epochs=20), model={})
    joint = dict(seed=42, paths={}, model=dict(freeze_encoder=False),
                 competing_risk=dict(readout_mode='shared'),
                 training=dict(batch_sampling=dict(type='random', seed=42)))
    return pretrain, joint


def test_matched_target_arms_and_joint_seeds(tmp_path):
    pretrain, joint = fixtures()
    originals = copy.deepcopy((pretrain, joint))
    root = tmp_path / 'pilot'
    prepare(pretrain, joint, root)
    load = lambda name: yaml.safe_load((root / 'configs' / f'{name}.yaml').read_text())
    excluded, included = load('pretrain_exclude_sep'), load('pretrain_include_sep')
    assert excluded['training'].pop('ignore_target_tokens') == ['[SEP]']
    assert included['training'].pop('ignore_target_tokens') == []
    assert included == excluded
    for seed in (42, 43, 44):
        left, right = load(f'joint_exclude_sep_s{seed}'), load(f'joint_include_sep_s{seed}')
        assert 'exclude_sep/pretrain/last.ckpt' in left.pop('dapt_ckpt')
        assert 'include_sep/pretrain/last.ckpt' in right.pop('dapt_ckpt')
        assert left == right
        assert left['seed'] == left['training']['batch_sampling']['seed'] == seed
    assert (pretrain, joint) == originals
    with pytest.raises(FileExistsError):
        prepare(pretrain, joint, root)


def test_reject_wrong_objective_and_router(tmp_path):
    pretrain, joint = fixtures()
    pretrain['paths']['dataset_class'] = 'x.MLMPretrainDataset'
    with pytest.raises(ValueError, match='code-only AR'):
        prepare(pretrain, joint, tmp_path / 'bad')
    pretrain, joint = fixtures()
    joint['competing_risk']['readout_mode'] = 'source_anchored_adapters'
    with pytest.raises(ValueError, match='shared joint'):
        prepare(pretrain, joint, tmp_path / 'bad')


def test_paired_tuning_summary_from_native_exports(tmp_path, monkeypatch):
    import json
    import sys
    import pandas as pd
    from opera.run.summarise_sep_sensitivity import main
    metadata = tmp_path / 'metadata.csv'
    pd.DataFrame({'subject_id': [str(i) for i in range(20)],
                  'cohort_fine': ['disease'] * 20}).to_csv(metadata, index=False)
    (tmp_path / 'sensitivity.json').write_text(json.dumps(
        dict(seeds=[42], patient_metadata=str(metadata))))
    for arm, strength in [('exclude_sep', 0.1), ('include_sep', 0.2)]:
        destination = tmp_path / arm / 'joint_s42/tuning_predictions'
        destination.mkdir(parents=True)
        pd.DataFrame([dict(subject_id=str(i), cohort='pooled', outcome='example',
                           time_days=10 if i % 2 else 1000, event=i % 2,
                           cif_365d=0.5 + strength * (1 if i % 2 else -1))
                      for i in range(20)]).to_parquet(destination / 'direct_cr_predictions.parquet')
    monkeypatch.setattr(sys, 'argv', ['summary', '--root', str(tmp_path)])
    main()
    summary = pd.read_csv(tmp_path / 'paired_tuning_analysis/paired_means.csv')
    assert set(summary.aggregation) == {'equal_cell', 'disease_balanced'}
    # Both rank perfectly; the better probabilities improve Brier without AUC gain.
    auc = summary.loc[summary.metric == 'auc', 'gain']
    brier = summary.loc[summary.metric == 'brier', 'gain']
    assert auc.eq(0).all() and brier.gt(0).all()


def test_generated_configs_compose_without_original_defaults(tmp_path):
    from hydra import compose, initialize_config_dir
    from opera.run.prepare_sep_sensitivity import load_config
    pretrain, joint = fixtures()
    source = tmp_path / 'saved.yaml'
    source.write_text(yaml.safe_dump(dict(pretrain, hydra={'run': {'dir': '${hydra:runtime.output_dir}'}})))
    resolved = load_config(source)
    assert 'hydra' not in resolved
    root = tmp_path / 'pilot'
    prepare(resolved, joint, root)
    with initialize_config_dir(config_dir=str(root / 'configs'), version_base='1.2'):
        config = compose(config_name='pretrain_include_sep')
    assert config.training.ignore_target_tokens == []
    assert config.seed == pretrain['seed']
