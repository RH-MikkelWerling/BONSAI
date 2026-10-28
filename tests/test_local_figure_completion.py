import hashlib
from pathlib import Path
import json
import numpy as np
import pandas as pd
import pytest
from opera.run.prepare_clinical_figure_metadata import prepare
from opera.run.complete_local_figures import selected_fit


def test_score_merge_preserves_population_and_derives_display_groups(tmp_path):
    base = tmp_path / 'base.parquet'
    source = tmp_path / 'scores.parquet'
    pd.DataFrame(dict(subject_id=[1, 2, 3], history_event_count=[7, 8, 9])).to_parquet(base)
    pd.DataFrame(dict(subject_id=['2', '1'], ipi_computed_score=[4, 2],
                     nccn_ipi_score=[6, 1])).to_parquet(source)
    before = base.read_bytes()
    result = prepare(base, [source], tmp_path / 'merged.parquet')
    merged = pd.read_parquet(tmp_path / 'merged.parquet')
    assert merged.subject_id.tolist() == ['1', '2', '3']
    assert merged.history_event_count.tolist() == [7, 8, 9]
    assert merged.ipi_computed_group.iloc[:2].tolist() == ['Intermediate', 'High']
    assert merged.nccn_ipi_group.iloc[:2].tolist() == ['Low', 'High']
    assert pd.isna(merged.ipi_computed_score.iloc[2])
    assert result['nccn_ipi']['n_complete'] == 2
    assert base.read_bytes() == before


def test_score_merge_rejects_conflicting_numeric_scores(tmp_path):
    base, other = tmp_path / 'a.parquet', tmp_path / 'b.parquet'
    pd.DataFrame(dict(subject_id=['1'], ipi_computed_score=[1])).to_parquet(base)
    pd.DataFrame(dict(subject_id=['1'], ipi_computed_score=[5])).to_parquet(other)
    with pytest.raises(ValueError, match='Conflicting'):
        prepare(base, [other], tmp_path / 'out.parquet')


def test_selected_fit_checks_prediction_checkpoint_and_config(tmp_path, monkeypatch):
    from opera.run import complete_local_figures as runner
    run = 'L/seed42/abc'
    pred = tmp_path / 'predictions' / run / 'tuning/direct_cr_predictions.parquet'
    pred.parent.mkdir(parents=True)
    pred.write_bytes(b'predictions')
    checkpoint = tmp_path / 'neural_runs' / run / 'best.ckpt'
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b'checkpoint')
    config = tmp_path / 'config.yaml'
    config.write_text('model: matched')
    digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    marker = {'inputs': {'checkpoint_sha256': digest(checkpoint), 'config_sha256': digest(config)},
              'prediction_sha256': digest(pred)}
    (pred.parent / 'evaluation_complete.json').write_text(json.dumps(marker))
    row = dict(scientific_hash='abc', run_directory=run)
    monkeypatch.setattr(runner, 'bundle_runs', lambda p: ({}, [(row, config, {'training': {'eval_monitor_metric': 'loss'}})]))
    monkeypatch.setattr(runner, 'canonical_models', lambda *a: ['R_AB_L'])
    monkeypatch.setattr(runner, 'verified_training_selection', lambda p: dict(best_checkpoint_sha256=digest(checkpoint), monitor='loss'))
    analysis = dict(opera_model='R_L', artifacts=[dict(model='R_L', split='tuning', seed=42, path=str(pred))])
    assert selected_fit(analysis, tmp_path, [tmp_path])['model'] == 'R_AB_L'
    pred.write_bytes(b'changed')
    with pytest.raises(ValueError, match='predictions changed'):
        selected_fit(analysis, tmp_path, [tmp_path])


def test_outcome_specific_vectors_keep_projection_and_mask_uncovered_patients(tmp_path):
    from opera.run.build_paper_embedding_tables import project
    rng = np.random.default_rng(70)
    values = rng.normal(size=(24, 5))
    frame = pd.DataFrame(dict(subject_id=[str(i) for i in range(24)],
                             split=['tuning'] * 12 + ['held_out'] * 12,
                             embedding=list(values)))
    vector = tmp_path / 'vectors.npz'
    np.savez(vector, subject_ids=np.array([str(i) for i in range(12, 18)]),
             shared=values[12:18], adapted=values[12:18] + .1)
    vector.with_suffix('.json').write_text(json.dumps(dict(target_outcome='example', split='held_out')))
    original = project(frame, method='pca')
    result = project(frame, method='pca', routed_vectors={'example': str(vector)}, allow_partial_routed=True)
    np.testing.assert_allclose(original[['x', 'y']], result[['x', 'y']])
    assert result['routed:example:x'].notna().sum() == 6
    with pytest.raises(ValueError, match='do not cover'):
        project(frame, method='pca', routed_vectors={'example': str(vector)})


def test_completion_orchestrates_fresh_exports_without_main_analysis(tmp_path, monkeypatch):
    from types import SimpleNamespace
    import yaml
    from opera.run import complete_local_figures as runner
    from opera.run.prepare_available_plot_configs import OUTCOMES
    paper = tmp_path / 'paper'
    (paper / 'bundles/stage1').mkdir(parents=True)
    (paper / 'bundles/stage1/bundle.json').write_text('{}')
    old = paper / 'analysis/old'
    (old / 'prediction_analysis').mkdir(parents=True)
    (old / 'analysis_inventory').mkdir()
    (old / 'prediction_analysis/original.txt').write_text('unchanged')
    metadata = tmp_path / 'metadata.parquet'
    pd.DataFrame(dict(subject_id=range(60), cohort_fine=['DLBCL']*60, age_at_index=[60]*60,
        index=['R-IPI']*60, IPI=['High']*60, ipi_computed_score=[4]*60,
        nccn_ipi_score=[6]*60)).to_parquet(metadata)
    prediction = tmp_path / 'pred/direct_cr_predictions.parquet'
    prediction.parent.mkdir()
    pd.DataFrame([dict(subject_id=i, outcome=name) for i in range(60) for name in OUTCOMES]).to_parquet(prediction)
    np.savez(prediction.parent / 'patient_embeddings.npz', embeddings=np.ones((60, 2)))
    analysis = dict(patient_metadata=str(metadata), opera_model='R_L', artifacts=[
        dict(model='R_L', split='tuning', seed=42, path=str(prediction))])
    (old / 'analysis_inventory/analysis.yaml').write_text(yaml.safe_dump(analysis))
    (old / 'overnight.env').write_text(f'OPERA_PAPER_ROOT={paper}\nBONSAI_COHORT_MEMBERSHIP={metadata}\n')
    config = tmp_path / 'fit.yaml'; config.write_text('matched')
    fit = dict(model='R_AB_L', own_reference='RBH_AB_L', config=str(config),
               checkpoint=str(tmp_path / 'best.ckpt'), checkpoint_sha256='fixed', config_sha256='config', predictions=str(prediction))
    monkeypatch.setattr(runner, 'selected_fit', lambda *a: fit)
    output = paper / 'analysis/completed'
    commands = []
    def write(relative, value='{}'):
        path = output / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(value)
    def execute(command, **kwargs):
        commands.append(command)
        module = command[2].rsplit('.', 1)[-1]
        if module == 'paper_experiments':
            action = command[3]
            paths = {'init-figures': 'figure_inputs/figures.yaml',
                'clinical-scores': 'clinical_scores/headline_scores.csv',
                'embedding-tables': 'figure_inputs/plot_tables/calibration_map_outcomes.json',
                'plot-tables': 'figure_inputs/plot_tables/plot_tables_manifest.json',
                'figures': 'figures/figure_manifest.json', 'paper-numbers': 'paper_numbers/results_macros.tex',
                'supplement-tables': 'supplement_tables/supplement_tables.tex'}
            if action in paths: write(paths[action])
            if action == 'clinical-scores': write('clinical_scores/score_inventory.csv', 'status\nrendered\n')
        elif module == 'evaluate_direct_competing_risk':
            destination = Path(command[command.index('--output-dir') + 1])
            destination.mkdir(parents=True, exist_ok=True)
            if '--export-routed-outcome' in command:
                (destination / 'routed_patient_vectors_INTERNAL.npz').write_bytes(b'export')
                (destination / 'routed_patient_vectors_INTERNAL.json').write_text('{}')
            else:
                write('borrowing_inference/source_removal/design.json', json.dumps({'variants': ['none'], 'ablations': {'mean': '', 'zero': ''}, 'horizons_days': [30,60,90,180,365,730]}))
                write('borrowing_inference/direct_cr_predictions.parquet')
                for ablation in ('mean', 'zero'):
                    write(f'borrowing_inference/source_removal/{ablation}/none.parquet')
        elif module == 'audit_figure_results':
            write('figure_results_audit.json')
        elif module == 'source_removal':
            assert command[command.index('--removal-dir') + 1] == str(output / 'borrowing_inference')
            suffix = '/zero_ablation' if '--ablation' in command and command[command.index('--ablation')+1] == 'zero' else ''
            write(f'source_removal/tuning{suffix}/borrowing_map.csv')
            write(f'source_removal/tuning{suffix}/analysis_provenance.json')
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(runner.subprocess, 'run', execute)
    monkeypatch.setattr('sys.argv', ['completion', '--analysis-root', str(old), '--output-dir', str(output)])
    runner.main()
    report = json.loads((output / 'completion.json').read_text())
    assert report['status'] == 'complete'
    assert sum('--export-routed-outcome' in c for c in commands) == 5
    assert not any('--outcome' in c for c in commands)
    assert sum('--borrowing-design' in c for c in commands) == 1
    assert not any(c[3] in ['analyse', 'inventory', 'train'] for c in commands)
    assert (old / 'prediction_analysis/original.txt').read_text() == 'unchanged'
    embed = yaml.safe_load((output / 'configs/embedding_tables.tuning.yaml').read_text())
    assert len(embed['calibration_maps']['routed_vectors']) == 5
    commands.clear()
    monkeypatch.setattr('sys.argv', ['completion', '--analysis-root', str(old), '--output-dir', str(output), '--resume'])
    runner.main()
    assert commands == []
