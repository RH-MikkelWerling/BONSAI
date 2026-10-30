import json
from pathlib import Path
from types import SimpleNamespace
import yaml
import pytest
from opera.run import assemble_figure_review as review
from opera.run.clinical_figure_provenance import render_receipt


@pytest.mark.parametrize('fresh_clinical', [False, True])
def test_collects_only_verified_clinical_plots_and_reports_missing_analysis(tmp_path, monkeypatch, fresh_clinical):
    root = tmp_path / 'run'
    (root / 'clinical_scores/cll').mkdir(parents=True)
    (root / 'clinical_scores/cll/risk_groups.pdf').write_bytes(b'figure')
    (root / 'clinical_scores/patients_INTERNAL.csv').write_text('private')
    if fresh_clinical:
        receipt = render_receipt(root / 'clinical_scores',
                                 [dict(id='cll', model='R_L', threshold_split='train')], 'tuning')
        (root / 'clinical_scores/report_manifest.json').write_text(json.dumps(receipt))
        (root / 'clinical_scores/score_inventory.csv').write_text(
            'score_id,status,reason\ncll,rendered,\nips,not_estimable,Two anchors require different cutoffs\n')
    (root / 'overnight.env').write_text('')
    (root / 'input_restrictions').mkdir()
    (root / 'input_restrictions/input_restriction_summary_by_horizon.csv').write_text('horizon_days\n60\n365\n')
    output = tmp_path / 'review'
    recipes = []
    def execute(command, **kwargs):
        recipe = yaml.safe_load(Path(command[command.index('--config')+1]).read_text())
        recipes.append(recipe)
        destination = Path(command[command.index('--output-dir')+1])
        destination.mkdir()
        (destination / 'figure_manifest.json').write_text(json.dumps(dict(complete=False, skipped={'x':'missing'})))
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(review.subprocess, 'run', execute)
    monkeypatch.setattr('sys.argv', ['review','--analysis-root',str(root),'--output-dir',str(output)])
    review.main()
    report = json.loads((output / 'review_manifest.json').read_text())
    assert report['status'] == 'partial_review'
    assert 'integrated_gradients' in {m['family'] for m in report['missing']}
    assert (output / 'clinical_scores/cll/risk_groups.pdf').exists() == fresh_clinical
    assert report['clinical_validation']['valid'] == fresh_clinical
    if fresh_clinical:
        assert any(row['family'] == 'clinical_score' and row['score_id'] == 'ips'
                   and row['status'] == 'not_estimable' for row in report['missing'])
    assert not list(output.rglob('*INTERNAL*'))
    borrowing = next(p for p in recipes[0]['figures'] if p['id']=='fig_borrowing_map')
    assert str(root / 'source_removal/tuning') in borrowing['tables']['table']
    recent = next(recipe for recipe in recipes if any(p['function'] == 'figure_recent_record_gap_horizons' for p in recipe['figures']))
    assert recent['figures'][1]['options']['horizons'] == [60, 365]
    assert (output / 'reports/input_restrictions/input_restriction_summary_by_horizon.csv').is_file()


def test_explicit_stale_clinical_directory_is_rejected_before_rendering(tmp_path, monkeypatch):
    clinical = tmp_path / 'old'
    clinical.mkdir()
    output = tmp_path / 'review'
    monkeypatch.setattr('sys.argv', ['review', '--analysis-root', str(tmp_path),
                                   '--clinical-dir', str(clinical), '--output-dir', str(output)])
    with pytest.raises(ValueError, match='stale or unverified'):
        review.main()
    assert not output.exists()


def test_held_out_assembly_keeps_test_paths_and_accepts_matching_clinical_receipt(tmp_path, monkeypatch):
    root = tmp_path / 'analysis'
    clinical = root / 'clinical_scores'
    clinical.mkdir(parents=True)
    (clinical / 'score.pdf').write_bytes(b'held-out figure')
    (clinical / 'report_manifest.json').write_text(json.dumps(render_receipt(
        clinical, [dict(id='score', model='R_L', threshold_split='train')], 'held_out')))
    (root / 'overnight.env').write_text('OPERA_PAPER_FIGURE_MARK=TUNING SET\n')
    output = tmp_path / 'review'
    recipes = []

    def execute(command, **kwargs):
        recipe_path = Path(command[command.index('--config')+1])
        recipes.append(yaml.safe_load(recipe_path.read_text()))
        assert kwargs['env']['OPERA_PAPER_PLOT_SPLIT'] == 'held_out'
        assert kwargs['env']['OPERA_PAPER_FIGURE_MARK'] == ''
        destination = Path(command[command.index('--output-dir')+1])
        destination.mkdir()
        (destination / 'figure_manifest.json').write_text('{"complete": false}')
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(review.subprocess, 'run', execute)
    monkeypatch.setattr('sys.argv', ['review', '--analysis-root', str(root),
        '--clinical-dir', str(clinical), '--output-dir', str(output), '--split', 'held_out'])
    review.main()
    manifest = json.loads((output / 'review_manifest.json').read_text())
    assert manifest['evaluation_split'] == 'held_out'
    assert manifest['clinical_validation']['valid']
    borrowing = next(p for p in recipes[0]['figures'] if p['id'] == 'fig_borrowing_map')
    assert '/source_removal/held_out/' in borrowing['tables']['table']
    assert not any('TUNING' in str(p.get('options')) for recipe in recipes for p in recipe['figures'])
