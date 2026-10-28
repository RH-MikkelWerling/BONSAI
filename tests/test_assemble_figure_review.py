import json
from pathlib import Path
from types import SimpleNamespace
import yaml
from opera.run import assemble_figure_review as review


def test_collects_nested_clinical_plots_and_reports_missing_analysis(tmp_path, monkeypatch):
    root = tmp_path / 'run'
    (root / 'clinical_scores/cll').mkdir(parents=True)
    (root / 'clinical_scores/cll/risk_groups.pdf').write_bytes(b'figure')
    (root / 'clinical_scores/patients_INTERNAL.csv').write_text('private')
    (root / 'overnight.env').write_text('')
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
    assert (output / 'clinical_scores/cll/risk_groups.pdf').read_bytes() == b'figure'
    assert not list(output.rglob('*INTERNAL*'))
    borrowing = next(p for p in recipes[0]['figures'] if p['id']=='fig_borrowing_map')
    assert str(root / 'source_removal/tuning') in borrowing['tables']['table']
