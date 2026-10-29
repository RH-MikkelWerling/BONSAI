import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from opera.run import rerun_full_tuning_rehearsal as runner


@pytest.mark.parametrize('clinical_gaps', [False, True])
@pytest.mark.parametrize('diagnostic_failure', [False, True])
@pytest.mark.parametrize('verify_sources', [False, True])
def test_fresh_full_rehearsal_uses_locked_inputs_and_new_auxiliary_outputs(
        tmp_path, monkeypatch, clinical_gaps, diagnostic_failure, verify_sources):
    source = tmp_path / 'old'
    (source / 'analysis_inventory').mkdir(parents=True)
    (source / 'analysis_inventory/analysis.yaml').write_text(yaml.safe_dump(dict(
        estimand='cif_ipcw', artifacts=[dict(model='R_L', split='tuning', path='/frozen/predictions')])) )
    (source / 'completion.json').write_text(json.dumps(dict(inputs=dict(fit=dict(checkpoint='/frozen/model')))))
    (source / 'overnight.env').write_text(f'OPERA_PAPER_ANALYSIS_ROOT={source}\n')
    (source / 'attributions').mkdir()
    (source / 'attributions/old_summary.csv').write_text('old,cached\n')
    (source / 'prediction_analysis').mkdir()
    (source / 'prediction_analysis/old_estimates.csv').write_text('old,cached\n')
    output = tmp_path / 'fresh'
    calls = []
    def execute(command, **kwargs):
        calls.append(command)
        name = command[2]
        if name.endswith('audit_router_vectors') and diagnostic_failure:
            return SimpleNamespace(returncode=1)
        if name.endswith('complete_local_figures'):
            complete = output / 'analysis'
            (complete / 'figure_inputs').mkdir(parents=True)
            (complete / 'figure_inputs/supplement_tables.yaml').write_text('tables: []')
            (complete / 'overnight.env').write_text(f'OPERA_PAPER_CELL_METADATA={tmp_path}/cells.csv\n')
        elif name.endswith('prepare_attribution_refresh'):
            summary = output / 'ig/attributions/summary_365d'
            summary.mkdir(parents=True)
            (summary / 'attribution_numbers.json').write_text('{"new": true}')
            (output / 'ig/overnight.env').write_text('')
        elif name.endswith('paper_experiments') and command[3] == 'input-restrictions':
            restrictions = output / 'ig/input_restrictions'
            restrictions.mkdir()
            (restrictions / 'input_restriction_summary.csv').write_text('label,value\nfresh,1\n')
        elif name.endswith('assemble_figure_review'):
            review = output / 'review'
            review.mkdir()
            (review / 'review_manifest.json').write_text(json.dumps(dict(
                status='partial_review' if clinical_gaps else 'review_ready')))
        return SimpleNamespace(returncode=0)
    verification_calls = []
    monkeypatch.setattr(runner, 'verify_sources', lambda repo: verification_calls.append(repo))
    original_digest = runner.digest
    def checked_digest(path):
        if Path(path).name == 'code_manifest.sha256':
            assert verify_sources, 'Default run must not require the source manifest'
        return original_digest(path)
    monkeypatch.setattr(runner, 'digest', checked_digest)
    monkeypatch.setattr(runner.subprocess, 'run', execute)
    monkeypatch.setattr('sys.argv', ['full', '--analysis-root', str(source), '--output-dir', str(output)]
                        + (['--verify-sources'] if verify_sources else []))
    runner.main()
    result = json.loads((output / 'full_rehearsal.json').read_text())
    assert result['status'] == ('finished_with_gaps' if clinical_gaps or diagnostic_failure else 'complete')
    assert len(verification_calls) == (2 if verify_sources else 0)
    assert result['source_verification_enabled'] is verify_sources
    if not verify_sources:
        assert result['code_manifest_sha256'] is None
        assert result['source_changed_during_run'] is None
    assert not result['reused_prior_analysis_outputs']
    assert not (output / 'paired/prediction_analysis/old_estimates.csv').exists()
    assert not (output / 'analysis/attributions/old_summary.csv').exists()
    assert json.loads((output / 'analysis/attributions/summary_365d/attribution_numbers.json').read_text()) == {'new': True}
    assert (output / 'analysis/input_restrictions/input_restriction_summary.csv').read_text().endswith('fresh,1\n')
    assert not any('--resume' in command for command in calls)
    assert not any('train' in command or 'held_out' in command for command in calls)
    assert any(c[2].endswith('complete_local_figures') and '--clinical-only' not in c for c in calls)
    assert any(c[2].endswith('prepare_attribution_refresh') and '--execute' in c for c in calls)
    assert any(c[2].endswith('package_figure_review') for c in calls)
    assert any(c[2].endswith('audit_router_vectors') and
               str(output / 'analysis/routed_vectors') in c for c in calls)
    whole, = [c for c in calls if c[2].endswith('evaluate_whole_cohort_outcomes')]
    assert whole[whole.index('--split') + 1] == 'tuning'
    assert whole[whole.index('--models') + 1:whole.index('--split')] == ['OPERA', 'J_L', 'X', 'XP']
    assert result['steps']['router_vectors']['status'] == ('failed' if diagnostic_failure else 'complete')
    assert result['steps']['whole_cohort']['status'] == 'complete'
    with pytest.raises(FileExistsError):
        runner.main()


def test_source_sync_preflight_checks_actual_bytes(tmp_path):
    (tmp_path / 'opera').mkdir()
    path = tmp_path / 'opera/runtime.py'
    path.write_text('current')
    (tmp_path / 'code_manifest.sha256').write_text(f'{runner.digest(path)}  opera/runtime.py\n')
    runner.verify_sources(tmp_path)
    path.write_text('old server copy')
    with pytest.raises(ValueError, match='Source checksum mismatch'):
        runner.verify_sources(tmp_path)
