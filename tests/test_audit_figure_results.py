import pandas as pd
from opera.run.audit_figure_results import audit


def test_audit_exposes_mean_median_difference_and_empty_auc(tmp_path):
    tables = tmp_path / 'figure_inputs/plot_tables'
    tables.mkdir(parents=True)
    pd.DataFrame(dict(contrast=['R1'] * 3, metric=['auc'] * 3,
                     horizon_days=[365] * 3, gain=[-.01, -.01, .08])).to_csv(
        tables / 'paired_cells.csv', index=False)
    pd.DataFrame(columns=['contrast', 'gain']).to_csv(tables / 'patient_subgroups_auc.csv', index=False)
    result = audit(tmp_path)
    assert result['plot_cell_summaries'][0]['mean'] > 0
    assert result['plot_cell_summaries'][0]['median'] < 0
    assert result['patient_subgroups_auc']['rows'] == 0


def test_audit_identifies_changed_inputs_and_pdf(tmp_path):
    import hashlib
    import json
    root = tmp_path / 'figures'
    root.mkdir()
    table = tmp_path / 'source.csv'
    table.write_text('before')
    old = hashlib.sha256(table.read_bytes()).hexdigest()
    table.write_text('changed')
    (root / 'figure_manifest.json').write_text(json.dumps({'figures': [dict(
        id='example', input_hashes={str(table): old}, output_pdf_sha='old')]}))
    result = audit(tmp_path)
    assert result['render_input_mismatches'][0]['figure'] == 'example'
    assert result['render_output_mismatches'] == ['example']


def test_audit_survives_empty_outputs(tmp_path):
    tables = tmp_path / 'figure_inputs/plot_tables'
    tables.mkdir(parents=True)
    for name in ['paired_cells', 'patient_subgroups_auc']:
        (tables / f'{name}.csv').write_text('')
    result = audit(tmp_path)
    assert result['plot_cells_error']
    assert result['patient_subgroups_auc']['rows'] == 0


def test_existing_workspace_reports_missing_clinical_score_columns(tmp_path):
    metadata = tmp_path / 'metadata.csv'
    pd.DataFrame({'subject_id': ['a'], 'ipi_computed_score': [1],
                  'ipi_computed_group': ['Low']}).to_csv(metadata, index=False)
    (tmp_path / 'overnight.env').write_text(f'OPERA_PAPER_PATIENT_METADATA={metadata}\n')
    result = audit(tmp_path)['computed_clinical_score_columns']
    assert result['IPI']['available']
    assert not result['NCCN-IPI']['available']
