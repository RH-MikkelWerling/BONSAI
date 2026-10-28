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
