import pandas as pd
from opera.run.diagnose_router_domains import diagnose


def test_trace_detects_source_and_plot_disagreement(tmp_path):
    plots = tmp_path / 'figure_inputs/plot_tables'
    analysis = tmp_path / 'prediction_analysis'
    plots.mkdir(parents=True)
    analysis.mkdir()
    rows = [dict(cohort=disease, outcome=outcome, endpoint_group=domain,
        metric='auc', contrast='R1', horizon_days=365, gain=gain, group_mean=export,
        n_cases=5, n_controls=10) for disease, outcome, domain, gain, export in [
        ('a', 'one', 'Disease course', .03, .02),
        ('b', 'one', 'Disease course', .01, .02),
        ('a', 'two', 'Cardiovascular', -.2, .1)]]
    # Aliased duplicates must not double the inventory.
    pd.concat([pd.DataFrame(rows), pd.DataFrame(rows).assign(contrast='OPERA-J')]).to_csv(
        plots / 'paired_cells.csv', index=False)
    pd.DataFrame([dict(seed=42, metric='auc', contrast='R1', stratum='endpoint_group',
        stratum_value=domain, horizon_days=365, gain=gain) for domain, gain in [
        ('Disease course', .02), ('Cardiovascular', -.2)]]).to_csv(
        analysis / 'effect_summaries.csv', index=False)
    result = diagnose(tmp_path)
    domains = {r['domain']: r for r in result['domains']}
    assert domains['Disease course']['n_cells'] == 2
    assert domains['Disease course']['source_matches_cell_mean']
    assert not domains['Cardiovascular']['export_matches_cell_mean']
    assert domains['Cardiovascular']['lowest_cells'][0]['outcome'] == 'two'
