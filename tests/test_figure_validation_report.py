import json
import sys
import matplotlib.pyplot as plt
import pandas as pd
import yaml
from opera.run import render_paper_results as renderer
from opera.visualization import paper_figures


def test_validation_report_draws_and_reports_uncertainty_without_writing_figures(tmp_path, monkeypatch):
    drawn = []
    def figure_audit_stub(frame, mark=None):
        fig, ax = plt.subplots()
        fig.canvas.mpl_connect('draw_event', lambda event: drawn.append(True))
        ax.plot([0, 1], [0, 1])
        return fig
    monkeypatch.setattr(paper_figures, 'figure_audit_stub', figure_audit_stub, raising=False)
    table = tmp_path / 'input.csv'
    pd.DataFrame({'interval_status':['insufficient_draws']}).to_csv(table, index=False)
    recipe = tmp_path / 'recipe.yaml'
    recipe.write_text(yaml.safe_dump({'figures':[{'id':'audit', 'function':'figure_audit_stub',
                                                 'tables':{'frame':str(table)}}]}))
    report = tmp_path / 'validation.json'
    output = tmp_path / 'figures'
    monkeypatch.setattr(sys, 'argv', ['render', '--config', str(recipe), '--output-dir', str(output),
                                     '--validate-render', '--report', str(report)])
    renderer.main()
    result = json.loads(report.read_text())
    assert drawn
    assert result['figures'][0]['status'] == 'rendered'
    assert result['figures'][0]['warnings']
    assert set(result['code_sha256']) == {'renderer','figures','style'}
    assert not output.exists()


def test_attribution_validation_cannot_accidentally_write_figures(tmp_path):
    from test_paper_workflow import args, env
    from opera.run.paper_experiments import build_commands
    option = args('attribution-figures', validate_render=True,
                  validation_report=tmp_path / 'validation.json')
    option.execute = True
    command = build_commands(option, env(tmp_path))[0]
    assert '--validate-render' in command
    assert '--execute' not in command
    assert '--report' in command


def test_validation_records_missing_inputs_instead_of_aborting_without_report(tmp_path, monkeypatch):
    import pytest
    recipe = tmp_path / 'recipe.yaml'
    recipe.write_text(yaml.safe_dump({'figures':[{'id':'headline', 'function':'figure_headline',
                       'tables':{'cells':str(tmp_path / 'missing.csv')}}]}))
    report = tmp_path / 'report.json'
    monkeypatch.setattr(sys,'argv',['render','--config',str(recipe),'--output-dir',str(tmp_path/'figures'),
                                   '--validate-render','--report',str(report)])
    with pytest.raises(ValueError,match='missing inputs'):
        renderer.main()
    assert json.loads(report.read_text())['figures'][0]['status']=='missing_inputs'


def test_prediction_commands_forward_exact_fit_names(tmp_path):
    from test_paper_workflow import args, env
    from opera.run.paper_experiments import build_commands
    for action in ['evaluate','status']:
        option=args(action,fit_name=['fix_J_IPW','fix_R_AB_IPW'])
        command=build_commands(option,env(tmp_path))[0]
        assert command.count('--fit-name')==2
        assert 'fix_J_IPW' in command and 'fix_R_AB_IPW' in command
