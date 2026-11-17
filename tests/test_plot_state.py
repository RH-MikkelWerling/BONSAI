import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pytest
from opera.visualization.plot_state import export_plot_state, render_plot_state


def test_presentation_exports_no_scatter_arrays_and_replays_labels_and_curves(tmp_path):
    fig, ax = plt.subplots()
    offsets = np.array([[0.123456789, .234567891], [.765432198, .876543219]])
    ax.scatter(*offsets.T, c=[.356789123, .567891234])
    ax.plot([0, .5, 1], [1, .8, .6], label='Group survival')
    ax.set_xlabel('Years'); ax.set_ylabel('Survival'); ax.legend()
    ax.text(.1, .9, 'A', transform=ax.transAxes)
    fig.opera_export_aggregate_curves = True
    path = tmp_path/'example.plot.json'
    before = (ax.axison, [x.get_visible() for x in ax.lines], ax.get_legend().get_visible())
    export_plot_state(fig,path)
    assert before == (ax.axison,[x.get_visible() for x in ax.lines],ax.get_legend().get_visible())
    text = path.read_text()
    assert '0.123456789' not in text and '0.356789123' not in text
    state = json.loads(text)
    assert len(state['axes'][0]['lines']) == 1
    assert state['axes'][0]['image'].endswith('.png')
    assert state['axes'][0]['legend']['handles'][0]['label']=='Group survival'
    replay = render_plot_state(path)
    assert replay.axes[0].get_xlabel() == 'Years'
    assert replay.axes[0].texts[0].get_text() == 'A'
    assert len(replay.axes[0].lines) == 1
    plt.close('all')


def test_colorbar_outline_does_not_break_ordinary_axes_replay(tmp_path):
    fig, ax = plt.subplots()
    fig.colorbar(ax.imshow([[0, 1], [2, 3]]), ax=ax)
    path = tmp_path / 'colorbar.plot.json'
    export_plot_state(fig, path)
    replay = render_plot_state(path)
    assert len(replay.axes) == 2
    plt.close('all')


def test_clinical_replay_migrates_math_panel_letters_to_the_house_font(tmp_path):
    from opera.visualization.paper_figures import _apply_clinical_font, _clinical_title
    fig, ax = plt.subplots()
    ax.set_title(r'$\mathbf{A}$  Model risk groups', loc='left')
    path = tmp_path / 'clinical.plot.json'
    export_plot_state(fig, path)
    replay = render_plot_state(path)
    _apply_clinical_font(replay)
    title = next(t for t in replay.axes[0].texts if 'Model risk groups' in t.get_text())
    assert title.get_text() == 'A  Model risk groups'
    assert '$' not in title.get_text()
    assert _clinical_title(ax, 'B', 'Index groups').get_text() == 'B  Index groups'
    plt.close('all')


def test_inset_counts_left_panel_titles_and_risk_table_positions_survive(tmp_path):
    fig,ax=plt.subplots()
    ax.set_title('A Model groups',loc='left')
    ax.table(cellText=[['30','25']],colLabels=['Start','1 year'],bbox=[.2,-.4,.8,.2])
    inset=ax.inset_axes([0,-.8,1,.25]);inset.imshow([[10,20]])
    inset.text(0,0,'10');inset.set_title('Counts',loc='left')
    path=tmp_path/'groups.plot.json';export_plot_state(fig,path)
    state=json.loads(path.read_text())
    assert len(state['axes'])==2
    assert any(t['text']=='A Model groups' for t in state['axes'][0]['texts'])
    replay=render_plot_state(path)
    assert len(replay.axes)==2
    # Risk-table numbers stay below the parent axes, not inside the curves.
    table_numbers=[t for t in replay.axes[0].texts if t.get_text()=='30']
    assert len(table_numbers)==1 and table_numbers[0].get_position()[1]<0
    assert any(t.get_text()=='10' for t in replay.axes[1].texts)
    assert ax.axison and inset.axison
    plt.close('all')


def test_unclassified_lines_stay_image_only_and_layer_paths_cannot_escape(tmp_path):
    fig,ax = plt.subplots();ax.plot([.123456789,.234567891],[.4,.6])
    path = tmp_path/'example.plot.json';export_plot_state(fig,path)
    state = json.loads(path.read_text())
    assert not state['axes'][0]['lines']
    assert '0.123456789' not in path.read_text()
    state['axes'][0]['image']='../private.png';path.write_text(json.dumps(state))
    with pytest.raises(ValueError,match='escapes'):
        render_plot_state(path)
    plt.close('all')
