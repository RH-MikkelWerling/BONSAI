import json
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch
import yaml


def test_cpu_map_vectors_reproduce_production_readout_and_risk_in_requested_order():
    from test_frozen_router_mechanism import fixture_head
    from opera.run.refresh_routed_map_vectors import replay_vectors
    model, head = fixture_head()
    x, fine, indices = torch.randn(6, 16), torch.tensor([0, 1, 0, 1, 0, 1]), torch.tensor([1, 0])
    vectors, risks = replay_vectors(head, x, fine, indices, 365, batch_size=2)
    with torch.inference_mode():
        np.testing.assert_allclose(vectors, model.source_adapter_readout(x, fine)[:, indices].numpy(), atol=1e-6)
        flat = model.competing_risk_log_hazards(x, fine).flatten(2)[:, indices]
        np.testing.assert_allclose(risks, head.cif(flat, indices, [365])[..., 0].numpy(), atol=1e-6)


def test_event_capture_counts_are_paired_and_zero_when_classifications_agree():
    from opera.run.render_clinical_score_panels import clinical_detection_counts
    patients = pd.DataFrame(dict(subject_id=['a', 'b', 'c', 'd'], model_risk=[.9, .8, .1, .2], ipi_score=[7, 6, 1, 0],
        event=[1, 1, 1, 0], time_days=[10., 20., 30., 500.]))
    spec = dict(id='nccn', score_label='NCCN-IPI', outcome='overall_survival',
        horizon_days=365, score_column='score', score_high_threshold=6)
    same = clinical_detection_counts(patients, spec, dict(high_cutoff=.5), draws=100)
    assert all(row['additional_events_per_1000'] == row['lower'] == row['upper'] == 0 for row in same)
    patients.loc[1, 'model_risk'] = .1
    different = clinical_detection_counts(patients, spec, dict(high_cutoff=.5), draws=100)
    assert all(row['additional_events_per_1000'] == -250 for row in different)
    assert all(row['lower'] <= -250 <= row['upper'] for row in different)
    assert clinical_detection_counts(patients, spec, {}) == []


def test_empty_development_does_not_block_outcome_blind_clinical_groups():
    from opera.run.render_clinical_score_panels import independent_clinical_anchors
    empty = pd.DataFrame(columns=['model_risk', 'ipi_score', 'ipi_group', 'time_days', 'event'])
    result = independent_clinical_anchors(empty, dict(horizon_days=730,
        death_in_endpoint=True, group_order=['Low', 'High']))
    assert result['high_reason'] == 'No development controls'
    assert result['km_low_status'] == 'not_estimable'


def test_refresh_remains_cpu_only_preserves_source_and_reports_partial_results(tmp_path, monkeypatch):
    from opera.run import refresh_reviewed_figures as refresh
    source, output = tmp_path / 'source', tmp_path / 'output'
    (source / 'analysis_inventory').mkdir(parents=True)
    (source / 'figure_inputs/plot_tables').mkdir(parents=True)
    (source / 'figure_inputs/plot_tables/atlas.csv').write_text('unchanged\n')
    (source / 'analysis_inventory/analysis.yaml').write_text(yaml.safe_dump(dict(
        artifacts=[dict(model='R_L', split='tuning')], cell_metadata='/frozen/cells.csv')))
    (source / 'completion.json').write_text('{}')
    (source / 'overnight.env').write_text('OPERA_PAPER_CELL_METADATA=/frozen/cells.csv\n')
    (source / 'prediction_analysis').mkdir()
    before = {str(p): p.read_bytes() for p in source.rglob('*') if p.is_file()}
    calls = []
    def execute(command, **kwargs):
        calls.append(command)
        assert kwargs['env']['CUDA_VISIBLE_DEVICES'] == ''
        if command[3].endswith('assemble_figure_review'):
            folder = output / 'review'
            folder.mkdir()
            (folder / 'review_manifest.json').write_text('{"status":"partial_review"}')
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(refresh.subprocess, 'run', execute)
    monkeypatch.setattr(sys, 'argv', ['refresh', '--analysis-root', str(source), '--output-dir', str(output)])
    refresh.main()
    report = json.loads((output / 'refresh_report.json').read_text())
    assert report['status'] == 'finished_with_gaps'
    assert len(calls) == 4
    assert before == {str(p): p.read_bytes() for p in source.rglob('*') if p.is_file()}
    assert not any('held_out' in str(c) or 'evaluate_direct_competing_risk' in str(c) for c in calls)


def test_paged_pooling_maps_keep_identical_scales_and_name_every_outcome():
    import matplotlib.pyplot as plt
    from opera.run.render_paper_figure_prototypes import synthetic_cells
    from opera.visualization.paper_figures import figure_pooling_cells
    cells = synthetic_cells(np.random.default_rng(2026))
    images = []
    for contrast in ('D-S', 'O-S', 'J-S', 'interaction'):
        fig = figure_pooling_cells(cells, contrast=contrast)
        ax = next(ax for ax in fig.axes if ax.images)
        images.append(ax.images[0].get_clim())
        assert len(ax.get_xticklabels()) == cells.outcome.nunique()
        assert all(t.get_text() for t in ax.get_xticklabels())
        plt.close(fig)
    assert all(limits == images[0] for limits in images)


def test_clinical_counts_explain_missing_low_anchor_without_layout_collisions():
    import matplotlib.pyplot as plt
    from opera.visualization.paper_figures import figure_clinical_operating_points
    from opera.visualization.layout_audit import audit_layout
    detection = pd.DataFrame([dict(score_id='nccn', label='NCCN-IPI', estimand='cif_ipcw',
        additional_events_per_1000=10., lower=-5., upper=25.)])
    fig = figure_clinical_operating_points(pd.DataFrame(), detection)
    assert any(t.get_text() == 'Not estimable' for t in fig.axes[0].texts)
    findings = audit_layout(fig)
    assert not findings['legend_overflow']
    assert not findings['text_collision_candidates']
    plt.close(fig)


def test_routing_and_support_figures_prefer_selected_alias_to_legacy_router():
    import matplotlib.pyplot as plt
    from opera.visualization.paper_figures import figure_routing, figure_support_sensitivity
    cells = pd.DataFrame([dict(cohort='CLL', outcome='overall_survival',
        endpoint_group='Disease course', horizon_days=365, metric='auc', contrast=code,
        gain=gain, lower=gain-.01, upper=gain+.01, n_cases=20, n_controls=50)
        for code, gain in [('J-S', .04), ('R1', -.2), ('OPERA-J', .1)]])
    fig = figure_routing(cells, pd.DataFrame(columns=['contrast']), horizon=365)
    np.testing.assert_allclose(fig.axes[0].collections[0].get_offsets()[:, 0], [.1])
    plt.close(fig)
    fig = figure_support_sensitivity(cells)
    assert fig.axes[0].lines[1].get_ydata()[0] == pytest.approx(.1)
    plt.close(fig)
