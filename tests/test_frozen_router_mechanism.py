import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
import yaml

from opera.diagnostics.router_mechanism import (
    COMPONENTS, VARIANTS, FrozenRouterHead, TaskReferences, variant_terms,
)
from opera.diagnostics.router_prediction_checks import component_cifs
from opera.run import audit_frozen_router_mechanism as audit
from test_router_axis_separation import _model


def fixture_head(*, borrowing=True):
    model = _model('both', adapter_global=True, adapter_borrowing=borrowing).eval()
    with torch.no_grad():
        model.source_adapter_readout.up.normal_(std=.12)
        model.source_adapter_readout.global_up.normal_(std=.12)
        model.source_adapter_readout.patient_query.weight.normal_(std=.2)
        model.source_adapter_readout.source_keys.normal_(std=2.)
        model.competing_risk_head.weight.normal_(std=.2)
    head = FrozenRouterHead(model.source_adapter_readout,
        model.competing_risk_head.weight, model.competing_risk_head.bias,
        [30, 365], no_competing_outcomes=[]).eval()
    return model, head


def references(head, x, fine, indices, eligible=None):
    if eligible is None:
        eligible = torch.ones(len(x), len(indices), dtype=torch.bool)
    ref = TaskReferences(head, len(indices))
    terms, gates = head.terms(x, fine, indices)
    ref.add(fine, eligible, terms, gates, head.borrowed_mask(fine, indices))
    return ref.finalize()


@pytest.mark.parametrize('indices', [[0, 1], [1], [1, 0]])
def test_terms_reconstruct_real_forward_and_original_component_checks(indices):
    model, head = fixture_head()
    x, fine, idx = torch.randn(6, 16), torch.tensor([0, 1, 1, 0, 1, 0]), torch.tensor(indices)
    terms, gates = head.terms(x, fine, idx)
    assert set(terms) == {'base', *COMPONENTS}
    production = model.competing_risk_log_hazards(x, fine).flatten(2)[:, idx]
    torch.testing.assert_close(sum(terms.values()), production)
    ref = references(head, x, fine, idx)
    variants = dict(variant_terms(head, x, fine, idx, ref))
    original = component_cifs(model, x, fine, [30, 365])
    torch.testing.assert_close(head.cif(variants['ENCODER_HEAD_ONLY'], idx, [30, 365]),
                               original['encoder_only'][:, idx])
    torch.testing.assert_close(head.cif(variants['ZERO_ALL_BORROWED'], idx, [30, 365]),
                               original['no_borrowing_zero'][:, idx])
    assert set(variants) == {'FULL', *VARIANTS}
    assert torch.isfinite(head.cif(sum(terms.values()), idx, [30, 365])).all()


def test_frozen_borrowed_composition_retains_budget_home_and_null():
    _, head = fixture_head()
    x, fine, idx = torch.randn(12, 16), torch.tensor([0, 1] * 6), torch.tensor([0, 1])
    ref = references(head, x, fine, idx)
    _, original = head.terms(x, fine, idx)
    mask = head.borrowed_mask(fine, idx)
    frozen = ref.frozen_composition(fine, original, mask)
    torch.testing.assert_close(frozen[..., 0], original[..., 0], rtol=0, atol=0)
    torch.testing.assert_close(frozen[..., 1:][~mask], original[..., 1:][~mask], rtol=0, atol=0)
    torch.testing.assert_close((frozen[..., 1:] * mask).sum(-1), (original[..., 1:] * mask).sum(-1))
    torch.testing.assert_close(frozen.sum(-1), torch.ones_like(frozen[..., 0]))
    assert not torch.allclose(frozen, original)


def test_zero_borrowing_and_unobserved_reference_slots_do_not_create_borrowing():
    _, head = fixture_head(borrowing=False)
    x, fine, idx = torch.randn(8, 16), torch.tensor([0, 1] * 4), torch.tensor([0, 1])
    eligible = torch.ones(8, 2, dtype=torch.bool)
    eligible[fine == 1, 1] = False
    ref = references(head, x, fine, idx, eligible)
    variants = dict(variant_terms(head, x, fine, idx, ref))
    torch.testing.assert_close(variants['TASK_MEAN_BORROWED_COMPOSITION'], variants['FULL'])
    torch.testing.assert_close(variants['ZERO_ALL_BORROWED'], variants['FULL'])
    assert ref.borrow_count.sum() == 0
    assert all(torch.isfinite(v).all() for v in variants.values())


def test_mean_ablation_keeps_eligible_task_mean_head_outputs():
    _, head = fixture_head()
    x, fine, idx = torch.randn(12, 16), torch.tensor([0, 1] * 6), torch.tensor([0, 1])
    eligible = torch.ones(12, 2, dtype=torch.bool)
    eligible[:3, 1] = False
    ref = references(head, x, fine, idx, eligible)
    variants = dict(variant_terms(head, x, fine, idx, ref))
    for d in range(2):
        for o in range(2):
            keep = (fine == d) & eligible[:, o]
            for name in VARIANTS:
                if name.startswith('MEAN_'):
                    torch.testing.assert_close(variants[name][keep, o].mean(0),
                                               variants['FULL'][keep, o].mean(0), atol=1e-6, rtol=1e-5)


def test_references_are_independent_of_batching_and_exclude_ineligible_rows():
    _, head = fixture_head()
    x, fine, idx = torch.randn(10, 16), torch.tensor([0, 1] * 5), torch.tensor([0, 1])
    e = torch.ones(10, 2, dtype=torch.bool)
    e[1, 0] = False
    whole = references(head, x, fine, idx, e)
    parts = TaskReferences(head, 2)
    for i in range(0, 10, 3):
        terms, gates = head.terms(x[i:i+3], fine[i:i+3], idx)
        parts.add(fine[i:i+3], e[i:i+3], terms, gates, head.borrowed_mask(fine[i:i+3], idx))
    parts.finalize()
    torch.testing.assert_close(parts.gates, whole.gates)
    torch.testing.assert_close(parts.composition, whole.composition)
    assert parts.count[1, 0] == 4
    assert parts.count[1, 1] == 5
    for name in COMPONENTS:
        torch.testing.assert_close(parts.terms[name], whole.terms[name])


def test_reconstruction_loader_preserves_source_order_support_and_head(tmp_path):
    model, head = fixture_head()
    path = tmp_path / 'best.ckpt'
    with torch.no_grad():
        model.source_adapter_readout.source_train_events[0] = 2**24 + 1
    torch.save({'state_dict': {'model.' + k: v for k, v in model.state_dict().items()},
                'hyper_parameters': {'outcome_names': ['x', 'y'],
                                     'model_init_config': model.model_init_config}}, path)
    loaded = FrozenRouterHead.from_checkpoint(path)
    assert loaded.readout.fine_names == ('A', 'B')
    assert loaded.readout.source_train_events[0] == 2**24 + 1
    x, fine, idx = torch.randn(4, 16), torch.tensor([0, 1, 0, 1]), torch.tensor([0, 1])
    before, _ = head.terms(x, fine, idx)
    after, _ = loaded.terms(x, fine, idx)
    for name in before:
        torch.testing.assert_close(before[name], after[name], rtol=0, atol=0)


def test_no_competing_head_is_ignored_in_cif():
    model, _ = fixture_head()
    head = FrozenRouterHead(model.source_adapter_readout,
        model.competing_risk_head.weight, model.competing_risk_head.bias,
        [30, 365], no_competing_outcomes=['x'])
    x, fine, idx = torch.randn(4, 16), torch.tensor([0, 1, 0, 1]), torch.tensor([0, 1])
    terms, _ = head.terms(x, fine, idx)
    flat = sum(terms.values())
    changed = flat.clone()
    changed[:, :, 3:] += 10
    torch.testing.assert_close(head.cif(flat, idx, [365])[:, 0], head.cif(changed, idx, [365])[:, 0])
    assert not torch.allclose(head.cif(flat, idx, [365])[:, 1], head.cif(changed, idx, [365])[:, 1])


@pytest.mark.parametrize('variant,zero_sources,zero_global', [
    ('ZERO_BORROWED_DISEASE', [1], False),
    ('ZERO_BORROWED_OUTCOME', [3], False),
    ('ZERO_OWN', [0, 2], False),
    ('ZERO_GLOBAL', [], True),
    ('GLOBAL_ONLY', [0, 1, 2, 3], False),
])
def test_individual_bank_interventions_match_explicit_production_source_zeroing(
        variant, zero_sources, zero_global):
    model, head = fixture_head()
    x, fine, idx = torch.randn(1, 16), torch.tensor([0]), torch.tensor([0])
    ref = references(head, x, fine, idx)
    before = dict(variant_terms(head, x, fine, idx, ref))[variant].clone()
    gates = model.source_adapter_readout.route_weights(x, fine).clone()
    with torch.no_grad():
        model.source_adapter_readout.up[zero_sources] = 0
        if zero_global:
            model.source_adapter_readout.global_up.zero_()
    torch.testing.assert_close(model.source_adapter_readout.route_weights(x, fine), gates, rtol=0, atol=0)
    production = model.competing_risk_log_hazards(x, fine).flatten(2)[:, idx]
    torch.testing.assert_close(before, production)


def make_workspace(tmp_path):
    model, head = fixture_head()
    root = tmp_path / 'analysis'
    (root / 'analysis_inventory').mkdir(parents=True)
    pred = tmp_path / 'predictions/tuning'
    pred.mkdir(parents=True)
    checkpoint = tmp_path / 'best.ckpt'
    torch.save({'state_dict': {'model.' + k: v for k, v in model.state_dict().items()},
                'hyper_parameters': {'outcome_names': ['x', 'y'],
                                     'model_init_config': model.model_init_config}}, checkpoint)
    ids = np.array([f'p{i:03}' for i in range(48)])
    x = torch.randn(48, 16)
    fine = torch.tensor([0, 1] * 24)
    idx = torch.tensor([0, 1])
    terms, _ = head.terms(x, fine, idx)
    scores = head.cif(sum(terms.values()), idx, [30, 365]).numpy()
    rows = []
    for p, subject in enumerate(ids):
        for o, outcome in enumerate(['x', 'y']):
            event = [1, 1, 1, 1, 1, 1, 2, 0, 0, 0, 0, 0][(p // 2 + o) % 12]
            time = 10 if event else (20 if p % 7 == 0 else 500)
            rows.append(dict(subject_id=subject, cohort=['A', 'B'][fine[p]], outcome=outcome,
                             time_days=time, event=event, cif_30d=scores[p, o, 0], cif_365d=scores[p, o, 1]))
    pd.DataFrame(rows).to_parquet(pred / 'direct_cr_predictions.parquet', index=False)
    metadata = tmp_path / 'patients.parquet'
    pd.DataFrame(dict(subject_id=ids, cohort_fine=[['A', 'B'][f] for f in fine])).to_parquet(metadata)
    np.savez(pred / 'patient_embeddings.npz', subject_ids=ids[::-1], embeddings=x.numpy()[::-1],
             splits=np.full(len(ids), 'tuning'))
    (pred / 'patient_embeddings.metadata.json').write_text(json.dumps(
        dict(checkpoint=str(checkpoint), split='tuning')))
    spec = dict(model='R_L', split='tuning', seed=42, format='direct_cif',
                path=str(pred / 'direct_cr_predictions.parquet'),
                cohort_from_metadata=True, data_contract='synthetic-release')
    joint_path = pred / 'joint.parquet'
    joint = pd.DataFrame(rows)
    joint[['cif_30d', 'cif_365d']] *= .9
    joint.to_parquet(joint_path, index=False)
    config = dict(estimand='cif_ipcw', patient_metadata=str(metadata), opera_model='R_L',
                  performance_min_cases=5, performance_min_controls=5,
                  artifacts=[spec, dict(spec, model='J_L', path=str(joint_path))])
    (root / 'analysis_inventory/analysis.yaml').write_text(yaml.safe_dump(config))
    (root / 'completion.json').write_text(json.dumps(dict(inputs=dict(fit=dict(
        predictions=spec['path'], checkpoint=str(checkpoint),
        checkpoint_sha256=audit.file_hash(checkpoint))))))
    return root, pred, config, spec


def test_full_cpu_audit_uses_existing_ipcw_and_exports_aggregate_only_zip(tmp_path, monkeypatch):
    import zipfile
    root, pred, _, _ = make_workspace(tmp_path)
    output = tmp_path / 'mechanism'
    actual_analysis = audit.paired_analysis
    resample_ids = []
    def checked_analysis(frame, contrasts, **kwargs):
        # Same full-risk-set censoring weights and the very same patient draws
        # for every intervention, not one resampling per model or outcome.
        resample_ids.append(id(kwargs['draws']))
        models = list(frame.model.unique())
        left = frame.loc[frame.model == models[0]].sort_values(
            ['subject_id', 'outcome', 'horizon_days'])
        right = frame.loc[frame.model == models[1]].sort_values(
            ['subject_id', 'outcome', 'horizon_days'])
        np.testing.assert_array_equal(left.ipcw_weight, right.ipcw_weight)
        assert (left.ipcw_weight == 0).any()
        assert (left.ipcw_weight > 1).any()
        assert kwargs['min_cases'] == kwargs['min_controls'] == 5
        return actual_analysis(frame, contrasts, **kwargs)
    monkeypatch.setattr(audit, 'paired_analysis', checked_analysis)
    monkeypatch.setenv('OPERA_ANALYSIS_WORKERS', '1')
    monkeypatch.setattr('sys.argv', ['audit', '--analysis-root', str(root), '--output-dir', str(output),
                                  '--bootstrap', '50', '--threads', '1', '--batch-size', '13'])
    audit.main()
    report = json.loads((output / 'report.json').read_text())
    assert report['status'] == 'complete'
    assert report['reconstruction']['passed']
    assert report['n_patients'] == 48
    assert len(report['completed_interventions']) == len(VARIANTS)
    assert len(resample_ids) == len(VARIANTS) + 1
    assert len(set(resample_ids)) == 1
    summary = pd.read_csv(output / 'effect_summaries.csv')
    assert set(summary.weighting) == {'equal_cell', 'disease_balanced'}
    assert set(summary.variant) == set(VARIANTS)
    assert report['joint_path_comparison']['status'] == 'available'
    joint = pd.read_csv(output / 'joint_path_summaries.csv')
    assert set(joint.contrast) == {'FULL_VS_JOINT', 'ENCODER_HEAD_ONLY_VS_JOINT', 'GLOBAL_ONLY_VS_JOINT'}
    assert {'auc', 'brier', 'ipa'} <= set(summary.metric)
    assert summary.n_valid_bootstrap.min() == 50
    with zipfile.ZipFile(output / 'router_mechanism_review.zip') as bundle:
        assert 'references_INTERNAL.pt' not in bundle.namelist()
        for name in bundle.namelist():
            assert 'INTERNAL' not in name
            if name.endswith('.csv'):
                columns = pd.read_csv(bundle.open(name), nrows=0).columns
                assert not {'subject_id', 'probability', 'ipcw_weight'} & set(columns)
    assert (output / 'router_mechanism.pdf').is_file()
    assert (output / 'router_mechanism_probability.pdf').is_file()
    assert (output / 'router_mechanism_sensitivity.pdf').is_file()
    assert (output / 'index.html').is_file()
    with pytest.raises(FileExistsError):
        audit.main()
    # Styling can be refreshed from aggregate tables, without loading fits or
    # recomputing any scientific result, including after a server code sync.
    before = {p.name: p.read_bytes() for p in output.glob('*.csv')}
    original_report = (output / 'report.json').read_bytes()
    monkeypatch.setattr(audit, 'selected_inputs', lambda *a: pytest.fail('Render-only must not load fits'))
    monkeypatch.setattr(audit, 'paired_analysis', lambda *a, **kw: pytest.fail('Render-only must not recompute metrics'))
    monkeypatch.setattr('sys.argv', ['audit', '--output-dir', str(output), '--render-only'])
    audit.main()
    assert before == {p.name: p.read_bytes() for p in output.glob('*.csv')}
    assert original_report == (output / 'report.json').read_bytes()


def test_mechanism_overview_does_not_magnify_roundoff_or_invent_missing_joint_paths():
    import matplotlib.pyplot as plt
    rows = [dict(variant=v, stratum='all_cells', horizon_days=0, metric='auc',
                 gain=1e-17, lower=-1e-17, upper=2e-17) for v in VARIANTS]
    fig = audit.figure_router_mechanism(pd.DataFrame(rows), None)
    a, b, c = fig.axes
    assert b.get_xlim() == c.get_xlim()
    assert c.get_xlim()[1] >= .001
    assert sum(t.get_text() == 'Comparison unavailable' for t in a.texts) == 3
    plt.close(fig)


def test_audit_rejects_wrong_saved_predictions_before_reporting_interventions(tmp_path, monkeypatch):
    root, pred, _, _ = make_workspace(tmp_path)
    path = pred / 'direct_cr_predictions.parquet'
    data = pd.read_parquet(path)
    data.loc[0, 'cif_30d'] = .99
    data.to_parquet(path, index=False)
    output = tmp_path / 'mechanism'
    monkeypatch.setattr('sys.argv', ['audit', '--analysis-root', str(root), '--output-dir', str(output),
                                  '--bootstrap', '50', '--threads', '1'])
    with pytest.raises(ValueError, match='do not reconstruct'):
        audit.main()
    assert json.loads((output / 'report.json').read_text())['status'] == 'failed'
    assert not (output / 'effect_summaries.csv').exists()


def test_joint_comparison_rejects_changed_followup(tmp_path, monkeypatch):
    root, pred, _, _ = make_workspace(tmp_path)
    joint = pd.read_parquet(pred / 'joint.parquet')
    joint.loc[0, 'event'] = 0
    joint.to_parquet(pred / 'joint.parquet', index=False)
    output = tmp_path / 'mechanism'
    monkeypatch.setattr('sys.argv', ['audit', '--analysis-root', str(root), '--output-dir', str(output),
                                  '--bootstrap', '50', '--threads', '1'])
    with pytest.raises(ValueError, match='parity failed'):
        audit.main()
    assert json.loads((output / 'report.json').read_text())['status'] == 'failed'


def test_audit_never_selects_a_held_out_artifact(tmp_path):
    root = tmp_path / 'analysis'
    (root / 'analysis_inventory').mkdir(parents=True)
    (root / 'analysis_inventory/analysis.yaml').write_text(yaml.safe_dump(dict(
        estimand='cif_ipcw', opera_model='R_L', artifacts=[dict(model='R_L', seed=42,
            split='held_out', path='/never/open/test.parquet', format='direct_cif')])))
    with pytest.raises(ValueError, match='native tuning artifact'):
        audit.selected_inputs(root, 42)


def test_vector_loading_rejects_test_split_and_missing_patients(tmp_path):
    root, pred, config, spec = make_workspace(tmp_path)
    _, _, fit = audit.selected_inputs(root, 42)
    marker_path = pred / 'patient_embeddings.metadata.json'
    marker = json.loads(marker_path.read_text())
    marker_path.write_text(json.dumps(dict(marker, split='held_out')))
    with pytest.raises(ValueError, match='tuning-only'):
        audit.load_vectors(root, fit)
    marker_path.write_text(json.dumps(marker))
    ids, values, _, _ = audit.load_vectors(root, fit)
    with pytest.raises(ValueError, match='lack 1 required'):
        audit.prepare_population(config, spec, FrozenRouterHead.from_checkpoint(Path(fit['checkpoint'])),
                                 ids[1:], values[1:], outcomes=['all'], horizons=None)


def test_control_ladder_uses_common_supported_roster_without_aliases(tmp_path):
    root, output = tmp_path / 'analysis', tmp_path / 'report'
    (root / 'prediction_analysis').mkdir(parents=True)
    output.mkdir()
    rows = []
    for model, delta in [('J_L', 0), ('N_A_L', -.01), ('RBH_AB_L', .02), ('R_AB_L', .03)]:
        for outcome in ['x', 'y']:
            rows.append(dict(split='tuning', seed=42, cohort='A', outcome=outcome,
                horizon_days=365, model=model, metric='auc',
                estimate=np.nan if model == 'N_A_L' and outcome == 'y' else .7 + delta))
    # Unrelated aliases/held-out/seed rows must not enter this matched ladder.
    rows.extend([dict(rows[0], model='OPERA', estimate=1),
                 dict(rows[0], split='held_out', estimate=.1)])
    pd.DataFrame(rows).to_csv(root / 'prediction_analysis/estimates.csv', index=False)
    result = audit.trained_control_ladder(root, 42, [365], ['x', 'y'], output)
    assert result['coverage'][0]['n_common_cell_horizons'] == 1
    assert result['coverage'][0]['n_any_supported_cell_horizons'] == 2
    table = pd.read_csv(output / 'trained_control_ladder.csv')
    assert set(table.model) == {'J_L', 'N_A_L', 'RBH_AB_L', 'R_AB_L'}
    assert table.n_cell_horizons.eq(1).all()
    assert table.loc[table.model == 'R_AB_L', 'mean_auc_difference_vs_joint'].iloc[0] == pytest.approx(.03)


def test_risk_change_directions_are_explicit_and_include_competing_censoring():
    base = pd.DataFrame(dict(cohort=['A']*4, outcome=['x']*4, horizon_days=[365]*4,
        time_days=[10, 20, 500, 30], event=[1, 2, 0, 0], probability=[.1]*4))
    candidate = base.assign(probability=[.2, .1, .05, .1])
    table = audit.risk_change_table(base, candidate, 'TEST').set_index('status')
    assert set(table.index) == {'all', 'event', 'competing_death', 'event_free', 'censored'}
    assert table.loc['event', 'mean_risk_change'] == pytest.approx(.1)
    assert table.loc['event_free', 'mean_risk_change'] == pytest.approx(-.05)
    assert table.direction.eq('intervention_minus_full').all()
