import json
import zipfile

import numpy as np
import pandas as pd
import pytest
import torch

from opera.run import audit_router_geometry as audit
from test_frozen_router_mechanism import fixture_head


def arrays(global_vector, routed):
    routed = np.asarray(routed, dtype=float)
    general = np.broadcast_to(np.asarray(global_vector, float), (len(routed), routed.shape[-1])).copy()
    return dict(total=routed + general[:, None], routed=routed, **{'global': general})


def test_exact_global_removal_can_reveal_different_routing_hidden_by_common_direction():
    values = arrays([100., 100.], [[[1., 0.], [0., 1.]]] * 4)
    risks = np.array([[.1, .9], [.2, .8], [.3, .7], [.4, .6]])
    energy, outcomes, pairs = audit.summarise(values, risks, np.ones((4, 2), bool), ['a', 'b'], minimum_common=2)
    assert pairs.iloc[0].total_median_cosine > .999
    assert pairs.iloc[0].routed_median_cosine == 0.
    assert pairs.iloc[0].predicted_risk_spearman == pytest.approx(-1)
    assert energy['routed_without_global']['outcome_specific_energy_fraction'] == pytest.approx(.5)
    assert energy['total']['outcome_specific_energy_fraction'] < .0001
    assert list(outcomes.n_patients) == [4, 4]
    assert energy['energy_identity_error'] < 1e-8


def test_cancellation_prevents_reading_energy_ratios_as_performance_shares():
    values = arrays([10., 0.], [[[-9., 1.], [-9., -1.]]] * 4)
    energy, _, _ = audit.summarise(values, np.ones((4, 2)), np.ones((4, 2), bool), ['a', 'b'], minimum_common=2)
    assert energy['global_energy'] > energy['total']['total_energy']
    assert energy['global_routed_cross_term'] < 0
    assert energy['energy_identity_error'] == 0


def test_pairs_require_common_eligibility_and_constant_risks_are_not_called_dissimilar():
    values = arrays([0., 0.], [[[1., 0.], [1., 0.]]] * 4)
    eligible = np.array([[1, 1], [1, 1], [1, 0], [0, 1]], bool)
    _, outcomes, pairs = audit.summarise(values, np.ones((4, 2)), eligible, ['a', 'b'], minimum_common=2)
    assert pairs.iloc[0].n_common == 2
    assert pairs.iloc[0].predicted_risk_spearman is None
    assert list(outcomes.n_patients) == [3, 3]
    _, _, sparse = audit.summarise(values, np.ones((4, 2)), eligible, ['a', 'b'], minimum_common=3)
    assert sparse.iloc[0].status == 'insufficient_common_patients'
    assert sparse.dropna(subset=['predicted_risk_spearman']).empty


def test_zero_corrections_have_no_fabricated_direction_or_energy_fraction():
    values = arrays([0., 0.], np.zeros((4, 2, 2)))
    energy, _, pairs = audit.summarise(values, np.ones((4, 2)), np.ones((4, 2), bool), ['a', 'b'], minimum_common=2)
    assert energy['total']['outcome_specific_energy_fraction'] is None
    assert pairs.iloc[0].total_median_cosine is None
    json.dumps(energy, allow_nan=False)


def test_production_replay_and_exact_global_source_split_are_batch_independent():
    _, head = fixture_head()
    x, fine, indices = torch.randn(8, 16), torch.tensor([0, 1] * 4), torch.tensor([1, 0])
    parts, risks, error = audit.reconstruct(head, x, fine, indices, 365, batch_size=3)
    whole, whole_risks, _ = audit.reconstruct(head, x, fine, indices, 365, batch_size=8)
    assert error < 2e-5
    for component in parts:
        np.testing.assert_allclose(parts[component], whole[component], rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(risks, whole_risks, atol=1e-6)
    terms, _ = head.terms(x, fine, indices)
    for o in range(2):
        projected = parts['global'] @ head.weight[indices[o]].numpy().T
        np.testing.assert_allclose(projected, terms['global'][:, o].numpy(), rtol=1e-5, atol=1e-6)


def test_energy_uses_within_patient_means_and_excludes_ineligible_outcomes():
    values = np.array([[[1., 0.], [0., 1.]], [[2., 3.], [999., 999.]]])
    eligible = np.array([[1, 1], [1, 0]], bool)
    energy = audit.outcome_energy(values, eligible)
    assert energy['total_energy'] == 15
    assert energy['outcome_specific_energy_fraction'] == pytest.approx(1 / 15)
    assert energy['patients_with_two_or_more_outcomes'] == 1


def test_cli_exports_only_aggregates_and_checks_saved_probabilities(tmp_path, monkeypatch):
    _, head = fixture_head()
    x, fine, indices = torch.randn(8, 16), torch.tensor([0, 1] * 4), torch.tensor([0, 1])
    _, risk, _ = audit.reconstruct(head, x, fine, indices, 365)
    row_index = (np.repeat(np.arange(8), 2), np.tile(np.arange(2), 8), np.zeros(16, int))
    baseline = pd.DataFrame(dict(probability=risk[row_index[:2]]))
    fit = dict(checkpoint='selected.ckpt', checkpoint_sha256='selected-hash')
    monkeypatch.setattr(audit, 'selected_inputs', lambda *a: ({}, {}, fit))
    monkeypatch.setattr(audit.FrozenRouterHead, 'from_checkpoint', lambda *a: head)
    monkeypatch.setattr(audit, 'load_vectors', lambda *a: (None, None, tmp_path / 'vectors.npz', tmp_path / 'marker.json'))
    monkeypatch.setattr(audit, 'prepare_population', lambda *a, **k:
                        (baseline, x, fine, torch.ones(8, 2, dtype=torch.bool), list(head.outcomes), [365], row_index))
    output = tmp_path / 'geometry'
    monkeypatch.setattr('sys.argv', ['audit', '--analysis-root', str(tmp_path), '--output-dir', str(output), '--minimum-common', '2'])
    audit.main()
    report = json.loads((output / 'report.json').read_text())
    assert report['reconstruction']['passed']
    with zipfile.ZipFile(output / 'router_geometry_review.zip') as archive:
        assert set(archive.namelist()) == {'report.json', 'outcome_geometry.csv', 'outcome_pairs.csv', 'different_risk_patterns.csv'}
        assert all('subject_id' not in archive.read(name).decode() for name in archive.namelist())
    with pytest.raises(FileExistsError):
        audit.main()
