import json

import numpy as np
import pytest

from opera.run.audit_router_vectors import audit


def export(root, outcome, adapted, *, ids=(1, 2, 3), checkpoint='fixed.ckpt'):
    directory = root / outcome
    directory.mkdir()
    path = directory / 'routed_patient_vectors_INTERNAL.npz'
    np.savez(path, subject_ids=ids, shared=np.zeros_like(adapted), adapted=adapted)
    path.with_suffix('.json').write_text(json.dumps(dict(target_outcome=outcome,
        checkpoint=checkpoint, split='tuning')))


def test_raw_vector_audit_aligns_patients_and_detects_equal_corrections(tmp_path):
    delta = np.arange(6).reshape(3, 2) + 1.
    export(tmp_path, 'a', delta)
    export(tmp_path, 'b', delta[::-1], ids=(3, 2, 1))
    result = audit(tmp_path)
    assert result['pairs'][0]['exactly_equal']
    assert result['pairs'][0]['median_cosine'] == pytest.approx(1.)
    assert result['outcome_specific_energy_fraction'] == 0.
    assert 'subject_ids' not in json.dumps(result)
    assert len(result['inputs']) == 2


def test_raw_vector_audit_measures_outcome_specific_corrections(tmp_path):
    export(tmp_path, 'a', np.array([[1., 0.]] * 3))
    export(tmp_path, 'b', np.array([[0., 1.]] * 3))
    result = audit(tmp_path)
    assert not result['pairs'][0]['exactly_equal']
    assert result['pairs'][0]['median_cosine'] == 0.
    assert result['outcome_specific_energy_fraction'] == pytest.approx(.5)


@pytest.mark.parametrize('kwargs', [dict(ids=(1, 2, 4)), dict(checkpoint='wrong.ckpt')])
def test_raw_vector_audit_rejects_incompatible_exports(tmp_path, kwargs):
    export(tmp_path, 'a', np.ones((3, 2)))
    export(tmp_path, 'b', np.ones((3, 2)), **kwargs)
    with pytest.raises(ValueError, match='differ'):
        audit(tmp_path)


def test_raw_vector_audit_handles_zero_corrections_without_fabricated_cosines(tmp_path):
    export(tmp_path, 'a', np.zeros((3, 2)))
    export(tmp_path, 'b', np.zeros((3, 2)))
    result = audit(tmp_path)
    assert result['pairs'][0]['median_cosine'] is None
    assert result['outcome_specific_energy_fraction'] is None
    json.dumps(result, allow_nan=False)
