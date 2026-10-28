import numpy as np
import pandas as pd
import pytest
from opera.run.prepare_available_plot_configs import build
from opera.run.render_clinical_score_panels import expand_index_scores


def fixture(tmp_path, category='High Risk'):
    metadata = pd.DataFrame(dict(subject_id=range(60), IPI=[category]*60,
        index=['IPI']*60, cohort_fine=['DLBCL']*60, age_at_index=[60]*60))
    meta = tmp_path / 'metadata.parquet'
    metadata.to_parquet(meta)
    prediction = tmp_path / 'direct_cr_predictions.parquet'
    pd.DataFrame(dict(outcome=['overall_survival', 'anemia_g2plus'])).to_parquet(prediction)
    np.savez(tmp_path / 'patient_embeddings.npz', embeddings=np.ones((60, 2)))
    return dict(patient_metadata=str(meta), opera_model='R_L', artifacts=[dict(
        path=str(prediction), model='R_L', split='tuning', seed=42)]), metadata


def test_available_configs_use_opera_and_harmonize_risk(tmp_path):
    analysis, metadata = fixture(tmp_path)
    embedding, clinical = build(analysis, tmp_path / 'recipes')
    assert embedding['ipi_group_map']['high risk'] == 'High'
    assert embedding['ipi_group_map']['low-intermediate'] == 'Intermediate'
    assert embedding['calibration_maps']['outcomes'] == ['overall_survival', 'anemia_g2plus']
    specs = expand_index_scores(clinical, metadata)
    assert len(specs) == 1
    assert specs[0]['model'] == 'R_L'
    assert specs[0]['cohorts'] == ['DLBCL']
    assert specs[0]['outcome'] == 'overall_survival'
    assert specs[0]['manuscript_figures']


def test_unknown_categories_fail_explicitly(tmp_path):
    analysis, _ = fixture(tmp_path, 'Unspecified strange category')
    with pytest.raises(ValueError, match='Unmapped IPI'):
        build(analysis, tmp_path / 'recipes')


def test_generated_clinical_recipe_passes_renderer_validation(tmp_path, monkeypatch):
    import yaml
    from opera.run.render_clinical_score_panels import main
    analysis, _ = fixture(tmp_path)
    build(analysis, tmp_path / 'recipes')
    path = tmp_path / 'analysis.yaml'
    path.write_text(yaml.safe_dump(analysis))
    monkeypatch.setattr('sys.argv', ['clinical', '--analysis-config', str(path),
        '--score-config', str(tmp_path / 'recipes/clinical_score_panels.tuning.yaml'),
        '--output-dir', str(tmp_path / 'unused'), '--split', 'tuning'])
    main()
    assert not (tmp_path / 'unused').exists()


def test_requested_map_set_and_legacy_infection_name(tmp_path):
    analysis, _ = fixture(tmp_path)
    pd.DataFrame(dict(outcome=['infections_all_definitions', 'overall_survival',
        'anemia_g2plus', 'hospitalisation', 'creatinine_increased_g2plus',
        'neutropenia_g3plus'])).to_parquet(analysis['artifacts'][0]['path'])
    embedding, _ = build(analysis, tmp_path / 'recipes')
    assert embedding['calibration_maps']['outcomes'] == [
        'infections_all_definitions', 'overall_survival', 'anemia_g2plus',
        'hospitalisation', 'creatinine_increased_g2plus']


def test_recorded_index_compares_all_scored_patients_once(tmp_path):
    analysis, metadata = fixture(tmp_path)
    metadata.loc[30:, 'cohort_fine'] = 'FL'
    metadata.to_parquet(analysis['patient_metadata'])
    _, clinical = build(analysis, tmp_path / 'recipes')
    specs = expand_index_scores(clinical, metadata)
    assert len(specs) == 1
    assert set(specs[0]['cohorts']) == {'DLBCL', 'FL'}
    assert specs[0]['index_values'] == ['IPI']
    assert specs[0]['manuscript_figures']


def test_computed_ipi_unavailability_is_explicit(tmp_path):
    import json
    analysis, _ = fixture(tmp_path)
    build(analysis, tmp_path / 'recipes')
    audit = json.loads((tmp_path / 'recipes/config_audit.json').read_text())
    for name in ('IPI (computed)', 'NCCN-IPI'):
        assert audit['computed_score_availability'][name]['status'] == 'missing_columns'
        assert audit['computed_score_availability'][name]['missing_columns']
