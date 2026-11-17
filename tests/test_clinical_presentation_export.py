import json
from zipfile import ZipFile

import pytest


def test_clinical_zip_retains_validated_editable_state_and_rejects_patient_fields(tmp_path):
    from opera.run.refresh_clinical_figures import package_clinical
    from opera.visualization.plot_state import SCHEMA
    source = tmp_path / 'clinical'
    source.mkdir()
    state = source / 'risk_groups.plot.json'
    state.write_text(json.dumps({'schema': SCHEMA, 'axes': [], 'texts': [], 'inches': [7, 5]}))
    archive = tmp_path / 'review.zip'
    package_clinical(source, archive)
    with ZipFile(archive) as z:
        assert 'risk_groups.plot.json' in z.namelist()
    state.write_text(json.dumps({'schema': SCHEMA, 'axes': [], 'patient_rows': [1]}))
    with pytest.raises(ValueError, match='Patient data'):
        package_clinical(source, tmp_path / 'bad.zip')


def test_remaining_input_export_keeps_private_views_and_patient_tables_out(tmp_path, monkeypatch):
    import importlib.util
    from pathlib import Path
    import yaml
    import opera.run.build_paper_plot_tables as builder
    script = Path(__file__).resolve().parents[1] / 'scripts/export_remaining_paper_inputs.py'
    spec = importlib.util.spec_from_file_location('remaining_export', script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    project = tmp_path / 'data/paper_256_v3_national_20261111/extended_test_comparison'
    results = project / 'results'
    results.mkdir(parents=True)
    (results / 'analysis_manifest.json').write_text('{}')
    (project / 'analysis.yaml').write_text(yaml.safe_dump({'cell_metadata': 'metadata.csv'}))
    (results / 'contrasts.csv').write_text('contrast,gain\nA,0.01\n')
    (results / 'patient_subgroup_panels.csv').write_text('subject_id,gain\n123,0.1\n')

    def build(view, metadata, tables, **kwargs):
        assert (view / 'prediction_analysis').resolve() == results
        tables.mkdir(parents=True)
        (tables / 'model_cells.csv').write_text('cohort,outcome,J\nA,x,0.8\n')
        return {'tables': {'model_cells.csv': {'status': 'written'}}}

    monkeypatch.setattr(builder, 'build', build)
    output = tmp_path / 'export'
    monkeypatch.setattr('sys.argv', ['export', '--repo-root', str(tmp_path), '--output-dir', str(output)])
    module.main()
    with ZipFile(output.with_suffix('.zip')) as z:
        names = z.namelist()
        assert 'extended/plot_tables/model_cells.csv' in names
        assert 'extended/aggregates/contrasts.csv' in names
        assert not any('INTERNAL' in name or 'patient_subgroup' in name for name in names)
