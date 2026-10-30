import json
from zipfile import ZipFile
import pytest
from opera.run.package_figure_review import package
from opera.run.clinical_figure_provenance import render_receipt


def test_partial_bundle_keeps_figures_and_aggregates_without_patient_data(tmp_path):
    review = tmp_path / 'review'
    (review / 'clinical_scores/cll').mkdir(parents=True)
    (review / 'review_manifest.json').write_text(json.dumps({'status':'partial_review','missing':[{'family':'IG'}]}))
    (review / 'clinical_scores/cll/risk.pdf').write_bytes(b'pdf')
    (review / 'clinical_scores/report_manifest.json').write_text(json.dumps(render_receipt(
        review / 'clinical_scores', [dict(id='cll', model='R_L', threshold_split='train')], 'tuning')))
    (review / 'patients_INTERNAL.csv').write_text('patient_id,risk\n123,0.4\n')
    (review / 'embeddings.npz').write_bytes(b'private')
    root = tmp_path / 'analysis'
    plots = root / 'figure_inputs/plot_tables'
    plots.mkdir(parents=True)
    (plots / 'panel_effects.csv').write_text('contrast,auc_gain\nO1,.01\n')
    (plots / 'model_cells.csv').write_text('patient_id,risk\n123,.4\n')
    destination = tmp_path / 'bundle.zip'
    result = package(review, destination, root)
    with ZipFile(destination) as archive:
        names = archive.namelist()
        assert 'review/clinical_scores/cll/risk.pdf' in names
        assert 'aggregates/figure_inputs/plot_tables/panel_effects.csv' in names
        assert not any('INTERNAL' in x or x.endswith('.npz') or 'model_cells.csv' in x for x in names)
        manifest = json.loads(archive.read('bundle_manifest.json'))
        assert manifest['missing'] == [{'family':'IG'}]
        assert len(manifest['files']['review/clinical_scores/cll/risk.pdf']['sha256']) == 64
    assert result['status'] == 'partial_review'
    with pytest.raises(FileExistsError):
        package(review, destination)


def test_rejects_unassembled_workspace(tmp_path):
    with pytest.raises(ValueError, match='assembled review'):
        package(tmp_path, tmp_path / 'bad.zip')


def test_rejects_old_clinical_figures_even_in_an_existing_review(tmp_path):
    (tmp_path / 'review_manifest.json').write_text('{}')
    (tmp_path / 'clinical_scores').mkdir()
    (tmp_path / 'clinical_scores/old.pdf').write_bytes(b'old')
    with pytest.raises(ValueError, match='stale clinical figures'):
        package(tmp_path, tmp_path / 'bad.zip')
    assert not (tmp_path / 'bad.zip').exists()


def test_held_out_package_keeps_the_declared_split_and_only_its_borrowing_outputs(tmp_path):
    review = tmp_path / 'review'
    clinical = review / 'clinical_scores'
    clinical.mkdir(parents=True)
    (review / 'review_manifest.json').write_text(json.dumps(dict(
        status='partial_review', evaluation_split='held_out')))
    (clinical / 'score.pdf').write_bytes(b'held-out clinical figure')
    (clinical / 'report_manifest.json').write_text(json.dumps(render_receipt(
        clinical, [dict(id='score', model='R_L', threshold_split='train')], 'held_out')))
    root = tmp_path / 'analysis'
    for split in ('tuning', 'held_out'):
        directory = root / 'source_removal' / split
        directory.mkdir(parents=True)
        (directory / 'borrowing_map.csv').write_text('source,gain\nA,.01\n')
    destination = tmp_path / 'held_out.zip'
    package(review, destination, root)
    with ZipFile(destination) as archive:
        names = archive.namelist()
        assert 'aggregates/source_removal/held_out/borrowing_map.csv' in names
        assert not any('/source_removal/tuning/' in name for name in names)
        assert json.loads(archive.read('bundle_manifest.json'))['evaluation_split'] == 'held_out'
