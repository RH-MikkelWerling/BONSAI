import json
from pathlib import Path
from zipfile import ZipFile
import pytest
import yaml
from opera.run.prepare_offline_figure_bundle import portable_recipe, bundle_files, copy_presentation
from opera.run.package_figure_review import package
from opera.run.render_offline_bundle import extract_offline, render


def test_portable_recipe_copies_only_aggregate_tables_and_removes_server_paths(tmp_path):
    source=tmp_path/'server';source.mkdir()
    table=source/'model_cells.csv';table.write_text('cohort,outcome,J\nDLBCL,OS,.8\n')
    offline=tmp_path/'offline';offline.mkdir()
    plate=dict(id='overview',function='figure_model_overview',tables=dict(models=str(table)))
    recipe,reasons=portable_recipe(plate,source/'original.yaml',offline,{})
    assert not reasons
    assert recipe['tables']['models'].startswith('../inputs/')
    assert str(source) not in yaml.safe_dump(recipe)
    assert len(list((offline/'inputs').glob('*.csv')))==1
    table.write_text('subject_id,model_risk\nprivate-person,.8\n')
    _,reasons=portable_recipe(plate,source/'original.yaml',offline,{})
    assert 'not an approved aggregate' in reasons[0]


def test_package_includes_only_referenced_validated_offline_payload(tmp_path):
    review=tmp_path/'review';review.mkdir()
    (review/'review_manifest.json').write_text(json.dumps(dict(status='partial_review',evaluation_split='tuning')))
    offline=tmp_path/'offline';(offline/'inputs').mkdir(parents=True)
    (offline/'recipes/main').mkdir(parents=True)
    (offline/'inputs/0123456789abcdef_model_cells.csv').write_text('cohort,outcome,J\nDLBCL,OS,.8\n')
    (offline/'inputs/secret.csv').write_text('subject_id,risk\nprivate-person,.8\n')
    (offline/'recipes/main/overview.yaml').write_text(yaml.safe_dump(dict(figures=[dict(
        id='overview',function='figure_model_overview',tables=dict(models='../../inputs/0123456789abcdef_model_cells.csv'))])))
    manifest=dict(schema='opera.offline.v1',status='partial',evaluation_split='tuning',entries=[
        dict(id='main/overview',mode='native',recipe='recipes/main/overview.yaml'),
        dict(id='main/IG',mode='missing',reasons=['IG pending'])])
    (offline/'manifest.json').write_text(json.dumps(manifest))
    archive=tmp_path/'review.zip';package(review,archive,offline_dir=offline)
    with ZipFile(archive) as z:
        assert 'offline/recipes/main/overview.yaml' in z.namelist()
        assert not any('secret' in n for n in z.namelist())
        assert json.loads(z.read('bundle_manifest.json'))['offline_status']=='partial'
    extracted=tmp_path/'extracted';extract_offline(archive,extracted)
    assert len(bundle_files(extracted/'offline'))==3
    with pytest.raises(ValueError,match='missing analyses'):
        render(extracted,tmp_path/'iteration',require_complete=True)
    (offline/'inputs/0123456789abcdef_model_cells.csv').write_text('time_days,event\n10,1\n')
    with pytest.raises(ValueError,match='approved aggregate'):
        bundle_files(offline)


def test_presentation_rejects_patient_payload_and_zip_traversal(tmp_path):
    source=tmp_path/'private.plot.json'
    source.write_text(json.dumps(dict(schema='opera.presentation.v1',axes=[],subject_id='private-person')))
    with pytest.raises(ValueError,match='Patient data'):
        copy_presentation(source,tmp_path/'public/plot.json')
    archive=tmp_path/'bad.zip'
    with ZipFile(archive,'w') as z:z.writestr('offline/../../escape.txt','private')
    with pytest.raises(ValueError,match='Unsafe path'):
        extract_offline(archive,tmp_path/'extracted')


def test_repository_synthetic_illustration_is_portable_without_dummy_patient_id(tmp_path):
    from opera.run.prepare_offline_figure_bundle import REPO
    source=REPO/'paper/figures/synthetic_patient_episodes_risk.csv'
    plate=dict(id='illustration',function='figure_patient_tracks',tables=dict(episodes=str(source)))
    result,reasons=portable_recipe(plate,tmp_path/'source.yaml',tmp_path/'offline',{})
    assert not reasons
    table=next((tmp_path/'offline/inputs').glob('*.csv'))
    assert 'subject_id' not in table.read_text().splitlines()[0]
    assert 'episode' in table.read_text().splitlines()[0]


def test_missing_rendered_figures_are_explicit_even_without_source_recipes(tmp_path):
    from opera.run.prepare_offline_figure_bundle import prepare
    review=tmp_path/'review';(review/'main').mkdir(parents=True)
    (review/'review_manifest.json').write_text(json.dumps(dict(status='partial_review',evaluation_split='tuning')))
    (review/'main/unregistered.png').write_bytes(b'previous publication figure')
    offline,manifest=prepare(review,tmp_path/'analysis',tmp_path/'prepared',clinical=False)
    assert manifest['status']=='partial'
    assert not manifest['complete_for_rendered_figures']
    assert manifest['missing_rendered_figures']==['main/unregistered']
