"""Refresh clinical and embedding inputs from corrected predictions; no model fitting."""
import argparse
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import shutil
from zipfile import ZipFile, ZIP_DEFLATED, BadZipFile

import yaml

from opera.run.finish_figure_inputs import immutable, sha
from opera.run.cutoff_recovery_worker import read
from opera.run.cutoff_recovery_results import rebound_artifacts


def archive_ready(path):
    """A producer's manifest can precede its ZIP; require a closed archive too."""
    try:
        with ZipFile(path) as archive:
            return bool(archive.namelist()) and archive.testzip() is None
    except (OSError, BadZipFile, EOFError):
        return False


def find_artifact(artifacts, model, split):
    choices = {a['path']: a for a in artifacts if a['model'] == model and a.get('split') == split
               and int(a.get('seed', 42)) == 42}
    if len(choices) != 1:
        raise ValueError(f'Need exactly one registered {model}/{split} prediction: {len(choices)}')
    return copy.deepcopy(next(iter(choices.values())))


def clinical(root, artifacts, model, metadata, score_path):
    from opera.run.finish_story_inputs import flipi2_os_config, refresh_discrimination
    from opera.run.launch_adapted_clinical_comparison import compact
    from opera.run.render_clinical_score_panels import expand_index_scores
    from opera.run.analyse_locked_predictions import read_table
    from opera.run.refresh_clinical_figures import package_clinical
    directory = root/'exports/clinical'/model
    directory.mkdir(parents=True, exist_ok=True)
    target = directory/'clinical_results'
    recipe = flipi2_os_config(read(score_path))
    meta = read_table(metadata)
    meta.subject_id = meta.subject_id.astype(str)
    specs = expand_index_scores(recipe, meta)
    for spec in specs: spec['model'] = model
    recipe.update(scores=specs, all_indices=None, native_endpoints={})
    primary = find_artifact(artifacts, model, 'tuning')
    prediction_root = Path(primary['path']).parent.parent
    selected = []
    hashes = {}
    for split in ('train', 'tuning', 'held_out'):
        source = prediction_root/split/'direct_cr_predictions.parquet'
        marker = read(source.parent/'evaluation_complete.json')
        if marker['prediction_sha256'] != sha(source) or marker['inputs']['split'] != split:
            raise ValueError('Clinical source predictions changed')
        hashes[split] = marker['prediction_sha256']
        selected.append({**copy.deepcopy(primary), 'split': split, 'path': str(source)})
    immutable(directory/'intent.yaml', dict(model=model, metadata_sha256=sha(metadata),
              score_config_sha256=sha(score_path), prediction_sha256=hashes))
    for declaration in selected:
        private = directory/'predictions_INTERNAL'/f'{declaration["split"]}.parquet'
        private.parent.mkdir(exist_ok=True)
        if not private.exists():
            temporary = private.with_suffix('.writing')
            compact(declaration['path'], temporary, specs, meta)
            temporary.replace(private)
        declaration['path'] = str(private)
    analysis = dict(opera_model=model, patient_metadata=str(metadata), patient_metadata_column_map={},
                    artifacts=[a for a in selected if a['split'] != 'train'],
                    clinical_threshold_artifacts=[a for a in selected if a['split'] == 'train'])
    immutable(directory/'analysis.yaml', analysis)
    immutable(directory/'clinical_scores.yaml', recipe)
    if not (target/'report_manifest.json').is_file():
        if target.exists(): target.rename(directory/f'clinical_incomplete_{time_ns()}_INTERNAL')
        subprocess.run([sys.executable, '-u', '-m', 'opera.run.render_clinical_score_panels',
            '--analysis-config', str(directory/'analysis.yaml'), '--score-config', str(directory/'clinical_scores.yaml'),
            '--output-dir', str(target), '--split', 'held_out', '--execute'], check=True,
            env={**os.environ, 'CUDA_VISIBLE_DEVICES': '', 'MPLBACKEND': 'Agg', 'OPERA_THESIS_STYLE': '1',
                 'OPERA_PAPER_EXPORT_PLOT_STATE': '1', 'OPERA_PAPER_FIGURE_MARK': ''})
    if not (target/'clinical_discrimination.csv').is_file():
        refresh_discrimination(directory/'analysis.yaml', directory/'clinical_scores.yaml', target/'clinical_discrimination.csv')
    archive = directory/f'download_clinical_{model}.zip'
    if not archive_ready(archive): package_clinical(target, archive)
    if not archive_ready(archive): raise ValueError(f'Clinical archive incomplete: {archive}')
    return dict(status='complete', archive=str(archive))


def time_ns():
    import time
    return time.time_ns()


def stages_config(root, artifacts):
    from bonsai.functional.checkpointing import verified_training_selection
    wanted = dict(joint={'Local joint': 'J_L', 'National joint': 'J_E', 'Adapted joint': 'J_EA'},
                  opera={'Local OPERA': 'J_OA_OWN_L', 'National OPERA': 'R_E', 'Adapted OPERA': 'R_EA'})
    stages = {}
    for family, models in wanted.items():
        stages[family] = {}
        for label, model in models.items():
            paths = []
            for split in ('tuning', 'held_out'):
                artifact = find_artifact(artifacts, model, split)
                directory = Path(artifact['path']).parent
                array, meta = directory/'patient_embeddings.npz', directory/'patient_embeddings.metadata.json'
                if not array.is_file(): raise FileNotFoundError(f'Required embedding export absent: {array}')
                provenance = read(meta)
                evaluation = read(directory/'evaluation_complete.json')
                selected = verified_training_selection(Path(provenance['checkpoint']).parent)
                if (provenance['split'] != split or evaluation['inputs']['checkpoint_sha256'] != selected['best_checkpoint_sha256']
                        or not evaluation['inputs'].get('export_embeddings')):
                    raise ValueError('Embedding/prediction checkpoint provenance differs')
                paths.append(str(array))
            stages[family][label] = paths
    path = root/'exports/embedding_stages.yaml'
    immutable(path, stages)
    return path


def task_structure(root, artifacts, env):
    from bonsai.functional.checkpointing import verified_training_selection
    from opera.run.package_figure_review import PATIENT_COLUMNS
    import csv
    directory = root/'exports/task_structure'
    directory.mkdir(parents=True, exist_ok=True)
    names = ['outcome_query_similarity.csv', 'outcome_correction_similarity.csv',
             'disease_correction_similarity.csv', 'outcome_specialisation.csv',
             'disease_specialisation.csv', 'task_structure_summary.json']
    for model in ('J_OA_OWN_L', 'R_E', 'R_EA'):
        artifact = find_artifact(artifacts, model, 'held_out')
        metadata = read(Path(artifact['path']).parent/'patient_embeddings.metadata.json')
        checkpoint = Path(metadata['checkpoint'])
        selected = verified_training_selection(checkpoint.parent)
        output = directory/f'task_structure_{model}'
        summary = output/'task_structure_summary.json'
        if summary.is_file():
            if read(summary)['checkpoint_sha256'] != selected['best_checkpoint_sha256']:
                raise ValueError('Task structure belongs to another checkpoint')
            continue
        old = Path('data/task_structure_comparison_20261119')/f'task_structure_{model}'
        if model == 'R_E' and (old/'task_structure_summary.json').is_file() and read(old/'task_structure_summary.json')['checkpoint_sha256'] == selected['best_checkpoint_sha256']:
            output.mkdir(exist_ok=True)
            for name in names: shutil.copyfile(old/name, output/name)
        else:
            subprocess.run([sys.executable, '-u', '-m', 'opera.run.export_opera_task_structure',
                '--run-dir', str(checkpoint.parent), '--output-dir', str(output)], check=True, env=env)
    target = directory/'download_task_structure.zip'
    temporary = target.with_suffix('.writing')
    with ZipFile(temporary, 'w', ZIP_DEFLATED) as archive:
        for model in ('J_OA_OWN_L', 'R_E', 'R_EA'):
            for name in names:
                path = directory/f'task_structure_{model}'/name
                if path.suffix == '.csv':
                    with path.open(newline='') as handle:
                        if {c.lower() for c in next(csv.reader(handle), [])} & PATIENT_COLUMNS:
                            raise ValueError('Patient columns in task aggregate')
                archive.write(path, str(path.relative_to(directory)))
    temporary.replace(target)
    return dict(status='complete', archive=str(target))


def run(root, *, metadata=None, scores=None):
    root = root.resolve()
    metadata = metadata or Path('data/clinical_registry_refresh/20261111/prepared_clinical/patient_metadata.parquet').resolve()
    scores = scores or Path('data/story_closure_20261116/server_only/score_config_flipi2_os.yaml').resolve()
    plan = read(root/'queue_plan.yaml')
    for label, item in plan.get('export_inputs', {}).items():
        if sha(Path(item['path'])) != item['sha256']:
            raise ValueError(f'Frozen export input changed: {label}')
    sources = read(plan['analyses'])
    declarations = []
    wanted = {'J_L', 'J_OA_OWN_L', 'J_EA', 'R_EA', 'R_AB_EA', 'J_E', 'R_E'}
    for source in sources:
        declarations.extend(d for d in source['artifacts'] if d['artifact']['model'] in wanted
                            and int(d['artifact'].get('seed', 42)) == 42)
    artifacts, _, waiting, _ = rebound_artifacts(root, plan, declarations)
    if waiting: return dict(status='waiting_for_main_predictions', jobs=waiting)
    env = {**os.environ, 'CUDA_VISIBLE_DEVICES': '', 'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1',
           'MKL_NUM_THREADS': '1', 'NUMEXPR_NUM_THREADS': '1', 'MPLBACKEND': 'Agg'}
    state = {}
    for model in ('R_EA', 'J_EA', 'R_AB_EA', 'J_OA_OWN_L'):
        state['clinical_'+model] = clinical(root, artifacts, model, metadata, scores)
    state['task_structure'] = task_structure(root, artifacts, env)
    stages = stages_config(root, artifacts)
    from opera.run.build_paper_embedding_tables import PROJECTION_VERSION
    story = root/'exports'/f'story_{PROJECTION_VERSION}'
    subprocess.run([sys.executable, '-u', '-m', 'opera.run.finish_story_inputs', '--only', 'embeddings',
        '--stages-config', str(stages), '--patient-metadata', str(metadata), '--survival-predictions',
        find_artifact(artifacts, 'R_EA', 'held_out')['path'], '--output-root', str(story), '--execute'],
        check=True, env=env)
    story_status = read(story/'story_status.json')
    if story_status.get('embeddings', {}).get('status') != 'complete':
        raise ValueError('Story embedding export incomplete')
    state['embeddings'] = dict(status='complete', archive=str(story/'download_story.zip'))
    for family in ('joint', 'opera'):
        output = root/'exports'/f'outcome_grids_{family}_{PROJECTION_VERSION}'
        archive = output/('download_outcome_grids_joint.zip' if family == 'joint' else 'download_outcome_grids.zip')
        if not (output/'manifest.json').is_file() or not archive_ready(archive):
            subprocess.run([sys.executable, '-u', '-m', 'opera.run.export_outcome_embedding_grids',
                '--story-root', str(story), '--family', family, '--output-dir', str(output)], check=True, env=env)
        if not archive_ready(archive): raise ValueError(f'Outcome-grid archive incomplete: {archive}')
        state['outcome_grids_'+family] = dict(status='complete', archive=str(archive))
    checkpoints = root/'analyses/core_damage/checkpoints'
    if not (root/'analyses/core_damage/results/analysis_manifest.json').is_file():
        return {**state, 'status': 'waiting_for_core_analysis'}
    deciles = root/'exports/deciles'
    if not (deciles/'manifest.json').is_file() or not archive_ready(deciles/'fixed_budget_review.zip'):
        subprocess.run([sys.executable, '-u', '-m', 'opera.analysis.fixed_budget_risk',
            '--checkpoint-dir', str(checkpoints), '--output-dir', str(deciles),
            '--models', 'R_EA', 'J_EA', 'J_OA_OWN_L', '--primary', 'R_EA'], check=True, env=env)
    dca = root/'exports/decision_curves'
    if not (dca/'manifest.json').is_file() or not archive_ready(dca/'download_decision_curves.zip'):
        subprocess.run([sys.executable, '-u', '-m', 'opera.run.export_decision_curves',
            '--checkpoint-dir', str(checkpoints), '--metadata', str(metadata), '--score-config',
            str(root/'exports/clinical/R_EA/clinical_scores.yaml'), '--clinical-results',
            str(root/'exports/clinical/R_EA/clinical_results'), '--output-dir', str(dca)], check=True, env=env)
    for archive in (story/'download_story.zip', deciles/'fixed_budget_review.zip', dca/'download_decision_curves.zip'):
        if not archive_ready(archive): raise ValueError(f'Final export archive incomplete: {archive}')
    state.update(deciles=dict(status='complete'), decision_curves=dict(status='complete'), status='complete')
    (root/'exports/export_status.json').write_text(json.dumps(state, indent=2))
    return state


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, default=Path('data/cutoff_recovery_20261121'))
    a = p.parse_args()
    report = run(a.root)
    print(json.dumps(report, indent=2))
    if report.get('status') != 'complete': sys.exit(3)


if __name__ == '__main__': main()
