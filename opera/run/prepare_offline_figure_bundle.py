"""Prepare portable plotting inputs, then package an existing server review.

CPU only: reads saved analyses/predictions for presentation, never trains or
runs the encoder. Native aggregate recipes are portable; patient-dependent
panels use editable presentation state with pixel-only patient point layers.
"""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import yaml

from opera.run.package_figure_review import PLOT_TABLES, REPORT_TABLES, PATIENT_COLUMNS

REPO = Path(__file__).resolve().parents[2]
DISPLAY_JSON = {'disease_groups.json', 'outcome_groups.json', 'calibration_examples.json',
                'patient_covariate_order.json', 'patient_contrast_styles.json',
                'patient_subgroups_styles.json', 'patient_subgroups_order.json',
                'national_subgroups_styles.json', 'national_subgroups_order.json',
                'correction_arrows.json', 'task_structure_summary.json'}
SYNTHETIC = {'synthetic_patient_episodes_risk.csv', 'synthetic_patient_episodes_logodds.csv'}
PATIENT_FUNCTIONS = {'figure_patient_map', 'figure_patient_disease_groups',
    'figure_calibration_maps', 'figure_router_arrows', 'figure_embedding_gallery',
    'figure_embedding_comparison'}


def digest(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024*1024), b''):
            result.update(chunk)
    return result.hexdigest()


def aggregate_input(path, *, original_name=None):
    """Positive filename allowlist and structural rejection, not just ID stripping."""
    import csv
    path = Path(path)
    if path.is_symlink() or not path.is_file() or 'INTERNAL' in path.name.upper():
        return False
    if (original_name or path.name) not in PLOT_TABLES | REPORT_TABLES | DISPLAY_JSON | SYNTHETIC | {
        'borrowing_hypothesis_test.csv', 'atlas.csv', 'origin_estimates.csv'}:
        return False
    if path.suffix == '.csv':
        with path.open() as handle:
            header = next(csv.reader(handle), [])
        columns = {c.strip().lower() for c in header}
        # These combinations describe individual patient rows even without IDs.
        if columns & PATIENT_COLUMNS or {'time_days', 'event'} <= columns or {
            'model_risk', 'ipi_group', 'age'} <= columns:
            return False
    elif path.suffix == '.json':
        def safe(value):
            if isinstance(value, dict):
                return not ({str(k).lower() for k in value} & PATIENT_COLUMNS) and all(safe(v) for v in value.values())
            return all(safe(v) for v in value) if isinstance(value, list) else True
        if not safe(json.loads(path.read_text())):
            return False
    else:
        return False
    return True


def portable_recipe(plate, source_recipe, offline, copied):
    """Return an entirely relative aggregate recipe, or identify private inputs."""
    from opera.run.render_paper_results import resolve_input
    result = copy.deepcopy(plate)
    if plate['function'] in {'figure_borrowing_map','figure_borrowing_detail'}:
        for source in plate.get('tables',{}).values():
            path=resolve_input(source,source_recipe)
            provenance=path.parent/'analysis_provenance.json' if path else None
            if provenance is not None and provenance.is_file():
                data=json.loads(provenance.read_text())
                result.setdefault('options',{}).update(ablation_mode=data.get('ablation'),
                    model_identity=str(data.get('model',''))+' / '+str(data.get('scientific_hash',''))[:12])
                break
    reasons = []
    for section in ('tables', 'json_inputs'):
        for key, source in result.get(section, {}).items():
            path = resolve_input(source, source_recipe)
            if path is None or not path.is_file():
                reasons.append(f'{key}: unavailable {Path(source).name}')
                continue
            if path.name in SYNTHETIC:
                # Repository-owned illustrations have a dummy subject_id=-1.
                # Verify that source, then remove even the dummy identifier.
                import pandas as pd
                canonical=REPO/'paper/figures'/path.name
                if path.resolve()!=canonical.resolve():
                    reasons.append(f'{key}: synthetic source is not the repository illustration');continue
                frame=pd.read_csv(path)
                if 'subject_id' in frame and not frame.subject_id.eq(-1).all():
                    raise ValueError('Synthetic illustration contains real subject identifiers')
                frame=frame.drop(columns=[c for c in frame if c.lower() in PATIENT_COLUMNS])
                payload=frame.to_csv(index=False).encode()
                sha=hashlib.sha256(payload).hexdigest()
                destination=offline/'inputs'/f'{sha[:16]}_{path.name}'
                destination.parent.mkdir(parents=True,exist_ok=True);destination.write_bytes(payload)
                result[section][key]='../inputs/'+destination.name
                continue
            if not aggregate_input(path):
                reasons.append(f'{key}: not an approved aggregate input ({path.name})')
                continue
            identity = str(path.resolve())
            if identity not in copied:
                sha = digest(path)
                destination = offline/'inputs'/f'{sha[:16]}_{path.name}'
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(path, destination)
                copied[identity] = destination
            # Recipes are in offline/recipes; no server paths survive.
            result[section][key] = '../inputs/' + copied[identity].name
    return result, reasons


def copy_presentation(source, destination):
    """Copy only a validated presentation JSON and its pixel assets."""
    from opera.visualization.plot_state import SCHEMA
    state = json.loads(source.read_text())
    if state.get('schema') != SCHEMA:
        raise ValueError('Unknown plot-state schema')
    def no_private_keys(value):
        if isinstance(value, dict):
            if {str(k).lower() for k in value} & (PATIENT_COLUMNS | {'offsets', 'model_risk', 'time_days', 'patient_rows'}):
                raise ValueError('Patient data in presentation state')
            for item in value.values():no_private_keys(item)
        elif isinstance(value,list):
            for item in value:no_private_keys(item)
    no_private_keys(state)
    destination.parent.mkdir(parents=True, exist_ok=True)
    for axis in state['axes']:
        for key in ('image', 'legend_image'):
            if not axis.get(key):
                continue
            path = (source.parent / axis[key]).resolve()
            if not path.is_relative_to(source.parent.resolve()) or path.suffix != '.png' or path.is_symlink():
                raise ValueError('Invalid presentation image layer')
            relative = destination.with_suffix('').name + '/' + path.name
            target = destination.parent/relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, target)
            axis[key] = relative
    destination.write_text(json.dumps(state, allow_nan=False))


def bundle_files(offline):
    """Validate the only offline payload types accepted by the ZIP packager."""
    offline=Path(offline).resolve()
    manifest=json.loads((offline/'manifest.json').read_text())
    if manifest.get('schema')!='opera.offline.v1':
        raise ValueError('Unknown offline bundle schema')
    selected={offline/'manifest.json'}
    if (offline/'README.md').is_file():selected.add(offline/'README.md')
    def local(source,base):
        path=(base/source).resolve()
        if not path.is_relative_to(offline) or not path.is_file() or (base/source).is_symlink():
            raise ValueError('Offline input escapes bundle or is missing')
        return path
    for entry in manifest['entries']:
        if entry['mode']=='native':
            recipe=local(entry['recipe'],offline);selected.add(recipe)
            for plate in yaml.safe_load(recipe.read_text())['figures']:
                for section in ('tables','json_inputs'):
                    for source in plate.get(section,{}).values():
                        path=local(source,recipe.parent)
                        original=path.name.split('_',1)[-1]
                        if not aggregate_input(path,original_name=original):
                            raise ValueError(f'Not an approved aggregate: {path.name}')
                        selected.add(path)
        elif entry['mode']=='presentation':
            state=local(entry['source'],offline);selected.add(state)
            data=json.loads(state.read_text())
            if data.get('schema')!='opera.presentation.v1':raise ValueError('Unknown presentation schema')
            def check(value):
                if isinstance(value,dict):
                    if set(value)&(PATIENT_COLUMNS|{'offsets','model_risk','time_days','patient_rows'}):
                        raise ValueError('Patient data in presentation state')
                    for item in value.values():check(item)
                elif isinstance(value,list):
                    for item in value:check(item)
            check(data)
            for axis in data['axes']:
                for key in ('image','legend_image'):
                    if axis.get(key):
                        path=local(axis[key],state.parent)
                        if path.suffix!='.png':raise ValueError('Presentation layers must be PNG')
                        selected.add(path)
        elif entry['mode']!='missing':raise ValueError('Unknown offline entry mode')
    return sorted(selected)


def prepare(review, analysis, output, *, clinical=True):
    output.mkdir(parents=True, exist_ok=False)
    offline = output/'offline';offline.mkdir()
    private = output/'server_only';private.mkdir()
    logs = output/'logs';logs.mkdir()
    report = json.loads((review/'review_manifest.json').read_text())
    if report.get('evaluation_split') not in {'tuning','held_out'}:
        raise ValueError('Review must declare tuning or held_out evaluation')
    entries, copied = [], {}
    environment = {**os.environ, 'CUDA_VISIBLE_DEVICES': '', 'OPERA_ANALYSIS_WORKERS': '1',
                   'OPERA_PAPER_EXPORT_PLOT_STATE': '1', 'MPLBACKEND': 'Agg'}
    environment['OPERA_PAPER_FIGURE_MARK']=('TUNING SET — not test results'
        if report['evaluation_split']=='tuning' else '')

    def run(name, command):
        print(f'Preparing offline presentation: {name}', flush=True)
        with (logs/(name.replace('/', '__')+'.log')).open('w') as log:
            return subprocess.run(command, cwd=REPO, env=environment,
                                  stdout=log, stderr=subprocess.STDOUT).returncode == 0

    for recipe_path in sorted(review.glob('*.yaml')):
        recipe = yaml.safe_load(recipe_path.read_text())
        for plate in recipe.get('figures', []):
            family, name = recipe_path.stem, plate['id']
            if Path(name).name!=name or name in {'.','..'}:
                raise ValueError('Unsafe figure ID in source recipe')
            identity = f'{family}/{name}'
            portable, reasons = portable_recipe(plate, recipe_path, offline, copied)
            if not reasons and plate['function'] not in PATIENT_FUNCTIONS:
                destination = offline/'recipes'/family/(name+'.yaml')
                # One extra family directory: adapt relative inputs once.
                for section in ('tables','json_inputs'):
                    portable[section] = {k: '../'+v for k,v in portable.get(section,{}).items()}
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_text(yaml.safe_dump(dict(figures=[portable]),sort_keys=False))
                entries.append(dict(id=identity, mode='native', recipe=str(destination.relative_to(offline))))
                continue
            unavailable = [x for x in reasons if ': unavailable ' in x]
            if unavailable:
                entries.append(dict(id=identity,mode='missing',reasons=unavailable));continue
            # Only patient-input figure functions are eligible for the visual
            # fallback; an unrecognised aggregate must not be silently exported.
            if plate['function'] not in PATIENT_FUNCTIONS:
                entries.append(dict(id=identity,mode='missing',reasons=reasons));continue
            state = review/family/(name+'.plot.json')
            if not state.is_file():
                one = private/(family+'__'+name+'.yaml')
                # Source recipes use absolute server paths or paths relative to
                # their original recipe. Resolve before moving the recipe.
                source = copy.deepcopy(plate)
                from opera.run.render_paper_results import resolve_input
                for section in ('tables','json_inputs'):
                    source[section] = {k:str(resolve_input(v,recipe_path)) for k,v in source.get(section,{}).items()}
                one.write_text(yaml.safe_dump(dict(figures=[source]),sort_keys=False))
                rendered = private/(family+'__'+name)
                if run(identity,[sys.executable,'-m','opera.run.render_paper_results',
                    '--config',str(one),'--output-dir',str(rendered),'--execute']):
                    state = rendered/(name+'.plot.json')
            if state.is_file():
                target = offline/'presentations'/family/(name+'.plot.json')
                copy_presentation(state,target)
                entries.append(dict(id=identity,mode='presentation',source=str(target.relative_to(offline))))
            else:
                entries.append(dict(id=identity,mode='missing',reasons=['Presentation export failed; inspect preparation log']))

    clinical_root = Path(report.get('clinical_dir',str(analysis/'clinical_scores')))
    clinical_states = list(clinical_root.rglob('*.plot.json')) if clinical_root.is_dir() else []
    expected_clinical={str(p.relative_to(review/'clinical_scores').with_suffix(''))
                       for p in (review/'clinical_scores').rglob('*.png')
                       if not any(part.endswith('.plot') for part in p.parts)}
    available_clinical={str(p.relative_to(clinical_root)).removesuffix('.plot.json') for p in clinical_states}
    if clinical and (expected_clinical-available_clinical) and report.get('clinical_validation',{}).get('valid'):
        from opera.run.paper_experiments import load_environment
        env = load_environment(analysis/'overnight.env')
        score_config = env.get('OPERA_PAPER_CLINICAL_SCORE_CONFIG')
        if score_config and Path(score_config).is_file():
            fresh = private/'clinical_scores'
            if run('clinical_scores',[sys.executable,'-m','opera.run.render_clinical_score_panels',
                    '--analysis-config',str(analysis/'analysis_inventory/analysis.yaml'),
                    '--score-config',score_config,'--split',report['evaluation_split'],
                    '--output-dir',str(fresh),'--execute']):
                clinical_states = list(fresh.rglob('*.plot.json'));clinical_root = fresh
    for state in sorted(clinical_states):
        relative = state.relative_to(clinical_root)
        target = offline/'presentations/clinical_scores'/relative
        copy_presentation(state,target)
        entries.append(dict(id='clinical_scores/'+str(relative).removesuffix('.plot.json'),
                            mode='presentation',source=str(target.relative_to(offline))))
    existing = {e['id'] for e in entries}
    for image in sorted(review.rglob('*.png')):
        if any(part.endswith('.plot') for part in image.parts):continue
        identity = str(image.relative_to(review).with_suffix(''))
        if identity not in existing:
            entries.append(dict(id=identity,mode='missing',reasons=['No portable recipe or presentation state for this rendered figure']))
    rendered_ids={str(path.relative_to(review).with_suffix(''))
                  for path in review.rglob('*.png') if not any(p.endswith('.plot') for p in path.parts)}
    missing_rendered=[e['id'] for e in entries if e['mode']=='missing' and e['id'] in rendered_ids]
    manifest = dict(schema='opera.offline.v1',evaluation_split=report['evaluation_split'],entries=entries,
        complete_for_rendered_figures=not missing_rendered, missing_rendered_figures=missing_rendered,
        status='complete' if all(e['mode']!='missing' for e in entries) else 'partial',
        notes=['Native recipes use only allowlisted aggregate inputs.',
               'Presentation mode retains dense patient marks only as publication-image pixels.',
               'Group curves, axes and labels remain editable. Patient-dot colour/data changes need server rendering.',
               'Missing analyses are explicit; no synthetic results or checkpoint inference were used.'])
    (offline/'manifest.json').write_text(json.dumps(manifest,indent=2))
    (offline/'README.md').write_text(
        '# Portable OPERA figures\n\nFrom the project checkout, with its plotting environment:\n\n'
        '```bash\npython -m opera.run.render_offline_bundle --bundle EXTRACTED_ZIP '
        '--output-dir NEW_LOCAL_ITERATION\n```\n\n'
        'Use --list to inspect availability, --figures main/fig_pooling to select figures, '
        'or --require-complete to reject missing entries. Native recipes are editable YAML '
        'with full aggregate tables. Presentation JSON exposes layout, labels, ticks and '
        'aggregate clinical curves; patient-dependent dense layers remain PNGs, without '
        'point coordinates, timelines or predictions. Statistical regrouping is not a '
        'presentation edit. Missing IG/national analyses cannot be supplied until computed.\n')
    return offline,manifest


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--review-dir',type=Path,required=True)
    parser.add_argument('--analysis-root',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--output-zip',type=Path)
    parser.add_argument('--prepare-only',action='store_true',help='Prepare payload for the existing packaging stage')
    parser.add_argument('--skip-clinical-refresh',action='store_true')
    args=parser.parse_args()
    if not args.prepare_only and args.output_zip is None:
        parser.error('Provide --output-zip or --prepare-only')
    offline,manifest=prepare(args.review_dir.resolve(),args.analysis_root.resolve(),
        args.output_dir.resolve(),clinical=not args.skip_clinical_refresh)
    from opera.run.package_figure_review import package
    result=(package(args.review_dir,args.output_zip,args.analysis_root,offline_dir=offline,include_analysis_backup=False)
            if not args.prepare_only else None)
    print(json.dumps(dict(package=result,offline_status=manifest['status'],
        missing=[e for e in manifest['entries'] if e['mode']=='missing']),indent=2))


if __name__=='__main__':main()
