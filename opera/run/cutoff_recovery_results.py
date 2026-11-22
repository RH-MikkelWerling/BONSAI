"""Watch corrected predictions; publish early damage checks and refresh registered analyses."""
import argparse
import copy
import csv
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from zipfile import ZipFile, ZIP_DEFLATED

import yaml

from opera.run.finish_figure_inputs import immutable, sha
from opera.run.cutoff_recovery_worker import read, completed
from opera.run.prepare_cutoff_recovery_queue import CORE, DAMAGE_CORE

AGGREGATES = {'effect_summaries.csv', 'effect_summaries_disease_balanced.csv', 'panel_estimates.csv',
    'estimates.csv', 'contrasts.csv', 'contrasts_all_horizons.csv', 'contrast_coverage.csv',
    'model_coverage.csv', 'calibration_summary.csv', 'operating_points.csv',
    'operating_point_differences.csv', 'seed_averaged_effects.csv', 'analysis_status.csv',
    'headline_holm.csv', 'strategy_selection.csv', 'negative_transfer.csv'}


def deduplicate_artifacts(records):
    """A forecast registered by several analyses must not create duplicate rows."""
    unique = {}
    for original in records:
        item = copy.deepcopy(original)
        if 'seed' in item: item['seed'] = int(item['seed'])
        if item.get('format') == 'direct_cif':
            item.setdefault('cohort_from_metadata', False)
            item.setdefault('probability_prefix', 'cif_')
            if 'horizon_days' in item:
                item['include_horizons_days'] = [int(item.pop('horizon_days'))]
        signature = json.dumps({k:v for k,v in item.items() if k != 'include_horizons_days'}, sort_keys=True)
        if signature not in unique:
            unique[signature] = item
            continue
        saved = unique[signature]
        if not saved.get('include_horizons_days') or not item.get('include_horizons_days'):
            saved.pop('include_horizons_days', None)
        else:
            saved['include_horizons_days'] = sorted(set(saved['include_horizons_days']) | set(item['include_horizons_days']))
    return list(unique.values())


def rebound_artifacts(root, plan, declarations):
    records, touched, waiting, omitted = [], set(), [], []
    for declaration in declarations:
        item = copy.deepcopy(declaration['artifact'])
        if declaration['action'] == 'replace':
            touched.add(item['model'])
            digest = declaration['job']
            job = plan['jobs'][digest]
            if not completed(root, digest, job):
                waiting.append(digest)
                continue
            done = read(root/'completed'/f'{digest}.json')
            if done['status'] == 'not_estimable':
                omitted.append(dict(job=digest, model=item['model'], reason='not_estimable'))
                continue
            suffix = declaration.get('relative_prediction') or f'{item["split"]}/direct_cr_predictions.parquet'
            item['path'] = str(root/'predictions'/job['row']['run_directory']/suffix)
            item.update(format='direct_cif', cohort_from_metadata=True)
        if not Path(item['path']).is_file():
            raise FileNotFoundError(item['path'])
        records.append(item)
    # Same forecast may have been registered in more than one source inventory.
    return deduplicate_artifacts(records), touched, sorted(set(waiting)), omitted


def prepare_analysis(root, plan, source, name):
    source_path = Path(source['source'])
    if sha(source_path) != source['source_sha256']:
        raise ValueError('Original analysis recipe changed after recovery preparation')
    deferred = source.get('recovery_deferred_artifacts', [])
    if source.get('recovery_defer_design_search') or (deferred and not source['artifacts']):
        return dict(status='deferred_historical_search', original=str(source_path),
                    note='Historical search is outside the active paper-first refresh')
    artifacts, touched, waiting, omitted = rebound_artifacts(root, plan, source['artifacts'])
    if waiting:
        return dict(status='waiting', jobs=waiting)
    if not touched:
        if deferred:
            return dict(status='deferred_historical_search', original=str(source_path),
                        note='Only unchanged baselines remain; affected search fits were deferred')
        return dict(status='reused_unaffected_analysis', original=str(source_path))
    if not ({a['model'] for a in artifacts} & touched):
        return dict(status='not_estimable', original=str(source_path), omitted=omitted,
                    note='Affected fits unavailable; original affected estimates are not reused')
    config = read(source_path)
    directory = root/'analyses'/name
    directory.mkdir(parents=True, exist_ok=True)
    original_manifest = Path(config['contrast_manifest'])
    manifest = read(original_manifest)
    if deferred:
        from opera.run.analyse_locked_predictions import models_for
        available = {a['model'] for a in artifacts}
        missing = {d['artifact']['model'] for d in deferred}-available
        for family, expressions in manifest['essential_contrasts'].items():
            manifest['essential_contrasts'][family] = {n: e for n, e in expressions.items()
                if not models_for({n: e}) & missing and not n.startswith('DS_')}
        config['extra_contrasts'] = {n: e for n, e in config.get('extra_contrasts', {}).items()
            if not models_for({n: e}) & missing and not n.startswith('DS_')}
        config.update(design_search=False, recovery_deferred_models=sorted(missing),
                      recovery_scope='paper_first: historical design sweep deferred')
        # Prevent old affected search results being merged as "unaffected".
        touched.update(missing)
    immutable(directory/'contrasts.yaml', manifest)
    config.update(artifacts=artifacts, contrast_manifest=str(directory/'contrasts.yaml'),
                  contrast_touch_models=sorted(touched), clinical_threshold_artifacts=[],
                  compute_inventory=None, recovery_original_analysis=str(source_path))
    if config.get('opera_model') in touched:
        config['contrast_touch_models'].append('OPERA')
    if config.get('route_weights'):
        route_model = config.get('route_model', 'R_L')
        choices = [a for a in artifacts if a['model'] == route_model
                   and a.get('split') == 'held_out' and int(a.get('seed', 42)) == int(config.get('route_seed', 42))]
        if len(choices) != 1:
            raise ValueError('Cannot uniquely rebind the route-weight export')
        config['route_weights'] = str(Path(choices[0]['path']).parent/'source_route_weights.parquet')
        if not Path(config['route_weights']).is_file():
            raise FileNotFoundError('Corrected route-weight export is absent; do not reuse old routes')
    immutable(directory/'analysis.yaml', config)
    report = dict(status='ready', analysis=str(directory/'analysis.yaml'),
                  source=str(source_path), source_sha256=source['source_sha256'],
                  refreshed_models=sorted(touched), omitted=omitted,
                  deferred_historical_fits=[dict(job=d['job'], model=d['artifact']['model'],
                      seed=d['artifact'].get('seed', 42)) for d in deferred],
                  note='Unchanged forecasts reused; contrasts wholly unaffected by recovery excluded')
    immutable(directory/'preparation.yaml', report)
    return report


def prepare_core(root, plan, sources):
    # Main fixed seed only. Specialists are deliberately not a gate for the first
    # damage assessment, which directly pairs each new prediction with its old one.
    declarations = []
    for source in sources:
        for declaration in source['artifacts']:
            item = declaration['artifact']
            if int(item.get('seed', 42)) != 42 or item.get('split') not in {'tuning', 'held_out'}:
                continue
            if item['model'] in DAMAGE_CORE or item['model'] in {'J_E', 'R_E', 'X', 'XP', 'XR'}:
                declarations.append(declaration)
    artifacts, touched, waiting, omitted = rebound_artifacts(root, plan, declarations)
    if waiting:
        return dict(status='waiting', jobs=waiting)
    required = {'J_L', 'J_OA_OWN_L', 'J_EA', 'R_EA'}
    present = {a['model'] for a in artifacts}
    if not required <= present:
        raise ValueError(f'Core damage assessment missing models: {sorted(required-present)}')
    old = []
    for declaration in declarations:
        if declaration['action'] == 'replace':
            item = copy.deepcopy(declaration['artifact'])
            if Path(item['path']).is_file():
                item['model_alias'] = 'UNCORRECTED_' + item['model']
                old.append(item)
    artifacts += old
    artifacts = deduplicate_artifacts(artifacts)
    contrasts = {f'CORRECTION_{m}': f'gain({m},UNCORRECTED_{m})' for m in sorted(touched)
                 if any(a.get('model_alias') == 'UNCORRECTED_'+m for a in old)}
    for model in touched:
        for baseline in ('XP', 'X', 'J_E', 'R_E'):
            if baseline in present:
                contrasts[f'CORRECTED_{model}_VS_{baseline}'] = f'gain({model},{baseline})'
    contrasts['ADAPTATION_J'] = 'gain(J_EA,J_E)'
    contrasts['ADAPTATION_OPERA'] = 'gain(R_EA,R_E)'
    contrasts['ARCHITECTURE_COHORT'] = 'gain(J_OA_OWN_L,J_L)'
    contrasts['ARCHITECTURE_ADAPTED'] = 'gain(R_EA,J_EA)'
    directory = root/'analyses/core_damage'
    directory.mkdir(parents=True, exist_ok=True)
    base = read(sources[0]['source'])
    manifest = read(base['contrast_manifest'])
    manifest['essential_contrasts'] = {'cutoff_recovery': contrasts}
    immutable(directory/'contrasts.yaml', manifest)
    base.update(artifacts=artifacts, contrast_manifest=str(directory/'contrasts.yaml'),
                extra_contrasts={}, opera_model=None, route_weights=None, design_search=False,
                strategy_candidates=[], clinical_threshold_artifacts=[],
                identity_date_sensitivity=False, stratified_reanalysis={}, patient_groups={},
                operating_points={'rules': [], 'pairs': []}, patient_gain_contrasts=[],
                contrast_touch_models=sorted(touched), coverage_roster=None, compute_inventory=None)
    immutable(directory/'analysis.yaml', base)
    return dict(status='ready', analysis=str(directory/'analysis.yaml'),
                scope='Exploratory correction audit; original results explicitly labelled UNCORRECTED', omitted=omitted)


def run_analysis(directory):
    if (directory/'results/analysis_manifest.json').is_file():
        build_plot_inputs(directory)
        return
    subprocess.run([sys.executable, 'scripts/check_results_resources.py', '--minimum-free-gib', '96',
                    '--require-no-coordinator'], check=True)
    env = {**os.environ, 'CUDA_VISIBLE_DEVICES': '', 'OPERA_ANALYSIS_CHECKPOINT_DIR': str(directory/'checkpoints'),
           'OPERA_ANALYSIS_WORKERS': '1', 'OPERA_SUMMARY_WORKERS': '1', 'OMP_NUM_THREADS': '1',
           'OPENBLAS_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1', 'NUMEXPR_NUM_THREADS': '1'}
    logs = directory/'logs'
    logs.mkdir(exist_ok=True)
    with (logs/'worker.log').open('a') as wl, (logs/'coordinator.log').open('a') as cl:
        worker = subprocess.Popen([sys.executable, '-u', '-m', 'opera.run.results_worker',
            '--checkpoint-dir', str(directory/'checkpoints'), '--wait-hours', '72'], env=env, stdout=wl, stderr=subprocess.STDOUT)
        process = None
        try:
            process = subprocess.Popen([sys.executable, '-u', '-m', 'opera.run.analyse_locked_predictions',
                '--config', str(directory/'analysis.yaml'), '--output-dir', str(directory/'results')],
                env=env, stdout=cl, stderr=subprocess.STDOUT)
            while process.poll() is None:
                if worker.poll() is not None:
                    raise RuntimeError(f'CPU helper stopped: {logs/"worker.log"}')
                time.sleep(5)
            if process.returncode:
                raise RuntimeError(f'Analysis failed; saved batches retained: {logs/"coordinator.log"}')
        finally:
            for child in (process, worker):
                if child is not None and child.poll() is None:
                    child.terminate()
                    try: child.wait(timeout=10)
                    except subprocess.TimeoutExpired: child.kill(); child.wait()
    build_plot_inputs(directory)


def build_plot_inputs(directory):
    config = read(directory/'analysis.yaml')
    if not config.get('cell_metadata'):
        raise ValueError('Registered analysis lacks the training-support metadata needed for figures')
    paper_tables = reuse_unaffected_summaries(directory, config)
    alias = directory/'prediction_analysis'
    if not alias.exists(): alias.symlink_to('paper_tables', target_is_directory=True)
    if alias.resolve() != paper_tables.resolve():
        raise ValueError('Plot input alias points outside this recovery analysis')
    output = directory/'plot_tables'
    if (output/'plot_tables_manifest.json').is_file(): return
    split = 'held_out' if any(a.get('split') == 'held_out' for a in config['artifacts']) else 'tuning'
    command = [sys.executable, '-u', '-m', 'opera.run.build_paper_plot_tables',
               '--analysis-root', str(directory), '--cell-metadata', str(config['cell_metadata']),
               '--output-dir', str(output), '--seed', '42', '--split', split]
    if config.get('patient_metadata'):
        command += ['--patient-metadata', str(config['patient_metadata'])]
    with (directory/'logs/plot_tables.log').open('a') as log:
        subprocess.run(command, check=True, stdout=log, stderr=subprocess.STDOUT,
                       env={**os.environ, 'CUDA_VISIBLE_DEVICES': ''})


def reuse_unaffected_summaries(directory, config):
    """Review tables combine new comparisons with demonstrably unchanged old rows.

    Raw new results and their completion manifest are never modified. Unknown
    contrast names are not guessed to be unaffected.
    """
    import pandas as pd
    from opera.run import analyse_locked_predictions as analysis
    destination = directory/'paper_tables'
    destination.mkdir(exist_ok=True)
    for path in (directory/'results').glob('*.csv'):
        target = destination/path.name
        if not target.exists(): target.symlink_to(Path('../results')/path.name)
    source = config.get('recovery_original_analysis')
    if not source: return destination
    original = read(source)
    manifest = read(original['contrast_manifest'])
    expressions = {name: value for family in manifest['essential_contrasts'].values() for name, value in family.items()}
    expressions = analysis.national_counterparts(expressions, original.get('opera_model'))
    for variable in ('NATIONAL_ROUTING', 'OPERA_CONTRASTS', 'SAMPLING_SENSITIVITY', 'TASK_ROUTED',
                     'TASK_DESIGN', 'TASK_SEEDS', 'TASK_ABLATION', 'ENCODER_LR', 'TASK_POOLING',
                     'FULL_OWN', 'PARTIAL_POOLING', 'ORIGINAL_ROUTER'):
        expressions.update(getattr(analysis, variable, {}))
    expressions.update(original.get('extra_contrasts', {}))
    touched = set(config['contrast_touch_models'])
    if set(original.get('strategy_candidates', [])) & touched: touched.add('BEST_TUNING')
    unaffected = {name for name, expression in expressions.items()
                  if isinstance(expression, str) and not (analysis.models_for({name: expression}) & touched)}
    source = Path(source)
    candidates = [source.parent/'results', source.parent.parent/'prediction_analysis', source.parent/'prediction_analysis']
    old_results = next((p for p in candidates if (p/'effect_summaries.csv').is_file()
                        and (p/'analysis_manifest.json').is_file()), None)
    if old_results is None:
        (destination/'reuse_report.json').write_text(json.dumps(dict(status='original_results_not_found',
            note='Retain already downloaded unchanged results separately; no unknown rows were reused'), indent=2))
        return destination
    report = []
    for name in ('effect_summaries.csv', 'effect_summaries_disease_balanced.csv'):
        new_path, old_path = directory/'results'/name, old_results/name
        if not new_path.is_file() or not old_path.is_file(): continue
        new = pd.read_csv(new_path)
        old = pd.read_csv(old_path)
        keep = old.loc[old.contrast.isin(unaffected)].copy()
        keep = keep.loc[~keep.contrast.isin(set(new.contrast))]
        new['recovery_provenance'] = 'corrected_prediction_analysis'
        keep['recovery_provenance'] = 'reused_unaffected_original'
        temporary = destination/(name+'.writing')
        pd.concat([new, keep], ignore_index=True).to_csv(temporary, index=False)
        temporary.replace(destination/name)
        report.append(dict(table=name, reused_rows=len(keep), source=str(old_path), source_sha256=sha(old_path)))
    (destination/'reuse_report.json').write_text(json.dumps(report, indent=2))
    return destination


def package(root):
    from opera.run.package_figure_review import PATIENT_COLUMNS, PLOT_TABLES
    target = root/'download_cutoff_recovery.zip'
    temporary = target.with_suffix('.writing')
    with ZipFile(temporary, 'w', ZIP_DEFLATED) as archive:
        for directory in sorted((root/'analyses').glob('*')):
            raw_results = directory/'results'
            results = directory/'paper_tables' if (directory/'paper_tables').is_dir() else raw_results
            if not (raw_results/'analysis_manifest.json').is_file():
                continue
            for name in sorted(AGGREGATES):
                path = results/name
                if not path.is_file(): continue
                with path.open(newline='') as handle:
                    columns = {c.lower().strip() for c in next(csv.reader(handle), [])}
                if columns & PATIENT_COLUMNS:
                    raise ValueError(f'Patient column in aggregate export: {path}')
                archive.write(path, f'{directory.name}/{name}')
            reuse_report = directory/'paper_tables/reuse_report.json'
            if reuse_report.is_file(): archive.write(reuse_report, f'{directory.name}/reuse_report.json')
            for path in (directory/'plot_tables').glob('*.csv'):
                if path.name not in PLOT_TABLES: continue
                with path.open(newline='') as handle:
                    columns = {c.lower().strip() for c in next(csv.reader(handle), [])}
                if columns & PATIENT_COLUMNS:
                    raise ValueError(f'Patient column in plot table: {path}')
                archive.write(path, f'{directory.name}/plot_tables/{path.name}')
        for name in ('queue_summary.json', 'pipeline_status.json', 'queue_index.csv', 'workload_policy.json'):
            if (root/name).is_file(): archive.write(root/name, name)
        for pattern in ('clinical/*/download_clinical_*.zip', 'story/download_story.zip',
                        'outcome_grids_*/download_outcome_grids*.zip', 'deciles/fixed_budget_review.zip',
                        'decision_curves/download_decision_curves.zip', 'model_exports/download_model_exports.zip',
                        'task_structure/download_task_structure.zip'):
            for path in (root/'exports').glob(pattern):
                archive.write(path, 'exports/'+str(path.relative_to(root/'exports')))
        # Interpretation outputs are known aggregate reports/figures only.
        from opera.run.package_figure_review import REPORT_TABLES
        interpretation = root/'final_interpretation'
        if interpretation.is_dir():
            for path in interpretation.rglob('*.csv'):
                if path.name not in REPORT_TABLES or 'INTERNAL' in str(path): continue
                with path.open(newline='') as handle:
                    columns = {c.lower().strip() for c in next(csv.reader(handle), [])}
                if columns & PATIENT_COLUMNS: raise ValueError('Patient column in interpretation aggregate')
                archive.write(path, 'interpretation/'+str(path.relative_to(interpretation)))
            figure_ids = {'fig_attribution', 'supp_attribution_atlas', 'supp_attribution_structure',
                          'supp_recent_record_gaps', 'supp_attribution_patient'}
            # The patient illustration in the registered recipe is synthetic.
            for path in interpretation.rglob('*'):
                if path.stem in figure_ids and path.suffix in {'.pdf', '.png'} and not path.is_symlink():
                    archive.write(path, 'interpretation/'+str(path.relative_to(interpretation)))
    temporary.replace(target)
    print('DOWNLOAD:', target, flush=True)


def watch(root, *, execute=False, wait_hours=96, poll_seconds=30, exports=False, wait_interpretation=False):
    plan = read(root/'queue_plan.yaml')
    if sha(Path(plan['analyses'])) != plan['analyses_sha256']:
        raise ValueError('Registered analysis inventory changed')
    sources = read(plan['analyses'])
    from opera.run.cutoff_recovery_workload import read_policy, filter_source
    policy = read_policy(root, plan)
    policy_digest = sha(root/'workload_policy.json') if (root/'workload_policy.json').is_file() else None
    if policy['profile'] == 'paper_first':
        sources = [filter_source(source, policy) for source in sources]
    deadline = time.monotonic()+wait_hours*3600
    completed_names = set()
    completed_reports = {}
    exports_attempted = False
    exports_report = None
    while time.monotonic() < deadline:
        current_digest = sha(root/'workload_policy.json') if (root/'workload_policy.json').is_file() else None
        if current_digest != policy_digest:
            raise RuntimeError('Workload scope changed; restart the results lane to use the separate paper-first outputs')
        state = {}
        if exports_report: state['exports'] = exports_report
        tasks = [('core_damage', lambda: prepare_core(root, plan, sources))]
        if (root/'epoch20_priority/manifest.yaml').is_file():
            from opera.run.cutoff_recovery_epoch20 import prepare_analysis as prepare_epoch20
            tasks.append(('epoch20_check', lambda: prepare_epoch20(root)))
        if (root/'epoch20_priority/opera_extension/manifest.yaml').is_file():
            from opera.run.cutoff_recovery_epoch20 import prepare_analysis as prepare_epoch20
            tasks.append(('epoch20_architecture_check', lambda: prepare_epoch20(root, architecture='opera')))
        prefix = 'paper_registered' if policy['profile'] == 'paper_first' else 'registered'
        tasks += [(f'{prefix}_{i:02d}', lambda s=s, i=i: prepare_analysis(root, plan, s, f'{prefix}_{i:02d}'))
                  for i, s in enumerate(sources)]
        for name, prepare in tasks:
            if name in completed_names:
                state[name] = completed_reports[name]
                continue
            report = prepare()
            state[name] = report
            if report['status'] in {'reused_unaffected_analysis', 'not_estimable', 'deferred_historical_search'}:
                completed_names.add(name)
                completed_reports[name] = report
            elif report['status'] == 'ready' and execute:
                run_analysis(root/'analyses'/name)
                completed_names.add(name)
                state[name] = {**report, 'status': 'complete'}
                completed_reports[name] = state[name]
                (root/'pipeline_status.json').write_text(json.dumps(state, indent=2))
                package(root)
        if execute and 'core_damage' in completed_names and exports and not exports_attempted:
            exports_attempted = True
            export_dir = root/'exports'
            export_dir.mkdir(exist_ok=True)
            try:
                with (export_dir/'pipeline.log').open('a') as log:
                    subprocess.run([sys.executable, '-u', '-m', 'opera.run.cutoff_recovery_exports',
                        '--root', str(root)], check=True, stdout=log, stderr=subprocess.STDOUT)
                exports_report = {'status': 'complete'}
                (export_dir/'failure.json').unlink(missing_ok=True)
            except subprocess.CalledProcessError as error:
                if error.returncode == 3:
                    exports_report = {'status': 'waiting_for_remaining_main_predictions'}
                    exports_attempted = False
                else:
                    exports_report = {'status': 'failed', 'log': str(export_dir/'pipeline.log')}
                    (export_dir/'failure.json').write_text(json.dumps(exports_report, indent=2))
            state['exports'] = exports_report
            package(root)
        (root/'pipeline_status.json').write_text(json.dumps(state, indent=2))
        print(json.dumps({name: spec['status'] for name, spec in state.items()}), flush=True)
        if not execute or len(completed_names) == len(tasks):
            if execute and exports_report and exports_report['status'] == 'failed':
                package(root)
                raise RuntimeError(f'Numerical refresh complete; exports failed. Inspect {exports_report["log"]}')
            if execute and exports and (not exports_report or exports_report['status'] != 'complete'):
                time.sleep(poll_seconds)
                continue
            if execute and wait_interpretation:
                manifest = root/'final_interpretation/interpretation_manifest.json'
                if not manifest.is_file() or read(manifest).get('status') != 'complete':
                    state['interpretation'] = {'status': 'waiting'}
                    (root/'pipeline_status.json').write_text(json.dumps(state, indent=2))
                    time.sleep(poll_seconds)
                    continue
                state['interpretation'] = {'status': 'complete'}
            if execute:
                state['overall'] = {'status': 'complete', 'workload_profile': policy['profile'],
                                    'deferred_jobs': len(policy['deferred_jobs'])}
                (root/'pipeline_status.json').write_text(json.dumps(state, indent=2))
            if execute: package(root)
            if execute: print('RECOVERY RESULTS COMPLETE:', root/'download_cutoff_recovery.zip', flush=True)
            return
        time.sleep(poll_seconds)
    raise TimeoutError('Results watcher wait limit reached; completed outputs and checkpoints retained')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, default=Path('data/cutoff_recovery_20261121'))
    p.add_argument('--execute', action='store_true')
    p.add_argument('--wait-hours', type=float, default=96)
    p.add_argument('--exports', action='store_true', help='After core results: refresh clinical, embeddings, deciles and decision curves')
    p.add_argument('--wait-interpretation', action='store_true', help='Refresh the final ZIP after the separately queued interpretation finishes')
    a = p.parse_args()
    if a.wait_hours <= 0: p.error('wait-hours must be positive')
    watch(a.root.resolve(), execute=a.execute, wait_hours=a.wait_hours, exports=a.exports,
          wait_interpretation=a.wait_interpretation)


if __name__ == '__main__':
    main()
