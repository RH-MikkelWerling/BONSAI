"""Independent, resumable final-figure jobs on the secure server.

No training, primary-model selection or analysis-lock changes. Existing IG and
clinical jobs are observed, never restarted. Each lane owns its own lock/logs.
"""
import argparse
import copy
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from zipfile import ZipFile, ZIP_DEFLATED

import yaml


def read(path):
    return yaml.safe_load(Path(path).read_text())


def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def immutable(path, data):
    import fcntl
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.with_suffix(path.suffix + '.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if path.exists():
            if read(path) != data:
                raise ValueError(f'Inputs changed; use a fresh figure workspace: {path}')
        else:
            path.write_text(yaml.safe_dump(data, sort_keys=False))


def final_fit(national):
    from bonsai.functional.checkpointing import verified_training_selection
    bundle = national / 'adapted_fits'
    row, = [r for r in read(bundle / 'bundle.json')['runs'] if r['name'] == 'seed42/EA/R']
    config = bundle / row['config_path']
    if sha(config) != row['config_sha256']:
        raise ValueError('Frozen adapted OPERA configuration changed')
    run = national / 'neural_runs' / row['run_directory']
    selection = verified_training_selection(run)
    return dict(row=row, config=config, run=run, checkpoint=run / 'best.ckpt',
                checkpoint_sha256=selection['best_checkpoint_sha256'])


def local_ablations(repo, national):
    from opera.run.paper_prediction_queue import bundle_runs, canonical_models
    directory = Path(read(national / 'direct_fits/bundle.json')['provenance']['frozen_local_bundle'])
    if directory.is_file(): directory = directory.parent
    bundle, rows = bundle_runs(directory)
    result = []
    for row, path, config in rows:
        models = set(canonical_models(row, config, bundle))
        selected = models & {'J_OA_OWN_NODIS_L', 'J_OA_OWN_NOOUT_L'}
        if int(config['seed']) == 42 and selected:
            result.append({'row': row, 'config': path, 'models': sorted(selected),
                           'bundle': directory, 'run': repo / 'data/paper_256_v3/neural_runs' / row['run_directory'],
                           'predictions': repo / 'data/paper_256_v3/predictions' / row['run_directory'] / 'held_out'})
    names = [model for item in result for model in item['models']]
    if len(names) != len(set(names)): raise ValueError('Ambiguous local ablation fits')
    return result


def verify_exports(national, fit):
    result = {}
    for split in ['tuning', 'held_out']:
        directory = national / 'predictions' / fit['row']['run_directory'] / split
        marker = read(directory / 'evaluation_complete.json')
        prediction = directory / 'direct_cr_predictions.parquet'
        if (marker['inputs']['checkpoint_sha256'] != fit['checkpoint_sha256']
                or marker['inputs']['config_sha256'] != fit['row']['config_sha256']
                or marker['inputs']['split'] != split or marker['prediction_sha256'] != sha(prediction)):
            raise ValueError(f'Prediction identity changed: {directory}')
        embedding = directory / 'patient_embeddings.npz'
        meta = read(directory / 'patient_embeddings.metadata.json')
        if (Path(meta['checkpoint']).resolve() != fit['checkpoint'].resolve()
                or meta['split'] != split or meta['config_name'] != fit['config'].stem):
            raise ValueError(f'Embedding identity differs from the final model: {directory}')
        if not embedding.is_file():
            raise FileNotFoundError(embedding)
        result[split] = dict(prediction=prediction, embedding=embedding,
                             prediction_sha256=marker['prediction_sha256'], embedding_sha256=sha(embedding))
    return result


def compact_predictions(source, destination):
    """Only five needed columns, streamed; no full raw panel in pandas."""
    import pyarrow.parquet as pq
    columns = ['subject_id', 'outcome', 'time_days', 'event', 'cif_365d']
    parquet = pq.ParquetFile(source)
    if not set(columns) <= set(parquet.schema_arrow.names):
        raise ValueError('Need native one-year prediction fields for final-model maps')
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix('.writing.parquet')
    writer = None
    outcomes = set()
    try:
        for batch in parquet.iter_batches(columns=columns, batch_size=8192):
            outcomes.update(batch.column(batch.schema.get_field_index('outcome')).to_pylist())
            if writer is None:
                writer = pq.ParquetWriter(temporary, batch.schema)
            writer.write_batch(batch)
    finally:
        if writer is not None:
            writer.close()
    if not outcomes:
        raise ValueError('Empty prediction export')
    temporary.replace(destination)
    return outcomes


class Lane:
    def __init__(self, root, name, execute):
        self.root, self.name, self.execute = root, name, execute
        self.path = root / f'{name}_status.json'
        self.report = read(self.path) if self.path.exists() else {'lane': name, 'steps': {}}
        self.env = {**os.environ, 'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1',
                    'MKL_NUM_THREADS': '1', 'NUMEXPR_NUM_THREADS': '1', 'NUMBA_NUM_THREADS': '1',
                    'OPERA_ANALYSIS_WORKERS': '1', 'OPERA_SUMMARY_WORKERS': '1',
                    'MPLBACKEND': 'Agg', 'OPERA_PAPER_EXPORT_PLOT_STATE': '1', 'OPERA_PAPER_FIGURE_MARK': ''}
        self.env.pop('OPERA_ANALYSIS_CHECKPOINT_DIR', None)
        self.env.pop('OPERA_ANALYSIS_DISTRIBUTED', None)
        if name != 'arrows':
            self.env['CUDA_VISIBLE_DEVICES'] = ''

    def save(self):
        temporary = self.path.with_suffix('.writing.json')
        temporary.write_text(json.dumps(self.report, indent=2, default=str))
        temporary.replace(self.path)

    def blocked(self, name, reason):
        self.report['steps'][name] = {'status': 'blocked', 'reason': str(reason)}
        self.save()
        print(f'{name}: BLOCKED: {reason}', flush=True)

    def step(self, name, command=None, *, action=None, outputs=(), identity=None):
        signature = json.loads(json.dumps({'command': command, 'identity': identity}, default=str))
        previous = self.report['steps'].get(name, {})
        if previous.get('signature') is not None and previous['signature'] != signature:
            raise ValueError(f'{name} inputs changed; use a fresh workspace')
        if previous.get('status') == 'complete':
            if previous['signature'] != signature:
                raise ValueError(f'Completed {name} inputs changed; use a fresh workspace')
            if all(Path(p).exists() for p in outputs):
                if any(sha(Path(path)) != digest for path, digest in previous.get('output_sha256', {}).items()):
                    raise ValueError(f'Completed {name} output changed')
                print(f'{name}: reused completed step', flush=True)
                return True
        record = {'status': 'prepared', 'signature': signature}
        self.report['steps'][name] = record
        self.save()
        if not self.execute:
            print(f'{name}: PREPARED', flush=True)
            return False
        log = self.root / 'logs' / f'{self.name}_{name}.log'
        log.parent.mkdir(exist_ok=True)
        record.update(status='running', log=str(log)); self.save()
        print(f'{name}: RUNNING; {log}', flush=True)
        try:
            if action is not None:
                action()
            else:
                with log.open('a') as handle:
                    subprocess.run(command, stdout=handle, stderr=subprocess.STDOUT, env=self.env, check=True)
            if not all(Path(p).exists() for p in outputs):
                raise ValueError(f'{name} did not produce every required completion output')
            record['status'] = 'complete'
            record['output_sha256'] = {str(p): sha(Path(p)) for p in outputs if Path(p).is_file()}
            print(f'{name}: COMPLETE', flush=True)
            return True
        except Exception as error:
            record.update(status='failed', error=str(error))
            print(f'{name}: FAILED: {error}', flush=True)
            return False
        finally:
            self.save()


def projects(repo, national, closure=None):
    result = {'local': repo / 'data/paper_256_v3/analysis/local_test_20261109_four_allocations',
            'extended': national / 'extended_test_comparison',
            'tuning': national / 'extended_tuning_comparison',
            'fractions': national / 'fraction_results_20261116'}
    if closure is not None:
        result.update({f'focused_{kind}': closure / f'comparisons_{kind}' for kind in ['core', 'ablations']})
    return result


def harvest(repo, national, output):
    """Copy finished aggregates; keep absent/non-final sources in the inventory."""
    from opera.run.package_figure_review import PATIENT_COLUMNS
    from opera.run.build_paper_plot_tables import model_cells, design_search_table
    import pandas as pd
    report = {}
    for name, project in projects(repo, national, output.parent.parent).items():
        result = next((p for p in [project / 'prediction_analysis', project / 'results']
                       if (p / 'analysis_manifest.json').is_file()), None)
        if result is None:
            report[name] = {'status': 'no_completed_analysis', 'project': str(project)}
            continue
        target = output / name; target.mkdir(parents=True, exist_ok=True)
        copied = []
        patterns = ['patient_subgroup_panels.csv', 'stratified_*_effects.csv', 'stratified_*_models.csv',
                    'strategy_selection*.csv', 'analysis_status.csv', 'effect_summaries*.csv',
                    'contrasts_all_horizons.csv', 'panel_estimates.csv', 'analysis_manifest.json']
        for pattern in patterns:
            for path in result.glob(pattern):
                if path.suffix == '.csv':
                    with path.open(newline='') as handle:
                        header = next(csv.reader(handle), [])
                    if set(c.strip().lower() for c in header) & (PATIENT_COLUMNS | {'subject_ids', 'patientid'}):
                        raise ValueError(f'Patient rows in supposed aggregate: {path}')
                shutil.copy2(path, target / path.name); copied.append(path.name)
        config = next((p for p in [project / 'analysis_inventory/analysis.yaml', project / 'analysis.yaml'] if p.exists()), None)
        if config is not None and (result / 'estimates.csv').is_file():
            spec = read(config)
            frames = []
            wanted = {'S_L', 'D_L', 'O_L', 'J_L', 'J_E', 'J_EA', 'R_L', 'R_E', 'R_EA',
                      'OPERA', 'J_OA_OWN_L', 'X', 'XP', 'XR', 'LR'}
            for chunk in pd.read_csv(result / 'estimates.csv', chunksize=50000):
                subset = chunk.loc[chunk['split'].eq('held_out') & chunk.seed.eq(42)
                                   & chunk.metric.eq('auc') & chunk.model.isin(wanted)].copy()
                subset['model'] = subset.model.replace({'J_OA_OWN_L': 'OPERA'})
                frames.append(subset)
            estimates = pd.concat(frames, ignore_index=True)
            if not estimates.empty:
                keys = ['cohort', 'outcome', 'horizon_days', 'model']
                if (estimates.groupby(keys).estimate.nunique() > 1).any():
                    raise ValueError('Conflicting aliases in absolute cell estimates')
                metadata = Path(spec['cell_metadata'])
                if not metadata.is_absolute(): metadata = repo / metadata
                model_cells(estimates, pd.read_csv(metadata), seed=42).to_csv(target / 'model_cells.csv', index=False)
                copied.append('model_cells.csv')
        if name == 'tuning' and (result / 'effect_summaries.csv').exists():
            from opera.run.build_paper_plot_tables import Missing
            try:
                designs = design_search_table(pd.read_csv(result / 'effect_summaries.csv', low_memory=False), seed=42)
                if not designs.empty:
                    designs.to_csv(target / 'design_search.csv', index=False); copied.append('design_search.csv')
            except Missing as error:
                report['design_search'] = {'status': 'blocked', 'reason': str(error)}
        report[name] = {'status': 'harvested', 'source': str(result), 'files': copied}
    try:
        report['local_ablations'] = [{'models': item['models'], 'run': str(item['run']),
             'checkpoint_exists': (item['run'] / 'best.ckpt').is_file(),
             'evaluation_marker_exists': (item['predictions'] / 'evaluation_complete.json').is_file()}
             for item in local_ablations(repo, national)]
    except (ValueError, FileNotFoundError, KeyError) as error:
        report['local_ablations'] = {'status': 'blocked', 'reason': str(error)}
    # Resolve the actual local-joint metric path from the frozen local bundle.
    try:
        from opera.run.paper_prediction_queue import bundle_runs, canonical_models
        local_bundle = Path(read(national / 'direct_fits/bundle.json')['provenance']['frozen_local_bundle'])
        if local_bundle.is_file(): local_bundle = local_bundle.parent
        bundle, rows = bundle_runs(local_bundle)
        matches = [(row, config) for row, _, config in rows
                   if 'J_L' in canonical_models(row, config, bundle) and int(config['seed']) == 42]
        if len(matches) != 1:
            raise ValueError('Ambiguous local joint fit')
        row, _ = matches[0]
        directory = repo / 'data/paper_256_v3/predictions' / row['run_directory'] / 'held_out'
        marker = read(directory / 'evaluation_complete.json')
        from bonsai.functional.checkpointing import verified_training_selection
        selection = verified_training_selection(repo / 'data/paper_256_v3/neural_runs' / row['run_directory'])
        if (marker['inputs']['config_sha256'] != row['config_sha256']
                or marker['inputs']['checkpoint_sha256'] != selection['best_checkpoint_sha256']
                or marker['prediction_sha256'] != sha(directory / 'direct_cr_predictions.parquet')):
            raise ValueError('Local atlas metric configuration differs')
        source = directory / 'direct_cr_pooled_metrics.csv'
        target = output / 'atlas'; target.mkdir(exist_ok=True)
        shutil.copy2(source, target / source.name)
        report['atlas'] = {'status': 'recovered', 'model': 'J_L', 'source': str(source), 'sha256': sha(source)}
    except (ValueError, FileNotFoundError, KeyError) as error:
        report['atlas'] = {'status': 'blocked', 'reason': str(error)}
    # Receipts, not cross-allocation process monitoring.
    report['job_receipts'] = {}
    for name, receipt in {
        'IG': national / 'final_interpretation_20261116/interpretation_manifest.json',
        'embeddings': national / 'encoder_embedding_review_20261115/review/review_manifest.json',
        'clinical_joint': national / 'clinical_20261115/J_EA/launch_manifest.json',
        'clinical_router': national / 'clinical_20261115/R_AB_EA/launch_manifest.json',
    }.items():
        report['job_receipts'][name] = read(receipt).get('status') if receipt.exists() else 'receipt_absent'
    # Repackage newly completed clinical runs without starting another inference.
    from opera.run.refresh_clinical_figures import package_clinical
    for model in ['J_EA', 'R_AB_EA']:
        source = national / 'clinical_20261115' / model
        if (source / 'clinical_results/report_manifest.json').is_file():
            target = output / f'clinical_{model}.zip'
            target.unlink(missing_ok=True)
            package_clinical(source / 'clinical_results', target)
            discrimination = source / 'clinical_discrimination.csv'
            if discrimination.is_file():
                with ZipFile(target, 'a') as z: z.write(discrimination, discrimination.name)
    embedding = national / 'encoder_embedding_review_20261115/embedding_figures.zip'
    if embedding.is_file(): shutil.copy2(embedding, output / embedding.name)
    interpretation = national / 'final_interpretation_20261116'
    if report['job_receipts']['IG'] == 'complete':
        from opera.run.prepare_offline_figure_bundle import aggregate_input, copy_presentation
        for folder in ['attributions/summary_365d', 'input_restrictions']:
            source = interpretation / folder
            for path in source.rglob('*'):
                if not path.is_file() or path.is_symlink(): continue
                target = output / 'interpretation' / folder / path.relative_to(source)
                if path.name.endswith('.plot.json'):
                    copy_presentation(path, target)
                elif path.suffix in {'.pdf', '.png'} or aggregate_input(path):
                    target.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(path, target)
    (output / 'harvest_report.json').write_text(json.dumps(report, indent=2))


def map_config(national, fit, output):
    exports = verify_exports(national, fit)
    metadata = national.parent / 'clinical_registry_refresh/20261111/prepared_clinical/patient_metadata.parquet'
    if not metadata.is_file(): raise FileNotFoundError(metadata)
    import pyarrow.parquet as pq
    source = exports['held_out']['prediction']
    outcomes = set()
    for batch in pq.ParquetFile(source).iter_batches(columns=['outcome'], batch_size=8192):
        outcomes.update(batch.column(0).to_pylist())
    planned = ['overall_survival', 'infection_diagnosis_or_antimicrobial_use_composite',
               'serious_infection_composite', 'neutropenia_g3plus']
    checks = [o for o in planned if o in outcomes]
    if not checks: raise ValueError('None of the fixed calibration outcomes is available')
    private = output / 'maps_SERVER_INTERNAL'; private.mkdir(exist_ok=True)
    slim = private / 'held_out.parquet'
    identity = {'model': 'R_EA', 'checkpoint_sha256': fit['checkpoint_sha256'],
                'config_sha256': fit['row']['config_sha256'], 'metadata_sha256': sha(metadata),
                'exports': {s: {k: str(v) for k, v in row.items()} for s, row in exports.items()}}
    immutable(private / 'identity.yaml', identity)
    recipe = {'method': 'umap', 'ipi_group_map': {'low': 'Low', 'medium': 'Intermediate', 'high': 'High'},
              'stages': {'Adapted OPERA': [str(exports[s]['embedding']) for s in ['tuning', 'held_out']]},
              'calibration_maps': {'stage': 'Adapted OPERA', 'predictions': str(slim),
                                   'horizon_days': 365, 'outcomes': checks},
              'gallery': {'stage': 'Adapted OPERA', 'predictions': str(slim), 'horizon_days': 365}}
    immutable(private / 'embedding_config.yaml', recipe)
    return private, metadata, identity, source


def package(root, lane):
    ready = root / 'ready' / lane
    archive = root / f'download_{lane}.zip'
    with ZipFile(archive, 'w', compression=ZIP_DEFLATED, compresslevel=1) as z:
        if ready.exists():
            for path in ready.rglob('*'):
                if not path.is_file() or path.is_symlink(): continue
                # ready/ is populated only by aggregate harvesters/checkpoint
                # exporters and native renderers; never by map input builders.
                if '_INTERNAL' in path.name.upper(): continue
                if path.suffix == '.csv':
                    from opera.run.package_figure_review import PATIENT_COLUMNS
                    with path.open(newline='') as h: columns = next(csv.reader(h), [])
                    if set(c.lower() for c in columns) & (PATIENT_COLUMNS | {'subject_ids', 'patientid'}):
                        raise ValueError(f'Patient table in ready directory: {path}')
                z.write(path, path.relative_to(root))
        z.write(root / f'{lane}_status.json', f'{lane}_status.json')
    print(f'DOWNLOAD: {archive}', flush=True)


def run(args):
    repo, root = Path.cwd().resolve(), args.output_root.resolve()
    national = repo / 'data/paper_256_v3_national_20261111'
    root.mkdir(parents=True, exist_ok=True)
    if args.status:
        for path in sorted(root.glob('*_status.json')):
            print(path.read_text())
        return
    import fcntl
    with (root / f'.{args.lane}.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        lane = Lane(root, args.lane, args.execute)
        if args.lane == 'recover' and args.refresh_harvest:
            lane.report['steps'].pop('existing_aggregates', None)
        if args.execute:
            minimum = {'recover': 12, 'maps': 24, 'arrows': 16}[args.lane]
            command = [sys.executable, 'scripts/check_results_resources.py', '--minimum-free-gib', str(minimum), '--require-no-coordinator']
            if args.lane == 'arrows': command.append('--require-free-gpu')
            subprocess.run(command, check=True)
        ready = root / 'ready' / args.lane; ready.mkdir(parents=True, exist_ok=True)
        if args.lane == 'recover':
            lane.step('existing_aggregates', action=lambda: harvest(repo, national, ready),
                      outputs=[ready / 'harvest_report.json'], identity='completed_sources_only_v1')
        try:
            fit = final_fit(national)
            identity = {'checkpoint_sha256': fit['checkpoint_sha256'], 'config_sha256': fit['row']['config_sha256']}
            if args.lane == 'recover':
                directory = ready / 'task_structure_R_EA'
                lane.step('task_structure', [sys.executable, '-u', '-m', 'opera.run.export_opera_task_structure',
                          '--run-dir', str(fit['run']), '--output-dir', str(directory)],
                          outputs=[directory / 'task_structure_summary.json'], identity=identity)
            else:
                private, metadata, binding, source = map_config(national, fit, root)
                config = private / 'embedding_config.yaml'
                if args.lane == 'maps':
                    compacted = lane.step('compact_prediction_columns', action=lambda: compact_predictions(source, private / 'held_out.parquet'),
                                          outputs=[private / 'held_out.parquet'], identity=binding)
                    if compacted and lane.step('map_tables', [sys.executable, '-u', '-m', 'opera.run.build_paper_embedding_tables',
                               '--config', str(config), '--patient-metadata', str(metadata), '--output-dir', str(private / 'plot_tables')],
                               outputs=[private / 'plot_tables/embedding_gallery.parquet'], identity=binding):
                        full = read(repo / 'opera/configs/paper_results_figures.full.example.yaml')
                        ids = ['fig_patient_map', 'supp_calibration_maps', *[f'supp_embedding_gallery_{i}' for i in range(1, 7)]]
                        plates = [p for p in full['figures'] if p['id'] in ids]
                        for plate in plates:
                            for section in ['tables', 'json_inputs']:
                                for key, value in plate.get(section, {}).items():
                                    plate[section][key] = str(private / 'plot_tables' / Path(value).name)
                        recipe = private / 'figures.yaml'
                        immutable(recipe, {'figures': plates, 'required_figures': ids})
                        lane.step('render_maps_and_gallery', [sys.executable, '-u', '-m', 'opera.run.render_paper_results',
                                  '--config', str(recipe), '--output-dir', str(ready / 'figures'), '--execute'],
                                  outputs=[ready / 'figures/figure_manifest.json'], identity=binding)
                else:
                    directory = ready / 'correction_arrows'
                    if lane.step('correction_arrows', [sys.executable, '-u', '-m', 'opera.run.export_opera_correction_arrows',
                              '--checkpoint', str(fit['checkpoint']), '--config-dir', str(fit['config'].parent),
                              '--config-name', fit['config'].stem, '--embedding-config', str(config),
                              '--batch-size', '1', '--num-workers', '0', '--device', 'cuda', '--output-dir', str(directory)],
                              outputs=[directory / 'correction_arrows.json'], identity=binding):
                        recipe = directory / 'figures.yaml'
                        immutable(recipe, {'figures': [{'id': 'supp_correction_arrows', 'function': 'figure_correction_arrows',
                            'tables': {'arrows': str(directory / 'correction_arrows.csv'), 'density': str(directory / 'correction_arrow_density.csv')},
                            'json_inputs': {'meta': str(directory / 'correction_arrows.json')}}]})
                        lane.step('render_arrows', [sys.executable, '-u', '-m', 'opera.run.render_paper_results', '--config', str(recipe),
                                  '--output-dir', str(ready / 'figures'), '--execute'], outputs=[ready / 'figures/figure_manifest.json'], identity=binding)
        except (ValueError, FileNotFoundError, KeyError) as error:
            lane.blocked('final_model_inputs', error)
        if args.lane == 'arrows':
            # These are evaluation passes over already completed local fits,
            # not new training or replacements for the adapted ablations.
            try:
                from bonsai.functional.checkpointing import verified_training_selection
                ablations = local_ablations(repo, national)
                if not ablations:
                    lane.blocked('local_ablation_evaluations', 'No local NODIS/NOOUT fits in the frozen bundle')
                for item in ablations:
                    model, = item['models']
                    try:
                        selected = verified_training_selection(item['run'])
                        lane.step('evaluate_' + model, [sys.executable, '-u', '-m', 'opera.run.paper_prediction_queue',
                            '--bundle-dir', str(item['bundle']), '--runs-dir', str(repo / 'data/paper_256_v3/neural_runs'),
                            '--output-dir', str(repo / 'data/paper_256_v3/predictions'), '--split', 'held_out',
                            '--fit-name', item['row']['name'], '--batch-size', '1', '--num-workers', '0',
                            '--report-path', str(ready / f'{model}_evaluation.json'), '--execute'],
                            outputs=[item['predictions'] / 'evaluation_complete.json'],
                            identity={'config_sha256': item['row']['config_sha256'],
                                      'checkpoint_sha256': selected['best_checkpoint_sha256']})
                    except (ValueError, FileNotFoundError) as error:
                        lane.blocked('evaluate_' + model, error)
            except (ValueError, FileNotFoundError, KeyError) as error:
                lane.blocked('local_ablation_evaluations', error)
        states = [s['status'] for s in lane.report['steps'].values()]
        source_gaps = {}
        if args.lane == 'recover' and (ready / 'harvest_report.json').exists():
            source_report = read(ready / 'harvest_report.json')
            source_gaps = {k: v for k, v in source_report.items()
                           if isinstance(v, dict) and v.get('status') in ['blocked', 'no_completed_analysis']}
            lane.report['source_gaps'] = source_gaps
        lane.report['status'] = ('complete' if states and all(s == 'complete' for s in states)
                                 and not source_gaps
                                 else 'complete_with_gaps' if args.execute else 'prepared')
        lane.save()
        if args.execute: package(root, args.lane)
        print(f'LANE {args.lane}: {lane.report["status"]}; inspect {lane.path}', flush=True)
        if 'failed' in states: raise SystemExit(1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--lane', choices=['recover', 'maps', 'arrows'], default='recover')
    parser.add_argument('--output-root', type=Path, default=Path('data/figure_closure_20261116'))
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--status', action='store_true')
    parser.add_argument('--refresh-harvest', action='store_true', help='Refresh recovered completed-source exports; does not repeat model diagnostics')
    run(parser.parse_args())


if __name__ == '__main__':
    main()
