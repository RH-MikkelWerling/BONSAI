"""Complete local tuning figures: clinical scores, routed arrows, fresh borrowing.

Run on the GPU server. Existing main analysis and prediction exports stay intact.
"""
from __future__ import annotations
import argparse
from datetime import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from zoneinfo import ZoneInfo
import yaml
from bonsai.functional.checkpointing import verified_training_selection
from opera.run.paper_experiments import load_environment, path_value, stage_bundles, OWN_REFERENCE
from opera.run.paper_prediction_queue import bundle_runs, canonical_models
from opera.run.prepare_available_plot_configs import build
from opera.run.prepare_clinical_figure_metadata import prepare


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def selected_fit(analysis, paper_root, bundles):
    selected = [a for a in analysis['artifacts'] if a['model'] == analysis.get('opera_model', 'R_L')
                and a.get('split') == 'tuning' and int(a.get('seed', 42)) == 42]
    if len(selected) != 1:
        raise ValueError('Need exactly one primary tuning artifact, seed 42')
    artifact = Path(selected[0]['path']).resolve()
    found = {}
    for directory in bundles:
        bundle, rows = bundle_runs(directory)
        for row, config_path, config in rows:
            expected = paper_root / 'predictions' / row['run_directory'] / 'tuning/direct_cr_predictions.parquet'
            if expected.resolve() == artifact:
                aliases = canonical_models(row, config, bundle)
                if aliases:
                    found[row['scientific_hash']] = (row, config_path, config, aliases)
    if len(found) != 1:
        raise ValueError('Primary prediction path does not identify exactly one audited bundle fit')
    row, config_path, config, aliases = next(iter(found.values()))
    model = next((a for a in aliases if a in OWN_REFERENCE), None)
    if model is None:
        raise ValueError(f'No supported own-task control for selected fit: {aliases}')
    checkpoint = paper_root / 'neural_runs' / row['run_directory'] / 'best.ckpt'
    selection = verified_training_selection(checkpoint.parent)
    marker = json.loads((artifact.parent / 'evaluation_complete.json').read_text())
    if marker['inputs']['config_sha256'] != sha(config_path):
        raise ValueError('Selected predictions and bundle configuration differ')
    if selection['monitor'] != config['training']['eval_monitor_metric']:
        raise ValueError('Selected checkpoint monitor differs from bundle configuration')
    if marker['inputs']['checkpoint_sha256'] != selection['best_checkpoint_sha256']:
        raise ValueError('Selected primary predictions and training checkpoint differ')
    if sha(checkpoint) != selection['best_checkpoint_sha256']:
        raise ValueError('Selected checkpoint content changed')
    if marker['prediction_sha256'] != sha(artifact):
        raise ValueError('Primary predictions changed after evaluation')
    return dict(model=model, own_reference=OWN_REFERENCE[model], checkpoint=str(checkpoint),
                checkpoint_sha256=selection['best_checkpoint_sha256'], config=str(config_path),
                config_sha256=sha(config_path), predictions=str(artifact))


def complete_borrowing_export(directory, fit, command):
    directory = Path(directory)
    design = json.loads((directory / 'source_removal/design.json').read_text())
    expected = [directory / 'direct_cr_predictions.parquet'] + [
        directory / 'source_removal' / ablation / (variant.replace(':', '_') + '.parquet')
        for ablation in ('mean', 'zero') for variant in design['variants']]
    if any(not path.is_file() for path in expected):
        raise ValueError('Borrowing inference is missing a variant export')
    marker = dict(inputs=dict(schema=1, split='tuning', checkpoint_sha256=fit['checkpoint_sha256'],
        config_sha256=fit['config_sha256'], borrowing_design='drop_source_term_v1',
        horizons=','.join(map(str, design['horizons_days']))), command=command,
        prediction_sha256=sha(directory / 'direct_cr_predictions.parquet'))
    (directory / 'evaluation_complete.json').write_text(json.dumps(marker, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--analysis-root', type=Path, required=True, help='Completed main tuning workspace')
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--scores', type=Path, action='append', default=[], help='Real computed score table; repeatable')
    parser.add_argument('--score-registry', type=Path, help='Optional real registry components')
    parser.add_argument('--score-column-map', type=Path, help='Explicit approved JSON mapping for those components')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    if args.resume and args.output_dir is None:
        parser.error('--resume requires --output-dir of the existing completion workspace')
    old = args.analysis_root.resolve()
    env = load_environment(old / 'overnight.env')
    os.environ.update(env)
    paper_root = path_value(env, 'OPERA_PAPER_ROOT')
    lock_path = paper_root / '.local_figure_completion.lock'
    with lock_path.open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        root = (args.output_dir or old.parent / ('local_figures_' + datetime.now(
            ZoneInfo('Europe/Copenhagen')).strftime('%Y%m%d_%H%M%S'))).resolve()
        if root == old or old in root.parents:
            parser.error('Use a fresh sibling workspace, not the completed analysis root')
        bundles = [p for p in stage_bundles(paper_root) if (p / 'bundle.json').is_file()]
        analysis = yaml.safe_load((old / 'analysis_inventory/analysis.yaml').read_text())
        fit = selected_fit(analysis, paper_root, bundles)
        clinical_sources = [Path(analysis['patient_metadata']), *args.scores]
        if env.get('OPERA_PAPER_CLINICAL_METADATA'):
            clinical_sources.append(path_value(env, 'OPERA_PAPER_CLINICAL_METADATA'))
        clinical_sources += [p for p in [args.score_registry, args.score_column_map] if p]
        repo = Path(__file__).resolve().parents[2]
        source_code = hashlib.sha256(b''.join(p.read_bytes() for p in sorted((repo / 'opera').rglob('*.py')))).hexdigest()
        scientific_inputs = dict(source_code_sha256=source_code,
                                 clinical_source_sha256={str(p.resolve()): sha(p) for p in clinical_sources},
                                 source_analysis_sha256=sha(old / 'analysis_inventory/analysis.yaml'),
                                 fit=fit, source_env_sha256=sha(old / 'overnight.env'))
        if args.resume:
            report = json.loads((root / 'completion.json').read_text())
            if report['inputs'] != scientific_inputs:
                raise ValueError('Source analysis/environment/checkpoint changed; use a new workspace')
        else:
            root.mkdir(parents=True, exist_ok=False)
            report = dict(inputs=scientific_inputs, source_analysis=str(old), steps={}, status='running')
            from opera.run.run_tuning_plot_rebuild import copy_aggregate_inputs
            report['reused_auxiliary_aggregates'] = {name: copy_aggregate_inputs(old / name, root / name)
                for name in ('attributions', 'input_restrictions')}
            report['notes'] = ['Main paired analysis reused read-only; borrowing is freshly inferred and analysed.',
                'Existing attribution/input-restriction aggregates are reused, not newly validated.']
            (root / 'prediction_analysis').symlink_to(old / 'prediction_analysis', target_is_directory=True)
            (root / 'analysis_inventory').mkdir()
            metadata = root / 'patient_metadata.parquet'
            extras = list(args.scores)
            if env.get('OPERA_PAPER_CLINICAL_METADATA'):
                extras.append(path_value(env, 'OPERA_PAPER_CLINICAL_METADATA'))
            if bool(args.score_registry) != bool(args.score_column_map):
                parser.error('Provide both --score-registry and --score-column-map')
            if args.score_registry:
                from opera.run.compute_prognostic_scores import compute_scores
                from opera.run.analyse_locked_predictions import read_table
                computed = root / 'computed_clinical_scores.parquet'
                compute_scores(read_table(args.score_registry), json.loads(args.score_column_map.read_text())).to_parquet(computed, index=False)
                extras.append(computed)
            report['clinical_availability'] = prepare(analysis['patient_metadata'], extras, metadata)
            analysis['patient_metadata'] = str(metadata)
            (root / 'analysis_inventory/analysis.yaml').write_text(yaml.safe_dump(analysis, sort_keys=False))
            embedding, clinical = build(analysis, root / 'configs')
            embedding['calibration_maps']['routed_vectors'] = {name: str(root / 'routed_vectors' / name /
                'routed_patient_vectors_INTERNAL.npz') for name in embedding['calibration_maps']['outcomes']}
            (root / 'configs/embedding_tables.tuning.yaml').write_text(yaml.safe_dump(embedding, sort_keys=False))
            env.update(OPERA_PAPER_ANALYSIS_ROOT=str(root), OPERA_PAPER_PATIENT_METADATA=str(metadata),
                OPERA_PAPER_EMBEDDING_CONFIG=str(root / 'configs/embedding_tables.tuning.yaml'),
                OPERA_PAPER_CLINICAL_SCORE_CONFIG=str(root / 'configs/clinical_score_panels.tuning.yaml'),
                OPERA_PAPER_PLOT_TABLE_ROOT=str(root / 'figure_inputs/plot_tables'),
                OPERA_PAPER_FIGURE_CONFIG=str(root / 'figure_inputs/figures.yaml'),
                OPERA_PAPER_PLOT_SPLIT='tuning', OPERA_PAPER_FIGURES_AVAILABLE_ONLY='1',
                OPERA_PAPER_ATTRIBUTION_SUMMARY_DIR=str(root / 'attributions/summary_365d'),
                OPERA_PAPER_INPUT_RESTRICTION_DIR=str(root / 'input_restrictions'))
            env_file = root / 'overnight.env'
            env_file.write_text(''.join(f'{k}={json.dumps(str(v))}\n' for k,v in env.items()
                if k.startswith(('OPERA_', 'BONSAI_'))))
            env_file.chmod(0o600)
        env = load_environment(root / 'overnight.env')
        os.environ.update(env)
        (root / 'logs').mkdir(exist_ok=True)
        manifest = root / 'completion.json'
        def save():
            manifest.write_text(json.dumps(report, indent=2))
        def run(name, command, needs=()):
            if report['steps'].get(name, {}).get('status') == 'complete':
                for path, digest in report['steps'][name].get('output_sha256', {}).items():
                    if not Path(path).is_file() or sha(path) != digest:
                        raise ValueError(f'Completed output changed: {path}')
                return True
            if any(report['steps'].get(n, {}).get('status') != 'complete' for n in needs):
                report['steps'][name] = dict(status='blocked', dependencies=list(needs)); save(); return False
            if name == 'clinical_scores' and (root / 'clinical_scores').exists():
                (root / 'clinical_scores').rename(root / ('clinical_scores_partial_' + datetime.now().strftime('%H%M%S%f')))
            print(f'{datetime.now(ZoneInfo("Europe/Copenhagen")):%H:%M:%S} {name}', flush=True)
            report['steps'][name] = dict(status='running'); save()
            with (root / 'logs' / f'{name}.log').open('a') as log:
                result = subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT)
            if name == 'borrowing_inference' and result.returncode == 0:
                complete_borrowing_export(root / 'borrowing_inference', fit, command)
            expected = {'init_figures': [root / 'figure_inputs/figures.yaml'],
                'clinical_scores': [root / 'clinical_scores/headline_scores.csv'],
                'borrowing_inference': [root / 'borrowing_inference/source_removal/design.json', root / 'borrowing_inference/evaluation_complete.json'],
                'borrowing_analysis_zero': [root / 'source_removal/tuning/zero_ablation/borrowing_map.csv', root / 'source_removal/tuning/zero_ablation/analysis_provenance.json'],
                'borrowing_analysis': [root / 'source_removal/tuning/borrowing_map.csv', root / 'source_removal/tuning/analysis_provenance.json'],
                'embedding_tables': [root / 'figure_inputs/plot_tables/calibration_map_outcomes.json'],
                'plot_tables': [root / 'figure_inputs/plot_tables/plot_tables_manifest.json'],
                'figures': [root / 'figures/figure_manifest.json'],
                'audit': [root / 'figure_results_audit.json'],
                'supplement_tables': [root / 'supplement_tables/supplement_tables.tex'],
                'paper_numbers': [root / 'paper_numbers/results_macros.tex']}.get(name, [])
            if name.startswith('arrows_'):
                directory = root / 'routed_vectors' / name.removeprefix('arrows_')
                expected = [directory / 'routed_patient_vectors_INTERNAL.npz', directory / 'routed_patient_vectors_INTERNAL.json']
            missing = [str(p) for p in expected if not p.is_file()]
            complete = result.returncode == 0 and not missing
            report['steps'][name] = dict(status='complete' if complete else 'failed', exit_code=result.returncode,
                missing_outputs=missing, output_sha256={str(p): sha(p) for p in expected if p.is_file()})
            save(); return complete
        def module(name, *args):
            return [sys.executable, '-m', 'opera.run.' + name, *map(str, args)]
        def action(name):
            return module('paper_experiments', name, '--env-file', root / 'overnight.env', '--execute')
        save(); print(f'Workspace: {root}', flush=True)
        print('Clinical input availability: ' + json.dumps(report['clinical_availability']), flush=True)
        run('init_figures', action('init-figures'))
        run('clinical_scores', action('clinical-scores'))
        inventory = root / 'clinical_scores/score_inventory.csv'
        if inventory.is_file():
            import pandas as pd
            report['clinical_score_inventory'] = pd.read_csv(inventory).fillna('').to_dict('records')
            save()
        embed = yaml.safe_load((root / 'configs/embedding_tables.tuning.yaml').read_text())
        export_steps = []
        for outcome in embed['calibration_maps']['outcomes']:
            name = 'arrows_' + outcome
            export_steps.append(name)
            destination = root / 'routed_vectors' / outcome
            run(name, module('evaluate_direct_competing_risk', '--checkpoint', fit['checkpoint'],
                '--config-dir', Path(fit['config']).parent, '--config-name', Path(fit['config']).stem,
                '--split', 'tuning', '--horizons', '365',
                '--export-routed-outcome', outcome, '--batch-size', env.get('OPERA_PAPER_BATCH_SIZE', '1'),
                '--output-dir', destination))
        removal = root / 'borrowing_inference'
        run('borrowing_inference', module('evaluate_direct_competing_risk', '--checkpoint', fit['checkpoint'],
            '--config-dir', Path(fit['config']).parent, '--config-name', Path(fit['config']).stem,
            '--split', 'tuning', '--borrowing-design', '--batch-size', env.get('OPERA_PAPER_BATCH_SIZE', '1'),
            '--output-dir', removal))
        bundle_args = [part for path in bundles for part in ['--bundle-dir', str(path)]]
        for ablation in ('mean', 'zero'):
            run('borrowing_analysis' + ('_zero' if ablation == 'zero' else ''), module('source_removal',
                *bundle_args, '--prediction-dir', paper_root / 'predictions', '--removal-dir', removal,
                '--membership', path_value(env, 'BONSAI_COHORT_MEMBERSHIP'),
                '--output-dir', root / 'source_removal', '--split', 'tuning', '--model', fit['model'],
                '--own-reference', fit['own_reference'], '--ablation', ablation,
                '--replicates', env.get('OPERA_PAPER_REMOVAL_REPLICATES', '1000')),
                needs=['borrowing_inference'])
        run('embedding_tables', action('embedding-tables'), needs=['init_figures', *export_steps])
        run('plot_tables', action('plot-tables'), needs=['init_figures'])
        if (root / 'figures').exists() and report['steps'].get('figures', {}).get('status') != 'complete':
            (root / 'figures').rename(root / ('figures_failed_' + datetime.now().strftime('%H%M%S%f')))
        run('figures', action('figures'), needs=['plot_tables'])
        if (root / 'attributions/summary_365d').is_dir():
            run('attribution_figures', action('attribution-figures'))
        run('readiness', action('figure-readiness'), needs=['init_figures'])
        run('audit', module('audit_figure_results', '--analysis-root', root, '--output', root / 'figure_results_audit.json'))
        run('paper_numbers', action('paper-numbers'), needs=['plot_tables'])
        run('supplement_tables', action('supplement-tables'), needs=['plot_tables'])
        report['status'] = 'complete' if all(s['status'] == 'complete' for s in report['steps'].values()) and all(
            s['status'] == 'available' for s in report['clinical_availability'].values()) and all(
            s.get('status') == 'rendered' for s in report.get('clinical_score_inventory', [])) else 'finished_with_gaps'
        save(); print(f"{report['status']}: {manifest}", flush=True)


if __name__ == '__main__':
    main()
