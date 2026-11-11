"""Two-allocation CPU handoff for the already running local test scoring shards.

The numerical allocation builds the common inventory, runs numerical analysis,
then joins the independent clinical allocation before rendering/numbers. Shared
markers coordinate tasks; neither role launches GPU inference or reads a new
test scope. Run pause_test_coordinator.py on shard zero first.
"""
from __future__ import annotations
import argparse
import csv
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import yaml
from dotenv import dotenv_values


def atomic_json(path, value):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2) + '\n')
    tmp.replace(path)


def validate_scoring(rows, expected):
    names = [row['name'] for row in rows]
    if len(names) != len(set(names)) or set(names) != set(expected):
        raise ValueError('Scoring report does not match this frozen fit shard')
    bad = [(r['name'], r['status']) for r in rows
           if r['status'] not in {'evaluated', 'reused_verified_evaluation', 'not_estimable'}]
    if bad:
        raise ValueError(f'Scoring is incomplete: {bad}')


def check_local_contrasts(path):
    required = {'P1','P2','P5','T2','O1','O2','router_borrow_AB',
                'O2_MATCHED','OFF_J','OAOWN_OFF','O_CORRECTIONS_MATCHED'}
    found = set()
    with path.open() as handle:
        for row in csv.DictReader(handle):
            if (row['split'] == 'held_out' and float(row['seed']) == 42
                    and row['metric'] == 'auc' and float(row['horizon_days']) == 0
                    and row['stratum'] == 'all_cells'
                    and row['aggregation'] == 'equal_cell_not_population_or_endpoint_group_macro'
                    and row.get('gain') and math.isfinite(float(row['gain']))):
                found.add(row['contrast'])
    if required - found:
        raise ValueError('Missing required local contrasts: ' + ', '.join(sorted(required - found)))


def wait_for(predicate, *, deadline, failed=()):
    while not predicate():
        for path in failed:
            if path.exists():
                raise RuntimeError(f'Another stage failed: {path}')
        if time.monotonic() > deadline:
            raise TimeoutError('Timed out waiting for the other allocation')
        time.sleep(15)


def prepare(args, root):
    from opera.run.paper_prediction_queue import bundle_runs
    from scripts.set_test_env import selected_fit
    env = {k: v for k, v in dotenv_values(args.env_file).items() if v is not None}
    if env.get('OPERA_PAPER_OPERA_MODEL') != 'J_OA_OWN_L':
        raise ValueError('This staged handoff is for the frozen local J_OA_OWN_L release')
    if env.get('OPERA_PAPER_PLOT_SPLIT') != 'held_out':
        raise ValueError('Run the agreed set_test_env.py held-out setup first')
    bundle, rows = bundle_runs(args.paper_root / 'bundles' / f'stage{args.stage}')
    row, config = selected_fit(bundle, rows, 'J_OA_OWN_L')
    checkpoint = args.paper_root / 'neural_runs' / row['run_directory'] / 'best.ckpt'
    source = yaml.safe_load(Path(env['OPERA_PAPER_CLINICAL_SOURCE_ANALYSIS']).read_text())
    if not source.get('clinical_threshold_artifacts'):
        raise ValueError('Missing registered clinical training thresholds')
    env.update(OPERA_PAPER_ANALYSIS_ROOT=str(root.resolve()),
        OPERA_PAPER_PATIENT_METADATA=source['patient_metadata'],
        OPERA_PAPER_ANALYSIS_ESTIMAND='cif_ipcw',
        OPERA_PAPER_ANALYSIS_HORIZONS='30,60,90,180,365,730',
        OPERA_PAPER_TABULAR_HORIZONS='30,60,90,180,365,730',
        OPERA_PAPER_INVENTORY_SPLITS='tuning,held_out',
        OPERA_PAPER_ROUTED_MODEL='R_AB_L',
        OPERA_PAPER_TASK_STRUCTURE_CHECKPOINT=str(checkpoint.resolve()),
        OPERA_PAPER_ATTRIBUTION_CHECKPOINT=str(checkpoint.resolve()),
        OPERA_PAPER_ATTRIBUTION_CONFIG_DIR=str(config.parent.resolve()),
        OPERA_PAPER_ATTRIBUTION_CONFIG_NAME=config.stem,
        OPERA_PAPER_FIGURES_AVAILABLE_ONLY='1',
        OPERA_ANALYSIS_WORKERS=str(args.analysis_workers),
        OPERA_SUMMARY_WORKERS=str(args.summary_workers),
        OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
        NUMEXPR_NUM_THREADS='1')
    recipe = root / 'release_clinical.yaml'
    shutil.copy2(env['OPERA_PAPER_CLINICAL_SCORE_CONFIG'], recipe)
    env['OPERA_PAPER_CLINICAL_SCORE_CONFIG'] = str(recipe.resolve())
    shutil.copy2(args.fit_names, root / 'release_fit_names.txt')
    (root / 'overnight.env').write_text(''.join(f'{k}={v}\n' for k, v in env.items()))
    files = {}
    for directory in ('opera', 'bonsai', 'scripts'):
        for p in Path(directory).rglob('*'):
            if p.is_file() and p.suffix in {'.py','.sh','.yaml','.json','.txt'}:
                files[str(p)] = hashlib.sha256(p.read_bytes()).hexdigest()
    atomic_json(root / 'release_source_hashes.json', files)
    atomic_json(root / 'release_record.json', dict(primary_model='J_OA_OWN_L', split='held_out',
        stage=args.stage, horizons=[30,60,90,180,365,730], bootstrap_replicates=1000,
        bootstrap_seed=2026, bootstrap_method='multiplier',
        fit_names_sha256=hashlib.sha256(args.fit_names.read_bytes()).hexdigest(),
        clinical_sha256=hashlib.sha256(recipe.read_bytes()).hexdigest(), cpu_handoff=True))
    atomic_json(root / 'cpu_execution.json', dict(analysis_workers=args.analysis_workers,
        summary_workers=args.summary_workers,
        note='Computational scheduling only; bootstrap draws/weights/contrasts remain unchanged.'))
    return source


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tag', required=True)
    parser.add_argument('--role', choices=['numerical', 'clinical'], required=True)
    parser.add_argument('--shards', type=int, default=4)
    parser.add_argument('--stage', type=int, default=16)
    parser.add_argument('--paper-root', type=Path, default=Path('data/paper_256_v3'))
    parser.add_argument('--env-file', type=Path, default=Path('.env'))
    parser.add_argument('--fit-names', type=Path,
        default=Path('opera/configs/releases/local_test_fit_names_20261109.txt'))
    parser.add_argument('--analysis-workers', type=int, default=1)
    parser.add_argument('--summary-workers', type=int, default=2)
    parser.add_argument('--wait-hours', type=float, default=48)
    parser.add_argument('--resume-analysis', action='store_true',
        help='Numerical recovery only: reuse the completed inventory and clinical results.')
    parser.add_argument('--distributed', action='store_true',
        help='Save prepared inputs and delegate paired batches to analysis_cpu_worker nodes.')
    args = parser.parse_args()
    if Path(args.tag).name != args.tag or args.shards < 1 or min(args.analysis_workers, args.summary_workers) < 1:
        raise ValueError('Invalid tag or shard count')
    root = args.paper_root / 'analysis' / args.tag
    marks = args.paper_root / 'predictions/.overnight' / args.tag
    marks.mkdir(parents=True, exist_ok=True)
    deadline = time.monotonic() + args.wait_hours * 3600
    with (marks / f'cpu_{args.role}.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        failed = marks / f'cpu_{args.role}.failed'
        if args.resume_analysis:
            if args.role != 'numerical':
                raise ValueError('--resume-analysis requires the numerical role')
            if not (marks / 'cpu_inventory_ready').exists() or not (root / 'analysis_inventory/analysis.yaml').is_file():
                raise RuntimeError('No completed inventory to resume')
            if (marks / 'cpu_numerical_done').exists():
                raise RuntimeError('Numerical analysis already completed; inspect downstream failure instead')
            if failed.exists():
                failed.rename(failed.with_name(failed.name + f'.previous_{time.time_ns()}'))
        elif failed.exists():
            raise RuntimeError(f'Previous role failed; inspect {failed} before retrying')
        env = {**os.environ, 'CUDA_VISIBLE_DEVICES': '',
               'OPERA_ANALYSIS_WORKERS': str(args.analysis_workers),
               'OPERA_SUMMARY_WORKERS': str(args.summary_workers),
               'OMP_NUM_THREADS':'1', 'OPENBLAS_NUM_THREADS':'1', 'MKL_NUM_THREADS':'1'}
        if args.distributed:
            if args.role != 'numerical':
                raise ValueError('--distributed requires the numerical role')
            env['OPERA_ANALYSIS_CHECKPOINT_DIR'] = str((root / '_cpu_checkpoints_INTERNAL').resolve())
        def action(name):
            print(f'{args.role}: {name}', flush=True)
            with (root / 'cpu_logs' / f'{name}.log').open('w') as log:
                subprocess.run([sys.executable, '-u', '-m', 'opera.run.paper_experiments', name,
                    '--stage', str(args.stage), '--env-file', str(root / 'overnight.env'), '--execute'],
                    env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
        try:
            if args.role == 'numerical':
                if not args.resume_analysis:
                    handoff = marks / 'cpu_handoff.json'
                    if not handoff.exists():
                        raise RuntimeError('Pause shard-zero coordinator first with pause_test_coordinator.py')
                    names = args.fit_names.read_text().splitlines()
                    shard_zero = args.paper_root / 'predictions/queue_held_out_shard0.json'
                    def scoring_done():
                        if not shard_zero.exists():
                            return False
                        try:
                            rows = json.loads(shard_zero.read_text())
                        except json.JSONDecodeError:
                            return False
                        # Stale reports with a different scope cannot unlock analysis.
                        if {r['name'] for r in rows} != set(names[::args.shards]):
                            return False
                        validate_scoring(rows, names[::args.shards])
                        return all((marks / f'shard_{i}_of_{args.shards}.done').is_file()
                                   for i in range(1, args.shards))
                    print('Waiting for all scoring shards; the paused shell does not pause GPU children.', flush=True)
                    wait_for(scoring_done, deadline=deadline,
                        failed=[marks / f'shard_{i}_of_{args.shards}.failed' for i in range(args.shards)])
                    for i in range(args.shards):
                        validate_scoring(json.loads((args.paper_root / f'predictions/queue_held_out_shard{i}.json').read_text()),
                                         names[i::args.shards])
                    record = json.loads(handoff.read_text()); record['scoring_complete'] = True
                    atomic_json(handoff, record)
                    (marks / f'shard_0_of_{args.shards}.done').touch()
                    root.mkdir(parents=True, exist_ok=False)
                    (root / 'cpu_logs').mkdir()
                    source = prepare(args, root)
                    action('inventory')
                    path = root / 'analysis_inventory/analysis.yaml'
                    config = yaml.safe_load(path.read_text())
                    config.update(bootstrap_method='multiplier', bootstrap_replicates=1000, bootstrap_seed=2026,
                        clinical_threshold_artifacts=source['clinical_threshold_artifacts'],
                        patient_metadata_column_map=source.get('patient_metadata_column_map', {}))
                    path.write_text(yaml.safe_dump(config, sort_keys=False))
                    (marks / 'cpu_inventory_ready').touch()
                action('analyse')
                check_local_contrasts(root / 'prediction_analysis/effect_summaries.csv')
                (marks / 'cpu_numerical_done').touch()
                print('Numerical results ready. Waiting for the independent clinical allocation.', flush=True)
                wait_for(lambda: (marks / 'cpu_clinical_done').exists(), deadline=deadline,
                         failed=[marks / 'cpu_clinical.failed'])
                for name in ['plot-tables','task-structure','paper-numbers','figure-readiness','init-figures','figures']:
                    action(name)
                (root / 'pipeline_actions_complete').touch()
            else:
                print('Waiting for the prepared shared clinical inventory.', flush=True)
                wait_for(lambda: (marks / 'cpu_inventory_ready').exists(), deadline=deadline,
                         failed=[marks / 'cpu_numerical.failed'])
                action('clinical-scores')
                (marks / 'cpu_clinical_done').touch()
            print(f'{args.role}: complete', flush=True)
        except BaseException as error:
            atomic_json(failed, dict(error=str(error), role=args.role))
            raise


if __name__ == '__main__':
    main()
