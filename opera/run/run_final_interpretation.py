"""Resume final adapted-OPERA IG, restrictions and native plotting in one workspace.

Uses the existing explanation and figure entry points; never trains a model.
The pilot and full run retain the same planned patients and frozen settings.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import yaml


def read(path):
    return yaml.load(Path(path).read_text(), Loader=getattr(yaml, 'CSafeLoader', yaml.SafeLoader))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def prepare(root, output, base_env):
    from opera.run.paper_experiments import load_environment
    from bonsai.functional.checkpointing import verified_training_selection
    bundle = root / 'adapted_fits'
    row, = [r for r in read(bundle / 'bundle.json')['runs'] if r['name'] == 'seed42/EA/R']
    config = bundle / row['config_path']
    if sha(config) != row['config_sha256']:
        raise ValueError('Frozen R_EA config changed')
    fit = root / 'neural_runs' / row['run_directory']
    selected = verified_training_selection(fit)
    checkpoint = fit / 'best.ckpt'
    identity = {'model': 'R_EA', 'split': 'held_out', 'config': str(config),
                'config_sha256': row['config_sha256'], 'checkpoint': str(checkpoint),
                'checkpoint_sha256': selected['best_checkpoint_sha256']}
    recipe = read(Path(__file__).resolve().parents[1] / 'configs/attribution_analysis.example.yaml')
    # Select checks by head availability only, not by observed performance.
    import pyarrow.parquet as pq
    predictions = root / 'predictions' / row['run_directory'] / 'held_out/direct_cr_predictions.parquet'
    available = set()
    for batch in pq.ParquetFile(predictions).iter_batches(columns=['outcome'], batch_size=8192):
        available.update(batch.column(0).to_pylist())
    checks = ['overall_survival', 'infection_diagnosis_or_antimicrobial_use_composite',
              'serious_infection_composite', 'neutropenia_g3plus']
    recipe['run'].update(split='held_out', outcomes='all', horizons=[365],
                         check_outcomes=[name for name in checks if name in available])
    receipt = output / 'interpretation_manifest.json'
    if output.exists():
        previous = read(receipt)
        if previous['fit'] != identity or read(output / 'attribution.yaml') != recipe:
            raise ValueError('Existing interpretation workspace belongs to different inputs/settings')
        return previous
    env = load_environment(base_env)
    output.mkdir(parents=True)
    recipe_path = output / 'attribution.yaml'
    recipe_path.write_text(yaml.safe_dump(recipe, sort_keys=False))
    env.update(OPERA_PAPER_ROOT=str(root), OPERA_PAPER_ANALYSIS_ROOT=str(output),
               OPERA_PAPER_OPERA_MODEL='R_EA', OPERA_PAPER_PLOT_SPLIT='held_out',
               OPERA_PAPER_FIGURE_MARK='', OPERA_PAPER_ATTRIBUTION_CONFIG=str(recipe_path),
               OPERA_PAPER_ATTRIBUTION_CHECKPOINT=str(checkpoint),
               OPERA_PAPER_ATTRIBUTION_CONFIG_DIR=str(config.parent),
               OPERA_PAPER_ATTRIBUTION_CONFIG_NAME=config.stem,
               OPERA_PAPER_ATTRIBUTION_SUMMARY_DIR=str(output / 'attributions/summary_365d'),
               OPERA_PAPER_INPUT_RESTRICTION_DIR=str(output / 'input_restrictions'))
    env_path = output / 'interpretation.env'
    env_path.write_text(''.join(f'{key}={json.dumps(str(value))}\n' for key, value in env.items()
                                if key.startswith(('OPERA_', 'BONSAI_'))))
    env_path.chmod(0o600)
    result = {'fit': identity, 'status': 'prepared', 'steps': {},
              'scope': 'Final-model held-out interpretation; no fitting or analysis-lock change',
              'restriction_reuse': 'Existing restriction outputs are reused by the original entry point; same-workspace inputs are immutable.'}
    receipt.write_text(json.dumps(result, indent=2))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--national-root', type=Path, default=Path('data/paper_256_v3_national_20261111'))
    parser.add_argument('--output-dir', type=Path, default=Path('data/paper_256_v3_national_20261111/final_interpretation_20261116'))
    parser.add_argument('--base-env', type=Path, default=Path('data/paper_256_v3/analysis/local_test_20261109_four_allocations/overnight.env'))
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--pilot-only', action='store_true')
    parser.add_argument('--pilot-new-patients', type=int, default=50,
                        help='Startup cap for additional uncached patients; 1 publishes the plan quickly before distributed IG')
    parser.add_argument('--distributed-ig', action='store_true',
                        help='After the pilot, wait for interpretation_worker helpers before collecting IG')
    args = parser.parse_args()
    if args.pilot_new_patients < 1:
        parser.error('--pilot-new-patients must be positive')
    root, output = args.national_root.resolve(), args.output_dir.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    import fcntl
    with (output.parent / ('.' + output.name + '.lock')).open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        report = prepare(root, output, args.base_env)
        base = [sys.executable, '-u', '-m', 'opera.run.paper_experiments']
        def command(action, *extra):
            return [*base, action, '--env-file', str(output / 'interpretation.env'), *extra, '--execute']
        commands = [('pilot', command('attributions', '--attribution-max-new-patients', str(args.pilot_new_patients)))]
        if not args.pilot_only:
            if args.distributed_ig:
                commands += [('ig_workers', [sys.executable, '-u', '-m', 'opera.run.interpretation_worker',
                                            'wait', '--workspace', str(output)])]
            commands += [('ig', command('attributions')), ('summary', command('attribution-summary')),
                         ('restrictions', command('input-restrictions')), ('figures', command('attribution-figures'))]
        print(json.dumps({'workspace': str(output), 'fit': report['fit'], 'commands': commands}, indent=2), flush=True)
        if not args.execute:
            return
        logs = output / 'logs'; logs.mkdir(exist_ok=True)
        receipt = output / 'interpretation_manifest.json'
        def save():
            temporary = receipt.with_suffix('.json.tmp')
            temporary.write_text(json.dumps(report, indent=2))
            os.replace(temporary, receipt)
        environment = {**os.environ, 'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1',
                       'MKL_NUM_THREADS': '1', 'OPERA_ANALYSIS_WORKERS': '1',
                       'MPLBACKEND': 'Agg', 'OPERA_PAPER_EXPORT_PLOT_STATE': '1'}
        report['status'] = 'running'; save()
        try:
            for name, cmd in commands:
                # Queue readiness must be checked on every resume, not treated as
                # a completed scientific analysis step from an earlier attempt.
                if name == 'ig_workers':
                    report['steps'].pop(name, None)
                if report['steps'].get(name, {}).get('status') == 'complete':
                    continue
                print(f'Interpretation step: {name}; log: {logs / (name + ".log")}', flush=True)
                with (logs / (name + '.log')).open('a') as handle:
                    if name == 'restrictions':
                        # Helpers evaluate separate restrictions; final serial
                        # dispatcher waits, skips these, then compares once.
                        with (output / '.restriction-helpers.lock').open('a') as gate:
                            fcntl.flock(gate, fcntl.LOCK_EX)
                            result = subprocess.run(cmd, stdout=handle, stderr=subprocess.STDOUT, env=environment,
                                                    pass_fds=(gate.fileno(),))
                    else:
                        result = subprocess.run(cmd, stdout=handle, stderr=subprocess.STDOUT, env=environment)
                report['steps'][name] = {'status': 'complete' if result.returncode == 0 else 'failed',
                                         'returncode': result.returncode}; save()
                if result.returncode:
                    raise RuntimeError(f'{name} failed; inspect {logs / (name + ".log")}')
                if name == 'ig' and not read(output / 'attributions/attribution_run.json').get('complete'):
                    report['steps'][name]['status'] = 'incomplete'; save()
                    raise RuntimeError('Full IG pass did not complete its planned sample; summary not launched')
            report['status'] = 'pilot_complete' if args.pilot_only else 'complete'
        except BaseException as error:
            report.update(status='failed', error=str(error))
            raise
        finally:
            save()
        print(f'Interpretation status: {report["status"]}; workspace: {output}', flush=True)


if __name__ == '__main__':
    main()
