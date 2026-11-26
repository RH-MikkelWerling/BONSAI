"""Add missing absolute-risk IG targets without repeating existing IG checks.

Uses the original calculation with a risk-only column selector. Existing four
risk targets stay in the original workspace; only missing targets are computed
here. No changes to the engine or running interpretation-worker code are needed.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import socket
import os
import uuid

from opera.run.interpretation_worker import claim, digest, progress, write_json, PatientQueue


def select_risk_columns(columns, existing_outcomes):
    selected = [c for c in columns if c[0] == 'risk' and c[1] not in existing_outcomes]
    if not selected:
        raise ValueError('No additional absolute-risk targets')
    columns[:] = selected


def prepare(source, output):
    intent = source / 'attribution_intent.json'
    run = source / 'attribution_run.json'
    if not intent.is_file() or not run.is_file():
        raise ValueError('Wait for the original startup pass to publish its intent and patient plan')
    original = json.loads(intent.read_text())
    plan = json.loads(run.read_text())['planned_patients']
    if not isinstance(plan, int) or plan < 1:
        raise ValueError('Original patient plan is not yet published')
    settings = original['settings']
    if settings['outcomes'] != 'all':
        raise ValueError('Extension requires the original all-outcome patient roster')
    # These outcomes already have absolute-risk IG in the main workspace.
    checks = settings['check_outcomes']
    if checks == 'all':
        raise ValueError('Original workspace already covers absolute risk for all outcomes')
    existing = sorted(name.strip() for name in checks.split(',') if name.strip())
    binding = {'source': str(source), 'source_intent_sha256': digest(intent),
               'baseline_sha256': digest(source / 'baseline_INTERNAL.pt'),
               'implementation_sha256': digest(Path(__file__)),
               'planned_patients': plan, 'existing_risk_outcomes': existing,
               'scope': 'Additional absolute-risk targets only; no perturbation, randomisation or alternative-baseline checks'}
    if binding['baseline_sha256'] != (source / 'baseline_sha256.txt').read_text().strip():
        raise ValueError('Original baseline changed')
    output.mkdir(parents=True, exist_ok=True)
    with claim(output / '.prepare.lock') as owned:
        if not owned:
            raise RuntimeError('Another extension preparation is active; retry shortly')
        receipt = output / 'extension_identity_INTERNAL.json'
        if receipt.exists():
            if json.loads(receipt.read_text()) != binding:
                raise ValueError('Extension inputs/code changed; do not overwrite the existing workspace')
            for name in ('baseline_INTERNAL.pt', 'baseline_sha256.txt'):
                if digest(source / name) != digest(output / name):
                    raise ValueError('Extension baseline changed')
        else:
            # Never seed a new target set with the old per-patient caches.
            if (output / 'patient_cache_INTERNAL').exists():
                raise ValueError('Unbound patient caches already exist in extension directory')
            for name in ('baseline_INTERNAL.pt', 'baseline_sha256.txt'):
                shutil.copyfile(source / name, output / name)
            write_json(output / 'attribution_run.json', {'planned_patients': plan, 'complete': False})
            write_json(receipt, binding)
    return binding, original


def verify_source_identity(new, original):
    """Permit only the explicit target/check changes, never input/model drift."""
    for key in ('torch_version', 'checkpoint_sha256', 'config', 'data_sha256', 'vocabulary_sha256'):
        if new[key] != original[key]:
            raise ValueError(f'Extension differs from original analysis: {key}')
    for name, value in original['implementation_sha256'].items():
        if new['implementation_sha256'].get(name) != value:
            raise ValueError(f'Original scientific implementation changed: {name}')
    allowed = {'check_outcomes', 'sanity_patients', 'faithfulness_patients',
               'alternative_baseline', 'no_deletion'}
    if {k: v for k, v in new['settings'].items() if k not in allowed} != {
            k: v for k, v in original['settings'].items() if k not in allowed}:
        raise ValueError('Extension changed settings beyond the additional targets/check suppression')


def run(source, output, *, collect=False):
    from opera.run import explain_direct_competing_risk as engine
    binding, original = prepare(source, output)
    with claim(output / '.attribution.lock', shared=not collect) as owned:
        if not owned:
            raise RuntimeError('Extension collector/helpers are active; wait before switching roles')
        if collect and not progress(output)['complete']:
            raise ValueError('Finish all extension patient caches before collecting')
        destination = output
        if not collect:
            destination = output / 'workers_INTERNAL' / f'{socket.gethostname()}-{os.getpid()}-{uuid.uuid4().hex[:8]}'
            destination.mkdir(parents=True)
            for name in ('baseline_INTERNAL.pt', 'baseline_sha256.txt'):
                shutil.copyfile(output / name, destination / name)
        settings = original['settings']
        args = engine.build_parser().parse_args([
            '--checkpoint', settings['checkpoint'], '--config-name', settings['config_name'],
            '--output-dir', str(destination)])
        if set(settings) - set(vars(args)):
            raise ValueError('Original argument schema changed')
        vars(args).update(settings)
        args.checkpoint, args.config_dir = Path(args.checkpoint), Path(args.config_dir)
        args.output_dir, args.resume, args.max_new_patients = destination, True, None
        args.check_outcomes = 'all'
        args.sanity_patients = args.faithfulness_patients = 0
        args.alternative_baseline, args.no_deletion = '', True
        args.token_budget = 16384
        original_explainer = engine.Explainer
        original_verify = engine.verify_intent
        original_files = engine.implementation_files
        original_cached = engine.cached_explanation

        def risk_only_explainer(*positional, columns, check_columns, **kwargs):
            # The engine owns these lists; update them before it constructs the
            # explainer so target values, metadata and saved names all agree.
            select_risk_columns(columns, binding['existing_risk_outcomes'])
            check_columns[:] = []
            return original_explainer(*positional, columns=columns, check_columns=check_columns, **kwargs)

        def verify(args, cfg, vocabulary):
            original_verify(args, cfg, vocabulary)
            new = json.loads((destination / 'attribution_intent.json').read_text())
            verify_source_identity(new, original)
            with (output / '.intent.lock').open('a') as handle:
                import fcntl
                fcntl.flock(handle, fcntl.LOCK_EX)
                central = output / 'attribution_intent.json'
                if central.exists() and json.loads(central.read_text()) != new:
                    raise ValueError('Extension workers disagree on frozen intent')
                if not central.exists():
                    write_json(central, new)

        queue = None if collect else PatientQueue(output, original_cached, destination)
        engine.Explainer = risk_only_explainer
        engine.verify_intent = verify
        engine.implementation_files = lambda: sorted(set(original_files() + [Path(__file__).resolve()]))
        if queue is not None:
            engine.cached_explanation = queue
        try:
            engine._run(args)
            if queue is not None and queue.visited != binding['planned_patients']:
                raise ValueError('Extension patient plan differs from original')
            if queue is not None:
                write_json(destination / 'status.json', {'status': 'pass_complete', 'completed_here': queue.completed})
        finally:
            engine.Explainer, engine.verify_intent = original_explainer, original_verify
            engine.implementation_files, engine.cached_explanation = original_files, original_cached


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['prepare', 'work', 'collect', 'status'])
    parser.add_argument('--source', type=Path, default=Path('data/cutoff_recovery_20261121/final_interpretation/attributions'))
    parser.add_argument('--output-dir', type=Path, default=Path('data/cutoff_recovery_20261121/absolute_risk_ig_extension'))
    args = parser.parse_args()
    source, output = args.source.resolve(), args.output_dir.resolve()
    if source == output:
        parser.error('Use a separate extension directory; never modify the live main workspace')
    if args.action == 'status':
        print(json.dumps(progress(output), indent=2))
    elif args.action == 'prepare':
        print(json.dumps(prepare(source, output)[0], indent=2))
    else:
        run(source, output, collect=args.action == 'collect')


if __name__ == '__main__':
    main()
