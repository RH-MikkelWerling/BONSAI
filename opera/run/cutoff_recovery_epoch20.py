"""Prioritise corrected epoch-20 joint/OPERA checks without changing frozen fits."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import socket
import subprocess
import sys
import time

from opera.run.cutoff_recovery_worker import read, completed, encoder_ready
from opera.run.finish_figure_inputs import immutable, sha


def reference_job(plan, model):
    choices = [(digest, job) for digest, job in plan['jobs'].items()
               if model in job['models'] and int(job['row']['seed']) == 42
               and job.get('pretraining_epochs') == 10]
    if len(choices) != 1:
        raise ValueError(f'Need one corrected seed-42 epoch10 {model} fit, found {len(choices)}')
    return choices[0]


def scientific_recipe(config):
    config = copy.deepcopy(config)
    config.pop('dapt_ckpt', None)
    config.pop('cutoff_recovery', None)
    return config


def prepare(root, architecture='joint'):
    from opera.run.prepare_epoch20_joint_finetune import clone_joint_config
    from opera.run.prepare_paper_run_bundle import write_bundle
    root = root.resolve()
    plan = read(root/'queue_plan.yaml')
    if sha(Path(plan['bundle'])/'bundle.json') != plan['bundle_sha256']:
        raise ValueError('Frozen recovery bundle changed')
    if architecture not in {'joint', 'opera'}:
        raise ValueError('Choose joint or opera')
    directory = root/'epoch20_priority'
    if architecture == 'opera':
        directory = directory/'opera_extension'
    directory.mkdir(parents=True, exist_ok=True)
    entries, specifications, references = [], {}, []
    auxiliary = directory/'additional_fits'
    designs = [('cohort', 'J_L', 'J_L_epoch20'), ('adapted', 'J_EA', 'J_EA_epoch20')]
    if architecture == 'opera':
        designs = [('cohort', 'J_OA_OWN_L', 'J_OA_OWN_L_epoch20'),
                   ('adapted', 'R_EA', 'R_EA_epoch20')]
    for family, model, alias in designs:
        digest, job = reference_job(plan, model)
        path = Path(plan['bundle'])/job['row']['config_path']
        if sha(path) != job['row']['config_sha256']:
            raise ValueError('Corrected reference config changed')
        source = read(path)
        candidate = clone_joint_config(source, root/family/'pretraining/version_0/last.ckpt')
        candidate['cutoff_recovery'].update(expected_epochs=20, export_embeddings=False)
        # Reuse an already queued cohort20 fit if it is exactly matched. Do not
        # create a second training job just to give it higher execution priority.
        existing = []
        for key, old in plan['jobs'].items():
            if old.get('family') != family or old.get('pretraining_epochs') != 20 or int(old['row']['seed']) != 42:
                continue
            old_path = Path(plan['bundle'])/old['row']['config_path']
            if sha(old_path) != old['row']['config_sha256']:
                raise ValueError('Queued epoch20 configuration changed')
            old_config = read(old_path)
            if (old_config['dapt_ckpt'] == candidate['dapt_ckpt']
                    and scientific_recipe(old_config) == scientific_recipe(candidate)):
                existing.append((key, old))
        if len(existing) > 1:
            raise ValueError('Multiple matched epoch20 fits; inspect instead of duplicating work')
        reference = dict(family=family, model=alias, reference_model=model,
                         reference_config_sha256=sha(path))
        if existing:
            key, _ = existing[0]
            reference.update(queue_root=str(root), job=key, reused_registered_fit=True)
        else:
            key = hashlib.sha256(json.dumps(candidate, sort_keys=True).encode()).hexdigest()
            specifications[key] = dict(config=candidate, family=family, models=[alias],
                splits=['held_out', 'tuning'], removal_splits=[], pretraining_epochs=20,
                priority=-1, dependencies=[], source_config=str(path), source_config_sha256=sha(path))
            entries.append(dict(name=f'seed42/{job["row"]["origin"]}/{alias}',
                phase='supervision', origin=job['row']['origin'], seed=42, condition={},
                config=candidate, recovery_models=[alias], source_id=key))
            reference.update(queue_root=str(auxiliary), job=key, reused_registered_fit=False)
        references.append(reference)
    if entries:
        bundle = auxiliary/'bundles/epoch20'
        if (bundle/'bundle.json').is_file():
            manifest = json.loads((bundle/'bundle.json').read_text())
            expected = {e['source_id']: e['config'] for e in entries}
            actual = {r['source_id']: read(bundle/r['config_path']) for r in manifest['runs']}
            if expected != actual:
                raise ValueError('Epoch20 add-on changed; do not overwrite its fits')
        else:
            manifest = write_bundle(entries, bundle, provenance=dict(scope=f'Corrected epoch20 exploratory {architecture} checks'))
        for row in manifest['runs']:
            if sha(bundle/row['config_path']) != row['config_sha256']:
                raise ValueError('Epoch20 add-on config changed')
            specifications[row['source_id']]['row'] = row
        (auxiliary/'jobs').mkdir(exist_ok=True)
        for key, spec in specifications.items():
            spec.pop('config')
            immutable(auxiliary/'jobs'/f'{key}.yaml', spec)
        immutable(auxiliary/'queue_plan.yaml', dict(bundle=str(bundle), bundle_sha256=sha(bundle/'bundle.json'),
                   jobs=specifications, settings='Matched corrected epoch10 downstream recipes; encoder checkpoint only'))
    receipt = dict(scope=f'Exploratory corrected epoch20 {architecture} checks; no primary alias changes',
                   recovery_plan_sha256=sha(root/'queue_plan.yaml'), fits=references)
    immutable(directory/'manifest.yaml', receipt)
    return receipt


def registry(root):
    paths = [root/'epoch20_priority/manifest.yaml', root/'epoch20_priority/opera_extension/manifest.yaml']
    receipts = [read(path) for path in paths if path.is_file()]
    if not receipts:
        return None
    for receipt in receipts:
        if receipt['recovery_plan_sha256'] != sha(root/'queue_plan.yaml'):
            raise ValueError('Epoch20 priority registry belongs to another recovery queue')
    result = copy.deepcopy(receipts[0])
    result['fits'] = [fit for receipt in receipts for fit in receipt['fits']]
    if len({r['model'] for r in result['fits']}) != len(result['fits']):
        raise ValueError('Duplicate epoch20 aliases')
    return result


def selected_fits(receipt, architecture=None):
    if architecture is None:
        return receipt['fits']
    models = {'J_L', 'J_EA'} if architecture == 'joint' else {'J_OA_OWN_L', 'R_EA'}
    return [r for r in receipt['fits'] if r['reference_model'] in models]


def fit_details(reference):
    base = Path(reference['queue_root'])
    plan = read(base/'queue_plan.yaml')
    job = plan['jobs'][reference['job']]
    if sha(Path(plan['bundle'])/'bundle.json') != plan['bundle_sha256']:
        raise ValueError('Epoch20 fit bundle changed')
    path = Path(plan['bundle'])/job['row']['config_path']
    if sha(path) != job['row']['config_sha256']:
        raise ValueError('Epoch20 fit config changed')
    return base, plan, job, read(path)


def try_one(root, worker, family=None, architecture=None):
    """Use the exact normal claim/receipt path; concurrent workers cannot duplicate."""
    from opera.run.paper_worker import Claim, record_attempt, record_failure, now
    receipt = registry(root)
    if receipt is None:
        return 'not_prepared'
    for reference in selected_fits(receipt, architecture):
        if family and family != reference['family']:
            continue
        base, plan, job, config = fit_details(reference)
        digest = reference['job']
        if completed(base, digest, job):
            continue
        if not encoder_ready(config):
            continue
        run = base/'neural_runs'/job['row']['run_directory']
        run.mkdir(parents=True, exist_ok=True)
        failed = run/'worker_failed.json'
        if failed.is_file() and read(failed).get('attempts', 0) >= 2:
            raise RuntimeError(f'Epoch20 fit needs inspection: {run/"recovery_job.log"}')
        claim = Claim(run, worker, 60)
        if not claim.acquire():
            continue
        started, status = now(), 'failed'
        try:
            if not completed(base, digest, job):
                print(f'URGENT EPOCH20 {reference["model"]}: {run}', flush=True)
                with (run/'recovery_job.log').open('a') as log:
                    subprocess.run([sys.executable, '-u', '-m', 'opera.run.cutoff_recovery_worker',
                        'one-job', '--root', str(base), '--job', digest], check=True,
                        stdout=log, stderr=subprocess.STDOUT)
            if not completed(base, digest, job):
                raise RuntimeError('Epoch20 fit exited without verified evaluations')
            status = 'evaluated'
            failed.unlink(missing_ok=True)
        except Exception as error:
            record_failure(run, worker, error)
            raise
        finally:
            record_attempt(run, worker, started, status)
            claim.release()
        return 'processed'
    return 'waiting_or_finished'


def status(root, family=None, architecture=None):
    receipt = registry(root)
    if receipt is None:
        return dict(status='not_prepared')
    records = []
    for reference in selected_fits(receipt, architecture):
        if family and reference['family'] != family:
            continue
        base, _, job, config = fit_details(reference)
        run = base/'neural_runs'/job['row']['run_directory']
        state = ('evaluated' if completed(base, reference['job'], job) else
                 'claimed_check_heartbeat' if (run/'.worker_claim.json').exists() else
                 'ready' if encoder_ready(config) else 'waiting_for_epoch20_encoder')
        records.append(dict(model=reference['model'], status=state, run=str(run)))
    return dict(status=('not_prepared' if not records else 'complete'
                if all(r['status'] == 'evaluated' for r in records) else 'pending'), fits=records)


def prepare_analysis(root, architecture='joint'):
    """Called sequentially by the existing high-RAM results coordinator."""
    from opera.run.cutoff_recovery_results import rebound_artifacts, deduplicate_artifacts
    receipt = registry(root)
    if receipt is None:
        return dict(status='not_estimable', note='Epoch20 priority checks not prepared')
    references = selected_fits(receipt, 'joint') if architecture == 'joint' else receipt['fits']
    required = {'J_L_epoch20', 'J_EA_epoch20'}
    if architecture == 'opera':
        required.update({'J_OA_OWN_L_epoch20', 'R_EA_epoch20'})
    if not required <= {r['model'] for r in references}:
        return dict(status='waiting', note='Prepare both joint and OPERA epoch20 checks')
    pending = [r['model'] for r in references if not completed(
        Path(r['queue_root']), r['job'], fit_details(r)[2])]
    if pending:
        return dict(status='waiting', models=pending)
    plan = read(root/'queue_plan.yaml')
    sources = read(plan['analyses'])
    wanted = {'J_L', 'J_EA', 'J_E', 'R_EA', 'X', 'XP'}
    if architecture == 'opera':
        wanted.add('J_OA_OWN_L')
    declarations = [d for source in sources for d in source['artifacts']
        if d['artifact']['model'] in wanted and int(d['artifact'].get('seed', 42)) == 42
        and d['artifact'].get('split') in {'tuning', 'held_out'}]
    artifacts, _, waiting, _ = rebound_artifacts(root, plan, declarations)
    if waiting:
        return dict(status='waiting', jobs=waiting)
    for reference in references:
        base, _, job, _ = fit_details(reference)
        for split in ('tuning', 'held_out'):
            templates = [a for a in artifacts if a['model'] == reference['reference_model'] and a['split'] == split]
            if len(templates) != 1:
                raise ValueError('Need one matched epoch10 prediction contract')
            declaration = copy.deepcopy(templates[0])
            native = reference['model'] if reference['model'] in job['models'] else job['models'][0]
            declaration.update(model=native, path=str(base/'predictions'/job['row']['run_directory']/split/'direct_cr_predictions.parquet'),
                               format='direct_cif', cohort_from_metadata=True)
            if native != reference['model']:
                declaration['model_alias'] = reference['model']
            artifacts.append(declaration)
    directory = root/'analyses'/('epoch20_check' if architecture == 'joint' else 'epoch20_architecture_check')
    directory.mkdir(parents=True, exist_ok=True)
    contrasts = dict(COHORT_20_MINUS_10='gain(J_L_epoch20,J_L)',
        ADAPTED_20_MINUS_10='gain(J_EA_epoch20,J_EA)',
        ADAPTED20_MINUS_COHORT20='gain(J_EA_epoch20,J_L_epoch20)')
    present = {a['model'] for a in artifacts}
    if architecture == 'opera':
        contrasts.update(
            COHORT_OPERA_20_MINUS_10='gain(J_OA_OWN_L_epoch20,J_OA_OWN_L)',
            ADAPTED_OPERA_20_MINUS_10='gain(R_EA_epoch20,R_EA)',
            ARCHITECTURE_COHORT20='gain(J_OA_OWN_L_epoch20,J_L_epoch20)',
            ARCHITECTURE_ADAPTED20='gain(R_EA_epoch20,J_EA_epoch20)',
            ARCH_GAIN_CHANGE_COHORT='gain(J_OA_OWN_L_epoch20,J_L_epoch20)-gain(J_OA_OWN_L,J_L)',
            ARCH_GAIN_CHANGE_ADAPTED='gain(R_EA_epoch20,J_EA_epoch20)-gain(R_EA,J_EA)')
    for model in sorted(required):
        for baseline in ('J_E', 'XP', 'X', 'R_EA'):
            if baseline in present:
                contrasts[f'{model}_VS_{baseline}'] = f'gain({model},{baseline})'
    config = read(sources[0]['source'])
    manifest = read(config['contrast_manifest'])
    manifest['essential_contrasts'] = {'corrected_epoch20': contrasts}
    immutable(directory/'contrasts.yaml', manifest)
    config.update(artifacts=deduplicate_artifacts(artifacts), contrast_manifest=str(directory/'contrasts.yaml'),
        extra_contrasts={}, opera_model=None, route_weights=None, design_search=False,
        strategy_candidates=[], clinical_threshold_artifacts=[], identity_date_sensitivity=False,
        stratified_reanalysis={}, patient_groups={}, operating_points={'rules': [], 'pairs': []},
        patient_gain_contrasts=[], contrast_touch_models=sorted(required),
        coverage_roster=None, compute_inventory=None)
    immutable(directory/'analysis.yaml', config)
    scope = (receipt['scope'] if architecture == 'joint' else
             'Exploratory corrected pretraining-budget by architecture comparisons; primary aliases unchanged')
    return dict(status='ready', analysis=str(directory/'analysis.yaml'), scope=scope)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage', choices=['prepare', 'prepare-opera', 'work', 'status', 'compare'])
    p.add_argument('--root', type=Path, default=Path('data/cutoff_recovery_20261121'))
    p.add_argument('--family', choices=['cohort', 'adapted'])
    p.add_argument('--architecture', choices=['joint', 'opera'])
    p.add_argument('--wait-hours', type=float, default=48)
    a = p.parse_args()
    root = a.root.resolve()
    if a.stage in {'prepare', 'prepare-opera'}:
        print(json.dumps(prepare(root, 'opera' if a.stage == 'prepare-opera' else 'joint'), indent=2))
    elif a.stage == 'status':
        print(json.dumps(status(root, a.family, a.architecture), indent=2))
    elif a.stage == 'compare':
        from opera.run.cutoff_recovery_results import run_analysis, package
        report = prepare_analysis(root, a.architecture or 'opera')
        print(json.dumps(report, indent=2), flush=True)
        if report['status'] != 'ready':
            raise SystemExit(3)
        run_analysis(Path(report['analysis']).parent)
        package(root)
        print('EPOCH20 COMPARISONS COMPLETE:', Path(report['analysis']).parent/'results', flush=True)
    else:
        if a.wait_hours <= 0:
            p.error('wait-hours must be positive')
        deadline = time.monotonic()+a.wait_hours*3600
        while time.monotonic() < deadline:
            report = status(root, a.family, a.architecture)
            if report['status'] == 'not_prepared':
                raise ValueError('Run epoch20-opera-prepare first' if a.architecture == 'opera' else 'Run epoch20-prepare first')
            if report['status'] == 'complete':
                print(json.dumps(report, indent=2)); return
            if try_one(root, f'{socket.gethostname()}-{__import__("os").getpid()}', a.family, a.architecture) != 'processed':
                print(json.dumps(report), flush=True)
                time.sleep(30)
        raise TimeoutError('Epoch20 wait limit reached; checkpoints retained')


if __name__ == '__main__':
    main()
