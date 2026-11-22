"""Reversible operational deferral of the historical recovery design sweep."""
import argparse
import json
from pathlib import Path
import re

import yaml

from opera.run.finish_figure_inputs import sha

POLICY = 'workload_policy.json'
SEED_DESIGNS = {'J', 'J_OA_OWN', 'R_AB', 'R_PF_OFF'}
KEEP_DESIGNS = {'N', 'N_A', 'R_A', 'R_O', 'RBH_AB', 'RBH_O',
                'J_OA_OWN_NODIS', 'J_OA_OWN_NOOUT'}


def read_policy(root, plan):
    path = root/POLICY
    if not path.is_file():
        return dict(deferred_jobs=[], profile='full_registered_recovery')
    policy = json.loads(path.read_text())
    if policy['queue_plan_sha256'] != sha(root/'queue_plan.yaml'):
        raise ValueError('Workload policy belongs to a different frozen queue')
    if not set(policy['deferred_jobs']) <= set(plan['jobs']):
        raise ValueError('Unknown jobs in workload policy')
    return policy


def build_policy(root, plan):
    # Explicit essential contrasts remain protected. DS_* describes historical
    # search, and must not accidentally protect the sweep we intend to defer.
    from opera.run.analyse_locked_predictions import models_for, SAMPLING_SENSITIVITY, ORIGINAL_ROUTER
    if sha(Path(plan['analyses'])) != plan['analyses_sha256']:
        raise ValueError('Registered analysis inventory changed')
    sources = yaml.safe_load(Path(plan['analyses']).read_text())
    protected = {'S_L', 'D_L', 'O_L', 'J_L', 'J_EA', 'R_EA', 'J_OA_OWN_L',
                 'R_AB_EA', 'R_PF_OFF_EA', 'R_AB_L', 'R_L', 'R_PF_OFF_L'}
    protected.update(models_for(SAMPLING_SENSITIVITY))
    protected.update(models_for(ORIGINAL_ROUTER))
    for source in sources:
        if sha(Path(source['source'])) != source['source_sha256']:
            raise ValueError('Source analysis changed after queue preparation')
        config = yaml.safe_load(Path(source['source']).read_text())
        manifest = yaml.safe_load(Path(config['contrast_manifest']).read_text())
        expressions = {name: expr for family in manifest['essential_contrasts'].values()
                       for name, expr in family.items() if not name.startswith('DS_')}
        protected.update(models_for(expressions))
        protected.update(config.get('strategy_candidates', ['S_L', 'D_L', 'O_L']))
    deferred = set()
    rows = []
    for digest, job in plan['jobs'].items():
        row = job['source_row']
        design = row['name'].split('/')[-1].removeprefix('fix_')
        design = re.sub(r'_s4[345]$', '', design)
        keep = (row['phase'] != 'router_fix' or job['priority'] < 20
                or bool(set(job['models']) & protected)
                or design in KEEP_DESIGNS or design in SEED_DESIGNS
                or job.get('pretraining_epochs') == 20)
        if not keep:
            deferred.add(digest)
        rows.append(dict(job=digest, models=job['models'], seed=row['seed'],
                         original_name=row['name'], action='keep' if keep else 'defer'))
    # A retained job must never wait for a deferred supervised predecessor.
    required = set(plan['jobs'])-deferred
    pending = list(required)
    while pending:
        for dependency in plan['jobs'][pending.pop()].get('dependencies', []):
            if dependency not in required:
                required.add(dependency)
                pending.append(dependency)
    deferred -= required
    for row in rows:
        row['action'] = 'defer' if row['job'] in deferred else 'keep'
    return dict(schema=1, profile='paper_first', queue_plan_sha256=sha(root/'queue_plan.yaml'),
                deferred_jobs=sorted(deferred), retained_jobs=len(plan['jobs'])-len(deferred),
                protected_models=sorted(protected),
                rows=rows, note='Operational deferral; frozen configurations and analysis lock unchanged. '
                'Historical search evidence is not refreshed or silently reused as corrected evidence.')


def filter_source(source, policy):
    deferred = set(policy['deferred_jobs'])
    omitted = [d for d in source['artifacts'] if d.get('job') in deferred]
    return {**source, 'artifacts': [d for d in source['artifacts'] if d.get('job') not in deferred],
            'recovery_deferred_artifacts': omitted,
            'recovery_defer_design_search': Path(source['source']).parent.name == 'comparisons_design_search'}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, default=Path('data/cutoff_recovery_20261121'))
    p.add_argument('--execute', action='store_true', help='Publish the reviewed paper-first policy atomically')
    a = p.parse_args()
    root = a.root.resolve()
    plan = yaml.safe_load((root/'queue_plan.yaml').read_text())
    policy = build_policy(root, plan)
    preview = root/'workload_policy_preview.json'
    preview.write_text(json.dumps(policy, indent=2))
    if a.execute:
        path = root/POLICY
        if path.exists() and json.loads(path.read_text()) != policy:
            raise ValueError('An existing workload policy differs; do not change an active analysis scope')
        temporary = path.with_suffix('.writing')
        temporary.write_text(json.dumps(policy, indent=2))
        temporary.replace(path)
    print(json.dumps(dict(retained_jobs=policy['retained_jobs'], deferred_jobs=len(policy['deferred_jobs']),
                          preview=str(preview), active=a.execute,
                          note='Existing workers need relaunching to load policy-aware code; do not rerun prepare'), indent=2))
    for row in policy['rows']:
        if row['action'] == 'defer':
            print('DEFER', row['seed'], ','.join(row['models']), row['original_name'])


if __name__ == '__main__':
    main()
