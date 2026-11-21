"""Bind registered analysis artifacts to only the fits affected by cutoff recovery."""
import argparse
import copy
import hashlib
import json
import csv
from pathlib import Path

import yaml

from opera.run.finish_figure_inputs import immutable, sha
from opera.run.prepare_cutoff_recovery import replacement, verify_affected_checkpoint

CORE = {'J_L', 'J_OA_OWN_L', 'OPERA', 'R_EA', 'J_EA', 'R_AB_EA',
        'R_PF_OFF_EA', 'R_AB_L', 'R_PF_OFF_L'}
DAMAGE_CORE = {'J_L', 'J_OA_OWN_L', 'J_EA', 'R_EA'}
UNCHANGED_TABULAR = {'X', 'XP', 'XR', 'XPR', 'LR'}


def absolute(path, repo):
    path = Path(path)
    return path.resolve() if path.is_absolute() else (repo / path).resolve()


def discover_sources(repo):
    national = repo / 'data/paper_256_v3_national_20261111'
    required = [repo / 'data/paper_256_v3/analysis/local_test_20261109_four_allocations/analysis_inventory/analysis.yaml',
                national / 'extended_test_comparison/analysis.yaml',
                national / 'extended_tuning_comparison/analysis.yaml']
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(f'Missing required registered analysis: {path}')
    # Latest existing recipe per requested lane. These are registered scopes,
    # not all historical search bundles. Exact source hashes are frozen below.
    optional = [national / 'fraction_results_20261116/analysis.yaml',
                repo / 'data/epoch20_matched_20261120/comparisons_epoch20/analysis.yaml']
    for kind in ('core', 'ablations', 'pretraining_controls', 'design_search'):
        candidates = list((repo / 'data').glob(f'figure_closure_*/comparisons_{kind}/analysis.yaml'))
        if candidates:
            optional.append(max(candidates, key=lambda p: p.stat().st_mtime_ns))
    return required + [p for p in optional if p.is_file()]


def discover_bundles(repo):
    paths = []
    for root in (repo / 'data/paper_256_v3', repo / 'data/paper_256_v3_national_20261111',
                 repo / 'data/epoch20_joint_control_20261119'):
        if root.is_dir():
            paths.extend(p.parent for p in root.rglob('bundle.json'))
    return sorted(set(paths))


def export_source_blockers(analyses):
    """Catch absent unchanged map inputs before spending GPU time on recovery."""
    declarations = [d for source in analyses for d in source['artifacts']
                    if int(d['artifact'].get('seed', 42)) == 42]
    blockers = []
    for model in ('J_L', 'J_OA_OWN_L', 'J_EA', 'R_EA', 'R_AB_EA', 'J_E', 'R_E'):
        for split in ('tuning', 'held_out'):
            choices = {d['artifact']['path']: d for d in declarations
                       if d['artifact']['model'] == model and d['artifact'].get('split') == split}
            reason = None
            if len(choices) != 1:
                reason = f'Need one registered seed-42 {model}/{split} source for final exports; got {len(choices)}'
            elif model in {'J_E', 'R_E'}:
                directory = Path(next(iter(choices))).parent
                required = ('patient_embeddings.npz', 'patient_embeddings.metadata.json', 'evaluation_complete.json')
                missing = [name for name in required if not (directory/name).is_file()]
                if missing:
                    reason = ('Unchanged national embedding inputs absent: '+', '.join(missing)
                              +'. Export these from the existing selected fit; no refitting is needed.')
                else:
                    evaluation = json.loads((directory/'evaluation_complete.json').read_text())
                    if not evaluation.get('inputs', {}).get('export_embeddings'):
                        reason = 'Unchanged national evaluation does not verify its embedding export'
            if reason:
                blockers.append(dict(model=model, split=split, reason=reason, scope='final_exports'))
    return blockers


def index_bundles(bundles):
    index, directories = {}, {}
    for directory in bundles:
        manifest = json.loads((directory / 'bundle.json').read_text())
        for row in manifest.get('runs', []):
            path = directory / row['config_path']
            digest = row['config_sha256']
            # Hash only referenced configs below; large historical bundle
            # inventories otherwise cost many thousands of network reads.
            index.setdefault(digest, []).append((row, path, manifest))
            directories.setdefault(row['run_directory'], set()).add(digest)
    return index, directories


def add_planned_controls(repo, root, sources):
    """Include approved fractions/replicates even if their old export was pending."""
    from opera.run.paper_prediction_queue import canonical_models
    national = repo/'data/paper_256_v3_national_20261111'
    direct = national/'direct_fits/bundle.json'
    if not direct.is_file(): return sources
    provenance = json.loads(direct.read_text()).get('provenance', {})
    local = provenance.get('frozen_local_bundle')
    if not local: raise ValueError('Direct national bundle lacks its frozen local bundle identity')
    local = absolute(local, repo)
    if local.is_file(): local = local.parent
    directories = [(local, None), (national/'adapted_fits', national)]
    specifications = [yaml.safe_load(p.read_text()) for p in sources]
    declared = [a for s in specifications for a in s.get('artifacts', [])]
    keys = {(str(absolute(a['path'], repo)), a['model'], int(a.get('seed', 42))) for a in declared}
    references = {a['model']: a for a in declared if a['model'] in {'J_L', 'J_EA'} and int(a.get('seed', 42)) == 42}
    prototypes = []
    for directory, _ in directories:
        if not (directory/'bundle.json').is_file(): raise FileNotFoundError(directory/'bundle.json')
        manifest = json.loads((directory/'bundle.json').read_text())
        for row in manifest['runs']:
            if row['seed'] != 42: continue
            cfg = yaml.safe_load((directory/row['config_path']).read_text())
            if set(canonical_models(row, cfg, manifest)) & CORE:
                prototypes.append(cfg)
    def same_design(a, b):
        def training(cfg):
            result = copy.deepcopy(cfg.get('training', {}))
            result.pop('train_fraction_seed', None)
            result.pop('seed', None)
            if 'batch_sampling' in result:
                result['batch_sampling'].pop('seed', None)
            return result
        fields = ('model', 'competing_risk', 'cross_outcome', 'outcomes', 'cohorts',
                  'dapt_ckpt', 'source_opera_ckpt', 'adapter_initialization')
        return all(a.get(k) == b.get(k) for k in fields) and training(a) == training(b)
    additions = []
    for directory, base in directories:
        manifest = json.loads((directory/'bundle.json').read_text())
        base = base or directory.parent.parent
        for row in manifest['runs']:
            planned = row['phase'] in {'data_efficiency', 'pretraining_control'}
            repeated = int(row['seed']) in {43, 44, 45}
            if not planned and not repeated: continue
            path = directory/row['config_path']
            if sha(path) != row['config_sha256']: raise ValueError('Approved control config changed')
            cfg = yaml.safe_load(path.read_text())
            if not planned and not any(same_design(cfg, prototype) for prototype in prototypes): continue
            aliases = canonical_models(row, cfg, manifest)
            reference = references.get('J_EA' if row['origin'] == 'EA' else 'J_L')
            if reference is None: raise ValueError('Missing registered contract for approved controls')
            for alias in aliases:
                for split in ('tuning', 'held_out'):
                    prediction = base/'predictions'/row['run_directory']/split/'direct_cr_predictions.parquet'
                    if (str(prediction.resolve()), alias, int(row['seed'])) in keys: continue
                    item = {**copy.deepcopy(reference), 'model': alias, 'seed': int(row['seed']),
                            'split': split, 'path': str(prediction.resolve()), 'format': 'direct_cif',
                            'cohort_from_metadata': True}
                    additions.append(item)
    if not additions: return sources
    config = copy.deepcopy(specifications[0])
    baseline_models = {'J_L', 'J_EA', 'J_OA_OWN_L', 'S_L', 'J_RI', 'J_FZ'}
    baselines = [a for a in declared if a['model'] in baseline_models]
    unique = {json.dumps(a, sort_keys=True): a for a in baselines+additions}
    config['artifacts'] = list(unique.values())
    config.update(route_weights=None, clinical_threshold_artifacts=[], design_search=False,
                  identity_date_sensitivity=False, stratified_reanalysis={})
    path = root/'registered_additions/analysis.yaml'
    immutable(path, config)
    return [*sources, path]


def match_artifact(item, index, directories):
    path = Path(item['path'])
    marker = path.parent / 'evaluation_complete.json'
    if marker.is_file():
        record = json.loads(marker.read_text())
        digest = record.get('inputs', {}).get('config_sha256')
        if digest in index:
            return digest
    matched = set()
    for directory, digests in directories.items():
        if f'/{directory}/' in str(path):
            matched.update(digests)
    if len(matched) != 1:
        raise ValueError(f'Cannot uniquely bind registered prediction to its frozen fit: {path}')
    return matched.pop()


def fit_priority(models, seed, cfg):
    if seed == 42 and models & DAMAGE_CORE:
        return 0
    if seed == 42 and models & CORE:
        return 1
    if cfg['cutoff_recovery']['expected_epochs'] == 20:
        return 40
    if models & {'S_L'}:
        return 10
    if models & {'D_L', 'O_L'}:
        return 15
    return 20


def clone_for_queue(cfg, root):
    # A supervised predecessor is mapped explicitly after collecting the graph.
    # Explicit resume checkpoints remain forbidden by replacement().
    temporary = copy.deepcopy(cfg)
    predecessor = temporary.get('source_opera_ckpt')
    temporary['source_opera_ckpt'] = None
    new, family = replacement(temporary, root)
    if new is not None:
        if predecessor:
            new['source_opera_ckpt'] = predecessor
        elif 'source_opera_ckpt' not in cfg:
            new.pop('source_opera_ckpt', None)
    return new, family


def prepare(repo, root, sources=None, bundles=None):
    from opera.run.prepare_paper_run_bundle import write_bundle
    default_sources = sources is None
    export_inputs = {}
    if default_sources:
        paths = dict(metadata=repo/'data/clinical_registry_refresh/20261111/prepared_clinical/patient_metadata.parquet',
                     scores=repo/'data/story_closure_20261116/server_only/score_config_flipi2_os.yaml',
                     interpretation_env=repo/'data/paper_256_v3/analysis/local_test_20261109_four_allocations/overnight.env')
        for label, path in paths.items():
            if not path.is_file(): raise FileNotFoundError(f'Recovery export prerequisite missing: {path}')
            export_inputs[label] = dict(path=str(path), sha256=sha(path))
        import importlib.util
        if importlib.util.find_spec('umap') is None:
            raise ValueError('UMAP is absent from this Python environment; activate the plotting-capable server environment')
    sources = add_planned_controls(repo, root, sources or discover_sources(repo))
    bundles = bundles or discover_bundles(repo)
    index, directories = index_bundles(bundles)
    jobs, analyses, blockers, reused, checked = {}, [], [], [], set()
    configs, affected_checkpoints, unaffected_checkpoints = {}, set(), set()
    for source in sources:
        config = yaml.safe_load(source.read_text())
        declarations = []
        for original in config.get('artifacts', []):
            item = copy.deepcopy(original)
            item['path'] = str(absolute(item['path'], repo))
            try:
                if item['model'] in UNCHANGED_TABULAR:
                    if not Path(item['path']).is_file():
                        raise FileNotFoundError(f'Missing unaffected comparator: {item["path"]}')
                    declarations.append(dict(artifact=item, action='reuse'))
                    reused.append(item['model'])
                    continue
                digest = match_artifact(item, index, directories)
                row, path, _ = index[digest][0]
                if digest not in checked and sha(path) != digest:
                    raise ValueError(f'Frozen source fit config changed: {path}')
                if digest not in configs:
                    configs[digest] = yaml.safe_load(path.read_text())
                cfg = configs[digest]
                new, family = clone_for_queue(cfg, root)
                if new is None:
                    if cfg.get('encoder_initialization') != 'random':
                        from opera.run.prepare_cutoff_recovery import NATIONAL_SHA
                        checkpoint = Path(cfg['dapt_ckpt'])
                        if checkpoint not in unaffected_checkpoints and sha(checkpoint) != NATIONAL_SHA:
                            raise ValueError('Unclassified encoder; cannot declare this result unaffected')
                        unaffected_checkpoints.add(checkpoint)
                    if not Path(item['path']).is_file():
                        raise FileNotFoundError(item['path'])
                    declarations.append(dict(artifact=item, action='reuse'))
                    reused.append(item['model'])
                    continue
                if digest not in checked:
                    checkpoint = Path(cfg['dapt_ckpt'])
                    if checkpoint not in affected_checkpoints:
                        verify_affected_checkpoint(checkpoint, family)
                        affected_checkpoints.add(checkpoint)
                    checked.add(digest)
                if int(item.get('seed', row['seed'])) != int(row['seed']):
                    raise ValueError('Registered artifact seed differs from the source fit')
                if digest not in jobs:
                    jobs[digest] = dict(source_config=str(path), source_config_sha256=digest,
                        source_bundle=str(path.parents[len(Path(row['config_path']).parts)-1]),
                        source_row=row, config=new, models=set(), splits=set(), family=family,
                        removal_splits=set())
                job = jobs[digest]
                job['models'].add(item['model'])
                job['splits'].add(item.get('split', 'held_out'))
                # Also evaluate tuning for every corrected fit, keeping checkpoint
                # selection and held-out assessment distinct.
                job['splits'].add('tuning')
                if set(job['models']) & {'R_EA', 'J_EA', 'R_AB_EA', 'J_OA_OWN_L'}:
                    job['splits'].add('train')
                relative = None
                if Path(item['path']).name != 'direct_cr_predictions.parquet' or '_source_removal/' in item['path']:
                    # Native source-removal exports share a fit but contain
                    # DIFFERENT predictions. Never map these onto its base CIF.
                    anchor = f'/{row["run_directory"]}/'
                    if not any(token in item['path'] for token in ('/source_removal/', '_source_removal/')) or anchor not in item['path']:
                        raise ValueError('Nonstandard affected forecast needs an explicit export mapping')
                    relative = item['path'].split(anchor, 1)[1]
                    if Path(relative).is_absolute() or '..' in Path(relative).parts:
                        raise ValueError('Unsafe source-removal path')
                    job['removal_splits'].add(item['split'])
                declarations.append(dict(artifact=item, action='replace', job=digest,
                                         relative_prediction=relative))
            except (ValueError, FileNotFoundError, KeyError) as error:
                blockers.append(dict(source_analysis=str(source), model=item.get('model'),
                                     artifact_path=item['path'], reason=str(error)))
        analyses.append(dict(source=str(source), source_sha256=sha(source), artifacts=declarations))
    if default_sources:
        blockers.extend(export_source_blockers(analyses))
    root.mkdir(parents=True, exist_ok=True)
    (root/'preparation_blockers.json').write_text(json.dumps(blockers, indent=2))
    if blockers:
        raise ValueError(f'{len(blockers)} unresolved artifact bindings; inspect {root / "preparation_blockers.json"}. No queue was published.')
    # Resolve supervised warm starts topologically. A predecessor with no
    # reported predictions is trained once, without unnecessary inference.
    visiting, emitted, dependency_rows = set(), set(), {}
    def emit(digest):
        if digest in visiting: raise ValueError('Cyclic supervised initialization dependency')
        if digest in emitted: return
        visiting.add(digest)
        job = jobs[digest]
        predecessor = job['config'].get('source_opera_ckpt')
        job['dependencies'] = []
        if predecessor:
            dependency = match_artifact(dict(path=predecessor), index, directories)
            if dependency not in jobs:
                row, path, _ = index[dependency][0]
                if sha(path) != dependency: raise ValueError('Predecessor config changed')
                original = yaml.safe_load(path.read_text())
                new, family = clone_for_queue(original, root)
                if new is None:
                    from opera.run.prepare_cutoff_recovery import NATIONAL_SHA
                    from bonsai.functional.checkpointing import verified_training_selection
                    if original.get('encoder_initialization') != 'random' and sha(Path(original['dapt_ckpt'])) != NATIONAL_SHA:
                        raise ValueError('Unclassified supervised predecessor')
                    selected = verified_training_selection(Path(predecessor).parent)
                    if sha(Path(predecessor)) != selected['best_checkpoint_sha256']:
                        raise ValueError('Supervised predecessor is not the verified selected checkpoint')
                    dependency = None
                else:
                    verify_affected_checkpoint(Path(original['dapt_ckpt']), family)
                    jobs[dependency] = dict(source_config=str(path), source_config_sha256=dependency,
                        source_row=row, config=new, models=set(), splits=set(), family=family,
                        removal_splits=set(), source_bundle='supervised_dependency')
            if dependency is not None:
                emit(dependency)
                parent = jobs[dependency]
                job['config']['source_opera_ckpt'] = str(root/'neural_runs'/dependency_rows[dependency]/'best.ckpt')
                job['dependencies'].append(dependency)
                parent['priority'] = min(parent['priority'], fit_priority(set(job['models']), job['source_row']['seed'], job['config']))
        job['models'] = sorted(job['models'])
        job['splits'] = sorted(job['splits'])
        job['removal_splits'] = sorted(job['removal_splits'])
        job['priority'] = fit_priority(set(job['models']), job['source_row']['seed'], job['config'])
        job['config']['cutoff_recovery']['export_embeddings'] = bool(set(job['models']) & CORE and job['source_row']['seed'] == 42)
        new_hash = hashlib.sha256(json.dumps(job['config'], sort_keys=True, default=str).encode()).hexdigest()
        dependency_rows[digest] = f'{job["source_row"]["origin"]}/seed{job["source_row"]["seed"]}/{new_hash[:20]}'
        visiting.remove(digest)
        emitted.add(digest)
    for digest in list(jobs): emit(digest)
    entries = []
    for digest, job in sorted(jobs.items()):
        row = job['source_row']
        entries.append(dict(name=f'recovery/{digest}', phase=row['phase'], origin=row['origin'],
                            seed=row['seed'], condition={}, config=job['config'],
                            recovery_models=job['models'], recovery_priority=job['priority'], source_id=digest))
    bundle_dir = root/'bundles/registered'
    identity = dict(analysis_sources={str(p): sha(p) for p in sources},
                    scope='Replace only encoder-dependent fits referenced by registered analyses; reuse unaffected artifacts')
    if bundle_dir.exists():
        saved = json.loads((bundle_dir/'bundle.json').read_text())
        expected = {e['source_id']: e['config'] for e in entries}
        actual = {r['source_id']: yaml.safe_load((bundle_dir/r['config_path']).read_text()) for r in saved['runs']}
        if actual != expected or saved['provenance'] != identity:
            raise ValueError('Prepared recovery queue changed; use a fresh recovery root')
        manifest = saved
    else:
        manifest = write_bundle(entries, bundle_dir, provenance=identity)
    rows = {r['source_id']: r for r in manifest['runs']}
    job_dir = root/'jobs'
    job_dir.mkdir(exist_ok=True)
    for digest, job in jobs.items():
        job['row'] = rows[digest]
        job['pretraining_epochs'] = job['config']['cutoff_recovery']['expected_epochs']
        job.pop('config')
        immutable(job_dir/f'{digest}.yaml', job)
    # Compatibility manifest for the existing interpretation entry point. It
    # names the SAME corrected fits/config bytes, not another training queue.
    adapted_view = root/'adapted_fits'
    adapted_rows = []
    for job in jobs.values():
        if job['family'] != 'adapted': continue
        row = copy.deepcopy(job['row'])
        row['name'] = job['source_row']['name']
        if row['name'] in {r['name'] for r in adapted_rows}: continue
        original_file = bundle_dir/row['config_path']
        row['config_path'] = f'configs/{row["name"]}.yaml'
        destination = adapted_view/row['config_path']
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists() and destination.read_bytes() != original_file.read_bytes():
            raise ValueError('Interpretation compatibility config changed')
        destination.write_bytes(original_file.read_bytes())
        adapted_rows.append(row)
    view = dict(runs=adapted_rows,
                provenance=dict(scope='Compatibility view of corrected fits; no additional training'))
    view_path = adapted_view/'bundle.json'
    adapted_view.mkdir(exist_ok=True)
    if view_path.exists() and json.loads(view_path.read_text()) != view:
        raise ValueError('Interpretation compatibility manifest changed')
    temporary_view = adapted_view/'bundle.json.writing'
    temporary_view.write_text(json.dumps(view, indent=2))
    temporary_view.replace(view_path)
    immutable(root/'registered_analyses.yaml', analyses)
    plan = dict(schema=1, root=str(root), bundle=str(bundle_dir), bundle_sha256=sha(bundle_dir/'bundle.json'),
                jobs=jobs, analyses=str(root/'registered_analyses.yaml'),
                analyses_sha256=sha(root/'registered_analyses.yaml'), reused_models=sorted(set(reused)),
                main_pretraining_epochs=10, longer_training='Separate exploratory epoch20 arms only',
                export_inputs=export_inputs,
                settings='Original fine-tuning, labels, splits and selection rules retained')
    immutable(root/'queue_plan.yaml', plan)
    from collections import Counter
    categories = Counter((job['family'], job['source_row']['phase'], job['source_row']['regime'],
                          int(job['source_row']['seed'])) for job in jobs.values())
    summary = dict(status='prepared_not_server_executed', affected_fits=len(jobs),
                   core_fits=sum(j['priority'] == 0 for j in jobs.values()),
                   reused_models=plan['reused_models'], analysis_sources=identity['analysis_sources'])
    summary['fit_categories'] = [dict(family=family, phase=phase, regime=regime, seed=seed, fits=count)
                                 for (family, phase, regime, seed), count in sorted(categories.items())]
    measured_seconds, measured_jobs = 0., 0
    for job in jobs.values():
        source_bundle = Path(job['source_bundle'])
        base = source_bundle.parent.parent if source_bundle.parent.name == 'bundles' else source_bundle.parent
        attempts = base/'neural_runs'/job['source_row']['run_directory']/'worker_attempts.jsonl'
        if not attempts.is_file(): continue
        records = [json.loads(line) for line in attempts.read_text().splitlines() if line.strip()]
        seconds = sum(float(r.get('wall_seconds', 0)) for r in records)
        if seconds > 0:
            measured_jobs += 1
            measured_seconds += seconds
    summary['historical_runtime'] = dict(measured_jobs=measured_jobs, unknown_jobs=len(jobs)-measured_jobs,
        recorded_gpu_allocation_hours=measured_seconds/3600,
        ideal_eight_gpu_hours_for_measured_subset=measured_seconds/3600/8,
        note='Original recorded attempts only; partial coverage, different stopping epochs and scheduling can change recovery time')
    (root/'queue_summary.json').write_text(json.dumps(summary, indent=2))
    with (root/'queue_index.csv').open('w', newline='') as handle:
        fields = ['models', 'seed', 'family', 'pretraining_epochs', 'priority', 'original_name', 'run_directory']
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for job in sorted(jobs.values(), key=lambda j: (j['priority'], j['row']['name'])):
            writer.writerow(dict(models='|'.join(job['models']), seed=job['row']['seed'], family=job['family'],
                pretraining_epochs=job['pretraining_epochs'], priority=job['priority'],
                original_name=job['source_row']['name'], run_directory=job['row']['run_directory']))
    return summary


def main():
    from dotenv import load_dotenv
    load_dotenv()
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, default=Path('data/cutoff_recovery_20261121'))
    p.add_argument('--source-analysis', action='append', type=Path)
    p.add_argument('--source-bundle', action='append', type=Path)
    a = p.parse_args()
    print(json.dumps(prepare(Path.cwd().resolve(), a.root.resolve(), a.source_analysis, a.source_bundle), indent=2))


if __name__ == '__main__':
    main()
