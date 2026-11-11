"""Durable analysis inputs and a shared-filesystem CPU queue (secure internal data).

Pickles are trusted, locally generated analysis artifacts, never export inputs.
Workers claim jobs with POSIX flock; killed jobs are reclaimable without leases.
"""
from __future__ import annotations

import fcntl
import hashlib
import json
import os
from pathlib import Path
import pickle
import socket
import time
import uuid
from functools import lru_cache

_NEXT_JOB = {}


def worker_identity():
    import numpy as np
    import pandas as pd
    directory = Path(__file__).resolve().parent
    return dict(numpy=np.__version__, pandas=pd.__version__, implementation={
        name: hashlib.sha256((directory / name).read_bytes()).hexdigest()
        for name in ('secondary_analysis.py', 'distributed_analysis.py')})


def atomic_write(path, writer):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    try:
        with temporary.open('wb') as handle:
            writer(handle)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def save_pickle(path, value):
    atomic_write(path, lambda handle: pickle.dump(value, handle, protocol=5))


def load_pickle(path):
    with Path(path).open('rb') as handle:
        return pickle.load(handle)


def save_json(path, value):
    atomic_write(path, lambda handle: handle.write(json.dumps(value, sort_keys=True).encode()))


def signature(config):
    """Bind cache to recipe, implementation and frozen input-file identities.

    Reuse checks size/mtime as well as paths; original content SHA256 values are
    retained in the prepared context. Replacing input files invalidates the cache.
    """
    paths = {str(a['path']) for a in config.get('artifacts', [])}
    for key in ('patient_metadata', 'cell_metadata', 'contrast_manifest',
                'coverage_roster', 'route_weights', 'compute_inventory'):
        if config.get(key):
            paths.add(str(config[key]))
    repo = Path(__file__).resolve().parents[2]
    code = {}
    for relative in ('opera/evaluation/secondary_analysis.py',
                     'opera/evaluation/distributed_analysis.py',
                     'opera/run/analyse_locked_predictions.py',
                     'opera/evaluation/operating_points.py'):
        code[relative] = hashlib.sha256((repo / relative).read_bytes()).hexdigest()
    identities = {}
    for name in sorted(paths):
        stat = Path(name).stat()
        identities[str(Path(name).resolve())] = [stat.st_size, stat.st_mtime_ns]
    payload = dict(config=config, inputs=identities, implementation=code,
                   environment=worker_identity(), version=1)
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()


def save_prepared(directory, frame, context, *, chunk_rows=1_000_000):
    """Publish a ready marker only after every parquet partition is durable."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    parts = []
    for number, start in enumerate(range(0, len(frame), chunk_rows)):
        name = f'predictions_{number:05d}.parquet'
        atomic_write(directory / name,
                     lambda handle, start=start: frame.iloc[start:start+chunk_rows].to_parquet(handle, index=True))
        parts.append(name)
    save_pickle(directory / 'context_INTERNAL.pkl', context)
    save_json(directory / 'prepared.json', dict(parts=parts, rows=len(frame)))


def load_prepared(directory):
    import pandas as pd
    directory = Path(directory)
    manifest = json.loads((directory / 'prepared.json').read_text())
    frame = pd.concat([pd.read_parquet(directory / name) for name in manifest['parts']])
    if len(frame) != manifest['rows']:
        raise ValueError('Incomplete prepared prediction snapshot')
    return frame, load_pickle(directory / 'context_INTERNAL.pkl')


def publish_job(directory, number, batch, options):
    directory = Path(directory)
    stem = f'batch_{number:05d}'
    ready = directory / f'{stem}.ready.json'
    if ready.exists():
        return
    # All tasks reuse the registered full-cohort patient draws. Store each
    # cohort's matrix once per queue instead of copying it into thousands of
    # input pickles. Each worker retains only a small LRU of these matrices.
    kind = options.get('kind', 'paired')
    if kind == 'paired':
        batch = [(key, cell, _draw_references(directory, draws)) for key,cell,draws in batch]
    elif kind == 'operating_points':
        frame, draws = batch
        batch = frame, _draw_references(directory, draws)
    save_pickle(directory / f'{stem}.input_INTERNAL.pkl', dict(batch=batch, options=options))
    save_json(ready, dict(number=number))


def _draw_references(directory, draws):
    refs = {}
    for key, value in draws.items():
        name = hashlib.sha256(repr(key).encode()).hexdigest()
        path = (Path(directory) / 'shared_draws' / f'{name}_INTERNAL.pkl').resolve()
        if not path.exists():
            save_pickle(path, value)
        refs[key] = str(path)
    return refs


@lru_cache(maxsize=4)
def _read_shared_draw(path):
    return load_pickle(path)


def _resolve_draws(refs):
    return {key: _read_shared_draw(path) for key,path in refs.items()}


def perform_one(directory):
    """Return True after one completed/reported job, False if none claimable."""
    from opera.evaluation.secondary_analysis import _paired_cells
    directory = Path(directory)
    if (directory / 'queue.json').exists():
        expected = json.loads((directory / 'queue.json').read_text())['implementation']
        if expected != worker_identity():
            raise RuntimeError('Worker implementation or numpy/pandas version differs from coordinator')
    jobs = sorted(directory.glob('batch_*.ready.json'))
    offset = _NEXT_JOB.get(str(directory), 0) % max(1, len(jobs))
    ordered = jobs[offset:] + jobs[:offset]
    for position, ready in enumerate(ordered):
        stem = ready.name.removesuffix('.ready.json')
        result = directory / f'{stem}.result_INTERNAL.pkl'
        failure = directory / f'{stem}.failed.json'
        if result.exists() or failure.exists():
            continue
        with (directory / f'{stem}.lock').open('a') as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                continue
            if result.exists() or failure.exists():
                continue
            save_json(directory / f'{stem}.running.json',
                      dict(host=socket.gethostname(), pid=os.getpid(), started=time.time()))
            try:
                job = load_pickle(directory / f'{stem}.input_INTERNAL.pkl')
                options = job['options']
                kind = options.get('kind', 'paired')
                if kind == 'paired':
                    batch = [(key, cell, _resolve_draws(refs)) for key,cell,refs in job['batch']]
                    result_data = _paired_cells(batch, options['parsed'],
                        options['patient_gain_contrasts'], options['min_cases'], options['min_controls'])
                elif kind == 'effect_summary':
                    from opera.evaluation.secondary_analysis import _effect_summary_task
                    result_data = _effect_summary_task(job['batch'])
                elif kind == 'calibration':
                    from opera.evaluation.secondary_analysis import reliability_tables, _calibration_summary
                    result_data = (reliability_tables(job['batch']), _calibration_summary(job['batch']))
                elif kind == 'operating_points':
                    from opera.evaluation.operating_points import operating_points
                    frame, refs = job['batch']
                    draws = _resolve_draws(refs)
                    result_data = [operating_points(frame, draws, rule=rule,
                        pairs=options['pairs'], minimum_events=options['minimum_events'])
                        for rule in options['rules']]
                elif kind == 'patient_subgroups':
                    from opera.evaluation.secondary_analysis import subgroup_patient_gains, panel_subgroup_gains
                    path, metadata, specifications, refs = job['batch']
                    gains = _read_gain_partition(path)
                    draws = _resolve_draws(refs)
                    result_data = (subgroup_patient_gains(gains, metadata, specifications, draws=draws),
                                   panel_subgroup_gains(gains, metadata, specifications, draws=draws))
                else:
                    raise ValueError(f'Unknown CPU task kind: {kind}')
                save_pickle(result, result_data)
            except Exception as error:
                import traceback
                save_json(failure, dict(error=repr(error), traceback=traceback.format_exc()))
                raise
            finally:
                (directory / f'{stem}.running.json').unlink(missing_ok=True)
            _NEXT_JOB[str(directory)] = offset + position + 1
            return True
    return False


def distributed_parts(directory, batches, total, options):
    """Coordinator publishes durable jobs; workers may start before preparation.

    Collection is in original batch order, regardless of worker completion order.
    The coordinator computes no paired jobs, leaving its RAM for preparation and
    pooled summaries. A worker is required even for a single-node deployment.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    descriptor = dict(total=total, options=options, implementation=worker_identity())
    expected = json.loads(json.dumps(descriptor, sort_keys=True))
    if (directory / 'queue.json').exists():
        if json.loads((directory / 'queue.json').read_text()) != expected:
            raise ValueError('Checkpoint queue options changed; use a new generation')
    else:
        save_json(directory / 'queue.json', descriptor)
    if not (directory / 'published.json').exists():
        for number, batch in enumerate(batches):
            publish_job(directory, number, batch, options)
            if number % 10 == 0 or number + 1 == total:
                print(f'{options.get("kind", "paired")} queue: published {number+1}/{total} jobs', flush=True)
        save_json(directory / 'published.json', dict(total=total))
    for number in range(total):
        stem = f'batch_{number:05d}'
        result = directory / f'{stem}.result_INTERNAL.pkl'
        while not result.exists():
            failures = list(directory.glob('batch_*.failed.json'))
            if failures:
                raise RuntimeError(f'CPU batch failed; inspect {failures[0]}')
            time.sleep(5)
        yield load_pickle(result)
    save_json(directory / 'collected.json', dict(total=total))


def cached_call(directory, name, function):
    """Restart expensive downstream summaries without changing their inputs."""
    if directory is None:
        return function()
    path = Path(directory) / 'stages' / f'{name}_INTERNAL.pkl'
    if path.exists():
        print(f'analysis checkpoint: reusing {name}', flush=True)
        return load_pickle(path)
    result = function()
    save_pickle(path, result)
    return result


def _columnwise_groups(frame, keys, columns):
    """Group only key columns; never reorder the full prediction payload.

    pandas group iteration sorts/copies all selected payload rows at once.
    Group indices on the narrow key table and take each payload column only
    for the current group, preserving original within-group row/index order.
    """
    import pandas as pd
    groups = frame[list(keys)].groupby(list(keys), sort=True).indices
    def parts():
        for key in sorted(groups):
            positions = groups[key]
            yield key, pd.DataFrame({column: frame[column].iloc[positions]
                                     for column in columns})
    return len(groups), parts()


def distributed_calibration(frame, directory):
    """Group-complete calibration jobs, with exactly the serial model order."""
    import pandas as pd
    from opera.evaluation.secondary_analysis import CELL
    columns = ['model', *CELL, 'label', 'probability']
    if 'ipcw_weight' in frame:
        columns.append('ipcw_weight')
    total, groups = _columnwise_groups(frame, ['model'], columns)
    results = list(distributed_parts(directory, (part for _, part in groups),
                                    total, dict(kind='calibration')))
    return tuple(pd.concat([part[i] for part in results], ignore_index=True) for i in (0, 1))


def distributed_operating_points(frame, draws, directory, *, rules, pairs, minimum_events):
    """Keep both splits and every model together for each task and every rule."""
    import pandas as pd
    from opera.evaluation.operating_points import TASK
    columns = [*TASK, 'model', 'split', 'subject_id', 'label', 'probability']
    if 'known' in frame:
        columns.append('known')
    # A task includes all models and both tuning/evaluation patients. Hence
    # thresholds and paired uncertainty are identical to the serial call.
    total, groups = _columnwise_groups(frame, TASK, columns)
    def payloads():
        for key, part in groups:
            relevant = {k:v for k,v in draws.items() if k[1:] == key[:2]}
            yield part, relevant
    options = dict(kind='operating_points', rules=list(rules), pairs=list(pairs),
                   minimum_events=minimum_events)
    # Accumulate only the small resulting tables, not all submitted inputs.
    collected = [[] for _ in rules]
    for results in distributed_parts(directory, payloads(), total, options):
        for target, result in zip(collected, results):
            target.append(result)
    combined = []
    for results in collected:
        points = pd.concat([r[0] for r in results], ignore_index=True)
        differences = pd.concat([r[1] for r in results], ignore_index=True)
        if not differences.empty:
            order = {tuple(pair): i for i,pair in enumerate(pairs)}
            differences['_pair_order'] = [order[(a,b)] for a,b in
                zip(differences.model_a, differences.model_b)]
            differences = differences.sort_values(['_pair_order', *TASK], kind='stable').drop(
                columns='_pair_order').reset_index(drop=True)
        combined.append((points, differences))
    return combined


@lru_cache(maxsize=1)
def _read_gain_partition(path):
    return load_pickle(path)


def distributed_patient_subgroups(gains, metadata, specifications, draws, directory):
    """Independent covariate/contrast jobs; pooling retains all cells/diseases."""
    import pandas as pd
    from opera.evaluation.secondary_analysis import CELL, _GAIN_METRICS
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    refs = _draw_references(directory, draws)
    contrasts = sorted(gains.contrast.unique())
    def payloads():
        for code, part in gains.groupby('contrast', sort=True):
            name = hashlib.sha256(str(code).encode()).hexdigest()
            path = (directory / f'gain_{name}_INTERNAL.pkl').resolve()
            if not path.exists():
                save_pickle(path, part)
            for column, spec in specifications.items():
                yield str(path), metadata[['subject_id', column]], {column:spec}, refs
    results = list(distributed_parts(directory, payloads(), len(contrasts)*len(specifications),
                                    dict(kind='patient_subgroups', specifications=specifications)))
    combined = []
    for index in (0, 1):
        table = pd.concat([part[index] for part in results], ignore_index=True)
        if not table.empty:
            table['_covariate_order'] = table.covariate.map({c:i for i,c in enumerate(specifications)})
            table['_metric_order'] = table.metric.map({m:i for i,m in enumerate(_GAIN_METRICS)})
            keys = (CELL + ['contrast','subgroup']) if index == 0 else [
                'split','seed','horizon_days','contrast','subgroup']
            table = table.sort_values(['_covariate_order', *keys, '_metric_order'], kind='stable').drop(
                columns=['_covariate_order','_metric_order']).reset_index(drop=True)
        combined.append(table)
    return tuple(combined)
