"""Read-only remaining fit-work estimate for the active recovery workload."""
import argparse
from collections import Counter
import json
from pathlib import Path

from opera.run.cutoff_recovery_worker import read, completed
from opera.run.cutoff_recovery_workload import read_policy


def completing_attempt_seconds(path, statuses):
    if not path.is_file():
        return None
    # Repeated launches can append fast reuse attempts. Use the first positive
    # completing attempt, not the sum of every historical invocation.
    for line in path.read_text().splitlines():
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            continue  # A live append may leave the final line unfinished.
        seconds = float(item.get('wall_seconds', 0))
        if item.get('status') in statuses and seconds > 0:
            return seconds
    return None


def estimate(root, *, gpus=8, productive_hours_per_day=24):
    if gpus < 1 or not 0 < productive_hours_per_day <= 24:
        raise ValueError('Use a positive GPU count and 0 < productive hours/day <= 24')
    plan = read(root/'queue_plan.yaml')
    policy = read_policy(root, plan)
    deferred = set(policy['deferred_jobs'])
    counts = Counter()
    remaining, corrected = [], []
    for digest, job in plan['jobs'].items():
        if digest in deferred:
            counts['deferred'] += 1
            continue
        run = root/'neural_runs'/job['row']['run_directory']
        if completed(root, digest, job):
            counts['finished'] += 1
            seconds = completing_attempt_seconds(run/'worker_attempts.jsonl', {'evaluated'})
            if seconds is not None:
                corrected.append(seconds)
            continue
        counts['unfinished'] += 1
        source = Path(job['source_bundle'])
        base = source.parent.parent if source.parent.name == 'bundles' else source.parent
        old = base/'neural_runs'/job['source_row']['run_directory']/'worker_attempts.jsonl'
        seconds = completing_attempt_seconds(old, {'completed'})
        if seconds is None:
            counts['unfinished_without_historical_timing'] += 1
        else:
            remaining.append(seconds)
            counts['unfinished_with_historical_timing'] += 1
    hours = sum(remaining)/3600
    # Running jobs are deliberately counted as whole jobs: elapsed time alone
    # does not identify the remaining number of early-stopped epochs/evaluations.
    active_hours = max(hours/gpus, max(remaining, default=0)/3600)
    return dict(workload_profile=policy['profile'], counts=dict(counts),
        remaining_measured_subset_gpu_hours=hours,
        assumed_concurrent_fit_gpus=gpus,
        ideal_remaining_fit_hours_for_measured_subset=active_hours,
        assumed_productive_hours_per_day=productive_hours_per_day,
        projected_days_for_measured_subset=active_hours/productive_hours_per_day,
        corrected_completed_jobs_with_runtime=len(corrected),
        corrected_completed_fit_and_evaluation_hours=sum(corrected)/3600,
        limitations=[
            'Partial fit-work estimate, not a full-paper completion ETA; unknown jobs are not imputed.',
            'Historical timings cover training; new evaluation/export work can add time.',
            'Pretraining, encoder/dependency waits, CPU analyses and interpretation are not budgeted here.',
            'Assumes the stated GPUs remain available; scheduler gaps and nightly restarts add calendar time.',
            'In-progress fits count at their full historical duration; corrected stopping times may differ.',
            'Unlike the original 844-hour summary, repeated historical reuse/retry attempts are not summed.'])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, default=Path('data/cutoff_recovery_20261121'))
    p.add_argument('--gpus', type=int, default=8)
    p.add_argument('--productive-hours-per-day', type=float, default=24)
    a = p.parse_args()
    print(json.dumps(estimate(a.root.resolve(), gpus=a.gpus,
        productive_hours_per_day=a.productive_hours_per_day), indent=2))


if __name__ == '__main__':
    main()
