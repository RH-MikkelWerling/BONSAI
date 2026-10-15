"""Run cell/horizon XGBoost and logistic controls under the paper label contract."""

from __future__ import annotations

import argparse
import fcntl
from functools import lru_cache
import json
from pathlib import Path
import subprocess
import sys

from opera.modules.datamodules.MultiCohortContrastiveDataModule import (
    _competing_path,
    _eligibility_path,
    _outcome_path,
)
from opera.functional.outcomes import resolve_registry_start_date
from opera.run.paper_prediction_queue import bundle_runs
from opera.run.run_paper_queue import file_hash


@lru_cache(maxsize=256)
def _cached_hash(path, mtime_ns, size):
    return file_hash(path)


def input_hash(path):
    path = Path(path).resolve()
    stat = path.stat()
    return _cached_hash(str(path), stat.st_mtime_ns, stat.st_size)


def tabular_jobs(bundle_dir, horizons):
    _, rows = bundle_runs(bundle_dir)
    jobs = []
    for row, _, config in rows:
        if row["phase"] != "supervision" or row["regime"] != "S":
            continue
        scope = config["supervision_factorial"]
        fine, outcome = scope["fine_cohort"], scope["outcome"]
        cohort = next(iter(config["cohorts"].values()))
        target = config["outcomes"][outcome]
        for horizon in horizons:
            jobs.append(
                {
                    "seed": row["seed"],
                    "cohort": fine,
                    "outcome": outcome,
                    "horizon_days": horizon,
                    "population": cohort["population_file"],
                    "outcome_path": _outcome_path(cohort["data_dir"], target),
                    "competing_path": _competing_path(cohort["data_dir"], target),
                    "eligibility": _eligibility_path(cohort["data_dir"], target),
                    "registry_start_date": resolve_registry_start_date(cohort, target),
                    "n_hours_start_include": target.get("n_hours_start_include", 1),
                    "source_config_sha256": row["config_sha256"],
                    "relative_dir": f"seed{row['seed']}/{fine}/{outcome}/{horizon}d",
                }
            )
    return sorted(
        jobs, key=lambda j: (j["seed"], j["cohort"], j["outcome"], j["horizon_days"])
    )


SHARED_INPUTS = ("population", "outcome_path", "competing_path", "eligibility",
                 "registry_start_date", "n_hours_start_include")


def pooled_tabular_jobs(bundle_dir, horizons):
    """One XGBoost per outcome and horizon over every disease.

    The tabular counterpart of the disease-pooled neural model: it separates the
    value of pooling across diseases from the value of the neural model. Like
    the neural pooled models it is not told the patient's disease (which is
    partly assigned from codes after the index date); it must infer it from the
    pre-index history. Inputs are those of the outcome's specialists, which must
    agree across diseases.
    """
    groups = {}
    for job in tabular_jobs(bundle_dir, horizons):
        groups.setdefault((job["seed"], job["outcome"], job["horizon_days"]), []).append(job)
    pooled = []
    for (seed, outcome, horizon), members in sorted(groups.items()):
        shared = {key: members[0][key] for key in SHARED_INPUTS}
        if any(member[key] != shared[key] for member in members for key in SHARED_INPUTS):
            raise ValueError(f"Specialist inputs differ across diseases for {outcome}")
        pooled.append({
            "seed": seed, "cohort": "pooled",
            "cohorts": sorted(member["cohort"] for member in members),
            "outcome": outcome, "horizon_days": horizon, **shared,
            "pooled": True,
            "source_config_sha256": sorted(m["source_config_sha256"] for m in members),
            "relative_dir": f"seed{seed}/pooled/{outcome}/{horizon}d",
        })
    return pooled


def job_models(job):
    return ("xgboost",) if job.get("pooled") else ("xgboost", "logistic")


def training_command(job, args):
    output = Path(args.output_dir) / job["relative_dir"]
    command = [
        sys.executable,
        "-m",
        "opera.run.train_tabular_baselines",
        "--features",
        str(args.features),
        "--population",
        job["population"],
        "--cohort_fine_col",
        "cohort_fine",
        "--cohort_fine_value",
        ",".join(job["cohorts"]) if job.get("pooled") else job["cohort"],
        "--cohort",
        job["cohort"],
        "--outcome_name",
        job["outcome"],
        "--outcome",
        job["outcome_path"],
        "--n_hours_start_include",
        str(job["n_hours_start_include"]),
        "--n_hours_end_include",
        str(job["horizon_days"] * 24),
        "--models",
        ",".join(job_models(job)),
        "--known_status_cif",
        "--selection_metric",
        "log_loss",
        "--tune",
        "--seed",
        str(job["seed"]),
        "--output_stem",
        "{model_name}",
        "--output_dir",
        str(output),
    ]
    for key, flag in (
        ("competing_path", "--competing_outcome"),
        ("eligibility", "--eligibility"),
        ("registry_start_date", "--registry_start_date"),
    ):
        if job[key] is not None:
            command += [flag, str(job[key])]
    return command


def run_job(job, args):
    directory = Path(args.output_dir) / job["relative_dir"]
    if not args.execute:
        return job | {"status": "planned", "command": training_command(job, args)}
    signature = {
        "job": job,
        "training_code_sha256": input_hash(
            Path(__file__).with_name("train_tabular_baselines.py")
        ),
        "feature_sha256": input_hash(args.features),
        "input_sha256": {
            key: input_hash(job[key])
            for key in ("population", "outcome_path", "competing_path", "eligibility")
            if job[key]
        },
    }
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / ".tabular.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        intent = directory / "intent.json"
        marker = directory / "training_complete.json"
        if intent.exists() and json.loads(intent.read_text()) != signature:
            raise ValueError("Tabular inputs changed; use a fresh output root")
        if marker.exists():
            saved = json.loads(marker.read_text())
            if saved["inputs"] != signature or any(
                file_hash(directory / name) != digest
                for name, digest in saved["outputs"].items()
            ):
                raise ValueError("Tabular outputs/provenance changed")
            return job | {"status": "reused_verified_fit"}
        intent.write_text(json.dumps(signature, indent=2, default=str) + "\n")
        with (directory / "training.log").open("a") as log:
            subprocess.run(
                training_command(job, args),
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
            )
        expected = [
            f"{model}_{suffix}"
            for model in job_models(job)
            for suffix in ("predictions.csv", "tuning_predictions.csv", "metadata.json")
        ]
        if not all((directory / name).is_file() for name in expected):
            raise RuntimeError("Tabular fit did not produce both models and splits")
        marker.write_text(
            json.dumps(
                {
                    "inputs": signature,
                    "outputs": {name: file_hash(directory / name) for name in expected},
                },
                indent=2,
                default=str,
            )
            + "\n"
        )
    return job | {"status": "fitted"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle-dir", type=Path, required=True)
    parser.add_argument(
        "--runs-dir", type=Path, required=True
    )  # Common dispatcher interface.
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--horizons", default="30,60,90,180,365,730")
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--max-runs", type=int)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--no-pooled", action="store_true",
                        help="Skip the disease-pooled XGBoost per outcome.")
    args = parser.parse_args()
    horizons = tuple(int(x) for x in args.horizons.split(","))
    if (
        not horizons
        or min(horizons) <= 0
        or len(set(horizons)) != len(horizons)
        or args.shard_count < 1
        or not 0 <= args.shard_index < args.shard_count
    ):
        parser.error("Invalid horizons or shard")
    jobs = [*tabular_jobs(args.bundle_dir, horizons),
            *([] if args.no_pooled else pooled_tabular_jobs(args.bundle_dir, horizons))]
    jobs = jobs[args.shard_index :: args.shard_count]
    if args.max_runs is not None:
        if args.max_runs < 1:
            parser.error("max-runs must be positive")
        jobs = jobs[: args.max_runs]
    reports = []
    for job in jobs:
        try:
            reports.append(run_job(job, args))
        except subprocess.CalledProcessError as exc:
            # Sparse/unsupported tasks remain explicit. Continue independent cells.
            reports.append(job | {"status": "failed_fit", "exit_code": exc.returncode})
    print(json.dumps(reports, indent=2, default=str))
    if args.execute:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        (args.output_dir / f"queue_shard{args.shard_index}.json").write_text(
            json.dumps(reports, indent=2, default=str) + "\n"
        )
        if any(report["status"] == "failed_fit" for report in reports):
            raise SystemExit(1)


if __name__ == "__main__":
    main()
