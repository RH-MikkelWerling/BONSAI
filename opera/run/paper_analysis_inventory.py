"""Build explicit analysis artifacts and coverage from completed paper queues."""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import os
import shutil
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
import yaml

from opera.functional.cohort_groups import ALL_EVALUATED_FINE, fine_to_grouped
from opera.run.paper_prediction_queue import bundle_runs, canonical_models
from opera.run.paper_tabular_queue import pooled_tabular_jobs, tabular_jobs
from opera.run.run_paper_queue import file_hash


def configured_cells(config):
    cells = []
    for parent, cohort in config["cohorts"].items():
        fines = (
            [cohort["cohort_fine_value"]]
            if cohort.get("cohort_fine_col") == "cohort_fine"
            else [
                fine for fine in ALL_EVALUATED_FINE if fine_to_grouped(fine) == parent
            ]
        )
        for fine in fines:
            for outcome in config["outcomes"]:
                if outcome not in cohort.get("exclude_outcomes", []):
                    cells.append((fine, outcome))
    return cells


def canonical_tabular(prediction_path, job, split, estimand="binary_known_status_unweighted"):
    """Join exact locked labels; extra or missing patient predictions are errors.

    For ``cif_ipcw`` the rows are the full survival risk set with follow-up, so
    censoring weights match the neural models. Tabular models predict only
    status-known patients; a patient censored before the horizon (weight zero)
    therefore has no prediction and gets a flagged placeholder that cannot
    influence any metric. Any other missing prediction is still an error.
    """
    if estimand == "cif_ipcw":
        return _canonical_tabular_risk_set(prediction_path, job, split)
    from opera.run.train_tabular_baselines import outcome_labels
    from opera.evaluation.cohorts import population_subject_ids

    predictions = pd.read_csv(prediction_path)
    allowed = population_subject_ids(
        job["population"],
        cohort_fine_col="cohort_fine",
        cohort_fine_value=job["cohort"],
    )
    if job.get("pooled_source"):
        predictions = predictions[
            predictions.subject_id.astype(str).isin({str(x) for x in allowed})]
    labels = outcome_labels(
        job["outcome_path"],
        split=split,
        n_hours_start_include=job["n_hours_start_include"],
        n_hours_end_include=job["horizon_days"] * 24,
        require_min_followup=True,
        known_status_cif=True,
        competing_outcome_path=job["competing_path"],
        registry_start_date=job["registry_start_date"],
        cohort=job["cohort"],
        outcome_name=job["outcome"],
        eligibility_path=job["eligibility"],
        allowed_subject_ids=allowed,
    )
    predictions["subject_id"] = predictions.subject_id.astype(str)
    labels["subject_id"] = labels.subject_id.astype(str)
    if predictions.subject_id.duplicated().any() or set(predictions.subject_id) != set(
        labels.subject_id
    ):
        raise ValueError(
            "Tabular prediction/label patient sets differ; no automatic intersection"
        )
    table = predictions.merge(
        labels, on="subject_id", how="left", validate="one_to_one"
    )
    return table.assign(
        cohort=job["cohort"],
        outcome=job["outcome"],
        horizon_days=job["horizon_days"],
        split=split,
    )


def _canonical_tabular_risk_set(prediction_path, job, split):
    from opera.run.train_tabular_baselines import outcome_labels
    from opera.evaluation.cohorts import population_subject_ids

    predictions = pd.read_csv(prediction_path)
    predictions["subject_id"] = predictions.subject_id.astype(str)
    allowed = population_subject_ids(
        job["population"], cohort_fine_col="cohort_fine", cohort_fine_value=job["cohort"]
    )
    if job.get("pooled_source"):
        predictions = predictions[predictions.subject_id.isin({str(x) for x in allowed})]
    risk_set = outcome_labels(
        job["outcome_path"],
        split=split,
        n_hours_start_include=job["n_hours_start_include"],
        n_hours_end_include=job["horizon_days"] * 24,
        require_min_followup=False,
        include_survival_fields=True,
        competing_outcome_path=job["competing_path"],
        registry_start_date=job["registry_start_date"],
        cohort=job["cohort"],
        outcome_name=job["outcome"],
        eligibility_path=job["eligibility"],
        allowed_subject_ids=allowed,
    )[["subject_id", "time_days", "event"]]
    risk_set["subject_id"] = risk_set.subject_id.astype(str)
    if predictions.subject_id.duplicated().any() or not set(predictions.subject_id) <= set(
        risk_set.subject_id
    ):
        raise ValueError("Tabular predictions include patients outside the locked risk set")
    table = risk_set.merge(
        predictions[["subject_id", "probability"]], on="subject_id", how="left",
        validate="one_to_one",
    )
    horizon = float(job["horizon_days"])
    unknown = (table.time_days < horizon) & (table.event == 0)
    absent = table.probability.isna()
    if (absent & ~unknown).any():
        raise ValueError(
            "Tabular predictions are missing for status-known patients; no automatic intersection"
        )
    table["prediction_absent_zero_weight"] = absent
    table["probability"] = table.probability.fillna(0.0)
    return table.assign(
        cohort=job["cohort"], outcome=job["outcome"], horizon_days=job["horizon_days"],
        split=split,
    )


def _stat_key(path):
    """Identity of an input file without reading it: path, size, modification time."""
    if not path:
        return None
    path = Path(path).resolve()
    stat = path.stat()
    return [str(path), stat.st_size, stat.st_mtime_ns]


def tabular_code_version():
    """Hash of the code that turns a prediction file into a canonical table."""
    from opera.run import train_tabular_baselines
    source = "".join(inspect.getsource(f) for f in (
        canonical_tabular, _canonical_tabular_risk_set, train_tabular_baselines.outcome_labels))
    return hashlib.sha256(source.encode()).hexdigest()


def tabular_cache_key(prediction_path, job, split, estimand, code_version):
    payload = {
        "prediction": _stat_key(prediction_path), "job": job, "split": split, "estimand": estimand,
        "inputs": {key: _stat_key(job.get(key))
                   for key in ("population", "outcome_path", "competing_path", "eligibility")},
        "code": code_version,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()


def _tabular_task(task):
    """Build (or reuse from the cache) one canonical tabular table and write it.

    Returns "built" or "cached", or "empty" (no file kept) when the disease has
    no patients in this split, e.g. NMZL on tuning."""
    prediction_path, job, split, estimand, destination, cache_dir, code_version = task
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    cached = None
    if cache_dir:
        key = tabular_cache_key(prediction_path, job, split, estimand, code_version)
        cached = Path(cache_dir) / key[:2] / f"{key}.parquet"
        if cached.is_file():
            if pq.read_metadata(cached).num_rows == 0:
                return "empty"
            shutil.copyfile(cached, destination)
            return "cached"
    table = canonical_tabular(prediction_path, job, split, estimand)
    table.to_parquet(destination, index=False)
    if cached is not None:
        cached.parent.mkdir(parents=True, exist_ok=True)
        temporary = cached.with_suffix(f".{os.getpid()}.tmp")
        shutil.copyfile(destination, temporary)
        os.replace(temporary, cached)  # atomic: concurrent inventories never see half a file
    if table.empty:
        destination.unlink()
        return "empty"
    return "built"


def _outputs_unchanged(directory, saved, verify):
    """Presence of a finished fit's outputs; their full hashes with ``verify``."""
    if verify:
        return all(file_hash(directory / name) == digest for name, digest in saved["outputs"].items())
    return all((directory / name).is_file() for name in saved["outputs"])


def build_inventory(args):
    output = Path(args.output_dir)
    if output.exists():
        raise FileExistsError("Use a fresh analysis inventory directory")
    # Fail before the long collection, not after it.
    missing = [f"{name}: {path}" for name, path in (
        ("template", args.template), ("patient metadata", args.patient_metadata),
        ("cell metadata", args.cell_metadata)) if not Path(path).is_file()]
    if missing:
        raise FileNotFoundError("Inventory inputs missing (build them first):\n  " + "\n  ".join(missing))
    bundle, rows = bundle_runs(args.bundle_dir)
    template = yaml.safe_load(Path(args.template).read_text())
    estimand = getattr(args, "estimand", None) or template.get(
        "estimand", "binary_known_status_unweighted"
    )
    artifacts, coverage, tabular_tables = [], [], []
    route_export = {}
    wanted_horizons = {int(x) for x in args.analysis_horizons.split(",")}
    splits = tuple(getattr(args, "splits", None) or ("tuning", "held_out"))
    if not set(splits) <= {"tuning", "held_out"}:
        raise ValueError("--splits takes tuning and/or held_out")
    for row, _, config in rows:
        aliases = canonical_models(row, config, bundle)
        if not aliases:
            continue
        horizons = [
            day
            for day in config["training"]["evaluation_horizons_days"]
            if day in wanted_horizons
        ]
        for split in splits:
            directory = Path(args.prediction_dir) / row["run_directory"] / split
            predictions = directory / "direct_cr_predictions.parquet"
            marker = directory / "evaluation_complete.json"
            source_run = Path(args.runs_dir) / row["run_directory"]
            ready = (
                predictions.is_file()
                and marker.is_file()
                and (source_run / "training_complete.json").is_file()
            )
            if ready:
                saved = json.loads(marker.read_text())
                if (
                    saved["inputs"]["checkpoint_sha256"]
                    != file_hash(source_run / "best.ckpt")
                    or saved["inputs"]["config_sha256"] != row["config_sha256"]
                    or saved["prediction_sha256"] != file_hash(predictions)
                ):
                    raise ValueError(
                        "Neural predictions differ from evaluation provenance"
                    )
            routes = directory / "source_route_weights.parquet"
            if (ready and split == "held_out" and routes.is_file()
                    and template.get("route_model", "R_L") in aliases):
                # The locked router's held-out routing weights describe which
                # adapters patients used; tuning candidates never supply them.
                route_export = {"route_weights": str(routes.resolve()),
                                "route_seed": row["seed"]}
            for alias in aliases:
                if ready:
                    artifacts.append(
                        {
                            "path": str(predictions.resolve()),
                            "format": "direct_cif",
                            "include_horizons_days": sorted(wanted_horizons),
                            "model": alias,
                            "seed": row["seed"],
                            "split": split,
                            "data_contract": args.data_contract,
                            "cohort_from_metadata": True,
                        }
                    )
                for cohort, outcome in configured_cells(config):
                    for horizon in horizons:
                        coverage.append(
                            {
                                "model": alias,
                                "seed": row["seed"],
                                "split": split,
                                "cohort": cohort,
                                "outcome": outcome,
                                "horizon_days": horizon,
                                "status": "prediction_available"
                                if ready
                                else "not_estimable"
                                if (source_run / "not_estimable.json").is_file()
                                else "evaluation_missing",
                            }
                        )
    horizons = tuple(int(x) for x in args.tabular_horizons.split(","))
    # Per-cell tabular arms: information-matched XGBoost and logistic regression,
    # and XGBoost with quality-registry variables added (XR) when it was run.
    arms = [(Path(args.tabular_dir), (("xgboost", "X"), ("logistic", "LR")), "")]
    registry_dir = getattr(args, "registry_tabular_dir", None)
    if registry_dir and Path(registry_dir).is_dir():
        arms.append((Path(registry_dir), (("xgboost", "XR"),), "registry__"))
    for (tabular_dir, models, prefix), job in (
            (arm, job) for arm in arms for job in tabular_jobs(args.bundle_dir, horizons)):
        if job["horizon_days"] not in wanted_horizons:
            continue
        directory = tabular_dir / job["relative_dir"]
        marker = directory / "training_complete.json"
        complete = marker.is_file()
        if complete:
            saved = json.loads(marker.read_text())
            if saved["inputs"]["job"] != job or not _outputs_unchanged(
                    directory, saved, getattr(args, "verify_outputs", False)):
                raise ValueError("Tabular outputs differ from completed-fit provenance")
        for model, alias in models:
            for split in splits:
                filename = f"{model}_{'tuning_predictions' if split == 'tuning' else 'predictions'}.csv"
                ready = complete and (directory / filename).is_file()
                coverage.append(
                    {
                        "model": alias,
                        "seed": job["seed"],
                        "split": split,
                        "cohort": job["cohort"],
                        "outcome": job["outcome"],
                        "horizon_days": job["horizon_days"],
                        "status": "prediction_available"
                        if ready
                        else "tabular_missing",
                    }
                )
                if ready:
                    destination = (
                        output
                        / "tabular"
                        / (
                            prefix + job["relative_dir"].replace("/", "__")
                            + f"__{model}__{split}.parquet"
                        )
                    )
                    tabular_tables.append((str(directory / filename), job, split, destination, alias))
                    artifacts.append(
                        {
                            "path": str(destination.resolve()),
                            "format": "long",
                            "model": alias,
                            "seed": job["seed"],
                            "data_contract": args.data_contract,
                        }
                    )
    for job in pooled_tabular_jobs(args.bundle_dir, horizons):
        # One pooled XGBoost per outcome, evaluated in each disease's cell (XP).
        if job["horizon_days"] not in wanted_horizons:
            continue
        directory = Path(args.tabular_dir) / job["relative_dir"]
        marker = directory / "training_complete.json"
        complete = marker.is_file()
        if complete:
            saved = json.loads(marker.read_text())
            if saved["inputs"]["job"] != job or not _outputs_unchanged(
                    directory, saved, getattr(args, "verify_outputs", False)):
                raise ValueError("Pooled tabular outputs differ from fit provenance")
        for split in splits:
            filename = f"xgboost_{'tuning_predictions' if split == 'tuning' else 'predictions'}.csv"
            ready = complete and (directory / filename).is_file()
            for fine in job["cohorts"]:
                cell_job = {**job, "cohort": fine, "pooled_source": True}
                coverage.append({
                    "model": "XP", "seed": job["seed"], "split": split, "cohort": fine,
                    "outcome": job["outcome"], "horizon_days": job["horizon_days"],
                    "status": "prediction_available" if ready else "tabular_missing"})
                if not ready:
                    continue
                destination = output / "tabular" / (
                    job["relative_dir"].replace("/", "__") + f"__{fine}__xgboost_pooled__{split}.parquet")
                tabular_tables.append((str(directory / filename), cell_job, split, destination, "XP"))
                artifacts.append({"path": str(destination.resolve()), "format": "long",
                                  "model": "XP", "seed": job["seed"],
                                  "data_contract": args.data_contract})
    if not artifacts:
        raise ValueError(
            "No completed prediction artifacts; evaluate completed fits first"
        )
    coverage_table = pd.DataFrame(coverage)
    if coverage_table.duplicated(
        ["model", "seed", "split", "cohort", "outcome", "horizon_days"]
    ).any():
        raise ValueError("Ambiguous duplicate canonical model cells")
    template.update(route_export)
    template.update(
        estimand=estimand,
        analysis_horizons_days=sorted(wanted_horizons),
        artifacts=artifacts,
        patient_metadata=str(Path(args.patient_metadata).resolve()),
        cell_metadata=str(Path(args.cell_metadata).resolve()),
        coverage_roster=str((output / "coverage.csv").resolve()),
    )
    # Resolve recipe paths now; later commands may run from another directory.
    template["contrast_manifest"] = str(
        (
            Path(__file__).resolve().parents[1]
            / "configs/manifests/final_contrasts_20261014.yaml"
        ).resolve()
    )
    report = {
        "status": "inventory_built_not_results_analysed",
        "data_contract": args.data_contract,
        "n_artifacts": len(artifacts),
        "coverage_counts": coverage_table.status.value_counts().to_dict(),
        "source_sha256": {
            str(Path(path).resolve()): file_hash(path)
            for path in (
                Path(args.bundle_dir) / "bundle.json",
                args.template,
                args.patient_metadata,
                args.cell_metadata,
            )
        },
    }
    # All joins/validation precede writes: tables are built into a staging
    # directory and moved into the inventory only when every one succeeded.
    staging = output.parent / f".{output.name}.staging"
    if staging.exists():
        shutil.rmtree(staging)
    cache_dir = getattr(args, "tabular_cache", None)
    code_version = tabular_code_version()
    tasks = [(path, job, split, estimand, str(staging / destination.relative_to(output)),
              str(cache_dir) if cache_dir else None, code_version)
             for path, job, split, destination, _ in tabular_tables]
    workers = max(1, int(getattr(args, "workers", 1) or 1))
    try:
        if workers == 1:
            outcomes = [_tabular_task(task) for task in tasks]
        else:
            with ProcessPoolExecutor(max_workers=workers) as pool:
                outcomes = list(pool.map(_tabular_task, tasks, chunksize=16))
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    report["tabular_tables"] = {"built": outcomes.count("built"), "from_cache": outcomes.count("cached"),
                                "empty_split": outcomes.count("empty")}
    # A disease with no patients in a split (NMZL on tuning) has nothing to
    # analyse there: keep it out of the analysis and say so in the coverage.
    empty = [entry for entry, outcome in zip(tabular_tables, outcomes) if outcome == "empty"]
    if empty:
        dropped = {str(destination.resolve()) for *_, destination, _ in empty}
        template["artifacts"] = [a for a in template["artifacts"] if a["path"] not in dropped]
        for _, job, split, _, alias in empty:
            row = ((coverage_table.model == alias) & (coverage_table.seed == job["seed"])
                   & (coverage_table.split == split) & (coverage_table.cohort == job["cohort"])
                   & (coverage_table.outcome == job["outcome"])
                   & (coverage_table.horizon_days == job["horizon_days"]))
            coverage_table.loc[row, "status"] = "no_patients_in_split"
        report["n_artifacts"] = len(template["artifacts"])
        report["coverage_counts"] = coverage_table.status.value_counts().to_dict()
    output.mkdir(parents=True)
    if (staging / "tabular").is_dir():
        os.replace(staging / "tabular", output / "tabular")
    shutil.rmtree(staging, ignore_errors=True)
    coverage_table.to_csv(output / "coverage.csv", index=False)
    (output / "analysis.yaml").write_text(yaml.safe_dump(template, sort_keys=False))
    (output / "inventory_manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "bundle-dir",
        "runs-dir",
        "prediction-dir",
        "tabular-dir",
        "template",
        "patient-metadata",
        "cell-metadata",
        "output-dir",
    ):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--data-contract", required=True)
    parser.add_argument("--analysis-horizons", default="365")
    parser.add_argument("--registry-tabular-dir", type=Path,
                        help="Registry-augmented XGBoost runs (model XR), if run.")
    parser.add_argument("--tabular-horizons", default="30,60,90,180,365,730")
    parser.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1),
                        help="Processes building the canonical tabular tables")
    parser.add_argument("--tabular-cache", type=Path,
                        help="Cache of canonical tabular tables, keyed by every input and the code")
    parser.add_argument("--splits", nargs="+", choices=("tuning", "held_out"),
                        help="Splits to collect (default both; a tuning rehearsal needs only tuning)")
    parser.add_argument("--verify-outputs", action="store_true",
                        help="Re-hash every finished tabular fit's outputs (slow on network storage)")
    parser.add_argument(
        "--estimand",
        choices=("binary_known_status_unweighted", "cif_ipcw"),
        default="binary_known_status_unweighted",
        help="Known-status binary or full-risk-set IPCW cumulative incidence",
    )
    args = parser.parse_args()
    if not args.data_contract or "REPLACE" in args.data_contract:
        parser.error("Use the actual frozen data-contract identity")
    print(json.dumps(build_inventory(args), indent=2))


if __name__ == "__main__":
    main()
