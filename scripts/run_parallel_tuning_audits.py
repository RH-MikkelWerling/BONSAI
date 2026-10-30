"""Independent CPU audits of existing tuning outputs, in one fresh directory.

Does not resume the main run, train, encode records, export new source-removal
predictions or inspect held-out data. Independent failures do not prevent the
remaining audits. Return the aggregate-only parallel_audits_review.zip.
"""
from __future__ import annotations

import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import zipfile

import yaml

REPO = Path(__file__).resolve().parents[1]
CLINICAL_FILES = (
    "score_inventory.csv", "cutoff_diagnostics.csv", "classification_at_anchors.csv",
    "classification_at_thresholds.csv", "score_population_flow.csv", "headline_scores.csv",
    "stratification_diagnostics.csv", "risk_group_point_estimates.csv", "score_availability.csv",
    "report_manifest.json", "layout_report.json",
)
WHOLE_COHORT_FILES = (
    "available_population_estimates.csv", "common_population_estimates.csv",
    "paired_contrasts.csv", "model_coverage.csv", "contrast_coverage.csv",
    "common_population_calibration.csv", "manifest.json",
)


def timestamp():
    return datetime.now().astimezone().isoformat(timespec="seconds")


def snapshot_clinical(analysis, output):
    """Copy explicitly named aggregate reports, never patient metadata."""
    output.mkdir()
    missing, copied = [], []
    for name in CLINICAL_FILES:
        path = analysis / "clinical_scores" / name
        if path.is_file():
            shutil.copy2(path, output / name)
            copied.append(name)
        else:
            missing.append(name)
    path = analysis / "configs/config_audit.json"
    if path.is_file():
        shutil.copy2(path, output / "config_audit.json")
        copied.append("config_audit.json")
    else:
        missing.append("config_audit.json")
    receipt = dict(captured=timestamp(), source=str(analysis), copied=copied,
                   missing=missing, interpretation="Existing reports under the current threshold rule. "
                   "No new thresholds or clinical groups were fitted by this snapshot.")
    (output / "snapshot.json").write_text(json.dumps(receipt, indent=2) + "\n")
    return receipt


def commands(analysis, output, *, bootstrap=400, threads=4):
    def module(name, *args):
        return [sys.executable, "-u", "-m", "opera.run." + name, *map(str, args)]
    return [
        ("router_vectors", module("audit_router_vectors", "--vectors-root", analysis / "routed_vectors",
                                  "--output", output / "router_vectors.json")),
        ("cross_family_metrics", module("audit_tuning_router", "--env-file", analysis / "overnight.env",
             "--analysis-root", analysis, "--recompute-metrics", "--metric-models", "X", "XP", "J_L", "OPERA")),
        ("mechanism", module("audit_frozen_router_mechanism", "--analysis-root", analysis,
             "--output-dir", output / "mechanism", "--bootstrap", bootstrap, "--threads", threads)),
        ("whole_cohort", module("evaluate_whole_cohort_outcomes", "--analysis-config",
             analysis / "analysis_inventory/analysis.yaml", "--models", "OPERA", "J_L", "X", "XP",
             "--split", "tuning", "--horizons", 30, 60, 90, 180, 365, 730,
             "--replicates", 1000, "--output-dir", output / "whole_cohort")),
    ]


def package(output):
    # Never recurse over server workspaces, logs or patient-level files.
    allowed = ["parallel_audits.json", "router_vectors.json", "cross_family_metrics.json",
               "mechanism/router_mechanism_review.zip", "clinical/snapshot.json",
               "clinical/config_audit.json"]
    allowed += ["clinical/" + name for name in CLINICAL_FILES]
    allowed += ["whole_cohort/" + name for name in WHOLE_COHORT_FILES]
    path = output / "parallel_audits_review.zip"
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as archive:
        for name in allowed:
            source = output / name
            if source.is_file():
                archive.write(source, name)
    return path


def run(analysis, output, *, workers=2, threads=4, bootstrap=400):
    analysis, output = Path(analysis).resolve(), Path(output).resolve()
    if output == analysis or analysis in output.parents:
        raise ValueError("Use a fresh sibling directory outside the active analysis workspace")
    inventory = yaml.safe_load((analysis / "analysis_inventory/analysis.yaml").read_text())
    artifacts = inventory.get("artifacts", [])
    if inventory.get("estimand") != "cif_ipcw" or not artifacts or any(
            a.get("split") != "tuning" for a in artifacts):
        raise ValueError("This independent launcher accepts a tuning-only CIF inventory")
    output.mkdir(parents=True, exist_ok=False)
    (output / "logs").mkdir()
    report = dict(started=timestamp(), status="running", analysis_root=str(analysis),
        split="tuning", workers=workers, threads=threads, mechanism_bootstrap=bootstrap,
        whole_cohort_bootstrap=1000, steps={},
        interpretation="Completed execution is not scientific sign-off. Existing inputs are read-only. "
        "The main rehearsal receipts and queued GPU stages are unchanged.")
    path = output / "parallel_audits.json"

    def save():
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(report, indent=2) + "\n")
        temporary.replace(path)

    save()
    try:
        snapshot = snapshot_clinical(analysis, output / "clinical")
        required = {"score_inventory.csv", "cutoff_diagnostics.csv", "classification_at_anchors.csv",
                    "score_population_flow.csv", "headline_scores.csv", "config_audit.json"}
        missing = sorted(required.intersection(snapshot["missing"]))
        report["steps"]["clinical_snapshot"] = dict(
            status="incomplete" if missing else "complete", missing_required=missing)
    except OSError as exc:
        report["steps"]["clinical_snapshot"] = dict(status="failed", error_type=type(exc).__name__)
        print(f"Clinical snapshot failed: {exc}. Independent audits will continue.", flush=True)
    save()
    env = {**os.environ, "OPERA_ANALYSIS_WORKERS": str(workers),
           "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
           "CUDA_VISIBLE_DEVICES": ""}
    for name, command in commands(analysis, output, bootstrap=bootstrap, threads=threads):
        print(f"{timestamp()} Starting {name}", flush=True)
        log = output / "logs" / (name + ".log")
        step = dict(status="running", started=timestamp(), command=command, log=str(log))
        report["steps"][name] = step
        save()
        try:
            with log.open("w") as handle:
                if name == "cross_family_metrics":
                    # Keep diagnostic warnings out of the JSON-producing stdout.
                    with (output / "cross_family_metrics.json.tmp").open("w") as stdout:
                        result = subprocess.run(command, cwd=REPO, env=env,
                                                stdout=stdout, stderr=handle)
                else:
                    result = subprocess.run(command, cwd=REPO, env=env,
                                            stdout=handle, stderr=subprocess.STDOUT)
            step.update(exit_code=result.returncode,
                        status="complete" if result.returncode == 0 else "failed")
            if name == "cross_family_metrics" and result.returncode == 0:
                # This command prints a JSON report, without a file-output flag.
                value = json.loads((output / "cross_family_metrics.json.tmp").read_text())
                (output / "cross_family_metrics.json").write_text(json.dumps(value, indent=2) + "\n")
        except (OSError, ValueError) as exc:
            step.update(status="failed", error_type=type(exc).__name__)
            print(f"{name} failed: {exc}. Other independent audits will continue.", flush=True)
        step["finished"] = timestamp()
        save()
        print(f"{timestamp()} {name}: {step['status']}", flush=True)
    report.update(finished=timestamp(), status=("complete" if all(
        step["status"] == "complete" for step in report["steps"].values()) else "finished_with_gaps"))
    save()
    archive = package(output)
    print(f"{report['status']}\nAggregate review ZIP: {archive}", flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--bootstrap", type=int, default=400)
    args = parser.parse_args()
    if args.workers < 1 or args.threads < 1 or args.bootstrap < 50:
        parser.error("Need positive worker/thread counts and at least 50 mechanism bootstrap draws")
    report = run(args.analysis_root, args.output_dir, workers=args.workers,
                 threads=args.threads, bootstrap=args.bootstrap)
    if report["status"] != "complete":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
