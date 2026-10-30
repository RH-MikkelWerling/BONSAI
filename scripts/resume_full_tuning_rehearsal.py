"""Recover an interrupted full tuning rehearsal without rerunning complete stages.

This wrapper uses the existing scientific commands, selected fits and workspace
receipts. complete_local_figures --resume accepts synced source code by default
while retaining scientific-input and completed-output validation.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from zoneinfo import ZoneInfo

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from opera.run.paper_experiments import load_environment, path_value
from opera.run.run_tuning_plot_rebuild import copy_aggregate_inputs


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def timestamp():
    return datetime.now(ZoneInfo("Europe/Copenhagen")).isoformat(timespec="seconds")


def save_report(path, report):
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(report, indent=2) + "\n")
    temporary.replace(path)


@contextmanager
def recovery_lock(root):
    with (root / ".recovery.lock").open("a") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("Another recovery process is using this workspace") from exc
        processes = subprocess.run(
            ["ps", "-ww", "-eo", "pid=,stat=,args="], text=True, capture_output=True, check=True
        ).stdout
        active = [line.strip() for line in processes.splitlines()
                  if root.name in line and " -m opera.run." in line
                  and len(line.split(maxsplit=2)) == 3
                  and not line.split(maxsplit=2)[1].startswith("Z")]
        if active:
            raise RuntimeError("The original run is still active; do not start a second copy:\n"
                               + "\n".join(active))
        yield


def archive_partial(path):
    if path.exists():
        suffix = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        path.rename(path.with_name(path.name + ".interrupted_" + suffix))


def recover(root):
    root = Path(root).resolve()
    report_path = root / "full_rehearsal.json"
    report = json.loads(report_path.read_text())
    source = Path(report["source_analysis"]).resolve()
    paired, complete, ig, review = [root / name for name in ("paired", "analysis", "ig", "review")]
    for name, expected in (("paired", paired), ("analysis", complete), ("ig", ig), ("review", review)):
        if Path(report["paths"][name]).resolve() != expected:
            raise ValueError(f"Recorded {name} path differs from this workspace")
    for path, key in ((source / "analysis_inventory/analysis.yaml", "inventory_sha256"),
                      (paired / "analysis_inventory/analysis.yaml", "prepared_inventory_sha256")):
        if digest(path) != report[key]:
            raise ValueError(f"Frozen inventory changed: {path}")
    # Do not read test rows or discover a new model roster during recovery.
    import yaml
    analysis = yaml.safe_load((paired / "analysis_inventory/analysis.yaml").read_text())
    if analysis.get("estimand") != "cif_ipcw" or not analysis.get("artifacts") or any(
        artifact.get("split") != "tuning" for artifact in analysis["artifacts"]
    ):
        raise ValueError("Recovery requires the prepared tuning-only CIF inventory")
    report.setdefault("recoveries", []).append(dict(started=timestamp(), pid=os.getpid()))
    report["status"] = "running"
    logs = root / "logs"
    logs.mkdir(exist_ok=True)

    def save():
        save_report(report_path, report)

    def module(name, *arguments):
        return [sys.executable, "-m", "opera.run." + name, *map(str, arguments)]

    def run(name, command, *, needs=(), capture=None, fresh_output=None):
        previous = report["steps"].get(name, {})
        if previous.get("status") == "complete":
            print(f"Already complete: {name}", flush=True)
            return True
        if any(report["steps"].get(n, {}).get("status") != "complete" for n in needs):
            report["steps"][name] = dict(status="blocked", dependencies=list(needs))
            save()
            return False
        downstream = {
            "paired_analysis": ["complete_figures"],
            "complete_figures": ["router_vectors", "whole_cohort", "cross_family_metrics", "plot_tables"],
            "integrated_gradients": ["input_restrictions", "plot_tables"],
            "input_restrictions": ["plot_tables"],
            "plot_tables": ["paper_numbers", "supplement_tables", "assemble"],
            "assemble": ["package"],
        }
        pending = list(downstream.get(name, []))
        while pending:
            dependent = pending.pop()
            record = report["steps"].get(dependent)
            if record and record.get("status") == "complete":
                record.update(status="pending", invalidated_by=name)
            pending.extend(downstream.get(dependent, []))
        # Fresh-only report writers cannot reuse an interrupted directory.
        # Preserve it for diagnosis rather than deleting partial results.
        if fresh_output is not None:
            archive_partial(fresh_output)
        log = logs / f"{name}.recovery.log"
        attempts = list(previous.get("previous_attempts", []))
        if previous:
            attempts.append({key: value for key, value in previous.items() if key != "previous_attempts"})
        report["steps"][name] = dict(status="running", command=command, log=str(log),
                                    previous_attempts=attempts, restarted=timestamp())
        save()
        print(f"{timestamp()} {name}", flush=True)
        with log.open("a") as handle:
            handle.write(f"\nRecovery attempt: {timestamp()}\n")
            handle.flush()
            result = subprocess.run(command, cwd=REPO, stdout=handle, stderr=subprocess.STDOUT)
        if capture and result.returncode == 0:
            # JSON-producing audits emit their report to stdout. Do not copy
            # the recovery log's timestamps into that machine-readable file.
            text = log.read_text()
            decoder = json.JSONDecoder()
            value = None
            for offset, char in enumerate(text):
                if char == "{":
                    try:
                        candidate, end = decoder.raw_decode(text, offset)
                    except json.JSONDecodeError:
                        continue
                    if not text[end:].strip():
                        value = candidate
                        break
            if value is None:
                raise ValueError(f"Audit did not end with a valid JSON report: {log}")
            Path(capture).write_text(json.dumps(value, indent=2) + "\n")
        report["steps"][name].update(status="complete" if result.returncode == 0 else "failed",
                                     exit_code=result.returncode)
        save()
        return result.returncode == 0

    save()
    # The replay and paired estimates normally completed before the interruption.
    run("router_replay", module("verify_router_predictions", "--env-file", source / "overnight.env",
        "--device", "cuda", "--patients-per-cohort", 2, "--output-dir", root / "replay"),
        fresh_output=root / "replay")
    run("paired_analysis", module("paper_experiments", "analyse", "--env-file", paired / "overnight.env", "--execute"),
        fresh_output=paired / "prediction_analysis")
    if (complete.exists() and not (complete / "completion.json").is_file()
            and report["steps"].get("complete_figures", {}).get("status") != "complete"):
        archive_partial(complete)
    completion_args = ["--resume"] if (complete / "completion.json").is_file() else []
    run("complete_figures", module("complete_local_figures", "--analysis-root", paired,
        "--output-dir", complete, *completion_args), needs=["paired_analysis"])
    if report["steps"]["complete_figures"]["status"] != "complete":
        report["status"] = "finished_with_gaps"
        save()
        raise RuntimeError(f"Figure completion failed; inspect {logs / 'complete_figures.recovery.log'}")
    run("router_vectors", module("audit_router_vectors", "--vectors-root", complete / "routed_vectors",
                                 "--output", root / "router_vectors.json"), needs=["complete_figures"])
    run("whole_cohort", module("evaluate_whole_cohort_outcomes", "--analysis-config",
        complete / "analysis_inventory/analysis.yaml", "--models", "OPERA", "J_L", "X", "XP",
        "--split", "tuning", "--horizons", 30, 60, 90, 180, 365, 730,
        "--output-dir", root / "whole_cohort"), needs=["complete_figures"],
        fresh_output=root / "whole_cohort")
    refresh = ig / "refresh_manifest.json"
    if report["steps"].get("integrated_gradients", {}).get("status") != "complete" and refresh.is_file():
        fit = json.loads((source / "completion.json").read_text())["inputs"]["fit"]
        if json.loads(refresh.read_text())["fit"] != fit:
            raise ValueError("Existing attribution workspace identifies a different fit")
        for key in ("checkpoint", "config"):
            if digest(fit[key]) != fit[key + "_sha256"]:
                raise ValueError(f"Selected attribution {key} changed")
        # Existing IG exports support their own checked patient-level resume.
        run("integrated_gradients", [sys.executable, str(Path(__file__).resolve()),
            "--attribution-workspace", str(ig)])
    else:
        if ig.exists() and not refresh.is_file() and report["steps"].get("integrated_gradients", {}).get("status") != "complete":
            archive_partial(ig)
        run("integrated_gradients", module("prepare_attribution_refresh", "--analysis-root", source,
                                         "--output-dir", ig, "--execute"))
    run("input_restrictions", module("paper_experiments", "input-restrictions", "--env-file",
        ig / "overnight.env", "--execute"), needs=["integrated_gradients"])
    if report["steps"]["integrated_gradients"]["status"] == "complete":
        report["fresh_ig_summary_files"] = copy_aggregate_inputs(ig / "attributions", complete / "attributions")
    if report["steps"]["input_restrictions"]["status"] == "complete":
        report["fresh_restriction_summary_files"] = copy_aggregate_inputs(ig / "input_restrictions", complete / "input_restrictions")
    run("cross_family_metrics", module("audit_tuning_router", "--env-file", complete / "overnight.env",
        "--analysis-root", complete, "--recompute-metrics", "--metric-models", "X", "XP", "J_L", "OPERA"),
        capture=root / "cross_family_metrics.json", needs=["paired_analysis"])
    run("plot_tables", module("paper_experiments", "plot-tables", "--env-file", complete / "overnight.env", "--execute"))
    env = load_environment(complete / "overnight.env")
    run("paper_numbers", module("build_paper_numbers", "--analysis-root", complete,
        "--cell-metadata", path_value(env, "OPERA_PAPER_CELL_METADATA"), "--split", "tuning",
        "--output-dir", root / "paper_numbers"), needs=["plot_tables"])
    run("supplement_tables", module("render_supplement_tables", "--config",
        complete / "figure_inputs/supplement_tables.yaml", "--output-dir", root / "supplement_tables",
        "--execute"), needs=["plot_tables"], fresh_output=root / "supplement_tables")
    run("assemble", module("assemble_figure_review", "--analysis-root", complete,
        "--attribution-dir", ig / "attributions", "--restriction-dir", ig / "input_restrictions",
        "--output-dir", review), needs=["plot_tables"], fresh_output=review)
    run("package", module("package_figure_review", "--review-dir", review, "--analysis-root", complete,
        "--output", root / "paper_review.zip"), needs=["assemble"], fresh_output=root / "paper_review.zip")
    manifest = review / "review_manifest.json"
    report["review_status"] = json.loads(manifest.read_text()).get("status") if manifest.is_file() else None
    report["status"] = "complete" if (all(s["status"] == "complete" for s in report["steps"].values())
        and report["review_status"] == "review_ready"
        and not report.get("source_changed_during_run")
        and not report.get("source_verification_error")) else "finished_with_gaps"
    report["recoveries"][-1]["finished"] = timestamp()
    save()
    print(f"{report['status']}: {report_path}\nReview ZIP: {root / 'paper_review.zip'}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--output-dir", type=Path, help="Existing full_tuning_rehearsal workspace")
    group.add_argument("--attribution-workspace", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.attribution_workspace is not None:
        for action in ("attributions", "attribution-summary"):
            subprocess.run([sys.executable, "-m", "opera.run.paper_experiments", action,
                            "--env-file", str(args.attribution_workspace / "overnight.env"), "--execute"],
                           cwd=REPO, check=True)
        return
    root = args.output_dir.resolve()
    with recovery_lock(root):
        recover(root)


if __name__ == "__main__":
    main()
