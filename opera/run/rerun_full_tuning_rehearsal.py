"""Fresh tuning analyses and explanatory inference from frozen model inputs.

No training or held-out evaluation. Previous analyses/figures are never copied.
The locked inventory and its original base prediction exports remain inputs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import yaml

from opera.run.paper_experiments import load_environment, path_value
from opera.run.run_tuning_plot_rebuild import copy_aggregate_inputs


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_sources(repo):
    """Catch incomplete server syncs before launching expensive work."""
    errors = []
    for line in (repo / "code_manifest.sha256").read_text().splitlines():
        expected, name = line.split("  ", 1)
        if not name.startswith(("opera/", "bonsai/")):
            continue
        path = repo / name
        if not path.is_file() or digest(path) != expected:
            errors.append(name)
    if errors:
        raise ValueError(
            "Source checksum mismatch. Sync source and code_manifest.sha256: "
            + ", ".join(errors)
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--analysis-root",
        type=Path,
        required=True,
        help="Existing completed tuning workspace with completion.json",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--verify-sources",
        action="store_true",
        help="Opt in to source-manifest checksum checks (disabled by default)",
    )
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[2]
    if args.verify_sources:
        verify_sources(repo)
    else:
        print("Source-manifest verification skipped (default).", flush=True)
    source = args.analysis_root.resolve()
    inventory = source / "analysis_inventory/analysis.yaml"
    analysis = yaml.safe_load(inventory.read_text())
    if analysis.get("estimand") != "cif_ipcw":
        raise ValueError("The locked source analysis must declare estimand=cif_ipcw")
    if any(a.get("split") != "tuning" for a in analysis["artifacts"]):
        raise ValueError("This full rehearsal accepts a tuning-only inventory")
    if any(
        analysis.get(key, 5) != 5
        for key in ("performance_min_cases", "performance_min_controls")
    ):
        raise ValueError(
            "Expected the frozen 5-case/5-control performance support rule"
        )
    if analysis.get("bootstrap_method", "multiplier") != "multiplier":
        raise ValueError("Expected the current multiplier-bootstrap analysis recipe")
    # Require the selected-fit receipt used by fresh attribution inference.
    json.loads((source / "completion.json").read_text())["inputs"]["fit"]
    env = load_environment(source / "overnight.env")
    root = args.output_dir.resolve()
    if root == source or source in root.parents:
        raise ValueError("Use a fresh sibling output directory")
    root.mkdir(parents=True, exist_ok=False)
    logs = root / "logs"
    logs.mkdir()
    paired, complete, ig, review = [
        root / n for n in ("paired", "analysis", "ig", "review")
    ]
    (paired / "analysis_inventory").mkdir(parents=True)
    # Reuse declarations, not estimates, figures or auxiliary aggregate caches.
    shutil.copytree(
        source / "analysis_inventory", paired / "analysis_inventory", dirs_exist_ok=True
    )
    env.update(
        OPERA_PAPER_ANALYSIS_ROOT=str(paired),
        OPERA_PAPER_PLOT_SPLIT="tuning",
        OPERA_PAPER_FIGURE_MARK="TUNING SET — not test results",
    )
    env_path = paired / "overnight.env"
    env_path.write_text(
        "".join(
            f"{k}={json.dumps(str(v), ensure_ascii=False)}\n"
            for k, v in env.items()
            if k.startswith(("OPERA_", "BONSAI_"))
        )
    )
    env_path.chmod(0o600)
    report_path = root / "full_rehearsal.json"
    report = dict(
        status="running",
        source_analysis=str(source),
        inventory_sha256=digest(inventory),
        source_verification_enabled=args.verify_sources,
        code_manifest_sha256=(
            digest(repo / "code_manifest.sha256") if args.verify_sources else None
        ),
        paths=dict(
            paired=str(paired), analysis=str(complete), ig=str(ig), review=str(review)
        ),
        frozen_inputs="Fitted checkpoints, original tuning predictions, metadata and outcome definitions",
        reused_prior_analysis_outputs=False,
        steps={},
    )

    def save():
        report_path.write_text(json.dumps(report, indent=2))

    def module(name, *arguments):
        return [sys.executable, "-m", "opera.run." + name, *map(str, arguments)]

    def run(name, command, *, needs=(), capture=None):
        if any(report["steps"].get(n, {}).get("status") != "complete" for n in needs):
            report["steps"][name] = dict(status="blocked", dependencies=list(needs))
            save()
            return False
        log = logs / f"{name}.log"
        report["steps"][name] = dict(status="running", command=command, log=str(log))
        save()
        print(
            f"{datetime.now(ZoneInfo('Europe/Copenhagen')):%H:%M:%S} {name}", flush=True
        )
        with log.open("w") as handle:
            result = subprocess.run(
                command, cwd=repo, stdout=handle, stderr=subprocess.STDOUT
            )
        if capture and result.returncode == 0:
            shutil.copy2(log, capture)
        report["steps"][name].update(
            status="complete" if result.returncode == 0 else "failed",
            exit_code=result.returncode,
        )
        save()
        return result.returncode == 0

    save()
    print(f"Fresh rehearsal: {root}\nProgress: {report_path}\nLogs: {logs}", flush=True)
    run(
        "router_replay",
        module(
            "verify_router_predictions",
            "--env-file",
            source / "overnight.env",
            "--device",
            "cuda",
            "--patients-per-cohort",
            "2",
            "--output-dir",
            root / "replay",
        ),
    )
    run(
        "paired_analysis",
        module("paper_experiments", "analyse", "--env-file", env_path, "--execute"),
    )
    run(
        "complete_figures",
        module(
            "complete_local_figures",
            "--analysis-root",
            paired,
            "--output-dir",
            complete,
        ),
        needs=["paired_analysis"],
    )
    run(
        "router_vectors",
        module(
            "audit_router_vectors",
            "--vectors-root",
            complete / "routed_vectors",
            "--output",
            root / "router_vectors.json",
        ),
        needs=["complete_figures"],
    )
    run(
        "whole_cohort",
        module(
            "evaluate_whole_cohort_outcomes",
            "--analysis-config",
            complete / "analysis_inventory/analysis.yaml",
            "--models",
            "OPERA",
            "J_L",
            "X",
            "XP",
            "--split",
            "tuning",
            "--horizons",
            "30",
            "60",
            "90",
            "180",
            "365",
            "730",
            "--output-dir",
            root / "whole_cohort",
        ),
        needs=["complete_figures"],
    )
    # This is a new IG directory. The attribution command's internal --resume
    # option cannot find or reuse any previous run's exports here.
    run(
        "integrated_gradients",
        module(
            "prepare_attribution_refresh",
            "--analysis-root",
            source,
            "--output-dir",
            ig,
            "--execute",
        ),
    )
    run(
        "input_restrictions",
        module(
            "paper_experiments",
            "input-restrictions",
            "--env-file",
            ig / "overnight.env",
            "--execute",
        ),
        needs=["integrated_gradients"],
    )
    if (complete / "overnight.env").is_file():
        # Transfer only outputs freshly computed above into the unified report
        # workspace. This also replaces the earlier recent-window-only summary
        # with the full newly computed restrictions, including those windows.
        if report["steps"]["integrated_gradients"]["status"] == "complete":
            report["fresh_ig_summary_files"] = copy_aggregate_inputs(
                ig / "attributions", complete / "attributions"
            )
        if report["steps"]["input_restrictions"]["status"] == "complete":
            report["fresh_restriction_summary_files"] = copy_aggregate_inputs(
                ig / "input_restrictions", complete / "input_restrictions"
            )
        run(
            "cross_family_metrics",
            module(
                "audit_tuning_router",
                "--env-file",
                complete / "overnight.env",
                "--analysis-root",
                complete,
                "--recompute-metrics",
                "--metric-models",
                "X",
                "XP",
                "J_L",
                "OPERA",
            ),
            capture=root / "cross_family_metrics.json",
            needs=["paired_analysis"],
        )
        run(
            "plot_tables",
            module(
                "paper_experiments",
                "plot-tables",
                "--env-file",
                complete / "overnight.env",
                "--execute",
            ),
        )
        current_env = load_environment(complete / "overnight.env")
        run(
            "paper_numbers",
            module(
                "build_paper_numbers",
                "--analysis-root",
                complete,
                "--cell-metadata",
                path_value(current_env, "OPERA_PAPER_CELL_METADATA"),
                "--split",
                "tuning",
                "--output-dir",
                root / "paper_numbers",
            ),
            needs=["plot_tables"],
        )
        run(
            "supplement_tables",
            module(
                "render_supplement_tables",
                "--config",
                complete / "figure_inputs/supplement_tables.yaml",
                "--output-dir",
                root / "supplement_tables",
                "--execute",
            ),
            needs=["plot_tables"],
        )
        # Implicit clinical validation preserves a partial review if an
        # independent clinical job failed. No legacy clinical images are used.
        run(
            "assemble",
            module(
                "assemble_figure_review",
                "--analysis-root",
                complete,
                "--attribution-dir",
                ig / "attributions",
                "--restriction-dir",
                ig / "input_restrictions",
                "--output-dir",
                review,
            ),
            needs=["plot_tables"],
        )
        run(
            "package",
            module(
                "package_figure_review",
                "--review-dir",
                review,
                "--analysis-root",
                complete,
                "--output",
                root / "paper_review.zip",
            ),
            needs=["assemble"],
        )
    else:
        report["missing_analysis_workspace"] = True
    review_manifest = review / "review_manifest.json"
    if review_manifest.is_file():
        report["review_status"] = json.loads(review_manifest.read_text()).get("status")
    report["source_changed_during_run"] = None
    if args.verify_sources:
        try:
            report["source_changed_during_run"] = (
                digest(repo / "code_manifest.sha256") != report["code_manifest_sha256"]
            )
            verify_sources(repo)
        except (OSError, ValueError) as exc:
            report["source_verification_error"] = str(exc)
    report["status"] = (
        "complete"
        if (
            all(s["status"] == "complete" for s in report["steps"].values())
            and report.get("review_status") == "review_ready"
            and not report["source_changed_during_run"]
            and not report.get("source_verification_error")
        )
        else "finished_with_gaps"
    )
    save()
    print(
        f"{report['status']}: {report_path}\nFinal figures: {review}\nReview ZIP: {root / 'paper_review.zip'}",
        flush=True,
    )


if __name__ == "__main__":
    main()
