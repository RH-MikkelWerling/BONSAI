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


def tuning_inventory(analysis):
    """Resolve legacy tabular split declarations without accepting test rows.

    Older inventories declare splits on native CIF artifacts, but canonical
    long tabular tables carry their split only in the table itself.
    """
    import pandas as pd

    artifacts = []
    checked = {}
    for artifact in analysis.get("artifacts", []):
        split = artifact.get("split")
        identity = f"{artifact.get('model', '?')}: {artifact['path']}"
        if split is None and artifact.get("format") == "long":
            path = Path(artifact["path"])
            if path not in checked:
                with path.open("rb") as handle:
                    parquet = handle.read(4) == b"PAR1"
                frame = (
                    pd.read_parquet(path, columns=["split"])
                    if parquet
                    else pd.read_csv(path, usecols=["split"], dtype={"split": str})
                )
                checked[path] = set(frame["split"].astype("string").fillna("<missing>"))
            splits = checked[path]
            if splits != {"tuning"}:
                raise ValueError(
                    f"This full rehearsal accepts tuning predictions only; "
                    f"{identity} contains splits {sorted(splits)}"
                )
            split = "tuning"
        if split != "tuning":
            raise ValueError(
                f"This full rehearsal accepts tuning predictions only; "
                f"{identity} declares split={split!r}"
            )
        artifacts.append({**artifact, "split": "tuning"})
    if not artifacts:
        raise ValueError("The source inventory has no tuning prediction artifacts")
    return {**analysis, "artifacts": artifacts}



def figure_checklist(review):
    """One auditable row per renderer-registered figure, including missing ones."""
    rows = []
    for path in sorted(Path(review).rglob('figure_manifest.json')):
        manifest = json.loads(path.read_text())
        family = str(path.parent.relative_to(review))
        for figure in manifest.get('figures', []):
            rows.append(dict(family=family, figure_id=figure['id'], status='rendered',
                             reason='', pdf=str(path.parent / (figure['id'] + '.pdf'))))
        for name, reason in manifest.get('skipped', {}).items():
            rows.append(dict(family=family, figure_id=name, status='missing', reason=reason, pdf=''))
    return rows

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--analysis-root",
        type=Path,
        required=True,
        help="Tuning input workspace with overnight.env and analysis_inventory/analysis.yaml",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--defer-national", action="store_true",
                        help="Keep missing national-only figures explicitly pending; do not block the local release")
    parser.add_argument("--clinical-config", type=Path,
                        help="Explicit approved clinical config; overrides the source workspace recipe")
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
    resolved_splits = sum(a.get("split") is None for a in analysis.get("artifacts", []))
    analysis = tuning_inventory(analysis)
    print(
        f"Tuning inventory: {len(analysis['artifacts'])} artifacts; "
        f"{resolved_splits} legacy split declarations verified from table rows.",
        flush=True,
    )
    if any(
        analysis.get(key, 5) != 5
        for key in ("performance_min_cases", "performance_min_controls")
    ):
        raise ValueError(
            "Expected the frozen 5-case/5-control performance support rule"
        )
    if analysis.get("bootstrap_method", "multiplier") != "multiplier":
        raise ValueError("Expected the current multiplier-bootstrap analysis recipe")
    analysis["bootstrap_method"] = "multiplier"
    # Require the selected-fit receipt used by fresh attribution inference.
    receipt = source / "completion.json"
    primary_fit = json.loads(receipt.read_text())["inputs"]["fit"] if receipt.is_file() else {}
    env = load_environment(source / "overnight.env")
    if primary_fit.get('checkpoint'):
        env['OPERA_PAPER_TASK_STRUCTURE_CHECKPOINT'] = primary_fit['checkpoint']
        env['OPERA_PAPER_ATTRIBUTION_CHECKPOINT'] = primary_fit['checkpoint']
    if primary_fit.get('config'):
        config_path = Path(primary_fit['config'])
        env['OPERA_PAPER_ATTRIBUTION_CONFIG_DIR'] = str(config_path.parent)
        env['OPERA_PAPER_ATTRIBUTION_CONFIG_NAME'] = config_path.stem
        env['OPERA_PAPER_TASK_STRUCTURE_CONFIG_NAME'] = config_path.stem
    if analysis.get('opera_model'):
        env['OPERA_PAPER_OPERA_MODEL'] = analysis['opera_model']
    if args.clinical_config:
        if not args.clinical_config.is_file():
            raise FileNotFoundError(args.clinical_config)
        env["OPERA_PAPER_CLINICAL_SCORE_CONFIG"] = str(args.clinical_config.resolve())
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
    prepared_inventory = paired / "analysis_inventory/analysis.yaml"
    prepared_inventory.write_text(yaml.safe_dump(analysis, sort_keys=False))
    env.update(
        OPERA_PAPER_ANALYSIS_ROOT=str(paired),
        OPERA_PAPER_INVENTORY_SPLITS="tuning",
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
        prepared_inventory_sha256=digest(prepared_inventory),
        legacy_split_declarations_resolved=resolved_splits,
        source_verification_enabled=args.verify_sources,
        national_arm="deferred" if args.defer_national else "included",
        code_manifest_sha256=(
            digest(repo / "code_manifest.sha256") if args.verify_sources else None
        ),
        paths=dict(
            paired=str(paired), analysis=str(complete), ig=str(ig), review=str(review)
        ),
        frozen_inputs="Fitted checkpoints, original tuning predictions, metadata and outcome definitions",
        reused_prior_analysis_outputs=False,
        cache_policy="New output tree; no prior estimates, explanatory exports, plot tables or images copied. Registered base predictions and checkpoints are fixed inputs.",
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
            complete,
            "--output-dir",
            ig,
            "--execute",
        ),
        needs=["complete_figures"],
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
        for name in ("task_structure", "correction_arrows"):
            run(name, module("paper_experiments", name.replace("_", "-"),
                "--env-file", complete / "overnight.env", "--execute"),
                needs=["plot_tables"])
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
                "--whole-cohort-dir",
                root / "whole_cohort",
                "--output-dir",
                review,
            ),
            needs=["plot_tables"],
        )
        if report['steps'].get('assemble', {}).get('status') == 'complete':
            # Only aggregate products created in this new run are added here.
            for name in ('paper_numbers', 'supplement_tables'):
                destination = review / 'reports' / name
                destination.mkdir(parents=True, exist_ok=True)
                for path in (root / name).glob('*'):
                    if path.is_file() and path.suffix in {'.tex', '.json', '.csv'}:
                        shutil.copy2(path, destination / path.name)
            checklist = figure_checklist(review)
            if checklist:
                import csv
                with (review / 'reports/figure_checklist.csv').open('w', newline='') as handle:
                    writer = csv.DictWriter(handle, fieldnames=['family', 'figure_id', 'status', 'reason', 'pdf'])
                    writer.writeheader()
                    writer.writerows(checklist)
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
    completion_manifest = complete / 'completion.json'
    report['nested_completion_status'] = (json.loads(completion_manifest.read_text()).get('status')
                                          if completion_manifest.is_file() else 'missing')
    review_manifest = review / "review_manifest.json"
    if review_manifest.is_file():
        review_report = json.loads(review_manifest.read_text())
        report["review_status"] = review_report.get("status")
        report["review_missing_inputs"] = review_report.get("missing", [])
    report['figure_checklist'] = figure_checklist(review)
    if report['figure_checklist']:
        import csv
        with (root / 'figure_checklist.csv').open('w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=['family', 'figure_id', 'status', 'reason', 'pdf'])
            writer.writeheader()
            writer.writerows(report['figure_checklist'])
    report['missing_figure_count'] = sum(r['status'] != 'rendered' for r in report['figure_checklist'])
    national_ids = {'fig_external', 'supp_national_where', 'supp_embedding_comparison'}
    report['deferred_national_figures'] = [r for r in report['figure_checklist']
        if args.defer_national and r['status'] == 'missing' and r['figure_id'] in national_ids
        and r['reason'].startswith('missing inputs:')]
    deferred = {(r['family'], r['figure_id']) for r in report['deferred_national_figures']}
    report['blocking_missing_figures'] = [r for r in report['figure_checklist']
        if r['status'] != 'rendered' and (r['family'], r['figure_id']) not in deferred]
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
            and report.get("nested_completion_status") == "complete"
            and report.get("review_status") == "review_ready"
            and not report["missing_figure_count"]
            and not report["source_changed_during_run"]
            and not report.get("source_verification_error")
        )
        else "finished_with_gaps"
    )
    if (args.defer_national and report['deferred_national_figures']
            and report.get('nested_completion_status') == 'complete'
            and not report['blocking_missing_figures']
            and not report.get('review_missing_inputs')
            and report.get('review_status') in {'review_ready', 'partial_review'}
            and all(s['status'] == 'complete' for s in report['steps'].values())
            and not report.get('source_verification_error')
            and not report['source_changed_during_run']):
        report['status'] = 'complete_local_national_pending'
    save()
    print(
        f"{report['status']}: {report_path}\nFinal figures: {review}\nReview ZIP: {root / 'paper_review.zip'}",
        flush=True,
    )
    if report["status"] not in {"complete", "complete_local_national_pending"}:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
