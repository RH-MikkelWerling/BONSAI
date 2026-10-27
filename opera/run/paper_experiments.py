"""Paper workflow dispatcher. Commands preview by default; --execute runs a stage.

Uses the selected .env without shell-sourcing it. Respects GPU allocation,
freezes configs at preparation, and never guesses the running encoder or router.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

import yaml

from dotenv import dotenv_values

REPO = Path(__file__).resolve().parents[2]
PHASES = (
    "supervision",
    "source_screen",
    "rank_screen",
    "query_screen",
    "final_sources",
    "external",
    "pretraining_control",
    "router_controls",
    "data_efficiency",
    "router_fix",
    "init_rule",  # router-decision only: R vs RSI on the stage-2 bundle
    "router_fix_rule",  # router-decision only: the router fix, stage-3 bundle
)


# CPU steps with process pools (OPERA_ANALYSIS_WORKERS, inventory workers).
CPU_POOL_ACTIONS = {"inventory", "analyse", "plot-tables", "borrowing-map", "paper-numbers",
                    "clinical-scores", "embedding-tables", "router-decision"}
BLAS_THREAD_VARIABLES = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                         "NUMEXPR_NUM_THREADS")


def load_environment(path):
    path = Path(path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Create/configure {path}; see .env.paper.example")
    env = os.environ.copy()
    values = dotenv_values(path)
    env.update({key: value for key, value in values.items() if value is not None})
    # A scheduler allocation takes precedence over any legacy dotenv GPU value.
    if "CUDA_VISIBLE_DEVICES" in os.environ:
        env["CUDA_VISIBLE_DEVICES"] = os.environ["CUDA_VISIBLE_DEVICES"]
    env.setdefault("BONSAI_CONFIG_PATH", str(REPO / "configs"))
    return env


def value(env, key, default=None):
    result = env.get(key) or default
    if result is None:
        raise ValueError(f"Set {key} in the selected .env")
    return str(result)


def path_value(env, key, default=None):
    path = Path(value(env, key, default)).expanduser()
    return (REPO / path).resolve() if not path.is_absolute() else path.resolve()


def external_sequence_dir(env, root):
    """Our patients tokenised in the national encoder's format (external-data)."""
    return path_value(env, "OPERA_PAPER_EXTERNAL_SEQUENCE_DIR",
                      str(root / "external_sequences" / "sequences"))


def module(name, *args):
    return [sys.executable, "-m", name, *map(str, args)]


# The own-sources-only control of each routed design (source removal levels).
OWN_REFERENCE = {"R_AB_L": "RBH_AB_L", "R_O_L": "RBH_O_L"}


def stage_bundles(root):
    """Bundle directories the worker reads: stage 1, stage 2 and any later stage.

    Follow-up runs are added as a new immutable bundle (stage 3, ...) that
    repeats earlier entries; identical runs share their run directory, so
    nothing is retrained. Restart workers after preparing a new stage.
    """
    stages = {root / "bundles" / "stage1", root / "bundles" / "stage2"}
    stages.update(path for path in (root / "bundles").glob("stage*") if path.is_dir())
    return sorted(stages, key=lambda path: int(path.name.removeprefix("stage") or 0))


def build_commands(args, env):
    root = path_value(env, "OPERA_PAPER_ROOT")
    estimand = value(env, "OPERA_PAPER_ANALYSIS_ESTIMAND", "binary_known_status_unweighted")
    # Each estimand gets its own immutable analysis tree by default.
    analysis_root = path_value(
        env,
        "OPERA_PAPER_ANALYSIS_ROOT",
        str(root / "analysis" / f"main365_{'ipcw' if estimand == 'cif_ipcw' else 'known_status'}"),
    )
    stage = (
        2
        if args.phase in ("final_sources", "external") and args.action == "train"
        else args.stage
    )
    bundle = root / "bundles" / f"stage{stage}"
    runs = root / "neural_runs"
    shared = [
        "--bundle-dir",
        bundle,
        "--runs-dir",
        runs,
        "--shard-count",
        args.shard_count,
        "--shard-index",
        args.shard_index,
    ]
    if args.action == "prepare":
        release = path_value(env, "OPERA_RELEASE_DIR")
        command = module(
            "opera.run.prepare_paper_run_bundle",
            "--base-config",
            path_value(env, "OPERA_PAPER_BASE_CONFIG"),
            "--release-registry",
            release / "experiment_registry_primary_v2.yaml",
            "--router-reference-config",
            path_value(env, "OPERA_PAPER_ROUTER_REFERENCE"),
            "--encoder-checkpoint",
            path_value(env, "OPERA_PAPER_LOCAL_CKPT"),
            "--sequence-data-dir",
            path_value(env, "OPERA_SEQUENCE_DATA_DIR"),
            "--seeds",
            value(env, "OPERA_PAPER_SEEDS", "42"),
            "--epochs",
            value(env, "OPERA_PAPER_EPOCHS", "30"),
            "--early-stopping-patience",
            value(env, "OPERA_PAPER_PATIENCE", "5"),
            "--batch-size",
            value(env, "OPERA_PAPER_BATCH_SIZE", "1"),
            *(("--num-workers", env["OPERA_PAPER_NUM_WORKERS"])
              if env.get("OPERA_PAPER_NUM_WORKERS") else ()),
            "--output-dir",
            bundle,
        )
        if stage >= 2:
            command += [
                "--router-lock",
                str(path_value(env, "OPERA_PAPER_ROUTER_LOCK")),
            ]
            if env.get("OPERA_PAPER_ROUTED_MODEL"):
                # The national routed fits use the primary router's readout.
                command += ["--routed-model", env["OPERA_PAPER_ROUTED_MODEL"]]
            if args.include_external:
                command += [
                    "--external-direct-checkpoint",
                    str(root / "external_direct" / "encoder.ckpt"),
                    "--external-dapt-checkpoint",
                    str(root / "external_dapt" / "version_0" / "best.ckpt"),
                    "--external-sequence-dir",
                    str(external_sequence_dir(env, root)),
                ]
            if stage >= 3:
                # Router fix (lock addendum 2026-10-26): stage 3 adds the six
                # fits; every earlier fit is regenerated identically and reused.
                command += ["--router-fix"]
            if stage >= 4:
                # Own sources outside the mix (same addendum, after the hub audits).
                command += ["--router-own"]
            if stage >= 5:
                # Sampling sensitivity (addendum 2026-10-27): J and R_AB with IPW
                # event sampling.
                command += ["--sampling-ipw"]
        elif args.include_external:
            raise ValueError("External downstream matrix requires --stage 2")
        return [command]
    if args.action == "smoke":
        return [
            module(
                "opera.run.paper_smoke",
                "--bundle-dir",
                bundle,
                "--runs-dir",
                root / "smoke_runs",
                "--minimum-local-pretraining-epochs",
                value(env, "OPERA_PAPER_MIN_LOCAL_EPOCHS", "10"),
                "--local-hidden-size",
                value(env, "OPERA_PAPER_LOCAL_WIDTH", "256"),
            )
            + (["--execute"] if args.execute else [])
        ]
    if args.action == "tuning-summary":
        return [
            module(
                "opera.run.paper_tuning_summary",
                "--bundle-dir",
                bundle,
                "--runs-dir",
                runs,
                "--output-dir",
                root / f"tuning_summary_stage{stage}_{args.phase}",
            )
        ]
    if args.action == "prepare-bayes":
        return [
            module(
                "opera.run.prepare_paper_bayes_recipes",
                "--template",
                path_value(
                    env,
                    "OPERA_PAPER_BAYES_TEMPLATE",
                    str(REPO / "opera/configs/bayesian_contrasts.example.yaml"),
                ),
                "--analysis-dir",
                analysis_root / "prediction_analysis",
                "--results-root",
                analysis_root / "bayesian",
                "--seeds",
                value(env, "OPERA_PAPER_SEEDS", "42"),
                "--output-dir",
                analysis_root / "bayesian_recipes",
            )
        ]
    if args.action == "train":
        command = module(
            "opera.run.run_paper_queue",
            *shared,
            "--phase",
            args.phase,
            "--minimum-local-pretraining-epochs",
            value(env, "OPERA_PAPER_MIN_LOCAL_EPOCHS", "10"),
            "--minimum-dapt-epochs",
            value(env, "OPERA_PAPER_MIN_DAPT_EPOCHS", "10"),
            "--external-hidden-size",
            value(env, "OPERA_PAPER_EXTERNAL_WIDTH", "256"),
            "--local-hidden-size",
            value(env, "OPERA_PAPER_LOCAL_WIDTH", "256"),
        )
        if args.phase in ("rank_screen", "query_screen"):
            command += ["--decisions", str(path_value(env, "OPERA_PAPER_DECISIONS"))]
        if args.regime:
            command += ["--regime", args.regime]
        if args.max_runs:
            command += ["--max-runs", str(args.max_runs)]
        return [command + (["--execute"] if args.execute else [])]
    if args.action == "external-check":
        # Preflight on the national checkpoint (CPU, seconds): vocabulary, width,
        # causality, value representation against the data config.
        rope_layout = env.get("OPERA_PAPER_EXTERNAL_SOURCE_ROPE_LAYOUT") or None
        return [module(
            "opera.run.check_external_checkpoint",
            "--checkpoint", path_value(env, "OPERA_PAPER_EXTERNAL_SOURCE_CKPT"),
            "--vocabulary", path_value(env, "OPERA_PAPER_EXTERNAL_SOURCE_VOCAB"),
            "--data-config", value(env, "OPERA_PAPER_EXTERNAL_DATA_CONFIG",
                                   "daly_care_joined_bins_data"),
            "--width", value(env, "OPERA_PAPER_EXTERNAL_WIDTH", "256"),
            *(("--source-rope-layout", rope_layout) if rope_layout else ()))]
    if args.action == "export-external":
        rope_layout = env.get("OPERA_PAPER_EXTERNAL_SOURCE_ROPE_LAYOUT") or None
        return [
            module(
                "bonsai.run.export_external_encoder",
                "--checkpoint",
                path_value(env, "OPERA_PAPER_EXTERNAL_SOURCE_CKPT"),
                "--source-vocab",
                path_value(env, "OPERA_PAPER_EXTERNAL_SOURCE_VOCAB"),
                # National vocabulary (our records tokenised in its format):
                # the export is an exact identity mapping.
                "--target-vocab",
                external_sequence_dir(env, root) / "vocabulary.pt",
                "--seed",
                "42",
                "--output-dir",
                root / "external_direct",
                *(("--source-rope-layout", rope_layout) if rope_layout else ()),
            )
        ]
    if args.action == "external-data":
        # Our patients from the national MEDS, tokenised with the national
        # vocabulary, then audited (coverage, [UNK] share).
        output = root / "external_sequences"
        script = "opera.run.prepare_external_sequences"
        return [
            module(script, "filter", "--output-dir", output,
                   "--external-meds", path_value(env, "OPERA_PAPER_EXTERNAL_MEDS_DIR"),
                   "--local-meds", path_value(env, "OPERA_PAPER_MEDS_DIR")),
            module(script, "tokenize", "--output-dir", output,
                   "--external-vocab", path_value(env, "OPERA_PAPER_EXTERNAL_SOURCE_VOCAB"),
                   "--data-config", value(env, "OPERA_PAPER_EXTERNAL_DATA_CONFIG",
                                          "daly_care_joined_bins_data")),
            module(script, "audit", "--output-dir", output,
                   "--membership", path_value(env, "BONSAI_COHORT_MEMBERSHIP"),
                   "--external-vocab", path_value(env, "OPERA_PAPER_EXTERNAL_SOURCE_VOCAB"),
                   *(("--required-latest", env["OPERA_PAPER_EXTERNAL_REQUIRED_LATEST"])
                     if env.get("OPERA_PAPER_EXTERNAL_REQUIRED_LATEST") else ())),
        ]
    if args.action == "dapt":
        return [
            module(
                "bonsai.run.pretrain",
                "--config-path",
                REPO / "configs",
                "--config-name",
                "pretrain_external_bonsai_dapt_v1",
                f"paths.initialization_ckpt={root / 'external_direct' / 'encoder.ckpt'}",
                f"paths.initialization_vocab={root / 'external_direct' / 'vocabulary.pt'}",
                f"paths.dir={external_sequence_dir(env, root)}",
                f"training.epochs={value(env, 'OPERA_PAPER_MIN_DAPT_EPOCHS', '10')}",
                # The inherited epoch-10 milestone is only valid for longer runs.
                *(() if int(value(env, "OPERA_PAPER_MIN_DAPT_EPOCHS", "10")) > 10
                  else ("training.checkpoint_milestone_epochs=[]",)),
                f"training.batch_size={value(env, 'OPERA_PAPER_DAPT_BATCH_SIZE', '1')}",
                f"training.accumulate_grad_batches={value(env, 'OPERA_PAPER_DAPT_ACCUMULATION', '128')}",
                f"hydra.run.dir={root / 'external_dapt'}",
            )
        ]
    if args.action == "features":
        # One raw matrix per cohort (<cohort_fine>.<ext>) plus all.<ext> for the
        # pooled XGBoost; each is prepared under both profiles.
        source_dir = Path(path_value(env, "OPERA_PAPER_FEATURE_DIR"))
        sources = sorted(
            path for path in source_dir.iterdir()
            if path.suffix.lower() in {".pkl", ".pickle", ".parquet", ".pq", ".csv"}
        ) if source_dir.is_dir() else []
        if not sources:
            raise ValueError(f"No feature matrices (.pkl/.parquet/.csv) in {source_dir}")
        return [
            module(
                "opera.run.prepare_tabular_features",
                "--features",
                *sources,
                "--population",
                path_value(env, "BONSAI_COHORT_MEMBERSHIP"),
                *(("--demographics", path_value(env, "OPERA_PAPER_DEMOGRAPHICS"))
                  if env.get("OPERA_PAPER_DEMOGRAPHICS") else ()),
                "--output_dir",
                root / "features",
                # Primary matrices, and ones keeping the quality-registry
                # predictors for the tabular reference arm (XR).
                "--feature_profile", "sequence_matched", "registry_augmented",
            )
        ]
    if args.action == "feature-check":
        # Do the raw matrices cover the neural cohorts at the same index dates?
        return [module(
            "opera.run.check_tabular_alignment",
            "--bundle-dir", bundle,
            "--feature-dir", path_value(env, "OPERA_PAPER_FEATURE_DIR"),
            "--index-date-column", value(env, "OPERA_PAPER_INDEX_DATE_COLUMN", "index_date"),
            "--output", root / "features" / "alignment.csv",
        )]
    if args.action == "tabular":
        command = module(
            "opera.run.paper_tabular_queue",
            *shared,
            "--features-dir",
            root / "features",
            "--feature-profile",
            "registry_augmented" if args.registry else "sequence_matched",
            "--output-dir",
            root / ("tabular_runs_registry" if args.registry else "tabular_runs"),
            "--horizons",
            value(env, "OPERA_PAPER_TABULAR_HORIZONS", "30,60,90,180,365,730"),
            # Registry arm: per-cell XGBoost only (model XR).
            *(("--per-cell-models", "xgboost", "--no-pooled") if args.registry else ()),
        )
        if args.max_runs:
            command += ["--max-runs", str(args.max_runs)]
        return [command + (["--execute"] if args.execute else [])]
    if args.action == "evaluate":
        command = module(
            "opera.run.paper_prediction_queue",
            *shared,
            "--split",
            args.split,
            "--output-dir",
            root / "predictions",
            "--batch-size",
            value(env, "OPERA_PAPER_BATCH_SIZE", "1"),
        )
        if args.max_runs:
            command += ["--max-runs", str(args.max_runs)]
        if args.export_routes:
            command += ["--export-routes"]
        for only in getattr(args, "only", []):
            command += ["--only", only]
        return [command + (["--execute"] if args.execute else [])]
    if args.action == "source-removal":
        # One evaluation of the locked router writing every borrowing variant:
        # no borrowing, each disease source alone, each left out (stage-2
        # bundle; one GPU pass per split, no training).
        # After the 2026-10-27 amendment the primary routed model is a router-fix
        # fit (stage 3); the original router lives in the stage-2 bundle.
        routed = env.get("OPERA_PAPER_ROUTED_MODEL") or "R_L"
        command = module(
            "opera.run.paper_prediction_queue",
            "--bundle-dir", root / "bundles" / ("stage2" if routed == "R_L" else "stage3"),
            "--runs-dir", runs,
            "--split", args.split,
            "--output-dir", root / "predictions",
            "--batch-size", value(env, "OPERA_PAPER_BATCH_SIZE", "1"),
            "--source-removal",
            "--removal-model", routed,
        )
        return [command + (["--execute"] if args.execute else [])]
    if args.action == "borrowing-map":
        # Borrowing map and prespecified pairs from the borrowing design (CPU).
        bundles = [part for path in stage_bundles(root) for part in ("--bundle-dir", path)]
        return [module(
            "opera.run.source_removal",
            *bundles,
            "--prediction-dir", root / "predictions",
            "--membership", path_value(env, "BONSAI_COHORT_MEMBERSHIP"),
            "--output-dir", analysis_root / "source_removal",
            "--split", args.split,
            "--replicates", value(env, "OPERA_PAPER_REMOVAL_REPLICATES", "1000"),
            "--ablation", getattr(args, "ablation", "mean"),
            "--shard-count", args.shard_count,
            "--shard-index", args.shard_index,
            *(("--model", env["OPERA_PAPER_ROUTED_MODEL"],
                "--own-reference", OWN_REFERENCE[env["OPERA_PAPER_ROUTED_MODEL"]])
              if env.get("OPERA_PAPER_ROUTED_MODEL") not in (None, "", "R_L") else ()),
        )]
    if args.action == "router-decision":
        # Prespecified router rules on tuning predictions (needs `evaluate
        # --split tuning --only supervision:J --only <phase>` first).
        command = module(
            "opera.run.router_decision",
            "--phase", args.phase,
            "--bundle-dir", bundle,
            "--prediction-dir", root / "predictions",
            "--membership", path_value(env, "BONSAI_COHORT_MEMBERSHIP"),
            "--decisions", path_value(env, "OPERA_PAPER_DECISIONS"),
            "--output-dir", root / "router_decisions",
            "--runs-dir", root / "neural_runs",
            "--router-lock", root / "router_lock.json",
        )
        return [command + (["--write"] if getattr(args, "record", False) else [])]
    if args.action == "inventory":
        return [
            module(
                "opera.run.paper_analysis_inventory",
                "--bundle-dir",
                bundle,
                "--runs-dir",
                runs,
                "--prediction-dir",
                root / "predictions",
                "--tabular-dir",
                root / "tabular_runs",
                "--registry-tabular-dir",
                root / "tabular_runs_registry",
                "--tabular-cache",
                root / "tabular_inventory_cache",
                "--workers",
                value(env, "OPERA_PAPER_INVENTORY_WORKERS", "8"),
                *(("--splits", *env["OPERA_PAPER_INVENTORY_SPLITS"].split(","))
                  if env.get("OPERA_PAPER_INVENTORY_SPLITS") else ()),
                *(("--opera-model", env["OPERA_PAPER_OPERA_MODEL"])
                  if env.get("OPERA_PAPER_OPERA_MODEL") else ()),
                *(("--routed-model", env["OPERA_PAPER_ROUTED_MODEL"])
                  if env.get("OPERA_PAPER_ROUTED_MODEL") else ()),
                "--template",
                path_value(
                    env,
                    "OPERA_PAPER_ANALYSIS_TEMPLATE",
                    str(REPO / "opera/configs/locked_prediction_analysis.example.yaml"),
                ),
                "--patient-metadata",
                path_value(env, "OPERA_PAPER_PATIENT_METADATA"),
                "--cell-metadata",
                path_value(env, "OPERA_PAPER_CELL_METADATA"),
                "--data-contract",
                value(env, "OPERA_PAPER_DATA_CONTRACT"),
                "--analysis-horizons",
                value(env, "OPERA_PAPER_ANALYSIS_HORIZONS", "365"),
                "--tabular-horizons",
                value(env, "OPERA_PAPER_TABULAR_HORIZONS", "30,60,90,180,365,730"),
                "--estimand",
                value(env, "OPERA_PAPER_ANALYSIS_ESTIMAND", "binary_known_status_unweighted"),
                "--output-dir",
                analysis_root / "analysis_inventory",
            )
        ]
    if args.action == "paper-numbers":
        # Every number reported in the text, and Table 2, from the aggregate outputs.
        return [module("opera.run.build_paper_numbers",
                       "--analysis-root", analysis_root,
                       "--cell-metadata", path_value(env, "OPERA_PAPER_CELL_METADATA"),
                       "--split", value(env, "OPERA_PAPER_PLOT_SPLIT", "held_out"),
                       "--output-dir", analysis_root / "paper_numbers")]
    if args.action == "supplement-tables":
        return [module("opera.run.render_supplement_tables",
                       "--config", path_value(env, "OPERA_PAPER_SUPPLEMENT_TABLE_CONFIG",
                                              str(analysis_root / "figure_inputs" / "supplement_tables.yaml")),
                       "--output-dir", analysis_root / "supplement_tables")
                + (["--execute"] if args.execute else [])]
    if args.action == "cell-metadata":
        return [module("opera.run.build_paper_cell_metadata",
                       "--registry",
                       path_value(env, "OPERA_RELEASE_DIR") / "experiment_registry_primary_v2.yaml",
                       "--outcomes-config", path_value(env, "OPERA_PAPER_BASE_CONFIG"),
                       "--output", path_value(env, "OPERA_PAPER_CELL_METADATA"))]
    if args.action == "patient-metadata":
        command = module(
            "opera.run.build_paper_patient_metadata",
            "--population", path_value(env, "BONSAI_COHORT_MEMBERSHIP"),
            "--subject-data-dir", path_value(env, "OPERA_PAPER_SUBJECT_DATA_DIR",
                                             value(env, "OPERA_SEQUENCE_DATA_DIR")),
            "--vocabulary", path_value(env, "OPERA_PAPER_LOCAL_VOCAB"),
            "--index-date-column", value(env, "OPERA_PAPER_INDEX_DATE_COLUMN", "index_date"),
            *(("--identity-date-column", env["OPERA_PAPER_IDENTITY_DATE_COLUMN"])
              if env.get("OPERA_PAPER_IDENTITY_DATE_COLUMN") else ()),
            "--output", path_value(env, "OPERA_PAPER_PATIENT_METADATA"))
        if env.get("OPERA_PAPER_DEMOGRAPHICS"):
            command += ["--demographics", str(path_value(env, "OPERA_PAPER_DEMOGRAPHICS"))]
        if env.get("OPERA_PAPER_CLINICAL_METADATA"):
            command += ["--clinical-metadata",
                        str(path_value(env, "OPERA_PAPER_CLINICAL_METADATA"))]
        if env.get("OPERA_PAPER_EXTERNAL_SOURCE_VOCAB"):
            command += ["--national-vocabulary",
                        str(path_value(env, "OPERA_PAPER_EXTERNAL_SOURCE_VOCAB"))]
            national = external_sequence_dir(env, root)
            if national.is_dir():  # after external-data: coverage in the national tokens
                command += ["--national-sequence-dir", str(national)]
        return [command]
    if args.action == "table1":
        # Supervised population with its splits, clinical metadata, overall
        # survival and pre-index MEDS history; small cells suppressed for release.
        outcomes = Path(value(env, "BONSAI_OUTCOMES_DIR"))
        survival = path_value(env, "OPERA_PAPER_OVERALL_SURVIVAL",
                              str(outcomes / "overall_survival.parquet"))
        if not env.get("OPERA_PAPER_DEMOGRAPHICS"):
            raise ValueError(
                "Table 1 needs sex and birth date from the DALY-CARE patient table. Run once:\n"
                f"  Rscript opera/R/export_table1_patient_demographics.R {survival} "
                "<new path>/table1_demographics.parquet\n"
                "then set OPERA_PAPER_DEMOGRAPHICS=<that path> in .env")
        command = module(
            "opera.run.export_table1",
            "--population", path_value(env, "BONSAI_COHORT_MEMBERSHIP"),
            # The membership file is diagnosis-indexed; Table 1 is the
            # supervised first-line cohort (index date and split), counted.
            "--supervised-only",
            "--metadata", path_value(env, "OPERA_PAPER_DEMOGRAPHICS"),
            *(("--metadata", path_value(env, "OPERA_PAPER_CLINICAL_METADATA"))
              if env.get("OPERA_PAPER_CLINICAL_METADATA") else ()),
            "--overall-survival", survival,
            "--meds-dir", path_value(env, "OPERA_PAPER_MEDS_DIR"),
            "--meds-code-prefixes", REPO / "opera/configs/table1_meds_prefixes.json",
            "--disclosure-policy", "targeted",
            "--output-dir", root / "table1")
        if args.inspect_inputs:
            return [command + ["--inspect-inputs"]]
        # Manuscript tables, rendered on the server from the exact export with
        # counts of 1-4 shown as <5 (no complementary suppression).
        manuscript = root / "table1" / "manuscript"
        return [command, module(
            "opera.run.format_table1_release",
            "--input", root / "table1" / "LOCAL_ONLY_internal" / "table1_values_long.csv",
            "--output-dir", manuscript,
            "--disclosure-policy", "primary-only",
            "--latex", manuscript / "table1.tex")]
    if args.action == "embedding-tables":
        return [module("opera.run.build_paper_embedding_tables",
                       "--config", path_value(env, "OPERA_PAPER_EMBEDDING_CONFIG"),
                       "--patient-metadata", path_value(env, "OPERA_PAPER_PATIENT_METADATA"),
                       "--output-dir", analysis_root / "figure_inputs" / "plot_tables")
                + (["--rehearse-on-tuning"]
                   if value(env, "OPERA_PAPER_PLOT_SPLIT", "held_out") == "tuning" else [])]
    if args.action == "plot-tables":
        command = module("opera.run.build_paper_plot_tables",
                         "--analysis-root", analysis_root,
                         "--cell-metadata", path_value(env, "OPERA_PAPER_CELL_METADATA"),
                         "--output-dir", analysis_root / "figure_inputs" / "plot_tables",
                         "--seed", value(env, "OPERA_PAPER_SEEDS", "42").split(",")[0],
                         "--split", value(env, "OPERA_PAPER_PLOT_SPLIT", "held_out"))
        if env.get("OPERA_PAPER_ATLAS_METRICS"):
            command += ["--atlas-metrics", str(path_value(env, "OPERA_PAPER_ATLAS_METRICS"))]
        if env.get("OPERA_PAPER_PATIENT_METADATA"):
            command += ["--patient-metadata",
                        str(path_value(env, "OPERA_PAPER_PATIENT_METADATA"))]
        return [command]
    if args.action == "init-figures":
        return [module("opera.run.prepare_paper_figure_inputs",
                       "--output-dir", analysis_root / "figure_inputs")
                + (["--execute"] if args.execute else [])]
    if args.action == "figures":
        # Recipe paths such as source_removal/<split>/ follow the plot split.
        env.setdefault("OPERA_PAPER_PLOT_SPLIT", "held_out")
        return [module("opera.run.render_paper_results",
                       "--config", path_value(env, "OPERA_PAPER_FIGURE_CONFIG",
                                              str(analysis_root / "figure_inputs" / "figures.yaml")),
                       "--output-dir", analysis_root / "figures")
                + (["--available-only"] if env.get("OPERA_PAPER_FIGURES_AVAILABLE_ONLY") else [])
                + (["--validate-render"] if getattr(args, "validate_render", False)
                   else (["--execute"] if args.execute else []))]
    if args.action == "figure-readiness":
        # Which manuscript figures can be rendered now, and what to run for the rest.
        return [module("opera.run.check_figure_inputs", "--analysis-root", analysis_root,
                       "--split", value(env, "OPERA_PAPER_PLOT_SPLIT", "held_out"))]
    if args.action == "input-restrictions":
        # The locked best model on restricted input: what each part adds.
        config_path = path_value(env, "OPERA_PAPER_ATTRIBUTION_CONFIG",
                                 str(REPO / "opera/configs/attribution_analysis.example.yaml"))
        config = yaml.safe_load(Path(config_path).read_text())
        out = analysis_root / "input_restrictions"
        base = module(
            "opera.run.evaluate_direct_competing_risk",
            "--checkpoint", path_value(env, "OPERA_PAPER_ATTRIBUTION_CHECKPOINT"),
            "--config-name", value(env, "OPERA_PAPER_ATTRIBUTION_CONFIG_NAME"),
            "--config-dir", path_value(env, "OPERA_PAPER_ATTRIBUTION_CONFIG_DIR",
                                       str(REPO / "opera/configs")),
            "--split", config.get("run", {}).get("split", "held_out"))
        def done(directory):  # restrictions already evaluated are reused
            return (directory / "direct_cr_predictions.parquet").is_file()

        commands = [] if done(out / "full") else [base + ["--output-dir", str(out / "full")]]
        compare = module("opera.run.compare_input_restrictions", "--full", out / "full",
                         "--output-dir", out)
        reference = analysis_root / "attributions" / "baseline_INTERNAL.pt"
        for item in config.get("restrictions", []):
            name, slug = item["label"], item["id"]
            flags = []
            if item.get("content_reference"):
                flags += ["--content-reference", str(reference)]
            if item.get("hide_prefixes"):
                flags += ["--hide-prefixes", ",".join(item["hide_prefixes"])]
            if item.get("keep_last_days") is not None:
                flags += ["--keep-last-days", str(item["keep_last_days"])]
            if item.get("drop_last_days") is not None:
                flags += ["--drop-last-days", str(item["drop_last_days"])]
            if not done(out / slug):
                commands.append(base + flags + ["--output-dir", str(out / slug)])
            compare += ["--restricted", f"{name}={out / slug}"]
        return commands + [compare]
    if args.action == "attribution-figures":
        config_path = path_value(env, "OPERA_PAPER_ATTRIBUTION_CONFIG",
                                 str(REPO / "opera/configs/attribution_analysis.example.yaml"))
        settings = yaml.safe_load(config_path.read_text())
        horizon = getattr(args, "attribution_horizon", None) or settings.get("summary", {}).get("horizon_days", 365)
        summary_dir = analysis_root / "attributions" / f"summary_{float(horizon):g}d"
        env["OPERA_PAPER_ATTRIBUTION_SUMMARY_DIR"] = str(summary_dir.resolve())
        env.setdefault("OPERA_PAPER_INPUT_RESTRICTION_DIR",
                       str((analysis_root / "input_restrictions").resolve()))
        return [module("opera.run.render_paper_results",
                       "--config", REPO / "opera/configs/attribution_figures.example.yaml",
                       "--output-dir", summary_dir / "figures")
                + (["--execute"] if args.execute else [])]
    if args.action == "vocabulary-audit":
        # What the tokens mean: names, groups and chapters for every code of the
        # locked model's vocabulary, and the unnamed codes that matter most.
        config_path = path_value(env, "OPERA_PAPER_ATTRIBUTION_CONFIG",
                                 str(REPO / "opera/configs/attribution_analysis.example.yaml"))
        horizon = yaml.safe_load(Path(config_path).read_text()).get("summary", {}).get(
            "horizon_days", 365)
        summary = analysis_root / "attributions" / f"summary_{float(horizon):g}d"
        return [module("opera.run.audit_vocabulary",
                       "--vocabulary", path_value(env, "OPERA_PAPER_LOCAL_VOCAB"),
                       "--config", config_path, "--output-dir", analysis_root / "vocabulary_audit")
                + (["--attribution-summary", summary]
                   if (summary / "attribution_top_concepts.csv").is_file() else [])]
    if args.action in {"attributions", "attribution-summary"}:
        # Integrated gradients on the locked best model: one pass over every
        # outcome, then aggregated tables (the only part that may be exported).
        config_path = path_value(env, "OPERA_PAPER_ATTRIBUTION_CONFIG",
                                 str(REPO / "opera/configs/attribution_analysis.example.yaml"))
        config = yaml.safe_load(Path(config_path).read_text())
        run_dir = analysis_root / "attributions"
        if args.action == "attribution-summary":
            horizon = getattr(args, "attribution_horizon", None) or config.get("summary", {}).get("horizon_days", 365)
            return [module("opera.run.summarise_attributions", "--input-dir", run_dir,
                           "--config", config_path, "--horizon-days", horizon,
                           "--output-dir", run_dir / f"summary_{float(horizon):g}d")]
        run = dict(config.get("run", {}))

        def listed(key, default=""):
            item = run.get(key, default)
            return ",".join(str(v) for v in item) if isinstance(item, list) else str(item)

        command = module(
            "opera.run.explain_direct_competing_risk",
            "--resume",
            "--checkpoint", path_value(env, "OPERA_PAPER_ATTRIBUTION_CHECKPOINT"),
            "--config-name", value(env, "OPERA_PAPER_ATTRIBUTION_CONFIG_NAME"),
            "--config-dir", path_value(env, "OPERA_PAPER_ATTRIBUTION_CONFIG_DIR", str(REPO / "opera/configs")),
            "--split", run.get("split", "held_out"),
            "--outcomes", listed("outcomes", "all"),
            "--horizons", listed("horizons", "365"),
            "--borrowed-outcomes", listed("borrowed_outcomes", ""),
            "--check-outcomes", listed("check_outcomes"),
            # "all" (default): every test patient; a number: stratified sample.
            "--max-patients", (0 if str(run.get("max_patients", "all")).lower() == "all"
                               else run.get("max_patients")),
            "--per-subtype", run.get("per_subtype", 40),
            "--steps", run.get("steps", 8),
            "--precision", run.get("precision", "fp32"),
            "--tolerance", run.get("tolerance", 0.05),
            "--baseline", run.get("baseline", "mean"),
            "--alternative-baseline", run.get("alternative_baseline", "source_mean"),
            "--k", listed("k", "1,5,10,20"),
            "--random-repeats", run.get("random_repeats", 5),
            "--faithfulness-patients", run.get("faithfulness_patients", 200),
            "--sanity-patients", run.get("sanity_patients", 100),
            # Tokens per forward/backward pass; halved automatically on CUDA
            # out of memory (results do not depend on it).
            "--token-budget", run.get("token_budget", 16384),
            "--output-dir", run_dir)
        if getattr(args, "attribution_max_new_patients", None):
            command += ["--max-new-patients", str(args.attribution_max_new_patients)]
        return [command]
    if args.action == "clinical-scores":
        return [module("opera.run.render_clinical_score_panels",
                       "--analysis-config", analysis_root / "analysis_inventory" / "analysis.yaml",
                       "--score-config", path_value(env, "OPERA_PAPER_CLINICAL_SCORE_CONFIG"),
                       "--output-dir", analysis_root / "clinical_scores",
                       "--split", value(env, "OPERA_PAPER_PLOT_SPLIT", "held_out"))
                + (["--execute"] if args.execute else [])]
    if args.action == "analyse":
        return [
            module(
                "opera.run.analyse_locked_predictions",
                "--config",
                analysis_root / "analysis_inventory" / "analysis.yaml",
                "--output-dir",
                analysis_root / "prediction_analysis",
            )
        ]
    if args.action == "bayes":
        if env.get("OPERA_PAPER_BAYES_RECIPES"):
            paths = json.loads(env["OPERA_PAPER_BAYES_RECIPES"])
        else:
            # Only primary recipes whose rosters validated; families without
            # their model arms (e.g. external) are listed as not estimable.
            manifest = analysis_root / "bayesian_recipes" / "recipe_manifest.json"
            if not manifest.is_file():
                raise FileNotFoundError(
                    f"Run prepare-bayes first; missing recipe manifest: {manifest}"
                )
            paths = json.loads(manifest.read_text())["primary_recipes"]
        if (
            not isinstance(paths, list)
            or not paths
            or not all(isinstance(p, str) for p in paths)
        ):
            raise ValueError(
                "OPERA_PAPER_BAYES_RECIPES must be a nonempty JSON list of recipe paths"
            )
        commands = []
        for raw in paths:
            path = Path(raw)
            path = path if path.is_absolute() else REPO / path
            command = module("opera.run.bayesian_contrasts", "--config", path)
            if args.prior_only:
                command += ["--prior-only"]
            commands.append(command)
        return commands
    if args.action == "check":
        return [
            module(
                "opera.run.paper_readiness", "--env-file", Path(args.env_file).resolve()
            )
        ]
    if args.action == "status":
        return [
            module(
                "opera.run.paper_prediction_queue",
                *shared,
                "--split",
                args.split,
                "--output-dir",
                root / "predictions",
                "--status",
            )
        ]
    if args.action == "compute-inventory":
        return [
            module(
                "opera.run.paper_compute_inventory",
                *[item for path in stage_bundles(root) for item in ("--bundle-dir", path)],
                "--runs-dir", runs,
                "--output-dir", root / "compute_inventory",
            )
        ]
    if args.action in {"worker", "queue-status"}:
        command = module(
            "opera.run.paper_worker",
            "next" if args.action == "worker" else "status",
            *[item for path in stage_bundles(root) for item in ("--bundle-dir", path)],
            "--runs-dir",
            runs,
            "--minimum-local-pretraining-epochs",
            value(env, "OPERA_PAPER_MIN_LOCAL_EPOCHS", "10"),
            "--minimum-dapt-epochs",
            value(env, "OPERA_PAPER_MIN_DAPT_EPOCHS", "10"),
            "--external-hidden-size",
            value(env, "OPERA_PAPER_EXTERNAL_WIDTH", "256"),
            "--local-hidden-size",
            value(env, "OPERA_PAPER_LOCAL_WIDTH", "256"),
        )
        if env.get("OPERA_PAPER_DECISIONS"):
            command += ["--decisions", str(path_value(env, "OPERA_PAPER_DECISIONS"))]
        if env.get("OPERA_PAPER_SPECULATIVE_SCREENS", "0") == "1":
            command += ["--speculative"]
        if args.action == "worker":
            if args.worker_name:
                command += ["--worker", args.worker_name]
            if args.wait:
                command += ["--wait"]
            if args.max_runs:
                command += ["--max-runs", str(args.max_runs)]
        return [command]
    if args.action == "stage1":
        commands = []
        for regime in ("J", "D", "O", "S"):
            command = module(
                "opera.run.run_paper_queue",
                *shared,
                "--phase",
                "supervision",
                "--regime",
                regime,
                "--minimum-local-pretraining-epochs",
                value(env, "OPERA_PAPER_MIN_LOCAL_EPOCHS", "10"),
            )
            if args.max_runs:
                command += ["--max-runs", str(args.max_runs)]
            commands.append(command + (["--execute"] if args.execute else []))
        commands.append(
            module("opera.run.run_paper_queue", *shared, "--phase", "source_screen",
                   "--minimum-local-pretraining-epochs",
                   value(env, "OPERA_PAPER_MIN_LOCAL_EPOCHS", "10"))
            + (["--execute"] if args.execute else [])
        )
        return commands
    raise ValueError(f"Unknown action: {args.action}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "action",
        choices=[
            "check",
            "prepare",
            "train",
            "stage1",
            "smoke",
            "tuning-summary",
            "prepare-bayes",
            "external-check",
            "external-data",
            "export-external",
            "dapt",
            "features",
            "feature-check",
            "tabular",
            "evaluate",
            "inventory",
            "analyse",
            "clinical-scores",
            "attributions",
            "attribution-summary",
            "attribution-figures",
            "vocabulary-audit",
            "input-restrictions",
            "figure-readiness",
            "figures",
            "init-figures",
            "cell-metadata",
            "patient-metadata",
            "table1",
            "embedding-tables",
            "plot-tables",
            "paper-numbers",
            "supplement-tables",
            "bayes",
            "status",
            "worker",
            "queue-status",
            "compute-inventory",
            "router-decision",
            "source-removal",
            "borrowing-map",
        ],
    )
    parser.add_argument("--env-file", type=Path, default=REPO / ".env")
    parser.add_argument("--stage", type=int, default=1,
                        help="Bundle stage: 1 local, 2 locked router and external, 3+ follow-ups")
    parser.add_argument("--phase", choices=PHASES, default="supervision")
    parser.add_argument("--regime", choices=("S", "D", "O", "J"))
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--max-runs", type=int)
    parser.add_argument("--split", choices=("tuning", "held_out"), default="tuning")
    parser.add_argument("--include-external", action="store_true")
    parser.add_argument("--export-routes", action="store_true")
    parser.add_argument("--only", action="append", default=[],
                        help="evaluate: only these phases, e.g. source_screen or supervision:J")
    parser.add_argument("--ablation", choices=("mean", "zero"), default="mean",
                        help="borrowing-map: mean ablation (primary) or zero (sensitivity)")
    parser.add_argument("--record", action="store_true",
                        help="router-decision: write the decision (default: report only)")
    parser.add_argument("--registry", action="store_true",
                        help="tabular: the registry-augmented XGBoost arm (XR).")
    parser.add_argument("--prior-only", action="store_true")
    parser.add_argument("--attribution-horizon", type=float, help="CPU attribution summary horizon")
    parser.add_argument("--attribution-max-new-patients", type=int, help="Reusable attribution pilot cap")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--validate-render", action="store_true",
                        help="figures: validate all render schemas without saving outputs")
    parser.add_argument("--inspect-inputs", action="store_true",
                        help="table1: print input coverage and split labels only")
    parser.add_argument("--worker-name", help="Label for this GPU worker's claims")
    parser.add_argument(
        "--wait",
        action="store_true",
        help="worker: keep polling while fits are blocked or running elsewhere",
    )
    args = parser.parse_args()
    if (
        args.shard_count < 1
        or not 0 <= args.shard_index < args.shard_count
        or (args.max_runs is not None and args.max_runs < 1)
    ):
        parser.error("Invalid shard count/index or max-runs")
    env = load_environment(args.env_file)
    if args.action in CPU_POOL_ACTIONS:
        # These steps run process pools; one BLAS thread per process keeps them
        # from oversubscribing a shared node (exported values still win).
        for name in BLAS_THREAD_VARIABLES:
            env.setdefault(name, "1")
    commands = build_commands(args, env)
    for command in commands:
        print(shlex.join(command), flush=True)
        if args.execute or args.action in {"check", "status", "queue-status", "feature-check"}:
            subprocess.run(command, cwd=REPO, env=env, check=True)
    if not args.execute and args.action not in {"check", "status", "queue-status", "feature-check"}:
        print("Preview only; use --execute to run this stage.")


if __name__ == "__main__":
    main()
