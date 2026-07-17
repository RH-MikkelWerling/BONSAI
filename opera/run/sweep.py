"""
OPERA Evaluation Sweep — runs finetune + evaluate across all
cohort × outcome × model_variant combinations.

This is the script that produces the main results table for the paper.

How it works
────────────
1. Reads a sweep config that defines:
   - Which cohorts to evaluate on
   - Which outcomes to evaluate on
   - Which model variants to compare (base, dapt, opera, tabular, ipi)
   - Checkpoint paths for each variant

2. For each (cohort, outcome, variant) triple:
   a. Runs finetuning (or skips if checkpoint already exists)
   b. Runs evaluation on the held-out test set
   c. Saves per-cell metrics.json

3. Aggregates all results into a single comparison DataFrame and saves
   as CSV + LaTeX table.

Usage
─────
python -m opera.run.sweep --config sweep_config.yaml [--dry-run] [--overwrite]

sweep_config.yaml structure
────────────────────────────
cohorts:
  dlbcl:
    data_dir: /data/dlbcl
    ipi_score_col: nccn_ipi    # column in population CSV, null if unavailable
  cll:
    data_dir: /data/cll
    ipi_score_col: cll_ipi

outcomes:
  - mortality_1y
  - treatment_failure
  - aki_30d

model_variants:
  base_pretrain:
    encoder_ckpt: /ckpts/pretrain/best.ckpt
    encoder_source: pretrain
  dapt:
    encoder_ckpt: /ckpts/dapt/best.ckpt
    encoder_source: dapt
  opera:
    encoder_ckpt: /ckpts/contrastive/best.ckpt
    encoder_source: contrastive
  # tabular predictions are evaluated through the shared metric pipeline
  tabular_rkkp:
    predictions_file: /results/tabular_rkkp_{cohort}_{outcome}_predictions.csv
  tabular_ehr:
    predictions_file: /results/tabular_ehr_{cohort}_{outcome}_predictions.csv

finetune_base_config: opera/configs/finetune.yaml
output_dir: /results/sweep
"""

import argparse
import json
import os
import subprocess
import time
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np
import pandas as pd

from opera.config_contracts import load_sweep_config, variant_applies_to_outcome
from opera.evaluation.cohorts import population_subject_ids
from opera.evaluation.cohort_flow import eligibility_file_path
from opera.evaluation.results_schema import build_result_row, write_result_artifacts
from opera.evaluation.tasks import normalize_outcome_config, outcome_file_path
from opera.functional.outcomes import (
    attach_prediction_censor_abspos,
    filter_outcome_eligibility,
    filter_registry_eligible_outcomes,
    resolve_registry_start_date,
)
from opera.run.sweep_commands import (
    build_evaluate_cmd,
    build_finetune_cmd,
    build_prediction_evaluate_cmd,
)
from opera.run.sweep_types import StatusTracker


def competing_outcome_path(data_dir: str, outcome_cfg: Dict) -> Optional[str]:
    """Resolve optional competing-event parquet from sweep outcome config."""
    if outcome_cfg.get("competing_outcome_path"):
        return os.path.expandvars(outcome_cfg["competing_outcome_path"])
    if outcome_cfg.get("competing_outcome_file"):
        raw = os.path.expandvars(outcome_cfg["competing_outcome_file"])
        path = Path(raw)
        return str(path if path.is_absolute() else Path(data_dir) / "outcomes" / path)
    return None


def format_variant_path(
    value: str,
    cohort: str,
    outcome: str,
    seed: int,
    training_cohort: Optional[str] = None,
) -> str:
    """Expand sweep placeholders in model artifact paths.

    Supported placeholders: ``{cohort}``, ``{outcome}``, ``{seed}``, and
    ``{training_cohort}`` (the grouped training cohort for train-on-grouped /
    eval-on-fine sweeps; falls back to ``cohort`` when not set).
    """
    return value.format(
        cohort=cohort,
        outcome=outcome,
        seed=seed,
        training_cohort=training_cohort if training_cohort is not None else cohort,
    )


def rank_normalize_scores(values: pd.Series) -> np.ndarray:
    """Map numeric or ordinal score values to rank-normalized [0, 1] scores."""
    numeric = pd.to_numeric(values, errors="coerce")
    if numeric.notna().all():
        ranks = numeric.rank(method="dense").to_numpy(float)
    else:
        ranks = pd.Series(pd.factorize(values.astype(str), sort=True)[0] + 1).to_numpy(
            float
        )
    denom = float(ranks.max() - ranks.min()) if len(ranks) else 0.0
    return np.full(len(ranks), 0.5) if denom == 0 else (ranks - ranks.min()) / denom


def run_logged_subprocess(
    cmd: list[str],
    log_dir: Path,
    step_name: str,
) -> subprocess.CompletedProcess:
    """Run a subprocess and persist command, stdout, stderr, and status.

    Inputs are the argv command, a per-cell log directory, and a short step
    label. The returned ``CompletedProcess`` mirrors ``subprocess.run``. The
    scientific workflow purpose is reproducibility: failed offline server runs
    leave enough context to diagnose environment, data, or CLI contract issues.
    """
    log_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()
    command_path = log_dir / f"{step_name}_command.json"
    stdout_path = log_dir / f"{step_name}_stdout.log"
    stderr_path = log_dir / f"{step_name}_stderr.log"
    status_path = log_dir / f"{step_name}_status.json"
    with open(command_path, "w") as f:
        json.dump({"cmd": cmd, "started_at": started}, f, indent=2)
    result = subprocess.run(cmd, capture_output=True, text=True)
    stdout_path.write_text(result.stdout or "")
    stderr_path.write_text(result.stderr or "")
    status = {
        "step": step_name,
        "cmd": cmd,
        "returncode": int(result.returncode),
        "started_at": started,
        "finished_at": time.time(),
        "duration_seconds": float(time.time() - started),
        "stdout_log": str(stdout_path),
        "stderr_log": str(stderr_path),
    }
    with open(status_path, "w") as f:
        json.dump(status, f, indent=2)
    return result


# ── IPI baseline ───────────────────────────────────────────────────────────


def compute_ipi_auroc(
    population_csv: str,
    outcome_parquet: str,
    ipi_score_col: str,
    split: str = "held_out",
    n_hours_start_include: int = 1,
    n_hours_end_include: Optional[int] = None,
    competing_outcome_parquet: Optional[str] = None,
    eligibility_path: Optional[str] = None,
    registry_start_date: Optional[str] = None,
    cohort: Optional[str] = None,
    outcome_name: Optional[str] = None,
) -> Optional[Dict]:
    """
    Compute binary and survival metrics for an IPI-style score.

    Binary metrics are computed on patients with full follow-up
    (require_min_followup=True). Survival metrics (C-index, IPCW-AUC)
    are computed on all patients via inverse probability of censoring
    weighting — matching the foundation model evaluator exactly.

    Returns None if the IPI column is missing or has insufficient coverage.
    """
    try:
        from sklearn.metrics import (
            roc_auc_score,
            average_precision_score,
            brier_score_loss,
        )
        from opera.compat.bonsai import binarize_outcomes
        from opera.evaluation.metrics import compute_survival_metrics

        population = pd.read_csv(population_csv)
        outcomes = pd.read_parquet(outcome_parquet)
        outcomes = filter_outcome_eligibility(
            outcomes,
            eligibility_path,
            cohort=cohort,
            outcome_name=outcome_name,
        )
        outcomes = attach_prediction_censor_abspos(outcomes)
        outcomes = filter_registry_eligible_outcomes(
            outcomes,
            registry_start_date,
            cohort=cohort,
            outcome_name=outcome_name,
        )
        test_df = outcomes[outcomes["split"] == split].copy()

        # Merge IPI score
        merged_all = test_df.merge(
            population[["subject_id", ipi_score_col]],
            on="subject_id",
            how="left",
        )
        ipi_coverage = float(merged_all[ipi_score_col].notna().mean())
        if ipi_coverage < 0.5:
            print(f"    IPI coverage {ipi_coverage:.0%} below 50%; skipping.")
            return None
        merged_all = merged_all.dropna(subset=[ipi_score_col])

        if len(merged_all) < 20:
            return None

        scores_norm = rank_normalize_scores(merged_all[ipi_score_col])

        competing_df = None
        if competing_outcome_parquet:
            competing_df = pd.read_parquet(competing_outcome_parquet)

        # ── All patients (for survival metrics) ────────────────────────
        all_outcomes = binarize_outcomes(
            merged_all,
            n_hours_start_include=n_hours_start_include,
            n_hours_end_include=n_hours_end_include,
            require_min_followup=False,
            competing_event_df=competing_df,
        )
        times_all = np.array(
            [
                all_outcomes[sid].get("time_days", float("nan"))
                for sid in merged_all["subject_id"]
            ]
        )
        events_all = np.array(
            [all_outcomes[sid].get("event", -1) for sid in merged_all["subject_id"]]
        )

        time_horizons = None
        if n_hours_end_include is not None:
            horizon_days = n_hours_end_include / 24.0
            defaults = [30.0, 90.0, 365.0, 730.0]
            time_horizons = sorted(
                set(h for h in defaults if h <= horizon_days) | {horizon_days}
            )

        survival_metrics = compute_survival_metrics(
            times_all, events_all, scores_norm, time_horizons=time_horizons
        )

        # ── Full-follow-up patients (for binary metrics) ────────────────
        full_fu = binarize_outcomes(
            merged_all,
            n_hours_start_include=n_hours_start_include,
            n_hours_end_include=n_hours_end_include,
            require_min_followup=True,
            competing_event_df=competing_df,
        )
        fu_sids = set(full_fu.keys())
        fu_mask = merged_all["subject_id"].isin(fu_sids).values
        labels_bin = np.array(
            [full_fu[sid]["label"] for sid in merged_all["subject_id"][fu_mask]]
        )
        scores_bin = scores_norm[fu_mask]

        binary_metrics: Dict = {}
        if len(labels_bin) >= 20 and len(np.unique(labels_bin)) == 2:
            binary_metrics = {
                "auroc": float(roc_auc_score(labels_bin, scores_bin)),
                "auprc": float(average_precision_score(labels_bin, scores_bin)),
                "brier_score": float(brier_score_loss(labels_bin, scores_bin)),
                "n_total": int(len(labels_bin)),
                "n_positive": int(labels_bin.sum()),
            }

        return {
            **binary_metrics,
            "coverage": ipi_coverage,
            "survival": survival_metrics,
        }
    except Exception as e:
        print(f"    IPI baseline failed: {e}")
        return None


def prepare_ipi_subset_predictions(
    population_csv: str,
    outcome_parquet: str,
    ipi_score_col: str,
    output_dir: Path,
    split: str = "held_out",
    eligibility_path: Optional[str] = None,
    registry_start_date: Optional[str] = None,
    cohort: Optional[str] = None,
    outcome_name: Optional[str] = None,
    cohort_fine_col: Optional[str] = None,
    cohort_fine_value: Optional[str] = None,
) -> tuple[Optional[Path], Optional[float], set]:
    """
    Write rank-normalized IPI predictions for IPI-complete test patients.

    Returns the prediction CSV path, IPI coverage among test-set patients, and
    the subject-id set defining the IPI-complete subset. This keeps the clinical
    score comparison restricted to identical patients.
    """
    population = pd.read_csv(population_csv)
    allowed_subject_ids = population_subject_ids(
        population_csv,
        cohort_fine_col=cohort_fine_col,
        cohort_fine_value=cohort_fine_value,
    )
    if allowed_subject_ids is not None:
        population = population[population["subject_id"].isin(allowed_subject_ids)].copy()
    outcomes = pd.read_parquet(outcome_parquet)
    outcomes = filter_outcome_eligibility(
        outcomes,
        eligibility_path,
        cohort=cohort,
        outcome_name=outcome_name,
    )
    outcomes = filter_registry_eligible_outcomes(
        attach_prediction_censor_abspos(outcomes),
        registry_start_date,
        cohort=cohort,
        outcome_name=outcome_name,
    )
    test_df = outcomes[outcomes["split"] == split][["subject_id"]].copy()
    if allowed_subject_ids is not None:
        test_df = test_df[test_df["subject_id"].isin(allowed_subject_ids)].copy()
    if test_df.empty or ipi_score_col not in population.columns:
        return None, None, set()
    merged = test_df.merge(
        population[["subject_id", ipi_score_col]],
        on="subject_id",
        how="left",
    )
    coverage = float(merged[ipi_score_col].notna().mean())
    complete = merged.dropna(subset=[ipi_score_col]).copy()
    if coverage < 0.5:
        print(f"  IPI coverage {coverage:.0%} below 50%; skipping IPI-complete rows.")
        return None, coverage, set(complete["subject_id"])
    if complete.empty:
        return None, coverage, set()
    complete["probability"] = rank_normalize_scores(complete[ipi_score_col])
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "ipi_subset_predictions.csv"
    complete[["subject_id", "probability"]].to_csv(path, index=False)
    complete[["subject_id"]].to_csv(
        output_dir / "ipi_subset_subjects.csv",
        index=False,
    )
    return path, coverage, set(complete["subject_id"])


def write_prediction_subset(
    predictions_path: str,
    subject_ids: set,
    output_path: Path,
    probability_col: str = "probability",
) -> Optional[Path]:
    """
    Restrict an existing prediction CSV/parquet to IPI-complete patients.

    The output uses the standard `subject_id`, `probability` columns so it can
    flow through the identical evaluation pipeline as full-cohort predictions.
    """
    source = Path(predictions_path)
    if not source.exists() or not subject_ids:
        return None
    pred = (
        pd.read_parquet(source)
        if source.suffix.lower() in {".parquet", ".pq"}
        else pd.read_csv(source)
    )
    if "subject_id" not in pred.columns or probability_col not in pred.columns:
        raise ValueError(f"Prediction file {predictions_path} lacks required columns.")
    subset = pred[pred["subject_id"].isin(subject_ids)].copy()
    subset = subset.rename(columns={probability_col: "probability"})
    output_path.parent.mkdir(parents=True, exist_ok=True)
    subset[["subject_id", "probability"]].to_csv(output_path, index=False)
    return output_path


def write_npz_prediction_subset(
    npz_path: Path, subject_ids: set, output_path: Path
) -> Optional[Path]:
    """
    Convert evaluator NPZ predictions to an IPI-complete prediction CSV.

    This is used for OPERA-style neural variants after their full-cohort
    evaluation has produced patient-level predictions.
    """
    if not npz_path.exists() or not subject_ids:
        return None
    data = np.load(npz_path, allow_pickle=True)
    frame = pd.DataFrame(
        {
            "subject_id": data["subject_ids"],
            "probability": data["probabilities"],
        }
    )
    subset = frame[frame["subject_id"].isin(subject_ids)].copy()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    subset.to_csv(output_path, index=False)
    return output_path


# ── Per-cell finetune + evaluate ───────────────────────────────────────────


def run_finetune(
    encoder_ckpt: str,
    encoder_source: str,
    cohort: str,
    cohort_data_dir: str,
    outcome_name: str,
    outcome_path: str,
    output_dir: Path,
    base_config: str,
    n_hours_start_include: int = 1,
    n_hours_end_include: Optional[int] = None,
    competing_outcome_path: Optional[str] = None,
    eligibility_path: Optional[str] = None,
    registry_start_date: Optional[str] = None,
    training_mode: Optional[str] = None,
    population_path: Optional[str] = None,
    cohort_fine_col: Optional[str] = None,
    cohort_fine_value: Optional[str] = None,
    extra_overrides: Optional[List[str]] = None,
    log_dir: Optional[Path] = None,
) -> Optional[Path]:
    """
    Launch a single finetune run via subprocess.

    Returns the checkpoint directory path.
    """
    cmd = build_finetune_cmd(
        encoder_ckpt=encoder_ckpt,
        encoder_source=encoder_source,
        cohort=cohort,
        cohort_data_dir=cohort_data_dir,
        outcome_name=outcome_name,
        outcome_path=outcome_path,
        output_dir=output_dir,
        base_config=base_config,
        n_hours_start_include=n_hours_start_include,
        n_hours_end_include=n_hours_end_include,
        competing_outcome_path=competing_outcome_path,
        eligibility_path=eligibility_path,
        registry_start_date=registry_start_date,
        training_mode=training_mode,
        population_path=population_path,
        cohort_fine_col=cohort_fine_col,
        cohort_fine_value=cohort_fine_value,
        extra_overrides=extra_overrides,
    )

    print(f"    Running finetune: {' '.join(cmd[4:8])} ...")
    result = run_logged_subprocess(cmd, log_dir or output_dir / "logs", "finetune")

    if result.returncode != 0:
        print(
            f"    FINETUNE FAILED (logs: {(log_dir or output_dir / 'logs')})\n{result.stderr[-2000:]}"
        )
        return None

    return output_dir / "best.ckpt"


def run_evaluate(
    ckpt_path: Path,
    cohort: str,
    cohort_data_dir: str,
    outcome_name: str,
    outcome_path: str,
    output_dir: Path,
    encoder_source: str = "contrastive",
    n_hours_start_include: int = 1,
    n_hours_end_include: Optional[int] = None,
    competing_outcome_path: Optional[str] = None,
    eligibility_path: Optional[str] = None,
    registry_start_date: Optional[str] = None,
    model_family: Optional[str] = None,
    training_stage: str = "evaluation",
    encoder_frozen: Optional[bool] = None,
    head_type: Optional[str] = None,
    cohort_fine_col: Optional[str] = None,
    cohort_fine_value: Optional[str] = None,
    log_dir: Optional[Path] = None,
) -> Optional[Dict]:
    """
    Launch evaluate.py and return parsed metrics dict.
    """
    cmd = build_evaluate_cmd(
        ckpt_path=ckpt_path,
        cohort=cohort,
        cohort_data_dir=cohort_data_dir,
        outcome_name=outcome_name,
        outcome_path=outcome_path,
        output_dir=output_dir,
        encoder_source=encoder_source,
        n_hours_start_include=n_hours_start_include,
        n_hours_end_include=n_hours_end_include,
        competing_outcome_path=competing_outcome_path,
        eligibility_path=eligibility_path,
        registry_start_date=registry_start_date,
        model_family=model_family,
        training_stage=training_stage,
        encoder_frozen=encoder_frozen,
        head_type=head_type,
        cohort_fine_col=cohort_fine_col,
        cohort_fine_value=cohort_fine_value,
    )
    result = run_logged_subprocess(cmd, log_dir or output_dir / "logs", "evaluate")

    if result.returncode != 0:
        print(
            f"    EVALUATE FAILED (logs: {(log_dir or output_dir / 'logs')})\n{result.stderr[-1000:]}"
        )
        return None

    metrics_path = output_dir / "metrics.json"
    if not metrics_path.exists():
        return None

    with open(metrics_path, encoding="utf-8") as f:
        return json.load(f)


def run_prediction_evaluate(
    predictions_path: str,
    cohort: str,
    outcome_name: str,
    outcome_path: str,
    model_family: str,
    output_dir: Path,
    n_hours_start_include: int,
    n_hours_end_include,
    competing_outcome_path: Optional[str] = None,
    eligibility_path: Optional[str] = None,
    registry_start_date: Optional[str] = None,
    rarity_mode: str = "none",
    baseline_model: Optional[str] = None,
    ipi_coverage: Optional[float] = None,
    evaluation_subset: str = "full",
    seed: int = 42,
    subgroup_path: Optional[str] = None,
    subgroup_columns: Optional[List[str]] = None,
    log_dir: Optional[Path] = None,
    population_path: Optional[str] = None,
    cohort_fine_col: Optional[str] = None,
    cohort_fine_value: Optional[str] = None,
    evaluation_subjects_path: Optional[str] = None,
    evaluation_regime: str = "both",
    probability_col: str = "probability",
    risk_col: Optional[str] = None,
) -> Optional[Dict]:
    cmd = build_prediction_evaluate_cmd(
        predictions_path=predictions_path,
        cohort=cohort,
        outcome_name=outcome_name,
        outcome_path=outcome_path,
        model_family=model_family,
        output_dir=output_dir,
        n_hours_start_include=n_hours_start_include,
        n_hours_end_include=n_hours_end_include,
        competing_outcome_path=competing_outcome_path,
        eligibility_path=eligibility_path,
        registry_start_date=registry_start_date,
        rarity_mode=rarity_mode,
        baseline_model=baseline_model,
        ipi_coverage=ipi_coverage,
        evaluation_subset=evaluation_subset,
        seed=seed,
        subgroup_path=subgroup_path,
        subgroup_columns=subgroup_columns,
        population_path=population_path,
        cohort_fine_col=cohort_fine_col,
        cohort_fine_value=cohort_fine_value,
        evaluation_subjects_path=evaluation_subjects_path,
        evaluation_regime=evaluation_regime,
        probability_col=probability_col,
        risk_col=risk_col,
    )
    result = run_logged_subprocess(
        cmd,
        log_dir or output_dir / "logs",
        f"evaluate_predictions_{evaluation_subset}",
    )
    if result.returncode != 0:
        print(
            f"    PREDICTION EVALUATE FAILED (logs: {(log_dir or output_dir / 'logs')})\n{result.stderr[-1000:]}"
        )
        return None
    metrics_path = output_dir / "metrics.json"
    if metrics_path.exists():
        with open(metrics_path, encoding="utf-8") as f:
            return json.load(f)
    return None


def _cfg_for_result_row(
    cohort: str,
    outcome: str,
    outcome_cfg: Dict,
    variant: str,
    cfg: Dict,
    checkpoint_path: str,
) -> Dict:
    rarity_cfg = cfg.get("rarity", {})
    row_cfg = {
        "model_family": variant,
        "training_stage": cfg.get("training_stage", "evaluation"),
        "cohort": cohort,
        "outcome": outcome,
        "seed": cfg.get("seed"),
        "ipi_coverage": cfg.get("ipi_coverage"),
        "evaluation_subset": cfg.get("evaluation_subset", "full"),
        "training_fraction": cfg.get("training_fraction"),
        "checkpoint_path": checkpoint_path,
        "labels": {
            "n_hours_end_include": outcome_cfg.get("n_hours_end_include"),
        },
        "rarity": {
            "mode": cfg.get("rarity_mode", rarity_cfg.get("mode", "none")),
            "tier": cfg.get("rarity_tier", rarity_cfg.get("tier")),
            "baseline_model": cfg.get(
                "baseline_model",
                rarity_cfg.get("baseline_model"),
            ),
            "size_metadata": rarity_cfg.get("size_metadata", {}),
        },
    }
    if cfg.get("run_id") is not None:
        row_cfg["run_id"] = cfg.get("run_id")
    return row_cfg


def write_sweep_result_artifact(
    metrics: Dict,
    output_dir: Path,
    cohort: str,
    outcome: str,
    outcome_cfg: Dict,
    variant: str,
    cfg: Dict,
    checkpoint_path: str,
) -> None:
    cfg = dict(cfg)
    if variant == "ipi":
        cfg["ipi_coverage"] = metrics.get("discrimination", {}).get("coverage")
        cfg["evaluation_subset"] = "ipi_complete"
    else:
        cfg.setdefault("evaluation_subset", "full")
    row = build_result_row(
        _cfg_for_result_row(
            cohort=cohort,
            outcome=outcome,
            outcome_cfg=outcome_cfg,
            variant=variant,
            cfg=cfg,
            checkpoint_path=checkpoint_path,
        ),
        metrics,
        checkpoint_path=checkpoint_path,
        split=cfg.get("test_key", "held_out"),
        model_family=variant,
    )
    write_result_artifacts(row, output_dir)


# ── Results aggregation ────────────────────────────────────────────────────


def flatten_metrics(metrics: Dict, prefix: str = "") -> Dict:
    """Flatten nested metrics dict to scalar values for the results table."""
    flat = {}
    disc = metrics.get("discrimination", {})
    cal = metrics.get("calibration", {})
    ci = metrics.get("bootstrap_ci", {})

    for key in (
        "auroc",
        "auprc",
        "sensitivity",
        "specificity",
        "f1",
        "n_total",
        "n_positive",
        "prevalence",
    ):
        if key in disc:
            flat[f"{prefix}{key}"] = disc[key]

    for key in ("brier_score", "ece"):
        if key in cal:
            flat[f"{prefix}{key}"] = cal[key]

    # Bootstrap CIs for key metrics
    for metric in ("auroc", "auprc"):
        if metric in ci:
            flat[f"{prefix}{metric}_lower"] = ci[metric]["lower"]
            flat[f"{prefix}{metric}_upper"] = ci[metric]["upper"]

    # Survival metrics
    sv = metrics.get("survival", {})
    if sv:
        flat[f"{prefix}concordance_index"] = sv.get("concordance_index", float("nan"))
        flat[f"{prefix}n_total_survival"] = sv.get("n_total", float("nan"))
        for label, hmet in sv.get("per_horizon", {}).items():
            flat[f"{prefix}ipcw_auc_{label}"] = hmet.get("ipcw_auc", float("nan"))
            flat[f"{prefix}ipcw_brier_{label}"] = hmet.get("ipcw_brier", float("nan"))

    # Survival bootstrap CIs
    sv_ci = metrics.get("survival_bootstrap_ci", {})
    if "concordance_index" in sv_ci:
        flat[f"{prefix}concordance_index_lower"] = sv_ci["concordance_index"].get(
            "lower", float("nan")
        )
        flat[f"{prefix}concordance_index_upper"] = sv_ci["concordance_index"].get(
            "upper", float("nan")
        )

    return flat


def build_results_table(all_results: List[Dict]) -> pd.DataFrame:
    """
    Assemble the main paper table from collected results.

    Rows: (cohort, outcome)
    Columns: metric × model_variant
    """
    rows = []
    for r in all_results:
        row = {
            "cohort": r["cohort"],
            "outcome": r["outcome"],
            "variant": r["variant"],
        }
        row.update(flatten_metrics(r.get("metrics", {})))
        rows.append(row)

    df = pd.DataFrame(rows)

    # Pivot to wide format: one row per (cohort, outcome), one col per (variant, metric)
    if df.empty:
        return df

    wide = df.pivot_table(
        index=["cohort", "outcome"],
        columns="variant",
        values=[c for c in df.columns if c not in ("cohort", "outcome", "variant")],
        aggfunc="first",
    )
    wide.columns = [f"{col[1]}__{col[0]}" for col in wide.columns]
    wide = wide.reset_index()

    return wide


def to_latex_table(df: pd.DataFrame, metric: str = "auroc") -> str:
    """
    Generate a LaTeX table showing `metric` for all cohort × outcome × variant.
    """
    variant_cols = sorted(
        set(
            col.split("__")[0]
            for col in df.columns
            if "__" in col and col.endswith(f"__{metric}")
        )
    )

    lines = [
        "\\begin{table}[ht]",
        "\\centering",
        "\\caption{AUROC by cohort, outcome, and model variant}",
        "\\begin{tabular}{ll" + "r" * len(variant_cols) + "}",
        "\\toprule",
        "Cohort & Outcome & "
        + " & ".join(v.replace("_", "\\_") for v in variant_cols)
        + " \\\\",
        "\\midrule",
    ]

    for _, row in df.iterrows():
        vals = []
        best_val = -1
        best_idx = -1
        cell_vals = []
        for i, v in enumerate(variant_cols):
            col = f"{v}__{metric}"
            val = row.get(col, float("nan"))
            cell_vals.append(val)
            if not np.isnan(val) and val > best_val:
                best_val = val
                best_idx = i

        for i, val in enumerate(cell_vals):
            if np.isnan(val):
                vals.append("--")
            elif i == best_idx:
                vals.append(f"\\textbf{{{val:.3f}}}")
            else:
                vals.append(f"{val:.3f}")

        cohort = str(row["cohort"]).upper()
        outcome = str(row["outcome"]).replace("_", "\\_")
        lines.append(f"{cohort} & {outcome} & " + " & ".join(vals) + " \\\\")

    lines += [
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}",
    ]
    return "\n".join(lines)


# ── Per-cell orchestration ─────────────────────────────────────────────────


def _run_ipi_baseline_cell(
    cohort_name: str,
    outcome_name: str,
    outcome_cfg: Dict,
    ipi_col: Optional[str],
    pop_file: str,
    data_dir: str,
    output_dir: Path,
    seeds: List[int],
    tracker: "StatusTracker",
    cfg: Dict,
    dry_run: bool,
    fail_fast: bool,
    rarity_mode: str,
    baseline_model: Optional[str],
    subgroup_path: Optional[str],
    subgroup_columns: Optional[List[str]],
    cohort_fine_col: Optional[str] = None,
    cohort_fine_value: Optional[str] = None,
) -> List[Dict]:
    """Evaluate the IPI clinical-score baseline for one cohort × outcome cell.

    Returns a list of result dicts (one per seed that produced metrics).
    Appends status records to tracker. Raises RuntimeError / FileNotFoundError
    only when fail_fast=True.
    """
    results: List[Dict] = []
    registry_start_date = resolve_registry_start_date(cfg, outcome_cfg)
    outcome_parquet = outcome_file_path(data_dir, outcome_name, outcome_cfg)
    eligibility = eligibility_file_path(
        data_dir,
        cohort_name,
        outcome_name,
        outcome_cfg,
    )
    eligibility = str(eligibility) if eligibility is not None else None
    if dry_run:
        print(f"  [DRY RUN] Would evaluate IPI for {cohort_name}/{outcome_name}")
        tracker.append(
            cohort=cohort_name,
            outcome=outcome_name,
            variant="ipi",
            seed=None,
            stage="ipi",
            status="dry_run",
            output_dir=str(output_dir / cohort_name / outcome_name / "ipi"),
        )
        return results
    if ipi_col and Path(outcome_parquet).exists():
        competing_parquet = competing_outcome_path(data_dir, outcome_cfg)
        subset_path, ipi_coverage, _ = prepare_ipi_subset_predictions(
            pop_file,
            outcome_parquet,
            ipi_col,
            output_dir / cohort_name / outcome_name / "ipi",
            eligibility_path=eligibility,
            registry_start_date=registry_start_date,
            cohort=cohort_name,
            outcome_name=outcome_name,
            cohort_fine_col=cohort_fine_col,
            cohort_fine_value=cohort_fine_value,
        )
        if subset_path is not None:
            for seed in seeds:
                metrics = run_prediction_evaluate(
                    predictions_path=str(subset_path),
                    cohort=cohort_name,
                    outcome_name=outcome_name,
                    outcome_path=outcome_parquet,
                    model_family="ipi",
                    output_dir=output_dir
                    / cohort_name
                    / outcome_name
                    / "ipi"
                    / f"seed_{seed}",
                    n_hours_start_include=outcome_cfg.get("n_hours_start_include", 1),
                    n_hours_end_include=outcome_cfg.get("n_hours_end_include"),
                    competing_outcome_path=competing_parquet,
                    eligibility_path=eligibility,
                    registry_start_date=registry_start_date,
                    rarity_mode=rarity_mode,
                    baseline_model=baseline_model,
                    ipi_coverage=ipi_coverage,
                    evaluation_subset="ipi_complete",
                    seed=seed,
                    subgroup_path=subgroup_path,
                    subgroup_columns=subgroup_columns,
                    log_dir=output_dir
                    / cohort_name
                    / outcome_name
                    / "ipi"
                    / f"seed_{seed}"
                    / "logs",
                    population_path=pop_file,
                    cohort_fine_col=cohort_fine_col,
                    cohort_fine_value=cohort_fine_value,
                    evaluation_subjects_path=str(
                        subset_path.parent / "ipi_subset_subjects.csv"
                    ),
                )
                if metrics:
                    results.append(
                        {
                            "cohort": cohort_name,
                            "outcome": outcome_name,
                            "variant": "ipi",
                            "metrics": metrics,
                        }
                    )
                    tracker.append(
                        cohort=cohort_name,
                        outcome=outcome_name,
                        variant="ipi",
                        seed=seed,
                        stage="evaluate_predictions",
                        status="success",
                        evaluation_subset="ipi_complete",
                        output_dir=str(
                            output_dir
                            / cohort_name
                            / outcome_name
                            / "ipi"
                            / f"seed_{seed}"
                        ),
                    )
                else:
                    tracker.append(
                        cohort=cohort_name,
                        outcome=outcome_name,
                        variant="ipi",
                        seed=seed,
                        stage="evaluate_predictions",
                        status="failed",
                        evaluation_subset="ipi_complete",
                        output_dir=str(
                            output_dir
                            / cohort_name
                            / outcome_name
                            / "ipi"
                            / f"seed_{seed}"
                        ),
                        reason="evaluate_predictions returned no metrics",
                    )
                    if fail_fast:
                        raise RuntimeError("IPI prediction evaluation failed")
            print(
                f"  IPI [{cohort_name} x {outcome_name}]: "
                f"coverage={ipi_coverage:.0%}, evaluated IPI-complete subset"
            )
            return results
        tracker.append(
            cohort=cohort_name,
            outcome=outcome_name,
            variant="ipi",
            seed=None,
            stage="coverage",
            status="skipped",
            output_dir=str(output_dir / cohort_name / outcome_name / "ipi"),
            reason="IPI coverage below threshold or no complete subset",
        )
        return results
    return results


def _run_variant_cell(
    cohort_name: str,
    outcome_name: str,
    outcome_cfg: Dict,
    variant_name: str,
    variant_cfg: Dict,
    result_variant: str,
    seed: int,
    training_cohort: str,
    cohort_fine_col: Optional[str],
    cohort_fine_value: Optional[str],
    data_dir: str,
    output_dir: Path,
    base_config: str,
    tracker: "StatusTracker",
    cfg: Dict,
    dry_run: bool,
    overwrite: bool,
    fail_fast: bool,
    rarity_mode: str,
    baseline_model: Optional[str],
    ipi_col: Optional[str],
    pop_file: str,
    subgroup_path: Optional[str],
    subgroup_columns: Optional[List[str]],
) -> Optional[Dict]:
    """Run one (cohort, outcome, variant, seed) cell of the evaluation sweep.

    Handles the three variant paths: results_file, predictions_file, and
    foundation-model finetune+evaluate. Returns a result dict or None.
    Appends status records to tracker.
    """
    registry_start_date = resolve_registry_start_date(cfg, outcome_cfg)
    n_hours_start = outcome_cfg.get("n_hours_start_include", 1)
    n_hours_end = outcome_cfg.get("n_hours_end_include")
    competing_outcome = competing_outcome_path(data_dir, outcome_cfg)
    eligibility = eligibility_file_path(
        data_dir,
        cohort_name,
        outcome_name,
        outcome_cfg,
    )
    eligibility = str(eligibility) if eligibility is not None else None

    cell_dir = output_dir / cohort_name / outcome_name / result_variant / f"seed_{seed}"
    cell_dir.mkdir(parents=True, exist_ok=True)

    # ── Pre-computed results (tabular baselines) ───────────
    if "results_file" in variant_cfg:
        results_path = format_variant_path(
            variant_cfg["results_file"],
            cohort_name,
            outcome_name,
            seed,
            training_cohort=training_cohort,
        )
        print(
            "  Warning: results_file bypasses shared prediction "
            "evaluation; calibration, DCA, bootstrap CI, and "
            "enrichment fields may be missing."
        )
        if dry_run:
            print(f"  [DRY RUN] Would load pre-computed results: {results_path}")
            tracker.append(
                cohort=cohort_name,
                outcome=outcome_name,
                variant=result_variant,
                seed=seed,
                stage="results_file",
                status="dry_run",
                output_dir=str(cell_dir),
            )
            return None
        if Path(results_path).exists():
            with open(results_path) as f:
                metrics = json.load(f)
            write_sweep_result_artifact(
                metrics=metrics,
                output_dir=cell_dir,
                cohort=cohort_name,
                outcome=outcome_name,
                outcome_cfg=outcome_cfg,
                variant=result_variant,
                cfg={**cfg, "seed": seed},
                checkpoint_path=results_path,
            )
            auroc = metrics.get("discrimination", {}).get("auroc", float("nan"))
            print(f"  Loaded pre-computed: AUROC={auroc:.3f}")
            tracker.append(
                cohort=cohort_name,
                outcome=outcome_name,
                variant=result_variant,
                seed=seed,
                stage="results_file",
                status="success",
                output_dir=str(cell_dir),
                artifact=results_path,
            )
            return {
                "cohort": cohort_name,
                "outcome": outcome_name,
                "variant": result_variant,
                "metrics": metrics,
            }
        print(f"  Pre-computed results not found: {results_path}")
        tracker.append(
            cohort=cohort_name,
            outcome=outcome_name,
            variant=result_variant,
            seed=seed,
            stage="results_file",
            status="failed",
            output_dir=str(cell_dir),
            artifact=results_path,
            reason="results_file not found",
        )
        if fail_fast:
            raise FileNotFoundError(results_path)
        return None

    if "predictions_file" in variant_cfg:
        predictions_path = format_variant_path(
            variant_cfg["predictions_file"],
            cohort_name,
            outcome_name,
            seed,
            training_cohort=training_cohort,
        )
        if dry_run:
            print(f"  [DRY RUN] Would evaluate prediction file: {predictions_path}")
            tracker.append(
                cohort=cohort_name,
                outcome=outcome_name,
                variant=result_variant,
                seed=seed,
                stage="evaluate_predictions",
                status="dry_run",
                output_dir=str(cell_dir),
                artifact=predictions_path,
            )
            return None
        if Path(predictions_path).exists():
            metrics = run_prediction_evaluate(
                predictions_path=predictions_path,
                cohort=cohort_name,
                outcome_name=outcome_name,
                outcome_path=outcome_file_path(
                    data_dir,
                    outcome_name,
                    outcome_cfg,
                ),
                model_family=result_variant,
                output_dir=cell_dir,
                n_hours_start_include=n_hours_start,
                n_hours_end_include=n_hours_end,
                competing_outcome_path=competing_outcome,
                eligibility_path=eligibility,
                registry_start_date=registry_start_date,
                rarity_mode=rarity_mode,
                baseline_model=baseline_model,
                seed=seed,
                subgroup_path=subgroup_path,
                subgroup_columns=subgroup_columns,
                log_dir=cell_dir / "logs",
                population_path=pop_file,
                cohort_fine_col=cohort_fine_col,
                cohort_fine_value=cohort_fine_value,
                evaluation_regime=variant_cfg.get("evaluation_regime", "both"),
                probability_col=variant_cfg.get("probability_col", "probability"),
                risk_col=variant_cfg.get("risk_col"),
            )
            if metrics is not None:
                if "discrimination" in metrics:
                    auroc = metrics["discrimination"].get("auroc", float("nan"))
                    print(f"  Evaluated predictions: AUROC={auroc:.3f}")
                else:
                    c_index = metrics.get("survival", {}).get(
                        "concordance_index", float("nan")
                    )
                    print(f"  Evaluated predictions: C-index={c_index:.3f}")
                tracker.append(
                    cohort=cohort_name,
                    outcome=outcome_name,
                    variant=result_variant,
                    seed=seed,
                    stage="evaluate_predictions",
                    status="success",
                    evaluation_subset="full",
                    output_dir=str(cell_dir),
                    artifact=predictions_path,
                )
                if ipi_col and result_variant in {"tabular_ehr", "opera"}:
                    ipi_path, ipi_coverage, ipi_subjects = (
                        prepare_ipi_subset_predictions(
                            pop_file,
                            outcome_file_path(data_dir, outcome_name, outcome_cfg),
                            ipi_col,
                            output_dir / cohort_name / outcome_name / "ipi",
                            eligibility_path=eligibility,
                            registry_start_date=registry_start_date,
                            cohort=cohort_name,
                            outcome_name=outcome_name,
                            cohort_fine_col=cohort_fine_col,
                            cohort_fine_value=cohort_fine_value,
                        )
                    )
                    if ipi_path is not None:
                        subset_pred = write_prediction_subset(
                            predictions_path,
                            ipi_subjects,
                            cell_dir / f"{result_variant}_ipi_subset_predictions.csv",
                        )
                        if subset_pred is not None:
                            run_prediction_evaluate(
                                predictions_path=str(subset_pred),
                                cohort=cohort_name,
                                outcome_name=outcome_name,
                                outcome_path=outcome_file_path(
                                    data_dir, outcome_name, outcome_cfg
                                ),
                                model_family=result_variant,
                                output_dir=cell_dir / "ipi_complete",
                                n_hours_start_include=n_hours_start,
                                n_hours_end_include=n_hours_end,
                                competing_outcome_path=competing_outcome,
                                eligibility_path=eligibility,
                                registry_start_date=registry_start_date,
                                rarity_mode=rarity_mode,
                                baseline_model=baseline_model,
                                ipi_coverage=ipi_coverage,
                                evaluation_subset="ipi_complete",
                                seed=seed,
                                subgroup_path=subgroup_path,
                                subgroup_columns=subgroup_columns,
                                log_dir=cell_dir / "ipi_complete" / "logs",
                                population_path=pop_file,
                                cohort_fine_col=cohort_fine_col,
                                cohort_fine_value=cohort_fine_value,
                                evaluation_subjects_path=str(
                                    ipi_path.parent / "ipi_subset_subjects.csv"
                                ),
                            )
                return {
                    "cohort": cohort_name,
                    "outcome": outcome_name,
                    "variant": result_variant,
                    "metrics": metrics,
                }
            tracker.append(
                cohort=cohort_name,
                outcome=outcome_name,
                variant=result_variant,
                seed=seed,
                stage="evaluate_predictions",
                status="failed",
                evaluation_subset="full",
                output_dir=str(cell_dir),
                artifact=predictions_path,
                reason="evaluate_predictions returned no metrics",
            )
            if fail_fast:
                raise RuntimeError(f"Prediction evaluation failed: {predictions_path}")
            return None
        print(f"  Prediction file not found: {predictions_path}")
        tracker.append(
            cohort=cohort_name,
            outcome=outcome_name,
            variant=result_variant,
            seed=seed,
            stage="evaluate_predictions",
            status="failed",
            output_dir=str(cell_dir),
            artifact=predictions_path,
            reason="predictions_file not found",
        )
        if fail_fast:
            raise FileNotFoundError(predictions_path)
        return None

    # ── Foundation model: finetune + evaluate ──────────────
    metrics_path = cell_dir / "metrics.json"
    if metrics_path.exists() and not overwrite:
        print("  Already done (use --overwrite to redo).")
        with open(metrics_path) as f:
            metrics = json.load(f)
        auroc = metrics.get("discrimination", {}).get("auroc", float("nan"))
        print(f"  Loaded cached: AUROC={auroc:.3f}")
        tracker.append(
            cohort=cohort_name,
            outcome=outcome_name,
            variant=result_variant,
            seed=seed,
            stage="cached",
            status="success",
            output_dir=str(cell_dir),
            artifact=str(metrics_path),
        )
        return {
            "cohort": cohort_name,
            "outcome": outcome_name,
            "variant": result_variant,
            "metrics": metrics,
        }

    encoder_source = variant_cfg.get("encoder_source", "contrastive")
    training_mode = variant_cfg.get("training_mode")
    if training_mode is not None and training_mode not in {
        "cox",
        "ipcw_bce",
    }:
        print(f"  Invalid training_mode={training_mode!r}, skipping.")
        tracker.append(
            cohort=cohort_name,
            outcome=outcome_name,
            variant=result_variant,
            seed=seed,
            stage="configuration",
            status="failed",
            output_dir=str(cell_dir),
            reason=f"invalid training_mode={training_mode!r}",
        )
        if fail_fast:
            raise ValueError(f"Invalid training_mode={training_mode!r}")
        return None
    if training_mode == "ipcw_bce" and n_hours_end is None:
        print("  IPCW-BCE requires n_hours_end_include, skipping.")
        tracker.append(
            cohort=cohort_name,
            outcome=outcome_name,
            variant=result_variant,
            seed=seed,
            stage="configuration",
            status="skipped",
            output_dir=str(cell_dir),
            reason="ipcw_bce requires n_hours_end_include",
        )
        return None
    if dry_run:
        action = (
            "evaluate joint checkpoint"
            if encoder_source == "joint"
            else (
                f"{training_mode} finetune and evaluate"
                if training_mode in {"cox", "ipcw_bce"}
                else "finetune and evaluate"
            )
        )
        print(
            f"  [DRY RUN] Would {action} {result_variant} "
            f"on {cohort_name}/{outcome_name}"
        )
        tracker.append(
            cohort=cohort_name,
            outcome=outcome_name,
            variant=result_variant,
            seed=seed,
            stage="finetune_evaluate",
            status="dry_run",
            output_dir=str(cell_dir),
        )
        return None

    encoder_ckpt_template = variant_cfg.get("encoder_ckpt")
    if encoder_ckpt_template is None and encoder_source not in {
        "random_init",
        "none",
        "scratch",
        "no_pretraining",
    }:
        print("  Missing encoder_ckpt for checkpointed variant, skipping.")
        tracker.append(
            cohort=cohort_name,
            outcome=outcome_name,
            variant=result_variant,
            seed=seed,
            stage="configuration",
            status="failed",
            output_dir=str(cell_dir),
            reason="missing encoder_ckpt",
        )
        if fail_fast:
            raise ValueError(f"Variant {result_variant!r} is missing encoder_ckpt.")
        return None
    encoder_ckpt = (
        format_variant_path(
            encoder_ckpt_template,
            cohort_name,
            outcome_name,
            seed,
            training_cohort=training_cohort,
        )
        if encoder_ckpt_template is not None
        else "null"
    )
    if encoder_source == "joint":
        ckpt_path = Path(encoder_ckpt)
    else:
        ckpt_path = cell_dir / "best.ckpt"
    if encoder_source != "joint" and (not ckpt_path.exists() or overwrite):
        ckpt_path = run_finetune(
            encoder_ckpt=encoder_ckpt,
            encoder_source=encoder_source,
            cohort=cohort_name,
            cohort_data_dir=data_dir,
            outcome_name=outcome_name,
            outcome_path=outcome_file_path(data_dir, outcome_name, outcome_cfg),
            output_dir=cell_dir,
            base_config=base_config,
            n_hours_start_include=n_hours_start,
            n_hours_end_include=n_hours_end,
            competing_outcome_path=competing_outcome,
            eligibility_path=eligibility,
            registry_start_date=registry_start_date,
            training_mode=training_mode,
            population_path=pop_file,
            cohort_fine_col=cohort_fine_col,
            cohort_fine_value=cohort_fine_value,
            extra_overrides=[
                *(variant_cfg.get("extra_overrides") or []),
                f"seed={seed}",
            ],
            log_dir=cell_dir / "logs",
        )

    if ckpt_path is None or not Path(ckpt_path).exists():
        print(f"  Checkpoint not found: {ckpt_path}, skipping.")
        tracker.append(
            cohort=cohort_name,
            outcome=outcome_name,
            variant=result_variant,
            seed=seed,
            stage="finetune",
            status="failed",
            output_dir=str(cell_dir),
            artifact=str(ckpt_path),
            reason="checkpoint not produced or not found",
        )
        if fail_fast:
            raise FileNotFoundError(str(ckpt_path))
        return None

    metrics = run_evaluate(
        ckpt_path=Path(ckpt_path),
        cohort=cohort_name,
        cohort_data_dir=data_dir,
        outcome_name=outcome_name,
        outcome_path=outcome_file_path(data_dir, outcome_name, outcome_cfg),
        output_dir=cell_dir,
        encoder_source=encoder_source,
        n_hours_start_include=n_hours_start,
        n_hours_end_include=n_hours_end,
        competing_outcome_path=competing_outcome,
        eligibility_path=eligibility,
        registry_start_date=registry_start_date,
        model_family=result_variant,
        training_stage=(
            "survival_finetuning"
            if training_mode in {"cox", "ipcw_bce"}
            else variant_cfg.get("training_stage", "per_task_finetuning")
        ),
        encoder_frozen=(
            True if variant_cfg.get("training_stage") == "linear_probe" else None
        ),
        head_type=(
            "linear_probe"
            if variant_cfg.get("training_stage") == "linear_probe"
            else None
        ),
        cohort_fine_col=cohort_fine_col,
        cohort_fine_value=cohort_fine_value,
        log_dir=cell_dir / "logs",
    )

    if metrics is not None:
        auroc = metrics.get("discrimination", {}).get("auroc", float("nan"))
        print(f"  Done: AUROC={auroc:.3f}")
        tracker.append(
            cohort=cohort_name,
            outcome=outcome_name,
            variant=result_variant,
            seed=seed,
            stage="evaluate",
            status="success",
            evaluation_subset="full",
            output_dir=str(cell_dir),
            artifact=str(cell_dir / "metrics.json"),
        )
        if ipi_col and result_variant == "opera":
            ipi_path, ipi_coverage, ipi_subjects = prepare_ipi_subset_predictions(
                pop_file,
                outcome_file_path(data_dir, outcome_name, outcome_cfg),
                ipi_col,
                output_dir / cohort_name / outcome_name / "ipi",
                eligibility_path=eligibility,
                registry_start_date=registry_start_date,
                cohort=cohort_name,
                outcome_name=outcome_name,
                cohort_fine_col=cohort_fine_col,
                cohort_fine_value=cohort_fine_value,
            )
            if ipi_path is not None:
                subset_pred = write_npz_prediction_subset(
                    cell_dir / "predictions.npz",
                    ipi_subjects,
                    cell_dir / "opera_ipi_subset_predictions.csv",
                )
                if subset_pred is not None:
                    run_prediction_evaluate(
                        predictions_path=str(subset_pred),
                        cohort=cohort_name,
                        outcome_name=outcome_name,
                        outcome_path=outcome_file_path(
                            data_dir, outcome_name, outcome_cfg
                        ),
                        model_family=result_variant,
                        output_dir=cell_dir / "ipi_complete",
                        n_hours_start_include=n_hours_start,
                        n_hours_end_include=n_hours_end,
                        competing_outcome_path=competing_outcome,
                        eligibility_path=eligibility,
                        registry_start_date=registry_start_date,
                        rarity_mode=rarity_mode,
                        baseline_model=baseline_model,
                        ipi_coverage=ipi_coverage,
                        evaluation_subset="ipi_complete",
                        seed=seed,
                        subgroup_path=subgroup_path,
                        subgroup_columns=subgroup_columns,
                        log_dir=cell_dir / "ipi_complete" / "logs",
                        population_path=pop_file,
                        cohort_fine_col=cohort_fine_col,
                        cohort_fine_value=cohort_fine_value,
                        evaluation_subjects_path=str(
                            ipi_path.parent / "ipi_subset_subjects.csv"
                        ),
                    )
        return {
            "cohort": cohort_name,
            "outcome": outcome_name,
            "variant": result_variant,
            "metrics": metrics,
        }
    tracker.append(
        cohort=cohort_name,
        outcome=outcome_name,
        variant=result_variant,
        seed=seed,
        stage="evaluate",
        status="failed",
        evaluation_subset="full",
        output_dir=str(cell_dir),
        reason="evaluate returned no metrics",
    )
    if fail_fast:
        raise RuntimeError(
            f"Evaluation failed for {cohort_name}/{outcome_name}/{result_variant}"
        )
    return None


# ── Main sweep loop ────────────────────────────────────────────────────────


def run_sweep(
    config_path: str,
    dry_run: bool = False,
    overwrite: bool = False,
    fail_fast: bool = False,
):
    cfg = load_sweep_config(config_path).to_mapping()
    # Sweep YAMLs may use server environment variables for shared data,
    # outcome, checkpoint, and result roots.  Expand them before paths are
    # handed to subprocesses (which are deliberately invoked without a shell).
    def expand_env(value):
        if isinstance(value, str):
            return os.path.expandvars(value)
        if isinstance(value, list):
            return [expand_env(item) for item in value]
        if isinstance(value, dict):
            return {key: expand_env(item) for key, item in value.items()}
        return value

    cfg = expand_env(cfg)

    output_dir = Path(cfg.get("output_dir", "./sweep_results"))
    output_dir.mkdir(parents=True, exist_ok=True)

    cohorts = cfg["cohorts"]
    model_variants = cfg["model_variants"]
    base_config = cfg.get("finetune_base_config", "opera/configs/finetune.yaml")
    rarity_cfg = cfg.get("rarity", {}) or {}
    rarity_mode = cfg.get("rarity_mode", rarity_cfg.get("mode", "none"))
    baseline_model = cfg.get("baseline_model", rarity_cfg.get("baseline_model"))
    seeds = cfg.get("seeds", [cfg.get("seed", 42)])
    subgroup_path = (cfg.get("paths", {}) or {}).get("subgroups") or (
        cfg.get("subgroups", {}) or {}
    ).get("path")
    subgroup_columns = (cfg.get("subgroups", {}) or {}).get("columns", [])
    if len(seeds) > 1:
        expanded_variants = {}
        for name, variant in model_variants.items():
            for seed in seeds:
                expanded = dict(variant)
                expanded["seed"] = seed
                expanded["model_family"] = name
                expanded_variants[f"{name}__seed{seed}"] = expanded
        model_variants = expanded_variants
    else:
        for variant in model_variants.values():
            variant.setdefault("seed", seeds[0])

    # outcomes: dict of {name: {n_hours_start_include, n_hours_end_include}}
    # Accept either the old list format (no windows → open-ended) or the new dict format.
    outcomes = normalize_outcome_config(cfg["outcomes"])

    all_results = []
    tracker = StatusTracker()
    planned_variant_cells = sum(
        1
        for variant in model_variants.values()
        for outcome_name in outcomes
        if variant_applies_to_outcome(variant, outcome_name)
    )
    total_cells = len(cohorts) * planned_variant_cells
    cell_idx = 0

    print(
        f"\nOPERA Sweep: {len(cohorts)} cohorts × {len(outcomes)} outcomes × "
        f"{len(model_variants)} variants = {total_cells} cells\n"
    )

    for cohort_name, cohort_cfg in cohorts.items():
        data_dir = cohort_cfg["data_dir"]
        ipi_col = cohort_cfg.get("ipi_score_col")
        pop_file = cohort_cfg.get(
            "population_file", str(Path(data_dir) / "population_full.csv")
        )
        cohort_fine_col = cohort_cfg.get("cohort_fine_col")
        cohort_fine_value = cohort_cfg.get("cohort_fine_value")
        # For train-on-grouped / eval-on-fine sweeps, training_cohort names the
        # grouped cohort whose checkpoint to use.  Falls back to cohort_name so
        # ordinary sweeps are unaffected.
        training_cohort = cohort_cfg.get("training_cohort", cohort_name)
        excluded_cohort_outcomes = set(cohort_cfg.get("exclude_outcomes", []))
        cohort_outcomes = {
            name: outcome
            for name, outcome in outcomes.items()
            if name not in excluded_cohort_outcomes
        }

        # ── IPI baseline (one per cohort × outcome, not per variant) ──
        for outcome_name, outcome_cfg in cohort_outcomes.items():
            results = _run_ipi_baseline_cell(
                cohort_name=cohort_name,
                outcome_name=outcome_name,
                outcome_cfg=outcome_cfg,
                ipi_col=ipi_col,
                pop_file=pop_file,
                data_dir=data_dir,
                output_dir=output_dir,
                seeds=seeds,
                tracker=tracker,
                cfg=cfg,
                dry_run=dry_run,
                fail_fast=fail_fast,
                rarity_mode=rarity_mode,
                baseline_model=baseline_model,
                subgroup_path=subgroup_path,
                subgroup_columns=subgroup_columns,
                cohort_fine_col=cohort_fine_col,
                cohort_fine_value=cohort_fine_value,
            )
            all_results.extend(results)

        # ── Foundation model variants ──────────────────
        for variant_name, variant_cfg in model_variants.items():
            result_variant = variant_cfg.get("model_family", variant_name)
            seed = variant_cfg.get("seed", cfg.get("seed", 42))
            for outcome_name, outcome_cfg in cohort_outcomes.items():
                if not variant_applies_to_outcome(variant_cfg, outcome_name):
                    continue
                cell_idx += 1
                window_str = (
                    f"{outcome_cfg.get('n_hours_end_include')}h"
                    if outcome_cfg.get("n_hours_end_include") is not None
                    else "open-ended"
                )
                print(
                    f"\n[{cell_idx}/{total_cells}] {cohort_name} × {outcome_name} ({window_str}) × {variant_name}"
                )
                result = _run_variant_cell(
                    cohort_name=cohort_name,
                    outcome_name=outcome_name,
                    outcome_cfg=outcome_cfg,
                    variant_name=variant_name,
                    variant_cfg=variant_cfg,
                    result_variant=result_variant,
                    seed=seed,
                    training_cohort=training_cohort,
                    cohort_fine_col=cohort_fine_col,
                    cohort_fine_value=cohort_fine_value,
                    data_dir=data_dir,
                    output_dir=output_dir,
                    base_config=base_config,
                    tracker=tracker,
                    cfg=cfg,
                    dry_run=dry_run,
                    overwrite=overwrite,
                    fail_fast=fail_fast,
                    rarity_mode=rarity_mode,
                    baseline_model=baseline_model,
                    ipi_col=ipi_col,
                    pop_file=pop_file,
                    subgroup_path=subgroup_path,
                    subgroup_columns=subgroup_columns,
                )
                if result is not None:
                    all_results.append(result)

    # ── Save sweep status and raw results ──────────────────────────────
    if len(tracker):
        tracker.write(output_dir)
        print(
            f"\nSweep status saved to {output_dir / 'sweep_cell_status.csv'} "
            f"({tracker.n_failed()} failed, {tracker.n_skipped()} skipped)"
        )

    with open(output_dir / "all_results_raw.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nRaw results saved to {output_dir / 'all_results_raw.json'}")

    # ── Build and save aggregated table ───────────────────────────────
    table = build_results_table(all_results)
    if not table.empty:
        table.to_csv(output_dir / "results_table.csv", index=False)
        print(f"Results table saved to {output_dir / 'results_table.csv'}")
        print(f"\n{table.to_string()}")

        # LaTeX
        latex = to_latex_table(table, metric="auroc")
        with open(output_dir / "results_table_auroc.tex", "w") as f:
            f.write(latex)
        print(f"LaTeX table saved to {output_dir / 'results_table_auroc.tex'}")

    return table


def main():
    parser = argparse.ArgumentParser(description="OPERA evaluation sweep")
    parser.add_argument("--config", required=True, help="Path to sweep_config.yaml")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be run without executing",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Rerun cells even if results already exist",
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop on the first failed or missing sweep cell instead of recording and continuing.",
    )
    args = parser.parse_args()

    run_sweep(
        args.config,
        dry_run=args.dry_run,
        overwrite=args.overwrite,
        fail_fast=args.fail_fast,
    )


if __name__ == "__main__":
    main()
