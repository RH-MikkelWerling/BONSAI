"""
OPERA Evaluation Runner.

Loads a fine-tuned model checkpoint, runs inference on the test set,
computes comprehensive metrics, and generates all evaluation plots.

Usage:
    python -m opera.run.evaluate \
        run_dir=/path/to/finetune/run \
        dataset=hematology_cohort \
        outcome=treatment_failure \
        output_dir=./evaluation_output
"""

import json
import warnings
import hydra
import torch
import pandas as pd
import numpy as np
from pathlib import Path
from dotenv import load_dotenv
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from opera.compat.bonsai import (
    FinetuneDataset,
    dynamic_padding,
    filter_subject_data,
)
from opera.evaluation.cohorts import (
    assert_cohort_parity,
    build_evaluation_cohorts,
    cohort_summary,
    population_subject_ids,
    population_subject_strata,
)
from opera.functional.checkpointing import load_opera_finetune_model_from_checkpoint
from opera.modules.datamodules.OutcomeFinetuneDataModule import load_subject_pool

from opera.evaluation.metrics import (
    compute_macro_stratified_concordance,
    compute_stratified_concordance,
    full_evaluation,
    format_evaluation_summary,
)
from opera.evaluation.results_schema import (
    bootstrap_ci_rows,
    build_result_row,
    write_per_cohort_concordance_artifact,
    write_result_artifacts,
)
from opera.evaluation.subgroups import compute_subgroup_metrics, load_subgroup_table
from opera.visualization.classification_plots import plot_full_evaluation
from opera.visualization.embedding_plots import (
    plot_embedding_projection,
    plot_similarity_distributions,
)

load_dotenv()


def resolve_device(device_cfg: str) -> str:
    if device_cfg in (None, "auto"):
        return "cuda" if torch.cuda.is_available() else "cpu"
    return str(device_cfg)


def resolve_attention_backend(backend_cfg: str, device: str) -> str | None:
    """Resolve the checkpoint backend while keeping Flash as the CUDA default."""
    if backend_cfg in (None, "auto"):
        return None if str(device).startswith("cuda") else "sdpa"
    if backend_cfg in {"checkpoint", "saved"}:
        return None
    if backend_cfg not in {"flash", "sdpa"}:
        raise ValueError("attention_backend must be auto, checkpoint, flash, or sdpa.")
    return str(backend_cfg)


def checkpoint_training_mode(ckpt_path: str) -> str:
    """Return the OPERA training mode stored in checkpoint metadata, if any."""
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    metadata = ckpt.get("hyper_parameters", {}).get("checkpoint_metadata", {})
    return metadata.get("training_mode", "bce")


def _read_checkpoint_sidecar(run_dir: Path) -> dict:
    sidecar = run_dir / "checkpoint_metadata.json"
    if not sidecar.exists():
        return {}
    return json.loads(sidecar.read_text(encoding="utf-8"))


def resolve_evaluation_checkpoint(cfg: DictConfig) -> tuple[Path, dict]:
    """Resolve the tuning-selected checkpoint unless an explicit path is set."""
    explicit_path = cfg.get("ckpt_path")
    if explicit_path not in (None, "", "???"):
        ckpt_path = Path(explicit_path)
        warning = (
            "Explicit ckpt_path override used; provenance cannot guarantee this "
            "was the tuning-selected best checkpoint."
        )
        warnings.warn(warning, RuntimeWarning, stacklevel=2)
        sidecar = _read_checkpoint_sidecar(ckpt_path.parent)
        metadata = sidecar.get("checkpoint_metadata", {})
        return ckpt_path, {
            "checkpoint_source": "explicit_override",
            "checkpoint_path": str(ckpt_path),
            "run_dir": str(ckpt_path.parent),
            "selection_split": metadata.get("selection_split"),
            "selection_metric": metadata.get("selection_metric"),
            "selection_mode": metadata.get("selection_mode"),
            "sidecar_path": str(ckpt_path.parent / "checkpoint_metadata.json"),
            "warning": warning,
        }

    run_dir = cfg.get("run_dir")
    if run_dir in (None, "", "???"):
        raise ValueError("Evaluation requires run_dir or explicit ckpt_path.")
    run_dir = Path(run_dir)
    ckpt_path = run_dir / "best.ckpt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Tuning-selected checkpoint not found: {ckpt_path}")
    sidecar = _read_checkpoint_sidecar(run_dir)
    metadata = sidecar.get("checkpoint_metadata", {})
    return ckpt_path, {
        "checkpoint_source": "run_dir_best",
        "checkpoint_path": str(ckpt_path),
        "run_dir": str(run_dir),
        "selection_split": metadata.get("selection_split", "tuning"),
        "selection_metric": metadata.get("selection_metric"),
        "selection_mode": metadata.get("selection_mode"),
        "sidecar_path": str(run_dir / "checkpoint_metadata.json"),
        "warning": None,
    }


def extract_patient_embeddings(model, batch: dict) -> torch.Tensor:
    """Return the pooled representation used by a supported prediction head."""
    if hasattr(model, "get_pooled_representation"):
        return model.get_pooled_representation(batch)
    if hasattr(model, "pooled_embedding"):
        return model.pooled_embedding(batch)
    raise TypeError(
        f"Cannot extract patient embeddings from model type {type(model).__name__}."
    )


def mark_probability_metrics_not_applicable(report: dict) -> None:
    """Neutralize calibrated-probability metrics for Cox risk-score checkpoints."""
    for key in (
        "sensitivity",
        "specificity",
        "ppv",
        "npv",
        "f1",
        "mcc",
        "accuracy",
        "log_loss",
        "threshold",
    ):
        if key in report.get("discrimination", {}):
            report["discrimination"][key] = float("nan")

    for key in (
        "brier_score",
        "ece",
        "mce",
        "hl_statistic",
        "hl_pvalue",
        "calibration_intercept",
        "calibration_slope",
    ):
        if key in report.get("calibration", {}):
            report["calibration"][key] = float("nan")
    if "calibration" in report:
        report["calibration"]["bin_data"] = []

    for key in (
        "brier_score",
        "sensitivity",
        "specificity",
        "ppv",
        "npv",
        "f1",
        "mcc",
        "accuracy",
        "log_loss",
    ):
        if key in report.get("bootstrap_ci", {}):
            report["bootstrap_ci"][key] = {
                "mean": float("nan"),
                "lower": float("nan"),
                "upper": float("nan"),
                "std": float("nan"),
            }

    report["threshold_sweep"] = pd.DataFrame()
    report["decision_curve"] = pd.DataFrame()
    for metrics in report.get("survival", {}).get("per_horizon", {}).values():
        metrics["ipcw_brier"] = float("nan")
    for metrics in report.get("competing_risk", {}).get("per_horizon", {}).values():
        metrics["cif_brier"] = float("nan")
    for metric, values in report.get("competing_risk_bootstrap_ci", {}).items():
        if metric.startswith("cif_brier_"):
            values.update(mean=float("nan"), lower=float("nan"), upper=float("nan"))


def format_cox_evaluation_summary(report: dict) -> str:
    """Summarize Cox checkpoints without implying calibrated probabilities."""
    lines = [
        "=" * 70,
        "OPERA COX RISK-SCORE EVALUATION REPORT",
        "=" * 70,
    ]
    discrimination = report.get("discrimination", {})
    lines.append("\nRanking discrimination on full-follow-up binary subset")
    lines.append(f"  AUROC:        {discrimination.get('auroc', float('nan')):.4f}")
    lines.append(f"  AUPRC:        {discrimination.get('auprc', float('nan')):.4f}")
    lines.append(
        "  Prevalence:   "
        f"{discrimination.get('prevalence', float('nan')):.4f} "
        f"({discrimination.get('n_positive')}/{discrimination.get('n_total')})"
    )
    lines.append(
        "\nCalibration, threshold, Brier, and decision-curve metrics are not "
        "reported for Cox checkpoints because the model outputs relative risk, "
        "not event probability."
    )

    survival = report.get("survival", {})
    if survival:
        ci = report.get("survival_bootstrap_ci", {}).get("concordance_index", {})
        lines.append(
            "\nSurvival metrics "
            f"(all patients, n={survival.get('n_total')}, "
            f"events={survival.get('n_events')})"
        )
        c_index = survival.get("concordance_index", float("nan"))
        if ci:
            lines.append(
                f"  C-index:      {c_index:.4f} "
                f"[{ci.get('lower', float('nan')):.4f}, "
                f"{ci.get('upper', float('nan')):.4f}]"
            )
        else:
            lines.append(f"  C-index:      {c_index:.4f}")
        for label, metrics in survival.get("per_horizon", {}).items():
            lines.append(
                f"  IPCW-AUC@{label:>4s}: "
                f"{metrics.get('ipcw_auc', float('nan')):.4f} "
                f"(cases={metrics.get('n_cases', 0)}, "
                f"controls={metrics.get('n_controls', 0)}, "
                f"excluded={metrics.get('n_excluded', 0)})"
            )
    return "\n".join(lines)


@hydra.main(
    config_path="../configs",
    config_name="evaluate",
    version_base="1.2",
)
def main(cfg: DictConfig) -> None:
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(cfg.get("device", "auto"))
    resolved_ckpt_path, checkpoint_provenance = resolve_evaluation_checkpoint(cfg)

    # ── Load model ───────────────────────────────────────────────────
    vocab = torch.load(cfg.paths.vocabulary)
    model = load_opera_finetune_model_from_checkpoint(
        str(resolved_ckpt_path),
        strict=cfg.get("strict_checkpoint_load", True),
        attn_type=resolve_attention_backend(
            cfg.get("attention_backend", "auto"), device
        ),
    )
    training_mode = checkpoint_training_mode(str(resolved_ckpt_path))
    if model.hparams["vocab_size"] != len(vocab):
        raise ValueError(
            f"Checkpoint vocab_size={model.hparams['vocab_size']} does not match "
            f"loaded vocabulary size={len(vocab)} from {cfg.paths.vocabulary}."
        )
    model = model.to(device)
    model.eval()

    # ── Load test data ───────────────────────────────────────────────
    test_key = cfg.labels.get("test_key", "held_out")
    competing_path = cfg.paths.get("competing_outcome")

    # ALL test patients — used for survival metrics (censoring handled by IPCW)
    cohort_fine_col = cfg.get("cohort_fine_col")
    cohort_fine_value = cfg.get("cohort_fine_value")
    population_ids = population_subject_ids(
        cfg.paths.population,
        cohort_fine_col=cohort_fine_col,
        cohort_fine_value=cohort_fine_value,
    )
    evaluation_cohorts = build_evaluation_cohorts(
        cfg.paths.outcome,
        split=test_key,
        n_hours_start_include=cfg.labels.n_hours_start_include,
        n_hours_end_include=cfg.labels.get("n_hours_end_include"),
        competing_outcomes=competing_path,
        eligibility=cfg.paths.get("eligibility"),
        registry_start_date=cfg.labels.get("registry_start_date"),
        cohort=cfg.get("dataset"),
        outcome_name=cfg.get("outcome"),
        allowed_subject_ids=population_ids,
    )
    all_test_outcomes = evaluation_cohorts.survival.records

    # Full-follow-up patients only — used for binary classification metrics
    full_fu_sids = evaluation_cohorts.fixed_horizon.subject_ids

    # Build dataset / loader over ALL test patients
    test_data = load_subject_pool(cfg.paths.subject_data_paths)
    # Optional fine-cohort subsetting for train-on-grouped / eval-on-fine.
    if cohort_fine_col and cohort_fine_value:
        print(
            f"cohort_fine filter: {cohort_fine_col}={cohort_fine_value!r} "
            f"-> {len(population_ids)} subjects"
        )

    test_data = [s for s in test_data if s["subject_id"] in all_test_outcomes]
    test_data = filter_subject_data(test_data, population_ids)
    if not test_data:
        raise ValueError(
            "No test subjects remain after filtering to outcome labels and "
            "population membership. Check paths.subject_data_paths, paths.population, "
            f"paths.outcome={cfg.paths.outcome}, and test_key={test_key!r}."
        )

    background_length = (test_data[0]["segment"] == 0).sum()
    max_len = cfg.get("max_len")
    if max_len is None:
        max_len = model.hparams["max_seqlen"]

    test_dataset = FinetuneDataset(
        test_data,
        outcomes=all_test_outcomes,
        predict_token_id=vocab["[CLS]"],
        background_length=background_length,
        max_len=int(max_len),
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg.get("batch_size", 64),
        num_workers=cfg.get("num_workers", 4),
        shuffle=False,
        collate_fn=dynamic_padding,
    )

    # ── Inference ────────────────────────────────────────────────────
    all_sids, all_labels, all_logits, all_embeddings = [], [], [], []

    with torch.no_grad():
        for batch in test_loader:
            batch = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }

            logits = model(batch).squeeze(-1)

            emb = extract_patient_embeddings(model, batch)

            all_sids.append(batch["subject_id"].cpu().numpy())
            all_labels.append(batch["target"].cpu().numpy().squeeze())
            all_logits.append(logits.cpu().numpy())
            all_embeddings.append(emb.cpu().numpy())

    labels_all = np.concatenate(all_labels)
    logits_all = np.concatenate(all_logits)
    probs_all = 1.0 / (1.0 + np.exp(-logits_all))
    sids_all = np.concatenate(all_sids)
    embeddings = np.concatenate(all_embeddings)

    # ── Survival fields (all patients) ───────────────────────────────
    times_all = np.array(
        [all_test_outcomes[int(sid)].get("time_days", float("nan")) for sid in sids_all]
    )
    events_all = np.array(
        [all_test_outcomes[int(sid)].get("event", -1) for sid in sids_all]
    )

    # Determine time horizons from config (n_hours_end_include → days)
    n_hours_end = cfg.labels.get("n_hours_end_include")
    if n_hours_end is not None:
        time_horizons = [float(n_hours_end) / 24.0]
    else:
        # Open-ended outcome: use calendar-year checkpoints up to observed data
        time_horizons = [365.0, 730.0]

    # ── Binary metrics (full-follow-up patients only) ────────────────
    binary_mask = np.array([int(sid) in full_fu_sids for sid in sids_all])
    labels_bin = labels_all[binary_mask]
    probs_bin = probs_all[binary_mask]

    print(
        f"Evaluation: {len(sids_all)} total patients | "
        f"{binary_mask.sum()} with full follow-up (binary metrics) | "
        f"{(~binary_mask).sum()} censored-only (survival metrics only)"
    )

    # ── Evaluation ───────────────────────────────────────────────────
    assert_cohort_parity(
        evaluation_cohorts.survival,
        sids_all,
        model_name="OPERA",
        outcome_name=cfg.get("outcome", "unknown"),
    )
    assert_cohort_parity(
        evaluation_cohorts.fixed_horizon,
        sids_all[binary_mask],
        model_name="OPERA",
        outcome_name=cfg.get("outcome", "unknown"),
    )
    print(
        cohort_summary(evaluation_cohorts.fixed_horizon, cfg.get("outcome", "unknown"))
    )
    print(cohort_summary(evaluation_cohorts.survival, cfg.get("outcome", "unknown")))
    report = full_evaluation(
        labels_bin,
        probs_bin,
        threshold=cfg.get("threshold", 0.5),
        n_bootstrap=cfg.get("n_bootstrap", 1000),
        times=times_all,
        events=events_all,
        survival_probabilities=probs_all,
        time_horizons=time_horizons,
        competing_risk=bool(competing_path),
    )
    stratified_cfg = cfg.get("stratified_concordance", {}) or {}
    if stratified_cfg.get("enabled", False):
        strata_col = stratified_cfg.get("strata_col", "cohort_fine")
        strata_all = population_subject_strata(
            cfg.paths.population,
            sids_all,
            strata_col,
        )
        survival_valid = (
            np.isfinite(times_all) & np.isfinite(probs_all) & (events_all >= 0)
        )
        stratified_n_bootstrap = stratified_cfg.get("n_bootstrap")
        if stratified_n_bootstrap is None:
            stratified_n_bootstrap = cfg.get("n_bootstrap", 1000)
        report["stratified_concordance"] = {
            "strata_col": strata_col,
            "micro": compute_stratified_concordance(
                times_all[survival_valid],
                events_all[survival_valid],
                probs_all[survival_valid],
                strata_all[survival_valid],
                n_bootstrap=int(stratified_n_bootstrap),
                seed=int(cfg.get("seed", 42)),
            ),
            "macro": compute_macro_stratified_concordance(
                times_all[survival_valid],
                events_all[survival_valid],
                probs_all[survival_valid],
                strata_all[survival_valid],
                n_bootstrap=int(stratified_n_bootstrap),
                seed=int(cfg.get("seed", 42)),
                min_events=int(stratified_cfg.get("min_events", 10)),
            ),
        }
    if training_mode == "cox":
        mark_probability_metrics_not_applicable(report)
    report["evaluation_notes"] = {
        "training_mode": training_mode,
        "survival_estimand": {
            "cox": "cause_specific_hazard",
            "ipcw_bce": "net_risk",
            "ipcw_cif_bce": "cumulative_incidence",
        }.get(training_mode),
        "time_horizons_days": time_horizons,
        "calibration_note": (
            "Cox checkpoints output relative risk scores. Binary AUROC/AUPRC "
            "and survival ranking metrics are meaningful; calibration, Brier, "
            "threshold, and decision-curve metrics are not reported for Cox."
            if training_mode == "cox"
            else None
        ),
    }
    report["checkpoint_provenance"] = checkpoint_provenance

    # Print summary
    summary = (
        format_cox_evaluation_summary(report)
        if training_mode == "cox"
        else format_evaluation_summary(report)
    )
    print(summary.encode("ascii", errors="replace").decode("ascii"))

    # Save text report
    with open(output_dir / "evaluation_report.txt", "w", encoding="utf-8") as f:
        f.write(summary)

    # Save detailed results
    np.savez(
        output_dir / "predictions.npz",
        subject_ids=sids_all,
        labels=labels_all,
        probabilities=probs_all,
        logits=logits_all,
        embeddings=embeddings,
        times=times_all,
        events=events_all,
        binary_mask=binary_mask.astype(np.uint8),
    )

    result_row = build_result_row(
        cfg,
        report,
        checkpoint_path=str(resolved_ckpt_path),
        split=test_key,
    )

    # Save metrics as JSON
    json_safe = {}
    for k, v in report.items():
        if isinstance(v, dict):
            json_safe[k] = {
                kk: (vv.tolist() if isinstance(vv, np.ndarray) else vv)
                for kk, vv in v.items()
                if not isinstance(vv, (pd.DataFrame, list))
            }
        elif isinstance(v, np.ndarray):
            json_safe[k] = v.tolist()
    json_safe["result_metadata"] = result_row
    with open(output_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(json_safe, f, indent=2, default=str)

    report["threshold_sweep"].to_csv(output_dir / "threshold_sweep.csv", index=False)
    report["decision_curve"].to_csv(output_dir / "decision_curve.csv", index=False)
    report["high_risk_enrichment"].to_csv(
        output_dir / "high_risk_enrichment.csv",
        index=False,
    )
    bootstrap_ci_rows(report).to_csv(output_dir / "bootstrap_ci.csv", index=False)
    subgroup_path = cfg.paths.get("subgroups")
    subgroup_columns = cfg.get("subgroups", {}).get("columns", [])
    if subgroup_path and subgroup_columns:
        subgroup_df = load_subgroup_table(subgroup_path)
        subgroup_metrics = compute_subgroup_metrics(
            subject_ids=sids_all[binary_mask],
            labels=labels_bin,
            probabilities=probs_bin,
            subgroup_df=subgroup_df,
            columns=subgroup_columns,
            threshold=cfg.get("threshold", 0.5),
        )
        subgroup_metrics.to_csv(output_dir / "subgroup_metrics.csv", index=False)

    write_result_artifacts(result_row, output_dir)
    write_per_cohort_concordance_artifact(report, result_row, output_dir)

    # ── Generate plots ───────────────────────────────────────────────
    print("Generating plots...")
    window_days = (n_hours_end / 24.0) if n_hours_end is not None else None
    if training_mode != "cox":
        plot_full_evaluation(
            labels_bin,
            probs_bin,
            output_dir=str(output_dir / "plots"),
            bootstrap_ci=report["bootstrap_ci"],
            times=times_all,
            events=events_all,
            survival_probabilities=probs_all,
            window_days=window_days,
            outcome_name=cfg.get("outcome", ""),
        )

    # Embedding visualizations
    print("Generating embedding visualizations...")
    try:
        plot_embedding_projection(
            embeddings,
            labels_all,
            method="umap",
            title="Test Set Embedding Space (UMAP)",
            save_path=str(output_dir / "plots" / "embedding_umap.png"),
        )
    except ImportError:
        print(
            "  UMAP not installed, skipping UMAP plot. Install with: pip install umap-learn"
        )

    plot_similarity_distributions(
        embeddings,
        labels_all,
        save_path=str(output_dir / "plots" / "similarity_distributions.png"),
    )

    print(f"\nEvaluation complete. Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
