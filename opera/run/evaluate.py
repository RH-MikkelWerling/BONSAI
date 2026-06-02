"""
OPERA Evaluation Runner.

Loads a fine-tuned model checkpoint, runs inference on the test set,
computes comprehensive metrics, and generates all evaluation plots.

Usage:
    python -m opera.run.evaluate \
        ckpt_path=/path/to/finetune/best.ckpt \
        dataset=hematology_cohort \
        outcome=treatment_failure \
        output_dir=./evaluation_output
"""

import json
import hydra
import torch
import pandas as pd
import numpy as np
from pathlib import Path
from dotenv import load_dotenv
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from opera.compat.bonsai import (
    BonsaiFinetune,
    FinetuneDataset,
    binarize_outcomes,
    dynamic_padding,
    filter_subject_data,
    split_and_binarize_outcomes,
)
from opera.functional.outcomes import attach_prediction_censor_abspos
from opera.functional.checkpointing import load_opera_finetune_model_from_checkpoint

from opera.evaluation.metrics import full_evaluation, format_evaluation_summary, _derive_time_horizons
from opera.evaluation.results_schema import (
    bootstrap_ci_rows,
    build_result_row,
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


def checkpoint_training_mode(ckpt_path: str) -> str:
    """Return the OPERA training mode stored in checkpoint metadata, if any."""
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    metadata = ckpt.get("hyper_parameters", {}).get("checkpoint_metadata", {})
    return metadata.get("training_mode", "bce")


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

    # ── Load model ───────────────────────────────────────────────────
    vocab = torch.load(cfg.paths.vocabulary)
    model = load_opera_finetune_model_from_checkpoint(
        cfg.ckpt_path,
        strict=cfg.get("strict_checkpoint_load", True),
    )
    training_mode = checkpoint_training_mode(cfg.ckpt_path)
    if model.config.vocab_size != len(vocab):
        raise ValueError(
            f"Checkpoint vocab_size={model.config.vocab_size} does not match "
            f"loaded vocabulary size={len(vocab)} from {cfg.paths.vocabulary}."
        )
    model = model.to(device)
    model.eval()

    # ── Load test data ───────────────────────────────────────────────
    outcomes = pd.read_parquet(cfg.paths.outcome)
    outcomes = attach_prediction_censor_abspos(outcomes)

    test_key = cfg.labels.get("test_key", "held_out")
    test_df  = outcomes[outcomes["split"] == test_key].copy()

    competing_df = None
    competing_path = cfg.paths.get("competing_outcome")
    if competing_path:
        competing_df = pd.read_parquet(competing_path)

    # ALL test patients — used for survival metrics (censoring handled by IPCW)
    all_test_outcomes = binarize_outcomes(
        test_df,
        n_hours_start_include=cfg.labels.n_hours_start_include,
        n_hours_end_include=cfg.labels.get("n_hours_end_include"),
        require_min_followup=False,
        competing_event_df=competing_df,
    )

    # Full-follow-up patients only — used for binary classification metrics
    full_fu_outcomes = binarize_outcomes(
        test_df,
        n_hours_start_include=cfg.labels.n_hours_start_include,
        n_hours_end_include=cfg.labels.get("n_hours_end_include"),
        require_min_followup=True,
        competing_event_df=competing_df,
    )
    full_fu_sids = set(full_fu_outcomes.keys())

    # Build dataset / loader over ALL test patients
    test_data = torch.load(cfg.paths.test_split)
    population = pd.read_csv(cfg.paths.population)
    test_data = [s for s in test_data if s["subject_id"] in all_test_outcomes]
    test_data = filter_subject_data(test_data, population["subject_id"])
    if not test_data:
        raise ValueError(
            "No test subjects remain after filtering to outcome labels and "
            "population membership. Check paths.test_split, paths.population, "
            f"paths.outcome={cfg.paths.outcome}, and test_key={test_key!r}."
        )

    background_length = (test_data[0]["segment"] == 0).sum()

    test_dataset = FinetuneDataset(
        test_data,
        outcomes=all_test_outcomes,
        predict_token_id=vocab["[CLS]"],
        background_length=background_length,
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
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}

            logits = model(batch).squeeze(-1)

            # Get embeddings (BiGRU pooled)
            enc_out = BonsaiFinetune.__bases__[0].forward(model, batch)
            hidden = enc_out[0]
            emb = model.cls(hidden, batch["attention_mask"], return_embedding=True)

            all_sids.append(batch["subject_id"].cpu().numpy())
            all_labels.append(batch["target"].cpu().numpy().squeeze())
            all_logits.append(logits.cpu().numpy())
            all_embeddings.append(emb.cpu().numpy())

    labels_all   = np.concatenate(all_labels)
    logits_all   = np.concatenate(all_logits)
    probs_all    = 1.0 / (1.0 + np.exp(-logits_all))
    sids_all     = np.concatenate(all_sids)
    embeddings   = np.concatenate(all_embeddings)

    # ── Survival fields (all patients) ───────────────────────────────
    times_all  = np.array([
        all_test_outcomes[int(sid)].get("time_days", float("nan"))
        for sid in sids_all
    ])
    events_all = np.array([
        all_test_outcomes[int(sid)].get("event", -1)
        for sid in sids_all
    ])

    # Determine time horizons from config (n_hours_end_include → days)
    n_hours_end = cfg.labels.get("n_hours_end_include")
    if n_hours_end is not None:
        if training_mode == "ipcw_bce":
            time_horizons = [float(n_hours_end) / 24.0]
        else:
            time_horizons = _derive_time_horizons(n_hours_end / 24.0)
    else:
        # Open-ended outcome: use calendar-year checkpoints up to observed data
        time_horizons = [365.0, 730.0]

    # ── Binary metrics (full-follow-up patients only) ────────────────
    binary_mask  = np.array([int(sid) in full_fu_sids for sid in sids_all])
    labels_bin   = labels_all[binary_mask]
    probs_bin    = probs_all[binary_mask]

    print(
        f"Evaluation: {len(sids_all)} total patients | "
        f"{binary_mask.sum()} with full follow-up (binary metrics) | "
        f"{(~binary_mask).sum()} censored-only (survival metrics only)"
    )

    # ── Evaluation ───────────────────────────────────────────────────
    report = full_evaluation(
        labels_bin, probs_bin,
        threshold=cfg.get("threshold", 0.5),
        n_bootstrap=cfg.get("n_bootstrap", 1000),
        times=times_all,
        events=events_all,
        survival_probabilities=probs_all,
        time_horizons=time_horizons,
    )
    if training_mode == "cox":
        mark_probability_metrics_not_applicable(report)
    report["evaluation_notes"] = {
        "training_mode": training_mode,
        "time_horizons_days": time_horizons,
        "calibration_note": (
            "Cox checkpoints output relative risk scores. Binary AUROC/AUPRC "
            "and survival ranking metrics are meaningful; calibration, Brier, "
            "threshold, and decision-curve metrics are not reported for Cox."
            if training_mode == "cox"
            else None
        ),
    }

    # Print summary
    summary = (
        format_cox_evaluation_summary(report)
        if training_mode == "cox"
        else format_evaluation_summary(report)
    )
    print(summary)

    # Save text report
    with open(output_dir / "evaluation_report.txt", "w") as f:
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
        checkpoint_path=cfg.ckpt_path,
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
    with open(output_dir / "metrics.json", "w") as f:
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

    # ── Generate plots ───────────────────────────────────────────────
    print("Generating plots...")
    window_days = (n_hours_end / 24.0) if n_hours_end is not None else None
    if training_mode != "cox":
        plot_full_evaluation(
            labels_bin, probs_bin,
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
            embeddings, labels_all, method="umap",
            title="Test Set Embedding Space (UMAP)",
            save_path=str(output_dir / "plots" / "embedding_umap.png"),
        )
    except ImportError:
        print("  UMAP not installed, skipping UMAP plot. Install with: pip install umap-learn")

    plot_similarity_distributions(
        embeddings, labels_all,
        save_path=str(output_dir / "plots" / "similarity_distributions.png"),
    )

    print(f"\nEvaluation complete. Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
