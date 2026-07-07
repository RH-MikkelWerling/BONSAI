"""
OPERA Joint Model Evaluator.

Evaluates the joint multi-task model on a specific cohort × outcome cell.
The joint model has one head per outcome — this script extracts predictions
from the relevant head and runs the full evaluation suite.

This is called by sweep.py when encoder_source == "joint".

Usage
─────
python -m opera.run.evaluate_joint \\
    ckpt_path=/ckpts/joint_finetune/best.ckpt \\
    paths.dir=/data/dlbcl \\
    paths.outcome=/data/dlbcl/outcomes/mortality.parquet \\
    outcome_name=mortality_1y \\
    output_dir=./results/dlbcl/mortality_1y/opera_joint
"""

import json
import hydra
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from opera.compat.bonsai import (
    FinetuneDataset,
    dynamic_padding,
    filter_subject_data,
)
from bonsai.functional.checkpointing import load_joint_model_from_checkpoint

from opera.evaluation.cohorts import (
    assert_cohort_parity,
    build_evaluation_cohorts,
    cohort_summary,
    population_subject_ids,
    population_subject_strata,
)
from opera.evaluation.metrics import (
    compute_macro_stratified_concordance,
    compute_stratified_concordance,
    full_evaluation,
    format_evaluation_summary,
    _derive_time_horizons,
)
from opera.evaluation.results_schema import (
    bootstrap_ci_rows,
    build_result_row,
    write_per_cohort_concordance_artifact,
    write_result_artifacts,
)
from opera.evaluation.subgroups import compute_subgroup_metrics, load_subgroup_table
from opera.visualization.classification_plots import plot_full_evaluation

load_dotenv()


def resolve_device(device_cfg: str) -> str:
    if device_cfg in (None, "auto"):
        return "cuda" if torch.cuda.is_available() else "cpu"
    return str(device_cfg)


@hydra.main(
    config_path="../configs",
    config_name="evaluate",  # reuse evaluate.yaml paths structure
    version_base="1.2",
)
def main(cfg: DictConfig) -> None:
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(cfg.get("device", "auto"))
    outcome_name = cfg.get("outcome_name")  # which head to evaluate
    if outcome_name is None:
        raise ValueError(
            "outcome_name must be set for joint model evaluation; "
            "pass outcome_name=<name> on the CLI."
        )
    ckpt_path = cfg.get("ckpt_path")
    if ckpt_path is None:
        raise ValueError(
            "ckpt_path must be set for joint model evaluation; "
            "pass ckpt_path=/path/to/best.ckpt on the CLI."
        )

    # ── Load model ─────────────────────────────────────────────────────
    model = load_joint_model_from_checkpoint(
        ckpt_path,
        strict=cfg.get("strict_checkpoint_load", True),
    )
    if outcome_name not in model.outcome_names:
        raise ValueError(
            f"outcome_name='{outcome_name}' not in joint model outcomes: "
            f"{model.outcome_names}"
        )
    model = model.to(device)
    model.eval()

    # ── Load canonical evaluation cohorts ──────────────────────────────
    test_key = cfg.labels.get("test_key", "held_out")
    competing_path = cfg.paths.get("competing_outcome")
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
        outcome_name=outcome_name or cfg.get("outcome"),
        allowed_subject_ids=population_ids,
    )
    all_test_outcomes = evaluation_cohorts.survival.records
    full_fu_sids = evaluation_cohorts.fixed_horizon.subject_ids

    # Build dataset / loader over ALL test patients
    test_data = torch.load(cfg.paths.test_split, weights_only=False)
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
            "population membership. Check paths.test_split, paths.population, "
            f"paths.outcome={cfg.paths.outcome}, and test_key={test_key!r}."
        )

    vocab = torch.load(cfg.paths.vocabulary, weights_only=False)
    if model.encoder.hparams["vocab_size"] != len(vocab):
        raise ValueError(
            f"Checkpoint vocab_size={model.encoder.hparams['vocab_size']} does not "
            f"match loaded vocabulary size={len(vocab)} from {cfg.paths.vocabulary}."
        )
    background_length = int((test_data[0]["segment"] == 0).sum())
    max_len = cfg.get("max_len")
    if max_len is None:
        max_len = model.encoder.hparams["max_seqlen"]

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

    # ── Inference via the correct outcome head ─────────────────────────
    all_sids, all_labels, all_logits = [], [], []
    with torch.no_grad():
        for batch in test_loader:
            batch = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }
            logits = model.predict(batch, outcome_name)
            all_sids.append(batch["subject_id"].cpu().numpy())
            all_labels.append(batch["target"].cpu().numpy().squeeze())
            all_logits.append(logits.cpu().numpy())

    labels_all = np.concatenate(all_labels)
    logits_all = np.concatenate(all_logits)
    probs_all = 1.0 / (1.0 + np.exp(-logits_all))
    sids_all = np.concatenate(all_sids)

    # ── Survival fields (all patients) ────────────────────────────────
    times_all = np.array(
        [all_test_outcomes[int(sid)].get("time_days", float("nan")) for sid in sids_all]
    )
    events_all = np.array(
        [all_test_outcomes[int(sid)].get("event", -1) for sid in sids_all]
    )

    # Determine time horizons from config (n_hours_end_include → days)
    n_hours_end = cfg.labels.get("n_hours_end_include")
    if n_hours_end is not None:
        time_horizons = _derive_time_horizons(n_hours_end / 24.0)
    else:
        # Open-ended outcome: use calendar-year checkpoints up to observed data
        time_horizons = [365.0, 730.0]

    # ── Binary metrics (full-follow-up patients only) ─────────────────
    binary_mask = np.array([int(sid) in full_fu_sids for sid in sids_all])
    labels_bin = labels_all[binary_mask]
    probs_bin = probs_all[binary_mask]

    print(
        f"Evaluation: {len(sids_all)} total patients | "
        f"{binary_mask.sum()} with full follow-up (binary metrics) | "
        f"{(~binary_mask).sum()} censored-only (survival metrics only)"
    )

    # ── Evaluate ───────────────────────────────────────────────────────
    reported_outcome = outcome_name or cfg.get("outcome", "unknown")
    assert_cohort_parity(
        evaluation_cohorts.survival,
        sids_all,
        model_name="OPERA-joint",
        outcome_name=reported_outcome,
    )
    assert_cohort_parity(
        evaluation_cohorts.fixed_horizon,
        sids_all[binary_mask],
        model_name="OPERA-joint",
        outcome_name=reported_outcome,
    )
    print(cohort_summary(evaluation_cohorts.fixed_horizon, reported_outcome))
    print(cohort_summary(evaluation_cohorts.survival, reported_outcome))
    report = full_evaluation(
        labels_bin,
        probs_bin,
        threshold=cfg.get("threshold", 0.5),
        n_bootstrap=cfg.get("n_bootstrap", 1000),
        times=times_all,
        events=events_all,
        survival_probabilities=probs_all,
        time_horizons=time_horizons,
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
    summary = format_evaluation_summary(report)
    print(summary.encode("ascii", errors="replace").decode("ascii"))

    with open(output_dir / "evaluation_report.txt", "w", encoding="utf-8") as f:
        f.write(summary)

    np.savez(
        output_dir / "predictions.npz",
        subject_ids=sids_all,
        labels=labels_all,
        probabilities=probs_all,
        logits=logits_all,
        embeddings=np.empty((len(sids_all), 0), dtype=np.float32),
        times=times_all,
        events=events_all,
        binary_mask=binary_mask.astype(np.uint8),
    )

    result_row = build_result_row(
        cfg,
        report,
        checkpoint_path=ckpt_path,
        split=test_key,
        model_family="joint",
        training_stage="joint_finetuning",
    )

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

    plot_full_evaluation(
        labels_bin,
        probs_bin,
        output_dir=str(output_dir / "plots"),
        bootstrap_ci=report["bootstrap_ci"],
        times=times_all,
        events=events_all,
        survival_probabilities=probs_all,
        window_days=(n_hours_end / 24.0) if n_hours_end is not None else None,
        outcome_name=outcome_name or "",
    )
    print(f"\nJoint evaluation complete: {output_dir}")


if __name__ == "__main__":
    main()
