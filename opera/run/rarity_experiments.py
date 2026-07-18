"""Run the controlled synthetic and natural OPERA rarity experiments.

This runner reuses the canonical outcome cohorts, existing finetune/tabular
entry points, and saved ``predictions.npz`` artifacts. Synthetic and natural
rarity are intentionally planned, executed, and summarized separately.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from omegaconf import OmegaConf

from bonsai.functional.outcomes import load_split_contract
from opera.config_contracts import load_sweep_config
from opera.evaluation.cohort_flow import eligibility_file_path
from opera.evaluation.rarity import (
    assert_prediction_parity,
    bootstrap_metric_table,
    build_nested_sample_manifest,
    build_task_label_table,
    classify_task_eligibility,
    estimate_label_savings,
    macro_synthetic_summary,
    paired_bootstrap_difference,
    read_table,
    resolve_sample_sizes,
    summarize_natural_differences,
    task_count_record,
    validate_nested_manifest,
    validate_patient_splits,
)
from opera.evaluation.tasks import (
    competing_outcome_file_path,
    normalize_outcome_config,
    outcome_file_path,
)
from opera.functional.outcomes import resolve_registry_start_date
from opera.run.label_efficiency import build_tabular_fraction_cmd
from opera.run.sweep_commands import build_evaluate_cmd, build_finetune_cmd, build_prediction_evaluate_cmd


def _load_config(path: str | Path) -> dict[str, Any]:
    payload = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
    if not isinstance(payload, dict):
        raise ValueError("Rarity configuration must be a mapping.")
    if payload.get("deprecated"):
        replacement = payload["deprecated"].get(
            "replacement_config", "the current generated sweep workflow"
        )
        raise ValueError(
            f"{path} is a deprecated rarity configuration and cannot be run. "
            f"Use {replacement}; create a registry-generated matched-comparator "
            "config before re-enabling this runner."
        )
    return payload


def _write_table(frame: pd.DataFrame, stem: Path) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(stem.with_suffix(".csv"), index=False)
    frame.to_parquet(stem.with_suffix(".parquet"), index=False)


def _format(template: str, **values) -> str:
    return os.path.expandvars(str(template)).format(**values)


def _run(cmd: list[str], log_dir: Path, name: str) -> None:
    log_dir.mkdir(parents=True, exist_ok=True)
    (log_dir / f"{name}_command.json").write_text(json.dumps(cmd, indent=2))
    result = subprocess.run(cmd, capture_output=True, text=True)
    (log_dir / f"{name}_stdout.log").write_text(result.stdout or "", encoding="utf-8")
    (log_dir / f"{name}_stderr.log").write_text(result.stderr or "", encoding="utf-8")
    if result.returncode != 0:
        raise RuntimeError(
            f"{name} failed with code {result.returncode}; see {log_dir}. "
            f"Tail: {(result.stderr or '')[-1000:]}"
        )


def _find_best_checkpoint(run_dir: Path) -> Path:
    candidates = sorted(run_dir.rglob("best.ckpt"), key=lambda path: path.stat().st_mtime)
    if not candidates:
        raise FileNotFoundError(f"No best.ckpt produced under {run_dir}.")
    return candidates[-1]


def _load_binary_predictions(path: str | Path) -> pd.DataFrame:
    data = np.load(path)
    required = {"subject_ids", "probabilities"}
    if not required.issubset(data.files):
        raise ValueError(f"Prediction artifact {path} is missing {sorted(required - set(data.files))}.")
    subject_ids = data["subject_ids"]
    probabilities = data["probabilities"]
    if "labels" in data.files:
        labels = data["labels"]
        if "binary_mask" in data.files:
            mask = data["binary_mask"].astype(bool)
            subject_ids, probabilities, labels = subject_ids[mask], probabilities[mask], labels[mask]
    else:
        raise ValueError(f"Prediction artifact {path} has no fixed-horizon labels.")
    return pd.DataFrame(
        {"subject_id": subject_ids, "label": labels.astype(int), "probability": probabilities.astype(float)}
    )


def _task_contexts(config: dict, *, dry_run: bool) -> list[dict]:
    sweep = load_sweep_config(config["sweep_config"]).to_mapping()
    outcomes_cfg = normalize_outcome_config(sweep["outcomes"])
    selected_outcomes = config["outcomes"].get("primary", [])
    if config["outcomes"].get("use_all_eligible"):
        selected_outcomes = list(outcomes_cfg)
    contexts = []
    for cohort_fine, cohort_cfg in sweep["cohorts"].items():
        if cohort_fine in set(config.get("exclude_diagnoses", [])):
            continue
        data_dir = cohort_cfg["data_dir"]
        population_path = cohort_cfg.get("population_file", str(Path(data_dir) / "population_full.csv"))
        population = read_table(population_path)
        cohort_col = cohort_cfg.get("cohort_fine_col")
        if cohort_col is None:
            cohort_col = "_rarity_cohort_fine"
            population[cohort_col] = cohort_fine
        for outcome in selected_outcomes:
            if outcome not in outcomes_cfg:
                continue
            outcome_cfg = outcomes_cfg[outcome]
            if outcome in set(cohort_cfg.get("exclude_outcomes", [])):
                continue
            path = outcome_file_path(data_dir, outcome, outcome_cfg)
            eligibility = eligibility_file_path(data_dir, cohort_fine, outcome, outcome_cfg)
            competing = competing_outcome_file_path(data_dir, outcome_cfg)
            labels = build_task_label_table(
                outcomes=path,
                population=population,
                cohort_fine=cohort_fine,
                cohort_fine_col=cohort_col,
                n_hours_start_include=outcome_cfg.get("n_hours_start_include", 1),
                n_hours_end_include=outcome_cfg.get("n_hours_end_include"),
                competing_outcomes=competing,
                eligibility=str(eligibility) if eligibility else None,
                registry_start_date=resolve_registry_start_date(cohort_cfg, outcome_cfg),
                outcome_name=outcome,
            )
            validate_patient_splits(read_table(path), config["temporal_split"])
            contexts.append(
                {
                    "cohort_fine": cohort_fine,
                    "cohort_grouped": labels["cohort_grouped"].dropna().iloc[0] if not labels.empty else cohort_cfg.get("clinical_group"),
                    "clinical_group": cohort_cfg.get("clinical_group", cohort_fine),
                    "cohort_cfg": cohort_cfg,
                    "outcome": outcome,
                    "outcome_cfg": outcome_cfg,
                    "data_dir": data_dir,
                    "population_path": population_path,
                    "outcome_path": path,
                    "eligibility_path": str(eligibility) if eligibility else None,
                    "competing_path": competing,
                    "registry_start_date": resolve_registry_start_date(cohort_cfg, outcome_cfg),
                    "labels": labels,
                }
            )
            if dry_run and len(contexts) >= int(config.get("dry_run", {}).get("max_tasks", 2)):
                return contexts
    return contexts


def _eligibility_table(contexts: list[dict], config: dict) -> pd.DataFrame:
    rules = config["eligibility"]
    rows = []
    for context in contexts:
        counts = task_count_record(context["labels"])
        decision = classify_task_eligibility(
            counts,
            min_train_patients=int(rules["min_train_patients"]),
            min_train_positive=int(rules["min_train_positive"]),
            min_train_negative=int(rules["min_train_negative"]),
            primary_test_positive=int(rules["primary_test_positive"]),
            primary_test_negative=int(rules["primary_test_negative"]),
            aggregate_test_positive=int(rules["aggregate_test_positive"]),
            aggregate_test_negative=int(rules["aggregate_test_negative"]),
            max_test_indeterminate_fraction=float(rules["max_test_indeterminate_fraction"]),
        )
        rows.append(
            {
                "cohort_fine": context["cohort_fine"],
                "cohort_grouped": context["cohort_grouped"],
                "outcome": context["outcome"],
                "horizon_hours": context["outcome_cfg"].get("n_hours_end_include"),
                **counts,
                **decision,
                "eligibility_basis": "prespecified counts only; no performance inspected",
            }
        )
    return pd.DataFrame(rows)


def _sampled_outcome(
    context: dict,
    manifest: pd.DataFrame,
    sample_size: int,
    seed: int,
    path: Path,
) -> Path:
    outcomes = read_table(context["outcome_path"])
    target_ids = set(context["labels"]["subject_id"])
    outcomes = outcomes[outcomes["subject_id"].isin(target_ids)].copy()
    selected = manifest[(manifest["sample_size"] == sample_size) & (manifest["seed"] == seed)]
    train_ids = set(selected.loc[selected["split"] == "train", "subject_id"])
    tune_ids = set(selected.loc[selected["split"] == "tuning", "subject_id"])
    keep = (
        ((outcomes["split"] == "train") & outcomes["subject_id"].isin(train_ids))
        | ((outcomes["split"] == "tuning") & outcomes["subject_id"].isin(tune_ids))
        | (outcomes["split"] == "held_out")
    )
    result = outcomes[keep].copy()
    path.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(path, index=False)
    return path


def _run_neural_cell(model_name: str, model_cfg: dict, context: dict, outcome_path: Path, cell: Path, seed: int, execute: bool, overwrite: bool = False) -> Path | None:
    prediction_path = cell / "eval" / "predictions.npz"
    if prediction_path.exists() and not overwrite:
        return prediction_path
    checkpoint = _format(
        model_cfg["encoder_ckpt"], cohort=context["cohort_fine"],
        clinical_group=context["clinical_group"], outcome=context["outcome"], seed=seed,
    )
    # NOTE: unlike the live sweep (opera/run/sweep.py), this disabled legacy
    # runner uses the clinical-group label as the actual finetune population
    # selector — a known conflation, intentionally not fixed here since this
    # runner is disabled pending a rebuild (see OPERA_EXPERIMENTS.md). Do not
    # replicate this pattern in the live sweep.
    finetune = build_finetune_cmd(
        encoder_ckpt=checkpoint,
        encoder_source=model_cfg.get("encoder_source", "contrastive"),
        cohort=context["clinical_group"],
        cohort_data_dir=context["data_dir"],
        outcome_name=context["outcome"], outcome_path=str(outcome_path),
        output_dir=cell, base_config=model_cfg.get("base_config", "opera/configs/finetune.yaml"),
        n_hours_start_include=context["outcome_cfg"].get("n_hours_start_include", 1),
        n_hours_end_include=context["outcome_cfg"].get("n_hours_end_include"),
        competing_outcome_path=context["competing_path"], eligibility_path=context["eligibility_path"],
        registry_start_date=context["registry_start_date"], extra_overrides=[f"seed={seed}"],
    )
    if not execute:
        print("DRY RUN:", " ".join(finetune))
        return None
    _run(finetune, cell / "logs", "finetune")
    best = _find_best_checkpoint(cell)
    # NOTE: same known clinical-group/population conflation as above.
    evaluate = build_evaluate_cmd(
        ckpt_path=best, cohort=context["clinical_group"], cohort_data_dir=context["data_dir"],
        outcome_name=context["outcome"], outcome_path=str(outcome_path), output_dir=cell / "eval",
        encoder_source=model_cfg.get("encoder_source", "contrastive"),
        n_hours_start_include=context["outcome_cfg"].get("n_hours_start_include", 1),
        n_hours_end_include=context["outcome_cfg"].get("n_hours_end_include"),
        competing_outcome_path=context["competing_path"], eligibility_path=context["eligibility_path"],
        registry_start_date=context["registry_start_date"], model_family=model_name,
        training_stage="synthetic_rarity_finetune",
    )
    _run(evaluate, cell / "logs", "evaluate")
    return prediction_path


def _run_tabular_cell(model_name: str, model_cfg: dict, context: dict, outcome_path: Path, cell: Path, seed: int, execute: bool, n_bootstrap: int, overwrite: bool = False) -> Path | None:
    prediction_npz = cell / "eval" / "predictions.npz"
    if prediction_npz.exists() and not overwrite:
        return prediction_npz
    features = _format(
        model_cfg["features"], cohort=context["cohort_fine"],
        clinical_group=context["clinical_group"], outcome=context["outcome"], seed=seed,
    )
    train_cmd = build_tabular_fraction_cmd(
        features_path=features, outcome_parquet=str(outcome_path), output_dir=cell / "train",
        cohort=context["cohort_fine"], outcome_name=context["outcome"], seed=seed,
        models=model_cfg.get("model", "xgboost"), tune=False,
        n_hours_start_include=context["outcome_cfg"].get("n_hours_start_include", 1),
        n_hours_end_include=context["outcome_cfg"].get("n_hours_end_include"),
        eligibility_path=context["eligibility_path"], competing_outcome_path=context["competing_path"],
        registry_start_date=context["registry_start_date"],
    )
    if not execute:
        print("DRY RUN:", " ".join(train_cmd))
        return None
    _run(train_cmd, cell / "logs", "train_tabular")
    candidates = list((cell / "train").glob("*_predictions.csv"))
    if len(candidates) != 1:
        raise RuntimeError(f"Expected one tabular prediction file, found {candidates}.")
    eval_cmd = build_prediction_evaluate_cmd(
        predictions_path=str(candidates[0]), cohort=context["cohort_fine"],
        outcome_name=context["outcome"], outcome_path=str(outcome_path), model_family=model_name,
        output_dir=cell / "eval", n_hours_start_include=context["outcome_cfg"].get("n_hours_start_include", 1),
        n_hours_end_include=context["outcome_cfg"].get("n_hours_end_include"),
        competing_outcome_path=context["competing_path"], eligibility_path=context["eligibility_path"],
        registry_start_date=context["registry_start_date"], rarity_mode="synthetic", seed=seed,
        evaluation_regime="fixed_horizon",
    )
    eval_cmd.extend(["--n_bootstrap", str(n_bootstrap)])
    _run(eval_cmd, cell / "logs", "evaluate_tabular")
    return prediction_npz


def _metadata_for_sample(context: dict, manifest: pd.DataFrame, size: int, seed: int, eligibility_row: pd.Series) -> dict:
    selected = manifest[(manifest["sample_size"] == size) & (manifest["seed"] == seed)]
    train = selected[selected["split"] == "train"]
    tuning = selected[selected["split"] == "tuning"]
    return {
        "cohort_fine": context["cohort_fine"], "cohort_grouped": context["cohort_grouped"],
        "outcome": context["outcome"], "horizon_hours": context["outcome_cfg"].get("n_hours_end_include"),
        "sample_size": size, "total_target_labels_available": int(eligibility_row["n_train_determinate"]),
        "target_training_patients": int(train["subject_id"].nunique()),
        "target_tuning_patients": int(tuning["subject_id"].nunique()),
        "training_positives": int((train["label"] == 1).sum()), "training_negatives": int((train["label"] == 0).sum()),
        "tuning_positives": int((tuning["label"] == 1).sum()), "tuning_negatives": int((tuning["label"] == 0).sum()),
        "test_positives": int(eligibility_row["n_test_positive"]), "test_negatives": int(eligibility_row["n_test_negative"]),
        "test_indeterminate": int(eligibility_row["n_test_indeterminate"]), "seed": seed,
        "train_patient_hash": train["patient_id_hash"].iloc[0] if not train.empty else None,
        "tuning_patient_hash": tuning["patient_id_hash"].iloc[0] if not tuning.empty else None,
    }


def run_synthetic(config: dict, contexts: list[dict], eligibility: pd.DataFrame, *, execute: bool, dry_run: bool, overwrite: bool) -> None:
    output = Path(config["output_dir"]) / "synthetic"
    manifest_rows, skipped_rows, metric_rows, difference_rows = [], [], [], []
    seeds = list(map(int, config["sampling"]["seeds"]))
    if dry_run:
        seeds = seeds[: int(config.get("dry_run", {}).get("max_seeds", 1))]
    n_bootstrap = int(config["bootstrap_replicates"])
    if dry_run:
        n_bootstrap = int(config.get("dry_run", {}).get("bootstrap_replicates", 20))
    models = config["models"]
    comparators = config.get("comparators", ["tabular"])
    for context in contexts:
        eligibility_row = eligibility[(eligibility["cohort_fine"] == context["cohort_fine"]) & (eligibility["outcome"] == context["outcome"])].iloc[0]
        if not bool(eligibility_row["synthetic_eligible"]):
            continue
        n_available = int(eligibility_row["n_train_determinate"])
        sizes = resolve_sample_sizes(n_available, config["sampling"]["absolute_sizes"], config["sampling"].get("percentage_sizes", []))
        if dry_run:
            sizes = sizes[: int(config.get("dry_run", {}).get("max_sample_sizes", 2))]
        task_predictions: dict[tuple[int, int], dict[str, pd.DataFrame]] = {}
        fixed_test_reference: pd.DataFrame | None = None
        for seed in seeds:
            manifest, skipped = build_nested_sample_manifest(
                context["labels"], sizes=sizes, seed=seed,
                min_positive=int(config["sampling"]["min_positive"]),
                min_negative=int(config["sampling"]["min_negative"]),
                downsample_tuning=bool(config["sampling"].get("downsample_tuning", False)),
            )
            validate_nested_manifest(manifest)
            if not manifest.empty:
                manifest = manifest.assign(cohort_fine=context["cohort_fine"], cohort_grouped=context["cohort_grouped"], outcome=context["outcome"])
                manifest_rows.append(manifest)
            if not skipped.empty:
                skipped_rows.append(skipped.assign(cohort_fine=context["cohort_fine"], outcome=context["outcome"]))
            for size in sorted(manifest["sample_size"].unique()) if not manifest.empty else []:
                sample_root = output / context["cohort_fine"] / context["outcome"] / f"n_{size}" / f"seed_{seed}"
                sampled_outcome = _sampled_outcome(context, manifest, int(size), seed, sample_root / "sampled_outcome.parquet")
                metadata = _metadata_for_sample(context, manifest, int(size), seed, eligibility_row)
                predictions_by_model = {}
                for model_name, model_cfg in models.items():
                    cell = sample_root / model_name
                    artifact = (
                        _run_tabular_cell(model_name, model_cfg, context, sampled_outcome, cell, seed, execute, n_bootstrap, overwrite)
                        if model_cfg["kind"] == "tabular"
                        else _run_neural_cell(model_name, model_cfg, context, sampled_outcome, cell, seed, execute, overwrite)
                    )
                    if artifact is None or not artifact.exists():
                        continue
                    prediction = _load_binary_predictions(artifact)
                    predictions_by_model[model_name] = prediction
                    table = bootstrap_metric_table(prediction, n_bootstrap=n_bootstrap, seed=seed)
                    metric_rows.append(table.assign(model=model_name, **metadata))
                if len(predictions_by_model) > 1:
                    assert_prediction_parity(predictions_by_model)
                if predictions_by_model:
                    current = next(iter(predictions_by_model.values()))[["subject_id", "label"]].drop_duplicates().sort_values("subject_id").reset_index(drop=True)
                    if fixed_test_reference is None:
                        fixed_test_reference = current
                    elif not fixed_test_reference.equals(current):
                        raise ValueError(
                            "Fixed test patients changed across synthetic sample sizes or seeds for "
                            f"{context['cohort_fine']}/{context['outcome']}."
                        )
                task_predictions[(int(size), seed)] = predictions_by_model
                for model_name, prediction in predictions_by_model.items():
                    for comparator in comparators:
                        if comparator == model_name or comparator not in predictions_by_model:
                            continue
                        table = paired_bootstrap_difference(
                            prediction, predictions_by_model[comparator], model_name=model_name,
                            comparator_name=comparator, n_bootstrap=n_bootstrap, seed=seed,
                        )
                        difference_rows.append(table.assign(**metadata))
    manifests = pd.concat(manifest_rows, ignore_index=True) if manifest_rows else pd.DataFrame()
    skipped = pd.concat(skipped_rows, ignore_index=True) if skipped_rows else pd.DataFrame()
    metrics = pd.concat(metric_rows, ignore_index=True) if metric_rows else pd.DataFrame()
    differences = pd.concat(difference_rows, ignore_index=True) if difference_rows else pd.DataFrame()
    _write_table(manifests, output / "sampled_patient_manifest")
    _write_table(skipped, output / "skipped_sample_sizes")
    if not execute:
        print(f"Synthetic rarity plan written to {output}; no models were trained.")
        return
    _write_table(metrics, output / "per_run_metrics")
    _write_table(differences, output / "paired_model_differences")
    summary = macro_synthetic_summary(metrics, n_bootstrap=n_bootstrap) if not metrics.empty else pd.DataFrame()
    difference_summary = (
        macro_synthetic_summary(differences.rename(columns={"difference": "estimate"}), n_bootstrap=n_bootstrap)
        if not differences.empty else pd.DataFrame()
    )
    _write_table(summary, output / "aggregate_learning_curves")
    _write_table(difference_summary, output / "aggregate_relative_benefit")
    savings = estimate_label_savings(summary, comparator=comparators[0], target=config.get("label_saving", {}).get("target_auroc")) if not summary.empty else pd.DataFrame()
    _write_table(savings, output / "label_saving_estimates")


def run_natural(config: dict, contexts: list[dict], eligibility: pd.DataFrame, *, dry_run: bool) -> None:
    output = Path(config["output_dir"]) / "natural"
    metric_rows, difference_rows = [], []
    seeds = list(map(int, config["sampling"]["seeds"]))
    if dry_run:
        seeds = seeds[: int(config.get("dry_run", {}).get("max_seeds", 1))]
    n_bootstrap = int(config.get("dry_run", {}).get("bootstrap_replicates", 20) if dry_run else config["bootstrap_replicates"])
    for context in contexts:
        eligibility_row = eligibility[(eligibility["cohort_fine"] == context["cohort_fine"]) & (eligibility["outcome"] == context["outcome"])].iloc[0]
        if eligibility_row["natural_viability_tier"] == "non_evaluable":
            continue
        fixed_test_reference: pd.DataFrame | None = None
        for seed in seeds:
            predictions = {}
            for model_name, model_cfg in config["models"].items():
                template = model_cfg.get("natural_predictions")
                if not template:
                    continue
                path = Path(_format(template, cohort=context["cohort_fine"], clinical_group=context["clinical_group"], outcome=context["outcome"], model=model_name, seed=seed))
                if not path.exists():
                    print(f"Natural prediction missing: {path}")
                    continue
                prediction = _load_binary_predictions(path)
                predictions[model_name] = prediction
                table = bootstrap_metric_table(prediction, n_bootstrap=n_bootstrap, seed=seed)
                metric_rows.append(table.assign(model=model_name, cohort_fine=context["cohort_fine"], cohort_grouped=context["cohort_grouped"], outcome=context["outcome"], seed=seed, natural_viability_tier=eligibility_row["natural_viability_tier"], n_train_determinate=eligibility_row["n_train_determinate"]))
            if len(predictions) > 1:
                assert_prediction_parity(predictions)
            if predictions:
                current = next(iter(predictions.values()))[["subject_id", "label"]].drop_duplicates().sort_values("subject_id").reset_index(drop=True)
                if fixed_test_reference is None:
                    fixed_test_reference = current
                elif not fixed_test_reference.equals(current):
                    raise ValueError(
                        "Fixed test patients changed across natural-analysis seeds for "
                        f"{context['cohort_fine']}/{context['outcome']}."
                    )
            for model_name, prediction in predictions.items():
                for comparator in config.get("comparators", ["tabular"]):
                    if model_name == comparator or comparator not in predictions:
                        continue
                    table = paired_bootstrap_difference(prediction, predictions[comparator], model_name=model_name, comparator_name=comparator, n_bootstrap=n_bootstrap, seed=seed)
                    difference_rows.append(table.assign(cohort_fine=context["cohort_fine"], cohort_grouped=context["cohort_grouped"], outcome=context["outcome"], seed=seed, natural_viability_tier=eligibility_row["natural_viability_tier"], n_train_determinate=eligibility_row["n_train_determinate"]))
    metrics = pd.concat(metric_rows, ignore_index=True) if metric_rows else pd.DataFrame()
    differences = pd.concat(difference_rows, ignore_index=True) if difference_rows else pd.DataFrame()
    _write_table(metrics, output / "per_task_metrics")
    _write_table(differences, output / "paired_model_differences")
    _write_table(summarize_natural_differences(differences), output / "macro_summaries")


def regenerate_plots(config: dict, eligibility: pd.DataFrame) -> None:
    root = Path(config["output_dir"])
    from opera.visualization.rarity_plots import write_rarity_experiment_plots

    def load(stem: Path) -> pd.DataFrame:
        return pd.read_csv(stem) if stem.exists() else pd.DataFrame()

    write_rarity_experiment_plots(
        output_dir=root / "plots", eligibility=eligibility,
        synthetic_metrics=load(root / "synthetic" / "per_run_metrics.csv"),
        synthetic_summary=load(root / "synthetic" / "aggregate_learning_curves.csv"),
        synthetic_difference_summary=load(root / "synthetic" / "aggregate_relative_benefit.csv"),
        natural_differences=load(root / "natural" / "paired_model_differences.csv"),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run OPERA rarity experiments")
    parser.add_argument("--config", default="opera/configs/rarity_experiments.yaml")
    parser.add_argument("--mode", choices=["synthetic", "natural", "plots", "all"], default="all")
    parser.add_argument("--execute", action="store_true", help="Run training/evaluation; otherwise only plan cells.")
    parser.add_argument("--dry-run", action="store_true", help="Limit tasks, seeds, sizes, and bootstrap replicates.")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = _load_config(args.config)
    config["temporal_split"] = load_split_contract(config["temporal_split"]["contract"])
    root = Path(config["output_dir"])
    if args.mode == "plots":
        eligibility_path = root / "task_eligibility.csv"
        if not eligibility_path.exists():
            raise FileNotFoundError(
                f"Cached eligibility table not found: {eligibility_path}. Run a planning mode first."
            )
        regenerate_plots(config, pd.read_csv(eligibility_path))
        return
    contexts = _task_contexts(config, dry_run=args.dry_run)
    eligibility = _eligibility_table(contexts, config)
    _write_table(eligibility, root / "task_eligibility")
    if args.mode in {"synthetic", "all"}:
        run_synthetic(config, contexts, eligibility, execute=args.execute, dry_run=args.dry_run, overwrite=args.overwrite)
    if args.mode in {"natural", "all"}:
        run_natural(config, contexts, eligibility, dry_run=args.dry_run)
    if args.mode in {"plots", "all"}:
        regenerate_plots(config, eligibility)


if __name__ == "__main__":
    main()
