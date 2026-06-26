"""
Label efficiency experiments.

Runs finetune + evaluate at multiple training set fractions to produce
learning curves. Supports either the legacy single cohort/outcome CLI or a
comma-separated list of cohort:outcome tasks.
"""

import argparse
import json
import subprocess
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml
from opera.compat.bonsai import binarize_outcomes
from opera.evaluation.tasks import (
    normalize_outcome_config,
    outcome_file_path,
    parse_task_ref,
)
from opera.functional.outcomes import (
    attach_prediction_censor_abspos,
    filter_registry_eligible_outcomes,
)


def _allocate_stratified_counts(
    train_df: pd.DataFrame, n_sample: int
) -> Dict[int, int]:
    counts = train_df["label"].value_counts().sort_index()
    labels = list(counts.index)
    if n_sample < len(labels):
        return {}

    total = float(counts.sum())
    raw = {label: n_sample * (count / total) for label, count in counts.items()}
    allocated = {
        label: min(int(counts[label]), max(1, int(np.floor(raw[label]))))
        for label in labels
    }

    while sum(allocated.values()) < n_sample:
        candidates = [
            label for label in labels if allocated[label] < int(counts[label])
        ]
        if not candidates:
            break
        label = max(candidates, key=lambda item: raw[item] - allocated[item])
        allocated[label] += 1

    while sum(allocated.values()) > n_sample:
        candidates = [label for label in labels if allocated[label] > 1]
        if not candidates:
            break
        label = min(candidates, key=lambda item: raw[item] - np.floor(raw[item]))
        allocated[label] -= 1

    return {int(label): int(n) for label, n in allocated.items() if n > 0}


def subsample_outcome_parquet(
    outcome_path: str,
    fraction: float,
    seed: int,
    output_path: str,
    split: str = "train",
    n_hours_start_include: int = 1,
    n_hours_end_include=None,
    registry_start_date: Optional[str] = None,
    cohort: Optional[str] = None,
    outcome_name: Optional[str] = None,
) -> str:
    """
    Subsample the training split of an outcome parquet.

    Validation and test splits are preserved unchanged. If a binary label column
    is already present, sampling is stratified to preserve event rate.
    """
    df = pd.read_parquet(outcome_path)
    if "index_date" in df.columns:
        df = attach_prediction_censor_abspos(df)
    elif registry_start_date not in (None, "", "null"):
        raise ValueError(
            "registry_start_date requires an index_date column in the outcome file."
        )
    df = filter_registry_eligible_outcomes(
        df,
        registry_start_date,
        cohort=cohort,
        outcome_name=outcome_name,
    )
    train_mask = df["split"] == split
    train_df = df[train_mask]
    other_df = df[~train_mask]

    n_sample = min(len(train_df), max(1, int(len(train_df) * fraction)))
    sampling_df = train_df
    sampling_label_col = "label"
    if sampling_label_col not in sampling_df.columns and {
        "subject_id",
        "outcome_date",
        "index_date",
        "censor_date",
    }.issubset(sampling_df.columns):
        derived = binarize_outcomes(
            sampling_df,
            n_hours_start_include=n_hours_start_include,
            n_hours_end_include=n_hours_end_include,
            require_min_followup=False,
            split_name=split,
        )
        sampling_df = train_df.copy()
        sampling_df["_sampling_label"] = sampling_df["subject_id"].map(
            {sid: record["label"] for sid, record in derived.items()}
        )
        sampling_label_col = "_sampling_label"

    if (
        sampling_label_col in sampling_df.columns
        and sampling_df[sampling_label_col].nunique() > 1
    ):
        allocation = _allocate_stratified_counts(
            sampling_df.rename(columns={sampling_label_col: "label"}),
            n_sample,
        )
        if allocation:
            sampled_parts = []
            for label, n_label in allocation.items():
                group = sampling_df[sampling_df[sampling_label_col] == label]
                sampled_parts.append(group.sample(n_label, random_state=seed))
            sampled = pd.concat(sampled_parts)[train_df.columns]
        else:
            warnings.warn(
                "Exact stratified subsampling is impossible for this tiny "
                f"fraction: requested n_sample={n_sample} across "
                f"{sampling_df[sampling_label_col].nunique()} classes. Falling back to "
                "unstratified sampling while preserving the requested total.",
                RuntimeWarning,
                stacklevel=2,
            )
            sampled = train_df.sample(n_sample, random_state=seed)
    else:
        sampled = train_df.sample(n_sample, random_state=seed)

    result = pd.concat([sampled, other_df])
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(output_path)
    return output_path


def outcome_split_size_metadata(
    outcome_path: str,
    registry_start_date: Optional[str] = None,
    cohort: Optional[str] = None,
    outcome_name: Optional[str] = None,
) -> Dict[str, float]:
    df = pd.read_parquet(outcome_path)
    if "index_date" in df.columns:
        df = attach_prediction_censor_abspos(df)
    elif registry_start_date not in (None, "", "null"):
        raise ValueError(
            "registry_start_date requires an index_date column in the outcome file."
        )
    df = filter_registry_eligible_outcomes(
        df,
        registry_start_date,
        cohort=cohort,
        outcome_name=outcome_name,
    )
    out: Dict[str, float] = {}
    for split_name, result_key in (
        ("train", "train"),
        ("tuning", "val"),
        ("held_out", "test"),
    ):
        split_df = df[df["split"] == split_name]
        if "subject_id" in split_df.columns:
            n_subjects = int(split_df["subject_id"].nunique())
        else:
            n_subjects = int(len(split_df))
        out[f"n_{result_key}"] = n_subjects
        if "label" in split_df.columns:
            n_events = int(split_df["label"].sum())
            out[f"n_events_{result_key}"] = n_events
            out[f"prevalence_{result_key}"] = (
                float(n_events / n_subjects) if n_subjects else float("nan")
            )
    return out


def run_finetune_and_evaluate(
    encoder_ckpt: str,
    encoder_source: str,
    cohort: str,
    outcome_name: str,
    cohort_data_dir: str,
    outcome_parquet: str,
    output_dir: Path,
    training_fraction: float,
    rarity_metadata: Optional[Dict[str, float]] = None,
    baseline_model: Optional[str] = None,
    base_config: str = "opera/configs/finetune.yaml",
    registry_start_date: Optional[str] = None,
) -> Dict:
    """Run finetune then evaluate, returning metrics.json contents."""
    ft_overrides = [
        f"encoder_ckpt={encoder_ckpt}",
        f"encoder_source={encoder_source}",
        f"dataset={cohort}",
        f"outcome={outcome_name}",
        f"paths.dir={cohort_data_dir}",
        f"paths.outcome={outcome_parquet}",
        f"labels.registry_start_date={'null' if registry_start_date is None else registry_start_date}",
        f"hydra.run.dir={output_dir}",
    ]
    ft_cmd = [
        sys.executable,
        "-m",
        "opera.run.finetune",
        f"--config-name={Path(base_config).stem}",
    ] + ft_overrides
    result = subprocess.run(ft_cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"    Finetune failed:\n{result.stderr[-500:]}")
        return {}

    ev_overrides = [
        f"run_dir={output_dir}",
        f"dataset={cohort}",
        f"outcome={outcome_name}",
        f"paths.dir={cohort_data_dir}",
        f"paths.outcome={outcome_parquet}",
        f"labels.registry_start_date={'null' if registry_start_date is None else registry_start_date}",
        f"output_dir={output_dir}/eval",
        f"+training_fraction={training_fraction}",
        "rarity.mode=synthetic",
    ]
    if baseline_model:
        ev_overrides.append(f"rarity.baseline_model={baseline_model}")
    for key, value in (rarity_metadata or {}).items():
        ev_overrides.append(f"rarity.size_metadata.{key}={value}")
    ev_cmd = [sys.executable, "-m", "opera.run.evaluate"] + ev_overrides
    result = subprocess.run(ev_cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"    Evaluate failed:\n{result.stderr[-500:]}")
        return {}

    metrics_path = output_dir / "eval" / "metrics.json"
    if metrics_path.exists():
        with open(metrics_path) as f:
            return json.load(f)
    return {}


def parse_tasks(args) -> List[Tuple[str, str]]:
    if args.tasks:
        return [parse_task_ref(item) for item in args.tasks.split(",")]
    if args.cohort and args.outcome:
        return [(args.cohort, args.outcome)]
    raise ValueError("Provide either --tasks or both --cohort and --outcome.")


def aggregate_task_results(all_task_results: Dict[str, Dict[str, Dict[float, list]]]):
    nested_summary = {}
    rows = []
    for task_key, task_results in all_task_results.items():
        cohort, outcome = task_key.split(":", 1)
        nested_summary[task_key] = {}
        for variant_name, frac_dict in task_results.items():
            nested_summary[task_key][variant_name] = {}
            for frac, aurocs in frac_dict.items():
                vals = [v for v in aurocs if not np.isnan(v)]
                if not vals:
                    continue
                ci = np.percentile(vals, [2.5, 97.5])
                agg = {
                    "mean": float(np.mean(vals)),
                    "lower": float(ci[0]),
                    "upper": float(ci[1]),
                    "std": float(np.std(vals)),
                    "n": len(vals),
                }
                nested_summary[task_key][variant_name][float(frac)] = agg
                rows.append(
                    {
                        "task": task_key,
                        "cohort": cohort,
                        "outcome": outcome,
                        "model_family": variant_name,
                        "training_fraction": frac,
                        **agg,
                    }
                )

    task_df = pd.DataFrame(rows)
    if len(task_df) == 0:
        return nested_summary, task_df, pd.DataFrame()

    pooled = task_df.groupby(["model_family", "training_fraction"], as_index=False).agg(
        median_auroc=("mean", "median"),
        lower=("mean", lambda x: float(np.percentile(x, 2.5))),
        upper=("mean", lambda x: float(np.percentile(x, 97.5))),
        n_tasks=("task", "nunique"),
    )
    return nested_summary, task_df, pooled


def compute_baseline_delta_tables(
    task_df: pd.DataFrame,
    baseline_model: Optional[str],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compute per-task and pooled label-efficiency gains over a baseline model.

    The headroom-normalized delta is `(model - baseline) / (1 - baseline)`, so
    gains at high baseline AUROC are not overstated.
    """
    if task_df.empty or not baseline_model:
        return pd.DataFrame(), pd.DataFrame()

    baseline = task_df[task_df["model_family"] == baseline_model][
        ["task", "training_fraction", "mean"]
    ].rename(columns={"mean": "baseline_mean"})
    if baseline.empty:
        return pd.DataFrame(), pd.DataFrame()

    merged = task_df.merge(
        baseline,
        on=["task", "training_fraction"],
        how="inner",
    )
    merged = merged[merged["model_family"] != baseline_model].copy()
    if merged.empty:
        return pd.DataFrame(), pd.DataFrame()

    denom = (1.0 - merged["baseline_mean"]).replace(0, np.nan)
    merged["delta_auroc"] = merged["mean"] - merged["baseline_mean"]
    merged["headroom_normalized_delta"] = merged["delta_auroc"] / denom

    pooled = merged.groupby(["model_family", "training_fraction"], as_index=False).agg(
        median_delta_auroc=("delta_auroc", "median"),
        lower_delta_auroc=("delta_auroc", lambda x: float(np.percentile(x, 2.5))),
        upper_delta_auroc=("delta_auroc", lambda x: float(np.percentile(x, 97.5))),
        median_headroom_normalized_delta=(
            "headroom_normalized_delta",
            "median",
        ),
        n_tasks=("task", "nunique"),
    )
    return merged, pooled


def _plot_summary_curves(summary: Dict[str, Dict[str, Dict]], save_path: Path) -> None:
    from opera.visualization.comparison_plots import plot_label_efficiency

    plot_label_efficiency(summary, save_path=str(save_path))


def _pooled_summary_for_plot(
    pooled_df: pd.DataFrame,
) -> Dict[str, Dict[float, Dict[str, float]]]:
    summary: Dict[str, Dict[float, Dict[str, float]]] = {}
    for _, row in pooled_df.iterrows():
        summary.setdefault(row["model_family"], {})[float(row["training_fraction"])] = {
            "mean": float(row["median_auroc"]),
            "lower": float(row["lower"]),
            "upper": float(row["upper"]),
        }
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sweep_config", required=True)
    parser.add_argument("--cohort")
    parser.add_argument("--outcome")
    parser.add_argument("--tasks", help="Comma-separated cohort:outcome pairs.")
    parser.add_argument("--output_dir", default="./results/label_efficiency")
    parser.add_argument("--fractions", default="0.01,0.02,0.05,0.1,0.25,0.5,1.0")
    parser.add_argument("--seeds", default="42,43,44")
    parser.add_argument(
        "--baseline_model",
        help="Model variant used for label-efficiency delta summaries. Defaults to the first variant.",
    )
    parser.add_argument(
        "--plot_tasks",
        help="Optional comma-separated task keys to plot as secondary panels.",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    with open(args.sweep_config) as f:
        cfg = yaml.safe_load(f)

    tasks = parse_tasks(args)
    outcomes_cfg = normalize_outcome_config(cfg["outcomes"])
    fractions = [float(x) for x in args.fractions.split(",")]
    seeds = [int(x) for x in args.seeds.split(",")]
    variants = {
        name: variant
        for name, variant in cfg["model_variants"].items()
        if "encoder_ckpt" in variant
    }
    baseline_model = args.baseline_model or next(iter(variants), None)
    all_task_results: Dict[str, Dict[str, Dict[float, list]]] = {}

    for cohort, outcome in tasks:
        task_key = f"{cohort}:{outcome}"
        cohort_cfg = cfg["cohorts"][cohort]
        data_dir = cohort_cfg["data_dir"]
        registry_start_date = cohort_cfg.get("registry_start_date")
        outcome_cfg = outcomes_cfg.get(outcome, {"outcome_file": f"{outcome}.parquet"})
        outcome_path = outcome_file_path(data_dir, outcome, outcome_cfg)
        task_output_dir = Path(args.output_dir) / cohort / outcome
        task_results: Dict[str, Dict[float, list]] = {name: {} for name in variants}

        print(f"\n=== Task {task_key} ===")
        for variant_name, variant_cfg in variants.items():
            print(f"\n-- {variant_name} --")
            for frac in fractions:
                print(f"  fraction={frac:.0%}")
                seed_aurocs = []
                for seed in seeds:
                    frac_dir = f"frac{frac:.2f}"
                    cell_dir = task_output_dir / variant_name / frac_dir / f"seed{seed}"
                    metrics_path = cell_dir / "eval" / "metrics.json"

                    if metrics_path.exists() and not args.overwrite:
                        with open(metrics_path) as f:
                            metrics = json.load(f)
                    else:
                        subsampled_path = str(cell_dir / "outcomes_subsampled.parquet")
                        subsample_outcome_parquet(
                            outcome_path,
                            frac,
                            seed,
                            subsampled_path,
                            n_hours_start_include=outcome_cfg.get(
                                "n_hours_start_include",
                                1,
                            ),
                            n_hours_end_include=outcome_cfg.get(
                                "n_hours_end_include",
                            ),
                            registry_start_date=registry_start_date,
                            cohort=cohort,
                            outcome_name=outcome,
                        )
                        metrics = run_finetune_and_evaluate(
                            encoder_ckpt=variant_cfg["encoder_ckpt"],
                            encoder_source=variant_cfg["encoder_source"],
                            cohort=cohort,
                            outcome_name=outcome,
                            cohort_data_dir=data_dir,
                            outcome_parquet=subsampled_path,
                            output_dir=cell_dir,
                            training_fraction=frac,
                            rarity_metadata=outcome_split_size_metadata(
                                subsampled_path,
                                registry_start_date=registry_start_date,
                                cohort=cohort,
                                outcome_name=outcome,
                            ),
                            baseline_model=baseline_model,
                            base_config=cfg.get(
                                "finetune_base_config",
                                "opera/configs/finetune.yaml",
                            ),
                            registry_start_date=registry_start_date,
                        )

                    auroc = metrics.get("discrimination", {}).get("auroc", float("nan"))
                    seed_aurocs.append(auroc)
                    print(f"    seed={seed}: AUROC={auroc:.3f}")
                task_results[variant_name][frac] = seed_aurocs

        all_task_results[task_key] = task_results

    nested_summary, task_df, pooled_df = aggregate_task_results(all_task_results)
    delta_df, pooled_delta_df = compute_baseline_delta_tables(task_df, baseline_model)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "label_efficiency_results.json", "w") as f:
        json.dump(
            {
                "tasks": nested_summary,
                "pooled_median": pooled_df.to_dict(orient="records"),
                "baseline_model": baseline_model,
                "pooled_delta": pooled_delta_df.to_dict(orient="records"),
            },
            f,
            indent=2,
        )
    task_df.to_csv(output_dir / "label_efficiency_summary.csv", index=False)
    pooled_df.to_csv(output_dir / "label_efficiency_pooled.csv", index=False)
    delta_df.to_csv(output_dir / "label_efficiency_delta.csv", index=False)
    pooled_delta_df.to_csv(
        output_dir / "label_efficiency_pooled_delta.csv",
        index=False,
    )
    print(f"\nResults saved to {output_dir / 'label_efficiency_results.json'}")

    try:
        pooled_summary = _pooled_summary_for_plot(pooled_df)
        _plot_summary_curves(pooled_summary, output_dir / "label_efficiency_pooled.png")

        selected_tasks = (
            args.plot_tasks.split(",")
            if args.plot_tasks
            else list(nested_summary.keys())[:4]
        )
        task_plot_dir = output_dir / "task_plots"
        task_plot_dir.mkdir(exist_ok=True)
        for task_key in selected_tasks:
            if task_key in nested_summary:
                safe_task_key = task_key.replace(":", "__")
                _plot_summary_curves(
                    nested_summary[task_key],
                    task_plot_dir / f"{safe_task_key}.png",
                )
        print(f"Main plot saved to {output_dir / 'label_efficiency_pooled.png'}")
    except Exception as exc:
        print(f"Plotting failed: {exc}")


if __name__ == "__main__":
    main()
