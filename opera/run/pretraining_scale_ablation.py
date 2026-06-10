"""
Run downstream finetune/evaluate for multiple upstream checkpoints.

This is a lightweight hook for pretraining-scale or pretraining-scope ablations.
The config intentionally reuses the sweep_config structure for cohorts/outcomes.

Example:
    python -m opera.run.pretraining_scale_ablation \
      --sweep_config opera/configs/sweep_example.yaml \
      --tasks dlbcl:mortality_1y,myeloma:aki_30d \
      --checkpoints small=/ckpts/small.ckpt,large=/ckpts/large.ckpt \
      --encoder_source pretrain \
      --output_dir ./results/pretraining_scale
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import yaml
from opera.evaluation.tasks import (
    normalize_outcome_config,
    outcome_file_path,
    parse_task_ref,
)


def parse_tasks(raw: str) -> List[Tuple[str, str]]:
    return [parse_task_ref(item) for item in raw.split(",")]


def parse_checkpoints(raw: str) -> Dict[str, str]:
    checkpoints = {}
    for item in raw.split(","):
        if "=" not in item:
            raise ValueError(f"Checkpoint {item!r} must be formatted as name=path")
        name, path = item.split("=", 1)
        checkpoints[name] = path
    return checkpoints


def run_cell(
    checkpoint_name: str,
    checkpoint_path: str,
    encoder_source: str,
    cohort: str,
    outcome: str,
    outcome_path: str,
    data_dir: str,
    output_dir: Path,
    base_config: str,
    n_hours_start_include: int,
    n_hours_end_include,
    overwrite: bool = False,
) -> Dict:
    eval_dir = output_dir / checkpoint_name / cohort / outcome / "eval"
    metrics_path = eval_dir / "metrics.json"
    if metrics_path.exists() and not overwrite:
        with open(metrics_path) as f:
            return json.load(f)

    train_dir = output_dir / checkpoint_name / cohort / outcome
    train_dir.mkdir(parents=True, exist_ok=True)
    end_value = "null" if n_hours_end_include is None else n_hours_end_include
    common = [
        f"encoder_ckpt={checkpoint_path}",
        f"encoder_source={encoder_source}",
        f"dataset={cohort}",
        f"outcome={outcome}",
        f"paths.dir={data_dir}",
        f"paths.outcome={outcome_path}",
        f"labels.n_hours_start_include={n_hours_start_include}",
        f"labels.n_hours_end_include={end_value}",
    ]

    finetune_cmd = [
        sys.executable,
        "-m",
        "opera.run.finetune",
        f"--config-name={Path(base_config).stem}",
        *common,
        f"hydra.run.dir={train_dir}",
        f"+model_family={checkpoint_name}",
        f"+pretraining_scale={checkpoint_name}",
    ]
    result = subprocess.run(finetune_cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Finetune failed for {checkpoint_name}/{cohort}/{outcome}:")
        print(result.stderr[-2000:])
        return {}

    eval_cmd = [
        sys.executable,
        "-m",
        "opera.run.evaluate",
        f"ckpt_path={train_dir}/best.ckpt",
        f"dataset={cohort}",
        f"outcome={outcome}",
        f"paths.dir={data_dir}",
        f"paths.outcome={outcome_path}",
        f"output_dir={eval_dir}",
        f"labels.n_hours_start_include={n_hours_start_include}",
        f"labels.n_hours_end_include={end_value}",
        f"+model_family={checkpoint_name}",
        f"+pretraining_scale={checkpoint_name}",
    ]
    result = subprocess.run(eval_cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Evaluate failed for {checkpoint_name}/{cohort}/{outcome}:")
        print(result.stderr[-2000:])
        return {}

    if metrics_path.exists():
        with open(metrics_path) as f:
            return json.load(f)
    return {}


def main():
    parser = argparse.ArgumentParser(description="Run pretraining scale ablations")
    parser.add_argument("--sweep_config", required=True)
    parser.add_argument(
        "--tasks", required=True, help="Comma-separated cohort:outcome pairs"
    )
    parser.add_argument(
        "--checkpoints", required=True, help="Comma-separated name=path pairs"
    )
    parser.add_argument("--encoder_source", default="pretrain")
    parser.add_argument("--output_dir", default="./results/pretraining_scale")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    with open(args.sweep_config) as f:
        cfg = yaml.safe_load(f)

    tasks = parse_tasks(args.tasks)
    checkpoints = parse_checkpoints(args.checkpoints)
    outcomes_cfg = normalize_outcome_config(cfg["outcomes"])
    base_config = cfg.get("finetune_base_config", "opera/configs/finetune.yaml")
    output_dir = Path(args.output_dir)

    rows = []
    for checkpoint_name, checkpoint_path in checkpoints.items():
        for cohort, outcome in tasks:
            cohort_cfg = cfg["cohorts"][cohort]
            outcome_cfg = outcomes_cfg.get(outcome, {})
            resolved_outcome_path = outcome_file_path(
                cohort_cfg["data_dir"],
                outcome,
                outcome_cfg,
            )
            metrics = run_cell(
                checkpoint_name=checkpoint_name,
                checkpoint_path=checkpoint_path,
                encoder_source=args.encoder_source,
                cohort=cohort,
                outcome=outcome,
                outcome_path=resolved_outcome_path,
                data_dir=cohort_cfg["data_dir"],
                output_dir=output_dir,
                base_config=base_config,
                n_hours_start_include=outcome_cfg.get("n_hours_start_include", 1),
                n_hours_end_include=outcome_cfg.get("n_hours_end_include"),
                overwrite=args.overwrite,
            )
            rows.append(
                {
                    "pretraining_scale": checkpoint_name,
                    "cohort": cohort,
                    "outcome": outcome,
                    "auroc": metrics.get("discrimination", {}).get("auroc"),
                    "auprc": metrics.get("discrimination", {}).get("auprc"),
                    "brier_score": metrics.get("calibration", {}).get("brier_score"),
                }
            )

    output_dir.mkdir(parents=True, exist_ok=True)
    import pandas as pd

    pd.DataFrame(rows).to_csv(output_dir / "pretraining_scale_cells.csv", index=False)
    print(f"Saved ablation cells to {output_dir / 'pretraining_scale_cells.csv'}")


if __name__ == "__main__":
    main()
