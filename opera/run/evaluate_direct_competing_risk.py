"""Evaluate a trained OPERA direct competing-risk head without refitting it."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from dotenv import load_dotenv
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from bonsai.functional.checkpointing import get_saved_encoder_config
from bonsai.functional.input_contract import resolve_numeric_value_control
from opera.diagnostics.representation_gradient_conflict import _load_model
from opera.evaluation.metrics import compute_competing_risk_metrics_at_horizon
from opera.modules.datamodules.ContrastiveDataModule import contrastive_collate
from opera.modules.datamodules.MultiCohortContrastiveDataModule import (
    MultiCohortContrastiveDataModule,
    compute_pooled_event_time_probability_grids,
)
from opera.modules.networks.competing_risk import piecewise_exponential_cif


def _device(value: str) -> torch.device:
    if value == "auto":
        value = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(value)


def _move(batch: dict, device: torch.device) -> dict:
    return {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config-name", required=True)
    parser.add_argument(
        "--config-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "configs",
    )
    parser.add_argument("--override", action="append", default=[])
    parser.add_argument("--split", choices=["tuning", "held_out"], default="held_out")
    parser.add_argument("--horizons", default="30,60,90,180,365,730")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    horizons = [float(value) for value in args.horizons.split(",") if value.strip()]
    if not horizons or any(value <= 0 for value in horizons):
        raise ValueError("--horizons must contain positive comma-separated days.")
    if args.batch_size <= 0 or args.num_workers < 0:
        raise ValueError("--batch-size must be positive and --num-workers non-negative.")

    load_dotenv()
    GlobalHydra.instance().clear()
    with initialize_config_dir(
        config_dir=str(args.config_dir.resolve()), version_base="1.2"
    ):
        cfg = compose(config_name=args.config_name, overrides=args.override)

    cohort_configs = OmegaConf.to_container(cfg.cohorts, resolve=True)
    outcome_configs = OmegaConf.to_container(cfg.outcomes, resolve=True)
    outcome_names = sorted(outcome_configs)
    time_grids, probability_grids = compute_pooled_event_time_probability_grids(
        cohort_configs,
        outcome_configs,
        split="train",
        require_all_configured_cells=bool(
            cfg.training.get("require_all_configured_cells", True)
        ),
    )
    vocabulary = torch.load(
        cfg.paths.vocabulary, map_location="cpu", weights_only=False
    )
    checkpoint = torch.load(
        args.checkpoint, map_location="cpu", weights_only=False
    )
    checkpoint_hparams = checkpoint.get("hyper_parameters", {})
    encoder_config = get_saved_encoder_config(checkpoint_hparams)
    max_len = int(encoder_config["max_seqlen"])
    data_module = MultiCohortContrastiveDataModule(
        cohort_configs=cohort_configs,
        outcome_configs=outcome_configs,
        predict_token_id=int(vocabulary["[CLS]"]),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        require_all_configured_cells=bool(
            cfg.training.get("require_all_configured_cells", True)
        ),
        require_min_followup_val=False,
        max_len=max_len,
        numeric_value_control=resolve_numeric_value_control(
            cfg.training.get("numeric_value_control", "inherit"), checkpoint_hparams
        ),
    )
    device = _device(args.device)
    model = _load_model(
        args.checkpoint,
        cfg,
        outcome_names,
        time_grids,
        probability_grids,
        device,
    )
    objective = model.competing_risk_loss
    if objective is None or model.competing_risk_head is None:
        raise ValueError("Checkpoint does not contain a direct competing-risk head.")
    no_competing = torch.tensor(
        [name in objective.no_competing_outcomes for name in outcome_names],
        dtype=torch.bool,
        device=device,
    )

    prediction_rows: list[dict] = []
    for cohort_name, cohort_cfg in cohort_configs.items():
        dataset = data_module._build_dataset_for_cohort(
            cohort_name, cohort_cfg, args.split
        )
        if dataset is None:
            continue
        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=args.num_workers,
            collate_fn=contrastive_collate,
        )
        with torch.inference_mode():
            for raw_batch in loader:
                batch = _move(raw_batch, device)
                pooled = model._pool(batch)
                hazards = model.competing_risk_head(pooled).reshape(
                    len(pooled), len(outcome_names), 2, objective.n_intervals
                )
                cif = piecewise_exponential_cif(
                    hazards,
                    objective.boundaries * objective.time_scale_days,
                    horizons,
                    time_scale_days=objective.time_scale_days,
                    no_competing_outcome_mask=no_competing,
                ).cpu().numpy()
                subject_ids = batch["subject_id"].detach().cpu().numpy()
                for outcome_index, outcome in enumerate(outcome_names):
                    times = batch[f"time_{outcome}"].detach().cpu().numpy()
                    events = batch[f"event_{outcome}"].detach().cpu().numpy()
                    for row_index, subject_id in enumerate(subject_ids):
                        if events[row_index] < 0 or times[row_index] < 0:
                            continue
                        row = {
                            "subject_id": int(subject_id),
                            "cohort": cohort_name,
                            "split": args.split,
                            "outcome": outcome,
                            "time_days": float(times[row_index]),
                            "event": int(events[row_index]),
                        }
                        for horizon_index, horizon in enumerate(horizons):
                            row[f"cif_{int(horizon)}d"] = float(
                                cif[row_index, outcome_index, horizon_index]
                            )
                        prediction_rows.append(row)

    predictions = pd.DataFrame(prediction_rows)
    if predictions.empty:
        raise ValueError(f"No eligible {args.split!r} predictions were produced.")
    duplicate_keys = ["subject_id", "cohort", "outcome"]
    if predictions.duplicated(duplicate_keys).any():
        raise ValueError("Duplicate subject/cohort/outcome predictions were produced.")

    metric_rows = []
    for (cohort, outcome), frame in predictions.groupby(["cohort", "outcome"]):
        for horizon in horizons:
            result = compute_competing_risk_metrics_at_horizon(
                frame["time_days"].to_numpy(),
                frame["event"].to_numpy(),
                frame[f"cif_{int(horizon)}d"].to_numpy(),
                horizon,
            )
            metric_rows.append(
                {"cohort": cohort, "outcome": outcome, "horizon_days": horizon, **result}
            )
    metrics = pd.DataFrame(metric_rows)
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions.to_parquet(output_dir / "direct_cr_predictions.parquet", index=False)
    metrics.to_csv(output_dir / "direct_cr_metrics.csv", index=False)
    summary = {
        "checkpoint": str(args.checkpoint),
        "split": args.split,
        "horizons_days": horizons,
        "n_prediction_rows": int(len(predictions)),
        "macro_cif_auc": float(metrics["cif_auc"].mean()),
        "macro_cif_brier": float(metrics["cif_brier"].mean()),
    }
    (output_dir / "direct_cr_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(metrics.to_string(index=False))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
