"""Evaluate a trained OPERA direct competing-risk head without refitting it."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

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
from opera.run.finetune_all_horizons import (
    _set_dataset_cohort_index,
    build_specialist_cohort_config,
    resolve_parent_cohort,
)


def _device(value: str) -> torch.device:
    if value == "auto":
        value = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(value)


def _move(batch: dict, device: torch.device) -> dict:
    return {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }


def _factorized_prototype_weights(model) -> pd.DataFrame | None:
    """Make the learned cohort-by-outcome attention routing inspectable."""
    if (
        getattr(model, "outcome_readout_mode", None)
        != "factorized_cohort_outcome_prototypes"
    ):
        return None
    readout = model.outcome_attention_readout
    reference = next(readout.parameters())
    cohort_indices = torch.arange(
        len(readout.cohort_names), device=reference.device
    )
    with torch.inference_mode():
        weights = readout.mixture_weights(cohort_indices).detach().cpu()
        gates = torch.tanh(readout.residual_gates).detach().cpu()
    rows = []
    for cohort_index, cohort in enumerate(readout.cohort_names):
        for outcome_index, outcome in enumerate(readout.outcome_names):
            for prototype_index, weight in enumerate(
                weights[cohort_index, outcome_index]
            ):
                rows.append(
                    {
                        "cohort": cohort,
                        "outcome": outcome,
                        "prototype": int(prototype_index),
                        "mixture_weight": float(weight),
                        "residual_gate": float(gates[outcome_index]),
                    }
                )
    return pd.DataFrame(rows)


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
    parser.add_argument(
        "--outcome",
        default=None,
        help="Evaluate only this outcome from a retained multi-outcome head.",
    )
    parser.add_argument(
        "--cohort-fine",
        default=None,
        help="Restrict evaluation to this fine diagnosis.",
    )
    parser.add_argument("--cohort-fine-col", default="cohort_fine")
    parser.add_argument("--export-routes", action="store_true", help="Export per-patient source route weights.")
    parser.add_argument("--remove-source", action="append", default=[], help="Diagnostic route ablation, e.g. disease:DLBCL.")
    parser.add_argument(
        "--parent-cohort",
        default=None,
        help="Grouped parent used by the source model; normally inferred.",
    )
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

    source_cohort_configs = OmegaConf.to_container(cfg.cohorts, resolve=True)
    source_outcome_configs = OmegaConf.to_container(cfg.outcomes, resolve=True)
    outcome_names = sorted(source_outcome_configs)
    if args.outcome is not None and args.outcome not in source_outcome_configs:
        raise ValueError(
            f"Outcome {args.outcome!r} is absent from the source config; "
            f"available={outcome_names}."
        )
    if bool(args.cohort_fine) != bool(args.outcome):
        raise ValueError(
            "Specialist evaluation requires --cohort-fine and --outcome together."
        )
    parent_cohort = None
    if args.cohort_fine:
        parent_cohort = resolve_parent_cohort(
            args.cohort_fine, args.parent_cohort, source_cohort_configs
        )
        cohort_configs = {
            parent_cohort: build_specialist_cohort_config(
                source_cohort_configs[parent_cohort],
                cohort_fine_col=args.cohort_fine_col,
                cohort_fine_value=args.cohort_fine,
                outcome=args.outcome,
            )
        }
        outcome_configs = {args.outcome: source_outcome_configs[args.outcome]}
        active_outcome_names = [args.outcome]
    else:
        cohort_configs = source_cohort_configs
        outcome_configs = source_outcome_configs
        active_outcome_names = outcome_names
    time_grids, probability_grids = compute_pooled_event_time_probability_grids(
        source_cohort_configs,
        source_outcome_configs,
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
        fine_names=cfg.competing_risk.get("fine_names") if cfg.competing_risk.get("readout_mode") in {"source_anchored_adapters", "unanchored_adapters"} else None,
        fine_column=str(cfg.competing_risk.get("fine_column", "cohort_fine")),
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
        [name in objective.no_competing_outcomes for name in active_outcome_names],
        dtype=torch.bool,
        device=device,
    )
    active_outcome_indices = [
        objective.outcome_index[name] for name in active_outcome_names
    ]
    checkpoint_cohorts = list(model.model_init_config.get("cohort_names", []))
    parent_index = (
        checkpoint_cohorts.index(parent_cohort)
        if parent_cohort is not None
        else None
    )

    prediction_rows: list[dict] = []
    route_rows: list[dict] = []
    source_names = []
    if model.source_adapter_readout is not None:
        source_names = [
            *("disease:" + name for name in model.source_adapter_readout.fine_names),
            *("outcome:" + name for name in model.source_adapter_readout.outcome_names),
        ]
    if (args.export_routes or args.remove_source) and not source_names:
        raise ValueError("Route export and removal require a source-adapter checkpoint")
    for cohort_name, cohort_cfg in cohort_configs.items():
        dataset = data_module._build_dataset_for_cohort(
            cohort_name, cohort_cfg, args.split
        )
        if dataset is None:
            continue
        if parent_index is not None:
            _set_dataset_cohort_index(dataset, parent_index)
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
                representations = model.get_objective_representations(batch)
                hazards, routes = model.competing_risk_log_hazards(
                    representations, batch.get("cohort_fine_index"), return_routes=True
                )
                removed_cif = None
                if args.remove_source:
                    removed_hazards = model.competing_risk_log_hazards(
                        representations, batch["cohort_fine_index"],
                        removed_sources=args.remove_source,
                    )
                    removed_cif = piecewise_exponential_cif(
                        removed_hazards[:, active_outcome_indices],
                        objective.boundaries * objective.time_scale_days,
                        horizons,
                        time_scale_days=objective.time_scale_days,
                        no_competing_outcome_mask=no_competing,
                    ).cpu().numpy()
                cif = piecewise_exponential_cif(
                    hazards[:, active_outcome_indices],
                    objective.boundaries * objective.time_scale_days,
                    horizons,
                    time_scale_days=objective.time_scale_days,
                    no_competing_outcome_mask=no_competing,
                ).cpu().numpy()
                subject_ids = batch["subject_id"].detach().cpu().numpy()
                for outcome_index, outcome in enumerate(active_outcome_names):
                    times = batch[f"time_{outcome}"].detach().cpu().numpy()
                    events = batch[f"event_{outcome}"].detach().cpu().numpy()
                    for row_index, subject_id in enumerate(subject_ids):
                        if events[row_index] < 0 or times[row_index] < 0:
                            continue
                        row = {
                            "subject_id": int(subject_id),
                            "cohort": args.cohort_fine or cohort_name,
                            "cohort_fine": args.cohort_fine,
                            "cohort_grouped": cohort_name,
                            "split": args.split,
                            "outcome": outcome,
                            "time_days": float(times[row_index]),
                            "event": int(events[row_index]),
                        }
                        for horizon_index, horizon in enumerate(horizons):
                            row[f"cif_{int(horizon)}d"] = float(
                                cif[row_index, outcome_index, horizon_index]
                            )
                            if removed_cif is not None:
                                row[f"removed_source_cif_{int(horizon)}d"] = float(
                                    removed_cif[row_index, outcome_index, horizon_index]
                                )
                        prediction_rows.append(row)
                        if args.export_routes and routes is not None:
                            target_index = active_outcome_indices[outcome_index]
                            weights = routes[row_index, target_index].detach().cpu().tolist()
                            route_rows.append({
                                "subject_id": int(subject_id),
                                "cohort": args.cohort_fine or cohort_name,
                                "split": args.split,
                                "outcome": outcome,
                                "fine_index": int(batch["cohort_fine_index"][row_index]),
                                "shared_weight": weights[0],
                                "disease_weights": json.dumps(dict(zip(source_names[:model.source_adapter_readout.n_disease], weights[1:1+model.source_adapter_readout.n_disease]))),
                                "outcome_weights": json.dumps(dict(zip(source_names[model.source_adapter_readout.n_disease:], weights[1+model.source_adapter_readout.n_disease:]))),
                            })

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
    pooled_rows = []
    for outcome, frame in predictions.groupby("outcome"):
        for horizon in horizons:
            result = compute_competing_risk_metrics_at_horizon(
                frame["time_days"].to_numpy(),
                frame["event"].to_numpy(),
                frame[f"cif_{int(horizon)}d"].to_numpy(),
                horizon,
            )
            pooled_rows.append(
                {"outcome": outcome, "horizon_days": horizon, **result}
            )
    pooled_metrics = pd.DataFrame(pooled_rows)
    outcome_families = OmegaConf.to_container(
        cfg.get("cross_outcome", {}).get("outcome_families", {}), resolve=True
    )
    outcome_to_family = {
        outcome: family
        for family, members in dict(outcome_families or {}).items()
        for outcome in members
    }
    metrics["outcome_family"] = metrics["outcome"].map(outcome_to_family)
    pooled_metrics["outcome_family"] = pooled_metrics["outcome"].map(
        outcome_to_family
    )
    family_summary = (
        pooled_metrics.dropna(subset=["outcome_family"])
        .groupby("outcome_family", as_index=False)[["cif_auc", "cif_brier"]]
        .mean()
    )
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions.to_parquet(output_dir / "direct_cr_predictions.parquet", index=False)
    if route_rows:
        pd.DataFrame(route_rows).to_parquet(output_dir / "source_route_weights.parquet", index=False)
    metrics.to_csv(output_dir / "direct_cr_metrics.csv", index=False)
    pooled_metrics.to_csv(output_dir / "direct_cr_pooled_metrics.csv", index=False)
    family_summary.to_csv(output_dir / "direct_cr_family_metrics.csv", index=False)
    prototype_weights = _factorized_prototype_weights(model)
    if prototype_weights is not None:
        prototype_weights.to_csv(
            output_dir / "cohort_outcome_prototype_weights.csv", index=False
        )
    summary = {
        "checkpoint": str(args.checkpoint),
        "split": args.split,
        "horizons_days": horizons,
        "n_prediction_rows": int(len(predictions)),
        "macro_cif_auc": float(metrics["cif_auc"].mean()),
        "macro_cif_brier": float(metrics["cif_brier"].mean()),
        "pooled_macro_cif_auc": float(pooled_metrics["cif_auc"].mean()),
        "pooled_macro_cif_brier": float(pooled_metrics["cif_brier"].mean()),
        "specialist_outcome": args.outcome,
        "cohort_fine": args.cohort_fine,
        "parent_cohort_grouped": parent_cohort,
        "route_removal_diagnostic": args.remove_source,
    }
    if not family_summary.empty:
        summary.update(
            {
                "family_macro_cif_auc": float(family_summary["cif_auc"].mean()),
                "family_macro_cif_brier": float(
                    family_summary["cif_brier"].mean()
                ),
            }
        )
    (output_dir / "direct_cr_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(metrics.to_string(index=False))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
