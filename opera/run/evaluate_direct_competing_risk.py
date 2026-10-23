"""Evaluate a trained OPERA direct competing-risk head without refitting it."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import numpy as np
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


def _tuning_source_means(model, data_module, cohort_configs, parent_index, collate,
                         device, args):
    """Each disease source's mean hazard term per target disease and outcome,
    over tuning patients; targets without tuning patients get the mean over
    all tuning patients."""
    sums, counts, total, n_total = None, None, None, 0
    n_fine = len(model.source_adapter_readout.fine_names)
    for cohort_name, cohort_cfg in cohort_configs.items():
        dataset = data_module._build_dataset_for_cohort(cohort_name, cohort_cfg, "tuning")
        if dataset is None:
            continue
        if parent_index is not None:
            _set_dataset_cohort_index(dataset, parent_index)
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                            drop_last=False, num_workers=args.num_workers,
                            collate_fn=collate)
        with torch.inference_mode():
            for raw_batch in loader:
                batch = _move(raw_batch, device)
                pooled = model.get_objective_representations(batch)
                fine = batch["cohort_fine_index"].long().reshape(-1)
                _, terms = model.borrowing_hazard_terms(pooled, fine)
                if sums is None:
                    sums = torch.zeros((n_fine, *terms.shape[1:]), device=terms.device)
                    counts = torch.zeros(n_fine, device=terms.device)
                    total = torch.zeros(terms.shape[1:], device=terms.device)
                valid = (fine >= 0) & (fine < n_fine)
                sums.index_add_(0, fine[valid], terms[valid].float())
                counts.index_add_(0, fine[valid], torch.ones(int(valid.sum()), device=terms.device))
                total += terms[valid].float().sum(0)
                n_total += int(valid.sum())
    if sums is None or n_total == 0:
        raise ValueError("Mean ablation needs tuning patients")
    overall = total / n_total
    means = torch.where(counts[:, None, None, None] > 0,
                        sums / counts.clamp_min(1)[:, None, None, None], overall[None])
    return {"means": means, "counts": counts}


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
    parser.add_argument("--export-embeddings", action="store_true",
                        help="Export pooled patient encoder embeddings for this evaluation split alongside predictions.")
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
    parser.add_argument("--export-routed-outcome", default=None,
                        help="Export shared and post-router patient vectors for this one target outcome.")
    parser.add_argument("--remove-source", action="append", default=[], help="Diagnostic route ablation, e.g. disease:DLBCL.")
    parser.add_argument("--borrowing-design", action="store_true",
                        help="Also write the CIF with disease-source terms dropped: no "
                             "borrowing, each source alone and each source left out "
                             "(source_removal/<variant>.parquet; one encoder pass).")
    parser.add_argument("--hide-prefixes", default="",
                        help="Input restriction: hide events whose code starts with these "
                             "comma-separated prefixes, e.g. LABTEST,MEDICATION.")
    parser.add_argument("--keep-last-days", type=float, default=None,
                        help="Input restriction: keep only events within this many days "
                             "of the last recorded event (background tokens are kept).")
    parser.add_argument("--content-reference", type=Path, default=None,
                        help="Timeline-only evaluation: replace every event's content by "
                             "this reference embedding (the IG baseline_INTERNAL.pt), keeping "
                             "times, order, age and subtype.")
    parser.add_argument(
        "--parent-cohort",
        default=None,
        help="Grouped parent used by the source model; normally inferred.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.export_routed_outcome and args.split != "tuning":
        raise ValueError("Routed visualization export is tuning-only; pass --split tuning")
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
    if args.export_routed_outcome and args.export_routed_outcome not in source_outcome_configs:
        raise ValueError("--export-routed-outcome must name a configured outcome")
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

    collate = contrastive_collate
    if args.hide_prefixes or args.keep_last_days is not None:
        from opera.evaluation.input_restriction import (
            hidden_code_ids, kept_code_ids, restricting_collate)
        collate = restricting_collate(
            contrastive_collate,
            hidden_ids=hidden_code_ids(vocabulary, args.hide_prefixes.split(",")),
            protected_ids=kept_code_ids(vocabulary),
            keep_last_days=args.keep_last_days)
    reference = None
    if args.content_reference is not None:
        from opera.interpretability.token_attribution import (
            non_attributable_ids, reference_embeddings)
        saved = torch.load(args.content_reference, map_location="cpu", weights_only=True)
        reference = (saved["mean"] if isinstance(saved, dict) else saved).to(device)
        fixed_ids = non_attributable_ids(vocabulary)
    prediction_rows: list[dict] = []
    embedding_ids, embedding_values = [], []
    routed_ids, routed_pre, routed_post = [], [], []
    route_rows: list[dict] = []
    source_names = []
    if model.source_adapter_readout is not None:
        source_names = [
            *("disease:" + name for name in model.source_adapter_readout.fine_names),
            *("outcome:" + name for name in model.source_adapter_readout.outcome_names),
        ]
    design_labels, design_cif = None, []
    if (args.export_routes or args.remove_source or args.export_routed_outcome
            or args.borrowing_design) and not source_names:
        raise ValueError("Route export and removal require a source-adapter checkpoint")
    borrowing_means = None
    if args.borrowing_design:
        # Mean ablation: each source's average hazard term per target disease and
        # outcome, from tuning patients only (also when evaluating held-out).
        borrowing_means = _tuning_source_means(
            model, data_module, cohort_configs, parent_index, collate, device, args)
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
            collate_fn=collate,
        )
        with torch.inference_mode():
            for raw_batch in loader:
                batch = _move(raw_batch, device)
                if reference is not None:
                    # Same timeline of care, every event an average event.
                    batch["token_embeddings"] = reference_embeddings(
                        model.encoder, batch, reference, fixed_ids)
                representations = model.get_objective_representations(batch)
                if args.export_routed_outcome:
                    if representations.ndim != 2:
                        raise ValueError("Routed vector export requires a shared pooled representation")
                    adapted, _ = model.source_adapter_readout(
                        representations, batch["cohort_fine_index"], return_weights=True
                    )
                    target_index = objective.outcome_index[args.export_routed_outcome]
                    routed_ids.append(batch["subject_id"].detach().cpu().numpy())
                    routed_pre.append(representations.detach().float().cpu().numpy())
                    routed_post.append(adapted[:, target_index].detach().float().cpu().numpy())
                if args.export_embeddings:
                    pooled = representations if representations.ndim == 2 else model.get_embeddings(batch, return_pre_projection=True)
                    if pooled.ndim != 2:
                        raise ValueError("Embedding export requires a pooled patient representation.")
                    embedding_ids.append(batch["subject_id"].detach().cpu().numpy())
                    embedding_values.append(pooled.detach().float().cpu().numpy())
                hazards, routes = model.competing_risk_log_hazards(
                    representations, batch.get("cohort_fine_index"), return_routes=True
                )
                variant_cif = None
                if args.borrowing_design:
                    from opera.modules.networks.source_anchored_readout import borrowing_design

                    design_labels, masks = borrowing_design(
                        model.source_adapter_readout.fine_names, batch["cohort_fine_index"])
                    masks = masks.to(device)
                    variants = torch.cat([
                        # Mean ablation (primary), then zero ablation (sensitivity).
                        model.competing_risk_log_hazard_masks(
                            representations, batch["cohort_fine_index"], masks,
                            means=borrowing_means["means"]),
                        model.competing_risk_log_hazard_masks(
                            representations, batch["cohort_fine_index"], masks),
                    ])
                    n_variants, n_batch = variants.shape[:2]
                    variant_cif = piecewise_exponential_cif(
                        variants.flatten(0, 1)[:, active_outcome_indices],
                        objective.boundaries * objective.time_scale_days,
                        horizons,
                        time_scale_days=objective.time_scale_days,
                        no_competing_outcome_mask=no_competing,
                    ).reshape(n_variants, n_batch, len(active_outcome_indices), -1)
                    variant_cif = variant_cif.float().cpu().numpy()
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
                        if variant_cif is not None:
                            design_cif.append(variant_cif[:, row_index, outcome_index, :])
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
    if args.export_embeddings:
        ids = np.concatenate(embedding_ids)
        values = np.concatenate(embedding_values)
        if len(np.unique(ids)) != len(ids):
            raise ValueError("Duplicate patients across cohort embedding exports; cannot create a unique projection.")
        np.savez_compressed(output_dir / "patient_embeddings.npz", subject_ids=ids,
                            embeddings=values, splits=np.full(len(ids), args.split))
        (output_dir / "patient_embeddings.metadata.json").write_text(json.dumps({
            "checkpoint": str(args.checkpoint.resolve()), "split": args.split,
            "config_name": args.config_name, "n_patients": len(ids),
            "representation": "pooled encoder (before risk head)",
        }, indent=2))
    if args.export_routed_outcome:
        ids = np.concatenate(routed_ids)
        if len(np.unique(ids)) != len(ids):
            raise ValueError("Duplicate patients in routed vector export")
        np.savez_compressed(output_dir / "routed_patient_vectors_INTERNAL.npz",
                            subject_ids=ids, shared=np.concatenate(routed_pre),
                            adapted=np.concatenate(routed_post))
        (output_dir / "routed_patient_vectors_INTERNAL.json").write_text(
            json.dumps({"checkpoint": str(args.checkpoint.resolve()),
                        "split": args.split, "target_outcome": args.export_routed_outcome,
                        "representation": "shared pooled encoder and target-specific post-router readout",
                        "n_patients": len(ids)}, indent=2) + "\n", encoding="utf-8"
        )
    if route_rows:
        pd.DataFrame(route_rows).to_parquet(output_dir / "source_route_weights.parquet", index=False)
    if args.borrowing_design:
        # One file per variant, rows aligned with direct_cr_predictions.parquet.
        values = np.stack(design_cif)  # (rows, 2 x variant, horizon)
        directory = output_dir / "source_removal"
        n_labels = len(design_labels)
        for offset, ablation in ((0, "mean"), (n_labels, "zero")):
            (directory / ablation).mkdir(parents=True, exist_ok=True)
            for index, label in enumerate(design_labels):
                frame = predictions[["subject_id", "outcome"]].copy()
                for horizon_index, horizon in enumerate(horizons):
                    frame[f"cif_{int(horizon)}d"] = values[:, offset + index, horizon_index]
                frame.to_parquet(directory / ablation / f"{label.replace(':', '_')}.parquet",
                                 index=False)
        np.savez_compressed(directory / "tuning_source_means.npz",
                            means=borrowing_means["means"].cpu().numpy(),
                            counts=borrowing_means["counts"].cpu().numpy(),
                            fine_names=np.array(model.source_adapter_readout.fine_names))
        (directory / "design.json").write_text(json.dumps({
            "variants": design_labels,
            "ablations": {
                "mean": "a switched-off source's term replaced by its tuning-set mean "
                        "for the patient's disease and outcome (primary)",
                "zero": "a switched-off source's term set to zero (sensitivity)"},
            "other_weights": "unchanged (no renormalisation)",
            "means_from": "tuning split; diseases without tuning patients use the "
                          "mean over all tuning patients",
            "horizons_days": [int(h) for h in horizons],
        }, indent=2) + "\n")
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
        "content_reference": str(args.content_reference) if args.content_reference else None,
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
