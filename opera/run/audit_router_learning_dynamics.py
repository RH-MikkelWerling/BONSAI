"""Audit source learning and projection precision on sampled training patients."""

from __future__ import annotations

import argparse
import csv
import json
from contextlib import nullcontext
from pathlib import Path

import torch
from dotenv import load_dotenv
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf
from torch.utils.data import ConcatDataset, DataLoader, Subset

from bonsai.functional.checkpointing import get_saved_encoder_config
from opera.diagnostics.representation_gradient_conflict import _load_model
from opera.modules.datamodules.ContrastiveDataModule import logical_contrastive_collate
from opera.modules.datamodules.MultiCohortContrastiveDataModule import (
    MultiCohortContrastiveDataModule, compute_pooled_event_time_probability_grids,
)


def projection_diagnostics(model, pooled, fine, eligible, *, amp_dtype=None):
    """Use the same projection for adapted and baseline vectors; expose old bias artifact."""
    readout = model.source_adapter_readout
    head = model.competing_risk_head
    n_outcome = readout.n_outcome
    weight = head.weight.reshape(n_outcome, -1, pooled.shape[-1])
    bias = head.bias.reshape(n_outcome, -1)
    mask = eligible[..., None].expand(-1, -1, weight.shape[1]).clone()
    objective = getattr(model, "competing_risk_loss", None)
    if objective is not None:
        for index, outcome in enumerate(readout.outcome_names):
            if outcome in objective.no_competing_outcomes:
                mask[:, index, objective.n_intervals:] = False
    if not bool(mask.any()):
        raise ValueError("Sample has no eligible patient/outcome pairs")
    with torch.no_grad(), torch.autocast(pooled.device.type, enabled=False):
        adapted, routes = readout(pooled.float(), fine, return_weights=True)
        direct_delta = torch.einsum("boh,okh->bok", adapted - pooled[:, None], weight.float())
    context = (torch.autocast(pooled.device.type, dtype=amp_dtype)
               if amp_dtype is not None else nullcontext())
    with torch.no_grad(), context:
        routed = torch.einsum("boh,okh->bok", adapted, weight) + bias[None]
        baseline = torch.einsum("boh,okh->bok", pooled[:, None].expand_as(adapted), weight) + bias[None]
        old_baseline = head(pooled).reshape_as(routed)
    cast_dtype = amp_dtype or torch.float32
    return {
        "true_fp32_projected_correction_mean_abs": float(direct_delta[mask].abs().mean()),
        "same_projection_correction_mean_abs": float((routed - baseline)[mask].abs().mean()),
        "fraction_eligible_logits_exactly_unchanged": float((routed == baseline)[mask].float().mean()),
        "zero_adapter_legacy_metric_artifact_mean_abs": float((baseline - old_baseline)[mask].abs().mean()),
        "fraction_eligible_embedding_coordinates_unchanged_after_cast": float(
            (adapted.to(cast_dtype) == pooled[:, None].to(cast_dtype)).float()[eligible].mean()),
        "shared_route_weight": float(routes[..., 0][eligible].mean()),
        "disease_route_mass": float(routes[..., 1:1 + readout.n_disease].sum(-1)[eligible].mean()),
        "outcome_route_mass": float(routes[..., 1 + readout.n_disease:].sum(-1)[eligible].mean()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--checkpoint", choices=("best", "last"), default="best")
    parser.add_argument("--patients", type=int, default=256)
    parser.add_argument("--physical-batch-size", type=int, default=8)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.patients < 1 or args.physical_batch_size < 1:
        raise ValueError("Patient count and physical batch size must be positive")
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    load_dotenv()
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str((args.run / "hydra").resolve()), version_base="1.2"):
        cfg = compose(config_name="config")
    cohorts = OmegaConf.to_container(cfg.cohorts, resolve=True)
    outcomes = OmegaConf.to_container(cfg.outcomes, resolve=True)
    names = sorted(outcomes)
    grids, probs = compute_pooled_event_time_probability_grids(
        cohorts, outcomes, split="train", require_all_configured_cells=True)
    checkpoint_path = args.run / f"{args.checkpoint}.ckpt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    encoder_cfg = get_saved_encoder_config(checkpoint["hyper_parameters"])
    checkpoint_epoch = checkpoint.get("epoch")
    del checkpoint
    vocabulary = torch.load(cfg.paths.vocabulary, map_location="cpu", weights_only=False)
    data = MultiCohortContrastiveDataModule(
        cohort_configs=cohorts, outcome_configs=outcomes,
        predict_token_id=int(vocabulary["[CLS]"]), batch_size=args.physical_batch_size,
        num_workers=0, require_all_configured_cells=True,
        require_min_followup_train=bool(cfg.training.get("require_min_followup_train", False)),
        max_len=int(encoder_cfg["max_seqlen"]), fine_names=list(cfg.competing_risk.fine_names),
        fine_column=str(cfg.competing_risk.get("fine_column", "cohort_fine")),
        outcome_label_cache_dir=cfg.paths.get("outcome_label_cache_dir"),
    )
    datasets = [data._build_dataset_for_cohort(name, config, "train")
                for name, config in cohorts.items()]
    dataset = ConcatDataset([item for item in datasets if item is not None and len(item)])
    generator = torch.Generator().manual_seed(42)
    indices = torch.randperm(len(dataset), generator=generator)[:args.patients].tolist()
    loader = DataLoader(Subset(dataset, indices), batch_size=len(indices), shuffle=False,
                        collate_fn=lambda batch: logical_contrastive_collate(batch, args.physical_batch_size))
    device = torch.device(args.device)
    model = _load_model(checkpoint_path, cfg, names, grids, probs, device)
    readout = model.source_adapter_readout
    if readout is None:
        raise ValueError("Audit requires a source router")
    parts, fines = [], []
    survival = {name: {"times": [], "events": []} for name in names}
    with torch.no_grad():
        for raw in next(iter(loader)):
            batch = {key: value.to(device) for key, value in raw.items()}
            parts.append(model.get_objective_representations(batch).float())
            fines.append(batch["cohort_fine_index"])
            for name in names:
                survival[name]["times"].append(batch[f"time_{name}"])
                survival[name]["events"].append(batch[f"event_{name}"])
    pooled, fine = torch.cat(parts), torch.cat(fines)
    survival = {name: {key: torch.cat(values) for key, values in fields.items()}
                for name, fields in survival.items()}
    eligible = torch.stack([
        (survival[name]["events"] >= 0) & (survival[name]["times"] >= 0)
        & model.competing_risk_loss._horizon_targets(
            survival[name]["times"], survival[name]["events"])[1].any(dim=1)
        for name in names], dim=1)
    precision = str(cfg.hardware.get("precision", "32"))
    dtype = torch.bfloat16 if "bf16" in precision else torch.float16
    adapter_epsilon = cfg.training.get("adapter_optimizer_epsilon")
    adapter_epsilon = float(cfg.training.optimizer_epsilon if adapter_epsilon is None else adapter_epsilon)
    summary = {
        "run": str(args.run), "checkpoint_epoch": checkpoint_epoch,
        "source_mode": readout.source_mode, "sampled_training_patients": len(pooled),
        "source_parameters_anchored": readout.anchored,
        "distinct_fine_diseases_in_sample": int(fine.unique().numel()),
        "fraction_unknown_fine_identity": float(((fine < 0) | (fine >= readout.n_disease)).float().mean()),
        "optimizer_epsilon": float(cfg.training.optimizer_epsilon),
        "adapter_optimizer_epsilon": adapter_epsilon,
        "curriculum": OmegaConf.to_container(cfg.competing_risk, resolve=True).get("curriculum", {}),
        "fp32": projection_diagnostics(model, pooled, fine, eligible),
        "gradient_measurement": "FP32 objective gradients on cached train representations; no optimizer step",
    }
    if device.type == "cuda":
        summary["configured_head_precision"] = projection_diagnostics(
            model, pooled, fine, eligible, amp_dtype=(dtype if "16" in precision else None))
    model.zero_grad(set_to_none=True)
    model.competing_risk_loss.set_curriculum_state(int(checkpoint_epoch or 0), training=True)
    hazards = model.competing_risk_log_hazards(pooled, fine)
    loss, _ = model.competing_risk_loss(hazards, survival)
    loss.backward()
    summary["sampled_training_loss"] = float(loss.detach())
    rows = []
    source_names = [*("disease:" + name for name in readout.fine_names),
                    *("outcome:" + name for name in readout.outcome_names)]
    with torch.no_grad():
        routes = readout.route_weights(pooled, fine)
        raw = torch.einsum("bh,srh->bsr", pooled, readout.down)
        raw = torch.einsum("bsr,shr->bsh", torch.nn.functional.gelu(raw), readout.up)
    for index, name in enumerate(source_names):
        row = {"source": name, "train_patients": int(readout.source_train_patients[index]),
               "train_events": int(readout.source_train_events[index]),
               "shrinkage": float(readout.source_shrinkage[index]),
               "raw_correction_rms": float(raw[:, index].square().mean().sqrt()),
               "mean_route_weight": float(routes[..., 1 + index][eligible].mean())}
        if index >= readout.n_disease:
            outcome_index = index - readout.n_disease
            outcome = readout.outcome_names[outcome_index]
            active = eligible[:, outcome_index]
            row["curriculum_multiplier"] = model.competing_risk_loss._curriculum_outcome_multiplier(outcome)
            row["sample_eligible_patients"] = int(active.sum())
            row["sample_target_events"] = int((survival[outcome]["events"][active] == 1).sum())
            row["mean_home_route_weight"] = (float(routes[:, outcome_index, 1 + index][active].mean())
                                               if bool(active.any()) else None)
        else:
            active = (fine == index)[:, None] & eligible
            row["curriculum_multiplier"] = None
            row["sample_eligible_patients"] = int((fine == index).sum())
            row["sample_target_events"] = None
            row["mean_home_route_weight"] = (float(routes[..., 1 + index][active].mean())
                                               if bool(active.any()) else None)
        for key in ("down", "up"):
            parameter = getattr(readout, key)
            grad = parameter.grad[index] if parameter.grad is not None else torch.zeros_like(parameter[index])
            row[f"{key}_rms"] = float(parameter[index].detach().square().mean().sqrt())
            row[f"{key}_gradient_rms"] = float(grad.square().mean().sqrt())
            row[f"{key}_fraction_gradient_below_adam_epsilon"] = float(
                (grad.abs() < adapter_epsilon).float().mean())
        rows.append(row)
    args.output_dir.mkdir(parents=True)
    with (args.output_dir / "source_diagnostics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
