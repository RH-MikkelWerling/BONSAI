"""Measure cross-outcome gradient conflict at the representation level.

The contrastive objective couples patients across each batch. A gradient row
for one patient therefore reflects that patient's relationship to all patients
eligible for the same outcome. Pairwise cosine similarity is restricted to
patients jointly eligible for both outcomes, which measures alignment on the
shared patient support and is the intended diagnostic quantity.

The script never computes full-model per-outcome gradients. It runs the encoder
once, creates a detached differentiable representation leaf, and computes each
outcome gradient only with respect to that leaf.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from dotenv import load_dotenv
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig, OmegaConf

from bonsai.functional.checkpointing import (
    MODEL_INIT_CONFIG_KEY,
    clean_lightning_state_dict,
    extract_encoder_state_dict,
    get_saved_encoder_config,
    load_state_dict_checked,
)
from opera.compat.bonsai import build_bonsai_encoder
from opera.modules.datamodules.ContrastiveDataModule import (
    ContrastiveDataModule,
    compute_event_time_probability_grids,
)
from opera.modules.datamodules.MultiCohortContrastiveDataModule import (
    MultiCohortContrastiveDataModule,
    compute_pooled_event_time_probability_grids,
)
from opera.modules.networks.opera_nets import (
    OperaContrastiveModel,
    outcome_eligibility_mask,
)
from opera.modules.networks.outcome_scaling import resolve_outcome_reference_scales
from opera.diagnostics.endpoint_dependencies import (
    classify_endpoint_pairs,
    dependency_config,
)


_CR_COMPONENT_KEYS = {
    "full_likelihood": "full_likelihood_loss",
    "exposure": "exposure_loss",
    "primary_event": "primary_event_loss",
    "competing_death": "competing_event_loss",
}


def _move_to_device(value: Any, device: torch.device) -> Any:
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, dict):
        return {key: _move_to_device(item, device) for key, item in value.items()}
    if isinstance(value, list):
        return [_move_to_device(item, device) for item in value]
    if isinstance(value, tuple):
        return tuple(_move_to_device(item, device) for item in value)
    return value


def _outcome_survival(
    batch: dict[str, Any],
    outcome_names: list[str],
) -> dict[str, dict[str, torch.Tensor]]:
    result = {}
    for name in outcome_names:
        time_key = f"time_{name}"
        event_key = f"event_{name}"
        if time_key in batch and event_key in batch:
            result[name] = {
                "times": batch[time_key],
                "events": batch[event_key],
            }
    return result


def _checkpoint_weighter(
    state_dict: dict[str, torch.Tensor],
    configured: dict[str, Any],
) -> dict[str, Any]:
    settings = dict(configured)
    keys = tuple(state_dict)
    if any(key.endswith("contrastive_loss.log_sigma") for key in keys):
        settings["weighter"] = "kendall"
    elif any(key.endswith("contrastive_loss.weighter.log_sigma") for key in keys):
        settings["weighter"] = "kendall"
    elif any(key.endswith("contrastive_loss.weighter.task_logits") for key in keys):
        settings["weighter"] = "famo"
    return settings


def _load_model(
    checkpoint_path: Path,
    cfg: DictConfig,
    outcome_names: list[str],
    time_grids: dict[str, torch.Tensor],
    probability_grids: dict[str, torch.Tensor],
    device: torch.device,
    seed: int = 17,
) -> OperaContrastiveModel:
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
    hparams = checkpoint.get("hyper_parameters", {})
    model_config = get_saved_encoder_config(hparams)
    if "hidden_size" not in model_config:
        raise ValueError(
            f"Checkpoint {checkpoint_path} does not contain a saved encoder config."
        )
    model_init = dict(hparams.get(MODEL_INIT_CONFIG_KEY, {}))

    clean_state = clean_lightning_state_dict(checkpoint["state_dict"])
    configured_weighter = model_init.get(
        "cross_outcome_config",
        OmegaConf.to_container(
            cfg.get("cross_outcome", {}),
            resolve=True,
        ),
    )
    cross_outcome = resolve_outcome_reference_scales(
        _checkpoint_weighter(clean_state, configured_weighter)
    )
    competing_risk_config = model_init.get(
        "competing_risk_config",
        OmegaConf.to_container(cfg.get("competing_risk", {}), resolve=True),
    )

    dapt_store = None
    if cfg.get("dapt_embedding_store") is not None:
        dapt_store = torch.load(
            cfg.dapt_embedding_store,
            map_location="cpu",
            weights_only=False,
        )

    torch.manual_seed(seed)
    encoder = build_bonsai_encoder(model_config)
    is_opera_checkpoint = any(key.startswith("projection.") for key in clean_state)
    if not is_opera_checkpoint:
        encoder.load_state_dict(
            extract_encoder_state_dict(checkpoint["state_dict"]),
            strict=True,
        )
    model = OperaContrastiveModel(
        encoder=encoder,
        outcome_names=outcome_names,
        hidden_size=int(model_init.get("hidden_size", model_config["hidden_size"])),
        projection_hidden_dim=int(
            model_init.get(
                "projection_hidden_dim",
                cfg.model.projection_hidden_dim,
            )
        ),
        projection_dim=int(model_init.get("projection_dim", cfg.model.projection_dim)),
        temperature=float(model_init.get("temperature", cfg.model.temperature)),
        outcome_sorted_event_times=time_grids,
        outcome_event_time_probs=probability_grids,
        dapt_lambda_floor=float(
            model_init.get(
                "dapt_lambda_floor",
                cfg.model.get("dapt_lambda_floor", 0.55),
            )
        ),
        dapt_anchor_weight=float(
            model_init.get(
                "dapt_anchor_weight",
                cfg.model.get("dapt_anchor_weight", 0.2),
            )
        ),
        competing_event_weight=float(
            model_init.get(
                "competing_event_weight",
                cfg.model.get("competing_event_weight", 0.0),
            )
        ),
        effective_pair_normalization=bool(
            model_init.get(
                "effective_pair_normalization",
                cfg.model.get("effective_pair_normalization", True),
            )
        ),
        cross_outcome_config=cross_outcome,
        projection_mode=str(
            model_init.get(
                "projection_mode", cfg.model.get("projection_mode", "shared")
            )
        ),
        freeze_encoder=bool(
            model_init.get(
                "freeze_encoder",
                cfg.model.get("freeze_encoder", False),
            )
        ),
        pooling=str(
            model_init.get(
                "pooling",
                cfg.model.get("pooling", "cls_last"),
            )
        ),
        dapt_embedding_store=dapt_store,
        competing_risk_config=competing_risk_config,
    )
    if is_opera_checkpoint:
        load_state_dict_checked(model, clean_state, strict=True)
    model.to(device)
    model.eval()
    return model


def _build_data(
    cfg: DictConfig,
    batch_size: int | None,
    logical_batch_size: int | None,
    num_workers: int,
    max_len: int,
):
    outcome_configs = OmegaConf.to_container(cfg.outcomes, resolve=True)
    outcome_names = sorted(outcome_configs)
    chosen_batch_size = int(batch_size or cfg.training.batch_size)

    if "cohorts" in cfg:
        cohort_configs = OmegaConf.to_container(cfg.cohorts, resolve=True)
        require_all = bool(cfg.training.get("require_all_configured_cells", True))
        time_grids, probability_grids = compute_pooled_event_time_probability_grids(
            cohort_configs,
            outcome_configs,
            split="train",
            require_all_configured_cells=require_all,
        )
        first_cohort = next(iter(cohort_configs.values()))
        vocabulary = torch.load(
            Path(first_cohort["data_dir"]) / "vocabulary.pt",
            map_location="cpu",
            weights_only=False,
        )
        data_module = MultiCohortContrastiveDataModule(
            cohort_configs=cohort_configs,
            outcome_configs=outcome_configs,
            predict_token_id=vocabulary["[CLS]"],
            batch_size=chosen_batch_size,
            logical_batch_size=logical_batch_size,
            num_workers=num_workers,
            require_all_configured_cells=require_all,
            max_len=max_len,
        )
    else:
        time_grids, probability_grids = compute_event_time_probability_grids(
            outcome_configs,
            split="train",
        )
        vocabulary = torch.load(
            cfg.paths.vocabulary,
            map_location="cpu",
            weights_only=False,
        )
        data_module = ContrastiveDataModule(
            path_train_data=cfg.paths.train_split,
            path_val_data=cfg.paths.val_split,
            path_population=cfg.paths.population,
            outcome_configs=outcome_configs,
            predict_token_id=vocabulary["[CLS]"],
            batch_size=chosen_batch_size,
            num_workers=num_workers,
            max_len=max_len,
        )

    data_module.setup("fit")
    return data_module, outcome_names, time_grids, probability_grids


def _shared_checkpoint_max_len(checkpoints: list[Path]) -> int:
    lengths = []
    for checkpoint_path in checkpoints:
        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=False,
        )
        model_config = get_saved_encoder_config(checkpoint.get("hyper_parameters", {}))
        length = model_config.get("max_seqlen")
        if length is None:
            raise ValueError(f"Checkpoint {checkpoint_path} has no max_seqlen.")
        lengths.append(int(length))
    return min(lengths)


def _gradient_cosine(
    first: torch.Tensor,
    second: torch.Tensor,
    joint_mask: torch.Tensor,
) -> float | None:
    first_flat = first[joint_mask].reshape(-1)
    second_flat = second[joint_mask].reshape(-1)
    denominator = first_flat.norm() * second_flat.norm()
    if not torch.isfinite(denominator) or denominator.item() <= 1e-12:
        return None
    return float(torch.dot(first_flat, second_flat).div(denominator).item())


def _cohort_by_subject(data_module) -> dict[int, str]:
    dataset = getattr(data_module, "train_dataset", None)
    datasets = getattr(dataset, "datasets", None)
    labels = getattr(data_module, "train_cohort_labels", None)
    if datasets is None or labels is None:
        return {}
    result: dict[int, str] = {}
    offset = 0
    for child in datasets:
        label = str(labels[offset])
        for subject in child.subjects:
            result[int(subject["subject_id"])] = label
        offset += len(child)
    return result


def _followup_strata(times: torch.Tensor, events: torch.Tensor) -> np.ndarray:
    time_values = times.detach().cpu().numpy()
    event_values = events.detach().cpu().numpy().astype(int)
    valid_times = time_values[np.isfinite(time_values) & (time_values >= 0)]
    cuts = (
        np.unique(np.quantile(valid_times, [0.25, 0.5, 0.75]))
        if valid_times.size >= 4
        else np.array([])
    )
    return event_values * 10 + np.digitize(time_values, cuts)


def _permuted_gradient(
    gradient: torch.Tensor,
    valid: torch.Tensor,
    times: torch.Tensor,
    events: torch.Tensor,
    cohorts: np.ndarray,
    rng: np.random.Generator,
) -> torch.Tensor:
    """Permute rows within cohort, event-status, and follow-up strata."""
    result = gradient.clone()
    strata = _followup_strata(times, events)
    valid_np = valid.detach().cpu().numpy().astype(bool)
    for cohort in np.unique(cohorts):
        for stratum in np.unique(strata[valid_np]):
            indices = np.flatnonzero(valid_np & (cohorts == cohort) & (strata == stratum))
            if indices.size > 1:
                shuffled = rng.permutation(indices)
                target = torch.as_tensor(indices, device=result.device)
                source = torch.as_tensor(shuffled, device=result.device)
                result[target] = gradient[source]
    return result


def _write_component_outputs(
    output_dir: Path,
    records: list[dict[str, Any]],
    outcome_names: list[str],
    dependencies: list[dict[str, Any]],
) -> None:
    dependency_frame = pd.DataFrame(dependencies)
    dependency_frame.to_csv(output_dir / "endpoint_dependencies.csv", index=False)
    frame = pd.DataFrame(records)
    if frame.empty:
        return
    frame = frame.merge(dependency_frame, on=["outcome_a", "outcome_b"], how="left")
    frame.to_csv(output_dir / "gradient_components_batches.csv", index=False)
    aggregate = frame.groupby(
        ["gradient_component", "outcome_a", "outcome_b", "relationship", "scaffold_eligible"],
        as_index=False,
        dropna=False,
    ).agg(
        mean_cosine=("cosine", "mean"),
        mean_null_cosine=("null_cosine", "mean"),
        mean_excess_alignment=("excess_alignment", "mean"),
        mean_joint_support=("joint_support", "mean"),
        supported_batches=("cosine", "count"),
    )
    aggregate.to_csv(output_dir / "gradient_components.csv", index=False)
    atomic = aggregate[
        (aggregate["gradient_component"] == "full_likelihood")
        & aggregate["scaffold_eligible"].fillna(False)
    ].sort_values("mean_excess_alignment", ascending=False)
    atomic.to_csv(output_dir / "atomic_scaffold_candidates.csv", index=False)
    atomic_matrix = pd.DataFrame(np.nan, index=outcome_names, columns=outcome_names)
    for row in atomic.itertuples():
        atomic_matrix.loc[row.outcome_a, row.outcome_b] = row.mean_excess_alignment
        atomic_matrix.loc[row.outcome_b, row.outcome_a] = row.mean_excess_alignment
    atomic_matrix.to_csv(output_dir / "atomic_scaffold_matrix.csv")
    for component, component_frame in aggregate.groupby("gradient_component"):
        matrix = pd.DataFrame(np.nan, index=outcome_names, columns=outcome_names)
        for row in component_frame.itertuples():
            matrix.loc[row.outcome_a, row.outcome_b] = row.mean_cosine
            matrix.loc[row.outcome_b, row.outcome_a] = row.mean_cosine
        np.fill_diagonal(matrix.values, 1.0)
        matrix.to_csv(output_dir / f"gradient_cosine_{component}.csv")


def _checkpoint_output_dir(base: Path, checkpoint: Path, index: int) -> Path:
    safe_stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", checkpoint.stem)
    output = base / f"{index:02d}_{safe_stem}"
    output.mkdir(parents=True, exist_ok=True)
    return output


def _pair_records(
    cosine_matrix: np.ndarray,
    support_matrix: np.ndarray,
    supported_batches: np.ndarray,
    outcome_names: list[str],
    n_batches: int,
    min_overlap: int,
) -> list[dict[str, Any]]:
    records = []
    minimum_batches = max(2, math.ceil(n_batches / 4))
    for first in range(len(outcome_names)):
        for second in range(first + 1, len(outcome_names)):
            cosine = cosine_matrix[first, second]
            if not np.isfinite(cosine):
                continue
            mean_support = float(support_matrix[first, second])
            n_supported = int(supported_batches[first, second])
            records.append(
                {
                    "outcome_a": outcome_names[first],
                    "outcome_b": outcome_names[second],
                    "mean_cosine": float(cosine),
                    "mean_joint_support": mean_support,
                    "supported_batches": n_supported,
                    "small_support": bool(
                        mean_support < 2 * min_overlap or n_supported < minimum_batches
                    ),
                }
            )
    return records


def _write_heatmap(
    cosine_matrix: np.ndarray,
    outcome_names: list[str],
    event_rates: np.ndarray,
    path: Path,
) -> None:
    sortable_rates = np.where(np.isfinite(event_rates), event_rates, np.inf)
    order = np.argsort(sortable_rates)
    ordered = cosine_matrix[np.ix_(order, order)]
    ordered_names = [outcome_names[index] for index in order]

    figure_size = max(8.0, min(24.0, 0.42 * len(outcome_names)))
    fig, ax = plt.subplots(figsize=(figure_size, figure_size))
    image = ax.imshow(ordered, vmin=-1.0, vmax=1.0, cmap="coolwarm")
    ax.set_xticks(np.arange(len(ordered_names)))
    ax.set_yticks(np.arange(len(ordered_names)))
    ax.set_xticklabels(ordered_names, rotation=90, fontsize=7)
    ax.set_yticklabels(ordered_names, fontsize=7)
    ax.set_title("Representation gradient cosine, ordered by event rate")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def _write_outputs(
    output_dir: Path,
    checkpoint: Path,
    outcome_names: list[str],
    cosine_sum: np.ndarray,
    cosine_count: np.ndarray,
    support_sum: np.ndarray,
    supported_batches: np.ndarray,
    event_sum: np.ndarray,
    event_count: np.ndarray,
    n_batches: int,
    min_overlap: int,
    negative_threshold: float,
    batch_pair_records: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    cosine_matrix = np.full_like(cosine_sum, np.nan, dtype=float)
    np.divide(
        cosine_sum,
        cosine_count,
        out=cosine_matrix,
        where=cosine_count > 0,
    )
    support_matrix = support_sum / max(n_batches, 1)
    event_rates = np.full_like(event_sum, np.nan, dtype=float)
    np.divide(
        event_sum,
        event_count,
        out=event_rates,
        where=event_count > 0,
    )

    np.save(output_dir / "gradient_cosine.npy", cosine_matrix)
    np.save(output_dir / "joint_support.npy", support_matrix)
    pd.DataFrame(
        cosine_matrix,
        index=outcome_names,
        columns=outcome_names,
    ).to_csv(output_dir / "gradient_cosine.csv")
    pd.DataFrame(
        support_matrix,
        index=outcome_names,
        columns=outcome_names,
    ).to_csv(output_dir / "joint_support.csv")
    pd.DataFrame(
        {
            "outcome": outcome_names,
            "event_rate": event_rates,
            "event_count": event_sum,
            "eligible_count": event_count,
        }
    ).to_csv(output_dir / "event_rates.csv", index=False)
    batch_columns = [
        "checkpoint",
        "batch_index",
        "outcome_a",
        "outcome_b",
        "cosine",
        "joint_support",
        "meets_min_overlap",
    ]
    pd.DataFrame(
        batch_pair_records or [],
        columns=batch_columns,
    ).to_csv(output_dir / "gradient_pair_batches.csv", index=False)
    _write_heatmap(
        cosine_matrix,
        outcome_names,
        event_rates,
        output_dir / "gradient_cosine_heatmap.png",
    )

    records = _pair_records(
        cosine_matrix,
        support_matrix,
        supported_batches,
        outcome_names,
        n_batches,
        min_overlap,
    )
    cosines = np.array([record["mean_cosine"] for record in records], dtype=float)
    summary = {
        "checkpoint": str(checkpoint),
        "n_batches": n_batches,
        "n_outcomes": len(outcome_names),
        "n_supported_pairs": len(records),
        "min_overlap": min_overlap,
        "negative_threshold": negative_threshold,
        "batch_pair_artifact": "gradient_pair_batches.csv",
        "fraction_below_zero": (
            float(np.mean(cosines < 0.0)) if cosines.size else None
        ),
        "fraction_below_negative_threshold": (
            float(np.mean(cosines < negative_threshold)) if cosines.size else None
        ),
        "most_conflicting": sorted(
            records,
            key=lambda record: record["mean_cosine"],
        )[:20],
        "most_aligned": sorted(
            records,
            key=lambda record: record["mean_cosine"],
            reverse=True,
        )[:20],
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    return summary


def diagnose_checkpoint(
    checkpoint: Path,
    cfg: DictConfig,
    data_module,
    outcome_names: list[str],
    time_grids: dict[str, torch.Tensor],
    probability_grids: dict[str, torch.Tensor],
    output_dir: Path,
    *,
    n_batches: int,
    min_overlap: int,
    negative_threshold: float,
    device: torch.device,
    seed: int,
    objective: str = "contrastive",
    gradient_components: tuple[str, ...] = ("full_likelihood",),
    null_permutations: int = 0,
    endpoint_dependency_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run the representation-gradient diagnostic for one checkpoint."""
    model = _load_model(
        checkpoint,
        cfg,
        outcome_names,
        time_grids,
        probability_grids,
        device,
        seed,
    )
    n_outcomes = len(outcome_names)
    cosine_sum = np.zeros((n_outcomes, n_outcomes), dtype=float)
    cosine_count = np.zeros((n_outcomes, n_outcomes), dtype=np.int64)
    support_sum = np.zeros((n_outcomes, n_outcomes), dtype=float)
    supported_batches = np.zeros((n_outcomes, n_outcomes), dtype=np.int64)
    event_sum = np.zeros(n_outcomes, dtype=float)
    event_count = np.zeros(n_outcomes, dtype=float)
    batch_pair_records: list[dict[str, Any]] = []
    outcome_batch_records: list[dict[str, Any]] = []
    projection_geometry_records: list[dict[str, Any]] = []
    component_pair_records: list[dict[str, Any]] = []
    composites, aliases = dependency_config(endpoint_dependency_config)
    endpoint_dependencies = classify_endpoint_pairs(
        outcome_names, composites=composites, aliases=aliases
    )
    family_by_outcome = {
        outcome: family
        for family, members in dict(
            cfg.get("cross_outcome", {}).get("outcome_families", {})
        ).items()
        for outcome in members
    }

    torch.manual_seed(seed)
    loader = data_module.train_dataloader()
    cohort_lookup = _cohort_by_subject(data_module)
    null_rng = np.random.default_rng(seed)
    processed_batches = 0
    for batch_index, cpu_batch in enumerate(loader):
        if batch_index >= n_batches:
            break
        microbatches = cpu_batch if isinstance(cpu_batch, list) else [cpu_batch]
        embedding_parts = []
        subject_parts = []
        survival_parts = {name: {"times": [], "events": []} for name in outcome_names}
        with torch.no_grad():
            for cpu_microbatch in microbatches:
                microbatch = _move_to_device(cpu_microbatch, device)
                embedding_parts.append(
                    model.get_embeddings(microbatch, return_pre_projection=True)
                )
                subject_parts.append(microbatch["subject_id"])
                micro_survival = _outcome_survival(microbatch, outcome_names)
                for name, fields in micro_survival.items():
                    survival_parts[name]["times"].append(fields["times"])
                    survival_parts[name]["events"].append(fields["events"])
        pooled = torch.cat(embedding_parts)
        subject_ids = torch.cat(subject_parts)
        survival = {
            name: {key: torch.cat(values) for key, values in fields.items()}
            for name, fields in survival_parts.items()
            if fields["times"]
        }

        with torch.enable_grad():
            representation = pooled.detach().clone().requires_grad_(True)
            projected = model.projection(representation)
            projected_spaces = (
                {"shared": projected}
                if isinstance(projected, torch.Tensor)
                else projected
            )
            for space, values in projected_spaces.items():
                similarities = values @ values.T
                off_diagonal = ~torch.eye(
                    values.shape[0], dtype=torch.bool, device=values.device
                )
                projection_geometry_records.append(
                    {
                        "checkpoint": str(checkpoint),
                        "batch_index": batch_index,
                        "projection_space": space,
                        "coordinate_std_mean": float(
                            values.std(dim=0, unbiased=False).mean().detach().item()
                        ),
                        "coordinate_std_min": float(
                            values.std(dim=0, unbiased=False).min().detach().item()
                        ),
                        "mean_off_diagonal_cosine": float(
                            similarities[off_diagonal].mean().detach().item()
                        ),
                        "std_off_diagonal_cosine": float(
                            similarities[off_diagonal]
                            .std(unbiased=False)
                            .detach()
                            .item()
                        ),
                    }
                )
            if objective == "competing_risk":
                if model.competing_risk_head is None:
                    raise ValueError(
                        f"Checkpoint {checkpoint} has no competing-risk head."
                    )
                log_hazards = model.competing_risk_head(representation).reshape(
                    representation.shape[0],
                    len(outcome_names),
                    2,
                    model.competing_risk_loss.n_intervals,
                )
                _, _, terms = model.competing_risk_loss(
                    log_hazards, survival, return_per_outcome=True
                )
                active_names = list(terms)
            else:
                terms, _ = model.contrastive_loss.compute_per_outcome_losses(
                    projected,
                    survival,
                    subject_ids=subject_ids,
                    dapt_embedding_store=model.dapt_embedding_store,
                )
                active_names = [
                    name
                    for name, term in terms.items()
                    if float(term["n_effective_pairs"].detach().item()) > 0.0
                ]
            requested_components = (
                gradient_components if objective == "competing_risk" else ("full_likelihood",)
            )
            gradient_jobs = [
                (component, name)
                for component in requested_components
                for name in active_names
            ]
            gradients_by_component: dict[str, dict[str, torch.Tensor]] = {
                component: {} for component in requested_components
            }
            for term_index, (component, name) in enumerate(gradient_jobs):
                loss_key = (
                    _CR_COMPONENT_KEYS[component]
                    if objective == "competing_risk"
                    else "loss"
                )
                gradient = torch.autograd.grad(
                    terms[name][loss_key],
                    representation,
                    retain_graph=term_index < len(gradient_jobs) - 1,
                )[0]
                gradients_by_component[component][name] = gradient.detach()
            gradients = gradients_by_component[requested_components[0]]
            null_gradient_cache: dict[
                tuple[str, str], list[torch.Tensor]
            ] = {}
            if null_permutations > 0:
                cohorts = np.asarray([
                    cohort_lookup.get(int(sid), "unknown")
                    for sid in subject_ids.detach().cpu().tolist()
                ])
                for component, component_gradients in gradients_by_component.items():
                    for name, gradient in component_gradients.items():
                        outcome = survival[name]
                        valid = outcome_eligibility_mask(
                            outcome["times"], outcome["events"]
                        )
                        null_gradient_cache[(component, name)] = [
                            _permuted_gradient(
                                gradient, valid, outcome["times"], outcome["events"],
                                cohorts, null_rng,
                            )
                            for _ in range(null_permutations)
                        ]

            for name, term in terms.items():
                outcome = survival[name]
                valid = outcome_eligibility_mask(outcome["times"], outcome["events"])
                events = outcome["events"][valid]
                record = {
                        "checkpoint": str(checkpoint),
                        "objective": objective,
                        "batch_index": batch_index,
                        "outcome": name,
                        "family": family_by_outcome.get(name, "unassigned"),
                        "n_valid": int(valid.sum().item()),
                        "n_event": int((events == 1).sum().item()),
                        "n_censored": int((events == 0).sum().item()),
                        "n_competing": int((events == 2).sum().item()),
                        "event_rate": float((events == 1).float().mean().item()),
                        "loss": float(term["loss"].detach().item()),
                        "gradient_norm": float(gradients[name].norm().item()),
                    }
                if objective == "contrastive":
                    record.update({
                        "target_entropy": float(term["target_entropy"].detach().item()),
                        "kl": float(term["excess_loss"].detach().item()),
                        "headroom": float(term["contrastive_headroom"].detach().item()),
                        "relative_kl": float(
                            (
                                term["excess_loss"]
                                / term["contrastive_headroom"].clamp_min(1e-6)
                            )
                            .detach()
                            .item()
                        ),
                        "n_effective_pairs": float(
                            term["n_effective_pairs"].detach().item()
                        ),
                        "effective_pair_fraction": float(
                            term["effective_pair_fraction"].detach().item()
                        ),
                    })
                outcome_batch_records.append(record)

        for index, name in enumerate(outcome_names):
            outcome = survival.get(name)
            if outcome is None:
                continue
            valid = outcome_eligibility_mask(
                outcome["times"],
                outcome["events"],
            )
            event_sum[index] += float(((outcome["events"] == 1) & valid).sum().item())
            event_count[index] += float(valid.sum().item())

        for first, first_name in enumerate(outcome_names):
            for second in range(first, n_outcomes):
                second_name = outcome_names[second]
                first_survival = survival.get(first_name)
                second_survival = survival.get(second_name)
                if first_survival is None or second_survival is None:
                    if second != first:
                        batch_pair_records.append(
                            {
                                "checkpoint": str(checkpoint),
                                "batch_index": batch_index,
                                "outcome_a": first_name,
                                "outcome_b": second_name,
                                "cosine": None,
                                "joint_support": 0,
                                "meets_min_overlap": False,
                            }
                        )
                    continue
                first_valid = outcome_eligibility_mask(
                    first_survival["times"],
                    first_survival["events"],
                )
                second_valid = outcome_eligibility_mask(
                    second_survival["times"],
                    second_survival["events"],
                )
                joint = first_valid & second_valid
                support = int(joint.sum().item())
                support_sum[first, second] += support
                support_sum[second, first] += support if second != first else 0
                cosine = None
                if support > 0 and first_name in gradients and second_name in gradients:
                    cosine = _gradient_cosine(
                        gradients[first_name],
                        gradients[second_name],
                        joint,
                    )
                if second != first:
                    batch_pair_records.append(
                        {
                            "checkpoint": str(checkpoint),
                            "batch_index": batch_index,
                            "outcome_a": first_name,
                            "outcome_b": second_name,
                            "cosine": cosine,
                            "joint_support": support,
                            "meets_min_overlap": support >= min_overlap,
                        }
                    )
                    for component, component_gradients in gradients_by_component.items():
                        component_cosine = None
                        if (
                            support > 0
                            and first_name in component_gradients
                            and second_name in component_gradients
                        ):
                            component_cosine = _gradient_cosine(
                                component_gradients[first_name],
                                component_gradients[second_name],
                                joint,
                            )
                        null_values: list[float] = []
                        if component_cosine is not None and null_permutations > 0:
                            first_nulls = null_gradient_cache[(component, first_name)]
                            second_nulls = null_gradient_cache[(component, second_name)]
                            for first_null, second_null in zip(first_nulls, second_nulls):
                                value = _gradient_cosine(first_null, second_null, joint)
                                if value is not None:
                                    null_values.append(value)
                        null_mean = float(np.mean(null_values)) if null_values else 0.0
                        component_pair_records.append({
                            "checkpoint": str(checkpoint),
                            "batch_index": batch_index,
                            "gradient_component": component,
                            "outcome_a": first_name,
                            "outcome_b": second_name,
                            "cosine": component_cosine,
                            "null_cosine": null_mean if null_permutations > 0 else np.nan,
                            "excess_alignment": (
                                component_cosine - null_mean
                                if component_cosine is not None and null_permutations > 0
                                else np.nan
                            ),
                            "joint_support": support,
                            "null_permutations": null_permutations,
                        })
                if support < min_overlap or cosine is None:
                    continue
                cosine_sum[first, second] += cosine
                cosine_count[first, second] += 1
                supported_batches[first, second] += 1
                if second != first:
                    cosine_sum[second, first] += cosine
                    cosine_count[second, first] += 1
                    supported_batches[second, first] += 1
        processed_batches += 1

    summary = _write_outputs(
        output_dir,
        checkpoint,
        outcome_names,
        cosine_sum,
        cosine_count,
        support_sum,
        supported_batches,
        event_sum,
        event_count,
        processed_batches,
        min_overlap,
        negative_threshold,
        batch_pair_records,
    )
    _write_component_outputs(
        output_dir,
        component_pair_records,
        outcome_names,
        endpoint_dependencies,
    )
    outcome_frame = pd.DataFrame(outcome_batch_records)
    outcome_frame.to_csv(output_dir / "outcome_batch_diagnostics.csv", index=False)
    if not outcome_frame.empty:
        numeric_columns = [
            column
            for column in outcome_frame.select_dtypes(include=[np.number]).columns
            if column != "batch_index"
        ]
        outcome_summary = outcome_frame.groupby(["outcome", "family"], as_index=False)[
            numeric_columns
        ].median()
        outcome_summary.to_csv(output_dir / "outcome_diagnostics.csv", index=False)
        family_summary = outcome_summary.groupby("family", as_index=False)[
            numeric_columns
        ].mean()
        family_summary.to_csv(output_dir / "family_diagnostics.csv", index=False)
        if "kl" in outcome_summary:
            references = {
                row["outcome"]: row["kl"]
                for row in outcome_summary.to_dict(orient="records")
                if np.isfinite(row["kl"]) and row["kl"] > 0
            }
            (output_dir / "normalization_references.json").write_text(
                json.dumps(
                    {
                        "scale": "median_initial_kl",
                        "checkpoint": str(checkpoint),
                        "outcome_reference_scales": references,
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
    geometry_frame = pd.DataFrame(projection_geometry_records)
    geometry_frame.to_csv(output_dir / "projection_geometry_batches.csv", index=False)
    if not geometry_frame.empty:
        geometry_frame.groupby("projection_space", as_index=False)[
            [
                "coordinate_std_mean",
                "coordinate_std_min",
                "mean_off_diagonal_cosine",
                "std_off_diagonal_cosine",
            ]
        ].mean().to_csv(output_dir / "projection_geometry.csv", index=False)
    pair_frame = pd.DataFrame(batch_pair_records)
    if not pair_frame.empty and family_by_outcome:
        pair_frame["family_a"] = pair_frame["outcome_a"].map(family_by_outcome)
        pair_frame["family_b"] = pair_frame["outcome_b"].map(family_by_outcome)
        pair_frame["relationship"] = np.where(
            pair_frame["family_a"] == pair_frame["family_b"],
            "within_family",
            "between_family",
        )
        pair_frame.to_csv(output_dir / "gradient_pair_batches.csv", index=False)
        valid_pairs = pair_frame[pair_frame["cosine"].notna()]
        if not valid_pairs.empty:
            conflict_summary = valid_pairs.groupby("relationship", as_index=False).agg(
                mean_cosine=("cosine", "mean"),
                median_cosine=("cosine", "median"),
                n=("cosine", "size"),
            )
            conflict_summary.to_csv(
                output_dir / "family_gradient_conflict.csv", index=False
            )
            family_pairs = valid_pairs.copy()
            family_pairs[["family_low", "family_high"]] = family_pairs.apply(
                lambda row: sorted([row["family_a"], row["family_b"]]),
                axis=1,
                result_type="expand",
            )
            (
                family_pairs.groupby(["family_low", "family_high"], as_index=False)
                .agg(
                    mean_cosine=("cosine", "mean"),
                    median_cosine=("cosine", "median"),
                    n_outcome_batch_pairs=("cosine", "size"),
                    mean_joint_support=("joint_support", "mean"),
                )
                .to_csv(output_dir / "family_pair_gradient_conflict.csv", index=False)
            )
    print(
        f"{checkpoint}: {summary['n_supported_pairs']} supported outcome pairs, "
        f"fraction below zero={summary['fraction_below_zero']}, "
        f"fraction below {negative_threshold}="
        f"{summary['fraction_below_negative_threshold']}"
    )
    for heading, key in (
        ("Most conflicting", "most_conflicting"),
        ("Most aligned", "most_aligned"),
    ):
        print(f"{heading}:")
        for record in summary[key]:
            support_note = " [small support]" if record["small_support"] else ""
            print(
                f"  {record['outcome_a']} vs {record['outcome_b']}: "
                f"cosine={record['mean_cosine']:.4f}, "
                f"mean support={record['mean_joint_support']:.1f}, "
                f"batches={record['supported_batches']}{support_note}"
            )
    return summary


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Measure per-outcome representation-gradient conflict.",
    )
    parser.add_argument("--checkpoints", nargs="+", type=Path, required=True)
    parser.add_argument("--config-name", required=True)
    parser.add_argument(
        "--config-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "configs",
    )
    parser.add_argument("--override", action="append", default=[])
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batches", type=int, default=16)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument(
        "--logical-batch-size",
        type=int,
        help="Pairwise diagnostic batch assembled from physical encoder microbatches",
    )
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--min-overlap", type=int, default=8)
    parser.add_argument("--negative-threshold", type=float, default=-0.1)
    parser.add_argument(
        "--objective",
        choices=("contrastive", "competing_risk"),
        default="contrastive",
        help="Objective whose per-outcome representation gradients are compared.",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument(
        "--gradient-components",
        nargs="+",
        choices=tuple(_CR_COMPONENT_KEYS),
        default=["full_likelihood"],
        help="Competing-risk likelihood components to diagnose.",
    )
    parser.add_argument(
        "--null-permutations",
        type=int,
        default=0,
        help="Within-cohort/status/follow-up permutations per batch and pair.",
    )
    parser.add_argument(
        "--dependency-config",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "configs" / "endpoint_dependencies.yaml",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.batches < 1:
        raise ValueError("--batches must be positive.")
    if args.min_overlap < 2:
        raise ValueError("--min-overlap must be at least 2.")
    if args.null_permutations < 0:
        raise ValueError("--null-permutations must be non-negative.")
    if args.objective == "competing_risk" and "full_likelihood" not in args.gradient_components:
        raise ValueError("--gradient-components must include full_likelihood.")

    load_dotenv()
    GlobalHydra.instance().clear()
    with initialize_config_dir(
        config_dir=str(args.config_dir.resolve()),
        version_base="1.2",
    ):
        cfg = compose(
            config_name=args.config_name,
            overrides=args.override,
        )

    max_len = _shared_checkpoint_max_len(args.checkpoints)
    data_module, outcome_names, time_grids, probability_grids = _build_data(
        cfg,
        args.batch_size,
        args.logical_batch_size,
        args.num_workers,
        max_len,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    dependency_cfg = OmegaConf.to_container(
        OmegaConf.load(args.dependency_config), resolve=True
    )

    summaries = []
    for index, checkpoint in enumerate(args.checkpoints):
        checkpoint_output = _checkpoint_output_dir(
            args.output_dir,
            checkpoint,
            index,
        )
        summaries.append(
            diagnose_checkpoint(
                checkpoint,
                cfg,
                data_module,
                outcome_names,
                time_grids,
                probability_grids,
                checkpoint_output,
                n_batches=args.batches,
                min_overlap=args.min_overlap,
                negative_threshold=args.negative_threshold,
                device=device,
                seed=args.seed,
                objective=args.objective,
                gradient_components=tuple(args.gradient_components),
                null_permutations=args.null_permutations,
                endpoint_dependency_config=dependency_cfg,
            )
        )

    (args.output_dir / "checkpoints_summary.json").write_text(
        json.dumps(summaries, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
