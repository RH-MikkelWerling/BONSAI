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
from transformers import ModernBertConfig

from bonsai.functional.checkpointing import (
    MODEL_INIT_CONFIG_KEY,
    clean_lightning_state_dict,
    get_saved_encoder_config,
    load_state_dict_checked,
)
from opera.compat.bonsai import BonsaiEncoder
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
    cross_outcome = _checkpoint_weighter(clean_state, configured_weighter)

    dapt_store = None
    if cfg.get("dapt_embedding_store") is not None:
        dapt_store = torch.load(
            cfg.dapt_embedding_store,
            map_location="cpu",
            weights_only=False,
        )

    encoder = BonsaiEncoder(ModernBertConfig(**model_config))
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
                cfg.model.get("dapt_lambda_floor", 0.3),
            )
        ),
        dapt_anchor_weight=float(
            model_init.get(
                "dapt_anchor_weight",
                cfg.model.get("dapt_anchor_weight", 0.0),
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
    )
    load_state_dict_checked(model, clean_state, strict=True)
    model.to(device)
    model.eval()
    return model


def _build_data(
    cfg: DictConfig,
    batch_size: int | None,
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
        length = model_config.get("max_position_embeddings")
        if length is None:
            raise ValueError(
                f"Checkpoint {checkpoint_path} has no max_position_embeddings."
            )
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
) -> dict[str, Any]:
    """Run the representation-gradient diagnostic for one checkpoint."""
    model = _load_model(
        checkpoint,
        cfg,
        outcome_names,
        time_grids,
        probability_grids,
        device,
    )
    n_outcomes = len(outcome_names)
    cosine_sum = np.zeros((n_outcomes, n_outcomes), dtype=float)
    cosine_count = np.zeros((n_outcomes, n_outcomes), dtype=np.int64)
    support_sum = np.zeros((n_outcomes, n_outcomes), dtype=float)
    supported_batches = np.zeros((n_outcomes, n_outcomes), dtype=np.int64)
    event_sum = np.zeros(n_outcomes, dtype=float)
    event_count = np.zeros(n_outcomes, dtype=float)
    batch_pair_records: list[dict[str, Any]] = []

    torch.manual_seed(seed)
    loader = data_module.train_dataloader()
    processed_batches = 0
    for batch_index, cpu_batch in enumerate(loader):
        if batch_index >= n_batches:
            break
        batch = _move_to_device(cpu_batch, device)
        survival = _outcome_survival(batch, outcome_names)
        with torch.no_grad():
            embedding = model.get_embeddings(batch)

        with torch.enable_grad():
            representation = embedding.detach().clone().requires_grad_(True)
            terms, _ = model.contrastive_loss.compute_per_outcome_losses(
                representation,
                survival,
                subject_ids=batch.get("subject_id"),
                dapt_embedding_store=model.dapt_embedding_store,
            )
            active_names = [
                name
                for name, term in terms.items()
                if float(term["n_effective_pairs"].detach().item()) > 0.0
            ]
            gradients: dict[str, torch.Tensor] = {}
            for term_index, name in enumerate(active_names):
                gradient = torch.autograd.grad(
                    terms[name]["loss"],
                    representation,
                    retain_graph=term_index < len(active_names) - 1,
                )[0]
                gradients[name] = gradient.detach()

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
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--min-overlap", type=int, default=8)
    parser.add_argument("--negative-threshold", type=float, default=-0.1)
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument("--seed", type=int, default=17)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.batches < 1:
        raise ValueError("--batches must be positive.")
    if args.min_overlap < 2:
        raise ValueError("--min-overlap must be at least 2.")

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
        args.num_workers,
        max_len,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

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
            )
        )

    (args.output_dir / "checkpoints_summary.json").write_text(
        json.dumps(summaries, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
