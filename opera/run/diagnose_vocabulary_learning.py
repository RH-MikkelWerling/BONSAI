"""Measure token exposure, geometry, loss, and contextual sensitivity."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GroupShuffleSplit

from bonsai.modules.datasets.PretrainDataset import ARPretrainDataset
from opera.compat.bonsai import dynamic_padding, encoder_hidden_state
from opera.evaluation.vocabulary_embeddings import (
    extract_vocabulary_embedding_frame,
    invert_vocabulary,
    load_checkpoint_state_dict,
)
from opera.evaluation.vocabulary_learning import (
    all_token_neighbours,
    count_token_exposure,
    neighbour_coherence,
    stratified_token_sample,
    token_geometry,
    vocabulary_coverage,
)
from opera.modules.datamodules.OutcomeFinetuneDataModule import load_subject_pool
from opera.run.extract_outcome_transfer_embeddings import (
    _resolve_device,
    load_frozen_encoder,
)


def _comma_ints(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def _to_device(batch: dict, device: torch.device) -> dict:
    return {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }


def _pool_valid(hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weights = mask.to(hidden.dtype).unsqueeze(-1)
    return (hidden * weights).sum(1) / weights.sum(1).clamp_min(1)


def contextual_sensitivity(
    encoder: torch.nn.Module,
    loader: DataLoader,
    *,
    device: torch.device,
    max_batches: int,
    max_events_per_batch: int = 256,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Counterfactually perturb value, calendar, code order, and context order."""
    per_subject = []
    event_samples = []
    with torch.inference_mode():
        for batch_index, batch in enumerate(loader):
            if batch_index >= max_batches:
                break
            original = _to_device(batch, device)
            mask = original["attention_mask"].bool()
            baseline_hidden = encoder_hidden_state(encoder(original))
            baseline = _pool_valid(baseline_hidden, mask)
            valid_positions = mask.nonzero(as_tuple=False)
            if len(valid_positions) > max_events_per_batch:
                selection = torch.linspace(
                    0,
                    len(valid_positions) - 1,
                    max_events_per_batch,
                    device=device,
                ).long()
                valid_positions = valid_positions[selection]
            raw_embeddings = encoder.embeddings.code_embedding(original["code"])
            for row, position in valid_positions.tolist():
                record = {
                    "subject_id": int(original["subject_id"][row].item()),
                    "token_id": int(original["code"][row, position].item()),
                    "abspos": float(original["abspos"][row, position].item()),
                    "numeric_value": (
                        float(original["numeric_value"][row, position].item())
                        if "numeric_value" in original
                        else np.nan
                    ),
                }
                for dim, value in enumerate(
                    baseline_hidden[row, position].float().cpu().tolist()
                ):
                    record[f"contextual_{dim}"] = value
                for dim, value in enumerate(
                    raw_embeddings[row, position].float().cpu().tolist()
                ):
                    record[f"raw_{dim}"] = value
                event_samples.append(record)
            variants: dict[str, dict] = {}

            if "numeric_value" in original:
                variant = dict(original)
                variant["numeric_value"] = torch.full_like(
                    original["numeric_value"], float("nan")
                )
                variants["mask_numeric_values"] = variant

            variant = dict(original)
            variant["abspos"] = original["abspos"] + 5.0 * 8766.0
            variants["calendar_plus_5y"] = variant

            # Reverse only real tokens. This preserves the multiset and padding,
            # while disrupting local chronology and event-to-time binding.
            variant = {
                key: value.clone() if isinstance(value, torch.Tensor) else value
                for key, value in original.items()
            }
            sequence_fields = [
                key
                for key, value in original.items()
                if isinstance(value, torch.Tensor)
                and value.ndim >= 2
                and value.shape[:2] == original["code"].shape
                and key != "attention_mask"
            ]
            for row, length in enumerate(mask.sum(1).tolist()):
                index = torch.arange(length - 1, -1, -1, device=device)
                for key in sequence_fields:
                    variant[key][row, :length] = original[key][row, index]
            variants["reverse_event_context"] = variant

            for name, variant_batch in variants.items():
                hidden = encoder_hidden_state(encoder(variant_batch))
                pooled = _pool_valid(hidden, mask)
                cosine = F.cosine_similarity(baseline.float(), pooled.float(), dim=1)
                relative_l2 = torch.linalg.vector_norm(
                    pooled.float() - baseline.float(), dim=1
                ) / torch.linalg.vector_norm(baseline.float(), dim=1).clamp_min(1e-8)
                subject_ids = original.get("subject_id")
                ids = (
                    subject_ids.detach().cpu().tolist()
                    if isinstance(subject_ids, torch.Tensor)
                    else list(range(len(cosine)))
                )
                for subject_id, cos, distance in zip(
                    ids, cosine.cpu().tolist(), relative_l2.cpu().tolist()
                ):
                    per_subject.append(
                        {
                            "subject_id": int(subject_id),
                            "perturbation": name,
                            "cosine_similarity": float(cos),
                            "cosine_distance": float(1.0 - cos),
                            "relative_l2": float(distance),
                        }
                    )
    details = pd.DataFrame(per_subject)
    summary = (
        details.groupby("perturbation")
        .agg(
            n_subjects=("subject_id", "nunique"),
            mean_cosine_distance=("cosine_distance", "mean"),
            median_cosine_distance=("cosine_distance", "median"),
            mean_relative_l2=("relative_l2", "mean"),
            median_relative_l2=("relative_l2", "median"),
        )
        .reset_index()
    )
    return details, summary, pd.DataFrame(event_samples)


def contextual_probes(events: pd.DataFrame, *, seed: int) -> pd.DataFrame:
    """Compare raw and contextual linear decodability on held-out subjects."""
    rows = []
    targets = {
        "numeric_value": np.isfinite(events["numeric_value"]),
        "calendar_abspos": np.isfinite(events["abspos"]) & (events["abspos"] > 0),
    }
    for target_name, valid in targets.items():
        target_col = "numeric_value" if target_name == "numeric_value" else "abspos"
        frame = events.loc[valid].reset_index(drop=True)
        if len(frame) < 50 or frame["subject_id"].nunique() < 10:
            continue
        splitter = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=seed)
        train_index, test_index = next(
            splitter.split(frame, groups=frame["subject_id"])
        )
        y_train = frame.loc[train_index, target_col].to_numpy(float)
        y_test = frame.loc[test_index, target_col].to_numpy(float)
        for representation in ("raw", "contextual"):
            columns = [
                column for column in frame if column.startswith(f"{representation}_")
            ]
            model = Ridge(alpha=1.0)
            model.fit(frame.loc[train_index, columns], y_train)
            prediction = model.predict(frame.loc[test_index, columns])
            rows.append(
                {
                    "target": target_name,
                    "representation": representation,
                    "n_train_events": len(train_index),
                    "n_test_events": len(test_index),
                    "n_train_subjects": int(
                        frame.loc[train_index, "subject_id"].nunique()
                    ),
                    "n_test_subjects": int(
                        frame.loc[test_index, "subject_id"].nunique()
                    ),
                    "r2": float(r2_score(y_test, prediction)),
                    "mae": float(mean_absolute_error(y_test, prediction)),
                }
            )
    return pd.DataFrame(rows)


def token_losses(
    encoder: torch.nn.Module,
    loader: DataLoader,
    *,
    device: torch.device,
    max_batches: int,
    logit_chunk_size: int,
    output_bias: torch.Tensor | None = None,
) -> pd.DataFrame:
    """Compute target-level next-code CE without creating giant 3D logits."""
    weight = encoder.embeddings.code_embedding.weight
    sums: dict[int, float] = {}
    counts: dict[int, int] = {}
    with torch.inference_mode():
        for batch_index, batch in enumerate(loader):
            if batch_index >= max_batches:
                break
            device_batch = _to_device(batch, device)
            hidden = encoder_hidden_state(encoder(device_batch))
            targets = device_batch["target"]
            valid = targets != -100
            states = hidden[valid]
            labels = targets[valid].long()
            for start in range(0, len(labels), logit_chunk_size):
                stop = min(start + logit_chunk_size, len(labels))
                logits = F.linear(
                    states[start:stop].float(),
                    weight.float(),
                    output_bias.float() if output_bias is not None else None,
                )
                losses = F.cross_entropy(logits, labels[start:stop], reduction="none")
                for token_id, loss in zip(
                    labels[start:stop].cpu().tolist(), losses.cpu().tolist()
                ):
                    sums[token_id] = sums.get(token_id, 0.0) + float(loss)
                    counts[token_id] = counts.get(token_id, 0) + 1
    return pd.DataFrame(
        [
            {
                "token_id": token_id,
                "n_loss_targets": counts[token_id],
                "mean_code_ce": sums[token_id] / counts[token_id],
                "total_code_ce": sums[token_id],
            }
            for token_id in sorted(counts)
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--initial-checkpoint")
    parser.add_argument("--vocabulary", required=True)
    parser.add_argument("--subject-data", action="append", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--max-len",
        type=int,
        default=None,
        help="Defaults to the checkpoint max_seqlen.",
    )
    parser.add_argument(
        "--background-length",
        type=int,
        default=None,
        help="Defaults to the segment==0 count in the first subject, as in training.",
    )
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-loss-batches", type=int, default=32)
    parser.add_argument("--max-context-batches", type=int, default=32)
    parser.add_argument("--logit-chunk-size", type=int, default=64)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument(
        "--max-neighbor-tokens",
        type=int,
        default=5000,
        help="Frequency-stratified cap for the quadratic exact-neighbour audit.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--frequency-thresholds", default="1,2,5,10,20,50,100,200,500,1000"
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--attention-backend", default="auto")
    args = parser.parse_args()

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    device = _resolve_device(args.device)
    encoder, vocabulary, metadata = load_frozen_encoder(
        args.checkpoint,
        vocabulary_path=args.vocabulary,
        device=device,
        attention_backend=args.attention_backend,
    )
    subjects = load_subject_pool(args.subject_data)
    if not subjects:
        raise ValueError("No subjects were loaded.")
    id_to_token = invert_vocabulary(vocabulary)

    print(f"[1/5] Counting exposure across {len(subjects):,} subjects...", flush=True)
    exposure = count_token_exposure(subjects, id_to_token)
    exposure.to_csv(output / "token_exposure.csv", index=False)
    coverage = vocabulary_coverage(exposure, _comma_ints(args.frequency_thresholds))
    coverage.to_csv(output / "vocabulary_frequency_coverage.csv", index=False)

    print("[2/5] Extracting token geometry...", flush=True)
    trained = extract_vocabulary_embedding_frame(
        args.checkpoint, args.vocabulary, stage="trained"
    )
    initial = (
        extract_vocabulary_embedding_frame(
            args.initial_checkpoint, args.vocabulary, stage="initial"
        )
        if args.initial_checkpoint
        else None
    )
    geometry = token_geometry(trained, exposure, initial)

    print(
        f"[3/5] Computing exact neighbours for up to {args.max_neighbor_tokens:,} tokens...",
        flush=True,
    )
    neighbor_ids = stratified_token_sample(
        geometry,
        max_tokens=args.max_neighbor_tokens,
        seed=args.seed,
    )["token_id"]
    neighbor_frame = trained[trained["token_id"].isin(neighbor_ids)].reset_index(
        drop=True
    )
    neighbours = all_token_neighbours(neighbor_frame, top_k=args.top_k)
    coherence = neighbour_coherence(neighbours, geometry)
    annotated_neighbours = neighbours.merge(
        exposure[["token_id", "n_occurrences"]], on="token_id", validate="many_to_one"
    )
    annotated_neighbours.to_csv(
        output / "frequency_stratified_neighbors.csv", index=False
    )
    coherence.to_csv(output / "neighbor_coherence_by_frequency.csv", index=False)

    max_len = int(args.max_len or encoder.hparams["max_seqlen"])
    background_length = (
        int(args.background_length)
        if args.background_length is not None
        else int((subjects[0]["segment"] == 0).sum())
    )
    dataset = ARPretrainDataset(
        subjects,
        max_len=max_len,
        background_length=background_length,
        vocabulary=vocabulary,
        value_embedding_mode=encoder.hparams.get("value_embedding_mode", "legacy"),
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=dynamic_padding,
    )
    print(
        f"[4/5] Sampling target loss over {args.max_loss_batches} batches...",
        flush=True,
    )
    checkpoint_state = load_checkpoint_state_dict(args.checkpoint)
    bias_candidates = [
        value
        for key, value in checkpoint_state.items()
        if key.endswith("pretrain_head.bias")
    ]
    output_bias = bias_candidates[0].to(device) if len(bias_candidates) == 1 else None
    losses = token_losses(
        encoder,
        loader,
        device=device,
        max_batches=args.max_loss_batches,
        logit_chunk_size=args.logit_chunk_size,
        output_bias=output_bias,
    )
    geometry = geometry.merge(losses, on="token_id", how="left", validate="one_to_one")
    geometry.to_csv(output / "token_learning_diagnostics.csv", index=False)

    print(
        f"[5/5] Running contextual perturbations over {args.max_context_batches} batches...",
        flush=True,
    )
    details, sensitivity, contextual_events = contextual_sensitivity(
        encoder,
        loader,
        device=device,
        max_batches=args.max_context_batches,
    )
    details.to_csv(output / "contextual_sensitivity_subjects.csv", index=False)
    sensitivity.to_csv(output / "contextual_sensitivity_summary.csv", index=False)
    contextual_events.to_parquet(output / "contextual_event_sample.parquet", index=False)
    contextual_probes(contextual_events, seed=args.seed).to_csv(
        output / "contextual_linear_probes.csv", index=False
    )
    (output / "summary.json").write_text(
        json.dumps(
            {
                "checkpoint": args.checkpoint,
                "initial_checkpoint": args.initial_checkpoint,
                "subject_data": args.subject_data,
                "n_subjects": len(subjects),
                "n_vocabulary_tokens": len(vocabulary),
                "n_neighbor_tokens": len(neighbor_frame),
                "max_len": max_len,
                "background_length": background_length,
                "checkpoint_metadata": metadata,
                "movement_note": (
                    "Exact checkpoint-to-checkpoint movement included."
                    if initial is not None
                    else "No initialization checkpoint supplied; initialization movement is unavailable."
                ),
            },
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )
    print(f"Wrote vocabulary learning diagnostics to {output}")


if __name__ == "__main__":
    main()
