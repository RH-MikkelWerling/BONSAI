"""Probe and visualize first-line regimen information in frozen embeddings.

Example
-------
python -m opera.run.treatment_embedding_atlas \
  --embeddings pretrain=/results/pretrain_embeddings.parquet \
  --embeddings dapt=/results/dapt_embeddings.parquet \
  --embeddings opera=/results/opera_embeddings.parquet \
  --metadata /data/hematology_first_line_metadata.parquet \
  --regimen_map opera/configs/treatment_regimen_groups.yaml \
  --atlas_model opera \
  --output_dir /results/treatment_atlas
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from opera.evaluation.treatment_embeddings import (
    build_atlas_tables,
    compute_disease_treatment_geometry,
    embedding_columns,
    merge_embeddings_with_metadata,
    normalize_regimens,
    probe_treatment_information,
)
from opera.visualization.treatment_atlas import (
    plot_disease_treatment_atlas,
    plot_treatment_probe_performance,
    project_embedding_frame,
)


def _read_table(path: str) -> pd.DataFrame:
    suffix = Path(path).suffix.lower()
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    if suffix == ".npz":
        data = np.load(path, allow_pickle=False)
        if "subject_ids" not in data or "embeddings" not in data:
            raise ValueError(f"{path} must contain subject_ids and embeddings arrays.")
        embeddings = np.asarray(data["embeddings"])
        subject_ids = np.asarray(data["subject_ids"])
        if embeddings.ndim != 2 or embeddings.shape[1] == 0:
            raise ValueError(f"{path} contains no usable embedding dimensions.")
        if len(subject_ids) != len(embeddings):
            raise ValueError(f"{path} subject_ids and embeddings lengths differ.")
        frame = pd.DataFrame(
            embeddings,
            columns=[f"embedding_{index}" for index in range(embeddings.shape[1])],
        )
        frame.insert(0, "subject_id", subject_ids)
        return frame
    if suffix in {".pt", ".pth"}:
        import torch

        loaded = torch.load(path, map_location="cpu", weights_only=True)
        if not isinstance(loaded, dict) or not loaded:
            raise ValueError(f"{path} must contain a non-empty subject embedding map.")
        subject_ids = list(loaded)
        vectors = np.vstack(
            [
                np.asarray(torch.as_tensor(loaded[item]).cpu(), dtype=float)
                for item in subject_ids
            ]
        )
        if vectors.ndim != 2 or vectors.shape[1] == 0:
            raise ValueError(f"{path} contains no usable embedding dimensions.")
        frame = pd.DataFrame(
            vectors,
            columns=[f"embedding_{index}" for index in range(vectors.shape[1])],
        )
        frame.insert(0, "subject_id", subject_ids)
        return frame
    return pd.read_csv(path)


def _embedding_spec(value: str) -> tuple[str, str]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("Embedding inputs must use NAME=PATH.")
    name, path = value.split("=", 1)
    if not name.strip() or not path.strip():
        raise argparse.ArgumentTypeError("Embedding inputs must use NAME=PATH.")
    return name.strip(), path.strip()


def _comma_list(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _read_regimen_mapping(path: str) -> dict[str, dict[str, list[Any]]]:
    with open(path, encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle) or {}
    mapping = loaded.get("diseases", loaded)
    if not isinstance(mapping, dict):
        raise ValueError("Regimen mapping must be a disease-keyed YAML mapping.")
    return mapping


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Measure disease-conditioned regimen accessibility in frozen "
            "embeddings and render a shared disease-treatment atlas."
        )
    )
    parser.add_argument(
        "--embeddings",
        action="append",
        type=_embedding_spec,
        required=True,
        metavar="NAME=PATH",
        help=(
            "Frozen embedding table or predictions.npz. Repeat for pretrain, "
            "DAPT, OPERA, or other representation stages."
        ),
    )
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--atlas_model", default="opera")
    parser.add_argument("--subject_col", default="subject_id")
    parser.add_argument("--disease_col", default="disease")
    parser.add_argument("--treatment_col", default="first_line_regimen")
    parser.add_argument("--regimen_group_col", default="regimen_group")
    parser.add_argument("--split_col", default="split")
    parser.add_argument("--train_splits", default="train,tuning")
    parser.add_argument("--test_split", default="held_out")
    parser.add_argument(
        "--evaluation_mode",
        choices=["auto", "held_out", "cv"],
        default="auto",
        help="Auto prefers held-out rows and labels CV output exploratory.",
    )
    parser.add_argument(
        "--regimen_map",
        default=None,
        help="Optional disease-specific raw-regimen normalization YAML.",
    )
    parser.add_argument(
        "--keep_unmapped",
        action="store_true",
        help="Keep raw treatment names not listed in --regimen_map.",
    )
    parser.add_argument("--projection", choices=["umap", "tsne", "pca"], default="umap")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min_disease_n", type=int, default=50)
    parser.add_argument("--min_class_n", type=int, default=10)
    parser.add_argument("--min_test_class_n", type=int, default=2)
    parser.add_argument("--min_joint_n", type=int, default=10)
    parser.add_argument("--n_splits", type=int, default=5)
    args = parser.parse_args()

    embedding_specs = dict(args.embeddings)
    if len(embedding_specs) != len(args.embeddings):
        raise ValueError("Embedding stage names must be unique.")
    if args.atlas_model not in embedding_specs:
        raise ValueError(
            f"atlas_model={args.atlas_model!r} is not one of {sorted(embedding_specs)}."
        )

    metadata = _read_table(args.metadata)
    if args.regimen_map:
        metadata = normalize_regimens(
            metadata,
            _read_regimen_mapping(args.regimen_map),
            disease_col=args.disease_col,
            treatment_col=args.treatment_col,
            output_col=args.regimen_group_col,
            keep_unmapped=args.keep_unmapped,
        )
    else:
        if args.treatment_col not in metadata.columns:
            raise ValueError(
                f"Metadata is missing treatment column {args.treatment_col!r}."
            )
        metadata = metadata.copy()
        metadata[args.regimen_group_col] = metadata[args.treatment_col]

    embedding_frames = {
        stage: _read_table(path) for stage, path in embedding_specs.items()
    }
    if args.subject_col not in metadata.columns:
        raise ValueError(f"Metadata is missing subject column {args.subject_col!r}.")
    common_subjects = set(metadata[args.subject_col])
    for stage, frame in embedding_frames.items():
        if args.subject_col not in frame.columns:
            raise ValueError(
                f"Embedding stage {stage!r} is missing {args.subject_col!r}."
            )
        columns = embedding_columns(frame)
        finite = np.isfinite(frame[columns].to_numpy(dtype=float)).all(axis=1)
        common_subjects &= set(frame.loc[finite, args.subject_col])
    if not common_subjects:
        raise ValueError("Embedding stages and metadata have no shared patients.")
    metadata = metadata[metadata[args.subject_col].isin(common_subjects)].copy()
    embedding_frames = {
        stage: frame[frame[args.subject_col].isin(common_subjects)].copy()
        for stage, frame in embedding_frames.items()
    }

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "analysis_metadata.json", "w", encoding="utf-8") as handle:
        json.dump(
            {
                "embedding_artifacts": embedding_specs,
                "atlas_model": args.atlas_model,
                "n_shared_patients": len(common_subjects),
                "disease_col": args.disease_col,
                "treatment_col": args.treatment_col,
                "regimen_group_col": args.regimen_group_col,
                "split_col": args.split_col,
                "train_splits": _comma_list(args.train_splits),
                "test_split": args.test_split,
                "evaluation_mode": args.evaluation_mode,
                "projection": args.projection,
                "regimen_map": args.regimen_map,
                "interpretation": (
                    "Treatment-selection accessibility analysis; not a treatment "
                    "recommendation or causal effect estimate."
                ),
            },
            handle,
            indent=2,
        )
    probe_results = []
    probe_predictions = []
    merged_by_stage: dict[str, pd.DataFrame] = {}
    for stage, frame in embedding_frames.items():
        merged = merge_embeddings_with_metadata(
            frame,
            metadata,
            subject_col=args.subject_col,
        )
        merged_by_stage[stage] = merged
        results, predictions = probe_treatment_information(
            merged,
            disease_col=args.disease_col,
            treatment_col=args.regimen_group_col,
            subject_col=args.subject_col,
            split_col=args.split_col if args.split_col in merged.columns else None,
            train_splits=_comma_list(args.train_splits),
            test_split=args.test_split,
            evaluation_mode=args.evaluation_mode,
            min_disease_n=args.min_disease_n,
            min_class_n=args.min_class_n,
            min_test_class_n=args.min_test_class_n,
            n_splits=args.n_splits,
            seed=args.seed,
        )
        if not results.empty:
            results.insert(0, "embedding_stage", stage)
            probe_results.append(results)
        if not predictions.empty:
            predictions.insert(0, "embedding_stage", stage)
            probe_predictions.append(predictions)

    all_results = (
        pd.concat(probe_results, ignore_index=True) if probe_results else pd.DataFrame()
    )
    all_predictions = (
        pd.concat(probe_predictions, ignore_index=True)
        if probe_predictions
        else pd.DataFrame()
    )
    all_results.to_csv(output_dir / "treatment_probe_results.csv", index=False)
    all_predictions.to_csv(output_dir / "treatment_probe_predictions.csv", index=False)

    atlas_frame = (
        merged_by_stage[args.atlas_model]
        .dropna(subset=[args.disease_col, args.regimen_group_col])
        .reset_index(drop=True)
    )
    if atlas_frame.empty:
        raise ValueError(
            "No atlas patients remain after disease and regimen normalization."
        )
    coordinates = project_embedding_frame(
        atlas_frame,
        method=args.projection,
        seed=args.seed,
    )
    patient_coordinates, centroids = build_atlas_tables(
        atlas_frame,
        coordinates,
        disease_col=args.disease_col,
        treatment_col=args.regimen_group_col,
        subject_col=args.subject_col,
    )
    patient_coordinates.to_csv(
        output_dir / "disease_treatment_atlas_coordinates.csv", index=False
    )
    centroids.to_csv(output_dir / "disease_treatment_atlas_centroids.csv", index=False)
    geometry = compute_disease_treatment_geometry(
        atlas_frame,
        disease_col=args.disease_col,
        treatment_col=args.regimen_group_col,
        min_joint_n=args.min_joint_n,
    )
    geometry.to_csv(output_dir / "disease_treatment_geometry.csv", index=False)
    plot_disease_treatment_atlas(
        patient_coordinates,
        centroids,
        disease_col=args.disease_col,
        treatment_col=args.regimen_group_col,
        min_disease_n=args.min_disease_n,
        min_treatment_n=args.min_class_n,
        min_joint_n=args.min_joint_n,
        save_path=str(output_dir / "disease_treatment_atlas.png"),
    )
    if not all_results.empty:
        plot_treatment_probe_performance(
            all_results,
            disease_col=args.disease_col,
            save_path=str(output_dir / "treatment_probe_performance.png"),
        )

    print(
        f"Wrote disease-treatment atlas artifacts for {len(atlas_frame):,} "
        f"patients to {output_dir}."
    )
    if all_results.empty:
        print(
            "No disease had enough supported treatment classes for a probe; "
            "review min_disease_n, min_class_n, and split coverage."
        )
    else:
        print(
            f"Treatment probes evaluated {len(all_results)} "
            "embedding-stage/disease cells."
        )


if __name__ == "__main__":
    main()
