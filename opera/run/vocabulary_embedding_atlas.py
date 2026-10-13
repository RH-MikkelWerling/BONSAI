"""Extract and visualize learned vocabulary/code embeddings.

Example
-------
python -m opera.run.vocabulary_embedding_atlas \
  --checkpoint pretrain=/results/pretrain.ckpt \
  --checkpoint dapt=/results/dapt.ckpt \
  --checkpoint opera=/results/opera.ckpt \
  --vocabulary /data/hematology/vocabulary.pt \
  --atlas_stage opera \
  --highlight_tokens LPR3//DC833,RKKP//ann_arbor_III \
  --output_dir /results/vocabulary_embedding_atlas
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib as mpl
from matplotlib import font_manager
import numpy as np
import pandas as pd

from opera.evaluation.treatment_embeddings import embedding_columns
from opera.evaluation.vocabulary_embeddings import (
    compute_token_movement,
    extract_vocabulary_embedding_frame,
    nearest_token_neighbors,
)
from opera.visualization.vocabulary_atlas import (
    plot_token_movement,
    plot_vocabulary_embedding_atlas,
    plot_vocabulary_family_facets,
    project_vocabulary_embeddings,
)


def _read_table(path: str) -> pd.DataFrame:
    suffix = Path(path).suffix.lower()
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    if suffix in {".json", ".jsonl"}:
        return pd.read_json(path, lines=suffix == ".jsonl")
    return pd.read_csv(path)


def _checkpoint_spec(value: str) -> tuple[str, str]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("Checkpoint inputs must use NAME=PATH.")
    name, path = value.split("=", 1)
    if not name.strip() or not path.strip():
        raise argparse.ArgumentTypeError("Checkpoint inputs must use NAME=PATH.")
    return name.strip(), path.strip()


def _comma_list(value: str | None) -> list[str]:
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_") or "stage"


def family_cosine_proximity(
    embeddings: pd.DataFrame, *, color_col: str = "token_family", min_group_n: int = 5,
) -> pd.DataFrame:
    """Mean token-pair cosine by family pair in the original embedding space."""
    cols = embedding_columns(embeddings)
    values = embeddings[cols].to_numpy(dtype=float)
    norms = np.linalg.norm(values, axis=1)
    if not np.isfinite(values).all() or (norms <= 0).any():
        raise ValueError("Proximity requires finite, nonzero token embeddings")
    unit = values / norms[:, None]
    labels = embeddings[color_col].fillna("unknown").astype(str).to_numpy()
    counts = pd.Series(labels).value_counts()
    families = counts[counts >= min_group_n].index.tolist()
    means = {family: unit[labels == family].mean(axis=0) for family in families}
    rows = []
    for left in families:
        for right in families:
            n_left, n_right = int(counts[left]), int(counts[right])
            if left == right:
                # Remove self-pairs (cosine=1) from the within-family mean.
                mean_cosine = (
                    (n_left * float(means[left] @ means[left]) - 1) / (n_left - 1)
                    if n_left > 1 else np.nan
                )
            else:
                mean_cosine = float(means[left] @ means[right])
            rows.append({"family_a": left, "family_b": right,
                         "n_a": n_left, "n_b": n_right,
                         "mean_pair_cosine": mean_cosine})
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Extract code_embedding.weight from one or more checkpoints and "
            "render a token-level vocabulary atlas."
        )
    )
    parser.add_argument(
        "--checkpoint",
        action="append",
        type=_checkpoint_spec,
        required=True,
        metavar="NAME=PATH",
        help="Checkpoint or state-dict artifact. Repeat for pretrain, DAPT, OPERA.",
    )
    parser.add_argument("--vocabulary", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument(
        "--metadata",
        default=None,
        help="Optional token metadata table to merge by token_id or token.",
    )
    parser.add_argument(
        "--atlas_stage",
        default=None,
        help="Stage to project for the atlas. Defaults to opera if present, else last.",
    )
    parser.add_argument(
        "--reference_stage",
        default=None,
        help="Reference stage for token movement. Defaults to the first checkpoint.",
    )
    parser.add_argument("--color_col", default="token_family")
    parser.add_argument("--projection", choices=["umap", "tsne", "pca"], default="umap")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min_group_n", type=int, default=5)
    parser.add_argument("--max_group_labels", type=int, default=20)
    parser.add_argument("--top_moved_tokens", type=int, default=25)
    parser.add_argument(
        "--highlight_tokens",
        default=None,
        help="Comma-separated token names to annotate in the atlas.",
    )
    parser.add_argument(
        "--neighbor_tokens",
        default=None,
        help="Comma-separated token names for cosine-nearest-neighbor tables.",
    )
    parser.add_argument("--top_neighbors", type=int, default=10)
    args = parser.parse_args()

    try:
        font_manager.findfont(
            font_manager.FontProperties(family="TeX Gyre Pagella"),
            fallback_to_default=False,
        )
        figure_font = "TeX Gyre Pagella"
    except ValueError:
        figure_font = "DejaVu Serif"
        print("TeX Gyre Pagella is unavailable; atlas figures use DejaVu Serif.")

    checkpoint_specs = dict(args.checkpoint)
    if len(checkpoint_specs) != len(args.checkpoint):
        raise ValueError("Checkpoint stage names must be unique.")
    atlas_stage = args.atlas_stage
    if atlas_stage is None:
        atlas_stage = "opera" if "opera" in checkpoint_specs else args.checkpoint[-1][0]
    if atlas_stage not in checkpoint_specs:
        raise ValueError(f"atlas_stage={atlas_stage!r} is not available.")
    reference_stage = args.reference_stage or args.checkpoint[0][0]
    if reference_stage not in checkpoint_specs:
        raise ValueError(f"reference_stage={reference_stage!r} is not available.")

    token_metadata = _read_table(args.metadata) if args.metadata else None
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    stage_frames: dict[str, pd.DataFrame] = {}
    for stage, checkpoint_path in checkpoint_specs.items():
        frame = extract_vocabulary_embedding_frame(
            checkpoint_path,
            args.vocabulary,
            stage=stage,
            token_metadata=token_metadata,
        )
        stage_frames[stage] = frame
        frame.to_csv(
            output_dir / f"vocabulary_embeddings_{_safe_name(stage)}.csv",
            index=False,
        )

    coordinates = project_vocabulary_embeddings(
        stage_frames[atlas_stage],
        method=args.projection,
        seed=args.seed,
    )
    coordinates.to_csv(output_dir / "vocabulary_atlas_coordinates.csv", index=False)
    family_cosine_proximity(
        stage_frames[atlas_stage], color_col=args.color_col,
        min_group_n=args.min_group_n,
    ).to_csv(output_dir / "vocabulary_family_cosine_proximity.csv", index=False)
    with mpl.rc_context({"font.family": figure_font, "mathtext.fontset": "dejavuserif"}):
        plot_vocabulary_embedding_atlas(
            coordinates,
            color_col=args.color_col,
            highlight_tokens=_comma_list(args.highlight_tokens),
            min_group_n=args.min_group_n,
            max_group_labels=args.max_group_labels,
            title=f"Vocabulary embedding atlas ({atlas_stage})",
            save_path=str(output_dir / "vocabulary_atlas.png"),
        )
        plot_vocabulary_family_facets(
            coordinates,
            color_col=args.color_col,
            min_group_n=args.min_group_n,
            title=f"Vocabulary families ({atlas_stage})",
            save_path=str(output_dir / "vocabulary_family_facets.png"),
        )

    neighbor_tokens = _comma_list(args.neighbor_tokens)
    if neighbor_tokens:
        neighbors = nearest_token_neighbors(
            stage_frames[atlas_stage],
            neighbor_tokens,
            top_k=args.top_neighbors,
        )
        neighbors.to_csv(output_dir / "vocabulary_neighbors.csv", index=False)

    if len(stage_frames) > 1:
        movement = compute_token_movement(
            stage_frames,
            reference_stage=reference_stage,
        )
        movement.to_csv(output_dir / "token_movement.csv", index=False)
        if not movement.empty:
            with mpl.rc_context({"font.family": figure_font, "mathtext.fontset": "dejavuserif"}):
                plot_token_movement(
                    movement,
                    top_n=args.top_moved_tokens,
                    save_path=str(output_dir / "token_movement.png"),
                )

    with open(output_dir / "analysis_metadata.json", "w", encoding="utf-8") as handle:
        json.dump(
            {
                "checkpoint_artifacts": checkpoint_specs,
                "vocabulary": args.vocabulary,
                "token_metadata": args.metadata,
                "atlas_stage": atlas_stage,
                "reference_stage": reference_stage,
                "projection": args.projection,
                "color_col": args.color_col,
                "figure_font": figure_font,
                "interpretation": (
                    "Token/code embedding geometry. These are vocabulary "
                    "lookup vectors, not patient embeddings and not outcome "
                    "predictions. Family proximity uses original embedding "
                    "cosines; the 2D projection is descriptive only."
                ),
            },
            handle,
            indent=2,
        )

    print(
        f"Wrote vocabulary embedding atlas for {atlas_stage!r} "
        f"({len(coordinates):,} tokens) to {output_dir}."
    )


if __name__ == "__main__":
    main()
