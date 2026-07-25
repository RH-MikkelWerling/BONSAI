"""Plot generic frozen patient embeddings with reusable 2D coordinates."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from opera.run.extract_patient_embeddings import read_index_table
from opera.visualization.style import CATEGORICAL, clean_2d_axes, save_fig


def load_embedding_frame(
    embedding_path: str | Path,
    metadata_paths: list[str],
    *,
    subject_col: str,
) -> tuple[np.ndarray, pd.DataFrame]:
    """Load embeddings and left-join one or more patient metadata tables."""
    artifact = np.load(embedding_path)
    if not {"subject_ids", "embeddings"} <= set(artifact.files):
        raise ValueError("Embedding NPZ must contain subject_ids and embeddings.")
    subject_ids = artifact["subject_ids"]
    embeddings = np.asarray(artifact["embeddings"], dtype=np.float32)
    if embeddings.ndim != 2 or len(subject_ids) != len(embeddings):
        raise ValueError("Embedding NPZ has inconsistent subject and embedding shapes.")

    frame_values = {
        "subject_id": subject_ids,
        "_join_id": pd.Series(subject_ids).astype(str),
    }
    if "splits" in artifact.files:
        if len(artifact["splits"]) != len(subject_ids):
            raise ValueError("Embedding NPZ has an inconsistent splits array.")
        frame_values["split"] = artifact["splits"].astype(str)
    frame = pd.DataFrame(frame_values)
    for raw_path in metadata_paths:
        metadata = read_index_table(raw_path)
        if subject_col not in metadata:
            raise ValueError(f"{raw_path} is missing subject column {subject_col!r}.")
        if metadata[subject_col].duplicated().any():
            raise ValueError(f"{raw_path} contains duplicate patient identifiers.")
        metadata = metadata.copy()
        metadata["_join_id"] = metadata[subject_col].astype(str)
        metadata = metadata.drop(columns=[subject_col], errors="ignore")
        overlap = (set(frame) & set(metadata)) - {"_join_id"}
        if overlap:
            raise ValueError(
                f"{raw_path} repeats metadata columns already loaded: {sorted(overlap)}."
            )
        frame = frame.merge(metadata, on="_join_id", how="left", validate="one_to_one")
    return embeddings, frame.drop(columns="_join_id")


def reduce_patient_embeddings(
    embeddings: np.ndarray,
    *,
    method: str,
    seed: int,
    n_neighbors: int,
    min_dist: float,
    perplexity: float = 30.0,
) -> tuple[np.ndarray, str]:
    """Return deterministic PCA, t-SNE, or UMAP coordinates."""
    if method == "pca":
        return PCA(n_components=2, random_state=seed).fit_transform(embeddings), "pca"
    if method == "tsne":
        if perplexity <= 0 or perplexity >= len(embeddings):
            raise ValueError(
                f"t-SNE perplexity must be between 0 and n_samples ({len(embeddings)})."
            )
        return (
            TSNE(
                n_components=2,
                perplexity=perplexity,
                learning_rate="auto",
                init="pca",
                random_state=seed,
            ).fit_transform(embeddings),
            "tsne",
        )
    try:
        from umap import UMAP
    except ImportError as exc:
        raise ImportError(
            "UMAP is not installed. Load/install umap-learn or rerun with "
            "--method pca."
        ) from exc
    reducer = UMAP(
        n_components=2,
        n_neighbors=min(n_neighbors, max(2, len(embeddings) - 1)),
        min_dist=min_dist,
        metric="cosine",
        random_state=seed,
    )
    return reducer.fit_transform(embeddings), "umap"


def plot_colored_coordinates(
    frame: pd.DataFrame,
    *,
    color_by: str,
    method: str,
    output_path: str | Path,
    max_categories: int = 20,
) -> None:
    """Plot one categorical or continuous metadata view of shared coordinates."""
    if color_by not in frame:
        raise ValueError(f"Unknown --color-by column {color_by!r}.")
    values = frame[color_by]
    valid = values.notna().to_numpy()
    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    unique = values[valid].nunique()
    categorical = (
        pd.api.types.is_bool_dtype(values)
        or not pd.api.types.is_numeric_dtype(values)
        or unique <= 20
    )
    if categorical:
        string_values = values.astype(str).to_numpy()
        counts = pd.Series(string_values[valid]).value_counts()
        if len(counts) > max_categories:
            retained = set(counts.head(max_categories - 1).index)
            string_values = np.asarray(
                [value if value in retained else "Other" for value in string_values]
            )
        categories = sorted(pd.Series(string_values[valid]).unique())
        for index, category in enumerate(categories):
            mask = valid & (string_values == category)
            ax.scatter(
                frame.loc[mask, "component_1"],
                frame.loc[mask, "component_2"],
                s=8,
                alpha=0.55,
                linewidths=0,
                rasterized=True,
                color=CATEGORICAL[index % len(CATEGORICAL)],
                label=f"{category} (n={int(mask.sum()):,})",
            )
        ax.legend(
            frameon=False,
            fontsize=7,
            markerscale=1.8,
            bbox_to_anchor=(1.02, 1),
            loc="upper left",
        )
    else:
        points = ax.scatter(
            frame.loc[valid, "component_1"],
            frame.loc[valid, "component_2"],
            c=pd.to_numeric(values[valid]),
            cmap="viridis",
            s=8,
            alpha=0.6,
            linewidths=0,
            rasterized=True,
        )
        fig.colorbar(points, ax=ax, label=color_by)
    ax.set_title(f"Patient embeddings colored by {color_by}")
    ax.set_xlabel(f"{method.upper()} 1")
    ax.set_ylabel(f"{method.upper()} 2")
    clean_2d_axes(ax)
    fig.tight_layout()
    save_fig(fig, str(output_path))
    plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Project and plot an extract_patient_embeddings NPZ artifact."
    )
    parser.add_argument("--embeddings", required=True)
    parser.add_argument("--metadata", action="append", default=[])
    parser.add_argument("--subject-col", default="subject_id")
    parser.add_argument("--color-by", action="append", default=[])
    parser.add_argument("--split", default=None)
    parser.add_argument("--split-col", default="split")
    parser.add_argument("--method", choices=["umap", "tsne", "pca"], default="umap")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-neighbors", type=int, default=30)
    parser.add_argument("--min-dist", type=float, default=0.25)
    parser.add_argument("--perplexity", type=float, default=30.0)
    parser.add_argument(
        "--max-categories",
        type=int,
        default=20,
        help="Keep the most frequent categorical levels and combine the rest.",
    )
    parser.add_argument(
        "--max-points",
        type=int,
        default=None,
        help="Deterministically subsample before projection (useful for t-SNE).",
    )
    parser.add_argument(
        "--index-date-col",
        default=None,
        help="Derive treatment_year from this metadata date column.",
    )
    parser.add_argument(
        "--birth-date-col",
        default=None,
        help="Derive age_at_index using this column and --index-date-col.",
    )
    parser.add_argument("--output-dir", required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    embeddings, frame = load_embedding_frame(
        args.embeddings, args.metadata, subject_col=args.subject_col
    )
    if args.index_date_col is not None:
        if args.index_date_col not in frame:
            raise ValueError(
                f"Missing --index-date-col {args.index_date_col!r} in metadata."
            )
        index_dates = pd.to_datetime(frame[args.index_date_col], errors="coerce")
        frame["treatment_year"] = index_dates.dt.year.astype("Int64")
        if args.birth_date_col is not None:
            if args.birth_date_col not in frame:
                raise ValueError(
                    f"Missing --birth-date-col {args.birth_date_col!r} in metadata."
                )
            birth_dates = pd.to_datetime(frame[args.birth_date_col], errors="coerce")
            frame["age_at_index"] = (
                (index_dates - birth_dates).dt.total_seconds()
                / (365.2425 * 24 * 60 * 60)
            )
    if args.split is not None:
        if args.split_col not in frame:
            raise ValueError(f"Cannot filter: missing split column {args.split_col!r}.")
        mask = frame[args.split_col].astype(str).to_numpy() == str(args.split)
        frame = frame.loc[mask].reset_index(drop=True)
        embeddings = embeddings[mask]
    if args.max_points is not None:
        if args.max_points < 3:
            raise ValueError("--max-points must be at least 3.")
        if len(frame) > args.max_points:
            selected = (
                frame.sample(n=args.max_points, random_state=args.seed)
                .index.sort_values()
                .to_numpy()
            )
            frame = frame.loc[selected].reset_index(drop=True)
            embeddings = embeddings[selected]
    if len(frame) < 3:
        raise ValueError("At least three patients are required for a projection.")
    if args.max_categories < 2:
        raise ValueError("--max-categories must be at least 2.")

    coords, resolved_method = reduce_patient_embeddings(
        embeddings,
        method=args.method,
        seed=args.seed,
        n_neighbors=args.n_neighbors,
        min_dist=args.min_dist,
        perplexity=args.perplexity,
    )
    frame["component_1"] = coords[:, 0]
    frame["component_2"] = coords[:, 1]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_dir / f"patient_{resolved_method}_coordinates.csv", index=False)
    color_columns = args.color_by or (
        [args.split_col] if args.split_col in frame else []
    )
    for column in color_columns:
        safe_name = "".join(
            character if character.isalnum() or character in "-_" else "_"
            for character in column
        )
        plot_colored_coordinates(
            frame,
            color_by=column,
            method=resolved_method,
            output_path=output_dir / f"patient_{resolved_method}_by_{safe_name}.png",
            max_categories=args.max_categories,
        )
    print(
        f"Wrote {len(frame):,} {resolved_method.upper()} coordinates and "
        f"{len(color_columns)} plot(s) to {output_dir}."
    )


if __name__ == "__main__":
    main()
