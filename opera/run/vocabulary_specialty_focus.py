"""Render contact-specialty neighbourhoods from an existing vocabulary atlas.

Uses the saved original token vectors for cosine neighbours and the saved atlas
coordinates for display. It does not reload checkpoints or rerun UMAP.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import font_manager
import pandas as pd

from opera.evaluation.vocabulary_embeddings import nearest_token_neighbors
from opera.visualization.vocabulary_atlas import plot_vocabulary_focus


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_") or "token"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--atlas-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--stage", default="time2vec")
    parser.add_argument("--family", default="CONTACT SPECIALTY")
    parser.add_argument("--focus-tokens", help="Comma-separated exact token names; default: first six in token-id order")
    parser.add_argument("--top-k", type=int, default=10)
    args = parser.parse_args()
    if not 1 <= args.top_k <= 10:
        parser.error("--top-k must be between 1 and 10 for the focus figure")
    if args.output_dir.exists():
        parser.error(f"Output directory already exists: {args.output_dir}")
    atlas_manifest = json.loads((args.atlas_dir / "analysis_metadata.json").read_text(encoding="utf-8"))
    if atlas_manifest.get("atlas_stage") != args.stage:
        parser.error("Requested embedding stage differs from the saved atlas projection stage")
    embeddings = pd.read_csv(args.atlas_dir / f"vocabulary_embeddings_{args.stage}.csv")
    coordinates = pd.read_csv(args.atlas_dir / "vocabulary_atlas_coordinates.csv")
    if (embeddings["token_id"].duplicated().any() or coordinates["token_id"].duplicated().any()
            or not embeddings[["token_id", "token"]].equals(
                coordinates[["token_id", "token"]])):
        parser.error("Saved embeddings and coordinates do not have the same ordered vocabulary")
    family_names = embeddings["token_family"].dropna().astype(str).unique().tolist()
    matching = [name for name in family_names
                if name.replace("_", " ").casefold() == args.family.replace("_", " ").casefold()]
    if len(matching) != 1:
        parser.error(f"Family {args.family!r} is not unique/present; available: {sorted(family_names)}")
    family = matching[0]
    specialty = embeddings.loc[embeddings["token_family"].eq(family),
                               ["token_id", "token", "token_family"]].sort_values("token_id")
    if args.focus_tokens:
        selected = [item.strip() for item in args.focus_tokens.split(",") if item.strip()]
        if len(selected) != len(set(selected)) or not set(selected).issubset(set(specialty["token"])):
            parser.error("Focus tokens must be unique exact members of the selected family")
    else:
        selected = specialty["token"].head(6).tolist()
    try:
        font_manager.findfont(font_manager.FontProperties(family="TeX Gyre Pagella"),
                              fallback_to_default=False)
        figure_font = "TeX Gyre Pagella"
    except ValueError:
        figure_font = "DejaVu Serif"
        print("TeX Gyre Pagella is unavailable; specialty figures use DejaVu Serif.")
    neighbors = nearest_token_neighbors(embeddings, selected, top_k=args.top_k)
    diagnosis_pool = embeddings.loc[
        embeddings["token_family"].isin(["DIAGNOSIS", "PATHOLOGY"])
        | embeddings["token"].isin(selected)
    ].copy()
    diagnosis_neighbors = nearest_token_neighbors(
        diagnosis_pool, selected, top_k=args.top_k + len(selected)
    )
    diagnosis_neighbors = diagnosis_neighbors.loc[
        diagnosis_neighbors["neighbor_family"].isin(["DIAGNOSIS", "PATHOLOGY"])
    ].groupby("query_token", sort=False).head(args.top_k).copy()
    diagnosis_neighbors["neighbor_rank"] = diagnosis_neighbors.groupby("query_token").cumcount() + 1
    args.output_dir.mkdir(parents=True)
    specialty.to_csv(args.output_dir / "contact_specialty_token_catalogue.csv", index=False)
    neighbors.to_csv(args.output_dir / "contact_specialty_cosine_neighbors.csv", index=False)
    diagnosis_neighbors.to_csv(args.output_dir / "contact_specialty_diagnosis_neighbors.csv", index=False)
    with mpl.rc_context({"font.family": figure_font, "mathtext.fontset": "dejavuserif"}):
        for token in selected:
            fig = plot_vocabulary_focus(
                coordinates, neighbors, query_token=token,
                save_path=str(args.output_dir / f"focus_{_safe_name(token)}.png"),
            )
            plt.close(fig)
    (args.output_dir / "manifest.json").write_text(json.dumps({
        "atlas_dir": str(args.atlas_dir.resolve()), "stage": args.stage,
        "family": family, "focus_tokens": selected, "top_k": args.top_k,
        "figure_font": figure_font,
        "interpretation": "Cosine-ranked token neighbours; 2D atlas coordinates are descriptive, not clinical validation",
    }, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {len(selected)} specialty focus figures to {args.output_dir}")


if __name__ == "__main__":
    main()
