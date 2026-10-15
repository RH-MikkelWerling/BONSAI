"""Projected patient representations for the embedding and calibration maps.

Each model stage supplies index-time embeddings (``subject_ids``,
``embeddings``, ``splits``), either from evaluation (``--export-embeddings``,
fine-tuned models) or from ``extract_patient_embeddings`` (pretrained
encoders). For every stage the projection -- standardisation, PCA and UMAP
(PCA when UMAP is unavailable) -- is fitted on tuning patients and then applied
unchanged to held-out patients, so held-out patients never shape the map. Only
held-out patients present in every stage are drawn.

    python -m opera.run.build_paper_embedding_tables --config <embedding.yaml> \
        --patient-metadata <metadata.parquet> --output-dir <plot_tables>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from opera.run.analyse_locked_predictions import read_table
from opera.visualization.outcome_names import outcome_display_name
from opera.visualization.paper_taxonomy import disease_group_name


def load_stage(paths) -> pd.DataFrame:
    """Embeddings of one stage from one or more npz exports."""
    frames = []
    for path in [paths] if isinstance(paths, (str, Path)) else paths:
        data = np.load(path, allow_pickle=False)
        frames.append(pd.DataFrame({"subject_id": data["subject_ids"],
                                    "split": data["splits"].astype(str),
                                    "embedding": list(data["embeddings"])}))
    frame = pd.concat(frames, ignore_index=True)
    if frame.subject_id.duplicated().any():
        raise ValueError("A patient appears twice within one stage")
    return frame


def project(frame: pd.DataFrame, *, method="umap", seed=0) -> pd.DataFrame:
    """2-D coordinates fitted on tuning patients, applied to held-out patients."""
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    tuning = frame.loc[frame.split == "tuning"]
    held = frame.loc[frame.split == "held_out"]
    if tuning.empty or held.empty:
        raise ValueError("Projection needs tuning (fit) and held-out (display) patients")
    fit = np.stack(tuning.embedding)
    scaler = StandardScaler().fit(fit)
    pca = PCA(n_components=min(50, fit.shape[1], len(fit) - 1), random_state=seed)
    reduced_fit = pca.fit_transform(scaler.transform(fit))
    reduced_held = pca.transform(scaler.transform(np.stack(held.embedding)))
    used = "pca"
    if method == "umap":
        try:
            from umap import UMAP

            reducer = UMAP(n_components=2, n_neighbors=30, min_dist=0.3,
                           random_state=seed).fit(reduced_fit)
            xy, used = reducer.transform(reduced_held), "umap"
        except ImportError:
            xy = reduced_held[:, :2]
    else:
        xy = reduced_held[:, :2]
    return pd.DataFrame({"subject_id": held.subject_id.to_numpy(), "x": xy[:, 0],
                         "y": xy[:, 1], "projection": used})


def embedding_comparison(stages: dict, metadata: pd.DataFrame, *, method="umap",
                         group_map=None):
    """Long table of held-out coordinates per stage with shared colour columns."""
    coordinates = {label: project(load_stage(paths), method=method)
                   for label, paths in stages.items()}
    common = set.intersection(*(set(c.subject_id) for c in coordinates.values()))
    meta = metadata.set_index("subject_id")
    rows = []
    for label, coords in coordinates.items():
        coords = coords.loc[coords.subject_id.isin(common)].copy()
        info = meta.loc[coords.subject_id]
        coords["model"] = label
        coords["patient_id"] = coords.subject_id
        coords["disease"] = info.cohort_grouped.map(disease_group_name).to_numpy()
        groups = {str(k).lower(): v for k, v in (group_map or {}).items()}
        coords["ipi_group"] = (info["IPI"].astype(str).str.strip().str.lower().map(groups)
                               .to_numpy() if "IPI" in info else np.nan)
        coords["age"] = info.age_at_index.to_numpy()
        coords["sequence_length"] = info.history_event_count.to_numpy()
        rows.append(coords)
    return pd.concat(rows, ignore_index=True).drop(columns="subject_id"), coordinates


def calibration_map_inputs(coords: pd.DataFrame, predictions: pd.DataFrame, outcomes,
                           *, horizon=365):
    """Coordinates and per-outcome (time, event, predicted risk) on one patient set.

    Only patients evaluated for every chosen outcome are kept, so all columns
    of the figure describe the same people.
    """
    column = f"cif_{horizon}d"
    data = predictions.loc[predictions.outcome.isin(outcomes)]
    eligible = data.groupby("subject_id").outcome.nunique()
    keep = set(eligible[eligible == len(outcomes)].index) & set(coords.subject_id)
    coords = coords.loc[coords.subject_id.isin(keep)].sort_values("subject_id")
    order = coords.subject_id.to_numpy()
    payload = {}
    for outcome in outcomes:
        part = data.loc[data.outcome == outcome].set_index("subject_id").loc[order]
        payload[outcome_display_name(outcome)] = [
            part.time_days.astype(float).tolist(), part.event.astype(int).tolist(),
            part[column].astype(float).tolist(), None]
    return coords[["x", "y"]], payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--patient-metadata", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    metadata = read_table(args.patient_metadata)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    comparison, coordinates = embedding_comparison(
        config["stages"], metadata, method=config.get("method", "umap"),
        group_map=config.get("ipi_group_map"))
    comparison.to_csv(args.output_dir / "matched_patient_embeddings.csv", index=False)
    maps = config["calibration_maps"]
    coords, payload = calibration_map_inputs(
        coordinates[maps["stage"]], read_table(maps["predictions"]), maps["outcomes"],
        horizon=maps.get("horizon_days", 365))
    coords.to_csv(args.output_dir / "patient_coordinates.csv", index=False)
    (args.output_dir / "calibration_map_outcomes.json").write_text(json.dumps(payload))
    print(json.dumps({"stages": list(config["stages"]),
                      "patients_compared": int(comparison.patient_id.nunique()),
                      "map_patients": len(coords),
                      "projection": sorted(set(comparison.projection))}))


if __name__ == "__main__":
    main()
