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


def _patient_ids(frame):
    frame = frame.copy()
    if frame.subject_id.isna().any():
        raise ValueError("Embedding patient identifiers must be complete")
    frame["subject_id"] = frame.subject_id.astype(str)
    return frame


def load_stage(paths) -> pd.DataFrame:
    """Embeddings of one stage from one or more npz exports."""
    frames = []
    for path in [paths] if isinstance(paths, (str, Path)) else paths:
        data = np.load(path, allow_pickle=False)
        frames.append(pd.DataFrame({"subject_id": data["subject_ids"],
                                    "split": data["splits"].astype(str),
                                    "embedding": list(data["embeddings"])}))
    frame = _patient_ids(pd.concat(frames, ignore_index=True))
    if frame.subject_id.duplicated().any():
        raise ValueError("A patient appears twice within one stage")
    return frame


def project(frame: pd.DataFrame, *, method="umap", seed=0, rehearse=False, routed_vectors=None) -> pd.DataFrame:
    """2-D coordinates fitted on tuning patients, applied to held-out patients.

    ``rehearse``: no held-out data; fit on half of the tuning patients and
    display the other half (fixed by subject id), to test the pipeline."""
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    tuning = frame.loc[frame.split == "tuning"]
    held = frame.loc[frame.split == "held_out"]
    if rehearse:
        if not held.empty:
            raise ValueError("Rehearsal mode is for tuning-only exports")
        shown = pd.util.hash_pandas_object(tuning.subject_id.astype(str), index=False) % 2 == 1
        tuning, held = tuning.loc[~shown.to_numpy()], tuning.loc[shown.to_numpy()]
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
    result = pd.DataFrame({"subject_id": held.subject_id.to_numpy(), "x": xy[:, 0],
                           "y": xy[:, 1], "projection": used})
    for outcome, path in (routed_vectors or {}).items():
        marker = json.loads(Path(path).with_suffix('.json').read_text())
        if marker.get('target_outcome') != outcome:
            raise ValueError('Routed vector target outcome does not match the requested panel')
        if set(held.split) != {marker.get('split')}:
            raise ValueError('Routed vector split does not match displayed patients')
        vectors = np.load(path, allow_pickle=False)
        ids = pd.Index(vectors['subject_ids'].astype(str))
        if ids.has_duplicates:
            raise ValueError('Duplicate IDs in routed vector export')
        positions = ids.get_indexer(held.subject_id.astype(str))
        if (positions < 0).any():
            raise ValueError('Routed vectors do not cover displayed patients')
        shared, adapted = vectors['shared'][positions], vectors['adapted'][positions]
        if not np.allclose(shared, np.stack(held.embedding), atol=1e-5, rtol=1e-4):
            raise ValueError('Router arrows require the same checkpoint shared embedding as the map')
        reduced = pca.transform(scaler.transform(adapted))
        after = reducer.transform(reduced) if used == 'umap' else reduced[:, :2]
        result[f'routed:{outcome}:x'] = after[:, 0]
        result[f'routed:{outcome}:y'] = after[:, 1]
    return result


def embedding_comparison(stages: dict, metadata: pd.DataFrame, *, method="umap",
                         group_map=None, rehearse=False):
    """Long table of held-out coordinates per stage with shared colour columns."""
    coordinates = {label: project(load_stage(paths), method=method, rehearse=rehearse)
                   for label, paths in stages.items()}
    common = set.intersection(*(set(c.subject_id) for c in coordinates.values()))
    meta = _patient_ids(metadata).set_index("subject_id")
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
    coords, predictions = _patient_ids(coords), _patient_ids(predictions)
    column = f"cif_{horizon}d"
    if not outcomes or len(set(outcomes)) != len(outcomes):
        raise ValueError('Calibration outcomes must be nonempty and unique')
    if coords.subject_id.duplicated().any():
        raise ValueError('Duplicate patient coordinates')
    data = predictions.loc[predictions.outcome.isin(outcomes)].drop_duplicates()
    if data.duplicated(['subject_id', 'outcome']).any():
        raise ValueError('Conflicting prediction rows for calibration map')
    eligible = data.groupby("subject_id").outcome.nunique()
    keep = set(eligible[eligible == len(outcomes)].index) & set(coords.subject_id)
    if not keep:
        raise ValueError('No common evaluated patients for calibration map outcomes')
    coords = coords.loc[coords.subject_id.isin(keep)].sort_values("subject_id")
    order = coords.subject_id.to_numpy()
    payload = {}
    for outcome in outcomes:
        part = data.loc[data.outcome == outcome].set_index("subject_id").loc[order]
        payload[outcome_display_name(outcome)] = [
            part.time_days.astype(float).tolist(), part.event.astype(int).tolist(),
            part[column].astype(float).tolist(),
            (coords[[f'routed:{outcome}:x', f'routed:{outcome}:y']].to_numpy(float).tolist()
             if f'routed:{outcome}:x' in coords else None)]
    return coords[["x", "y"]], payload


def embedding_gallery_inputs(coords: pd.DataFrame, predictions: pd.DataFrame,
                             metadata: pd.DataFrame | None = None, *, horizon=365):
    """Every outcome's (time, event, predicted risk) on the joint model's projection.

    Each outcome keeps its own evaluated patients; the projection is shared.
    With ``metadata`` the coordinates also carry each patient's disease group.
    """
    from opera.visualization.paper_taxonomy import DISPLAY_GROUPS, outcome_endpoint_groups

    coords, predictions = _patient_ids(coords), _patient_ids(predictions)
    if metadata is not None:
        metadata = _patient_ids(metadata)
    domains = outcome_endpoint_groups()
    domain_order = {name: i for i, name in enumerate(
        yaml.safe_load(Path(DISPLAY_GROUPS).read_text()))}
    risk = f"cif_{horizon}d"
    columns = ["subject_id", "outcome", "time_days", "event",
               *([risk] if risk in predictions else [])]
    data = predictions.loc[predictions.subject_id.isin(set(coords.subject_id)),
                           columns].rename(columns={risk: "risk"})
    missing = sorted(set(data.outcome) - set(domains))
    if missing:
        raise ValueError(f"Outcomes without a clinical domain: {missing}")
    data["domain"] = data.outcome.map(domains)
    data["domain_order"] = data.domain.map(domain_order)
    data["outcome"] = data.outcome.map(outcome_display_name)
    coords = coords[["subject_id", "x", "y"]].copy()
    if metadata is not None and "cohort_grouped" in metadata:
        group = metadata.set_index("subject_id").cohort_grouped
        coords["disease"] = coords.subject_id.map(group).map(disease_group_name)
    data = data.drop_duplicates()
    if data.duplicated(['subject_id', 'outcome']).any():
        raise ValueError('Conflicting prediction rows for embedding gallery')
    return coords, data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--patient-metadata", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--rehearse-on-tuning", action="store_true",
                        help="tuning-only exports: fit on half the tuning patients, "
                             "show the other half (pipeline rehearsal)")
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    metadata = read_table(args.patient_metadata)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    comparison, coordinates = embedding_comparison(
        config["stages"], metadata, method=config.get("method", "umap"),
        group_map=config.get("ipi_group_map"), rehearse=args.rehearse_on_tuning)
    maps = config["calibration_maps"]
    map_coords = coordinates[maps["stage"]]
    if maps.get('routed_vectors'):
        map_coords = project(load_stage(config['stages'][maps['stage']]),
                             method=config.get('method', 'umap'),
                             rehearse=args.rehearse_on_tuning,
                             routed_vectors=maps['routed_vectors'])
    coords, payload = calibration_map_inputs(
        map_coords, read_table(maps["predictions"]), maps["outcomes"],
        horizon=maps.get("horizon_days", 365))
    # Supplementary gallery: every outcome on the same projection, by domain.
    gallery = config.get("gallery", maps)
    gallery_coords, gallery_outcomes = embedding_gallery_inputs(
        coordinates[gallery["stage"]], read_table(gallery["predictions"]), metadata,
        horizon=gallery.get("horizon_days", 365))
    selected = [outcome_display_name(name) for name in maps["outcomes"]]
    gallery_outcomes["highlight_order"] = gallery_outcomes.outcome.map(
        {name: i for i, name in enumerate(selected)})
    # Finish validation before replacing any existing derived tables.
    comparison.to_csv(args.output_dir / "matched_patient_embeddings.csv", index=False)
    coords.to_csv(args.output_dir / "patient_coordinates.csv", index=False)
    gallery_coords.to_csv(args.output_dir / "gallery_coordinates.csv", index=False)
    gallery_outcomes.to_parquet(args.output_dir / "embedding_gallery.parquet", index=False)
    (args.output_dir / "calibration_map_outcomes.json").write_text(json.dumps(payload))
    print(json.dumps({"stages": list(config["stages"]),
                      "patients_compared": int(comparison.patient_id.nunique()),
                      "map_patients": len(coords),
                      "projection": sorted(set(comparison.projection))}))


if __name__ == "__main__":
    main()
