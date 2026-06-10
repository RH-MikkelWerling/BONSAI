"""Patient-level transfer analyses for OPERA model contrasts.

These helpers turn a model-level claim such as "joint OPERA improves over
per-cohort OPERA" into patient-level evidence: who gained, whether their
nearest neighbours came from other diseases, and which shared clinical
concepts describe the beneficiaries.
"""

from __future__ import annotations

from typing import Iterable, Optional

import numpy as np
import pandas as pd


def _as_frame(data: pd.DataFrame | str) -> pd.DataFrame:
    if isinstance(data, pd.DataFrame):
        return data.copy()
    if str(data).lower().endswith((".parquet", ".pq")):
        return pd.read_parquet(data)
    return pd.read_csv(data)


def _clip_prob(values: pd.Series | np.ndarray) -> np.ndarray:
    return np.clip(np.asarray(values, dtype=float), 1e-6, 1.0 - 1e-6)


def compute_patient_prediction_gain(
    baseline_predictions: pd.DataFrame | str,
    comparator_predictions: pd.DataFrame | str,
    *,
    subject_col: str = "subject_id",
    label_col: str = "label",
    probability_col: str = "probability",
    weight_col: Optional[str] = None,
    contrast_name: str = "comparator_minus_baseline",
) -> pd.DataFrame:
    """Compute per-patient loss reduction for a model contrast.

    Positive gain means the comparator predicted the observed binary endpoint
    better than the baseline. If an IPCW or other analytic weight is supplied,
    weighted Brier and log-loss gains are included while the unweighted
    contribution remains visible.
    """
    base = _as_frame(baseline_predictions)
    comp = _as_frame(comparator_predictions)
    base_cols = [subject_col, label_col, probability_col]
    comp_cols = [subject_col, probability_col]
    if weight_col and weight_col in base.columns:
        base_cols.append(weight_col)
    merged = base[base_cols].merge(
        comp[comp_cols],
        on=subject_col,
        how="inner",
        suffixes=("_baseline", "_comparator"),
    )
    if merged.empty:
        return pd.DataFrame()

    y = np.asarray(merged[label_col], dtype=float)
    p_base = _clip_prob(merged[f"{probability_col}_baseline"])
    p_comp = _clip_prob(merged[f"{probability_col}_comparator"])
    brier_base = (y - p_base) ** 2
    brier_comp = (y - p_comp) ** 2
    logloss_base = -(y * np.log(p_base) + (1.0 - y) * np.log(1.0 - p_base))
    logloss_comp = -(y * np.log(p_comp) + (1.0 - y) * np.log(1.0 - p_comp))

    out = pd.DataFrame(
        {
            subject_col: merged[subject_col].values,
            "label": y.astype(int),
            "baseline_probability": p_base,
            "comparator_probability": p_comp,
            "brier_baseline": brier_base,
            "brier_comparator": brier_comp,
            "brier_gain": brier_base - brier_comp,
            "logloss_baseline": logloss_base,
            "logloss_comparator": logloss_comp,
            "logloss_gain": logloss_base - logloss_comp,
            "contrast": contrast_name,
        }
    )
    if weight_col and weight_col in merged.columns:
        weights = np.asarray(merged[weight_col], dtype=float)
        out[weight_col] = weights
        out["weighted_brier_gain"] = weights * out["brier_gain"]
        out["weighted_logloss_gain"] = weights * out["logloss_gain"]
    return out


def compute_embedding_neighbor_support(
    embeddings: pd.DataFrame | str,
    metadata: pd.DataFrame | str,
    *,
    subject_col: str = "subject_id",
    cohort_col: str = "cohort",
    label_col: Optional[str] = "label",
    feature_cols: Optional[Iterable[str]] = None,
    k: int = 20,
) -> pd.DataFrame:
    """Describe each patient's nearest-neighbour support in embedding space.

    The embedding table must contain `subject_id` plus numeric embedding
    columns. The metadata table can include cohort, labels, EHR features, and
    harmonized registry concepts. Missing optional features are skipped, which
    lets the same function work when RKKP columns are cohort-specific.
    """
    emb = _as_frame(embeddings)
    meta = _as_frame(metadata)
    merged = emb.merge(meta, on=subject_col, how="inner")
    if merged.empty:
        return pd.DataFrame()
    exclude = {subject_col, cohort_col}
    if label_col:
        exclude.add(label_col)
    meta_cols = set(meta.columns)
    embed_cols = [
        col
        for col in emb.columns
        if col != subject_col
        and col not in meta_cols
        and pd.api.types.is_numeric_dtype(emb[col])
    ]
    if not embed_cols:
        embed_cols = [
            col
            for col in emb.columns
            if col != subject_col and pd.api.types.is_numeric_dtype(emb[col])
        ]
    if not embed_cols:
        raise ValueError("No numeric embedding columns found.")

    x = merged[embed_cols].to_numpy(dtype=float)
    x = x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)
    sim = x @ x.T
    np.fill_diagonal(sim, -np.inf)
    k_eff = min(k, max(len(merged) - 1, 0))
    if k_eff == 0:
        return pd.DataFrame()
    neigh_idx = np.argpartition(-sim, kth=k_eff - 1, axis=1)[:, :k_eff]

    rows = []
    requested_features = [c for c in (feature_cols or []) if c in merged.columns]
    cohorts = (
        merged[cohort_col].astype(str).to_numpy() if cohort_col in merged else None
    )
    labels = merged[label_col].to_numpy() if label_col and label_col in merged else None

    for i, idx in enumerate(neigh_idx):
        order = idx[np.argsort(-sim[i, idx])]
        row = {
            subject_col: merged.iloc[i][subject_col],
            "n_neighbors": int(len(order)),
            "mean_neighbor_similarity": float(np.mean(sim[i, order])),
        }
        if cohorts is not None:
            cross = cohorts[order] != cohorts[i]
            row["cross_cohort_neighbor_fraction"] = float(np.mean(cross))
            row["dominant_neighbor_cohort"] = pd.Series(cohorts[order]).mode().iloc[0]
        if labels is not None:
            same_label = labels[order] == labels[i]
            row["neighbor_outcome_concordance"] = float(np.mean(same_label))
            if cohorts is not None and np.any(cross):
                row["cross_cohort_outcome_concordance"] = float(
                    np.mean(same_label[cross])
                )
        for feature in requested_features:
            values = merged.iloc[order][feature]
            if pd.api.types.is_numeric_dtype(values):
                row[f"neighbor_mean_{feature}"] = float(
                    pd.to_numeric(values, errors="coerce").mean()
                )
                row[f"patient_{feature}"] = merged.iloc[i][feature]
            else:
                mode = values.dropna().mode()
                row[f"neighbor_mode_{feature}"] = (
                    mode.iloc[0] if not mode.empty else np.nan
                )
                row[f"patient_{feature}"] = merged.iloc[i][feature]
        rows.append(row)
    return pd.DataFrame(rows)


def build_patient_transfer_table(
    baseline_predictions: pd.DataFrame | str,
    comparator_predictions: pd.DataFrame | str,
    embeddings: pd.DataFrame | str,
    metadata: pd.DataFrame | str,
    *,
    subject_col: str = "subject_id",
    contrast_name: str = "joint_opera_minus_per_cohort_opera",
    feature_cols: Optional[Iterable[str]] = None,
    k: int = 20,
) -> pd.DataFrame:
    """Combine prediction gains and embedding-neighbour descriptors."""
    gains = compute_patient_prediction_gain(
        baseline_predictions,
        comparator_predictions,
        subject_col=subject_col,
        contrast_name=contrast_name,
    )
    support = compute_embedding_neighbor_support(
        embeddings,
        metadata,
        subject_col=subject_col,
        feature_cols=feature_cols,
        k=k,
    )
    if gains.empty:
        return gains
    if support.empty:
        return gains
    return gains.merge(support, on=subject_col, how="left")


def summarize_beneficiary_profile(
    patient_transfer: pd.DataFrame,
    *,
    gain_col: str = "brier_gain",
    top_fraction: float = 0.25,
    descriptor_cols: Optional[Iterable[str]] = None,
) -> pd.DataFrame:
    """Compare top-gain patients with the rest of the evaluated patients."""
    if patient_transfer.empty or gain_col not in patient_transfer.columns:
        return pd.DataFrame()
    frame = patient_transfer.sort_values(gain_col, ascending=False).copy()
    n_top = max(1, int(np.ceil(len(frame) * top_fraction)))
    frame["beneficiary_group"] = "other"
    frame.iloc[:n_top, frame.columns.get_loc("beneficiary_group")] = "top_gain"
    if descriptor_cols is None:
        descriptor_cols = [
            col
            for col in frame.columns
            if col.startswith("cross_")
            or col.startswith("neighbor_")
            or col.startswith("patient_")
        ]
    rows = []
    for col in descriptor_cols:
        if col not in frame.columns:
            continue
        for group, values in frame.groupby("beneficiary_group")[col]:
            if pd.api.types.is_numeric_dtype(values):
                rows.append(
                    {
                        "descriptor": col,
                        "beneficiary_group": group,
                        "summary": "mean",
                        "value": float(pd.to_numeric(values, errors="coerce").mean()),
                        "n": int(values.notna().sum()),
                    }
                )
            else:
                mode = values.dropna().mode()
                rows.append(
                    {
                        "descriptor": col,
                        "beneficiary_group": group,
                        "summary": "mode",
                        "value": mode.iloc[0] if not mode.empty else np.nan,
                        "n": int(values.notna().sum()),
                    }
                )
    return pd.DataFrame(rows)


def compute_atypicality_scores(
    embeddings: pd.DataFrame | str,
    metadata: pd.DataFrame | str,
    *,
    subject_col: str = "subject_id",
    cohort_col: str = "cohort",
    feature_cols: Optional[Iterable[str]] = None,
    min_cohort_size: int = 10,
) -> pd.DataFrame:
    """Compute patient atypicality in the baseline embedding space.

    `atypicality_own` is the within-cohort z-scored distance to the patient's
    own disease centroid. `atypicality_nearest` is the globally z-scored
    distance to the nearest valid disease centroid. Distances are always
    computed in the full high-dimensional embedding space.
    """
    emb = _as_frame(embeddings)
    meta = _as_frame(metadata)
    feature_set = set(feature_cols or [])
    excluded = {subject_col, cohort_col, *feature_set}
    embed_cols = [
        col
        for col in emb.columns
        if col not in excluded and pd.api.types.is_numeric_dtype(emb[col])
    ]
    if not embed_cols:
        raise ValueError("No numeric embedding columns found for atypicality scoring.")

    meta_cols = [subject_col]
    if cohort_col in meta.columns:
        meta_cols.append(cohort_col)
    merged = emb.merge(
        meta[meta_cols].drop_duplicates(subject_col),
        on=subject_col,
        how="left",
    ).reset_index(drop=True)
    if cohort_col not in merged.columns and f"{cohort_col}_x" in merged.columns:
        merged[cohort_col] = merged[f"{cohort_col}_x"]
    if cohort_col not in merged.columns:
        raise ValueError(f"Missing required cohort column {cohort_col!r}.")

    vectors = merged[embed_cols].to_numpy(dtype=float)
    cohorts = merged[cohort_col].astype(str).to_numpy()
    valid_centroids = {}
    for cohort, idx in merged.groupby(cohort_col, dropna=False).groups.items():
        if len(idx) >= min_cohort_size:
            valid_centroids[str(cohort)] = vectors[np.asarray(list(idx))].mean(axis=0)
    if not valid_centroids:
        raise ValueError(
            f"No cohorts have at least min_cohort_size={min_cohort_size} patients."
        )

    centroid_names = np.asarray(list(valid_centroids.keys()), dtype=object)
    centroid_matrix = np.vstack([valid_centroids[name] for name in centroid_names])
    distances = np.linalg.norm(
        vectors[:, None, :] - centroid_matrix[None, :, :], axis=2
    )
    nearest_idx = np.argmin(distances, axis=1)
    nearest_raw = distances[np.arange(len(merged)), nearest_idx]
    nearest_mean = float(np.nanmean(nearest_raw))
    nearest_std = float(np.nanstd(nearest_raw))
    nearest_z = (
        (nearest_raw - nearest_mean) / nearest_std
        if nearest_std > 0
        else np.zeros_like(nearest_raw, dtype=float)
    )

    own_raw = np.full(len(merged), np.nan, dtype=float)
    for name, centroid in valid_centroids.items():
        mask = cohorts == name
        own_raw[mask] = np.linalg.norm(vectors[mask] - centroid, axis=1)

    own_z = np.full(len(merged), np.nan, dtype=float)
    for name in np.unique(cohorts):
        mask = cohorts == name
        values = own_raw[mask]
        valid = np.isfinite(values)
        if not valid.any():
            continue
        mean = float(np.nanmean(values))
        std = float(np.nanstd(values))
        own_z[mask] = (values - mean) / std if std > 0 else 0.0

    return pd.DataFrame(
        {
            subject_col: merged[subject_col].values,
            cohort_col: cohorts,
            "atypicality_own": own_z,
            "atypicality_nearest": nearest_z,
            "nearest_centroid_cohort": centroid_names[nearest_idx],
            "own_centroid_distance_raw": own_raw,
            "nearest_centroid_distance_raw": nearest_raw,
        }
    )
