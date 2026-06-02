"""
Scalable stratified survival sampler for OPERA contrastive learning.

Random batches produce mostly easy pairs: patients who differ substantially in
survival time and are already well-separated by the model. Gradient signal comes
mostly from the few hard pairs the batch happens to contain by chance.

This sampler builds per-patient quantile ranks for all configured outcomes,
projects those ranks to two dimensions, and samples inversely to the frequency
of each projected 4x4 bucket. The bucket space is fixed at 16 cells regardless
of outcome count, so stratification still works when K is large.
"""

from __future__ import annotations

from collections import Counter
from typing import Dict, List

import numpy as np
import torch
from torch.utils.data import WeightedRandomSampler, ConcatDataset


def _extract_patient_times(
    dataset,
    outcome_names: List[str],
) -> List[Dict[str, float]]:
    """
    Walk a dataset (or ConcatDataset of ContrastiveDatasets) and return,
    for each patient in order, a dict {outcome_name: time_days or nan}.
    """
    sub_datasets = dataset.datasets if isinstance(dataset, ConcatDataset) else [dataset]

    all_times: List[Dict[str, float]] = []
    for ds in sub_datasets:
        for subject in ds.subjects:
            sid = subject["subject_id"]
            patient_times: Dict[str, float] = {}
            for name in outcome_names:
                rec = ds.outcome_dicts.get(name, {}).get(sid)
                if rec is not None:
                    t = rec.get("time_days", float("nan"))
                    patient_times[name] = float(t) if t is not None else float("nan")
                else:
                    patient_times[name] = float("nan")
            all_times.append(patient_times)

    return all_times


def _quantile_rank_matrix(
    all_times: List[Dict[str, float]],
    outcome_names: List[str],
    n_quantiles: int,
) -> np.ndarray:
    """Build an N x K matrix of observed quantile ranks plus missing sentinel."""
    n = len(all_times)
    k = len(outcome_names)
    sentinel = float(n_quantiles)
    ranks = np.full((n, k), sentinel, dtype=np.float32)

    for col, name in enumerate(outcome_names):
        times = np.array([pt[name] for pt in all_times], dtype=np.float64)
        valid = np.isfinite(times)
        if valid.sum() == 0:
            continue

        if valid.sum() < n_quantiles:
            ranks[valid, col] = 0.0
            continue

        qs = np.linspace(0.0, 1.0, n_quantiles + 1)[1:-1]
        boundaries = np.nanquantile(times[valid], qs)
        ranks[valid, col] = np.searchsorted(
            boundaries, times[valid], side="right"
        )

    return ranks


def _standardize_rank_matrix(ranks: np.ndarray, n_quantiles: int) -> np.ndarray:
    """
    Standardize observed ranks column-wise while preserving missingness.

    Missing entries are excluded from column moments and then encoded as 0.0,
    so they contribute no projection signal instead of acting as large outliers.
    """
    sentinel = float(n_quantiles)
    standardized = ranks.astype(np.float32, copy=True)

    for col in range(standardized.shape[1]):
        valid = standardized[:, col] != sentinel
        if not valid.any():
            standardized[:, col] = 0.0
            continue

        values = standardized[valid, col]
        mean = float(values.mean())
        std = float(values.std())
        standardized[valid, col] = 0.0 if std <= 1e-12 else (values - mean) / std
        standardized[~valid, col] = 0.0

    return standardized


def _project_rank_matrix(ranks: np.ndarray, n_quantiles: int) -> np.ndarray:
    """
    Project rank matrix to two dimensions.

    For K <= 2, use raw rank columns directly. If sklearn is unavailable, fall
    back to the same raw-column projection.
    """
    n, k = ranks.shape
    if n == 0 or k == 0:
        return np.zeros((n, 2), dtype=np.float32)
    if k == 1:
        return np.repeat(ranks[:, :1], 2, axis=1).astype(np.float32)
    if k == 2:
        return ranks[:, :2].astype(np.float32)

    standardized = _standardize_rank_matrix(ranks, n_quantiles)
    if np.allclose(standardized, 0.0):
        return np.zeros((n, 2), dtype=np.float32)

    try:
        from sklearn.decomposition import TruncatedSVD

        return TruncatedSVD(n_components=2, random_state=0).fit_transform(
            standardized
        ).astype(np.float32)
    except Exception:
        return ranks[:, :2].astype(np.float32)


def _quartile_bins(values: np.ndarray) -> np.ndarray:
    """Assign values to quartile bins 0..3."""
    if len(values) == 0:
        return np.array([], dtype=np.int64)
    boundaries = np.percentile(values, [25, 50, 75])
    bins = np.searchsorted(boundaries, values, side="right")
    return np.clip(bins, 0, 3).astype(np.int64)


def _projected_buckets(
    all_times: List[Dict[str, float]],
    outcome_names: List[str],
    n_quantiles: int,
) -> List[tuple]:
    ranks = _quantile_rank_matrix(all_times, outcome_names, n_quantiles)
    projection = _project_rank_matrix(ranks, n_quantiles)
    pc1_bins = _quartile_bins(projection[:, 0])
    pc2_bins = _quartile_bins(projection[:, 1])
    return list(zip(pc1_bins.tolist(), pc2_bins.tolist()))


def build_stratified_sampler(
    dataset,
    outcome_names: List[str],
    n_quantiles: int = 4,
    missing_bucket: bool = True,
) -> WeightedRandomSampler:
    """
    Build a WeightedRandomSampler that spans survival-time structure.

    Parameters
    ----------
    dataset : ContrastiveDataset or ConcatDataset of ContrastiveDatasets
    outcome_names : list of outcome names
    n_quantiles : number of time quantile bins per outcome before projection
    missing_bucket : retained for backwards compatibility. Missing outcomes are
        always represented by sentinel encoding before projection.

    Returns
    -------
    WeightedRandomSampler with replacement=True, num_samples=len(dataset).
    """
    del missing_bucket
    all_times = _extract_patient_times(dataset, outcome_names)
    n = len(all_times)
    buckets = _projected_buckets(all_times, outcome_names, n_quantiles)

    bucket_counts = Counter(buckets)
    weights = np.array([1.0 / bucket_counts[b] for b in buckets], dtype=np.float64)
    weights = weights / weights.mean()

    return WeightedRandomSampler(
        weights=torch.from_numpy(weights).float(),
        num_samples=n,
        replacement=True,
    )


def log_bucket_stats(
    dataset,
    outcome_names: List[str],
    n_quantiles: int = 4,
) -> str:
    """
    Return a human-readable summary of the projected PC bucket distribution.
    """
    all_times = _extract_patient_times(dataset, outcome_names)
    n = len(all_times)
    buckets = _projected_buckets(all_times, outcome_names, n_quantiles)
    bucket_counts = Counter(buckets)

    lines = [
        f"Stratified sampler - {n} patients, {len(outcome_names)} outcomes",
        "Projected PC bucket distribution (4x4 cells):",
    ]
    for pc1 in range(4):
        cells = [
            f"pc1={pc1},pc2={pc2}: {bucket_counts.get((pc1, pc2), 0)}"
            for pc2 in range(4)
        ]
        lines.append("  " + " | ".join(cells))

    lines.append(f"Outcomes: {outcome_names}")
    for name in outcome_names:
        times = np.array([pt[name] for pt in all_times])
        valid = np.isfinite(times)
        if valid.sum() > 0:
            lines.append(
                f"  {name}: {valid.sum()}/{n} patients have data  "
                f"(median {np.nanmedian(times):.1f}d, "
                f"p25={np.nanquantile(times[valid], 0.25):.1f}d, "
                f"p75={np.nanquantile(times[valid], 0.75):.1f}d)"
            )
        else:
            lines.append(f"  {name}: no data")

    return "\n".join(lines)

