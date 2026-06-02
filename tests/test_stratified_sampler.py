"""Tests for the PCA-based stratified sampler."""

import numpy as np
from unittest.mock import MagicMock
from torch.utils.data import WeightedRandomSampler


def _make_mock_dataset(n_patients, outcome_names, seed=0):
    """Build a mock ContrastiveDataset-like object with fake outcome_dicts."""
    rng = np.random.default_rng(seed)
    ds = MagicMock()
    ds.datasets = None  # not a ConcatDataset

    subjects = [{"subject_id": i} for i in range(n_patients)]
    ds.subjects = subjects

    outcome_dicts = {}
    for name in outcome_names:
        outcome_dicts[name] = {
            i: {"time_days": float(rng.uniform(0, 1000))}
            for i in range(n_patients)
            if rng.random() > 0.1
        }
    ds.outcome_dicts = outcome_dicts
    return ds


def test_sampler_returns_weighted_random_sampler():
    from opera.functional.stratified_sampling import build_stratified_sampler

    ds = _make_mock_dataset(100, ["mortality", "aki"])
    sampler = build_stratified_sampler(ds, ["mortality", "aki"])
    assert isinstance(sampler, WeightedRandomSampler)
    assert sampler.num_samples == 100


def test_sampler_always_produces_at_most_16_buckets_regardless_of_outcome_count():
    """Core scaling test: bucket count must not grow with K."""
    from opera.functional.stratified_sampling import build_stratified_sampler

    for n_outcomes in [2, 10, 25, 50, 100]:
        outcome_names = [f"outcome_{k}" for k in range(n_outcomes)]
        ds = _make_mock_dataset(500, outcome_names, seed=n_outcomes)
        sampler = build_stratified_sampler(ds, outcome_names, n_quantiles=4)
        unique_weights = len(set(sampler.weights.tolist()))
        assert unique_weights <= 16, (
            f"Expected <=16 unique weights at K={n_outcomes}, got {unique_weights}"
        )


def test_sampler_handles_all_missing_outcome():
    """A completely missing outcome should not crash the sampler."""
    from opera.functional.stratified_sampling import build_stratified_sampler

    ds = _make_mock_dataset(50, ["present", "always_missing"])
    ds.outcome_dicts["always_missing"] = {}
    sampler = build_stratified_sampler(ds, ["present", "always_missing"])
    assert isinstance(sampler, WeightedRandomSampler)


def test_sampler_handles_single_outcome():
    from opera.functional.stratified_sampling import build_stratified_sampler

    ds = _make_mock_dataset(80, ["only_outcome"])
    sampler = build_stratified_sampler(ds, ["only_outcome"])
    assert isinstance(sampler, WeightedRandomSampler)


def test_sampler_weights_are_positive_and_finite():
    import torch
    from opera.functional.stratified_sampling import build_stratified_sampler

    outcome_names = [f"o{k}" for k in range(20)]
    ds = _make_mock_dataset(200, outcome_names)
    sampler = build_stratified_sampler(ds, outcome_names)
    w = sampler.weights
    assert (w > 0).all(), "All weights must be positive"
    assert torch.isfinite(w).all(), "All weights must be finite"


def test_log_bucket_stats_returns_string():
    from opera.functional.stratified_sampling import log_bucket_stats

    ds = _make_mock_dataset(100, ["mortality", "aki", "infection"])
    result = log_bucket_stats(ds, ["mortality", "aki", "infection"])
    assert isinstance(result, str)
    assert len(result) > 0

