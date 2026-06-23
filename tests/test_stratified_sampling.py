"""Unit tests for opera.functional.stratified_sampling.

The stratified sampler operates on dataset-like objects exposing ``.subjects``
(a list of ``{"subject_id": ...}`` dicts) and ``.outcome_dicts`` (a mapping from
outcome name to ``{subject_id: {"time_days": float}}``). These tests build tiny
synthetic fakes that satisfy that contract so no real data is required.
"""

from __future__ import annotations

import numpy as np
import pytest
from torch.utils.data import ConcatDataset, WeightedRandomSampler

from opera.functional.stratified_sampling import (
    EventAwareSurvivalBatchSampler,
    _extract_patient_times,
    _project_rank_matrix,
    _projected_buckets,
    _quantile_rank_matrix,
    _quartile_bins,
    _standardize_rank_matrix,
    build_event_aware_batch_sampler,
    build_stratified_sampler,
    log_bucket_stats,
)


class _FakeDataset:
    """Minimal stand-in for a ContrastiveDataset.

    ``subjects`` is a list of subject dicts; ``outcome_dicts`` maps each outcome
    name to ``{subject_id: {"time_days": value}}``. ``__len__`` is provided so the
    object behaves like a torch Dataset for ConcatDataset.
    """

    def __init__(self, subjects, outcome_dicts):
        self.subjects = subjects
        self.outcome_dicts = outcome_dicts

    def __len__(self):
        return len(self.subjects)

    def __getitem__(self, index):  # pragma: no cover - not exercised by sampler
        return self.subjects[index]


def _make_dataset(n_subjects=40, outcomes=("os", "pfs"), seed=0, missing_every=0):
    """Build a synthetic dataset with deterministic per-outcome survival times.

    ``missing_every`` > 0 omits every k-th subject from each outcome's dict so
    the missing-outcome code paths are exercised.
    """
    rng = np.random.default_rng(seed)
    subjects = [{"subject_id": f"P{i:03d}"} for i in range(n_subjects)]
    outcome_dicts = {}
    for col, name in enumerate(outcomes):
        records = {}
        times = rng.uniform(10.0, 2000.0, size=n_subjects) + col * 100.0
        for i, subject in enumerate(subjects):
            if missing_every and (i % missing_every == 0):
                continue
            records[subject["subject_id"]] = {"time_days": float(times[i])}
        outcome_dicts[name] = records
    return _FakeDataset(subjects, outcome_dicts)


def _make_event_dataset(n_subjects=96):
    subjects = [{"subject_id": f"P{i:03d}"} for i in range(n_subjects)]
    outcome_dicts = {"common": {}, "rare": {}}
    for i, subject in enumerate(subjects):
        sid = subject["subject_id"]
        outcome_dicts["common"][sid] = {
            "time_days": float(20 + i),
            "event": 1 if i % 4 == 0 else 0,
            "label": 1 if i % 4 == 0 else 0,
        }
        if i < 36:
            is_rare_event = i in {0, 5, 10, 15}
            outcome_dicts["rare"][sid] = {
                "time_days": float(10 + i if is_rare_event else 120 + i),
                "event": 1 if is_rare_event else 0,
                "label": 1 if is_rare_event else 0,
            }
    return _FakeDataset(subjects, outcome_dicts)


def test_extract_patient_times_returns_list_keyed_by_outcome():
    outcomes = ["os", "pfs"]
    ds = _make_dataset(n_subjects=10, outcomes=outcomes, seed=1)

    times = _extract_patient_times(ds, outcomes)

    assert isinstance(times, list)
    assert len(times) == 10
    for patient in times:
        assert set(patient.keys()) == set(outcomes)
        assert all(isinstance(v, float) for v in patient.values())


def test_extract_patient_times_handles_missing_outcomes():
    outcomes = ["os", "pfs"]
    # Every 3rd subject has no record in either outcome dict.
    ds = _make_dataset(n_subjects=12, outcomes=outcomes, seed=2, missing_every=3)

    times = _extract_patient_times(ds, outcomes)

    assert len(times) == 12
    # Subjects at index 0, 3, 6, 9 are missing -> nan for both outcomes.
    for missing_index in (0, 3, 6, 9):
        for name in outcomes:
            assert np.isnan(times[missing_index][name])
    # A present subject (index 1) should have finite values.
    assert np.isfinite(times[1]["os"])


def test_extract_patient_times_traverses_concat_dataset():
    outcomes = ["os"]
    ds_a = _make_dataset(n_subjects=5, outcomes=outcomes, seed=3)
    ds_b = _make_dataset(n_subjects=7, outcomes=outcomes, seed=4)
    concat = ConcatDataset([ds_a, ds_b])

    times = _extract_patient_times(concat, outcomes)

    assert len(times) == 12


def test_quantile_rank_matrix_shape():
    outcomes = ["os", "pfs", "trm"]
    ds = _make_dataset(n_subjects=25, outcomes=outcomes, seed=5)
    all_times = _extract_patient_times(ds, outcomes)

    ranks = _quantile_rank_matrix(all_times, outcomes, n_quantiles=4)

    assert ranks.shape == (25, 3)
    assert ranks.dtype == np.float32


def test_quantile_rank_matrix_fills_sentinel_for_missing():
    outcomes = ["os", "pfs"]
    n_quantiles = 4
    sentinel = float(n_quantiles)
    ds = _make_dataset(n_subjects=20, outcomes=outcomes, seed=6, missing_every=4)
    all_times = _extract_patient_times(ds, outcomes)

    ranks = _quantile_rank_matrix(all_times, outcomes, n_quantiles=n_quantiles)

    # Missing rows (every 4th) carry the sentinel value in every column.
    for missing_index in (0, 4, 8, 12, 16):
        assert np.all(ranks[missing_index] == sentinel)
    # Observed ranks never exceed the sentinel and are non-negative.
    assert ranks.min() >= 0.0
    assert ranks.max() <= sentinel


def test_standardize_rank_matrix_zero_mean_unit_std():
    outcomes = ["os", "pfs"]
    ds = _make_dataset(n_subjects=60, outcomes=outcomes, seed=7)
    all_times = _extract_patient_times(ds, outcomes)
    n_quantiles = 4
    ranks = _quantile_rank_matrix(all_times, outcomes, n_quantiles=n_quantiles)

    standardized = _standardize_rank_matrix(ranks, n_quantiles=n_quantiles)

    sentinel = float(n_quantiles)
    for col in range(standardized.shape[1]):
        valid = ranks[:, col] != sentinel
        values = standardized[valid, col]
        # Only meaningful when there is spread in the column.
        if values.std() > 0:
            assert abs(float(values.mean())) < 1e-5
            assert abs(float(values.std()) - 1.0) < 1e-4


def test_standardize_rank_matrix_preserves_sentinel():
    outcomes = ["os", "pfs"]
    n_quantiles = 4
    sentinel = float(n_quantiles)
    ds = _make_dataset(n_subjects=24, outcomes=outcomes, seed=8, missing_every=3)
    all_times = _extract_patient_times(ds, outcomes)
    ranks = _quantile_rank_matrix(all_times, outcomes, n_quantiles=n_quantiles)

    standardized = _standardize_rank_matrix(ranks, n_quantiles=n_quantiles)

    # Missing entries are encoded as 0.0 (no projection signal), never sentinel.
    for missing_index in (0, 3, 6, 9):
        assert np.all(standardized[missing_index] == 0.0)
    assert not np.any(standardized == sentinel)


def test_project_rank_matrix_reduces_to_2d():
    outcomes = ["os", "pfs", "trm", "relapse"]  # K > 2 forces dimensionality reduction
    ds = _make_dataset(n_subjects=80, outcomes=outcomes, seed=9)
    all_times = _extract_patient_times(ds, outcomes)
    n_quantiles = 4
    ranks = _quantile_rank_matrix(all_times, outcomes, n_quantiles=n_quantiles)

    projection = _project_rank_matrix(ranks, n_quantiles=n_quantiles)

    assert projection.shape == (80, 2)
    assert projection.dtype == np.float32


def test_project_rank_matrix_single_outcome_duplicates_column():
    outcomes = ["os"]
    ds = _make_dataset(n_subjects=15, outcomes=outcomes, seed=10)
    all_times = _extract_patient_times(ds, outcomes)
    ranks = _quantile_rank_matrix(all_times, outcomes, n_quantiles=4)

    projection = _project_rank_matrix(ranks, n_quantiles=4)

    assert projection.shape == (15, 2)
    # K == 1 is repeated across both projected dimensions.
    np.testing.assert_array_equal(projection[:, 0], projection[:, 1])


def test_quartile_bins_returns_values_in_0_3():
    values = np.linspace(0.0, 100.0, 200)

    bins = _quartile_bins(values)

    assert bins.dtype == np.int64
    assert bins.min() >= 0
    assert bins.max() <= 3
    # All four quartile bins should be represented for evenly spread data.
    assert set(np.unique(bins).tolist()) == {0, 1, 2, 3}


def test_quartile_bins_empty_input():
    bins = _quartile_bins(np.array([]))
    assert bins.shape == (0,)


def test_projected_buckets_in_0_15_grid():
    outcomes = ["os", "pfs", "trm"]
    ds = _make_dataset(n_subjects=64, outcomes=outcomes, seed=11)
    all_times = _extract_patient_times(ds, outcomes)

    buckets = _projected_buckets(all_times, outcomes, n_quantiles=4)

    assert len(buckets) == 64
    for pc1, pc2 in buckets:
        # 4x4 grid, 0-indexed -> each coordinate in 0..3, flat index in 0..15.
        assert 0 <= pc1 <= 3
        assert 0 <= pc2 <= 3
        flat = pc1 * 4 + pc2
        assert 0 <= flat <= 15


def test_build_stratified_sampler_returns_weighted_sampler():
    outcomes = ["os", "pfs"]
    ds = _make_dataset(n_subjects=50, outcomes=outcomes, seed=12)

    sampler = build_stratified_sampler(ds, outcomes, n_quantiles=4)

    assert isinstance(sampler, WeightedRandomSampler)
    assert sampler.num_samples == 50
    assert sampler.replacement is True
    weights = np.asarray(sampler.weights)
    assert len(weights) == 50
    assert np.all(weights > 0)
    # Drawing from the sampler yields valid in-range indices.
    drawn = list(sampler)
    assert len(drawn) == 50
    assert min(drawn) >= 0
    assert max(drawn) < 50


def test_build_stratified_sampler_deterministic_weights():
    outcomes = ["os", "pfs"]
    ds = _make_dataset(n_subjects=30, outcomes=outcomes, seed=13)

    first = build_stratified_sampler(ds, outcomes, n_quantiles=4)
    second = build_stratified_sampler(ds, outcomes, n_quantiles=4)

    np.testing.assert_allclose(
        np.asarray(first.weights), np.asarray(second.weights), rtol=0, atol=0
    )


def test_build_stratified_sampler_empty_dataset_raises():
    ds = _FakeDataset(subjects=[], outcome_dicts={"os": {}})

    # An empty dataset cannot produce a valid weighted sampler (num_samples == 0
    # is rejected by torch, or weight normalisation degenerates).
    with pytest.raises(Exception):
        build_stratified_sampler(ds, ["os"], n_quantiles=4)


def test_build_event_aware_batch_sampler_returns_batch_sampler():
    ds = _make_event_dataset()

    sampler = build_event_aware_batch_sampler(
        ds,
        ["common", "rare"],
        batch_size=16,
        min_events_per_batch=2,
        min_valid_per_batch=8,
        seed=7,
    )

    assert isinstance(sampler, EventAwareSurvivalBatchSampler)
    assert len(sampler) == 6
    batches = list(sampler)
    assert len(batches) == 6
    assert all(len(batch) == 16 for batch in batches)
    assert all(min(batch) >= 0 and max(batch) < len(ds) for batch in batches)


def test_event_aware_batch_sampler_enriches_rare_outcome_batches():
    ds = _make_event_dataset()
    sampler = build_event_aware_batch_sampler(
        ds,
        ["common", "rare"],
        batch_size=16,
        min_events_per_batch=2,
        min_valid_per_batch=8,
        seed=3,
    )

    rare_records = ds.outcome_dicts["rare"]
    rare_eventful_batches = 0
    rare_valid_batches = 0
    for batch in sampler:
        rare_events = 0
        rare_valid = 0
        for index in batch:
            sid = ds.subjects[index]["subject_id"]
            rec = rare_records.get(sid)
            if rec is None:
                continue
            rare_valid += 1
            rare_events += int(rec["event"] == 1)
        if rare_events >= 2:
            rare_eventful_batches += 1
        if rare_valid >= 8:
            rare_valid_batches += 1

    # The focus schedule alternates common and rare outcomes across six batches.
    assert rare_eventful_batches >= 3
    assert rare_valid_batches >= 3


def test_event_aware_batch_sampler_avoids_duplicates_when_pool_is_large_enough():
    ds = _make_event_dataset(n_subjects=128)
    sampler = build_event_aware_batch_sampler(
        ds,
        ["common", "rare"],
        batch_size=16,
        min_events_per_batch=2,
        min_valid_per_batch=8,
        seed=11,
    )

    for batch in sampler:
        assert len(batch) == len(set(batch))


def test_event_aware_batch_sampler_falls_back_without_events():
    ds = _make_dataset(n_subjects=32, outcomes=["os"], seed=16)

    sampler = build_event_aware_batch_sampler(
        ds,
        ["os"],
        batch_size=8,
        min_events_per_batch=2,
        min_valid_per_batch=4,
        seed=13,
    )

    batches = list(sampler)
    assert len(batches) == 4
    assert all(len(batch) == 8 for batch in batches)


def test_log_bucket_stats_does_not_crash():
    outcomes = ["os", "pfs"]
    ds = _make_dataset(n_subjects=40, outcomes=outcomes, seed=14, missing_every=5)

    summary = log_bucket_stats(ds, outcomes, n_quantiles=4)

    assert isinstance(summary, str)
    assert "Stratified sampler" in summary
    assert "Projected PC bucket distribution" in summary
    for name in outcomes:
        assert name in summary


def test_log_bucket_stats_handles_outcome_with_no_data():
    outcomes = ["os", "empty"]
    ds = _make_dataset(n_subjects=20, outcomes=["os"], seed=15)
    # Add an outcome key that has no records at all.
    ds.outcome_dicts["empty"] = {}

    summary = log_bucket_stats(ds, outcomes, n_quantiles=4)

    assert "empty: no data" in summary


def test_log_bucket_stats_reports_event_signal_for_batch_size():
    ds = _make_event_dataset()

    summary = log_bucket_stats(ds, ["common", "rare"], batch_size=16)

    assert "primary events" in summary
    assert "expected random batch" in summary
