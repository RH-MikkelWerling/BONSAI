"""Unit tests for opera.functional.stratified_sampling.

The stratified sampler operates on dataset-like objects exposing ``.subjects``
(a list of ``{"subject_id": ...}`` dicts) and ``.outcome_dicts`` (a mapping from
outcome name to ``{subject_id: {"time_days": float}}``). These tests build tiny
synthetic fakes that satisfy that contract so no real data is required.
"""

from __future__ import annotations

import numpy as np
import pytest
from torch.utils.data import ConcatDataset, RandomSampler, WeightedRandomSampler

import opera.functional.stratified_sampling as stratified_sampling
from opera.functional.stratified_sampling import (
    CoverageBalancedSurvivalBatchSampler,
    EventAwareSurvivalBatchSampler,
    _extract_patient_times,
    _project_rank_matrix,
    _projected_buckets,
    _quantile_rank_matrix,
    _quartile_bins,
    _standardize_rank_matrix,
    build_coverage_balanced_survival_batch_sampler,
    build_event_aware_batch_sampler,
    build_stratified_sampler,
    log_bucket_stats,
)
from opera.modules.datamodules.SurvivalFinetuneDataModule import (
    SurvivalFinetuneDataModule,
    resolve_survival_batch_sampler_type,
)
from opera.modules.datamodules.MultiCohortContrastiveDataModule import (
    MultiCohortContrastiveDataModule,
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
    assert sampler.replacement is False
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


def test_coverage_balanced_survival_sampler_preserves_epoch_coverage_and_signal():
    ds = _make_event_dataset()
    sampler = build_coverage_balanced_survival_batch_sampler(
        ds,
        outcome_name="common",
        batch_size=16,
        seed=31,
    )

    assert isinstance(sampler, CoverageBalancedSurvivalBatchSampler)
    batches = list(sampler)
    flattened = [index for batch in batches for index in batch]

    assert len(batches) == 6
    assert sorted(flattened) == list(range(len(ds)))
    assert all(len(batch) == len(set(batch)) for batch in batches)
    assert (
        max(sampler.last_epoch_event_counts) - min(sampler.last_epoch_event_counts) <= 1
    )
    assert min(sampler.last_epoch_event_counts) > 0
    assert min(sampler.last_epoch_comparable_event_counts) > 0
    assert sampler.last_epoch_usage_counts.tolist() == [1] * len(ds)


def test_coverage_balanced_survival_sampler_is_deterministic_per_epoch():
    ds = _make_event_dataset()
    first = build_coverage_balanced_survival_batch_sampler(
        ds, "common", batch_size=16, seed=37
    )
    second = build_coverage_balanced_survival_batch_sampler(
        ds, "common", batch_size=16, seed=37
    )

    first_epoch = list(first)
    assert first_epoch == list(second)
    assert first_epoch != list(first)


def test_coverage_balanced_survival_sampler_rejects_missing_survival_records():
    ds = _make_event_dataset()
    del ds.outcome_dicts["common"]["P003"]

    with pytest.raises(ValueError, match="invalid records"):
        build_coverage_balanced_survival_batch_sampler(
            ds,
            "common",
            batch_size=16,
        )


def test_survival_datamodule_auto_sampling_is_objective_aware():
    source = _make_event_dataset()
    ds = _FakeDataset(source.subjects, {"survival": source.outcome_dicts["common"]})

    cox = object.__new__(SurvivalFinetuneDataModule)
    cox.batch_sampling = {"type": "auto", "seed": 5}
    cox.training_mode = "cox"
    cox.train_dataset = ds
    cox.batch_size = 16
    cox.train_sampler = None
    cox.train_batch_sampler = None
    cox._setup_train_sampling()
    assert isinstance(
        cox.train_batch_sampler,
        CoverageBalancedSurvivalBatchSampler,
    )

    ipcw = object.__new__(SurvivalFinetuneDataModule)
    ipcw.batch_sampling = {"type": "auto", "seed": 5}
    ipcw.training_mode = "ipcw_bce"
    ipcw.train_dataset = ds
    ipcw.batch_size = 16
    ipcw.train_sampler = None
    ipcw.train_batch_sampler = None
    ipcw._setup_train_sampling()
    assert ipcw.train_batch_sampler is None
    assert ipcw.train_sampler is None


def test_survival_sampler_type_resolution_is_objective_aware():
    assert resolve_survival_batch_sampler_type("cox", {"type": "auto"}) == (
        "coverage_balanced"
    )
    assert resolve_survival_batch_sampler_type("ipcw_bce", {"type": "auto"}) == ("none")
    assert (
        resolve_survival_batch_sampler_type("cox", {"type": "risk_set_balanced"})
        == "coverage_balanced"
    )
    assert (
        resolve_survival_batch_sampler_type("cox", {"type": "survival_event_aware"})
        == "event_aware"
    )


@pytest.mark.parametrize(
    "builder",
    [
        lambda ds: build_coverage_balanced_survival_batch_sampler(
            ds, "common", batch_size=16, seed=43
        ),
        lambda ds: build_event_aware_batch_sampler(
            ds, ["common"], batch_size=16, seed=43
        ),
    ],
)
def test_survival_batch_samplers_give_ddp_ranks_equal_step_counts(
    monkeypatch,
    builder,
):
    ds = _make_event_dataset(n_subjects=80)  # five batches before DDP padding
    rank_batches = []
    for rank in (0, 1):
        monkeypatch.setattr(
            stratified_sampling,
            "_distributed_context",
            lambda rank=rank: (rank, 2),
        )
        sampler = builder(ds)
        rank_batches.append(list(sampler))

    assert len(rank_batches[0]) == len(rank_batches[1]) == 3


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


def test_event_aware_batch_sampler_never_clones_sparse_outcome_patients():
    subjects = [{"subject_id": f"P{i:03d}"} for i in range(24)]
    common = {
        subject["subject_id"]: {
            "time_days": float(20 + index),
            "event": int(index % 5 == 0),
        }
        for index, subject in enumerate(subjects)
    }
    rare = {
        "P000": {"time_days": 10.0, "event": 1},
        "P001": {"time_days": 100.0, "event": 0},
    }
    ds = _FakeDataset(subjects, {"common": common, "rare": rare})
    sampler = build_event_aware_batch_sampler(
        ds,
        ["common", "rare"],
        batch_size=12,
        min_events_per_batch=4,
        min_valid_per_batch=8,
        min_unique_events_for_focus=2,
        min_unique_valid_for_focus=4,
        seed=19,
    )

    assert "rare" not in sampler.focus_outcomes
    batches = list(sampler)
    assert all(len(batch) == len(set(batch)) for batch in batches)
    assert sampler.last_epoch_usage_counts.max() <= len(batches)


def test_event_aware_batch_sampler_caps_sparse_quotas_at_unique_pool_size():
    subjects = [{"subject_id": f"P{i:03d}"} for i in range(24)]
    common = {
        subject["subject_id"]: {
            "time_days": float(20 + index),
            "event": int(index % 5 == 0),
        }
        for index, subject in enumerate(subjects)
    }
    rare = {
        "P000": {"time_days": 10.0, "event": 1},
        "P001": {"time_days": 15.0, "event": 1},
        "P002": {"time_days": 100.0, "event": 0},
        "P003": {"time_days": 120.0, "event": 0},
    }
    ds = _FakeDataset(subjects, {"common": common, "rare": rare})
    sampler = build_event_aware_batch_sampler(
        ds,
        ["common", "rare"],
        batch_size=12,
        min_events_per_batch=4,
        min_valid_per_batch=8,
        min_unique_events_for_focus=2,
        min_unique_valid_for_focus=4,
        seed=23,
    )

    assert "rare" in sampler.focus_outcomes
    batches = list(sampler)
    assert all(len(batch) == len(set(batch)) for batch in batches)
    assert any(
        sum(ds.subjects[index]["subject_id"] in rare for index in batch) == 4
        for batch in batches
    )


def test_event_aware_batch_sampler_returns_short_unique_batch_for_tiny_dataset():
    ds = _make_event_dataset(n_subjects=5)
    sampler = build_event_aware_batch_sampler(
        ds,
        ["common", "rare"],
        batch_size=16,
        seed=29,
    )

    batch = next(iter(sampler))
    assert len(batch) == 5
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


def test_event_aware_batch_sampler_default_focus_threshold_admits_single_event_outcome():
    """min_unique_events_for_focus now defaults to 1, so a genuinely
    single-event outcome enters the focus rotation without needing an
    explicit override -- mirrors
    test_event_aware_batch_sampler_never_clones_sparse_outcome_patients'
    fixture but exercises the default rather than pinning the old threshold."""
    subjects = [{"subject_id": f"P{i:03d}"} for i in range(24)]
    common = {
        subject["subject_id"]: {
            "time_days": float(20 + index),
            "event": int(index % 5 == 0),
        }
        for index, subject in enumerate(subjects)
    }
    rare = {
        "P000": {"time_days": 10.0, "event": 1},
        "P001": {"time_days": 100.0, "event": 0},
        "P002": {"time_days": 110.0, "event": 0},
        "P003": {"time_days": 120.0, "event": 0},
    }
    ds = _FakeDataset(subjects, {"common": common, "rare": rare})

    sampler = build_event_aware_batch_sampler(
        ds,
        ["common", "rare"],
        batch_size=12,
        min_events_per_batch=4,
        min_valid_per_batch=8,
        seed=41,
    )

    assert sampler.min_unique_events_for_focus == 1
    assert "rare" in sampler.focus_outcomes
    batches = list(sampler)
    assert all(len(batch) == len(set(batch)) for batch in batches)


def test_event_aware_batch_sampler_never_clones_sparse_outcome_patients_still_holds_with_explicit_threshold():
    """The original threshold=2 test remains valid when the threshold is
    passed explicitly, confirming the new default (1) doesn't silently
    change behavior for callers who still pin the old value."""
    subjects = [{"subject_id": f"P{i:03d}"} for i in range(24)]
    common = {
        subject["subject_id"]: {
            "time_days": float(20 + index),
            "event": int(index % 5 == 0),
        }
        for index, subject in enumerate(subjects)
    }
    rare = {
        "P000": {"time_days": 10.0, "event": 1},
        "P001": {"time_days": 100.0, "event": 0},
    }
    ds = _FakeDataset(subjects, {"common": common, "rare": rare})
    sampler = build_event_aware_batch_sampler(
        ds,
        ["common", "rare"],
        batch_size=12,
        min_events_per_batch=4,
        min_valid_per_batch=8,
        min_unique_events_for_focus=2,
        min_unique_valid_for_focus=4,
        seed=19,
    )

    assert "rare" not in sampler.focus_outcomes
    batches = list(sampler)
    assert all(len(batch) == len(set(batch)) for batch in batches)
    assert sampler.last_epoch_usage_counts.max() <= len(batches)


def test_event_aware_batch_sampler_cohort_labels_rejects_wrong_length():
    ds = _make_event_dataset(n_subjects=24)

    with pytest.raises(ValueError, match="cohort_labels"):
        build_event_aware_batch_sampler(
            ds,
            ["common", "rare"],
            batch_size=12,
            cohort_labels=["big"] * 10,  # wrong length: dataset has 24 patients
        )


def test_event_aware_batch_sampler_cohort_summary_flags_undercovered_cohort():
    rng = np.random.default_rng(51)
    n_big, n_small = 200, 12
    subjects = [{"subject_id": f"P{i:03d}"} for i in range(n_big + n_small)]
    common = {
        subject["subject_id"]: {
            "time_days": float(rng.uniform(10, 2000)),
            "event": int(rng.random() < 0.3),
        }
        for subject in subjects
    }
    ds = _FakeDataset(subjects, {"common": common})
    cohort_labels = ["big"] * n_big + ["small"] * n_small

    sampler = build_event_aware_batch_sampler(
        ds,
        ["common"],
        batch_size=16,
        cohort_labels=cohort_labels,
        seed=5,
    )

    summary = sampler.summary()
    assert "cohort epoch coverage" in summary
    assert "big: patients=200" in summary
    assert "small: patients=12" in summary
    # The small cohort is undercovered relative to a full epoch given its
    # tiny population share -- this is the diagnostic the plan calls for,
    # not a sampling behavior change.
    small_line = next(line for line in summary.splitlines() if "small:" in line)
    assert "epoch_coverage" in small_line


def test_event_aware_batch_sampler_cohort_labels_do_not_change_sampling_behavior():
    """cohort_labels is diagnostic-only: passing it must not change which
    batches get drawn for a fixed seed."""
    ds = _make_event_dataset(n_subjects=48)
    cohort_labels = ["a"] * 24 + ["b"] * 24

    without_labels = build_event_aware_batch_sampler(
        ds, ["common", "rare"], batch_size=16, seed=8
    )
    with_labels = build_event_aware_batch_sampler(
        ds, ["common", "rare"], batch_size=16, seed=8, cohort_labels=cohort_labels
    )

    assert list(without_labels) == list(with_labels)


def test_multicohort_datamodule_threads_cohort_labels_to_sampler():
    """_setup_train_sampling wires self.train_cohort_labels through to the
    sampler in the same order ConcatDataset concatenates sub-datasets --
    built via object.__new__ + manual attrs, matching the existing
    test_survival_datamodule_auto_sampling_is_objective_aware pattern, since
    setup() itself does real file I/O this test deliberately avoids."""
    big = _make_event_dataset(n_subjects=40)
    small = _make_event_dataset(n_subjects=6)
    concatenated = ConcatDataset([big, small])

    module = object.__new__(MultiCohortContrastiveDataModule)
    module.train_dataset = concatenated
    module.train_cohort_labels = np.repeat(
        ["big_cohort", "small_cohort"], [len(big), len(small)]
    )
    module.outcome_names = ["common", "rare"]
    module.batch_size = 8
    module.batch_sampling = {"type": "event_aware", "seed": 3}
    module.train_sampler = None
    module.train_batch_sampler = None

    module._setup_train_sampling()

    sampler = module.train_batch_sampler
    assert isinstance(sampler, EventAwareSurvivalBatchSampler)
    assert sampler.cohort_indices is not None
    assert sorted(sampler.cohort_indices) == ["big_cohort", "small_cohort"]
    assert sampler.cohort_indices["big_cohort"].size == len(big)
    assert sampler.cohort_indices["small_cohort"].size == len(small)
    # Ordering contract: small_cohort's global indices must be the tail
    # range, matching ConcatDataset's concatenation order.
    assert sampler.cohort_indices["small_cohort"].min() == len(big)


def test_multicohort_random_training_loader_really_shuffles():
    module = object.__new__(MultiCohortContrastiveDataModule)
    module.train_dataset = _make_event_dataset(n_subjects=12)
    module.train_batch_sampler = None
    module.train_sampler = None
    module.batch_size = 4
    module.num_workers = 0

    loader = module.train_dataloader()

    assert isinstance(loader.sampler, RandomSampler)
