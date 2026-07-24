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
from typing import Dict, Iterator, List, Optional, Sequence

import numpy as np
import torch
from torch.utils.data import ConcatDataset, Sampler, WeightedRandomSampler


def _as_float(value, default: float = float("nan")) -> float:
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _as_event(value, default: int = -1) -> int:
    if value is None:
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _sub_datasets(dataset) -> list:
    return list(dataset.datasets) if isinstance(dataset, ConcatDataset) else [dataset]


def _extract_patient_records(
    dataset,
    outcome_names: Sequence[str],
) -> List[Dict[str, Optional[dict]]]:
    """
    Return per-index outcome records for ContrastiveDataset-like objects.

    The helper also accepts SurvivalFinetuneDataset-like objects exposing a
    single ``.outcomes`` mapping; in that case the first outcome name is used.
    """
    records: List[Dict[str, Optional[dict]]] = []
    for ds in _sub_datasets(dataset):
        for subject in ds.subjects:
            sid = subject["subject_id"]
            patient_records: Dict[str, Optional[dict]] = {}
            if hasattr(ds, "outcome_dicts"):
                for name in outcome_names:
                    patient_records[name] = ds.outcome_dicts.get(name, {}).get(sid)
            elif hasattr(ds, "outcomes"):
                if len(outcome_names) != 1:
                    raise ValueError(
                        "Single-outcome datasets must be sampled with exactly "
                        "one outcome name."
                    )
                patient_records[outcome_names[0]] = ds.outcomes.get(sid)
            else:
                raise TypeError(
                    "Event-aware sampling requires a dataset exposing either "
                    "outcome_dicts or outcomes."
                )
            records.append(patient_records)
    return records


def _extract_patient_times(
    dataset,
    outcome_names: List[str],
) -> List[Dict[str, float]]:
    """
    Walk a dataset (or ConcatDataset of ContrastiveDatasets) and return,
    for each patient in order, a dict {outcome_name: time_days or nan}.
    """
    all_times: List[Dict[str, float]] = []
    for patient_records in _extract_patient_records(dataset, outcome_names):
        patient_times: Dict[str, float] = {}
        for name in outcome_names:
            rec = patient_records.get(name)
            patient_times[name] = (
                _as_float(rec.get("time_days")) if rec is not None else float("nan")
            )
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
        ranks[valid, col] = np.searchsorted(boundaries, times[valid], side="right")

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

        return (
            TruncatedSVD(n_components=2, random_state=0)
            .fit_transform(standardized)
            .astype(np.float32)
        )
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
    WeightedRandomSampler without replacement, covering every patient once.
    """
    del missing_bucket
    all_times = _extract_patient_times(dataset, outcome_names)
    n = len(all_times)
    if n == 0:
        raise ValueError("Cannot build a stratified sampler for an empty dataset.")
    buckets = _projected_buckets(all_times, outcome_names, n_quantiles)

    bucket_counts = Counter(buckets)
    weights = np.array([1.0 / bucket_counts[b] for b in buckets], dtype=np.float64)
    weights = weights / weights.mean()

    return WeightedRandomSampler(
        weights=torch.from_numpy(weights).float(),
        num_samples=n,
        replacement=False,
    )


def _normalise_probabilities(weights: np.ndarray) -> np.ndarray:
    weights = np.asarray(weights, dtype=np.float64).copy()
    weights[~np.isfinite(weights)] = 0.0
    weights = np.clip(weights, 0.0, None)
    total = float(weights.sum())
    if total <= 0.0:
        return np.full(weights.shape, 1.0 / len(weights), dtype=np.float64)
    return weights / total


def _distributed_context() -> tuple[int, int]:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_rank(), torch.distributed.get_world_size()
    return 0, 1


def _draw_from_pool(
    rng: np.random.Generator,
    pool: np.ndarray,
    n: int,
    selected: set[int],
    probabilities: Optional[np.ndarray] = None,
) -> list[int]:
    """Draw unique indices that are not already present in the batch."""
    if n <= 0 or pool.size == 0:
        return []

    unique_pool = np.unique(pool.astype(np.int64, copy=False))
    available = np.array(
        [idx for idx in unique_pool.tolist() if idx not in selected],
        dtype=np.int64,
    )

    if available.size == 0:
        return []
    n_unique = min(n, available.size)
    p = None
    if probabilities is not None:
        p = _normalise_probabilities(probabilities[available])
    return rng.choice(available, size=n_unique, replace=False, p=p).tolist()


class CoverageBalancedSurvivalBatchSampler(Sampler[list[int]]):
    """Spread survival signal across batches without changing epoch exposure.

    Every eligible patient appears exactly once per single-process epoch. Primary
    events are assigned as evenly as possible across batches, then later
    non-event comparators are paired where capacity permits. Remaining patients
    are distributed across follow-up-time bins. Unlike outcome-weighted random
    sampling, this does not duplicate rare events or alter patient-level epoch
    weights.

    This improves the usefulness of mini-batch Cox updates but does not make
    their risk sets exact: the full Cox partial likelihood still requires the
    complete training risk set (or a formally corrected sampled-risk objective).
    """

    def __init__(
        self,
        dataset,
        outcome_name: str,
        batch_size: int,
        seed: int = 0,
    ):
        if batch_size <= 0:
            raise ValueError("batch_size must be positive.")
        self.dataset = dataset
        self.outcome_name = str(outcome_name)
        self.batch_size = int(batch_size)
        self.seed = int(seed)
        self.epoch = 0

        records = _extract_patient_records(dataset, [self.outcome_name])
        self.n = len(records)
        if self.n == 0:
            raise ValueError("Cannot build a batch sampler for an empty dataset.")

        self.times = np.full(self.n, np.nan, dtype=np.float64)
        self.events = np.full(self.n, -1, dtype=np.int64)
        for index, patient_records in enumerate(records):
            record = patient_records.get(self.outcome_name)
            if record is None:
                continue
            self.times[index] = _as_float(record.get("time_days"))
            self.events[index] = _as_event(record.get("event", record.get("label", -1)))

        self.valid_mask = np.isfinite(self.times) & (self.events >= 0)
        if not self.valid_mask.all():
            invalid = int((~self.valid_mask).sum())
            raise ValueError(
                "Coverage-balanced survival sampling requires finite times and "
                f"event indicators for every patient; found {invalid} invalid records."
            )
        self.event_indices = np.flatnonzero(self.events == 1).astype(np.int64)
        self.non_event_indices = np.flatnonzero(self.events != 1).astype(np.int64)
        self.num_batches = max(1, (self.n + self.batch_size - 1) // self.batch_size)
        self.last_epoch_usage_counts = np.zeros(self.n, dtype=np.int64)
        self.last_epoch_event_counts: list[int] = []
        self.last_epoch_comparable_event_counts: list[int] = []

    def __len__(self) -> int:
        _, world_size = _distributed_context()
        return (self.num_batches + world_size - 1) // world_size

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def _capacities(self) -> list[int]:
        base, remainder = divmod(self.n, self.num_batches)
        return [base + int(index < remainder) for index in range(self.num_batches)]

    @staticmethod
    def _least_loaded_batch(
        batches: Sequence[list[int]],
        capacities: Sequence[int],
        rng: np.random.Generator,
        *,
        event_counts: Optional[np.ndarray] = None,
    ) -> int:
        candidates = [
            index
            for index, batch in enumerate(batches)
            if len(batch) < capacities[index]
        ]
        if not candidates:
            raise RuntimeError("No batch capacity remains while assigning patients.")
        if event_counts is not None:
            minimum_events = min(event_counts[index] for index in candidates)
            candidates = [
                index for index in candidates if event_counts[index] == minimum_events
            ]
        minimum_size = min(len(batches[index]) for index in candidates)
        candidates = [
            index for index in candidates if len(batches[index]) == minimum_size
        ]
        return int(rng.choice(candidates))

    def _build_epoch_batches(self, rng: np.random.Generator) -> list[list[int]]:
        capacities = self._capacities()
        batches: list[list[int]] = [[] for _ in capacities]
        event_counts = np.zeros(self.num_batches, dtype=np.int64)
        assigned = np.zeros(self.n, dtype=bool)

        shuffled_events = rng.permutation(self.event_indices)
        for patient_index in shuffled_events:
            batch_index = self._least_loaded_batch(
                batches,
                capacities,
                rng,
                event_counts=event_counts,
            )
            batches[batch_index].append(int(patient_index))
            event_counts[batch_index] += 1
            assigned[int(patient_index)] = True

        # Reserve long-follow-up non-events for the latest events first. This
        # increases the chance that every event has a genuine at-risk comparator
        # without duplicating either cases or controls.
        available_controls = set(self.non_event_indices.tolist())
        events_latest_first = sorted(
            self.event_indices.tolist(),
            key=lambda index: self.times[index],
            reverse=True,
        )
        event_to_batch = {
            patient_index: batch_index
            for batch_index, batch in enumerate(batches)
            for patient_index in batch
            if self.events[patient_index] == 1
        }
        for event_index in events_latest_first:
            batch_index = event_to_batch[event_index]
            if len(batches[batch_index]) >= capacities[batch_index]:
                continue
            candidates = np.array(
                [
                    index
                    for index in available_controls
                    if self.times[index] >= self.times[event_index]
                ],
                dtype=np.int64,
            )
            if candidates.size == 0:
                continue
            comparator = int(rng.choice(candidates))
            batches[batch_index].append(comparator)
            assigned[comparator] = True
            available_controls.remove(comparator)

        remaining = np.flatnonzero(~assigned).astype(np.int64)
        if remaining.size:
            time_bins = _quartile_bins(self.times[remaining])
            queues = []
            for bin_id in np.unique(time_bins):
                queue = remaining[time_bins == bin_id].copy()
                rng.shuffle(queue)
                queues.append(queue.tolist())
            rng.shuffle(queues)

            queue_index = 0
            while any(queues):
                queue = queues[queue_index % len(queues)]
                queue_index += 1
                if not queue:
                    continue
                patient_index = int(queue.pop())
                batch_index = self._least_loaded_batch(batches, capacities, rng)
                batches[batch_index].append(patient_index)
                assigned[patient_index] = True

        if not assigned.all():
            raise RuntimeError(
                "Coverage-balanced sampler failed to assign all patients."
            )
        for batch, capacity in zip(batches, capacities):
            if len(batch) != capacity or len(batch) != len(set(batch)):
                raise RuntimeError(
                    "Coverage-balanced sampler produced an invalid batch."
                )
            rng.shuffle(batch)
        return batches

    def _comparable_event_count(self, batch: Sequence[int]) -> int:
        indices = np.asarray(batch, dtype=np.int64)
        times = self.times[indices]
        events = self.events[indices]
        count = 0
        for local_index in np.flatnonzero(events == 1):
            at_risk = times >= times[local_index]
            if int(at_risk.sum()) >= 2:
                count += 1
        return count

    def __iter__(self) -> Iterator[list[int]]:
        rng = np.random.default_rng(self.seed + self.epoch)
        self.epoch += 1
        batches = self._build_epoch_batches(rng)
        rank, world_size = _distributed_context()

        # All ranks must execute the same number of optimizer steps. Pad the
        # global batch list only for distributed execution; single-process
        # training retains exact once-per-epoch coverage.
        target_global_batches = self.__len__() * world_size
        if target_global_batches > len(batches):
            padding = target_global_batches - len(batches)
            for index in range(padding):
                batches.append(list(batches[index % len(batches)]))

        usage_counts = np.zeros(self.n, dtype=np.int64)
        event_counts: list[int] = []
        comparable_counts: list[int] = []
        for batch in batches:
            usage_counts[np.asarray(batch, dtype=np.int64)] += 1
            event_counts.append(int((self.events[np.asarray(batch)] == 1).sum()))
            comparable_counts.append(self._comparable_event_count(batch))
        self.last_epoch_usage_counts = usage_counts
        self.last_epoch_event_counts = event_counts
        self.last_epoch_comparable_event_counts = comparable_counts

        yield from batches[rank::world_size]

    def summary(self) -> str:
        expected_random_empty = (
            (1.0 - self.event_indices.size / self.n) ** self.batch_size
            if self.n
            else float("nan")
        )
        return "\n".join(
            [
                "Coverage-balanced survival batch sampler",
                f"  patients={self.n}, events={self.event_indices.size}, "
                f"batch_size={self.batch_size}, batches_per_epoch={self.num_batches}",
                "  each patient appears exactly once per single-process epoch",
                "  primary events are spread across batches; later non-event "
                "comparators are paired when available",
                f"  random-shuffle probability of an event-free full batch≈"
                f"{expected_random_empty:.1%}",
                "  note: mini-batch Cox risk sets remain an approximation of the "
                "full partial likelihood",
            ]
        )


def build_coverage_balanced_survival_batch_sampler(
    dataset,
    outcome_name: str,
    batch_size: int,
    seed: int = 0,
) -> CoverageBalancedSurvivalBatchSampler:
    return CoverageBalancedSurvivalBatchSampler(
        dataset=dataset,
        outcome_name=outcome_name,
        batch_size=batch_size,
        seed=seed,
    )


class EventAwareSurvivalBatchSampler(Sampler[list[int]]):
    """
    Compose batches with outcome-specific survival signal.

    Each batch focuses on one outcome in a rotating schedule. For that outcome,
    the sampler tries to include a minimum number of primary events, later
    at-risk comparators for those events, and a minimum number of eligible
    patients before filling the rest of the batch from the survival-time bucket
    distribution used by the legacy weighted sampler.
    """

    def __init__(
        self,
        dataset,
        outcome_names: Sequence[str],
        batch_size: int,
        n_quantiles: int = 4,
        min_events_per_batch: int = 4,
        min_valid_per_batch: Optional[int] = None,
        min_unique_events_for_focus: int = 1,
        min_unique_valid_for_focus: int = 4,
        batches_per_epoch: Optional[int] = None,
        seed: int = 0,
        cohort_labels: Optional[Sequence[str]] = None,
    ):
        if batch_size <= 0:
            raise ValueError("batch_size must be positive.")
        self.dataset = dataset
        self.outcome_names = list(outcome_names)
        self.batch_size = int(batch_size)
        self.n_quantiles = int(n_quantiles)
        self.min_events_per_batch = max(0, int(min_events_per_batch))
        if min_valid_per_batch is None:
            min_valid_per_batch = max(2 * self.min_events_per_batch, batch_size // 4)
            min_valid_per_batch = max(2, min_valid_per_batch)
        self.min_valid_per_batch = min(int(min_valid_per_batch), self.batch_size)
        self.min_unique_events_for_focus = max(1, int(min_unique_events_for_focus))
        self.min_unique_valid_for_focus = max(2, int(min_unique_valid_for_focus))
        self.seed = int(seed)
        self.epoch = 0

        self.records = _extract_patient_records(dataset, self.outcome_names)
        self.n = len(self.records)
        if self.n == 0:
            raise ValueError("Cannot build a batch sampler for an empty dataset.")

        # Diagnostics only: cohort_labels never affects sampling probability
        # or draw order, only the per-cohort coverage estimate in summary().
        # Deliberately not an oversampling lever — see OPERA_EXPERIMENTS.md.
        self.cohort_indices: Optional[dict[str, np.ndarray]] = None
        if cohort_labels is not None:
            labels_array = np.asarray(list(cohort_labels))
            if labels_array.shape[0] != self.n:
                raise ValueError(
                    "cohort_labels must have one entry per patient; got "
                    f"{labels_array.shape[0]} labels for {self.n} patients."
                )
            self.cohort_indices = {
                str(label): np.flatnonzero(labels_array == label).astype(np.int64)
                for label in np.unique(labels_array)
            }

        all_times = _extract_patient_times(dataset, self.outcome_names)
        buckets = _projected_buckets(all_times, self.outcome_names, self.n_quantiles)
        bucket_counts = Counter(buckets)
        base_weights = np.array(
            [1.0 / bucket_counts[bucket] for bucket in buckets],
            dtype=np.float64,
        )
        self.base_probabilities = _normalise_probabilities(base_weights)
        self._draw_probabilities = self.base_probabilities.copy()
        self.last_epoch_usage_counts = np.zeros(self.n, dtype=np.int64)

        self.times_by_outcome: dict[str, np.ndarray] = {}
        self.events_by_outcome: dict[str, np.ndarray] = {}
        self.valid_indices: dict[str, np.ndarray] = {}
        self.event_indices: dict[str, np.ndarray] = {}
        for name in self.outcome_names:
            times = np.full(self.n, np.nan, dtype=np.float64)
            events = np.full(self.n, -1, dtype=np.int64)
            for index, patient_records in enumerate(self.records):
                rec = patient_records.get(name)
                if rec is None:
                    continue
                times[index] = _as_float(rec.get("time_days"))
                events[index] = _as_event(rec.get("event", rec.get("label", -1)))

            valid = np.isfinite(times) & (events >= 0)
            event = valid & (events == 1)
            self.times_by_outcome[name] = times
            self.events_by_outcome[name] = events
            self.valid_indices[name] = np.where(valid)[0].astype(np.int64)
            self.event_indices[name] = np.where(event)[0].astype(np.int64)

        self.focus_outcomes = sorted(
            [
                name
                for name in self.outcome_names
                if self.valid_indices[name].size >= self.min_unique_valid_for_focus
                and self.event_indices[name].size >= self.min_unique_events_for_focus
            ],
            key=lambda name: (
                self.event_indices[name].size,
                self.valid_indices[name].size,
                name,
            ),
        )
        self.num_batches = (
            int(batches_per_epoch)
            if batches_per_epoch is not None
            else max(1, (self.n + self.batch_size - 1) // self.batch_size)
        )
        if self.num_batches <= 0:
            raise ValueError("batches_per_epoch must be positive.")

    def __len__(self) -> int:
        _, world_size = _distributed_context()
        return (self.num_batches + world_size - 1) // world_size

    def _add(
        self, batch: list[int], selected: set[int], indices: Sequence[int]
    ) -> None:
        for index in indices:
            if len(batch) >= self.batch_size:
                return
            batch.append(int(index))
            selected.add(int(index))

    def _draw_events(
        self,
        name: str,
        rng: np.random.Generator,
        n: int,
        selected: set[int],
    ) -> list[int]:
        pool = self.event_indices[name]
        if n <= 0 or pool.size == 0:
            return []
        if n == 1 or pool.size <= 1:
            return _draw_from_pool(rng, pool, n, selected, self._draw_probabilities)

        times = self.times_by_outcome[name][pool]
        finite = np.isfinite(times)
        if finite.sum() < 2:
            return _draw_from_pool(rng, pool, n, selected, self._draw_probabilities)

        bins = _quartile_bins(times)
        drawn: list[int] = []
        bin_ids = rng.permutation(np.unique(bins))
        while len(drawn) < n and len(bin_ids) > 0:
            made_progress = False
            for bin_id in bin_ids:
                candidates = pool[bins == bin_id]
                chosen = _draw_from_pool(
                    rng,
                    candidates,
                    1,
                    selected | set(drawn),
                    self._draw_probabilities,
                )
                if chosen:
                    drawn.extend(chosen)
                    made_progress = True
                    if len(drawn) >= n:
                        break
            if not made_progress:
                break
        if len(drawn) < n:
            drawn.extend(
                _draw_from_pool(
                    rng,
                    pool,
                    n - len(drawn),
                    selected | set(drawn),
                    self._draw_probabilities,
                )
            )
        return drawn[:n]

    def _draw_later_comparators(
        self,
        name: str,
        event_indices: Sequence[int],
        rng: np.random.Generator,
        selected: set[int],
    ) -> list[int]:
        valid_pool = self.valid_indices[name]
        if valid_pool.size == 0:
            return []
        times = self.times_by_outcome[name]
        drawn: list[int] = []
        for event_index in event_indices:
            if len(drawn) >= self.batch_size:
                break
            event_time = times[int(event_index)]
            later_pool = valid_pool[times[valid_pool] > event_time]
            if later_pool.size == 0:
                later_pool = valid_pool[times[valid_pool] >= event_time]
            chosen = _draw_from_pool(
                rng,
                later_pool,
                1,
                selected | set(drawn),
                self._draw_probabilities,
            )
            drawn.extend(chosen)
        return drawn

    def _fill_batch(
        self,
        batch: list[int],
        selected: set[int],
        rng: np.random.Generator,
    ) -> None:
        # Fill from the union of valid patients so batch slots aren't wasted
        # on patients who have all-missing outcome labels (time=-1, event=-1)
        # and would contribute zero pairs to every outcome in the loss.
        valid_arrays = [v for v in self.valid_indices.values() if v.size > 0]
        fill_pool = (
            np.unique(np.concatenate(valid_arrays))
            if valid_arrays
            else np.arange(self.n, dtype=np.int64)
        )
        needed = self.batch_size - len(batch)
        self._add(
            batch,
            selected,
            _draw_from_pool(
                rng,
                fill_pool,
                needed,
                selected,
                self._draw_probabilities,
            ),
        )

    def _build_batch_for_outcome(
        self,
        name: str,
        rng: np.random.Generator,
    ) -> list[int]:
        batch: list[int] = []
        selected: set[int] = set()

        event_quota = min(self.min_events_per_batch, self.batch_size)
        events = self._draw_events(name, rng, event_quota, selected)
        self._add(batch, selected, events)

        comparators = self._draw_later_comparators(name, events, rng, selected)
        self._add(batch, selected, comparators)

        valid_set = set(self.valid_indices[name].tolist())
        valid_now = sum(index in valid_set for index in batch)
        valid_needed = max(0, self.min_valid_per_batch - valid_now)
        self._add(
            batch,
            selected,
            _draw_from_pool(
                rng,
                self.valid_indices[name],
                valid_needed,
                selected,
                self._draw_probabilities,
            ),
        )

        self._fill_batch(batch, selected, rng)
        return batch[: self.batch_size]

    def set_epoch(self, epoch: int) -> None:
        """Set the epoch counter externally (called by Lightning on resume)."""
        self.epoch = epoch

    def __iter__(self) -> Iterator[list[int]]:
        rng = np.random.default_rng(self.seed + self.epoch)
        self.epoch += 1
        usage_counts = np.zeros(self.n, dtype=np.int64)
        rank, world_size = _distributed_context()
        if self.focus_outcomes:
            focus_order = list(self.focus_outcomes)
            rng.shuffle(focus_order)
        else:
            focus_order = []

        # Pad the global schedule to a multiple of world size so every DDP rank
        # executes the same number of optimizer steps. Unequal iterator lengths
        # can otherwise deadlock gradient synchronization.
        global_num_batches = self.__len__() * world_size
        for batch_index in range(global_num_batches):
            should_yield = batch_index % world_size == rank
            self._draw_probabilities = _normalise_probabilities(
                self.base_probabilities / (1.0 + usage_counts)
            )
            if focus_order:
                focus = focus_order[batch_index % len(focus_order)]
                batch = self._build_batch_for_outcome(focus, rng)
                usage_counts[np.asarray(batch, dtype=np.int64)] += 1
                if should_yield:
                    yield batch
            else:
                batch: list[int] = []
                selected: set[int] = set()
                self._fill_batch(batch, selected, rng)
                usage_counts[np.asarray(batch, dtype=np.int64)] += 1
                if should_yield:
                    yield batch[: self.batch_size]
        self.last_epoch_usage_counts = usage_counts

    def summary(self) -> str:
        lines = [
            "Event-aware survival batch sampler",
            f"  patients={self.n}, batch_size={self.batch_size}, "
            f"batches_per_epoch={self.num_batches}",
            f"  min_events_per_batch={self.min_events_per_batch}, "
            f"min_valid_per_batch={self.min_valid_per_batch}",
            f"  focus threshold: {self.min_unique_events_for_focus} unique events, "
            f"{self.min_unique_valid_for_focus} unique valid patients",
        ]
        if not self.focus_outcomes:
            lines.append(
                "  no outcomes meet the unique-evidence threshold; "
                "falling back to diversity-balanced fill"
            )
            for name in self.outcome_names:
                lines.append(
                    f"  {name}: valid={self.valid_indices[name].size}, "
                    f"primary_events={self.event_indices[name].size} [not focused]"
                )
            self._append_cohort_summary(lines)
            return "\n".join(lines)
        lines.append(f"  focus outcomes: {self.focus_outcomes}")
        batches_per_focus = max(1, self.num_batches // len(self.focus_outcomes))
        for name in self.outcome_names:
            n_events = self.event_indices[name].size
            unique_event_draws = batches_per_focus * min(
                self.min_events_per_batch,
                n_events,
            )
            coverage = (
                unique_event_draws / max(1, n_events)
                if name in self.focus_outcomes
                else 0.0
            )
            coverage_str = f", epoch_coverage≈{coverage:.2f}"
            if name in self.focus_outcomes and coverage < 1.0:
                coverage_str += " [WARNING: <1× per epoch, consider batches_per_epoch]"
            lines.append(
                f"  {name}: valid={self.valid_indices[name].size}, "
                f"primary_events={n_events}{coverage_str}"
            )

        self._append_cohort_summary(lines)
        return "\n".join(lines)

    def _append_cohort_summary(self, lines: list[str]) -> None:
        if not self.cohort_indices:
            return
        lines.append(
            "  cohort epoch coverage (diagnostic only; does not affect sampling):"
        )
        total_draws = self.num_batches * self.batch_size
        for label in sorted(self.cohort_indices):
            indices = self.cohort_indices[label]
            weight_share = float(self.base_probabilities[indices].sum())
            expected_draws_per_patient = (
                total_draws * weight_share / indices.size if indices.size else 0.0
            )
            coverage_str = f"epoch_coverage≈{expected_draws_per_patient:.2f}"
            if expected_draws_per_patient < 1.0:
                coverage_str += " [WARNING: <1× per epoch on average]"
            lines.append(f"    {label}: patients={indices.size}, {coverage_str}")


def build_event_aware_batch_sampler(
    dataset,
    outcome_names: Sequence[str],
    batch_size: int,
    n_quantiles: int = 4,
    min_events_per_batch: int = 4,
    min_valid_per_batch: Optional[int] = None,
    min_unique_events_for_focus: int = 1,
    min_unique_valid_for_focus: int = 4,
    batches_per_epoch: Optional[int] = None,
    seed: int = 0,
    cohort_labels: Optional[Sequence[str]] = None,
) -> EventAwareSurvivalBatchSampler:
    return EventAwareSurvivalBatchSampler(
        dataset=dataset,
        outcome_names=outcome_names,
        batch_size=batch_size,
        n_quantiles=n_quantiles,
        min_events_per_batch=min_events_per_batch,
        min_valid_per_batch=min_valid_per_batch,
        min_unique_events_for_focus=min_unique_events_for_focus,
        min_unique_valid_for_focus=min_unique_valid_for_focus,
        batches_per_epoch=batches_per_epoch,
        seed=seed,
        cohort_labels=cohort_labels,
    )


def log_bucket_stats(
    dataset,
    outcome_names: List[str],
    n_quantiles: int = 4,
    batch_size: Optional[int] = None,
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

    records = _extract_patient_records(dataset, outcome_names)

    lines.append(f"Outcomes: {outcome_names}")
    for name in outcome_names:
        times = np.array([pt[name] for pt in all_times])
        valid = np.isfinite(times)
        events = np.array(
            [
                _as_event(rec[name].get("event", rec[name].get("label", -1)))
                if rec.get(name) is not None
                else -1
                for rec in records
            ]
        )
        primary_events = valid & (events == 1)
        if valid.sum() > 0:
            expected = ""
            if batch_size is not None and n > 0:
                expected = (
                    f", expected random batch: "
                    f"{batch_size * valid.sum() / n:.1f} valid / "
                    f"{batch_size * primary_events.sum() / n:.1f} events"
                )
            lines.append(
                f"  {name}: {valid.sum()}/{n} patients have data  "
                f"({primary_events.sum()} primary events"
                f"{expected})  "
                f"(median {np.nanmedian(times):.1f}d, "
                f"p25={np.nanquantile(times[valid], 0.25):.1f}d, "
                f"p75={np.nanquantile(times[valid], 0.75):.1f}d)"
            )
        else:
            lines.append(f"  {name}: no data")

    return "\n".join(lines)
