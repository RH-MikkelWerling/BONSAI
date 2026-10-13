from collections import Counter

import pytest

import numpy as np
import torch

from opera.modules.datamodules.ContrastiveDataModule import contrastive_collate
from opera.functional.stratified_sampling import (
    ImportanceWeightedHorizonSampler,
    MinorityEventBatchSampler,
)


class _Dataset:
    def __init__(self, n=120):
        self.subjects = [{"subject_id": index} for index in range(n)]
        self.outcome_dicts = {
            "common": {
                index: {"event": int(index < 60), "time_days": 100 if index < 60 else 500}
                for index in range(n)
            },
            "tiny": {
                index: {"event": int(index < 5), "time_days": 100 if index < 5 else 500}
                for index in range(n)
            },
        }

    def __len__(self):
        return len(self.subjects)


def test_minority_sampler_respects_support_and_exposure_cap():
    sampler = MinorityEventBatchSampler(
        _Dataset(), ["common", "tiny"], batch_size=20,
        min_unique_events=50, min_unique_controls=50,
        event_slots_per_batch=4, max_draws_per_patient=2, seed=4,
    )
    assert sampler.focus_names == ["common"]
    batches = list(sampler)
    assert len(batches) == 6
    assert all(len(batch) == len(set(batch)) == 20 for batch in batches)
    assert max(Counter(index for batch in batches for index in batch).values()) <= 2
    assert all(sum(index < 60 for index in batch) >= 4 for batch in batches)


def test_minority_sampler_rejects_empty_focus_panel():
    with pytest.raises(ValueError, match="No outcomes meet"):
        MinorityEventBatchSampler(
            _Dataset(), ["tiny"], batch_size=20, min_unique_events=50,
        )


def test_ipw_sampler_covers_all_horizons_and_corrects_proposal():
    dataset = _Dataset()
    sampler = ImportanceWeightedHorizonSampler(
        dataset, ["common", "tiny"], batch_size=20,
        horizons_days=[180, 365], event_fraction=0.5,
        min_unique_events=5, min_unique_controls=50, seed=4,
    )
    assert len(sampler.event_indices) == 4
    assert sampler.proposal.sum() == pytest.approx(1.0)
    assert sampler.importance_weights.max() <= 2.0 + 1e-10
    assert 0 < sampler.effective_sample_fraction <= 1.0
    assert 0 < sampler.expected_unique_fraction <= 1.0
    assert np.sum(sampler.proposal * sampler.importance_weights) == pytest.approx(1.0)
    values = np.arange(len(dataset), dtype=np.float64)
    assert np.sum(
        sampler.proposal * sampler.importance_weights * values
    ) == pytest.approx(values.mean())
    assert sampler.proposal[0] > sampler.proposal[100]
    assert len(list(sampler)) == 120


def test_ipw_sampler_rejects_unsupported_horizon():
    with pytest.raises(ValueError, match="No supported event focus"):
        ImportanceWeightedHorizonSampler(
            _Dataset(), ["common"], batch_size=20,
            horizons_days=[30, 180], min_unique_events=5,
            min_unique_controls=50,
        )


def test_sampling_weights_survive_contrastive_collation():
    samples = [
        {
            "subject_id": index,
            "code": torch.tensor([1, 2], dtype=torch.long),
            "time_common": torch.tensor(100.0),
            "event_common": torch.tensor(index % 2),
            "sampling_weight": torch.tensor(weight),
        }
        for index, weight in enumerate((0.5, 1.5))
    ]
    batch = contrastive_collate(samples)
    assert torch.equal(batch["sampling_weight"], torch.tensor([0.5, 1.5]))
