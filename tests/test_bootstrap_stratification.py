import numpy as np

from opera.evaluation.metrics import _stratified_bootstrap_indices


def test_stratified_bootstrap_indices_preserve_events():
    events = np.array([1, 1, 0, 0, 0, 2])
    rng = np.random.RandomState(7)

    for _ in range(50):
        idx = _stratified_bootstrap_indices(len(events), events, rng)
        assert (events[idx] == 1).sum() == 2
        assert len(idx) == len(events)
