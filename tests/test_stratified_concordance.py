import numpy as np
import pytest

from opera.evaluation.metrics import (
    _within_stratum_bootstrap_indices,
    compute_concordance_index,
    compute_macro_stratified_concordance,
    compute_stratified_concordance,
)


def test_single_stratum_matches_existing_pooled_concordance():
    times = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    events = np.array([1, 0, 1, 1, 0, 1])
    risk = np.array([0.9, 0.7, 0.8, 0.2, 0.4, 0.1])
    strata = np.repeat("DLBCL", len(times))

    pooled = compute_concordance_index(times, events, risk)
    stratified = compute_stratified_concordance(
        times,
        events,
        risk,
        strata,
        n_bootstrap=20,
        seed=7,
    )

    assert stratified["c_index"] == pooled
    assert stratified["n_strata"] == 1
    assert stratified["n_comparable_per_stratum"] == {
        "DLBCL": stratified["n_comparable"]
    }


def test_between_group_separation_does_not_inflate_stratified_concordance():
    n_per_group = 40
    times = np.concatenate(
        [
            np.arange(1, n_per_group + 1, dtype=float),
            np.arange(101, 101 + n_per_group, dtype=float),
        ]
    )
    events = np.ones(2 * n_per_group, dtype=int)
    risk = np.concatenate(
        [
            np.ones(n_per_group),
            np.zeros(n_per_group),
        ]
    )
    strata = np.repeat(["aggressive", "indolent"], n_per_group)

    pooled = compute_concordance_index(times, events, risk)
    stratified = compute_stratified_concordance(
        times,
        events,
        risk,
        strata,
        n_bootstrap=0,
    )

    assert pooled > 0.74
    assert stratified["c_index"] == pytest.approx(0.5)
    assert stratified["n_comparable"] == 2 * (n_per_group * (n_per_group - 1) // 2)


def test_macro_concordance_keeps_small_strata_and_flags_reliability():
    times = np.arange(1, 16, dtype=float)
    events = np.ones(15, dtype=int)
    risk = np.linspace(1.0, 0.0, 15)
    strata = np.array(["rare"] * 5 + ["common"] * 10)

    macro = compute_macro_stratified_concordance(
        times,
        events,
        risk,
        strata,
        n_bootstrap=10,
        seed=3,
        min_events=8,
    )
    by_stratum = {item["stratum"]: item for item in macro["per_stratum"]}

    assert macro["c_index"] == pytest.approx(1.0)
    assert macro["n_strata"] == 2
    assert by_stratum["rare"]["n_events"] == 5
    assert by_stratum["rare"]["reliable"] is False
    assert by_stratum["common"]["n_events"] == 10
    assert by_stratum["common"]["reliable"] is True


def test_micro_and_macro_weight_strata_differently():
    times = np.concatenate([np.arange(1, 4), np.arange(1, 11)]).astype(float)
    events = np.ones(13, dtype=int)
    risk = np.concatenate([np.array([0.9, 0.6, 0.3]), np.zeros(10)])
    strata = np.array(["small"] * 3 + ["large"] * 10)

    micro = compute_stratified_concordance(
        times,
        events,
        risk,
        strata,
        n_bootstrap=0,
    )
    macro = compute_macro_stratified_concordance(
        times,
        events,
        risk,
        strata,
        n_bootstrap=0,
        min_events=0,
    )

    assert micro["c_index"] == pytest.approx((3.0 + 0.5 * 45.0) / 48.0)
    assert macro["c_index"] == pytest.approx(0.75)


def test_none_strata_delegates_to_existing_pooled_point_estimate():
    times = np.array([1.0, 2.0, 3.0, 4.0])
    events = np.array([1, 1, 0, 1])
    risk = np.array([0.9, 0.8, 0.2, 0.1])

    result = compute_stratified_concordance(
        times,
        events,
        risk,
        None,
        n_bootstrap=0,
    )

    assert result["c_index"] == compute_concordance_index(times, events, risk)
    assert result["n_strata"] == 1


def test_bootstrap_preserves_event_counts_inside_each_stratum():
    events = np.array([1, 0, 0, 1, 1, 0, 0])
    strata = np.array(["A", "A", "A", "B", "B", "B", "B"])
    rng = np.random.RandomState(11)

    for _ in range(30):
        idx = _within_stratum_bootstrap_indices(events, strata, rng)
        for label in ("A", "B"):
            original = (events[strata == label] == 1).sum()
            sampled = (events[idx][strata[idx] == label] == 1).sum()
            assert sampled == original
