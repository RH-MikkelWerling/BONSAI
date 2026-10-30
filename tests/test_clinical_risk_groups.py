"""Scientific constraints for the optional grouping proposal, not server results."""

from itertools import combinations

import numpy as np
import pytest

from opera.evaluation.clinical_risk_groups import (
    RiskGroupsNotEstimable, fit_low_anchored_groups, cohort_size_matched_groups,
    fit_km_low_cutoff, km_survival,
)


NAMES = ["Low", "Low-intermediate", "High-intermediate", "High"]


def test_cohort_size_matching_uses_only_baseline_risk_and_categories():
    risk = np.arange(80) / 80
    native = np.array(NAMES * 20)
    match = cohort_size_matched_groups(risk, native, NAMES)
    assert match['model_counts'] == match['index_counts'] == [20] * 4
    assert match['exact_size_match']
    assert list(match['assignment'][:20]) == ['Low'] * 20
    # A wholly absent native category remains absent, not an invented stratum.
    native[native == NAMES[1]] = NAMES[2]
    match = cohort_size_matched_groups(risk, native, NAMES)
    assert match['model_counts'] == [20, 0, 40, 20]
    assert np.bincount(np.digitize(risk, match['cutoffs']), minlength=4).tolist() == [20, 0, 40, 20]


def test_size_matching_preserves_ties_and_reports_approximate_counts():
    risk = np.repeat([.1, .2, .3, .4], [15, 25, 30, 10])
    match = cohort_size_matched_groups(risk, np.array(NAMES * 20), NAMES)
    assert not match['exact_size_match']
    assert match['model_counts'] == [15, 25, 30, 10]


def test_direct_km_matching_includes_censored_patients_and_matches_exhaustive_prefix_search():
    rng = np.random.default_rng(922)
    risk = np.repeat(np.arange(25) / 25, 4)
    time = rng.choice([20., 50., 200., 500.], 100)
    event = rng.choice([0, 1], 100)
    native = np.array(['Low'] * 40 + ['High'] * 60)
    # Give the native reference enough horizon follow-up for a real KM target.
    time[:20], event[:20] = 500., 0
    target = km_survival(time[native == 'Low'], event[native == 'Low'], 365)
    feasible = []
    for cut in np.r_[np.unique(risk), np.nextafter(risk.max(), np.inf)]:
        low = risk < cut
        if low.sum() >= 20 and (time[low] >= 365).sum() >= 10 and km_survival(time[low], event[low], 365) >= target:
            feasible.append((cut, low.sum()))
    fitted = fit_km_low_cutoff(risk, time, event, native, 'Low')
    assert fitted['cutoff'] == feasible[-1][0]
    assert fitted['model_low_n'] == feasible[-1][1]
    assert fitted['target_survival'] == target


def test_km_match_refuses_unsupported_horizon_and_competing_events():
    risk = np.arange(100) / 100
    native = np.array(['Low'] * 40 + ['High'] * 60)
    with pytest.raises(RiskGroupsNotEstimable, match='follow-up'):
        fit_km_low_cutoff(risk, np.full(100, 100.), np.zeros(100), native, 'Low')
    with pytest.raises(ValueError, match='death-inclusive'):
        fit_km_low_cutoff(risk, np.full(100, 500.), np.full(100, 2), native, 'Low')


def example():
    risk = np.arange(200) / 200
    label = np.zeros(200)
    label[80::2] = 1
    native = np.array(["Low-intermediate"] * 200, dtype=object)
    native[100:160] = "High-intermediate"
    native[160:] = "High"
    native[:19] = "Low"
    native[90] = "Low"
    return risk, label, native


def test_can_expand_low_group_while_preserving_four_supported_native_categories():
    risk, label, native = example()
    fit = fit_low_anchored_groups(risk, label, native, NAMES)
    assert len(fit["cutoffs"]) == 3
    assert np.all(np.diff(fit["cutoffs"]) > 0)
    assert fit["groups"][0]["n"] == 88  # next attainable prefix exceeds 5%
    assert fit["groups"][0]["n"] > (native == "Low").sum()
    assert fit["groups"][0]["event_risk"] <= fit["index_low_risk"] == .05
    assert min(row["n"] for row in fit["groups"]) >= 20


def test_declining_event_risks_are_flagged_not_hidden_or_forced_into_order():
    # A misranking model still has an ordered *score* partition. It must not
    # be presented as successful clinical risk separation.
    risk = np.arange(100) / 100
    label = np.r_[np.ones(50), np.zeros(50)]
    native = np.array(["Low"] * 50 + ["Intermediate"] * 25 + ["High"] * 25)
    fit = fit_low_anchored_groups(risk, label, native, ["Low", "Intermediate", "High"])
    assert [g["n"] for g in fit["groups"]] == [60, 20, 20]
    assert not fit["development_group_risks_ordered"]


def test_binary_index_needs_only_one_cutoff_not_two_incompatible_anchors():
    risk, label, native = example()
    native[native != "Low"] = "High"
    fit = fit_low_anchored_groups(risk, label, native, ["Low", "High"])
    assert len(fit["cutoffs"]) == 1
    assert [g["n"] for g in fit["groups"]] == [88, 112]


def test_ties_cannot_be_split_to_manufacture_supported_groups():
    risk = np.repeat([.1, .2, .3, .4], [100, 20, 20, 60])
    label = np.r_[np.zeros(100), np.ones(100)]
    native = np.repeat(NAMES, 50)
    fit = fit_low_anchored_groups(risk, label, native, NAMES)
    assert fit["cutoffs"] == [.2, .3, .4]
    assert [g["n"] for g in fit["groups"]] == [100, 20, 20, 60]
    assert all(g["n"] >= fit["minimum_group"] for g in fit["groups"])
    risk[100:] = .2
    with pytest.raises(RiskGroupsNotEstimable) as exc:
        fit_low_anchored_groups(risk, label, native, NAMES)
    assert exc.value.code == "insufficient_distinct_scores"


def test_no_low_risk_match_is_an_explicit_failure_not_a_minimum_size_fallback():
    risk = np.arange(100) / 100
    label = np.r_[np.ones(50), np.zeros(50)]
    native = np.array(["High"] * 80 + ["Low"] * 20)
    with pytest.raises(RiskGroupsNotEstimable) as exc:
        fit_low_anchored_groups(risk, label, native, ["Low", "High"])
    assert exc.value.code == "no_supported_low_risk_match"


def test_ipcw_weights_affect_the_low_risk_constraint_and_effective_support():
    risk, label, native = example()
    weights = np.ones(200)
    weights[80] = 100
    fit = fit_low_anchored_groups(risk, label, native, NAMES, weights=weights)
    assert fit["groups"][0]["n"] == 80
    assignment = np.digitize(risk, fit["cutoffs"])
    for rank, group in enumerate(fit["groups"]):
        inside = assignment == rank
        assert group["event_risk"] == pytest.approx(np.average(label[inside], weights=weights[inside]))
        assert group["effective_n"] <= group["n"] + 1e-10


def test_reordering_patients_changes_neither_cutoffs_nor_diagnostics():
    risk, label, native = example()
    permutation = np.random.default_rng(42).permutation(len(risk))
    assert fit_low_anchored_groups(risk, label, native, NAMES) == fit_low_anchored_groups(
        risk[permutation], label[permutation], native[permutation], NAMES)


@pytest.mark.parametrize("seed", range(12))
def test_matches_exhaustive_joint_partition_search_on_small_tied_samples(seed):
    # Independent enumeration verifies both the largest feasible Low prefix
    # and the optimal tail allocation. It catches greedy quantile failures.
    rng = np.random.default_rng(seed)
    risk = np.repeat(np.arange(8) / 8, rng.integers(1, 4, 8))
    label = rng.integers(0, 2, len(risk))
    label[0], label[-1] = 1, 0
    native = rng.choice(NAMES, len(risk)).astype(object)
    native[0], native[-1] = "Low", "High"
    minimum = 2
    target = label[native == "Low"].mean()
    counts = np.r_[0, np.cumsum(np.unique(risk, return_counts=True)[1])]
    shares = np.array([(native == name).sum() for name in NAMES[1:]])
    choices = []
    for blocks in combinations(range(1, len(counts) - 1), 3):
        ends = counts[list(blocks)]
        if min(np.diff(np.r_[0, ends, len(risk)])) < minimum:
            continue
        if label[:ends[0]].mean() > target:
            continue
        ideal = ends[0] + (len(risk) - ends[0]) * np.cumsum(shares / shares.sum())[:-1]
        objective = float(np.sum((ends[1:] - ideal) ** 2))
        choices.append((ends[0], objective, blocks))
    if not choices:
        with pytest.raises(RiskGroupsNotEstimable):
            fit_low_anchored_groups(risk, label, native, NAMES, minimum_group=minimum, minimum_share=0)
        return
    largest = max(choice[0] for choice in choices)
    expected = min(choice[1] for choice in choices if choice[0] == largest)
    fit = fit_low_anchored_groups(risk, label, native, NAMES, minimum_group=minimum, minimum_share=0)
    assert fit["groups"][0]["n"] == largest
    assert fit["tail_share_objective"] == pytest.approx(expected)


@pytest.mark.parametrize("change", ["censored", "weight_zero", "missing_category", "nonfinite"])
def test_invalid_or_unknown_outcome_data_cannot_be_treated_as_clean_controls(change):
    risk, label, native = example()
    weights = np.ones(len(risk))
    if change == "censored":
        label[0] = np.nan
    elif change == "weight_zero":
        weights[0] = 0
    elif change == "missing_category":
        native[0] = "unmapped"
    else:
        risk[0] = np.inf
    with pytest.raises(ValueError):
        fit_low_anchored_groups(risk, label, native, NAMES, weights=weights)
