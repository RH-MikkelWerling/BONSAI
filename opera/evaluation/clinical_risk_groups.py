"""Explicit clinical comparison rules, without predictive model retraining.

The cohort size/KM rules are available through an explicit renderer mode;
the existing recipe remains unchanged unless that mode is selected. The
low-anchored full-partition candidate remains optional. Outcome-based cutoffs
use declared development data, never evaluation outcomes. Cohort size matching
uses baseline categories and predictions only.
"""

from __future__ import annotations

import numpy as np


def cohort_size_matched_groups(risk, index_group, group_order):
    """Baseline-only rank benchmark: no event or follow-up inputs are accepted.

    Match cumulative native category sizes as closely as attainable without
    splitting tied model scores. Native empty categories remain empty.
    """
    risk, native = np.asarray(risk, float), np.asarray(index_group)
    names = list(group_order)
    if (risk.ndim != 1 or not len(risk) or native.shape != risk.shape
            or len(names) < 2 or len(set(names)) != len(names)
            or not np.isfinite(risk).all() or not np.isin(native, names).all()):
        raise ValueError('Need finite model risks and mapped baseline index categories')
    native_n = np.array([(native == name).sum() for name in names])
    active = np.flatnonzero(native_n)
    scores, sizes = np.unique(risk, return_counts=True)
    if len(scores) < len(active):
        raise RiskGroupsNotEstimable('insufficient_distinct_scores', 'Too few distinct model scores for the populated native categories')
    ends = np.r_[0, np.cumsum(sizes)]
    blocks, objective = _partition_tail(ends, 0, native_n[active], 1)
    active_cuts = scores[blocks]
    original_rank = active[np.digitize(risk, active_cuts)]
    cuts = []
    for rank in range(1, len(names)):
        higher = np.flatnonzero(active >= rank)
        cuts.append(float(np.nextafter(scores[-1], np.inf)) if not len(higher) else
                    float(scores[0]) if higher[0] == 0 else float(active_cuts[higher[0] - 1]))
    model_n = np.bincount(original_rank, minlength=len(names))
    return dict(cutoffs=cuts, assignment=np.asarray(names)[original_rank],
                index_counts=native_n.tolist(), model_counts=model_n.tolist(),
                exact_size_match=bool(np.array_equal(native_n, model_n)),
                cumulative_count_objective=objective)


def km_survival(time, event, horizon):
    """KM survival for all-cause death/death-inclusive composites only."""
    time, event = np.asarray(time, float), np.asarray(event)
    if len(time) == 0:
        return float('nan')
    if time.shape != event.shape or not np.isin(event, [0, 1]).all():
        raise ValueError('KM matching requires event 0/1; use CIF for competing outcomes')
    at = np.unique(time[(event == 1) & (time <= horizon)])
    return float(np.prod([1 - np.sum((time == t) & (event == 1)) / np.sum(time >= t) for t in at]))


def fit_km_low_cutoff(risk, time, event, index_group, low_group, *, horizon=365,
                      minimum_group=20, minimum_at_risk=10):
    """Largest attainable development prefix with native Low KM survival.

    All follow-up records are used, including early-censored patients. Unlike
    the full-partition candidate, no arbitrary middle-stratum support limits
    the binary Low-group objective. No matching fallback is substituted.
    """
    risk, time, event, native = (np.asarray(risk, float), np.asarray(time, float),
                                np.asarray(event), np.asarray(index_group))
    if (risk.ndim != 1 or not len(risk) or any(a.shape != risk.shape for a in (time, event, native))
            or not np.isfinite(risk).all() or not np.isfinite(time).all()
            or (time < 0).any() or not np.isin(event, [0, 1]).all()
            or horizon <= 0 or minimum_group < 1 or minimum_at_risk < 1):
        raise ValueError('Need finite aligned death-inclusive development data and positive support settings')
    low = native == low_group
    if low.sum() < minimum_group or (time[low] >= horizon).sum() < minimum_at_risk:
        raise RiskGroupsNotEstimable('insufficient_native_low_followup', 'Native Low has insufficient patients or horizon follow-up for KM matching')
    target = km_survival(time[low], event[low], horizon)
    order = np.argsort(risk, kind='stable')
    r, t, e = risk[order], time[order], event[order]
    event_times = np.unique(t[(e == 1) & (t <= horizon)])
    survivals = np.ones(len(r))
    previous_risk = np.zeros(len(event_times))
    previous_deaths = np.zeros(len(event_times))
    # All prefix KMs, computed in small blocks rather than refitting N curves.
    for start in range(0, len(r), 128):
        stop = min(start + 128, len(r))
        at_risk = (t[start:stop, None] >= event_times).cumsum(0) + previous_risk
        deaths = ((t[start:stop, None] == event_times) & (e[start:stop, None] == 1)).cumsum(0) + previous_deaths
        fractions = np.divide(deaths, at_risk, out=np.zeros_like(deaths, float), where=at_risk > 0)
        survivals[start:stop] = np.prod(1 - fractions, axis=1)
        if len(event_times):
            previous_risk, previous_deaths = at_risk[-1], deaths[-1]
    ends = np.r_[np.flatnonzero(r[:-1] != r[1:]) + 1, len(r)]
    horizon_n = np.cumsum(t >= horizon)
    feasible = ends[(ends >= minimum_group) & (horizon_n[ends - 1] >= minimum_at_risk)
                    & (survivals[ends - 1] >= target)]
    if not len(feasible):
        raise RiskGroupsNotEstimable('no_supported_km_match', 'No supported model Low group reaches native Low KM survival')
    end = int(feasible[-1])
    cutoff = float(r[end]) if end < len(r) else float(np.nextafter(r[-1], np.inf))
    return dict(cutoff=cutoff, target_survival=target, model_development_survival=float(survivals[end - 1]),
                index_low_n=int(low.sum()), model_low_n=end, at_risk_horizon_n=int(horizon_n[end - 1]),
                feasible_boundaries=len(feasible), horizon_days=horizon)


class RiskGroupsNotEstimable(ValueError):
    """The declared risk and support constraints have no attainable partition."""

    def __init__(self, code, message):
        super().__init__(message)
        self.code = code


def _partition_tail(counts, low_block, shares, minimum):
    """Nearest cumulative share targets, subject to indivisible ties/support.

    Dynamic programming minimises the sum of squared deviations in cumulative
    patient counts. Ties in that objective choose earlier boundaries. Counts
    include a zero at the beginning and the total at the end.
    """
    remaining_groups = len(shares)
    total, low_n = counts[-1], counts[low_block]
    targets = low_n + (total - low_n) * np.cumsum(shares / shares.sum())[:-1]
    previous = np.full(len(counts), np.inf)
    previous[low_block] = 0.
    parents = []
    positions = np.arange(len(counts))
    for stage, target in enumerate(targets, start=1):
        # A prefix minimum supplies the best preceding boundary that leaves
        # at least `minimum` patients in this group: O(groups * score blocks).
        best = np.minimum.accumulate(previous)
        improved = np.r_[True, previous[1:] < best[:-1]]
        best_index = np.maximum.accumulate(np.where(improved, positions, 0))
        last_predecessor = np.searchsorted(counts, counts - minimum, side="right") - 1
        safe = np.maximum(last_predecessor, 0)
        current = best[safe] + (counts - target) ** 2
        possible = ((last_predecessor >= 0) & (positions > low_block)
                    & (counts <= total - (remaining_groups - stage) * minimum))
        current[~possible] = np.inf
        parents.append(best_index[safe])
        previous = current
    if not len(targets):
        return [], 0.
    possible_end = (counts <= total - minimum) & np.isfinite(previous)
    if not possible_end.any():
        raise RiskGroupsNotEstimable("insufficient_tail_support", "No supported ordered tail partition")
    last = int(np.argmin(np.where(possible_end, previous, np.inf)))
    objective = float(previous[last])
    boundaries = []
    for parent in reversed(parents):
        boundaries.append(last)
        last = int(parent[last])
    return list(reversed(boundaries)), objective


def fit_low_anchored_groups(risk, label, index_group, group_order, *, weights=None,
                            minimum_group=20, minimum_share=.05):
    """Fit an ordered model classification from a paired development sample.

    The largest supported model low-risk prefix must have an IPCW-weighted
    event rate no greater than the native index Low group in this SAME sample.
    All K groups must retain min support. The remaining K-1 groups approximate
    the native non-Low category proportions, without a high-specificity anchor.
    They are descriptive model strata, not established equivalent clinical risks.

    Inputs contain patients with known horizon status and strictly positive
    censoring weights. Early-censored patients must not be coded as controls.
    Equal model scores always remain together. Membership is risk >= cutoff
    at each boundary (np.digitize with its default right=False).

    This uses development labels and estimates no out-of-sample performance.
    The minimum size is a support constraint, not a precision guarantee. Event
    risks need not increase across fitted groups and are returned for auditing.
    """
    risk, label = np.asarray(risk, float), np.asarray(label, float)
    groups = np.asarray(index_group)
    names = list(group_order)
    if (risk.ndim != 1 or label.shape != risk.shape or groups.shape != risk.shape
            or len(names) < 2 or len(set(names)) != len(names)):
        raise ValueError("Aligned one-dimensional arrays and at least two distinct ordered categories required")
    if (not np.isfinite(risk).all() or not np.isfinite(label).all()
            or not np.isin(label, [0, 1]).all() or not np.isin(groups, names).all()):
        raise ValueError("Finite scores, binary known labels and mapped native categories required")
    if (isinstance(minimum_group, (bool, np.bool_)) or int(minimum_group) != minimum_group
            or minimum_group < 1 or not np.isfinite(minimum_share)
            or not 0 <= minimum_share < 1):
        raise ValueError("Positive integer minimum_group and minimum_share in [0, 1) required")
    weights = np.ones(len(risk)) if weights is None else np.asarray(weights, float)
    if weights.shape != risk.shape or not np.isfinite(weights).all() or (weights <= 0).any():
        raise ValueError("Finite positive weights aligned with known labels required")
    k, n = len(names), len(risk)
    minimum = max(int(minimum_group), int(np.ceil(minimum_share * n)))
    if n < k * minimum:
        raise RiskGroupsNotEstimable("insufficient_group_support", "Too few known patients for the requested categories and minimum sizes")
    if np.unique(label).size != 2:
        raise RiskGroupsNotEstimable("missing_cases_or_controls", "Development requires both cases and controls")
    native_counts = np.array([(groups == name).sum() for name in names])
    if not native_counts[0] or not native_counts[1:].sum():
        raise RiskGroupsNotEstimable("missing_index_anchor", "The index must have Low and non-Low patients")
    low_native = groups == names[0]
    target = float(np.average(label[low_native], weights=weights[low_native]))
    ranked = np.argsort(risk, kind="stable")
    scores, sizes = np.unique(risk[ranked], return_counts=True)
    if len(scores) < k:
        raise RiskGroupsNotEstimable("insufficient_distinct_scores", "Too few distinct scores to keep ties intact across all categories")
    counts = np.r_[0, np.cumsum(sizes)]
    cumulative_weight = np.r_[0., np.cumsum(weights[ranked])][counts]
    cumulative_events = np.r_[0., np.cumsum(weights[ranked] * label[ranked])][counts]
    prefix_rate = np.divide(cumulative_events, cumulative_weight,
                            out=np.full(len(counts), np.inf), where=cumulative_weight > 0)
    candidates = np.arange(1, len(scores))
    feasible = (counts[candidates] >= minimum) & (prefix_rate[candidates] <= target)
    # Greedily pack the remaining groups at their earliest attainable ends.
    # If that cannot fit, no partition of that tail can meet all minimum sizes.
    end = candidates.copy()
    for _ in range(k - 1):
        end = np.searchsorted(counts, counts[np.minimum(end, len(scores))] + minimum)
    feasible &= end <= len(scores)
    if not feasible.any():
        raise RiskGroupsNotEstimable("no_supported_low_risk_match", "No low-risk match leaves enough supported groups; no fallback cutoff applied")
    low_block = int(candidates[feasible][-1])
    inner, objective = _partition_tail(counts, low_block, native_counts[1:], minimum)
    boundary_blocks = [low_block, *inner]
    cuts = scores[boundary_blocks]
    assignment = np.digitize(risk, cuts)
    rows = []
    for rank, name in enumerate(names):
        inside = assignment == rank
        mass = weights[inside]
        rows.append(dict(group=name, n=int(inside.sum()),
                         index_n=int(native_counts[rank]),
                         event_risk=float(np.average(label[inside], weights=mass)),
                         effective_n=float(mass.sum() ** 2 / (mass @ mass))))
    return dict(rule="low_anchored_ordered_candidate", cutoffs=cuts.tolist(),
                n_known=n, minimum_group=minimum, index_low_risk=target,
                feasible_low_boundaries=int(feasible.sum()),
                tail_share_objective=objective, groups=rows,
                development_group_risks_ordered=bool(np.all(np.diff([row["event_risk"] for row in rows]) >= 0)),
                note="Development fit only. No high-specificity match or risk equivalence for middle/high groups is imposed.")
