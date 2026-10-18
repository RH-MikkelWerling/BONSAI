"""Threshold metrics at decision points fixed on tuning data.

For every model, disease, outcome and horizon a threshold is chosen on the
model's own tuning predictions and applied unchanged to held-out patients.
Confusion counts, sensitivity, specificity, predictive values and the Matthews
correlation coefficient use patients whose status at the horizon is known.
Intervals reuse the shared patient bootstrap of the paired analysis, so
differences between models are paired.

Rules (all fixed on tuning data):
  ``tuning_event_rate``        flag the share of patients equal to the tuning
                               event rate (every model raises as many alerts);
  ``tuning_youden``            maximise sensitivity + specificity - 1;
  ``tuning_max_mcc``           maximise the Matthews correlation coefficient;
  ``tuning_specificity:0.90``  the lowest threshold whose tuning specificity is
                               at least 0.90 (any target in (0, 1)).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TASK = ["seed", "cohort", "outcome", "horizon_days"]
COUNTS = ["tp", "fp", "fn", "tn"]
METRICS = ["sensitivity", "specificity", "ppv", "npv", "mcc", "flagged"]
RULES = ("tuning_event_rate", "tuning_youden", "tuning_max_mcc", "tuning_specificity")
DEFAULT_RULES = ("tuning_event_rate", "tuning_youden", "tuning_max_mcc",
                 "tuning_specificity:0.80", "tuning_specificity:0.90",
                 "tuning_specificity:0.95")


def _check_rule(rule):
    name, _, target = str(rule).partition(":")
    if name not in RULES or (name == "tuning_specificity") != bool(target):
        raise ValueError(f"Unknown operating-point rule: {rule}; use one of {DEFAULT_RULES}")
    if target and not 0 < float(target) < 1:
        raise ValueError("Target specificity must lie in (0, 1)")
    return name, float(target) if target else None


def threshold_metrics(tp, fp, fn, tn):
    """Metrics from (possibly weighted) confusion counts; arrays broadcast."""
    tp, fp, fn, tn = (np.asarray(value, float) for value in (tp, fp, fn, tn))
    with np.errstate(divide="ignore", invalid="ignore"):
        denominator = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
        return {
            "sensitivity": tp / (tp + fn),
            "specificity": tn / (tn + fp),
            "ppv": tp / (tp + fp),
            "npv": tn / (tn + fn),
            "mcc": np.where(denominator > 0, (tp * tn - fp * fn) / denominator, np.nan),
            "flagged": (tp + fp) / (tp + fp + fn + tn),
        }


def choose_threshold(probability, label, rule):
    """Threshold on tuning predictions; patients at or above it are flagged."""
    probability = np.asarray(probability, float)
    label = np.asarray(label, int)
    if rule == "tuning_event_rate":
        rate = label.mean()
        # The flagged share equals the event rate: the top-ranked rate * n patients.
        flagged = max(1, int(round(rate * len(probability))))
        return float(np.sort(probability)[::-1][flagged - 1])
    name, target = _check_rule(rule)
    candidates = np.unique(probability)
    controls = np.sort(probability[label == 0])
    cases = np.sort(probability[label == 1])
    # Flagged if probability >= threshold: specificity counts controls below it.
    specificity = np.searchsorted(controls, candidates, side="left") / len(controls)
    sensitivity = 1 - np.searchsorted(cases, candidates, side="left") / len(cases)
    if name == "tuning_specificity":
        eligible = np.flatnonzero(specificity >= target)
        if not len(eligible):
            return float(np.inf)
        return float(candidates[eligible[0]])
    if name == "tuning_youden":
        return float(candidates[int(np.argmax(sensitivity + specificity - 1))])
    tp = sensitivity * len(cases)
    fn = len(cases) - tp
    tn = specificity * len(controls)
    fp = len(controls) - tn
    mcc = np.nan_to_num(threshold_metrics(tp, fp, fn, tn)["mcc"], nan=-np.inf)
    return float(candidates[int(np.argmax(mcc))])


def _weighted_counts(flag, label, weights):
    """Confusion counts for every bootstrap replicate (rows of ``weights``)."""
    flag, label = np.asarray(flag, bool), np.asarray(label, int)
    return (
        weights @ (flag & (label == 1)),
        weights @ (flag & (label == 0)),
        weights @ (~flag & (label == 1)),
        weights @ (~flag & (label == 0)),
    )


def _interval(draws):
    finite = draws[np.isfinite(draws)]
    if len(finite) < 0.8 * len(draws):
        return np.nan, np.nan
    return tuple(np.quantile(finite, [0.025, 0.975]))


def operating_points(predictions, draws, *, rule="tuning_event_rate", pairs=(),
                     minimum_events=5):
    """Per-model metrics and paired model differences at tuning-fixed thresholds.

    ``predictions`` is the normalised long table of the locked analysis (both
    splits); ``draws`` is its ``patient_bootstrap`` output. Returns
    ``(points, differences)``: one row per model and task with counts, metrics
    and 95% intervals, and one row per ``(model_a, model_b)`` pair and task
    with the paired difference ``a - b`` of every metric.
    """
    _check_rule(rule)
    known = predictions.loc[predictions.known.astype(bool)] if "known" in predictions else predictions
    rows, replicate_metrics = [], {}
    for (seed, cohort, outcome, horizon, model), task in known.groupby([*TASK, "model"], sort=True):
        tuning = task.loc[task.split == "tuning"]
        test = task.loc[task.split == "held_out"].sort_values("subject_id")
        record = {"seed": seed, "cohort": cohort, "outcome": outcome,
                  "horizon_days": horizon, "model": model, "rule": rule}
        events = int(tuning.label.sum())
        if test.empty:
            continue
        if min(events, len(tuning) - events) < minimum_events:
            rows.append(record | {"status": "too_few_tuning_events"})
            continue
        threshold = choose_threshold(tuning.probability, tuning.label, rule)
        flag = test.probability.to_numpy() >= threshold
        label = test.label.to_numpy(int)
        counts = _weighted_counts(flag, label, np.ones((1, len(test))))
        point = {name: float(value[0]) for name, value in threshold_metrics(*counts).items()}
        record |= {"status": "estimated", "threshold": threshold, "n": len(test),
                   **{name: int(value[0]) for name, value in zip(COUNTS, counts)}, **point}
        key = ("held_out", seed, cohort)
        if key in draws:
            index, weights = draws[key]
            weights = weights[:, [index[str(subject)] for subject in test.subject_id]]
            sampled = threshold_metrics(*_weighted_counts(flag, label, weights))
            replicate_metrics[(seed, cohort, outcome, horizon, model)] = sampled
            for name in METRICS:
                record[f"{name}_lower"], record[f"{name}_upper"] = _interval(sampled[name])
        rows.append(record)
    points = pd.DataFrame(rows)
    differences = []
    for model_a, model_b in pairs:
        for (seed, cohort, outcome, horizon, model), sampled in replicate_metrics.items():
            if model != model_a:
                continue
            other = replicate_metrics.get((seed, cohort, outcome, horizon, model_b))
            if other is None:
                continue
            a = points.loc[(points.model == model_a) & (points.seed == seed) & (points.cohort == cohort)
                           & (points.outcome == outcome) & (points.horizon_days == horizon)].iloc[0]
            b = points.loc[(points.model == model_b) & (points.seed == seed) & (points.cohort == cohort)
                           & (points.outcome == outcome) & (points.horizon_days == horizon)].iloc[0]
            row = {"seed": seed, "cohort": cohort, "outcome": outcome, "horizon_days": horizon,
                   "model_a": model_a, "model_b": model_b, "rule": rule}
            for name in METRICS:
                row[name] = a[name] - b[name]
                row[f"{name}_lower"], row[f"{name}_upper"] = _interval(sampled[name] - other[name])
            differences.append(row)
    return points, pd.DataFrame(differences)


def suppress_small_counts(points, minimum=5):
    """Copy safe to export: confusion counts below ``minimum`` become missing.

    Metrics are kept; the counts that would identify small groups are not.
    """
    safe = points.copy()
    for column in COUNTS:
        if column in safe:
            small = safe[column].notna() & (safe[column] < minimum)
            safe[column] = safe[column].astype("Float64").mask(small)
            safe[f"{column}_suppressed"] = small
    return safe
