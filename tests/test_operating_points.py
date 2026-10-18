import numpy as np
import pandas as pd
import pytest

from opera.evaluation.operating_points import (
    choose_threshold,
    operating_points,
    suppress_small_counts,
    threshold_metrics,
)
from opera.evaluation.secondary_analysis import patient_bootstrap


def test_threshold_metrics_match_hand_counts():
    metrics = threshold_metrics(8, 2, 2, 88)
    assert metrics["sensitivity"] == pytest.approx(0.8)
    assert metrics["specificity"] == pytest.approx(88 / 90)
    assert metrics["ppv"] == pytest.approx(0.8)
    expected = (8 * 88 - 2 * 2) / np.sqrt(10 * 10 * 90 * 90)
    assert metrics["mcc"] == pytest.approx(expected)


def test_event_rate_rule_flags_as_many_patients_as_events():
    probability = np.linspace(0, 1, 100)
    label = np.r_[np.zeros(80), np.ones(20)]
    threshold = choose_threshold(probability, label, "tuning_event_rate")
    assert (probability >= threshold).sum() == 20


def test_max_mcc_rule_finds_the_separating_threshold():
    probability = np.r_[np.full(50, 0.1), np.full(10, 0.9)]
    label = np.r_[np.zeros(50), np.ones(10)]
    assert choose_threshold(probability, label, "tuning_max_mcc") == pytest.approx(0.9)


def _predictions(rng):
    rows = []
    for split, n in (("tuning", 300), ("held_out", 400)):
        label = rng.random(n) < 0.15
        for model, noise in (("J_L", 0.5), ("X", 1.5)):
            score = label + rng.normal(0, noise, n)
            for subject, (y, p) in enumerate(zip(label, 1 / (1 + np.exp(-score)))):
                rows.append({"split": split, "seed": 42, "cohort": "DLBCL",
                             "outcome": "TF", "horizon_days": 365,
                             "subject_id": f"{split}{subject}", "model": model,
                             "label": int(y), "probability": p, "known": True})
    return pd.DataFrame(rows)


def test_operating_points_threshold_comes_from_tuning_and_pairs_are_paired():
    predictions = _predictions(np.random.default_rng(0))
    draws = patient_bootstrap(predictions, 200, 2026)
    points, differences = operating_points(predictions, draws, pairs=[("J_L", "X")])
    joint = points.loc[points.model == "J_L"].iloc[0]
    tuning = predictions.loc[(predictions.split == "tuning") & (predictions.model == "J_L")]
    assert joint.threshold == pytest.approx(
        choose_threshold(tuning.probability, tuning.label, "tuning_event_rate"))
    assert joint.tp + joint.fp + joint.fn + joint.tn == 400
    assert joint.mcc_lower <= joint.mcc <= joint.mcc_upper
    row = differences.iloc[0]
    assert row.mcc > 0 and row.mcc_lower > 0  # the less noisy model is better


def test_unknown_status_is_excluded_and_small_counts_suppressed():
    predictions = _predictions(np.random.default_rng(1))
    predictions.loc[predictions.subject_id == "held_out0", "known"] = False
    points, _ = operating_points(predictions, {}, rule="tuning_max_mcc")
    assert (points[["tp", "fp", "fn", "tn"]].sum(axis=1) == 399).all()
    safe = suppress_small_counts(points.assign(fn=3))
    assert safe.fn.isna().all() and safe.fn_suppressed.all()
    assert safe.mcc.notna().all()


def test_specificity_rule_reaches_target_on_tuning_data():
    rng = np.random.default_rng(3)
    label = rng.random(500) < 0.2
    probability = 1 / (1 + np.exp(-(label + rng.normal(0, 1, 500))))
    for target in (0.8, 0.9, 0.95):
        threshold = choose_threshold(probability, label, f"tuning_specificity:{target}")
        specificity = (probability[~label] < threshold).mean()
        assert specificity >= target
        # The lowest such threshold: one step lower falls below the target.
        lower = probability[probability < threshold].max()
        assert (probability[~label] < lower).mean() < target


def test_youden_rule_and_rule_validation():
    probability = np.r_[np.full(50, 0.1), np.full(10, 0.9)]
    label = np.r_[np.zeros(50), np.ones(10)]
    assert choose_threshold(probability, label, "tuning_youden") == pytest.approx(0.9)
    with pytest.raises(ValueError):
        choose_threshold(probability, label, "tuning_specificity")
    with pytest.raises(ValueError):
        choose_threshold(probability, label, "youden")
