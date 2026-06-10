import numpy as np

from opera.evaluation.metrics import (
    _km_censoring_fn,
    calibration_intercept_slope,
    compute_survival_metrics,
    full_evaluation,
    high_risk_enrichment,
)


def test_high_risk_enrichment_reports_top_fraction_lift():
    labels = np.array([1, 1, 0, 0, 0])
    probabilities = np.array([0.9, 0.8, 0.4, 0.2, 0.1])

    enrichment = high_risk_enrichment(
        labels,
        probabilities,
        fractions=[0.4, 1.0],
    )

    top = enrichment.iloc[0]
    assert top["n_top"] == 2
    assert top["n_events"] == 2
    assert top["event_rate"] == 1.0
    assert top["enrichment"] == 2.5


def test_calibration_intercept_slope_returns_values_for_two_class_labels():
    labels = np.array([0, 0, 1, 1])
    probabilities = np.array([0.1, 0.2, 0.8, 0.9])

    result = calibration_intercept_slope(labels, probabilities)

    assert "calibration_intercept" in result
    assert "calibration_slope" in result
    assert np.isfinite(result["calibration_intercept"])
    assert np.isfinite(result["calibration_slope"])


def test_full_evaluation_uses_explicit_survival_probabilities():
    labels = np.array([0, 1])
    probabilities = np.array([0.2, 0.8])
    times = np.array([10.0, 20.0, 30.0])
    events = np.array([0, 1, 0])
    survival_probabilities = np.array([0.2, 0.8, 0.4])

    report = full_evaluation(
        labels,
        probabilities,
        n_bootstrap=5,
        times=times,
        events=events,
        survival_probabilities=survival_probabilities,
        time_horizons=[15.0],
    )

    assert report["survival"]["n_total"] == 3


def test_survival_metrics_n_events_counts_only_primary_events():
    # With event=2 (competing death) in the cohort, n_events must count only
    # event==1 patients, not inflate by summing raw values (1+2=3 per pair).
    times = np.array([30.0, 60.0, 90.0, 120.0, 150.0])
    events = np.array([1, 2, 0, 1, 2])  # 2 primary, 2 competing, 1 admin
    preds = np.array([0.8, 0.4, 0.3, 0.7, 0.5])

    result = compute_survival_metrics(times, events, preds, time_horizons=[100.0])

    assert result["n_events"] == 2


def test_ipcw_metrics_exclude_competing_deaths_before_horizon():
    # Competing-death patients before the horizon should be excluded from
    # binary IPCW metrics (same as admin-censored patients before horizon).
    times = np.array([30.0, 60.0, 150.0, 200.0])
    events = np.array([1, 2, 0, 0])  # 1 primary event, 1 competing death
    preds = np.array([0.9, 0.4, 0.2, 0.1])

    result = compute_survival_metrics(times, events, preds, time_horizons=[100.0])

    per_h = result["per_horizon"].get("100d", {})
    # Patient 1 (competing death at 60d < 100d) must be excluded, not treated as a case
    assert per_h.get("n_cases", 0) == 1
    assert per_h.get("n_excluded", 0) == 1  # the competing death patient


def test_ipcw_brier_uses_event_probability_target_orientation():
    times = np.array([30.0, 150.0])
    events = np.array([1, 0])
    good_preds = np.array([0.9, 0.1])
    bad_preds = np.array([0.1, 0.9])

    good = compute_survival_metrics(
        times,
        events,
        good_preds,
        time_horizons=[100.0],
    )
    bad = compute_survival_metrics(
        times,
        events,
        bad_preds,
        time_horizons=[100.0],
    )

    assert (
        good["per_horizon"]["100d"]["ipcw_brier"]
        < bad["per_horizon"]["100d"]["ipcw_brier"]
    )


def test_km_censoring_function_returns_survival_just_before_tied_time():
    times = np.array([10.0, 10.0, 20.0])
    events = np.array([1, 0, 0])
    G_fn = _km_censoring_fn(times, events)

    assert G_fn(10.0) == 1.0
    assert G_fn(10.01) < 1.0


def test_full_evaluation_rejects_mismatched_survival_lengths():
    labels = np.array([0, 1])
    probabilities = np.array([0.2, 0.8])

    try:
        full_evaluation(
            labels,
            probabilities,
            n_bootstrap=5,
            times=np.array([10.0, 20.0, 30.0]),
            events=np.array([0, 1, 0]),
        )
    except ValueError as exc:
        assert "same patients" in str(exc)
    else:
        raise AssertionError("Expected mismatched survival inputs to fail.")
