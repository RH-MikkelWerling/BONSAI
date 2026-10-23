import numpy as np
import pandas as pd

from opera.evaluation.decision_curves import (
    decision_curves,
    figure_decision_curve,
    index_risk,
    ipcw_status,
    net_benefit,
)


def test_net_benefit_by_hand_without_censoring():
    risk = np.array([0.9, 0.6, 0.3, 0.1])
    case = np.array([1, 0, 1, 0])
    # At 0.5: acts on the first two, one true and one false positive among 4.
    value = net_benefit(risk, case, np.ones(4), [0.5])[0, 0]
    assert np.isclose(value, 1 / 4 - 1 / 4 * 0.5 / 0.5)
    # Acting for everyone: prevalence minus non-cases at the threshold odds.
    p = 0.2
    everyone = net_benefit(np.ones(4), case, np.ones(4), [p])[0, 0]
    assert np.isclose(everyone, 0.5 - 0.5 * p / (1 - p))


def test_censoring_weights_recover_the_uncensored_net_benefit():
    rng = np.random.default_rng(0)
    n, horizon = 40_000, 730
    risk = rng.uniform(0, 0.6, n)
    event_time = np.where(rng.uniform(size=n) < risk, rng.uniform(0, horizon, n), 5_000)
    censor = rng.uniform(200, 3_000, n)  # independent of the event
    time = np.minimum(event_time, censor)
    event = (event_time <= censor).astype(int)
    truth = net_benefit(risk, event_time <= horizon, np.ones(n), [0.2, 0.3])[0]
    case, weight = ipcw_status(time, event, horizon)
    estimate = net_benefit(risk, case, weight, [0.2, 0.3])[0]
    assert np.allclose(estimate, truth, atol=0.01)
    # Dropping censored patients instead would overstate the benefit.
    known = (time >= horizon) | (event == 1)
    naive = net_benefit(risk[known], case[known], np.ones(known.sum()), [0.2])[0, 0]
    assert naive > truth[0] + 0.01


def test_index_categories_take_their_tuning_risk():
    groups = np.array(["low", "low", "high", "high"])
    risk = index_risk(groups, np.array([0, 1, 1, 1]), np.ones(4), np.array(["high", "low"]))
    assert np.allclose(risk, [1.0, 0.5])


def test_a_perfect_model_beats_the_index_and_the_figure_renders():
    rng = np.random.default_rng(1)

    def patients(n):
        dead = rng.uniform(size=n) < 0.3
        return pd.DataFrame({
            "model_risk": np.where(dead, 0.9, 0.05),
            "ipi_group": rng.choice(["low", "high"], n),
            "time_days": np.where(dead, 100.0, 900.0),
            "event": dead.astype(int),
            "X": rng.uniform(size=n),
        })

    table = decision_curves(patients(400), patients(200), 730, [0.1, 0.2, 0.3],
                            comparators=["X"], n_boot=200)
    gain = table[table.strategy == "model_minus_index"]
    assert (gain.lower > 0).all()
    assert set(table.strategy) >= {"model", "index", "X", "treat_all", "treat_none",
                                   "model_minus_X"}
    figure_decision_curve(table, labels={"X": "XGBoost"})


def test_operating_points_are_computed_exactly_and_drawn():
    import pytest
    from opera.evaluation.decision_curves import operating_point_rows

    rng = np.random.default_rng(2)

    def patients(n):
        dead = rng.uniform(size=n) < 0.3
        return pd.DataFrame({"model_risk": np.clip(np.where(dead, 0.6, 0.15)
                                                   + rng.normal(0, 0.1, n), 0, 1),
                             "ipi_group": rng.choice(["low", "high"], n),
                             "time_days": np.where(dead, 100.0, 900.0),
                             "event": dead.astype(int)})

    points = {"Low-risk cutoff": 0.137, "High-risk cutoff": 0.4123}
    table = decision_curves(patients(500), patients(200), 730,
                            sorted({0.1, 0.2, 0.3, *points.values()}), n_boot=50)
    rows = operating_point_rows(table, points)
    assert set(rows.operating_point) == set(points)
    assert np.allclose(sorted(rows.threshold.unique()), sorted(points.values()))
    with pytest.raises(ValueError, match="not in the curve"):
        operating_point_rows(table, {"Elsewhere": 0.25})
    figure_decision_curve(table, points=points)
