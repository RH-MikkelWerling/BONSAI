import numpy as np
import pandas as pd

from opera.evaluation.secondary_analysis import attach_ipcw_weights


def _rows(times, events, horizons, *, cut):
    rows = []
    for h in horizons:
        t = np.minimum(times, h) if cut else times
        e = np.where(cut & (times > h), 0, events)
        rows.append(pd.DataFrame({
            "model": "X", "split": "tuning", "seed": 42, "cohort": "CLL",
            "outcome": "infection", "horizon_days": h, "subject_id": range(len(times)),
            "probability": 0.2, "time_days": t, "event": e}))
    return pd.concat(rows, ignore_index=True)


def test_cutting_follow_up_at_the_horizon_leaves_the_weights_unchanged():
    # Tabular tables cut follow-up at each horizon; neural ones keep it whole.
    rng = np.random.default_rng(0)
    times = rng.uniform(1, 900, 400).round()
    events = rng.choice([0, 1, 2], 400, p=[0.5, 0.3, 0.2])
    horizons = (90, 365, 730)
    full = attach_ipcw_weights(_rows(times, events, horizons, cut=False))
    cut = attach_ipcw_weights(_rows(times, events, horizons, cut=True))
    np.testing.assert_allclose(full.ipcw_weight.to_numpy(), cut.ipcw_weight.to_numpy())
    assert (full.label.to_numpy() == cut.label.to_numpy()).all()


def test_models_with_cut_and_whole_follow_up_align_in_a_cell():
    from opera.evaluation.secondary_analysis import align_models

    rng = np.random.default_rng(1)
    times = rng.uniform(1, 900, 200).round()
    events = rng.choice([0, 1, 2], 200)
    whole = attach_ipcw_weights(_rows(times, events, (365,), cut=False)).assign(data_contract="c")
    cut = attach_ipcw_weights(_rows(times, events, (365,), cut=True)).assign(data_contract="c")
    aligned = align_models({"neural": whole, "tabular": cut})
    assert set(aligned) == {"neural", "tabular"}

    # A real disagreement (an event moved inside the horizon) still fails.
    broken = cut.copy()
    first = broken.index[(broken.time_days < 300) & (broken.event == 0)][0]
    broken.loc[first, "event"] = 2
    import pytest
    with pytest.raises(ValueError):
        align_models({"neural": whole, "tabular": broken})
