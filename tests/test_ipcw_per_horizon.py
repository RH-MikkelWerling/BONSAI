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


def test_parallel_paired_analysis_equals_the_serial_one(monkeypatch):
    from opera.evaluation import secondary_analysis as sa

    rng = np.random.default_rng(3)
    rows = []
    for cohort in ("CLL", "DLBCL", "MM"):
        times = rng.uniform(1, 900, 120).round()
        events = rng.choice([0, 1, 2], 120)
        for model, noise in (("A", 0.1), ("B", 0.2)):
            frame = _rows(times, events, (90, 365), cut=False)
            frame["cohort"], frame["model"], frame["data_contract"] = cohort, model, "c"
            frame["subject_id"] = [f"{cohort}{i}" for i in frame.subject_id]
            frame["probability"] = np.clip(0.3 + rng.normal(0, noise, len(frame)), 0.01, 0.99)
            rows.append(frame)
    frame, _ = sa.normalize_predictions(pd.concat(rows, ignore_index=True), "cif_ipcw")
    monkeypatch.setattr(sa, "CELL_BATCH", 2)
    serial = sa.paired_analysis(frame, {"d": "gain(A,B)"}, replicates=50)
    monkeypatch.setenv("OPERA_ANALYSIS_WORKERS", "3")
    parallel = sa.paired_analysis(frame, {"d": "gain(A,B)"}, replicates=50)
    for name in ("estimates", "contrasts", "contrast_coverage"):
        pd.testing.assert_frame_equal(serial[name], parallel[name])


def _reference_weights(frame):
    """The former group-by-group loop, kept as the reference implementation."""
    from opera.evaluation.secondary_analysis import _censoring_survival_before

    parts = []
    keys = ["model", "split", "seed", "cohort", "outcome", "horizon_days"]
    for _, group in frame.groupby(keys, sort=False):
        before = _censoring_survival_before(group.time_days.to_numpy(float),
                                            group.event.to_numpy(int))
        t, e, h = (group[c].to_numpy(v) for c, v in
                   (("time_days", float), ("event", int), ("horizon_days", float)))
        case, competing = (t <= h) & (e == 1), (t <= h) & (e == 2)
        control = competing | (t > h) | ((t >= h) & (e == 0))
        weight = np.zeros(len(group))
        weight[case | competing] = 1.0 / before(t[case | competing])
        weight[control & ~competing] = 1.0 / before(h[control & ~competing])
        parts.append(group.assign(label=case.astype(int), ipcw_weight=weight, known=case | control))
    return pd.concat(parts).sort_index()


def test_vectorised_weights_equal_the_group_loop():
    rng = np.random.default_rng(7)
    rows = []
    for cohort in ("CLL", "DLBCL", "MM", "HL"):
        n = int(rng.integers(5, 80))
        times = rng.integers(1, 900, n).astype(float)  # integer days: many ties
        events = rng.choice([0, 1, 2], n, p=[0.6, 0.25, 0.15])
        for model in ("A", "B", "C"):
            frame = _rows(times, events, (30, 90, 365, 730), cut=(model == "C"))
            frame["cohort"], frame["model"] = cohort, model
            rows.append(frame)
    # A group where every patient is censored at the same time (curve drops to 0).
    rows.append(_rows(np.full(6, 50.0), np.zeros(6, int), (30, 90), cut=False).assign(
        cohort="SolM", model="A"))
    frame = pd.concat(rows, ignore_index=True)
    fast = attach_ipcw_weights(frame)
    slow = _reference_weights(frame)
    np.testing.assert_allclose(fast.ipcw_weight.to_numpy(), slow.ipcw_weight.to_numpy(), rtol=1e-12)
    assert (fast.label.to_numpy() == slow.label.to_numpy()).all()
    assert (fast.known.to_numpy() == slow.known.to_numpy()).all()


def _reference_seed_averaged(contrasts, bootstrap_draws):
    from opera.evaluation.secondary_analysis import _interval

    keys = ["split", "cohort", "outcome", "horizon_days", "contrast", "metric"]
    rows = []
    for key, group in contrasts.groupby(keys):
        samples = bootstrap_draws
        for column, value in zip(keys, key):
            samples = samples.loc[samples[column] == value]
        arrays = np.stack(samples.loc[samples.seed.isin(group.seed)].gain_draws)
        average = arrays.mean(axis=0)
        average[~np.isfinite(arrays).all(axis=0)] = np.nan
        rows.append({**dict(zip(keys, key)),
                     "gain": group.gain.mean() if np.isfinite(group.gain).all() else np.nan,
                     "n_training_seeds": len(group), "seed_minimum": group.gain.min(),
                     "seed_maximum": group.gain.max(), "seed_sd": group.gain.std(),
                     "uncertainty": "patients_conditional_on_fitted_seed_roster",
                     **_interval(average)})
    return pd.DataFrame(rows)


def test_vectorised_seed_averaging_equals_the_loop():
    from opera.evaluation.secondary_analysis import seed_averaged_effects

    rng = np.random.default_rng(11)
    contrasts, draws = [], []
    for cohort in ("CLL", "MM"):
        for outcome in ("os", "inf"):
            for horizon in (90, 365):
                seeds = (42,) if outcome == "os" else (42, 43)  # one and two seeds
                for seed in seeds:
                    key = {"split": "tuning", "cohort": cohort, "outcome": outcome,
                           "horizon_days": horizon, "contrast": "d", "metric": "auc", "seed": seed}
                    sample = rng.normal(0.01, 0.02, 200)
                    if cohort == "MM" and horizon == 90:
                        sample[:150] = np.nan  # too few valid draws
                    contrasts.append({**key, "gain": float(np.nanmean(sample))})
                    draws.append({**key, "gain_draws": sample})
    contrasts, draws = pd.DataFrame(contrasts), pd.DataFrame(draws)
    fast = seed_averaged_effects(contrasts, draws)
    slow = _reference_seed_averaged(contrasts, draws)
    pd.testing.assert_frame_equal(fast.reset_index(drop=True), slow[fast.columns].reset_index(drop=True),
                                  check_dtype=False)


def test_parallel_calibration_summary_equals_serial(monkeypatch):
    from opera.evaluation import secondary_analysis as sa

    rng = np.random.default_rng(5)
    rows = []
    for model in ("A", "B", "C"):
        times = rng.uniform(1, 900, 300).round()
        events = rng.choice([0, 1, 2], 300, p=[0.4, 0.4, 0.2])
        frame = _rows(times, events, (365, 730), cut=False)
        frame["model"], frame["data_contract"] = model, "c"
        frame["probability"] = np.clip(rng.uniform(0.05, 0.9, len(frame)), 0.01, 0.99)
        rows.append(frame)
    frame, _ = sa.normalize_predictions(pd.concat(rows, ignore_index=True), "cif_ipcw")
    serial = sa.calibration_summary(frame)
    monkeypatch.setenv("OPERA_ANALYSIS_WORKERS", "3")
    parallel = sa.calibration_summary(frame)
    pd.testing.assert_frame_equal(serial, parallel)


# The former per-subgroup loops, kept as reference implementations.
from opera.evaluation.secondary_analysis import CELL, _interval, subgroup_labels as _subgroup_labels  # noqa: E402

def _ref_subgroup_patient_gains(gains, metadata, specifications, *, draws):
    """Pre-index, prespecified subgroup loss means with joint patient bootstrap."""
    rows = []
    for column, enriched in _subgroup_labels(gains, metadata, specifications):
        for key, group in enriched.groupby([*CELL, "contrast", "subgroup"]):
            identity = dict(zip([*CELL, "contrast", "subgroup"], key))
            mapping, counts = draws[key[:3]]
            weights = counts[:, [mapping[sid] for sid in group.subject_id]]
            denominators = weights.sum(axis=1)
            for metric in ("brier_gain", "log_loss_gain"):
                samples = np.divide(
                    weights @ group[metric].to_numpy(),
                    denominators,
                    out=np.full(len(weights), np.nan),
                    where=denominators > 0,
                )
                rows.append(
                    {
                        **identity,
                        "covariate": column,
                        "metric": metric,
                        "gain": group[metric].mean(),
                        "n_subjects": group.subject_id.nunique(),
                        **_interval(samples),
                    }
                )
    return pd.DataFrame(rows)


def _ref_panel_subgroup_gains(gains, metadata, specifications, *, draws):
    """Subgroup loss gains pooled over every cell at a horizon.

    Each patient contributes once per cell in which they are evaluated, so the
    estimate is the patient-weighted mean gain across the panel. Intervals use
    the shared within-disease patient resamples, so a patient appearing in
    several cells is resampled jointly across them.
    """
    keys = ["split", "seed", "horizon_days", "contrast", "subgroup"]
    rows = []
    for column, enriched in _subgroup_labels(gains, metadata, specifications):
        for key, panel in enriched.groupby(keys):
            totals = {metric: 0.0 for metric in ("brier_gain", "log_loss_gain")}
            denominator = 0.0
            for cell_key, group in panel.groupby(CELL):
                mapping, counts = draws[cell_key[:3]]
                weights = counts[:, [mapping[sid] for sid in group.subject_id]]
                denominator = denominator + weights.sum(axis=1)
                for metric in totals:
                    totals[metric] = totals[metric] + weights @ group[metric].to_numpy()
            for metric, numerator in totals.items():
                samples = np.divide(
                    numerator, denominator,
                    out=np.full(len(np.atleast_1d(denominator)), np.nan),
                    where=np.atleast_1d(denominator) > 0,
                )
                rows.append({
                    **dict(zip(keys, key)),
                    "covariate": column,
                    "metric": metric,
                    "gain": panel[metric].mean(),
                    "n_subjects": panel.subject_id.nunique(),
                    "n_patient_cells": len(panel),
                    "aggregation": "patient_weighted_across_cells",
                    **_interval(samples),
                })
    return pd.DataFrame(rows)




def _gains_fixture():
    rng = np.random.default_rng(13)
    rows, meta = [], []
    for cohort in ("CLL", "MM"):
        ids = [f"{cohort}{i}" for i in range(60)]
        meta += [{"subject_id": s, "age_at_index": float(rng.integers(30, 90)),
                  "sex": rng.choice(["F", "M"])} for s in ids]
        for outcome in ("os", "inf"):
            for horizon in (365, 730):
                for contrast in ("P5", "T2"):
                    for s in ids[: int(rng.integers(30, 60))]:
                        rows.append({"split": "tuning", "seed": 42, "cohort": cohort,
                                     "outcome": outcome, "horizon_days": horizon,
                                     "contrast": contrast, "subject_id": s,
                                     "brier_gain": rng.normal(0, 0.01),
                                     "log_loss_gain": rng.normal(0, 0.05)})
    gains, metadata = pd.DataFrame(rows), pd.DataFrame(meta)
    draws = {}
    for cohort in ("CLL", "MM"):
        ids = [f"{cohort}{i}" for i in range(60)]
        draws[("tuning", 42, cohort)] = (dict(zip(ids, range(60))),
                                         rng.multinomial(60, np.full(60, 1 / 60), size=200))
    specs = {"age_at_index": {"provenance": "pre_index", "bins": [0, 50, 65, 80, np.inf]},
             "sex": {"provenance": "pre_index"}}
    return gains, metadata, specs, draws


def test_vectorised_subgroup_gains_equal_the_loops():
    from opera.evaluation.secondary_analysis import panel_subgroup_gains, subgroup_patient_gains

    gains, metadata, specs, draws = _gains_fixture()
    for fast_fn, slow_fn in ((subgroup_patient_gains, _ref_subgroup_patient_gains),
                             (panel_subgroup_gains, _ref_panel_subgroup_gains)):
        fast = fast_fn(gains, metadata, specs, draws=draws)
        if "horizon_days" in fast and fast_fn is panel_subgroup_gains:
            fast = fast.loc[fast.horizon_days != 0]  # pooled rows are new; see below
        slow = slow_fn(gains, metadata, specs, draws=draws)
        pd.testing.assert_frame_equal(fast.reset_index(drop=True),
                                      slow[fast.columns].reset_index(drop=True),
                                      check_dtype=False, rtol=1e-10)


def _reference_coverage(frame, roster):
    required = {"model", "seed", "cohort", "outcome", "horizon_days", "status"}
    if "split" in roster:
        required.add("split")
    rows = []
    for record in roster.to_dict("records"):
        subset = frame.loc[frame.split == record.get("split", "held_out")]
        for column in required - {"status"}:
            subset = subset.loc[subset[column] == record[column]]
        cases = int(subset.label.sum())
        rows.append({**record, "n_known_predictions": len(subset), "n_cases": cases,
                     "n_controls": len(subset) - cases,
                     "prediction_status": "absent" if subset.empty else
                     "single_class" if cases in (0, len(subset)) else "two_class",
                     "auc_estimable": cases > 0 and cases < len(subset)})
    return pd.DataFrame(rows)


def test_vectorised_coverage_equals_the_loop():
    from opera.evaluation.secondary_analysis import coverage_inventory

    rng = np.random.default_rng(2)
    frame = pd.DataFrame({"model": rng.choice(["J_L", "X"], 3000), "seed": 42,
                          "split": rng.choice(["tuning", "held_out"], 3000),
                          "cohort": rng.choice(["CLL", "MM"], 3000),
                          "outcome": rng.choice(["os", "inf"], 3000),
                          "horizon_days": rng.choice([365.0, 730.0], 3000),
                          "label": rng.integers(0, 2, 3000)})
    frame.loc[(frame.cohort == "MM") & (frame.outcome == "inf"), "label"] = 0  # single class
    roster = pd.DataFrame([{"model": m, "seed": 42, "split": sp, "cohort": c, "outcome": o,
                            "horizon_days": h, "status": "prediction_available"}
                           for m in ("J_L", "X", "S_L") for sp in ("tuning", "held_out")
                           for c in ("CLL", "MM") for o in ("os", "inf") for h in (365, 730)])
    fast = coverage_inventory(frame, roster)
    slow = _reference_coverage(frame, roster)
    pd.testing.assert_frame_equal(fast, slow, check_dtype=False)
    no_split = roster.drop(columns="split").drop_duplicates()
    pd.testing.assert_frame_equal(coverage_inventory(frame, no_split).reset_index(drop=True),
                                  _reference_coverage(frame, no_split).reset_index(drop=True),
                                  check_dtype=False)
