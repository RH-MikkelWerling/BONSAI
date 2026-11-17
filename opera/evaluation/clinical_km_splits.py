"""Survival curves that show what a continuous model adds to a categorical clinical index.

Equal-size groups inherit an index's wide categories and hide most of a model's ranking. Two
displays show it in the clinicians' own terms (Kaplan--Meier curves):

  within   each index category split at the model's median risk within that category: does the
           model separate patients the index treats alike?
  tails    the model's own risk groups cut at fixed shares of patients (lowest 25%, 25-75%,
           75-90%, top 10%), next to the index's categories.

Only aggregates leave the server: survival on a fixed time grid with Greenwood (log-log) 95%
intervals and the number at risk, plus one summary row per group at the horizon.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

TAIL_GROUPS = (("Lowest 25%", 0.00, 0.25), ("25-75%", 0.25, 0.75), ("75-90%", 0.75, 0.90),
               ("Top 10%", 0.90, 1.00))


def km_curve(time, event, grid):
    """Kaplan--Meier survival, 95% log-log interval and number at risk at each grid time."""
    time, event = np.asarray(time, float), np.asarray(event, int)
    if not np.isin(event, [0, 1]).all():
        raise ValueError("KM curves need event 0/1 (death-inclusive endpoints)")
    order = np.unique(time[event == 1])
    surv, var, s, v = [], [], 1.0, 0.0
    for t in order:
        n = np.sum(time >= t)
        d = np.sum((time == t) & (event == 1))
        s *= 1 - d / n
        v += d / (n * (n - d)) if n > d else 0.0
        surv.append(s)
        var.append(v)
    out = []
    for g in grid:
        k = np.searchsorted(order, g, side="right") - 1
        st, vt = (1.0, 0.0) if k < 0 else (surv[k], var[k])
        if 0 < st < 1 and vt > 0:
            se = np.sqrt(vt) / abs(np.log(st))
            lo, hi = st ** np.exp(1.96 * se), st ** np.exp(-1.96 * se)
        else:
            lo = hi = st
        out.append((float(g), float(st), float(lo), float(hi), int(np.sum(time >= g))))
    return out


def _rows(score_id, display, group, category, data, grid, horizon):
    curve = km_curve(data.time_days, data.event, grid)
    rows = [dict(score_id=score_id, display=display, group=group, category=category, time_days=t,
                 survival=s, lower=lo, upper=hi, at_risk=n) for t, s, lo, hi, n in curve]
    at = km_curve(data.time_days, data.event, [horizon])[0]
    summary = dict(score_id=score_id, display=display, group=group, category=category,
                   horizon_days=horizon, n=int(len(data)), events=int(data.event[data.time_days <= horizon].sum()),
                   survival=at[1], lower=at[2], upper=at[3], at_risk_horizon=at[4])
    return rows, summary


def km_splits(patients, spec, *, step_days=7):
    """Curves and summaries for one index population (held-out ``score_patients`` output)."""
    horizon = int(spec["horizon_days"])
    grid = np.arange(0, int(horizon * 1.25) + 1, step_days)
    p = patients.assign(event=patients.event.astype(int))
    curves, summary = [], []
    for category in spec["group_order"]:
        part = p[p.ipi_group == category]
        if part.empty:
            continue
        r, s = _rows(spec["id"], "index", category, category, part, grid, horizon)
        curves += r
        summary.append(s)
        # Split the category at the model's median risk within it (ties by rank).
        rank = part.model_risk.rank(method="first")
        lower = rank <= len(part) / 2
        for name, mask in (("OPERA lower half", lower), ("OPERA upper half", ~lower)):
            if mask.sum():
                r, s = _rows(spec["id"], "within", name, category, part[mask], grid, horizon)
                curves += r
                summary.append(s)
    share = p.model_risk.rank(method="first", pct=True)
    for name, lo, hi in TAIL_GROUPS:
        mask = (share > lo) & (share <= hi)
        if mask.sum():
            r, s = _rows(spec["id"], "tails", name, "", p[mask], grid, horizon)
            curves += r
            summary.append(s)
    return curves, summary
