"""Median follow-up by the reverse Kaplan-Meier method.

Follow-up time is the time from the index date to death or censoring. In the
reverse method the roles are swapped: censoring is the event and deaths are
censored observations, so a death does not shorten the follow-up the cohort
would otherwise have had. ``event`` follows the outcome tables (0 censored,
1 the event, 2 a competing death); every non-zero value ends observation
without being censoring.
"""

from __future__ import annotations

import numpy as np


def reverse_km_quantiles(time_days, event, quantiles=(0.25, 0.5, 0.75)):
    """Follow-up quantiles in days; NaN where the curve never reaches them."""
    time = np.asarray(time_days, float)
    censored = np.asarray(event) == 0
    keep = np.isfinite(time)
    time, censored = time[keep], censored[keep]
    if not len(time):
        return {q: np.nan for q in quantiles}
    order = np.argsort(time, kind="stable")
    time, censored = time[order], censored[order]
    unique, first = np.unique(time, return_index=True)
    at_risk = len(time) - first
    events = np.add.reduceat(censored.astype(int), first)
    survival = np.cumprod(1 - events / at_risk)
    # Quantile q of follow-up: the first time the probability of still being
    # under follow-up falls to 1 - q or below.
    return {q: (float(unique[np.argmax(survival <= 1 - q)])
                if (survival <= 1 - q).any() else np.nan)
            for q in quantiles}


def median_followup(time_days, event):
    """Median and interquartile range of follow-up, in days."""
    values = reverse_km_quantiles(time_days, event)
    return {"median_days": values[0.5], "q25_days": values[0.25], "q75_days": values[0.75],
            "n": int(np.isfinite(np.asarray(time_days, float)).sum())}
