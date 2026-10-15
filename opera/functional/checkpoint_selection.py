"""Full-split horizon BCE for comparable supervised checkpoint selection."""

import numpy as np


def horizon_bce_values(times, events, probabilities, horizons):
    """Return horizon means over known status, including competing negatives.

    Means are formed on the complete tuning split, not averaged over batches.
    A single-class horizon remains usable; an entirely unknown horizon does not.
    """
    times = np.asarray(times, dtype=float).reshape(-1)
    events = np.asarray(events).reshape(-1)
    probabilities = np.asarray(probabilities, dtype=float)
    horizons = np.asarray(horizons, dtype=float).reshape(-1)
    if events.shape != times.shape or probabilities.shape != (
        len(times),
        len(horizons),
    ):
        raise ValueError("Selection predictions must align with patients and horizons")
    if (
        not np.isfinite(probabilities).all()
        or ((probabilities < 0) | (probabilities > 1)).any()
    ):
        raise ValueError(
            "Selection probabilities must be finite and between zero and one"
        )
    valid = np.isfinite(times) & (times >= 0) & np.isin(events, [0, 1, 2])
    positive = (events[:, None] == 1) & (times[:, None] <= horizons)
    known = valid[:, None] & (
        positive
        | (times[:, None] >= horizons)
        | ((events[:, None] == 2) & (times[:, None] <= horizons))
    )
    p = probabilities.clip(1e-6, 1 - 1e-6)
    losses = -(positive * np.log(p) + (~positive) * np.log1p(-p))
    return [
        float(losses[known[:, index], index].mean())
        for index in range(len(horizons))
        if known[:, index].any()
    ]
