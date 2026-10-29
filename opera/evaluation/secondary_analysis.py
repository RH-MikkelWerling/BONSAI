"""Paired analyses of locked predictions; no model training or selection on test.

Two estimands are supported and never mixed. ``binary_known_status_unweighted``
scores only patients whose horizon status is known. ``cif_ipcw`` scores the full
risk set for cumulative incidence at each horizon with inverse probability of
censoring weights, reproducing the survival evaluator's CIF AUC and Brier score
(competing events before the horizon are controls; administrative censoring
before the horizon carries weight zero but stays in the Brier denominator).
"""

from __future__ import annotations

import hashlib
import re
from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

CELL = ["split", "seed", "cohort", "outcome", "horizon_days"]
PATIENT = [*CELL, "subject_id"]
# ipa: index of prediction accuracy, 1 - Brier / Brier of the observed event
# rate in the same (weighted) sample; comparable across outcomes.
METRICS = ("auc", "brier", "ipa", "log_loss", "calibration_bias", "ece")
ESTIMANDS = ("binary_known_status_unweighted", "cif_ipcw")
# horizon_days of summaries pooled over every horizon.
ALL_HORIZONS = 0


def _censoring_survival_before(times, events):
    """Kaplan-Meier G(t-) of administrative censoring, floored at 1e-6.

    Same estimator as ``fast_cif_auc``/``compute_competing_risk_metrics_at_horizon``:
    censoring is event code 0; events 1 and 2 are not censoring.
    """
    order = np.argsort(times, kind="mergesort")
    unique, starts = np.unique(times[order], return_index=True)
    at_risk = len(times) - starts
    censored = np.add.reduceat((events[order] == 0).astype(int), starts)
    after = np.cumprod(1.0 - censored / at_risk)

    def before(query):
        index = np.searchsorted(unique, query, side="left") - 1
        result = np.ones(len(query), dtype=float)
        valid = index >= 0
        result[valid] = after[index[valid]]
        return np.maximum(result, 1e-6)

    return before


def attach_ipcw_weights(frame):
    """Full-risk-set CIF labels and IPCW weights per model/seed/disease/outcome/horizon.

    Per horizon as well: tabular tables are fitted per horizon, with follow-up
    cut at that horizon. Cutting at h leaves the Kaplan-Meier censoring curve
    before h unchanged (the extra censorings enter at h itself), so the weights
    equal those from full follow-up, for neural and tabular rows alike.

    Every group's Kaplan-Meier curve of censoring is computed at once (sorting
    and cumulative products), not group by group; the result is identical to
    ``_censoring_survival_before`` applied per group.
    """
    if not {"time_days", "event"} <= set(frame) or frame[["time_days", "event"]].isna().any().any():
        raise ValueError("cif_ipcw requires follow-up time_days and event for every model and row")
    keys = ["model", "split", "seed", "cohort", "outcome", "horizon_days"]
    group = frame.groupby(keys, sort=False).ngroup().to_numpy()
    frame = frame.loc[group >= 0].copy()  # rows with a missing key, as groupby drops them
    group = group[group >= 0]
    if frame.assign(_group=group).duplicated(["_group", "subject_id"]).any():
        raise ValueError("A patient appears twice in one model, cell and horizon")
    t = frame.time_days.to_numpy(float)
    e = frame.event.to_numpy(int)
    h = frame.horizon_days.to_numpy(float)

    # Censoring curve per group, on the group's distinct times.
    order = np.lexsort((t, group))
    g_sorted, t_sorted = group[order], t[order]
    censored_sorted = (e[order] == 0).astype(float)
    distinct = np.r_[True, (g_sorted[1:] != g_sorted[:-1]) | (t_sorted[1:] != t_sorted[:-1])]
    starts = np.flatnonzero(distinct)
    unique_group, unique_time = g_sorted[starts], t_sorted[starts]
    censored = np.add.reduceat(censored_sorted, starts)
    group_size = np.bincount(group)
    first_row = np.r_[0, np.cumsum(group_size)[:-1]]  # first sorted row of each group
    at_risk = group_size[unique_group] - (starts - first_row[unique_group])
    step = 1.0 - censored / at_risk
    zero = step <= 0  # everyone still at risk censored: the curve stays at 0
    by_group = pd.Series(unique_group)
    log_cumulative = pd.Series(np.log(np.where(zero, 1.0, step))).groupby(by_group).cumsum()
    reached_zero = pd.Series(zero.astype(int)).groupby(by_group).cumsum().to_numpy() > 0
    after = np.where(reached_zero, 0.0, np.exp(log_cumulative.to_numpy()))  # G just after each time

    # G(q-): the curve at the group's last distinct time strictly before q.
    scale = 10.0 ** np.ceil(np.log10(max(np.abs(t).max(), np.abs(h).max()) + 2))
    key = unique_group * scale + unique_time

    def before(rows, query):
        index = np.searchsorted(key, group[rows] * scale + query, side="left") - 1
        valid = (index >= 0) & (unique_group[np.maximum(index, 0)] == group[rows])
        value = np.where(valid, after[np.maximum(index, 0)], 1.0)
        return np.maximum(value, 1e-6)

    case = (t <= h) & (e == 1)
    competing = (t <= h) & (e == 2)
    control = competing | (t > h) | ((t >= h) & (e == 0))
    weight = np.zeros(len(frame))
    event_rows = np.flatnonzero(case | competing)
    weight[event_rows] = 1.0 / before(event_rows, t[event_rows])
    horizon_rows = np.flatnonzero(control & ~competing)
    weight[horizon_rows] = 1.0 / before(horizon_rows, h[horizon_rows])
    frame["label"] = case.astype(int)
    frame["ipcw_weight"] = weight
    frame["known"] = case | control
    if "prediction_absent_zero_weight" in frame:
        placeholder = frame.prediction_absent_zero_weight.eq(True)  # NaN counts as False
        if (placeholder & frame.known).any():
            raise ValueError("A status-known patient lacks a real prediction")
    return frame.sort_index()


def normalize_predictions(
    frame: pd.DataFrame, estimand: str = "binary_known_status_unweighted", *,
    allow_training: bool = False,
) -> tuple[pd.DataFrame, dict]:
    """Validate before casting, then apply the requested estimand.

    Known-status analysis retains only rows whose horizon status is known.
    ``cif_ipcw`` retains the full risk set with IPCW weights (zero for rows
    censored before the horizon).
    Training rows are accepted only by an explicit clinical threshold-development
    caller. Ordinary evaluation keeps its tuning/held_out restriction.
    """
    if estimand not in ESTIMANDS:
        raise ValueError(f"estimand must be one of {ESTIMANDS}")
    frame = frame.copy()
    if estimand == "cif_ipcw" and "label" not in frame and {"time_days", "event"} <= set(frame):
        # Full-risk-set rows: status follows from follow-up, not a stored label.
        horizon = pd.to_numeric(frame.horizon_days, errors="raise")
        positive = (frame.event == 1) & (frame.time_days <= horizon)
        frame["label"] = positive.astype(int)
        frame["known"] = (
            positive
            | (frame.time_days >= horizon)
            | ((frame.event == 2) & (frame.time_days <= horizon))
        )
    required = {*PATIENT, "model", "label", "probability", "data_contract"}
    missing = required - set(frame)
    if missing:
        raise ValueError(f"Missing canonical prediction fields: {sorted(missing)}")
    if frame.empty:
        raise ValueError("Empty predictions")
    gaps = frame[sorted(required - {"label"})].isna().sum()
    if gaps.any():
        raise ValueError(
            "Missing prediction identity/probability "
            f"(rows with gaps per column, of {len(frame)}): {gaps[gaps > 0].to_dict()}"
        )
    for key in ("model", "subject_id", "cohort", "outcome", "data_contract"):
        frame[key] = frame[key].astype(str)
        if frame[key].str.strip().eq("").any():
            raise ValueError(f"Empty identity field: {key}")
    for key in ("horizon_days", "seed"):
        values = pd.to_numeric(frame[key], errors="raise")
        if not np.isfinite(values).all() or (values < 0).any():
            raise ValueError(f"Invalid {key}")
        if key == "seed" and (values != np.floor(values)).any():
            raise ValueError("Seeds must be integers")
        frame[key] = values
    allowed_splits = ["tuning", "held_out", *(["train"] if allow_training else [])]
    if (frame.horizon_days <= 0).any() or not frame.split.isin(allowed_splits).all():
        raise ValueError(f"Need positive horizons and {'/'.join(allowed_splits)} splits")
    if frame.duplicated(["model", *PATIENT]).any():
        raise ValueError("Duplicate model/patient/task predictions")
    if (frame.groupby("subject_id").split.nunique() > 1).any():
        raise ValueError("Patient overlap between prediction splits")
    if (frame.groupby("subject_id").cohort.nunique() > 1).any():
        raise ValueError("A patient must have one fine disease assignment")
    frame["probability"] = pd.to_numeric(frame.probability, errors="raise").astype(
        float
    )
    if (
        not np.isfinite(frame.probability).all()
        or not frame.probability.between(0, 1).all()
    ):
        raise ValueError("Probabilities must be finite and within [0,1]")
    if "known" not in frame:
        frame["known"] = True
    if not frame.known.isin([True, False, 0, 1]).all():
        raise ValueError("known must be explicit boolean/0/1, not truthy strings")
    frame["known"] = frame.known.astype(bool)
    if not frame.loc[frame.known, "label"].isin([0, 1]).all():
        raise ValueError("Known labels must be exactly binary before casting")
    if ("time_days" in frame) != ("event" in frame):
        raise ValueError("Follow-up time and event columns must be provided together")
    if {"time_days", "event"} <= set(frame):
        observed = frame.time_days.notna() & frame.event.notna()
        if not frame.time_days.notna().equals(frame.event.notna()):
            raise ValueError("Follow-up time and event must be provided together")
        records = frame.loc[observed]
        if (
            not np.isfinite(records.time_days).all()
            or (records.time_days < 0).any()
            or not records.event.isin([0, 1, 2]).all()
        ):
            raise ValueError("Invalid observed follow-up/event codes")
        positive = (records.event == 1) & (records.time_days <= records.horizon_days)
        known = (
            positive
            | (records.time_days >= records.horizon_days)
            | ((records.event == 2) & (records.time_days <= records.horizon_days))
        )
        if not known.equals(records.known) or not np.array_equal(
            positive[known].astype(int), records.loc[known, "label"].astype(int)
        ):
            raise ValueError("Label/known status disagrees with event and follow-up")
    if estimand == "cif_ipcw":
        frame = attach_ipcw_weights(frame)
        audit = {
            "input_rows": len(frame),
            "known_rows": int(frame.known.sum()),
            "zero_weight_censored_rows": int((~frame.known).sum()),
            "estimand": estimand,
        }
        return frame.sort_values(["model", *PATIENT]).reset_index(drop=True), audit
    audit = {
        "input_rows": len(frame),
        "known_rows": int(frame.known.sum()),
        "excluded_unknown_rows": int((~frame.known).sum()),
        "estimand": estimand,
    }
    return frame.loc[frame.known].sort_values(["model", *PATIENT]).reset_index(
        drop=True
    ), audit


def contrast_coefficients(expression: str) -> dict[str, int]:
    """Parse only sums/differences of gain(A,B), never execute input text."""
    compact = re.sub(r"\s+", "", expression)
    matches = list(
        re.finditer(r"([+-]?)gain\(([A-Za-z0-9_]+),([A-Za-z0-9_]+)\)", compact)
    )
    if not matches or "".join(m.group() for m in matches) != compact:
        raise ValueError(f"Invalid contrast expression: {expression}")
    coefficients = defaultdict(int)
    for index, match in enumerate(matches):
        sign, candidate, reference = match.groups()
        if index and not sign:
            raise ValueError("Contrast terms require a sign")
        value = -1 if sign == "-" else 1
        coefficients[candidate] += value
        coefficients[reference] -= value
    result = {name: value for name, value in coefficients.items() if value}
    if not result:
        raise ValueError("Contrast expression cancels identically")
    return result


def align_models(frames: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """Exact identity/label/contract parity; no intersecting patient sets."""
    reference = None
    result = {}
    for name, frame in frames.items():
        frame = frame.sort_values("subject_id").reset_index(drop=True)
        if frame.subject_id.duplicated().any():
            raise ValueError("Duplicate patients within a comparison cell")
        if reference is not None:
            parity = ["subject_id", "label", "data_contract"]
            if "ipcw_weight" in frame and "ipcw_weight" in reference:
                parity.append("ipcw_weight")
            for column in parity:
                if not np.array_equal(
                    frame[column].to_numpy(), reference[column].to_numpy()
                ):
                    raise ValueError(
                        f"Patient/label/data-contract parity failed: {name}/{column}"
                    )
        reference = frame if reference is None else reference
        result[name] = frame
    # Follow-up must agree as seen at the cell's horizon: tabular tables are
    # fitted per horizon with follow-up cut there, neural tables keep it whole.
    seen = [
        _follow_up_at_horizon(group)
        for group in result.values()
        if {"time_days", "event"} <= set(group)
        and group[["time_days", "event"]].notna().all().all()
    ]
    for index, column in enumerate(("time_days", "event")):
        if seen and any(not np.array_equal(other[index], seen[0][index]) for other in seen[1:]):
            raise ValueError(f"Follow-up parity failed: {column}")
    return result


def _follow_up_at_horizon(frame):
    """(time, event) censored at the horizon: time cut at h, events after h as 0."""
    time = frame.time_days.to_numpy(float)
    event = frame.event.to_numpy(int)
    if "horizon_days" not in frame:
        return time, event
    horizon = frame.horizon_days.to_numpy(float)
    return np.minimum(time, horizon), np.where(time > horizon, 0, event)


def metric_values(labels, probabilities, weights=None, ipcw=None):
    """Point metrics; ``ipcw`` switches losses to full-risk-set IPCW estimates.

    With IPCW, AUC is the weighted case/control AUC and Brier/log loss are
    ``sum(ipcw * loss) / N`` over all N risk-set rows, as in the evaluator.
    Calibration bias and ECE are IPCW-weighted means over status-known rows.
    """
    labels, probabilities = np.asarray(labels, float), np.asarray(probabilities, float)
    count = np.ones(len(labels)) if weights is None else np.asarray(weights, float)
    if count.sum() <= 0:
        return dict.fromkeys(METRICS, float("nan"))
    p = probabilities.clip(1e-6, 1 - 1e-6)
    weights = count if ipcw is None else count * np.asarray(ipcw, float)
    if weights.sum() <= 0:
        return dict.fromkeys(METRICS, float("nan"))

    def average(values):
        return float(np.average(values, weights=weights))

    def risk_set_average(values):
        if ipcw is None:
            return average(values)
        return float((weights * values).sum() / count.sum())

    auc = float("nan")
    if weights[labels == 1].sum() > 0 and weights[labels == 0].sum() > 0:
        auc = float(roc_auc_score(labels, probabilities, sample_weight=weights))
    brier = risk_set_average((labels - probabilities) ** 2)
    event_rate = risk_set_average(labels.astype(float))
    null_brier = risk_set_average((labels - event_rate) ** 2)
    return {
        "auc": auc,
        "brier": brier,
        "ipa": 1 - brier / null_brier if null_brier > 0 else float("nan"),
        "log_loss": risk_set_average(-(labels * np.log(p) + (1 - labels) * np.log1p(-p))),
        "calibration_bias": average(probabilities - labels),
        "ece": float(
            sum(
                abs(
                    np.dot(
                        weights[(probabilities * 10).astype(int).clip(0, 9) == index],
                        (probabilities - labels)[
                            (probabilities * 10).astype(int).clip(0, 9) == index
                        ],
                    )
                )
                for index in range(10)
            )
            / weights.sum()
        ),
    }


def bootstrap_metric_values(labels, probabilities, weights, ipcw=None):
    """Vectorized weighted ROC/loss draws in memory-bounded chunks.

    ``weights`` are bootstrap multiplicities; optional ``ipcw`` row weights
    give the full-risk-set CIF estimand (see ``metric_values``).
    """
    y, p = np.asarray(labels, float), np.asarray(probabilities, float)
    row_weight = np.ones(len(y)) if ipcw is None else np.asarray(ipcw, float)
    clipped = p.clip(1e-6, 1 - 1e-6)
    order = np.argsort(p, kind="stable")
    starts = np.r_[0, np.flatnonzero(np.diff(p[order])) + 1]
    bins = (p * 10).astype(int).clip(0, 9)
    result = {metric: [] for metric in METRICS}
    for offset in range(0, len(weights), 128):
        counts = weights[offset : offset + 128].astype(float)
        w = counts * row_weight
        denominator = w.sum(axis=1)
        risk_set = counts.sum(axis=1) if ipcw is not None else denominator
        positive = np.add.reduceat(w[:, order] * y[order], starts, axis=1)
        negative = np.add.reduceat(w[:, order] * (1 - y[order]), starts, axis=1)
        pair_denominator = positive.sum(axis=1) * negative.sum(axis=1)
        concordance = (positive * (negative.cumsum(axis=1) - 0.5 * negative)).sum(
            axis=1
        )
        scores = {
            "auc": np.divide(
                concordance,
                pair_denominator,
                out=np.full(len(w), np.nan),
                where=pair_denominator > 0,
            )
        }
        for metric, values, normaliser in (
            ("brier", (y - p) ** 2, risk_set),
            ("log_loss", -(y * np.log(clipped) + (1 - y) * np.log1p(-clipped)), risk_set),
            ("calibration_bias", p - y, denominator),
        ):
            scores[metric] = np.divide(
                w @ values,
                normaliser,
                out=np.full(len(w), np.nan),
                where=denominator > 0,
            )
        # Brier of the event rate in each draw: (S - 2 r S + r^2 W) / R with
        # S = weighted events, W = total weight, r = S / R.
        events, total = w @ y, w.sum(axis=1)
        rate = np.divide(events, risk_set, out=np.full(len(w), np.nan), where=risk_set > 0)
        null_brier = np.divide(events - 2 * rate * events + rate**2 * total, risk_set,
                               out=np.full(len(w), np.nan), where=risk_set > 0)
        scores["ipa"] = 1 - np.divide(scores["brier"], null_brier,
                                      out=np.full(len(w), np.nan), where=null_brier > 0)
        error = sum(
            np.abs(w[:, bins == index] @ (p - y)[bins == index]) for index in range(10)
        )
        scores["ece"] = np.divide(
            error, denominator, out=np.full(len(w), np.nan), where=denominator > 0
        )
        for metric in METRICS:
            result[metric].extend(scores[metric])
    return {metric: np.asarray(values) for metric, values in result.items()}


def patient_bootstrap(frame: pd.DataFrame, replicates: int, seed: int, *, method="multinomial"):
    """One subject-count draw reused across outcomes, horizons and methods."""
    if replicates < 0:
        raise ValueError("Bootstrap replicates cannot be negative")
    if method not in {'multinomial', 'multiplier', 'bayesian'}:
        raise ValueError('Unknown patient bootstrap method')
    draws = {}
    for key, group in frame.groupby(["split", "cohort"], sort=True):
        ids = sorted(group.subject_id.unique())
        digest = hashlib.sha256(repr((seed, key)).encode()).digest()
        rng = np.random.default_rng(int.from_bytes(digest[:8], "little"))
        if method in {'multiplier', 'bayesian'}:
            # Positive exchangeable patient weights; shared across tasks/models.
            # Multiplier inference is frequentist, not a posterior interpretation.
            counts = rng.exponential(size=(replicates, len(ids)))
            counts = np.maximum(counts, np.finfo(float).tiny)
            counts *= len(ids) / counts.sum(axis=1, keepdims=True)
        else:
            counts = rng.multinomial(len(ids), np.full(len(ids), 1 / len(ids)), size=replicates)
        shared = (dict(zip(ids, range(len(ids)))), counts)
        for training_seed in group.seed.unique():
            draws[(key[0], training_seed, key[1])] = shared
    return draws


def seed_averaged_effects(contrasts, bootstrap_draws):
    """Patient uncertainty for the mean of the fitted seed roster; not seed CI.

    Vectorised: each contrast row is joined once to its own draws; cells fitted
    with one seed need no averaging, and only cells with several seeds are
    averaged one by one. Identical to averaging each group's draws in a loop.
    """
    keys = ["split", "cohort", "outcome", "horizon_days", "contrast", "metric"]
    columns = [*keys, "gain", "n_training_seeds", "seed_minimum", "seed_maximum",
               "seed_sd", "uncertainty", "lower", "upper", "difference_se",
               "n_valid_bootstrap", "interval_status"]
    if contrasts.empty:
        return pd.DataFrame(columns=columns)
    merged = contrasts[[*keys, "seed", "gain"]].merge(
        bootstrap_draws[[*keys, "seed", "gain_draws"]], on=[*keys, "seed"],
        how="left", validate="one_to_one")
    if merged.gain_draws.isna().any():
        raise ValueError("A contrast has no bootstrap draws")
    sizes = merged.groupby(keys).seed.transform("size").to_numpy()
    single = merged.loc[sizes == 1]
    rows = pd.DataFrame({
        **{key: single[key].to_numpy() for key in keys},
        "gain": single.gain.to_numpy(float),
        "n_training_seeds": 1,
        "seed_minimum": single.gain.to_numpy(float),
        "seed_maximum": single.gain.to_numpy(float),
        "seed_sd": np.nan,
    })
    averaged = list(single.gain_draws)
    parts = [rows]
    multi_rows, multi_draws = [], []
    for key, group in merged.loc[sizes > 1].groupby(keys):
        arrays = np.stack(group.gain_draws)
        finite = np.isfinite(arrays).all(axis=0)
        average = arrays.mean(axis=0)
        average[~finite] = np.nan
        multi_rows.append({
            **dict(zip(keys, key)),
            "gain": group.gain.mean() if np.isfinite(group.gain).all() else np.nan,
            "n_training_seeds": len(group),
            "seed_minimum": group.gain.min(),
            "seed_maximum": group.gain.max(),
            "seed_sd": group.gain.std(),
        })
        multi_draws.append(average)
    if multi_rows:
        parts.append(pd.DataFrame(multi_rows))
        averaged += multi_draws
    result = pd.concat(parts, ignore_index=True)
    result["uncertainty"] = "patients_conditional_on_fitted_seed_roster"
    result = pd.concat([result, _intervals(averaged)], axis=1)
    return result.sort_values(keys, kind="stable").reset_index(drop=True)[columns]


def _intervals(draws, chunk=20000):
    """``_interval`` for many draw vectors at once (rows of equal length)."""
    out = []
    for start in range(0, len(draws), chunk):
        block = np.stack(draws[start : start + chunk]).astype(float)
        finite = np.isfinite(block)
        n_valid = finite.sum(axis=1)
        enough = n_valid >= max(20, int(np.ceil(0.8 * block.shape[1])))
        with np.errstate(all="ignore"):
            import warnings

            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                lower, upper = np.nanquantile(block, [0.025, 0.975], axis=1)
                se = np.nanstd(block, axis=1, ddof=1)
        out.append(pd.DataFrame({
            "lower": np.where(enough, lower, np.nan),
            "upper": np.where(enough, upper, np.nan),
            "difference_se": np.where(enough, se, np.nan),
            "n_valid_bootstrap": n_valid,
            "interval_status": np.where(enough, "conditional_on_fitted_models",
                                        "insufficient_draws"),
        }))
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame(
        columns=["lower", "upper", "difference_se", "n_valid_bootstrap", "interval_status"])


def _interval(values):
    values = np.asarray(values, float)
    finite = values[np.isfinite(values)]
    minimum = max(20, int(np.ceil(0.8 * len(values))))
    if len(finite) < minimum:
        return {
            "lower": np.nan,
            "upper": np.nan,
            "difference_se": np.nan,
            "n_valid_bootstrap": len(finite),
            "interval_status": "insufficient_draws",
        }
    lo, hi = np.quantile(finite, [0.025, 0.975])
    return {
        "lower": lo,
        "upper": hi,
        "difference_se": finite.std(ddof=1),
        "n_valid_bootstrap": len(finite),
        "interval_status": "conditional_on_fitted_models",
    }


def pool_cells_over_horizons(contrasts, bootstrap_draws):
    """Per disease-outcome cell, the gain averaged over its evaluated horizons
    (horizon_days == ALL_HORIZONS), each horizon weighing equally.

    The patient resample of a disease is shared across outcomes and horizons,
    so averaging a cell's draws over horizons gives the pooled gain's
    bootstrap distribution; a replicate in which any horizon is not finite is
    not finite for the pooled gain either (as in effect_summaries)."""
    keys = ["split", "seed", "cohort", "outcome", "contrast", "metric"]
    values = contrasts.loc[np.isfinite(contrasts.gain), keys + ["horizon_days", "gain"]]
    if values.empty:
        return pd.DataFrame(columns=[*keys, "horizon_days", "gain", "n_horizons",
                                     "lower", "upper"])
    draws = values[keys + ["horizon_days"]].merge(
        bootstrap_draws, on=keys + ["horizon_days"], how="left",
        validate="one_to_one", indicator=True)
    if not draws['_merge'].eq('both').all():
        raise ValueError("Missing bootstrap cells for horizon pooling")
    draws = draws.drop(columns='_merge')
    parts = []
    # One contrast and metric at a time keeps the stacked draws small.
    for _, part in draws.groupby(["contrast", "metric"], sort=False):
        part = part.sort_values(keys, kind="stable")
        starts = np.flatnonzero(part[keys].ne(part[keys].shift()).any(axis=1).to_numpy())
        block = np.stack(part.gain_draws).astype(float)
        finite = np.isfinite(block)
        counts = np.diff(np.append(starts, len(part)))
        mean = np.add.reduceat(np.where(finite, block, 0.0), starts, axis=0) / counts[:, None]
        mean[np.add.reduceat(~finite, starts, axis=0) > 0] = np.nan
        identity = part.iloc[starts][keys].reset_index(drop=True)
        parts.append(pd.concat([identity.assign(n_horizons=counts),
                                _intervals(list(mean))], axis=1))
    pooled = pd.concat(parts, ignore_index=True)
    gains = values.groupby(keys).gain.mean().reset_index()
    pooled = pooled.merge(gains, on=keys, validate="one_to_one")
    pooled["horizon_days"] = ALL_HORIZONS
    return pooled


CELL_BATCH = 200  # cells per parallel task


def analysis_workers():
    """Processes for the paired analysis: OPERA_ANALYSIS_WORKERS (default 1)."""
    import os

    return max(1, int(os.environ.get("OPERA_ANALYSIS_WORKERS", "1")))


def _with_progress(results, total):
    import sys
    import time

    started = time.time()
    for done, result in enumerate(results, 1):
        if done % 5 == 0 or done == total:
            print(f"paired analysis: {done}/{total} cell batches, "
                  f"{(time.time() - started) / 60:.1f} min", file=sys.stderr, flush=True)
        yield result


def _paired_cells(batch, parsed, patient_gain_contrasts, min_cases=0, min_controls=0):
    """Estimates, contrasts and draws for a batch of cells (run in parallel)."""
    estimates, differences, draw_rows, contributions, coverage = [], [], [], [], []
    model_draw_rows = []
    for key, cell, draws in batch:
        identity = dict(zip(CELL, key))
        models = {name: group for name, group in cell.groupby("model")}
        # All available methods must agree, even if a requested contrast is absent.
        models = align_models(models)
        reference = next(iter(models.values()))
        y = reference.label.to_numpy(float)
        ipcw = (
            reference.ipcw_weight.to_numpy(float)
            if "ipcw_weight" in reference
            else None
        )
        status_known = np.ones(len(y), bool) if ipcw is None else ipcw > 0
        n_cases = int((y[status_known] == 1).sum())
        n_controls = int((y[status_known] == 0).sum())
        eligible = n_cases >= min_cases and n_controls >= min_controls
        values = {
            name: (metric_values(y, group.probability.to_numpy(float), ipcw=ipcw)
                   if eligible else dict.fromkeys(METRICS, np.nan))
            for name, group in models.items()
        }
        mapping, counts = draws[key[:3]]
        positions = [mapping[sid] for sid in reference.subject_id]
        weights = counts[:, positions]
        scores_by_draw = {
            name: (bootstrap_metric_values(
                y, group.probability.to_numpy(float), weights, ipcw=ipcw
            ) if eligible else {metric: np.full(len(weights), np.nan) for metric in METRICS})
            for name, group in models.items()
        }
        for name, scores in values.items():
            for metric, estimate in scores.items():
                sampled = np.asarray(scores_by_draw[name][metric], float)
                interval = _interval(sampled)
                estimates.append(
                    {
                        **identity,
                        "model": name,
                        "metric": metric,
                        "estimate": estimate,
                        "n": len(y),
                        "n_cases": n_cases,
                        "n_controls": n_controls,
                        "lower": interval["lower"],
                        "upper": interval["upper"],
                        "interval_status": interval["interval_status"],
                    }
                )
                model_draw_rows.append(
                    {**identity, "model": name, "metric": metric,
                     "estimate_draws": sampled}
                )
        for name, coefficients in parsed.items():
            missing = set(coefficients) - set(models)
            coverage.append(
                {
                    **identity,
                    "contrast": name,
                    "status": ("missing_models" if missing else "paired" if eligible
                               else "insufficient_cases_controls"),
                    "missing_models": ",".join(sorted(missing)),
                }
            )
            if missing:
                continue
            for metric in METRICS:
                # Bias is signed calibration error, not a higher-is-better skill.
                orientation = 1 if metric in {"auc", "ipa", "calibration_bias"} else -1
                estimate = orientation * sum(
                    coef * values[model][metric] for model, coef in coefficients.items()
                )
                sampled = orientation * sum(
                    coef * np.asarray(scores_by_draw[model][metric])
                    for model, coef in coefficients.items()
                )
                differences.append(
                    {
                        **identity,
                        "contrast": name,
                        "metric": metric,
                        "gain": estimate,
                        "n": len(y),
                        "n_cases": n_cases,
                        "n_controls": n_controls,
                        **_interval(sampled),
                    }
                )
                draw_rows.append(
                    {
                        **identity,
                        "contrast": name,
                        "metric": metric,
                        "gain_draws": sampled,
                    }
                )
            if not eligible:
                continue
            if (
                patient_gain_contrasts is not None
                and name not in patient_gain_contrasts
            ):
                continue
            loss_gains = {}
            for metric in ("brier", "log_loss"):
                losses = {}
                for model in coefficients:
                    p = models[model].probability.to_numpy(float)
                    if metric == "log_loss":
                        p = p.clip(1e-6, 1 - 1e-6)
                    losses[model] = (
                        (y - p) ** 2
                        if metric == "brier"
                        else (-(y * np.log(p) + (1 - y) * np.log1p(-p)))
                    )
                loss_gains[metric + "_gain"] = -sum(
                    coefficients[model] * losses[model] for model in coefficients
                ) * (1.0 if ipcw is None else ipcw)
            contributions.append(
                pd.DataFrame(
                    {
                        **identity,
                        "subject_id": reference.subject_id,
                        "contrast": name,
                        **loss_gains,
                    }
                )
            )
    return estimates, differences, draw_rows, contributions, coverage, model_draw_rows


def paired_analysis(
    frame,
    contrasts,
    *,
    replicates=1000,
    seed=2026,
    draws=None,
    patient_gain_contrasts=None,
    min_cases=0,
    min_controls=0,
):
    """Cell estimates, paired contrasts, joint draws and patient loss contributions."""
    if min_cases < 0 or min_controls < 0:
        raise ValueError('Performance support thresholds must be nonnegative')
    draws = patient_bootstrap(frame, replicates, seed) if draws is None else draws
    estimates, differences, draw_rows, contributions, coverage = [], [], [], [], []
    model_draw_rows = []
    parsed = {name: contrast_coefficients(expr) for name, expr in contrasts.items()}
    batches, batch = [], []
    for key, cell in frame.groupby(CELL, sort=True):
        batch.append((key, cell, {key[:3]: draws[key[:3]]}))
        if len(batch) == CELL_BATCH:
            batches.append(batch)
            batch = []
    if batch:
        batches.append(batch)
    workers = analysis_workers()
    if workers > 1 and len(batches) > 1:
        from concurrent.futures import ProcessPoolExecutor
        from collections import deque

        def ordered_results(pool):
            # At most two batches per worker in flight, results in submission order.
            pending = deque()
            for batch in batches:
                pending.append(pool.submit(_paired_cells, batch, parsed, patient_gain_contrasts, min_cases, min_controls))
                if len(pending) >= 2 * workers:
                    yield pending.popleft().result()
            while pending:
                yield pending.popleft().result()

        with ProcessPoolExecutor(max_workers=workers) as pool:
            parts = list(_with_progress(ordered_results(pool), len(batches)))
    else:
        parts = list(_with_progress(
            (_paired_cells(b, parsed, patient_gain_contrasts, min_cases, min_controls) for b in batches), len(batches)))
    for part in parts:  # in cell order, exactly as a serial run
        for target, rows in zip((estimates, differences, draw_rows, contributions, coverage,
                                 model_draw_rows), part):
            target.extend(rows)
    if not differences:
        raise ValueError("No requested contrast has all required model predictions")
    return {
        "estimates": pd.DataFrame(estimates),
        "contrasts": pd.DataFrame(differences),
        "bootstrap_draws": pd.DataFrame(draw_rows),
        "patient_gains": pd.concat(contributions, ignore_index=True)
        if contributions
        else pd.DataFrame(),
        "contrast_coverage": pd.DataFrame(coverage),
        "model_bootstrap_draws": pd.DataFrame(model_draw_rows),
    }


def panel_estimates(estimates, model_draws, groups, *, stable_cells_only=False,
                    pooled_horizons=False):
    """Equal-cell panel mean of absolute metrics for sets of compared models.

    Each group (for example the joint and routed models under the three encoder
    origins) is averaged over the cells in which every member was evaluated, so
    its models are compared on an identical panel. Intervals reuse the shared
    patient resamples of each cell.
    """
    keys = ["split", "seed", "horizon_days", "metric"]
    rows = []
    for group_name, members in groups.items():
        subset = estimates.loc[estimates.model.isin(members)
                               & np.isfinite(estimates.estimate)]
        if subset.empty:
            continue
        for identity, part in subset.groupby(keys):
            complete = (part.groupby(["cohort", "outcome"]).model.nunique()
                        == len(set(members)))
            cells = complete[complete].index
            if not len(cells):
                continue
            per_model = {}
            for model in members:
                selected = part.set_index(["cohort", "outcome"]).loc[cells]
                selected = selected.loc[selected.model == model].reset_index()
                draws = model_draws.loc[model_draws.model == model]
                for column, value in zip(keys, identity):
                    draws = draws.loc[draws[column] == value]
                draws = draws.merge(selected[["cohort", "outcome"]],
                                    on=["cohort", "outcome"])
                per_model[model] = (selected, draws)
            if stable_cells_only:
                # As in effect_summaries, on one panel shared by every member.
                stable = None
                for _, draws in per_model.values():
                    ok = draws.loc[np.isfinite(np.stack(draws.estimate_draws)).all(axis=1),
                                   ["cohort", "outcome"]]
                    stable = ok if stable is None else stable.merge(ok)
                if stable is None or stable.empty:
                    continue
                per_model = {m: (sel.merge(stable), dr.merge(stable))
                             for m, (sel, dr) in per_model.items()}
            for model in members:
                selected, draws = per_model[model]
                sampled = np.stack(draws.estimate_draws).mean(axis=0)
                sampled[~np.isfinite(np.stack(draws.estimate_draws)).all(axis=0)] = np.nan
                rows.append({
                    **dict(zip(keys, identity)),
                    "group": group_name,
                    "model": model,
                    "estimate": selected.estimate.mean(),
                    "n_cells": len(selected),
                    "aggregation": "equal_cell_common_panel",
                    **_interval(sampled),
                })
    table = pd.DataFrame(rows)
    if pooled_horizons and len(estimates):
        # Also every disease-outcome-horizon cell weighing equally
        # (horizon_days == ALL_HORIZONS), as in effect_summaries.
        def pooled(frame):
            return frame.assign(outcome=frame.outcome.astype(str) + "@"
                                + frame.horizon_days.astype(str), horizon_days=ALL_HORIZONS)
        table = pd.concat([table, panel_estimates(
            pooled(estimates), pooled(model_draws), groups,
            stable_cells_only=stable_cells_only)], ignore_index=True)
    return table


ROUTER_GATED_CONTRASTS = ("R2", "R3", "R4", "R5", "R6", "R7")


def router_null_equivalence(
    summaries, tolerance, *, contrast="R0", gated=ROUTER_GATED_CONTRASTS
):
    """Check that the no-source adapter N reproduces shared J within tolerance.

    Uses the all-cell panel summary of ``contrast`` with its patient-bootstrap
    interval. Equivalence requires the whole interval inside
    ``[-tolerance, +tolerance]`` on that metric's gain scale; an inconclusive
    interval is not evidence of equivalence. Returns one row per split, seed,
    metric and horizon, listing the bank contrasts whose interpretation the
    gate controls.
    """
    columns = ["split", "seed", "metric", "horizon_days"]
    panel = summaries.loc[
        (summaries.contrast == contrast) & (summaries.stratum == "all_cells")
    ] if not summaries.empty else summaries
    gated_present = sorted(set(summaries.contrast) & set(gated)) if not summaries.empty else []
    if panel.empty:
        if not gated_present:
            return pd.DataFrame(columns=[*columns, "status"])
        keys = summaries.loc[summaries.contrast.isin(gated_present), columns]
        return keys.drop_duplicates().assign(
            contrast=contrast, status="missing_null_router_contrast",
            gated_contrasts=",".join(gated_present),
        )
    rows = []
    for _, row in panel.iterrows():
        limit = (tolerance or {}).get(row.metric)
        record = {
            **{key: row[key] for key in columns},
            "contrast": contrast,
            "gain": row.gain,
            "lower": row.lower,
            "upper": row.upper,
            "tolerance": np.nan if limit is None else float(limit),
            "gated_contrasts": ",".join(gated_present),
        }
        if limit is None:
            record["status"] = "tolerance_not_locked"
        elif not (np.isfinite(row.lower) and np.isfinite(row.upper)):
            record["status"] = "insufficient_bootstrap_draws"
        elif -float(limit) <= row.lower and row.upper <= float(limit):
            record["status"] = "equivalent_within_tolerance"
        else:
            record["status"] = "equivalence_not_shown"
        rows.append(record)
    return pd.DataFrame(rows)


def annotate_router_gate(table, gate, *, gated=ROUTER_GATED_CONTRASTS):
    """Attach the N-vs-J gate status to bank-effect rows; others get NaN."""
    if table.empty:
        return table
    columns = ["split", "seed", "metric", "horizon_days"]
    status = (
        gate[[*columns, "status"]].rename(columns={"status": "router_null_gate"})
        if not gate.empty
        else pd.DataFrame(columns=[*columns, "router_null_gate"])
    )
    merged = table.merge(status, on=columns, how="left", validate="many_to_one")
    in_scope = merged.contrast.isin(gated)
    merged.loc[in_scope & merged.router_null_gate.isna(), "router_null_gate"] = (
        "missing_null_router_contrast"
    )
    merged.loc[~in_scope, "router_null_gate"] = np.nan
    return merged


def bootstrap_p_value(draws):
    """Two-sided paired bootstrap test of a zero difference.

    Twice the smaller share of finite draws on either side of zero, at most 1.
    """
    finite = np.asarray(draws, float)
    finite = finite[np.isfinite(finite)]
    if not len(finite):
        return np.nan
    return float(min(1.0, 2 * min((finite <= 0).mean(), (finite >= 0).mean())))


def claim_label(gain, lower, upper, margin):
    """Prespecified label of a pooled test-set contrast (lock, 2026-10-21).

    better: the 95% interval lies above zero; worse: below zero; equivalent:
    the interval lies within +-margin; otherwise inconclusive. A difference
    whose interval lies above zero and whose estimate exceeds the margin is
    also flagged as practically meaningful.
    """
    if not (np.isfinite(lower) and np.isfinite(upper)):
        return "not estimable"
    if lower > 0:
        return "better, above margin" if gain > margin else "better"
    if upper < 0:
        return "worse, beyond margin" if gain < -margin else "worse"
    if -margin <= lower and upper <= margin:
        return "equivalent"
    return "inconclusive"


def effect_summaries(contrasts, bootstrap_draws, cell_metadata, *, strata=None,
                     stable_cells_only=False, aggregation="equal_cell"):
    """Paired means with equal-cell or disease/outcome/horizon-balanced weights.

    No across-seed pooling: bootstrap intervals condition on one training seed.
    Strict fixed finite cell inventory is used in each replicate.
    ``stable_cells_only`` (patient-subset re-analyses, e.g. per region): the
    panel is the cells whose gain is finite in every replicate, so a few small
    cells that resample to one class cannot void the whole interval; the cells
    left out are counted in ``n_cells_unstable``.
    """
    if aggregation not in {"equal_cell", "disease_balanced"}:
        raise ValueError("Unknown effect-summary aggregation")
    keys = ["cohort", "outcome", "horizon_days"]
    if cell_metadata.duplicated(keys).any():
        raise ValueError("Duplicate cell metadata")
    values = contrasts.merge(
        cell_metadata, on=keys, how="left", validate="many_to_one", indicator=True
    )
    if not values._merge.eq("both").all():
        raise ValueError("Cell metadata must cover every evaluated task")
    values = values.drop(columns="_merge")
    strata = [
        key
        for key in (
            strata or ["support_bin", "outcome_family", "endpoint_group", "cohort"]
        )
        if key in values
    ]
    rows = []
    # Per horizon, and pooled over all horizons (horizon_days == ALL_HORIZONS),
    # every disease-outcome-horizon cell weighing equally.
    # Split the draws by contrast and metric once, not once per group.
    draws_by_contrast = dict(tuple(bootstrap_draws.groupby(["contrast", "metric"], sort=False)))
    for stratum, by_horizon in [(s, h) for s in [None, *strata] for h in (True, False)]:
        groups = ["split", "seed", "contrast", "metric"]
        if by_horizon:
            groups.append("horizon_days")
        if stratum:
            groups.append(stratum)
        for identity, group in values.groupby(groups, dropna=False):
            group = group.loc[np.isfinite(group.gain)]
            if group.empty:
                continue
            selected = group[CELL]
            samples = draws_by_contrast[(group.contrast.iloc[0], group.metric.iloc[0])].merge(
                selected, on=CELL, validate="many_to_one")
            arrays = np.stack(samples.gain_draws)
            unstable = int((~np.isfinite(arrays).all(axis=1)).sum())
            if stable_cells_only:
                stable = np.isfinite(arrays).all(axis=1)
                unstable = int((~stable).sum())
                if not stable.any():
                    continue
                kept = samples.loc[stable, CELL]
                samples, arrays = samples.loc[stable], arrays[stable]
                group = group.merge(kept, on=CELL, validate="one_to_one")
            def weights_for(frame):
                if aggregation == "equal_cell":
                    return np.full(len(frame), 1.0 / len(frame))
                diseases = frame.cohort.nunique()
                outcomes = frame.groupby("cohort").outcome.transform("nunique")
                horizons = frame.groupby(["cohort", "outcome"]).horizon_days.transform("nunique")
                return (1.0 / (diseases * outcomes * horizons)).to_numpy(float)
            weights = weights_for(samples)
            sampled = weights @ arrays
            sampled[~np.isfinite(arrays).all(axis=0)] = np.nan
            if len(samples) != len(group):
                raise ValueError(
                    "Bootstrap cell inventory does not match point-estimate inventory"
                )
            rows.append(
                {
                    **dict(zip(groups, identity)),
                    **({} if by_horizon else {"horizon_days": ALL_HORIZONS}),
                    "stratum": stratum or "all_cells",
                    "stratum_value": str(identity[-1]) if stratum else "all",
                    "gain": float(weights_for(group) @ group.gain.to_numpy(float)),
                    "n_diseases": group.cohort.nunique(),
                    "n_cells": len(group),
                    "n_cells_unstable": unstable,
                    "bootstrap_cell_policy": "stable_cells_only" if stable_cells_only else "strict_fixed_roster",
                    "n_bootstrap_replicates": arrays.shape[1],
                    "minimum_finite_cells_per_draw": int(np.isfinite(arrays).sum(axis=0).min()),
                    "aggregation": ("equal_cell_not_population_or_endpoint_group_macro"
                                    if aggregation == "equal_cell" else
                                    "equal_disease_then_outcome_then_horizon"),
                    "p_value": bootstrap_p_value(sampled),
                    **_interval(sampled),
                }
            )
    return values, pd.DataFrame(rows)


def tuning_selected_strategies(frame, candidates=("S_L", "D_L", "O_L")):
    """One strategy per disease/outcome/seed, chosen only on tuning log loss."""
    tuning = frame.loc[(frame.split == "tuning") & frame.model.isin(candidates)]
    # Split the tuning rows once, not once per disease and outcome.
    tuning_by_cell = dict(tuple(tuning.groupby(["seed", "cohort", "outcome"], sort=False)))
    empty = tuning.iloc[0:0]
    decisions, selected = [], []
    for key, cell in frame.groupby(["seed", "cohort", "outcome"]):
        identity = dict(zip(["seed", "cohort", "outcome"], key))
        identity.update(candidate_models='|'.join(candidates),
                        selection_metric='mean_horizon_ipcw_log_loss' if 'ipcw_weight' in frame
                        else 'mean_horizon_log_loss')
        eligible = tuning_by_cell.get(key, empty)
        by_model = {name: eligible.loc[eligible.model == name] for name in candidates}
        if any(group.empty for group in by_model.values()):
            decisions.append(
                {
                    **identity,
                    "status": "missing_tuning_candidates",
                    "selected_model": None,
                }
            )
            continue
        horizons = set().union(
            *(set(group.horizon_days) for group in by_model.values())
        )
        if any(set(group.horizon_days) != horizons for group in by_model.values()):
            raise ValueError(
                "Strategy candidates have different tuning horizon inventories"
            )
        losses = defaultdict(list)
        for horizon in sorted(horizons):
            aligned = align_models(
                {
                    name: group.loc[group.horizon_days == horizon]
                    for name, group in by_model.items()
                }
            )
            for name, group in aligned.items():
                losses[name].append(
                    metric_values(
                        group.label,
                        group.probability,
                        ipcw=group.ipcw_weight if "ipcw_weight" in group else None,
                    )["log_loss"]
                )
        winner = min(
            candidates, key=lambda name: (np.mean(losses[name]), candidates.index(name))
        )
        output = cell.loc[(cell.split == "held_out") & (cell.model == winner)].copy()
        if output.empty:
            decisions.append(
                {
                    **identity,
                    "status": "selected_held_out_missing",
                    "selected_model": winner,
                }
            )
            continue
        output["model"] = "BEST_TUNING"
        selected.append(output)
        decisions.append(
            {
                **identity,
                "status": "selected_on_tuning",
                "selected_model": winner,
                "selection_log_loss": np.mean(losses[winner]),
            }
        )
    return pd.concat(selected, ignore_index=True) if selected else frame.iloc[
        :0
    ], pd.DataFrame(decisions)


def negative_transfer_strata(
    contrasts, *, pooling="P5", rescue="R1", tolerance=0.0, min_cases=5, min_controls=5
):
    """Define harmed cells using tuning only; inspect held-out rescue separately."""
    keys = ["seed", "cohort", "outcome", "horizon_days", "metric"]
    tuning = contrasts.loc[
        (contrasts.split == "tuning")
        & (contrasts.contrast == pooling)
        & (contrasts.metric != "calibration_bias")
    ].copy()
    tuning["stratum"] = np.where(
        (tuning.n_cases >= min_cases)
        & (tuning.n_controls >= min_controls)
        & np.isfinite(tuning.gain),
        np.where(tuning.gain < -tolerance, "tuning_negative_pooling", "tuning_other"),
        "tuning_insufficient_support",
    )
    held = contrasts.loc[
        (contrasts.split == "held_out")
        & contrasts.contrast.isin([pooling, rescue])
        & (contrasts.metric != "calibration_bias")
    ]
    return held.merge(
        tuning[keys + ["stratum"]], on=keys, how="left", validate="many_to_one"
    ).fillna({"stratum": "missing_tuning_evidence"})


def subgroup_labels(gains, metadata, specifications):
    """Yield (covariate, gains with a ``subgroup`` column) for each covariate."""
    if metadata.subject_id.duplicated().any():
        raise ValueError("Metadata must have one row per patient")
    missing = set(specifications) - set(metadata)
    if missing:
        raise ValueError(f"Missing subgroup columns: {sorted(missing)}")
    enriched = gains.merge(
        metadata[["subject_id", *specifications]],
        on="subject_id",
        how="left",
        validate="many_to_one",
        indicator=True,
    )
    if not enriched._merge.eq("both").all():
        raise ValueError("Missing patient metadata; cannot silently drop patients")
    for column, specification in specifications.items():
        if specification.get("provenance") not in {"pre_index", "train_only"}:
            raise ValueError(
                f"Subgroup {column} requires pre_index/train_only provenance"
            )
        value = enriched[column]
        if specification.get("bins"):
            value = pd.cut(
                pd.to_numeric(value, errors="raise"),
                bins=specification["bins"],
                include_lowest=True,
            )
        elif column == "index_year" or specification.get("kind") == "calendar_year":
            # Calendar years are discrete strata, even across a long registry.
            # Retain the cardinality safeguard for other numeric covariates.
            year = pd.to_numeric(value, errors="raise")
            known = year.dropna()
            if not (known.between(1800, 2200) & known.eq(np.floor(known))).all():
                raise ValueError(f"Invalid calendar-year subgroup: {column}")
            value = year.astype("Int64")
        elif value.nunique() > specification.get("max_categories", 20):
            raise ValueError(
                f"Continuous/high-cardinality subgroup needs prespecified bins: {column}"
            )
        labelled = enriched.copy()
        labelled["subgroup"] = (
            value.astype(object).where(value.notna(), "missing").astype(str)
        )
        yield column, labelled


_GAIN_METRICS = ("brier_gain", "log_loss_gain")


def _subgroup_cell_sums(enriched):
    """The patient gains of each cell and contrast (all subgroups together)."""
    yield from enriched.groupby([*CELL, "contrast"], sort=True)


def subgroup_patient_gains(gains, metadata, specifications, *, draws):
    """Pre-index, prespecified subgroup loss means with joint patient bootstrap.

    Vectorised over subgroups: one weight matrix per cell and contrast, summed
    over all subgroups with an indicator matrix. Same rows, in the same order,
    as computing every subgroup separately.
    """
    rows, samples = [], []
    for column, enriched in subgroup_labels(gains, metadata, specifications):
        for key, group in _subgroup_cell_sums(enriched):
            mapping, counts = draws[key[:3]]
            weights = counts[:, [mapping[sid] for sid in group.subject_id]].astype(float)
            codes, labels = pd.factorize(group.subgroup, sort=True)
            indicator = np.zeros((len(group), len(labels)))
            indicator[np.arange(len(group)), codes] = 1.0
            denominators = weights @ indicator
            sizes = indicator.sum(axis=0)
            for s, label in enumerate(labels):
                members = codes == s
                n_subjects = group.subject_id.to_numpy()[members]
                for metric in _GAIN_METRICS:
                    values = group[metric].to_numpy(float)
                    numerator = weights @ (indicator[:, s] * values)
                    sample = np.divide(numerator, denominators[:, s],
                                       out=np.full(len(weights), np.nan),
                                       where=denominators[:, s] > 0)
                    rows.append({
                        **dict(zip([*CELL, "contrast", "subgroup"], (*key, label))),
                        "covariate": column,
                        "metric": metric,
                        "gain": values[members].mean() if sizes[s] else np.nan,
                        "n_subjects": len(pd.unique(n_subjects)),
                    })
                    samples.append(sample)
    if not rows:
        return pd.DataFrame()
    return pd.concat([pd.DataFrame(rows), _intervals(samples)], axis=1)


def panel_subgroup_gains(gains, metadata, specifications, *, draws):
    """Subgroup loss gains pooled over every cell at a horizon.

    Each patient contributes once per cell in which they are evaluated, so the
    estimate is the patient-weighted mean gain across the panel. Intervals use
    the shared within-disease patient resamples, so a patient appearing in
    several cells is resampled jointly across them. Vectorised over subgroups
    (one weight matrix per cell and contrast); same rows as the per-subgroup loop.
    """
    keys = ["split", "seed", "horizon_days", "contrast", "subgroup"]
    rows, samples = [], []
    for column, enriched in subgroup_labels(gains, metadata, specifications):
        totals = {}
        for key, group in _subgroup_cell_sums(enriched):
            mapping, counts = draws[key[:3]]
            weights = counts[:, [mapping[sid] for sid in group.subject_id]].astype(float)
            codes, labels = pd.factorize(group.subgroup, sort=True)
            indicator = np.zeros((len(group), len(labels)))
            indicator[np.arange(len(group)), codes] = 1.0
            denominators = weights @ indicator
            numerators = {m: weights @ (indicator * group[m].to_numpy(float)[:, None])
                          for m in _GAIN_METRICS}
            split, seed, _, _, horizon, contrast = key
            for s, label in enumerate(labels):
                # Each horizon, and every horizon together (ALL_HORIZONS: each
                # patient once per cell and horizon in which they are evaluated).
                for pooled in (horizon, ALL_HORIZONS):
                    panel_key = (split, seed, pooled, contrast, label)
                    entry = totals.setdefault(panel_key, {"denominator": 0.0,
                                                          **{m: 0.0 for m in _GAIN_METRICS}})
                    entry["denominator"] = entry["denominator"] + denominators[:, s]
                    for m in _GAIN_METRICS:
                        entry[m] = entry[m] + numerators[m][:, s]
        aggregations = dict(
            n_subjects=("subject_id", "nunique"), n_patient_cells=("subject_id", "size"),
            **{f"mean_{m}": (m, "mean") for m in _GAIN_METRICS})
        summary = pd.concat([
            enriched.groupby(keys).agg(**aggregations),
            enriched.assign(horizon_days=ALL_HORIZONS).groupby(keys).agg(**aggregations)])
        for panel_key in sorted(totals):
            entry, info = totals[panel_key], summary.loc[panel_key]
            denominator = np.atleast_1d(entry["denominator"])
            for metric in _GAIN_METRICS:
                rows.append({
                    **dict(zip(keys, panel_key)),
                    "covariate": column,
                    "metric": metric,
                    "gain": info[f"mean_{metric}"],
                    "n_subjects": int(info.n_subjects),
                    "n_patient_cells": int(info.n_patient_cells),
                    "aggregation": "patient_weighted_across_cells",
                })
                samples.append(np.divide(entry[metric], denominator,
                                         out=np.full(len(denominator), np.nan),
                                         where=denominator > 0))
    if not rows:
        return pd.DataFrame()
    return pd.concat([pd.DataFrame(rows), _intervals(samples)], axis=1)


def reliability_tables(frame, bins=10):
    """Fixed-bin descriptive calibration; no calibrator fit on held-out labels.

    Under the full-risk-set estimand the observed risk in each bin is the
    IPCW-weighted event fraction; otherwise the plain known-status fraction.
    """
    frame = frame.copy()
    frame["probability_bin"] = (frame.probability * bins).astype(int).clip(0, bins - 1)
    frame["w"] = frame.ipcw_weight if "ipcw_weight" in frame else 1.0
    frame["wy"] = frame.w * frame.label
    frame["wp"] = frame.w * frame.probability
    table = (
        frame.groupby(["model", *CELL, "probability_bin"], observed=True)
        .agg(n=("label", "size"), w=("w", "sum"), wy=("wy", "sum"), wp=("wp", "sum"),
             predicted_mean=("probability", "mean"))
        .reset_index()
    )
    table["observed_fraction"] = np.where(table.w > 0, table.wy / table.w, np.nan)
    table["predicted_weighted_mean"] = np.where(table.w > 0, table.wp / table.w, np.nan)
    return table.drop(columns=["wy", "wp"])


CALIBRATION_LIMIT = 50.0  # |intercept| or |slope| beyond this: separation, not a fit


def _calibration_fit(labels, probabilities, weights, iterations=50):
    """Weighted logistic recalibration ``logit P(y) = a + b logit(p)``.

    Newton steps with a numerically stable sigmoid. When the predictions (nearly)
    separate events from non-events the likelihood has no finite maximum and
    Newton diverges; such fits, and fits that do not converge, are returned as
    not estimable (NaN) rather than as a meaningless slope."""
    from scipy.special import expit

    p = np.clip(probabilities, 1e-6, 1 - 1e-6)
    x = np.column_stack([np.ones(len(p)), np.log(p / (1 - p))])
    beta = np.array([0.0, 1.0])
    for _ in range(iterations):
        mu = expit(x @ beta)
        gradient = x.T @ (weights * (labels - mu))
        hessian = (x * (weights * mu * (1 - mu))[:, None]).T @ x
        try:
            step = np.linalg.solve(hessian + 1e-9 * np.eye(2), gradient)
        except np.linalg.LinAlgError:
            return np.nan, np.nan
        beta = beta + step
        if not np.isfinite(beta).all() or np.abs(beta).max() > CALIBRATION_LIMIT:
            return np.nan, np.nan
        if np.max(np.abs(step)) < 1e-8:
            return float(beta[0]), float(beta[1])
    return np.nan, np.nan  # did not converge


def parallel_ordered(function, tasks, workers):
    """``function`` over ``tasks`` in a process pool, results in task order, with at
    most two tasks per worker in flight (bounded memory)."""
    from collections import deque
    from concurrent.futures import ProcessPoolExecutor

    with ProcessPoolExecutor(max_workers=workers) as pool:
        pending = deque()
        for task in tasks:
            pending.append(pool.submit(function, task))
            if len(pending) >= 2 * workers:
                yield pending.popleft().result()
        while pending:
            yield pending.popleft().result()


def calibration_summary(frame, *, min_events=10):
    """Per model and cell (see ``_calibration_summary``), in parallel by model with
    OPERA_ANALYSIS_WORKERS; the same rows in the same order as a serial run."""
    workers = analysis_workers()
    models = sorted(frame.model.unique())
    if workers <= 1 or len(models) < 2:
        return _calibration_summary(frame, min_events=min_events)
    tasks = ((frame.loc[frame.model == model], min_events) for model in models)
    parts = list(parallel_ordered(_calibration_summary_task, tasks, workers))
    return pd.concat(parts, ignore_index=True)


def _calibration_summary_task(task):
    frame, min_events = task
    return _calibration_summary(frame, min_events=min_events)


def _calibration_summary(frame, *, min_events=10):
    """Per model and cell: calibration slope, intercept, O/E and ECE.

    The slope comes from a weighted logistic recalibration of the outcome on
    the logit of the prediction (ideal 1); calibration-in-the-large is
    observed minus mean predicted risk (ideal 0); O/E is their ratio. Weights
    are the IPCW weights under the full-risk-set estimand. Nothing is refitted
    for prediction: these describe calibration, they do not change it.
    """
    rows = []
    for key, group in frame.groupby(["model", *CELL], sort=True):
        weights = (group.ipcw_weight.to_numpy(float) if "ipcw_weight" in group
                   else np.ones(len(group)))
        y = group.label.to_numpy(float)
        p = group.probability.to_numpy(float)
        used = weights > 0
        y, p, weights = y[used], p[used], weights[used]
        events = int(y.sum())
        identity = dict(zip(["model", *CELL], key))
        if events < min_events or events == len(y):
            rows.append({**identity, "status": "too_few_events", "n_events": events})
            continue
        intercept, slope = _calibration_fit(y, p, weights)
        observed = float(np.average(y, weights=weights))
        expected = float(np.average(p, weights=weights))
        rows.append({
            **identity, "status": "estimated", "n_events": events,
            "calibration_slope": slope, "recalibration_intercept": intercept,
            "calibration_in_the_large": observed - expected,
            "observed_expected_ratio": observed / expected if expected > 0 else np.nan,
            "ece": metric_values(y, p, weights)["ece"],
        })
    return pd.DataFrame(rows)


def audit_router_identity(metadata):
    """Audit identity provenance without changing cohort membership or routing.

    Later registry coding is a review flag, not proof the clinician did not know
    the disease. An independently documented baseline identity can clear it.
    """
    required = {"subject_id", "index_date", "cohort_identity_evidence_date"}
    if required - set(metadata):
        return pd.DataFrame(
            [
                {
                    "status": "missing_identity_time_provenance",
                    "missing_columns": ",".join(sorted(required - set(metadata))),
                }
            ]
        )
    if metadata.subject_id.duplicated().any():
        raise ValueError("Identity provenance must have one row per patient")
    frame = metadata.copy()
    index = pd.to_datetime(frame.index_date, errors="raise")
    evidence = pd.to_datetime(frame.cohort_identity_evidence_date, errors="raise")
    if index.isna().any():
        raise ValueError("Missing prediction index date in identity provenance")
    verified = pd.Series(False, index=frame.index)
    if "baseline_disease_clinically_confirmed" in frame:
        values = frame.baseline_disease_clinically_confirmed
        if not values.isin([True, False, 0, 1]).all():
            raise ValueError("Baseline clinical confirmation must be boolean")
        verified = values.astype(bool)
        if verified.any() and (
            "baseline_confirmation_reference" not in frame
            or frame.loc[verified, "baseline_confirmation_reference"].isna().any()
            or frame.loc[verified, "baseline_confirmation_reference"]
            .astype(str)
            .str.strip()
            .eq("")
            .any()
        ):
            raise ValueError(
                "Clinical confirmation requires an independent provenance reference"
            )
    frame["status"] = np.where(
        verified,
        "clinically_confirmed_at_index",
        np.where(
            evidence < index,
            "evidence_preindex",
            np.where(
                evidence.notna(),
                "same_or_postindex_evidence_needs_review",
                "missing_identity_evidence",
            ),
        ),
    )
    if "cohort_source" not in frame:
        frame["cohort_source"] = "unspecified"
    frame["diagnosis_delay_days"] = (evidence - index).dt.total_seconds() / 86400
    frame["coding_delay_category"] = pd.cut(
        frame.diagnosis_delay_days,
        [-np.inf, -1e-9, 0, 14, 90, np.inf],
        labels=[
            "before_index",
            "same_timestamp",
            "later_1_to_14_days",
            "later_15_to_90_days",
            "later_over_90_days",
        ],
    )
    frame["coding_delay_category"] = frame.coding_delay_category.astype(object).where(
        frame.coding_delay_category.notna(), "missing_date"
    )
    keys = ["cohort_source", "status", "coding_delay_category"]
    if "cohort_fine" in frame:
        keys.insert(0, "cohort_fine")
    return (
        frame.groupby(keys, dropna=False)
        .agg(
            n_patients=("subject_id", "size"),
            median_diagnosis_delay_days=("diagnosis_delay_days", "median"),
            maximum_diagnosis_delay_days=("diagnosis_delay_days", "max"),
        )
        .reset_index()
    )


def coverage_inventory(frame, roster):
    """Explicit planned cell availability, including failures/unsupported fits."""
    required = {"model", "seed", "cohort", "outcome", "horizon_days", "status"}
    if "split" in roster:
        required.add("split")
    if required - set(roster) or roster.duplicated(list(required - {"status"})).any():
        raise ValueError("Coverage roster requires unique model/seed/task and status")
    # One aggregation per model and cell, merged onto the roster (filtering the
    # whole prediction table once per roster row is quadratic).
    keys = sorted(required - {"status"})
    if "split" not in roster:
        frame = frame.loc[frame.split == "held_out"]
    numeric = {"seed", "horizon_days"}

    def keyed(table):
        table = table.copy()
        for key in keys:
            table[key] = (pd.to_numeric(table[key]).astype(float) if key in numeric
                          else table[key].astype(str))
        return table

    known_frame = frame.loc[frame.ipcw_weight > 0] if "ipcw_weight" in frame else frame
    counts = (keyed(known_frame[[*keys, "label"]]).groupby(keys, sort=False)
              .label.agg(n_known_predictions="size", n_cases="sum").reset_index())
    merged = keyed(roster[keys]).merge(counts, on=keys, how="left", validate="many_to_one")
    n = merged.n_known_predictions.fillna(0).astype(int).to_numpy()
    cases = merged.n_cases.fillna(0).astype(int).to_numpy()
    result = roster.reset_index(drop=True).copy()
    result["n_known_predictions"] = n
    result["n_cases"] = cases
    result["n_controls"] = n - cases
    result["prediction_status"] = np.where(
        n == 0, "absent", np.where((cases == 0) | (cases == n), "single_class", "two_class"))
    result["auc_estimable"] = (cases > 0) & (cases < n)
    return result


def route_summaries(routes):
    """Describe exported gates without calling them predictive contributions."""
    keys = ["model", "seed", "split", "cohort", "outcome", "subject_id"]
    if {*keys, "source", "bank", "weight"} - set(routes):
        raise ValueError("Routes require model/seed/task/patient/source/bank/weight")
    if (
        routes.duplicated([*keys, "source"]).any()
        or not np.isfinite(routes.weight).all()
        or not routes.weight.between(0, 1).all()
    ):
        raise ValueError("Duplicate or invalid route weights")
    if not np.allclose(routes.groupby(keys).weight.sum(), 1, atol=1e-5):
        raise ValueError(
            "Export all sources including shared; gate weights must sum to one"
        )
    return (
        routes.groupby(
            ["model", "seed", "split", "cohort", "outcome", "bank", "source"]
        )
        .weight.agg(
            mean="mean", median="median", n="size", minimum="min", maximum="max"
        )
        .reset_index()
    )
