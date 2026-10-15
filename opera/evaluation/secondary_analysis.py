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
METRICS = ("auc", "brier", "log_loss", "calibration_bias", "ece")
ESTIMANDS = ("binary_known_status_unweighted", "cif_ipcw")


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
    """Full-risk-set CIF labels and IPCW weights per model/seed/disease/outcome."""
    if not {"time_days", "event"} <= set(frame) or frame[["time_days", "event"]].isna().any().any():
        raise ValueError("cif_ipcw requires follow-up time_days and event for every model and row")
    parts = []
    for _, group in frame.groupby(["model", "split", "seed", "cohort", "outcome"], sort=False):
        patients = group.drop_duplicates("subject_id")
        if (group.groupby("subject_id")[["time_days", "event"]].nunique() > 1).any().any():
            raise ValueError("A patient's follow-up must not vary across horizons")
        before = _censoring_survival_before(
            patients.time_days.to_numpy(float), patients.event.to_numpy(int)
        )
        t = group.time_days.to_numpy(float)
        e = group.event.to_numpy(int)
        h = group.horizon_days.to_numpy(float)
        case = (t <= h) & (e == 1)
        competing = (t <= h) & (e == 2)
        control = competing | (t > h) | ((t >= h) & (e == 0))
        weight = np.zeros(len(group))
        weight[case | competing] = 1.0 / before(t[case | competing])
        horizon_control = control & ~competing
        weight[horizon_control] = 1.0 / before(h[horizon_control])
        group = group.copy()
        group["label"] = case.astype(int)
        group["ipcw_weight"] = weight
        group["known"] = case | control
        if "prediction_absent_zero_weight" in group:
            placeholder = group.prediction_absent_zero_weight.fillna(False).astype(bool)
            if (placeholder & group.known).any():
                raise ValueError("A status-known patient lacks a real prediction")
        parts.append(group)
    return pd.concat(parts).sort_index()


def normalize_predictions(
    frame: pd.DataFrame, estimand: str = "binary_known_status_unweighted"
) -> tuple[pd.DataFrame, dict]:
    """Validate before casting, then apply the requested estimand.

    Known-status analysis retains only rows whose horizon status is known.
    ``cif_ipcw`` retains the full risk set with IPCW weights (zero for rows
    censored before the horizon).
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
    if frame.empty or frame[list(required - {"label"})].isna().any().any():
        raise ValueError("Empty predictions or missing prediction identity/probability")
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
    if (frame.horizon_days <= 0).any() or not frame.split.isin(
        ["tuning", "held_out"]
    ).all():
        raise ValueError("Need positive horizons and tuning/held_out splits")
    if frame.duplicated(["model", *PATIENT]).any():
        raise ValueError("Duplicate model/patient/task predictions")
    if (frame.groupby("subject_id").split.nunique() > 1).any():
        raise ValueError("Patient overlap between tuning and held_out predictions")
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
    for column in ("time_days", "event"):
        available = [
            group
            for group in result.values()
            if column in group and group[column].notna().all()
        ]
        if available and any(
            not np.array_equal(
                group[column].to_numpy(), available[0][column].to_numpy()
            )
            for group in available[1:]
        ):
            raise ValueError(f"Follow-up parity failed: {column}")
    return result


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
    return {
        "auc": auc,
        "brier": risk_set_average((labels - probabilities) ** 2),
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
        error = sum(
            np.abs(w[:, bins == index] @ (p - y)[bins == index]) for index in range(10)
        )
        scores["ece"] = np.divide(
            error, denominator, out=np.full(len(w), np.nan), where=denominator > 0
        )
        for metric in METRICS:
            result[metric].extend(scores[metric])
    return {metric: np.asarray(values) for metric, values in result.items()}


def patient_bootstrap(frame: pd.DataFrame, replicates: int, seed: int):
    """One subject-count draw reused across outcomes, horizons and methods."""
    if replicates < 0:
        raise ValueError("Bootstrap replicates cannot be negative")
    draws = {}
    for key, group in frame.groupby(["split", "cohort"], sort=True):
        ids = sorted(group.subject_id.unique())
        digest = hashlib.sha256(repr((seed, key)).encode()).digest()
        rng = np.random.default_rng(int.from_bytes(digest[:8], "little"))
        shared = (
            dict(zip(ids, range(len(ids)))),
            rng.multinomial(len(ids), np.full(len(ids), 1 / len(ids)), size=replicates),
        )
        for training_seed in group.seed.unique():
            draws[(key[0], training_seed, key[1])] = shared
    return draws


def seed_averaged_effects(contrasts, bootstrap_draws):
    """Patient uncertainty for the mean of the fitted seed roster; not seed CI."""
    keys = ["split", "cohort", "outcome", "horizon_days", "contrast", "metric"]
    rows = []
    for key, group in contrasts.groupby(keys):
        samples = bootstrap_draws
        for column, value in zip(keys, key):
            samples = samples.loc[samples[column] == value]
        samples = samples.loc[samples.seed.isin(group.seed)]
        arrays = np.stack(samples.gain_draws)
        finite = np.isfinite(arrays).all(axis=0)
        average = arrays.mean(axis=0)
        average[~finite] = np.nan
        rows.append(
            {
                **dict(zip(keys, key)),
                "gain": group.gain.mean() if np.isfinite(group.gain).all() else np.nan,
                "n_training_seeds": len(group),
                "seed_minimum": group.gain.min(),
                "seed_maximum": group.gain.max(),
                "seed_sd": group.gain.std(),
                "uncertainty": "patients_conditional_on_fitted_seed_roster",
                **_interval(average),
            }
        )
    return pd.DataFrame(rows)


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


def paired_analysis(
    frame,
    contrasts,
    *,
    replicates=1000,
    seed=2026,
    draws=None,
    patient_gain_contrasts=None,
):
    """Cell estimates, paired contrasts, joint draws and patient loss contributions."""
    draws = patient_bootstrap(frame, replicates, seed) if draws is None else draws
    estimates, differences, draw_rows, contributions, coverage = [], [], [], [], []
    model_draw_rows = []
    parsed = {name: contrast_coefficients(expr) for name, expr in contrasts.items()}
    for key, cell in frame.groupby(CELL, sort=True):
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
        values = {
            name: metric_values(y, group.probability.to_numpy(float), ipcw=ipcw)
            for name, group in models.items()
        }
        mapping, counts = draws[key[:3]]
        positions = [mapping[sid] for sid in reference.subject_id]
        weights = counts[:, positions]
        scores_by_draw = {
            name: bootstrap_metric_values(
                y, group.probability.to_numpy(float), weights, ipcw=ipcw
            )
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
                    "status": "missing_models" if missing else "paired",
                    "missing_models": ",".join(sorted(missing)),
                }
            )
            if missing:
                continue
            for metric in METRICS:
                # Bias is signed calibration error, not a higher-is-better skill.
                orientation = 1 if metric in {"auc", "calibration_bias"} else -1
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


def panel_estimates(estimates, model_draws, groups):
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
            for model in members:
                selected = part.set_index(["cohort", "outcome"]).loc[cells]
                selected = selected.loc[selected.model == model].reset_index()
                draws = model_draws.loc[model_draws.model == model]
                for column, value in zip(keys, identity):
                    draws = draws.loc[draws[column] == value]
                draws = draws.merge(selected[["cohort", "outcome"]],
                                    on=["cohort", "outcome"])
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
    return pd.DataFrame(rows)


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


def effect_summaries(contrasts, bootstrap_draws, cell_metadata, *, strata=None):
    """Equal-cell descriptive summaries with subject-correlated intervals.

    No across-seed pooling: bootstrap intervals condition on one training seed.
    Strict fixed finite cell inventory is used in each replicate.
    """
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
    for stratum in [None, *strata]:
        groups = ["split", "seed", "contrast", "metric", "horizon_days"]
        if stratum:
            groups.append(stratum)
        for identity, group in values.groupby(groups, dropna=False):
            group = group.loc[np.isfinite(group.gain)]
            if group.empty:
                continue
            selected = group[CELL]
            samples = bootstrap_draws.loc[
                (bootstrap_draws.contrast == group.contrast.iloc[0])
                & (bootstrap_draws.metric == group.metric.iloc[0])
            ].merge(selected, on=CELL, validate="many_to_one")
            arrays = np.stack(samples.gain_draws)
            sampled = arrays.mean(axis=0)
            sampled[~np.isfinite(arrays).all(axis=0)] = np.nan
            if len(samples) != len(group):
                raise ValueError(
                    "Bootstrap cell inventory does not match point-estimate inventory"
                )
            rows.append(
                {
                    **dict(zip(groups, identity)),
                    "stratum": stratum or "all_cells",
                    "stratum_value": str(identity[-1]) if stratum else "all",
                    "gain": group.gain.mean(),
                    "n_cells": len(group),
                    "aggregation": "equal_cell_not_population_or_endpoint_group_macro",
                    **_interval(sampled),
                }
            )
    return values, pd.DataFrame(rows)


def tuning_selected_strategies(frame, candidates=("S_L", "D_L", "O_L")):
    """One strategy per disease/outcome/seed, chosen only on tuning log loss."""
    tuning = frame.loc[frame.split == "tuning"]
    decisions, selected = [], []
    for key, cell in frame.groupby(["seed", "cohort", "outcome"]):
        identity = dict(zip(["seed", "cohort", "outcome"], key))
        eligible = tuning
        for column, value in identity.items():
            eligible = eligible.loc[eligible[column] == value]
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


def _subgroup_labels(gains, metadata, specifications):
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
        elif value.nunique() > specification.get("max_categories", 20):
            raise ValueError(
                f"Continuous/high-cardinality subgroup needs prespecified bins: {column}"
            )
        labelled = enriched.copy()
        labelled["subgroup"] = (
            value.astype(object).where(value.notna(), "missing").astype(str)
        )
        yield column, labelled


def subgroup_patient_gains(gains, metadata, specifications, *, draws):
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


def panel_subgroup_gains(gains, metadata, specifications, *, draws):
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


def _calibration_fit(labels, probabilities, weights, iterations=50):
    """Weighted logistic recalibration ``logit P(y) = a + b logit(p)``."""
    p = np.clip(probabilities, 1e-6, 1 - 1e-6)
    x = np.column_stack([np.ones(len(p)), np.log(p / (1 - p))])
    beta = np.array([0.0, 1.0])
    for _ in range(iterations):
        mu = 1 / (1 + np.exp(-(x @ beta)))
        gradient = x.T @ (weights * (labels - mu))
        hessian = (x * (weights * mu * (1 - mu))[:, None]).T @ x
        try:
            step = np.linalg.solve(hessian + 1e-9 * np.eye(2), gradient)
        except np.linalg.LinAlgError:
            return np.nan, np.nan
        beta = beta + step
        if np.max(np.abs(step)) < 1e-8:
            break
    return float(beta[0]), float(beta[1])


def calibration_summary(frame, *, min_events=10):
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
    rows = []
    for record in roster.to_dict("records"):
        subset = frame.loc[frame.split == record.get("split", "held_out")]
        for column in required - {"status"}:
            subset = subset.loc[subset[column] == record[column]]
        cases = int(subset.label.sum())
        rows.append(
            {
                **record,
                "n_known_predictions": len(subset),
                "n_cases": cases,
                "n_controls": len(subset) - cases,
                "prediction_status": "absent"
                if subset.empty
                else "single_class"
                if cases in (0, len(subset))
                else "two_class",
                "auc_estimable": cases > 0 and cases < len(subset),
            }
        )
    return pd.DataFrame(rows)


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
