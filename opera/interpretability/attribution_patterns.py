"""Episodes of care and clinical names for local explanations.

The unit of a local explanation is the clinical *episode*: events within
``gap_days`` of each other (an admission, a diagnostic work-up), not single
timestamps, because one admission's diagnoses, results and drugs are recorded
at different times. Concepts are named from a reviewed list of codes, with
laboratory results named by the value third of the result.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from opera.interpretability.attribution_summaries import (
    SOURCE_ORDER, _lab_third, rollup, source_of)

def episode_ids(tokens: pd.DataFrame, gap_days: float = 1.0) -> pd.Series:
    """Episode index per token: events within ``gap_days`` belong together.

    Background tokens (no time) form episode -1. Episodes are numbered from the
    oldest, per patient.
    """
    days = tokens.days_before_index.astype(float)
    background = tokens.code.astype(str).str.startswith("BG_")
    order = tokens.assign(_d=days.where(~background)).sort_values(
        ["subject_id", "_d"], ascending=[True, False])
    timed = order.loc[order._d.notna()]
    gap = timed.groupby("subject_id")._d.diff().abs()
    new = gap.isna() | (gap > gap_days)
    ids = new.astype(int).groupby(timed.subject_id).cumsum() - 1
    out = pd.Series(-1, index=tokens.index, dtype=int)
    out.loc[ids.index] = ids.astype(int)
    return out


def _concept(code, counts):
    concept = rollup(code)
    if counts and concept.startswith("LABTEST//") and concept.endswith("//BIN"):
        third = _lab_third(code, counts)
        return concept[:-3] + third if third else concept
    return concept


def concept_label(concept, labels):
    """Clinical name of a rolled-up concept, e.g. ``LABTEST//NPU02902//low``.

    Reviewed labels are looked up for the concept itself, then for its test
    (``LABTEST//<test>//BIN``) with the value third added; unknown concepts keep
    their code.
    """
    if concept in labels:
        return labels[concept]
    if str(concept).startswith("LABTEST//"):
        head, _, third = str(concept).rpartition("//")
        base = labels.get(f"{head}//BIN")
        if base and third in {"low", "high"}:
            return f"{base}, {'lower' if third == 'low' else 'upper'} bins"
        if base and third == "mid":
            return f"{base}, middle bins"
    return _plain(concept)


def _lower_first(text):
    """Lower-case the first letter unless the word is an acronym (LDH, CRP)."""
    word = text.split(" ", 1)[0]
    return text if word.isupper() or len(word) < 2 or word[1].isupper() else \
        text[0].lower() + text[1:]


def _plain(concept):
    """Readable fallback for an unlabelled concept: ``Pathology M96``.

    Contact attributes, regions and patient types hold words (``ACUTE``), which
    are lower-cased; codes (ICD, ATC, SNOMED, procedures) are kept as they are.
    """
    source, _, code = str(concept).partition("//")
    if not code:
        return str(concept)
    name = source.replace("_", " ").capitalize()
    value = code.replace("//", " ")
    if source.startswith(("CONTACT_", "REGION", "PATIENT_TYPE")):
        value = value.replace("_", " ").lower()
    return f"{name} {value}"


def _code_label(code, labels, counts=None):
    """Reviewed label for a code's concept; laboratory results keep their value.

    A laboratory result reads as its value third when the bin ranges are known
    (``Albumin, lower bins``, ``Albumin, middle bins``), otherwise as its bin.
    Relative bins do not establish whether a result is clinically abnormal.
    """
    if code in labels:
        return labels[code]
    concept = _concept(code, counts)
    if concept in labels:
        return labels[concept]
    base = rollup(code)
    text = labels.get(base, base)
    if base.startswith("LABTEST//") and "//BIN_" in str(code):
        third = concept.rsplit("//", 1)[1] if concept != base else ""
        if third == "low":
            return f"{text}, lower bins"
        if third == "high":
            return f"{text}, upper bins"
        if third == "mid":
            return f"{text}, middle bins"
        return f"{text} (bin {str(code).rsplit('BIN_', 1)[1]})"
    return text


def patient_episodes(tokens, *, gap_days=1.0, labels=None, top_codes=3, counts=None):
    """INTERNAL: one patient's episodes with contribution and their main codes."""
    labels = labels or {}
    frame = tokens.assign(episode=episode_ids(tokens, gap_days),
                          source=tokens.code.map(source_of))
    rows = []
    for episode, part in frame.groupby("episode"):
        by_code = part.groupby("code").attribution.sum()
        main = by_code.abs().sort_values(ascending=False).index[:top_codes]
        rows.append({"episode": int(episode), "attribution": float(part.attribution.sum()),
                     "start_days": float(part.days_before_index.max()),
                     "end_days": float(part.days_before_index.min()),
                     "sources": ", ".join(sorted(set(part.source),
                                                 key=SOURCE_ORDER.index)),
                     "main_codes": "; ".join(_code_label(c, labels, counts) for c in main),
                     "n_events": len(part)})
    return pd.DataFrame(rows).sort_values("start_days", ascending=False)


def when(days):
    """Plain timing of an episode before treatment: ``2 mo before``."""
    days = float(days)
    if days < 7:
        return f"{days:.0f} d before"
    if days < 60:
        return f"{days / 7:.0f} wk before"
    if days < 730:
        return f"{days / 30.4:.0f} mo before"
    return f"{days / 365.25:.0f} y before"


def _patient_episodes(tokens, columns, *, gap_days, labels, counts):
    """Episodes of one patient: contribution per outcome (pp) and a description."""
    frame = tokens.assign(episode=episode_ids(tokens, gap_days))
    names = list(columns.values())
    sums = frame.groupby("episode")[names].sum(min_count=1) * 100
    sums.columns = list(columns)
    info = frame.groupby("episode").days_before_index.agg(start_days="max", end_days="min")
    descriptions = {}
    for episode, part in frame.groupby("episode"):
        weight = part[names].abs().sum(1).groupby(part.code).sum()
        main = weight.sort_values(ascending=False).index[:3]
        descriptions[episode] = "; ".join(_code_label(c, labels, counts) for c in main)
    info["description"] = pd.Series(descriptions)
    info["when"] = [("background" if e < 0 else when(d))
                    for e, d in zip(info.index, info.start_days)]
    return sums, info


def episode_contributions(tokens, patients, subject_id, *, columns, gap_days=1.0, labels=None,
                          counts=None, scale=100.0):
    """INTERNAL: every episode of one patient, with its contribution to each outcome.

    One row per episode and outcome: timing (``start_days``, ``end_days``,
    ``when``), clinical description, contribution (``scale`` x the summed
    credit, percentage points for absolute-risk columns) and the outcome's
    reference and predicted values. Background tokens (no time) are left out.
    """
    labels = labels or {}
    tokens = tokens.loc[tokens.subject_id == subject_id]
    sums, info = _patient_episodes(tokens, columns, gap_days=gap_days, labels=labels,
                                   counts=counts)
    sums = sums.loc[sums.index >= 0] * (scale / 100)
    rows = []
    for outcome, column in columns.items():
        found = patients.loc[(patients.subject_id == subject_id) & (patients.column == column)]
        if found.empty or sums[outcome].isna().all():
            continue
        prediction = float(found.prediction.iloc[0])
        reference = float(found.baseline_prediction.iloc[0])
        for episode, value in sums[outcome].dropna().items():
            rows.append({"subject_id": subject_id, "outcome": outcome, "episode": int(episode),
                         **info.loc[episode].to_dict(), "contribution": float(value),
                         "prediction": prediction, "reference": reference})
    return pd.DataFrame(rows)


def explanation_table(tokens, patients, *, columns, gap_days=1.0, labels=None, counts=None,
                      n=3):
    """INTERNAL: for every patient and outcome, the episodes that most raise and lower risk.

    One row per patient, outcome, direction and rank, with the predicted and
    reference risk, the episode's timing and description and its contribution
    in percentage points: the format a clinical tool would show beside each
    prediction.
    """
    labels = labels or {}
    rows = []
    for subject_id, part in tokens.groupby("subject_id"):
        sums, info = _patient_episodes(part, columns, gap_days=gap_days, labels=labels,
                                       counts=counts)
        timed = sums.loc[sums.index >= 0]
        for outcome, column in columns.items():
            found = patients.loc[(patients.subject_id == subject_id)
                                 & (patients.column == column)]
            if found.empty or timed[outcome].isna().all():
                continue
            values = timed[outcome].dropna()
            base = {"subject_id": subject_id, "outcome": outcome,
                    "risk": float(found.prediction.iloc[0]),
                    "reference_risk": float(found.baseline_prediction.iloc[0])}
            for direction, picked in (("raises", values[values > 0].nlargest(n)),
                                      ("lowers", values[values < 0].nsmallest(n))):
                for rank, (episode, value) in enumerate(picked.items(), start=1):
                    rows.append({**base, "direction": direction, "rank": rank,
                                 "when": info.loc[episode, "when"],
                                 "description": info.loc[episode, "description"],
                                 "contribution_pp": float(value)})
    return pd.DataFrame(rows)


REFERENCE = "Timing of care, age and subtype"
RESIDUAL = "Numerical residual"


def variance_decomposition(tokens, patients, *, by="source", draws=1000, seed=2026,
                           min_count=20):
    """Share of the between-patient variance in predictions explained by each group.

    Completeness gives ``prediction = reference + sum_g c_g + residual`` per
    patient, so ``var(prediction) = cov(reference, prediction)
    + sum_g cov(c_g, prediction) + cov(residual, prediction)``. Each share is
    ``cov(x, prediction) / var(prediction)``; shares are signed and sum to one.
    The reference keeps the timing of care, age and subtype with average events.
    Weighted by sampling weights, with patient-bootstrap intervals.
    """
    frame = patients.drop_duplicates("subject_id").set_index("subject_id")
    ids = frame.index  # patients without attributed events contribute zero
    if len(ids) < min_count:
        return pd.DataFrame()
    contributions = (tokens.pivot_table(index="subject_id", columns=by, values="attribution",
                                        aggfunc="sum", fill_value=0.0)
                     .reindex(ids, fill_value=0.0))
    f = frame.prediction.to_numpy(float)
    base = frame.baseline_prediction.to_numpy(float)
    residual = f - base - contributions.sum(1).to_numpy()
    names = [REFERENCE, *contributions.columns, RESIDUAL]
    x = np.column_stack([base, contributions.to_numpy(float), residual])
    w = frame.weight.to_numpy(float) if "weight" in frame else np.ones(len(ids))
    rng = np.random.default_rng(seed)
    counts = rng.multinomial(len(ids), np.full(len(ids), 1 / len(ids)), size=draws)
    W = np.vstack([w[None, :], counts * w[None, :]])
    total = W.sum(1, keepdims=True)
    mean_f = (W @ f)[:, None] / total
    var_f = (W @ (f * f))[:, None] / total - mean_f ** 2
    cov = (W @ (x * f[:, None])) / total - (W @ x) / total * mean_f
    shares = cov / np.where(var_f > 0, var_f, np.nan)
    rows = [{"group": name, "share": shares[0, j],
             "share_lower": np.nanquantile(shares[1:, j], 0.025),
             "share_upper": np.nanquantile(shares[1:, j], 0.975),
             "n_patients": len(ids)} for j, name in enumerate(names)]
    return pd.DataFrame(rows)


def outcome_similarity(wide, columns, *, gap_days=1.0, min_episodes=5, min_count=20):
    """Mean rank agreement between outcomes of which episodes move a patient's risk.

    For each patient, episode contributions are ranked within each outcome and
    correlated between outcomes (Spearman); the patient correlations are
    averaged on the Fisher-z scale with sampling weights. High agreement means
    two outcomes rest on the same parts of the history; a model that had only
    learned a general measure of frailty would agree everywhere.
    """
    frame = wide.assign(episode=episode_ids(wide, gap_days))
    frame = frame.loc[frame.episode >= 0]
    sums = frame.groupby(["subject_id", "episode"])[list(columns)].sum(min_count=1)
    weights = (wide.drop_duplicates("subject_id").set_index("subject_id").weight
               if "weight" in wide else None)
    k = len(columns)
    total, mass, count = np.zeros((k, k)), np.zeros((k, k)), np.zeros((k, k))
    for subject, part in sums.groupby(level=0):
        if len(part) < min_episodes:
            continue
        values = part.to_numpy(float)
        present = ~np.isnan(values).all(0)
        if present.sum() < 2:
            continue
        ranks = pd.DataFrame(values[:, present]).rank().to_numpy()
        with np.errstate(invalid="ignore", divide="ignore"):
            rho = np.corrcoef(ranks, rowvar=False)
        block = np.ix_(np.flatnonzero(present), np.flatnonzero(present))
        w = float(weights.get(subject, 1.0)) if weights is not None else 1.0
        z = np.arctanh(np.clip(rho, -0.999, 0.999))
        ok = np.isfinite(z)
        total[block] += np.where(ok, w * z, 0)
        mass[block] += np.where(ok, w, 0)
        count[block] += ok
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = np.tanh(total / mass)
    mean[count < min_count] = np.nan
    return pd.DataFrame([{"column_a": a, "column_b": b, "spearman": mean[i, j],
                          "n_patients": int(count[i, j])}
                         for i, a in enumerate(columns) for j, b in enumerate(columns)])


__all__ = ["REFERENCE", "RESIDUAL", "episode_contributions", "outcome_similarity",
           "variance_decomposition",
           "concept_label", "episode_ids", "explanation_table",
           "patient_episodes", "when"]
