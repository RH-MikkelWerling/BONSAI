"""Population summaries of event attributions, with patient-bootstrap intervals.

Input is the internal token table written by
``opera.run.explain_direct_competing_risk`` (one row per patient, outcome and
attributed token, with the attribution in absolute risk) and its patient table.
Every exported number is an average over patients with a 95% patient-bootstrap
interval, and every cell with fewer than ``min_count`` patients is suppressed.
Attributions describe the model, not causes.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

from opera.interpretability.code_meaning import FAMILIES, OTHER_SOURCE, SOURCES
from opera.interpretability.code_meaning import source_of as _source_of

# Sources follow the registers in Figure S3 and share its colours; the token
# prefixes of each source are defined once, in code_meaning.
SOURCE_FAMILIES = tuple(
    (name, tuple(prefix for prefix, source, _ in FAMILIES if source == name)) for name in SOURCES)
SOURCE_ORDER = SOURCES + (OTHER_SOURCE,)
WINDOW_ORDER = ("Last 30 days", "31 days to 1 year", "Over 1 year", "No time")
BIN_PATTERN = re.compile(r"^LABTEST//(?P<test>[^/]+)//BIN_(?P<bin>\d+)$")


def source_of(code: str) -> str:
    return _source_of(code)


def source_index(vocabulary) -> dict[int, int]:
    """Vocabulary id to source position (for per-source baselines)."""
    return {
        int(i): SOURCE_ORDER.index(source_of(name)) for name, i in vocabulary.items()
    }


def window_of(days_before, source, edges=(30, 365)) -> np.ndarray:
    days = np.asarray(days_before, float)
    labels = np.where(
        days <= edges[0],
        WINDOW_ORDER[0],
        np.where(days <= edges[1], WINDOW_ORDER[1], WINDOW_ORDER[2]),
    )
    return np.where(np.asarray(source) == "Demographics", WINDOW_ORDER[3], labels)


def bin_counts(vocabulary) -> dict[str, dict[str, int]]:
    """Observed lower/upper bin identifiers, preserving zero- or one-based indexing."""
    bounds = {}
    for name in vocabulary:
        match = BIN_PATTERN.match(str(name))
        if match:
            test, k = match["test"], int(match["bin"])
            old = bounds.get(test, {"min": k, "max": k})
            bounds[test] = {"min": min(old["min"], k), "max": max(old["max"], k)}
    return bounds


def label_tokens(tokens: pd.DataFrame, *, edges=(30, 365)) -> pd.DataFrame:
    frame = tokens.copy()
    frame["source"] = frame["code"].map(source_of)
    frame["window"] = window_of(frame["days_before_index"], frame["source"], edges)
    return frame


def _bootstrap(matrix, draws, seed, weights=None):
    """Weighted mean and 95% interval of each column, resampling rows (patients).

    ``weights`` are sampling weights (inverse inclusion probabilities); each
    replicate resamples patients with replacement and takes the weighted mean.
    """
    matrix = np.asarray(matrix, float)
    if matrix.ndim == 1:
        matrix = matrix[:, None]
    n = len(matrix)
    w = np.ones(n) if weights is None else np.asarray(weights, float)
    mean = (w[:, None] * matrix).sum(0) / w.sum()
    if n < 2 or draws < 1:
        return mean, np.full_like(mean, np.nan), np.full_like(mean, np.nan)
    rng = np.random.default_rng(seed)
    counts = rng.multinomial(n, np.full(n, 1 / n), size=draws) * w[None, :]
    resampled = (counts @ matrix) / counts.sum(1, keepdims=True)
    return mean, np.quantile(resampled, 0.025, 0), np.quantile(resampled, 0.975, 0)


TOKEN_COLUMNS = (
    "subject_id",
    "cohort",
    "subtype",
    "weight",
    "position",
    "event",
    "code",
    "days_before_index",
)


def long_tokens(tokens, column, *, outcome=None):
    """One target column of the wide token table as a long frame with ``attribution``."""
    keep = [c for c in TOKEN_COLUMNS if c in tokens]
    part = tokens[keep + [column]].rename(columns={column: "attribution"})
    part = part.loc[part.attribution.notna()].copy()
    part["outcome"] = outcome if outcome is not None else column
    return part


def _weights(frame, ids):
    if "weight" not in frame:
        return np.ones(len(ids))
    return (
        frame.drop_duplicates("subject_id")
        .set_index("subject_id")
        .weight.reindex(ids)
        .to_numpy()
    )


def source_window_summary(
    tokens, *, by=(), draws=1000, seed=2026, min_count=5, edges=(30, 365)
):
    """Share of each patient's total absolute attribution by source and window.

    Shares are computed within each patient and then averaged (weighted), so
    every patient counts once whatever the length of their history. Also
    reports the mean signed contribution in percentage points of risk. ``by``
    groups the patients (e.g. ``("subtype",)``) on top of the outcome.
    """
    frame = label_tokens(tokens, edges=edges) if "source" not in tokens else tokens
    rows = []
    keys = ["outcome", *by]
    cells = [(s, w) for s in SOURCE_ORDER for w in WINDOW_ORDER]
    for key, part in frame.groupby(keys):
        key = key if isinstance(key, tuple) else (key,)
        absolute = part.assign(a=part.attribution.abs()).pivot_table(
            index="subject_id", columns=["source", "window"], values="a", aggfunc="sum"
        )
        signed = part.pivot_table(
            index="subject_id",
            columns=["source", "window"],
            values="attribution",
            aggfunc="sum",
        )
        columns = pd.MultiIndex.from_tuples(cells)
        absolute = absolute.reindex(columns=columns).fillna(0)
        signed = signed.reindex(index=absolute.index, columns=columns).fillna(0)
        totals = absolute.sum(1).replace(0, np.nan)
        share = absolute.div(totals, axis=0).dropna()
        if len(share) < min_count:
            continue
        weights = _weights(part, share.index)
        s_mean, s_low, s_high = _bootstrap(share.to_numpy(), draws, seed, weights)
        p_mean, p_low, p_high = _bootstrap(
            100 * signed.to_numpy(), draws, seed, _weights(part, signed.index)
        )
        carriers = (absolute.loc[share.index] > 0).sum(0).to_numpy()
        for j, (source, window) in enumerate(cells):
            if not carriers[j]:
                continue
            rows.append(
                {
                    **dict(zip(keys, key)),
                    "source": source,
                    "window": window,
                    "n_patients": len(signed),
                    "n_share_patients": len(share),
                    "n_carriers": int(carriers[j])
                    if carriers[j] >= min_count
                    else np.nan,
                    "share": s_mean[j],
                    "share_lower": s_low[j],
                    "share_upper": s_high[j],
                    "signed_pp": p_mean[j],
                    "signed_pp_lower": p_low[j],
                    "signed_pp_upper": p_high[j],
                }
            )
    return pd.DataFrame(rows)


def _bin_position(code, counts):
    match = BIN_PATTERN.match(str(code))
    if not match:
        return np.nan
    specification = counts.get(match["test"], 0)
    if isinstance(specification, dict):
        low, high = specification["min"], specification["max"]
    else:  # Backward-compatible one-based legacy metadata.
        low, high = 1, specification
    if high <= low:
        return np.nan
    return (int(match["bin"]) - low) / (high - low)


def concept_tokens(tokens, concept, counts):
    """Rows of ``tokens`` belonging to one prespecified concept."""
    prefixes = tuple(concept["codes"])
    part = tokens.loc[tokens.code.astype(str).str.startswith(prefixes)]
    if concept.get("outcome"):
        part = part.loc[part.outcome == concept["outcome"]]
    if concept.get("within_days") is not None:
        part = part.loc[part.days_before_index <= float(concept["within_days"])]
    if concept.get("latest_only"):
        part = part.sort_values("position").groupby(["subject_id", "outcome"]).tail(1)
    if concept.get("bin_position") is not None:
        low, high = concept["bin_position"]
        position = part.code.map(lambda code: _bin_position(code, counts))
        part = part.loc[(position >= low) & (position <= high)]
    return part


def concept_effects(tokens, concepts, counts, *, draws=1000, seed=2026, min_count=20):
    """Mean summed attribution of each concept among the patients who carry it.

    The verdict compares the interval with the prespecified direction:
    supported (excludes 0 on the expected side), contradicted (excludes 0 on
    the other side) or inconclusive.
    """
    rows = []
    for concept in concepts:
        part = concept_tokens(tokens, concept, counts)
        per_patient = 100 * part.groupby("subject_id").attribution.sum()
        n = len(per_patient)
        row = {
            "concept": concept["name"],
            "label": concept.get("label", concept["name"]),
            "outcome": concept.get("outcome"),
            "expected": concept.get("expected"),
            "group": concept.get("group", ""),
            "n_carriers": n,
        }
        if n < min_count:
            rows.append({**row, "n_carriers": np.nan, "verdict": "too few patients"})
            continue
        mean, low, high = _bootstrap(
            per_patient.to_numpy(), draws, seed, _weights(part, per_patient.index)
        )
        row.update(
            mean_pp=mean[0],
            lower_pp=low[0],
            upper_pp=high[0],
            median_pp=float(per_patient.median()),
            share_raising=float((per_patient > 0).mean()),
        )
        expected = {"+": 1, "-": -1}.get(str(concept.get("expected")), 0)
        if expected and low[0] > 0:
            row["verdict"] = "supported" if expected > 0 else "contradicted"
        elif expected and high[0] < 0:
            row["verdict"] = "supported" if expected < 0 else "contradicted"
        else:
            row["verdict"] = "inconclusive" if expected else "descriptive"
        rows.append(row)
    return pd.DataFrame(rows)


def lab_response(
    tokens, tests, counts, *, within_days=90, draws=1000, seed=2026, min_count=20
):
    """Attribution of each patient's latest result of a test, by its value bin.

    One row per test, outcome and bin; the latest result within ``within_days``
    of the index date, as a clinician would read "the LDH before treatment".
    """
    rows = []
    for test in tests:
        code = f"LABTEST//{test['npu']}//BIN_"
        part = tokens.loc[
            tokens.code.astype(str).str.startswith(code)
            & (tokens.days_before_index <= within_days)
        ]
        if test.get("outcome"):
            part = part.loc[part.outcome == test["outcome"]]
        latest = part.sort_values("position").groupby(["subject_id", "outcome"]).tail(1)
        latest = latest.assign(
            bin=latest.code.str.rsplit("BIN_", n=1).str[1].astype(int)
        )
        # Bin range from the vocabulary (zero- or one-based); without it, the
        # range observed in these results.
        observed = ({"min": int(latest.bin.min()), "max": int(latest.bin.max())}
                    if len(latest) else {"min": 0, "max": 0})
        span = {test["npu"]: counts.get(test["npu"]) or observed}
        bounds = span[test["npu"]]
        if not isinstance(bounds, dict):  # legacy one-based bin count
            bounds = {"min": 1, "max": int(bounds)}
        for (outcome, k), group in latest.groupby(["outcome", "bin"]):
            n = group.subject_id.nunique()
            row = {
                "test": test.get("label", test["npu"]),
                "npu": test["npu"],
                "outcome": outcome,
                "bin": int(k),
                "n_bins": bounds["max"] - bounds["min"] + 1,
                "bin_position": _bin_position(f"{code}{int(k)}", span),
                "n_patients": n,
            }
            if n < min_count:
                rows.append({**row, "n_patients": np.nan})
                continue
            mean, low, high = _bootstrap(
                100 * group.attribution.to_numpy(),
                draws,
                seed,
                _weights(group, group.subject_id),
            )
            rows.append(
                {**row, "mean_pp": mean[0], "lower_pp": low[0], "upper_pp": high[0]}
            )
    return pd.DataFrame(rows)


def faithfulness_summary(
    faithfulness, patients, *, draws=1000, seed=2026, min_count=20
):
    """Drop in the prediction when events are removed, by how they were chosen.

    ``drop = prediction - prediction_after`` in percentage points. Removing the
    events with the largest attributions should lower the prediction more than
    removing random events; ``gap_vs_random`` is that paired difference.
    """
    base = patients.set_index(["subject_id", "column"])[["prediction", "weight"]]
    frame = faithfulness.drop(
        columns=[c for c in ("weight",) if c in faithfulness]
    ).join(base, on=["subject_id", "column"])
    frame["drop_pp"] = 100 * (frame.prediction - frame.prediction_after)
    per_patient = (
        frame.groupby(
            ["column", "outcome", "perturbation", "k", "ranking", "subject_id"]
        )
        .drop_pp.mean()
        .unstack("ranking")
    )
    weights = frame.drop_duplicates("subject_id").set_index("subject_id").weight
    rows = []
    for (column, outcome, perturbation, k), part in per_patient.groupby(
        level=[0, 1, 2, 3]
    ):
        part = part.dropna(subset=[c for c in ("top", "random") if c in part])
        if len(part) < min_count:
            continue
        w = weights.reindex(part.index.get_level_values("subject_id")).to_numpy()
        row = {
            "column": column,
            "outcome": outcome,
            "perturbation": perturbation,
            "k": int(k),
            "n_patients": len(part),
        }
        for ranking in ("top", "random", "bottom"):
            if ranking in part:
                mean, low, high = _bootstrap(part[ranking].to_numpy(), draws, seed, w)
                row.update(
                    {
                        f"{ranking}_pp": mean[0],
                        f"{ranking}_lower": low[0],
                        f"{ranking}_upper": high[0],
                    }
                )
        if {"top", "random"} <= set(part):
            mean, low, high = _bootstrap(
                (part.top - part.random).to_numpy(), draws, seed, w
            )
            row.update(gap_vs_random=mean[0], gap_lower=low[0], gap_upper=high[0])
        rows.append(row)
    return pd.DataFrame(rows)


def sanity_summary(sanity):
    """Rank agreement of attributions with a randomised or re-baselined model."""
    return (
        sanity.groupby(["check", "outcome"])
        .spearman.agg(
            median="median",
            q25=lambda s: s.quantile(0.25),
            q75=lambda s: s.quantile(0.75),
            n_patients="count",
        )
        .reset_index()
    )


def completeness_summary(patients, tolerance=0.05):
    gap = patients.completeness_gap.abs()
    return {
        "n_patient_outcomes": int(len(patients)),
        "median_relative_gap": float(gap.median()),
        "p95_relative_gap": float(gap.quantile(0.95)),
        "share_above_tolerance": float((gap > tolerance).mean()),
        "tolerance": tolerance,
    }


def profile_agreement(left, right, *, on=("concept",), value="mean_pp"):
    """Spearman agreement of two concept tables (e.g. joint vs routed, L vs RI)."""
    merged = left.merge(right, on=list(on), suffixes=("_a", "_b")).dropna(
        subset=[f"{value}_a", f"{value}_b"]
    )
    if len(merged) < 3:
        return np.nan
    return float(merged[f"{value}_a"].rank().corr(merged[f"{value}_b"].rank()))


def headline_numbers(
    sources, concepts, faithfulness, sanity, completeness, *, verdict_k=5
):
    """Scalars for the manuscript (``\\result`` keys), aggregated only."""
    out = {}
    tested = concepts.loc[
        concepts.verdict.isin(["supported", "contradicted", "inconclusive"])
    ]
    out["ig_expect_total"] = int(len(tested))
    out["ig_expect_supported"] = int((tested.verdict == "supported").sum())
    out["ig_expect_contradicted"] = int((tested.verdict == "contradicted").sum())
    if len(sources):
        by_source = (
            sources.groupby(["outcome", "source"]).share.sum().groupby("source").mean()
        )
        by_window = (
            sources.groupby(["outcome", "window"]).share.sum().groupby("window").mean()
        )
        out["ig_share_laboratory"] = float(by_source.get("Laboratory", np.nan))
        out["ig_share_medication"] = float(by_source.get("Medication", np.nan))
        out["ig_share_recent"] = float(by_window.get(WINDOW_ORDER[0], np.nan))
    primary = (
        faithfulness.loc[
            (faithfulness.perturbation == "replace") & (faithfulness.k == verdict_k)
        ]
        if len(faithfulness)
        else faithfulness
    )
    if len(primary):
        out["ig_faithful_outcomes"] = int((primary.gap_lower > 0).sum())
        out["ig_faithful_total"] = int(len(primary))
        out["ig_gap_min"] = float(primary.gap_vs_random.min())
        out["ig_gap_max"] = float(primary.gap_vs_random.max())
    if len(sanity):
        for check, part in sanity.groupby("check"):
            out[f"ig_rho_{check}"] = float(part["median"].median())
    out["ig_completeness_p95"] = completeness["p95_relative_gap"]
    return out


__all__ = [
    "SOURCE_FAMILIES",
    "SOURCE_ORDER",
    "TOKEN_COLUMNS",
    "WINDOW_ORDER",
    "bin_counts",
    "completeness_summary",
    "long_tokens",
    "concept_effects",
    "concept_tokens",
    "faithfulness_summary",
    "headline_numbers",
    "lab_response",
    "label_tokens",
    "profile_agreement",
    "rollup",
    "sanity_summary",
    "source_index",
    "top_concepts",
    "source_of",
    "source_window_summary",
    "window_of",
]


def rollup(code: str) -> str:
    """Clinical concept level for data-driven summaries.

    ICD-10 to three characters (``DIAGNOSIS//DC83``), ATC to the
    pharmacological subgroup (``MEDICATION//J01C``), procedures to four
    characters, pathology SNOMED to three, laboratory results to the test and
    its value third (``LABTEST//NPU02319//low``); contacts are kept as they are.
    """
    code = str(code)
    match = BIN_PATTERN.match(code)
    if match:
        return f"LABTEST//{match['test']}//BIN"
    for prefix, width in (
        ("DIAGNOSIS//", 4),
        ("MEDICATION//", 4),
        ("PROCEDURE//", 4),
        ("PATHOLOGY//", 3),
    ):
        if code.startswith(prefix):
            value = code[len(prefix):]
            # Pathology: morphology and topography codes form families (M96:
            # lymphomas); other axes, such as molecular findings (TP53), are
            # specific and kept whole.
            if prefix == "PATHOLOGY//" and not value[:1] in {"M", "T"}:
                return code
            return prefix + value[:width]
    return code


def _lab_third(code, counts):
    position = _bin_position(code, counts)
    if np.isnan(position):
        return ""
    return "low" if position < 1 / 3 else ("high" if position > 2 / 3 else "mid")


def top_concepts(tokens, counts, *, n=10, draws=1000, seed=2026, min_count=20):
    """Concepts that most consistently raise or lower one column's predictions.

    Per patient, attributions are summed within each rolled-up concept (for a
    laboratory test, its latest result's value third); concepts carried by at
    least ``min_count`` patients are ranked by their weighted mean and the
    ``n`` most risk-raising and ``n`` most risk-lowering are returned.
    """
    frame = tokens.copy()
    frame["concept"] = frame.code.map(rollup)
    labs = frame.concept.str.endswith("//BIN")
    if labs.any():
        latest = (
            frame.loc[labs]
            .sort_values("position")
            .groupby(["subject_id", "concept"])
            .tail(1)
        )
        latest = latest.assign(
            concept=latest.concept.str[:-3]
            + latest.code.map(lambda c: _lab_third(c, counts))
        )
        frame = pd.concat([frame.loc[~labs], latest])
    per = frame.groupby(["concept", "subject_id"]).attribution.sum().mul(100)
    rows = []
    for concept, part in per.groupby(level=0):
        if len(part) < min_count:
            continue
        ids = part.index.get_level_values(1)
        mean, low, high = _bootstrap(part.to_numpy(), draws, seed, _weights(frame, ids))
        rows.append(
            {
                "concept": concept,
                "source": source_of(concept),
                "n_carriers": len(part),
                "mean_pp": mean[0],
                "lower_pp": low[0],
                "upper_pp": high[0],
                "share_raising": float((part > 0).mean()),
            }
        )
    table = pd.DataFrame(rows)
    if table.empty:
        return table
    table = table.sort_values("mean_pp")
    picked = pd.concat(
        [
            table.tail(n).assign(direction="raises"),
            table.head(n).assign(direction="lowers"),
        ]
    )
    return picked.drop_duplicates("concept").sort_values("mean_pp", ascending=False)
