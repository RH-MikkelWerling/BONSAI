"""One row per clinical prognostic index: OPERA against the index at both ends.

Reads only the aggregate tables written by
``opera.run.render_clinical_score_panels`` (no patient rows) and derives four
signed comparisons per score population (``score_id``), each oriented so that
a positive value favours the model:

``low_size`` (a)
    Equal group sizes, lowest group: survival of the model's lowest stratum
    minus the index's lowest category at the index's horizon, in percentage
    points (``size_matched_survival_comparisons.csv``, first native group).
``high_size`` (b)
    Equal group sizes, highest group: survival of the index's highest category
    minus the model's highest stratum (same table, last native group). For a
    competing-risk endpoint the table holds cumulative incidence, and both
    signs are flipped so that "positive favours the model" still holds.
``low_risk`` (c)
    Equal risk, low end: additional patients in the model's low-risk group
    (frozen development cutoff matched to the survival of the index's lowest
    category) as a percentage of patients, with the count
    (``km_matched_low_groups.csv``: ``additional_low_per_1000`` / 10 and
    ``model_low_n - index_low_n``).
``high_risk`` (d)
    Equal risk, high end: additional events captured per 1,000 patients by the
    model's high-risk group at the index's specificity
    (``clinical_operating_point_counts.csv``: ``additional_events_per_1000``).

Intervals are the pipeline's paired patient-bootstrap percentiles, transformed
with the point estimate. When the size-matched tables are absent (the
``risk_matched_low`` recipe writes no paired comparisons), (c) and (d) fall
back to ``headline_scores.csv`` point estimates without intervals, and (a) and
(b) are reported as not estimable with that reason.

Per end the analysis-lock rule (30 October 2026) applies: *at least as good*
when the point estimate is >= 0, *better* when the paired 95% interval
excludes zero in the model's favour. The same rule is applied descriptively to
(c) and (d).
"""

from __future__ import annotations

import json
import re

import numpy as np
import pandas as pd

MEASURES = ("low_size", "high_size", "low_risk", "high_risk")
MEASURE_LABELS = {
    "low_size": "Lowest group, equal size",
    "high_size": "Highest group, equal size",
    "low_risk": "Low-risk group, equal risk",
    "high_risk": "High-risk group, equal specificity",
}
VERDICTS = ("better", "at_least_as_good", "not_at_least_as_good", "not_estimable")

# Short, reader-facing reasons for the pipeline's failure codes.
FAILURE_REASONS = {
    "crossed_anchors": "low and high training cutoffs crossed",
    "two_group_anchor_conflict": "two categories, but the training anchors disagree",
    "coincident_or_missing_middle_cutoffs": "intermediate training cutoffs coincide",
    "empty_model_group": "a model group is empty on training data",
    "no_low_risk_match": "no model group as low-risk as the index's on training data",
    "missing_cases_or_controls": "no events or no event-free patients on training data",
    "missing_low_group": "no lowest-category patients on training data",
    "missing_high_group": "no highest-category patients on training data",
    "empty_development_population": "no training predictions for this population",
    "insufficient_distinct_scores": "too few distinct model scores",
    "insufficient_native_low_followup": "too little follow-up in the lowest category",
    "no_supported_km_match": "no supported model group as low-risk as the index's",
}


def _json_list(value):
    if isinstance(value, (list, tuple)):
        return list(value)
    try:
        parsed = json.loads(value)
        return list(parsed) if isinstance(parsed, (list, tuple)) else []
    except (TypeError, ValueError):
        return []


def _finite(*values):
    try:
        return bool(np.isfinite(np.asarray(values, float)).all())
    except (TypeError, ValueError):
        return False


def _float(value):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return np.nan
    return value


def _text(value):
    return value if isinstance(value, str) and value.strip() else None


def verdict(point, lower):
    """Analysis-lock label for one end: better / at least as good / neither."""
    if not _finite(point):
        return "not_estimable"
    if _finite(lower) and lower > 0:
        return "better"
    return "at_least_as_good" if point >= 0 else "not_at_least_as_good"


def short_reason(code=None, reason=None, *, limit=64):
    """A failure code's reader-facing reason, else the pipeline text, shortened."""
    if isinstance(code, str) and code in FAILURE_REASONS:
        return FAILURE_REASONS[code]
    text = _text(reason)
    if not text:
        return "not estimable"
    text = re.split(r"(?<=[a-z0-9\)])[.;:] ", text.strip(), maxsplit=1)[0].rstrip(".")
    text = text[0].lower() + text[1:] if text[:2] != text[:2].upper() else text
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


def _taxonomy():
    try:
        from opera.visualization.paper_taxonomy import (
            FINE_LONG, GROUP_ORDER, disease_name, fine_cohort_groups)
        groups = fine_cohort_groups()
    except Exception:  # pragma: no cover - the registry is part of the repo
        return {}, (), [], (lambda code: code)
    return groups, GROUP_ORDER, list(FINE_LONG), disease_name


def _split_label(label, score_id):
    """Index name and any population qualifier ("IPI (computed), lymphomas ...")."""
    label = _text(label) or re.sub(r"_native$", "", str(score_id)).replace("_", " ").upper()
    label = re.sub(r"\s*·.*$", "", label)
    name, _, qualifier = label.partition(", ")
    return name.strip(), qualifier.strip()


def _horizon_text(days):
    days = _float(days)
    if not np.isfinite(days):
        return ""
    return f"{days / 365:g} y" if days % 365 == 0 else f"{days:g} d"


def _rows_for(table, score_id):
    if table is None or not len(table) or "score_id" not in table:
        return pd.DataFrame()
    return table.loc[table.score_id.astype(str).eq(str(score_id))]


def _first(frame):
    return frame.iloc[0] if len(frame) else None


def summarise_clinical_indices(inventory, size_matched=None, km_low=None, detection=None,
                               headline=None, *, estimand="cif_ipcw"):
    """Per-index table of the four comparisons (a–d) with intervals and labels.

    Every row of ``inventory`` (``score_inventory.csv``) is kept: a population
    that was not rendered or whose comparison is not estimable keeps its row
    with ``status`` and ``reason``. Returns columns ``score_id``, ``index``,
    ``qualifier``, ``disease``, ``disease_order``, ``pooled``, ``endpoint``,
    ``horizon_days``, ``n_patients``, ``status``, ``reason`` and per measure
    ``<m>``, ``<m>_lower``, ``<m>_upper``, ``<m>_verdict``, ``<m>_reason``,
    ``<m>_source``, plus ``low_risk_count``, ``both_ends_size_verdict``.
    """
    if inventory is None or not len(inventory):
        return pd.DataFrame(columns=["score_id", "status"])
    groups_of, group_order, fine_order, disease_name = _taxonomy()
    detection = detection if detection is not None else pd.DataFrame()
    if len(detection) and "estimand" in detection:
        detection = detection.loc[detection.estimand.eq(estimand)]
    rows = []
    for record in inventory.drop_duplicates("score_id").itertuples(index=False):
        record = record._asdict()
        score_id = str(record["score_id"])
        status = str(record.get("status") or "unknown")
        sizes = _rows_for(size_matched, score_id)
        km = _first(_rows_for(km_low, score_id))
        found = _first(_rows_for(detection, score_id))
        head = _first(_rows_for(headline, score_id))
        label = record.get("score_label")
        if not _text(label) and found is not None:
            label = found.get("label")
        if not _text(label) and head is not None:
            label = head.get("label")
        index, qualifier = _split_label(label, score_id)
        cohorts = _json_list(record.get("cohorts"))
        population = _text(record.get("population_label"))
        pooled = len(cohorts) != 1
        if not pooled:
            disease = disease_name(cohorts[0])
        elif qualifier:
            disease = f"{qualifier[0].upper()}{qualifier[1:]} ({len(cohorts)})"
        elif population:
            disease = f"{population[0].upper()}{population[1:]} ({len(cohorts)})"
        else:
            disease = f"{len(cohorts)} subtypes" if cohorts else "Unspecified"
        # Clinical disease-group order (paper_taxonomy.GROUP_ORDER), then subtype;
        # comparisons pooled over several subtypes come last.
        if pooled:
            order = 1000 * (len(group_order) + 1)
        else:
            code = cohorts[0]
            group = groups_of.get(code)
            order = (1000 * (list(group_order).index(group) if group in group_order
                             else len(group_order))
                     + (fine_order.index(code) if code in fine_order else len(fine_order)))
        horizon = _float(record.get("horizon_days"))
        if not np.isfinite(horizon) and len(sizes) and "horizon_days" in sizes:
            horizon = _float(sizes.horizon_days.iloc[0])
        if not np.isfinite(horizon) and found is not None:
            horizon = _float(found.get("horizon_days"))
        endpoint = _text(record.get("outcome_label")) or (
            _text(found.get("outcome")) if found is not None else None) or ""
        n = _float(record.get("score_available_at_index_n"))
        if not np.isfinite(n) and len(sizes) and "index_n" in sizes:
            n = float(sizes.index_n.sum())
        reason = None
        if status != "rendered":
            reason = short_reason(record.get("cutoff_failure_code"), record.get("reason"))
            if status == "insufficient_support":
                reason = "too few test patients with the index recorded"
        row = dict(score_id=score_id, index=index, qualifier=qualifier, disease=disease,
                   disease_order=order, pooled=pooled, endpoint=endpoint,
                   horizon_days=horizon, n_patients=n, status=status, reason=reason)
        for measure in MEASURES:
            row.update({measure: np.nan, f"{measure}_lower": np.nan,
                        f"{measure}_upper": np.nan, f"{measure}_reason": None,
                        f"{measure}_source": None})
        row["low_risk_count"] = np.nan
        order_names = _json_list(record.get("group_order"))
        if status == "rendered":
            # (a), (b): paired size-matched comparison at the index's horizon.
            if len(sizes) and order_names:
                for measure, group, sign in (("low_size", order_names[0], 1.0),
                                             ("high_size", order_names[-1], -1.0)):
                    match = _first(sizes.loc[sizes.group.astype(str).eq(str(group))])
                    if match is None or not _finite(match.get("difference")):
                        row[f"{measure}_reason"] = "group empty in the test population"
                        continue
                    # Survival tables: positive difference = model survives longer.
                    flip = -1.0 if str(match.get("estimand")) == "cumulative_incidence" else 1.0
                    scale = 100 * sign * flip
                    low, high = sorted([scale * _float(match.get("lower")),
                                        scale * _float(match.get("upper"))])
                    row.update({measure: scale * float(match["difference"]),
                                f"{measure}_lower": low if _finite(low, high) else np.nan,
                                f"{measure}_upper": high if _finite(low, high) else np.nan,
                                f"{measure}_source": "size_matched_survival_comparisons"})
            else:
                for measure in ("low_size", "high_size"):
                    rule = head.get("grouping_rule") if head is not None else None
                    row[f"{measure}_reason"] = (
                        f"equal-size comparison not run ({rule})" if _text(rule)
                        and rule != "cohort_size_matched" else "equal-size comparison not run")
            # (c): KM-matched low-risk group at a frozen development cutoff.
            if km is not None and str(km.get("status")) == "estimable" \
                    and _finite(km.get("additional_low_per_1000")):
                row.update(low_risk=float(km["additional_low_per_1000"]) / 10,
                           low_risk_lower=_float(km.get("additional_low_per_1000_lower")) / 10,
                           low_risk_upper=_float(km.get("additional_low_per_1000_upper")) / 10,
                           low_risk_source="km_matched_low_groups")
                if _finite(km.get("model_low_n"), km.get("index_low_n")):
                    row["low_risk_count"] = int(km["model_low_n"]) - int(km["index_low_n"])
            elif km is not None:
                row["low_risk_reason"] = short_reason(None, km.get("reason"))
            elif head is not None and _finite(head.get("model_low_per_1000"),
                                              head.get("index_low_per_1000")):
                # risk_matched_low recipe: shares among known-status patients.
                gained = float(head["model_low_per_1000"]) - float(head["index_low_per_1000"])
                row.update(low_risk=gained / 10, low_risk_source="headline_scores")
                if _finite(head.get("n_known")):
                    row["low_risk_count"] = int(round(gained * float(head["n_known"]) / 1000))
            else:
                row["low_risk_reason"] = "no low-risk match reported"
            # (d): events captured at the index's specificity.
            if found is not None and _finite(found.get("additional_events_per_1000")):
                row.update(high_risk=float(found["additional_events_per_1000"]),
                           high_risk_lower=_float(found.get("lower")),
                           high_risk_upper=_float(found.get("upper")),
                           high_risk_source=f"clinical_operating_point_counts:{estimand}")
            elif head is not None and _finite(head.get("extra_events_flagged_per_1000")):
                row.update(high_risk=float(head["extra_events_flagged_per_1000"]),
                           high_risk_source="headline_scores:known_status")
            else:
                row["high_risk_reason"] = "no specificity-matched cutoff"
        for measure in MEASURES:
            if not _finite(row[f"{measure}_lower"], row[f"{measure}_upper"]):
                row[f"{measure}_lower"] = row[f"{measure}_upper"] = np.nan
            row[f"{measure}_verdict"] = verdict(row[measure], row[f"{measure}_lower"])
            if row[f"{measure}_verdict"] == "not_estimable" and not row[f"{measure}_reason"]:
                row[f"{measure}_reason"] = reason or "not estimable"
        ends = [row["low_size_verdict"], row["high_size_verdict"]]
        row["both_ends_size_verdict"] = (
            "not_estimable" if "not_estimable" in ends else
            "better" if ends == ["better", "better"] else
            "at_least_as_good" if "not_at_least_as_good" not in ends else "not_at_least_as_good")
        if status == "rendered" and all(row[f"{m}_verdict"] == "not_estimable" for m in MEASURES):
            row["status"] = "not_estimable"
            row["reason"] = row["low_size_reason"]
        rows.append(row)
    table = pd.DataFrame(rows)
    table = table.sort_values(["pooled", "disease_order", "disease", "index", "score_id"],
                              kind="stable")
    return table.reset_index(drop=True)


def across_index_counts(summary):
    """Counts per measure: indices compared, better, at least as good, not estimable."""
    counts = {}
    if summary is None or not len(summary):
        return counts
    for measure in (*MEASURES, "both_ends_size"):
        labels = summary[f"{measure}_verdict"]
        estimable = int(labels.ne("not_estimable").sum())
        counts[measure] = dict(
            total=int(len(labels)), estimable=estimable,
            better=int(labels.eq("better").sum()),
            at_least_as_good=int(labels.isin(["better", "at_least_as_good"]).sum()),
            not_at_least_as_good=int(labels.eq("not_at_least_as_good").sum()),
            not_estimable=int(labels.eq("not_estimable").sum()))
    return counts
