"""Borrowing map: what each source disease contributes to the locked router.

Input is one evaluation of the locked routed model R_L (no retraining) that
writes the cumulative incidence of every variant of its disease sources
(`paper_prediction_queue --source-removal`). Switching a source off replaces
its routed correction by its tuning-set mean for the patient's disease and
outcome (mean ablation, primary) or by zero (sensitivity); nothing else changes
(the other weights are not renormalised), so every variant is the same fitted
model on the same patients. The router spreads its weight widely, so zeroing
many terms would move the model far from anything it saw in training; the mean
keeps the average shift and removes only the patient-specific part:

* full: every disease source; none: other-disease terms replaced by their means;
  the patient's own disease, outcome bank and shared encoder remain active;
* in:X: own disease and X, so X's contribution on its own, which redundant
  sources cannot mask (primary; the map and the named claims);
* out:X: every source but X, so what only X provides (necessity).

Leave-one-in and leave-one-out measure two contexts, with no other borrowed
disease sources and with all of them. They do not bound a Shapley value when
source interactions are arbitrary. Read them descriptively: both positive, a main source; in only,
useful but redundant with similar sources; out only, useful in combination.

* Effects are AUC differences in AUC points (x100) on the full-risk-set IPCW
  estimand with one paired patient bootstrap (the same draws for every
  variant): dependence on borrowing = full - none; leave-one-in = in:X - none;
  leave-one-out = full - out:X.
* Per target disease, source and clinical domain the effect is averaged over
  estimable cells (>= 5 cases and >= 5 controls), each outcome weighing equally
  and each horizon equally within an outcome; "All outcomes" pools every
  outcome the same way. Domains are the paper's endpoint groups.
* The map is descriptive and partially pooled: within each domain, all pairs
  share one fitted prior centred on no borrowing (adaptive shrinkage, a
  zero-centred normal scale mixture; Stephens 2017), so noisy near-null pairs
  are pulled to zero and large precise ones kept. Lineage is deliberately not
  used in the pooling, so any lineage pattern comes from the data. The
  unshrunk estimates are kept alongside. The map's dots use the pooled
  interval (unpooled intervals would mark about one in twenty null pairs).
* Candidate pairs and draft negative controls are exploratory: the clinician
  review/analysis lock is not established. The set permutation statistic is
  retained as an exploratory diagnostic, not confirmatory evidence.
  IPA is reported as a secondary metric.
* Descriptive interval classifications for the candidate pairs and controls
  (opera/configs/borrowing_hypotheses.yaml), on the unshrunk leave-one-in
  estimate, with the prespecified labels and the plan's practical margin
  (``margin_points``, AUC points; smaller than the model-comparison margin,
  because a single source's share is a fraction of the routing gain).

Three levels put the design in context, per target disease and domain (the
same paired bootstrap; models are scored on identical patients):

* routing = R_L - J: what routing adds over the joint model (two fits);
* trained borrowing = R_L - RDH: what borrowing adds over the same router
  trained with its own disease's adapter only (two fits);
* dependence = full - none: how much the fitted router depends on borrowing
  (one fit; the encoder was trained alongside the borrowed corrections, so this
  can exceed the trained gain).

Fit-to-fit differences carry the noise floor of refitting (router decision
report); the within-model ones do not. Each target's patient count is
attached, so dependence on borrowing can be read against disease size.

The effects describe the fitted model, not biology.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from opera.evaluation.secondary_analysis import (
    CELL,
    bootstrap_p_value,
    claim_label,
    normalize_predictions,
    paired_analysis,
    patient_bootstrap,
)
from opera.run.router_decision import panel

REPO = Path(__file__).resolve().parents[2]
REGISTRY = REPO / "opera/configs/experiment_registry.yaml"
DISPLAY_GROUPS = REPO / "opera/configs/outcome_display_groups_paper_v1.yaml"
HYPOTHESES = REPO / "opera/configs/borrowing_hypotheses.yaml"
HORIZONS = (30, 60, 90, 180, 365, 730)
ALL_OUTCOMES = "All outcomes"
MARGIN_POINTS = 0.25  # default practical margin for pair claims, AUC points
MINIMUM = 5  # cases and controls per estimable cell
METRICS = ("auc", "ipa")  # AUC primary, IPA secondary
ALL_DISEASES = "All diseases"
SPARSE_CELLS = 24  # targets with fewer estimable cells are left out of summaries
LEVELS = "__levels__"  # the work unit holding routing, trained borrowing, dependence
REFERENCES = {"J": "J_L", "RDH": "RDH_L"}  # level model -> paper alias
# Short disease-group names used on the map's axes.
GROUP_SHORT = {"AMYLOIDOSIS": "Amyloid", "BL_LBL": "BL/LBL", "CLL_SLL": "CLL/SLL",
               "DLBCL_like": "LBCL", "HCL": "HCL", "HL": "HL",
               "Indolent_B_NHL": "Indolent B", "MCL": "MCL", "MM": "Plasma cell",
               "T_NHL": "T-cell"}


def disease_sizes(registry=REGISTRY):
    """Fine disease -> patients in the population (registry)."""
    registry = yaml.safe_load(Path(registry).read_text())
    return {fine: int(n) for spec in registry["cohort_groups"].values()
            for fine, n in spec["fine"].items()}


def disease_groups(registry=REGISTRY):
    registry = yaml.safe_load(Path(registry).read_text())
    return {fine: GROUP_SHORT.get(group, group)
            for group, spec in registry["cohort_groups"].items() for fine in spec["fine"]}


def endpoint_groups(path=DISPLAY_GROUPS):
    groups = yaml.safe_load(Path(path).read_text())
    return {outcome: group for group, members in groups.items() for outcome in members}


ABLATIONS = ("mean", "zero")  # mean ablation is primary; zero ablation a sensitivity


def variant_file(directory, variant, ablation="mean"):
    """CIF of one variant: a switched-off source's term replaced by its tuning
    mean (``mean``) or by zero (``zero``)."""
    if ablation not in ABLATIONS:
        raise ValueError(f"ablation must be one of {ABLATIONS}")
    return (Path(directory) / "source_removal" / ablation
            / f"{variant.replace(':', '_')}.parquet")


def wide_predictions(directory, metadata, files, *, split, horizons=HORIZONS):
    """Normalized rows of several models scored on the same patients.

    ``files`` maps a model name to a table with subject_id, outcome and
    cif_<h>d; follow-up comes from the design's direct_cr_predictions.parquet,
    and every model must cover exactly its rows.
    """
    directory = Path(directory)
    keys = ["subject_id", "outcome"]
    base = pd.read_parquet(directory / "direct_cr_predictions.parquet",
                           columns=[*keys, "time_days", "event"])
    base["subject_id"] = base.subject_id.astype(str)
    # The export's cohort is the grouped training cohort; analyses use the fine one.
    base = base.merge(
        metadata[["subject_id", "cohort_fine"]].rename(columns={"cohort_fine": "cohort"}),
        on="subject_id", how="left", validate="many_to_one")
    if base.cohort.isna().any():
        raise ValueError("Missing fine disease membership")
    columns = {f"cif_{h}d": h for h in horizons}
    parts = []
    for model, path in files.items():
        cif = pd.read_parquet(path)
        missing = set(columns) - set(cif)
        if missing:
            raise ValueError(f"{model} ({path}) lacks {sorted(missing)}")
        cif = cif[[*keys, *columns]].assign(subject_id=lambda t: t.subject_id.astype(str))
        wide = base.merge(cif, on=keys, how="left", validate="one_to_one")
        if len(cif) != len(base) or wide[list(columns)].isna().any().any():
            raise ValueError(f"{model} was not scored on exactly the design's patients")
        long = wide.melt(id_vars=list(base), var_name="horizon", value_name="probability")
        long["horizon_days"] = long.pop("horizon").map(columns)
        parts.append(long.assign(model=model))
    rows = pd.concat(parts, ignore_index=True).assign(
        split=split, seed=0, data_contract="borrowing_design")
    normalized, _ = normalize_predictions(rows, "cif_ipcw")
    return normalized


def design_predictions(directory, metadata, variants, *, split, horizons=HORIZONS,
                       ablation="mean"):
    """Normalized rows of the requested variants (model = label, ':' as '_')."""
    return wide_predictions(
        directory, metadata,
        {v.replace(":", "_"): variant_file(directory, v, ablation) for v in variants},
        split=split, horizons=horizons)


def estimable(predictions, minimum=MINIMUM):
    """Only cells with enough status-known cases and controls are bootstrapped."""
    first = predictions.loc[predictions.model == predictions.model.iloc[0]]
    known = first.known.astype(bool)
    counts = first.assign(case=known & first.label.eq(1),
                          control=known & first.label.eq(0)) \
        .groupby(CELL)[["case", "control"]].sum()
    cells = counts.loc[(counts.case >= minimum) & (counts.control >= minimum)].index
    keyed = predictions.set_index(CELL)
    return keyed.loc[keyed.index.isin(cells)].reset_index()


def pooled_effects(predictions, contrasts, domains, *, replicates=1000, seed=2026,
                   metrics=METRICS, overall=False, keep_draws=False):
    """Per contrast, metric, target and domain: the difference in points (x100).

    ``contrasts`` maps a name to an expression such as ``gain(full,none)``.
    ``overall`` adds target "All diseases": every disease weighing equally.
    ``keep_draws`` also returns the bootstrap draws (points) per
    (contrast, metric, target, domain), shared across calls with the same seed.
    """
    predictions = estimable(predictions)
    tables = paired_analysis(predictions, contrasts, replicates=replicates, seed=seed,
                             draws=patient_bootstrap(predictions, replicates, seed, method="multiplier"),
                             patient_gain_contrasts=(), min_cases=MINIMUM, min_controls=MINIMUM)
    every = tables["contrasts"].assign(
        endpoint_group=lambda t: t.outcome.map(domains))
    rows, kept = [], {}
    for metric in metrics:
        cells = every.loc[every.metric == metric]
        draws = tables["bootstrap_draws"].loc[lambda t: t.metric == metric]
        groups = list(cells.groupby(["contrast", "cohort"]))
        if overall:
            groups += [((name, ALL_DISEASES), part)
                       for name, part in cells.groupby("contrast")]
        for (name, target), part in groups:
            domains_of = {ALL_OUTCOMES: part, **dict(tuple(part.groupby("endpoint_group")))}
            for domain, subset in domains_of.items():
                result = panel(subset, draws, name, metric, minimum=MINIMUM,
                               return_draws=True)
                if not result["n_cells"]:
                    continue
                sampled = 100 * np.asarray(result["draws"], float)
                if keep_draws:
                    kept[(name, metric, target, domain)] = sampled
                rows.append({"contrast": name, "metric": metric, "target": target,
                             "endpoint_group": domain,
                             "effect": 100 * result["gain"], "lower": 100 * result["lower"],
                             "upper": 100 * result["upper"],
                             "se": float(np.nanstd(sampled, ddof=1)),
                             "p_value": bootstrap_p_value(sampled),
                             "n_cells": result["n_cells"],
                             "n_outcomes": int(subset.outcome.nunique())})
    result = pd.DataFrame(rows), every.loc[every.metric.isin(metrics)]
    return (*result, kept) if keep_draws else result


def source_effects(directory, metadata, source, domains, *, split, replicates=1000,
                   ablation="mean"):
    """Leave-one-in and leave-one-out of one source, one row per target and domain.

    AUC columns are ``in_*``/``out_*``; IPA (secondary) ``in_ipa_*``/``out_ipa_*``.
    Also returns the source-alone AUC draws per (target, domain), for summaries
    over pairs that keep the pairing of the bootstrap.
    """
    predictions = design_predictions(
        directory, metadata, ["full", "none", f"in:{source}", f"out:{source}"], split=split,
        ablation=ablation)
    pooled, cells, kept = pooled_effects(
        predictions, {"in": f"gain(in_{source},none)", "out": f"gain(full,out_{source})"},
        domains, replicates=replicates, keep_draws=True)
    draws = {(target, domain): values
             for (name, metric, target, domain), values in kept.items()
             if name == "in" and metric == "auc" and target != source}
    # A disease alone with itself is the no-borrowing model.
    pooled = pooled.loc[~((pooled.contrast == "in") & (pooled.target == source))]
    keys = ["target", "endpoint_group"]
    columns = ["effect", "lower", "upper", "se", "p_value", "n_cells"]
    wide = None
    for metric in METRICS:
        for name in ("in", "out"):
            prefix = name if metric == "auc" else f"{name}_{metric}"
            part = pooled.loc[(pooled.contrast == name) & (pooled.metric == metric),
                              [*keys, *columns]]
            part = part.rename(columns={c: f"{prefix}_{c}" for c in columns})
            wide = part if wide is None else wide.merge(part, on=keys, how="outer")
    return wide.assign(source=source), cells.assign(source=source), draws


def lineage_summary(effects, draws, groups, *, exclude=()):
    """Per clinical domain: mean source-alone gain of pairs within the target's
    disease group and across groups, and their difference.

    ``effects`` maps (target, source, domain) to a pair's estimate and ``draws``
    to its bootstrap draws (all from the same patient resamples); the estimate
    is the mean of the pairs' estimates, and each replicate averages its finite
    pairs, so the interval keeps the correlation between pairs. Targets in
    ``exclude`` (few estimable cells) are left out.
    """
    rows = []
    for domain in sorted({key[2] for key in draws}):
        kinds = {"within": [], "across": []}
        for key, values in draws.items():
            target, source, d = key
            if (d != domain or target == source or target in exclude
                    or not np.isfinite(effects.get(key, np.nan))):
                continue
            same = groups.get(target) == groups.get(source)
            kinds["within" if same else "across"].append((effects[key], values))
        points, means = {}, {}
        for kind, pairs in kinds.items():
            if not pairs:
                continue
            points[kind] = float(np.mean([e for e, _ in pairs]))
            means[kind] = np.nanmean(np.vstack([v for _, v in pairs]), axis=0)
            rows.append({"endpoint_group": domain, "kind": kind, "n_pairs": len(pairs),
                         "effect": points[kind], **_bounds(means[kind])})
        if len(means) == 2:
            difference = means["within"] - means["across"]
            rows.append({"endpoint_group": domain, "kind": "within - across",
                         "n_pairs": len(kinds["within"]) + len(kinds["across"]),
                         "effect": points["within"] - points["across"],
                         **_bounds(difference), "p_value": bootstrap_p_value(difference)})
    return pd.DataFrame(rows)


def _bounds(values):
    finite = np.asarray(values, float)
    finite = finite[np.isfinite(finite)]
    if len(finite) < 20:
        return {"lower": np.nan, "upper": np.nan}
    lower, upper = np.quantile(finite, [0.025, 0.975])
    return {"lower": float(lower), "upper": float(upper)}


def borrowing_levels(directory, metadata, domains, *, split, references=None,
                     replicates=1000, ablation="mean"):
    """Cross-disease dependence (full - none; outcome bank retained) and routing
    (full - J) and trained borrowing (full - RDH); per target and over all
    diseases."""
    references = {name: path for name, path in (references or {}).items()
                  if path is not None and Path(path).is_file()}
    files = {"full": variant_file(directory, "full", ablation),
             "none": variant_file(directory, "none", ablation), **references}
    contrasts = {"dependence": "gain(full,none)"}
    if "J" in references:
        contrasts["routing"] = "gain(full,J)"
    if "RDH" in references:
        contrasts["trained_borrowing"] = "gain(full,RDH)"
    predictions = wide_predictions(directory, metadata, files, split=split)
    return pooled_effects(predictions, contrasts, domains, replicates=replicates,
                          overall=True)


def hypothesis_test(raw, plan, *, permutations=10000, seed=2026):
    """One test of the prespecified pairs as a set (primary borrowing claim).

    Statistic: mean source-alone AUC gain of the hypothesis pairs minus the mean
    of the other off-diagonal pairs in the same domains. Null: hypotheses drawn
    at random from the other pairs of each hypothesis's domain (permutation, one
    draw per hypothesis, without replacement). Also reported for the negative
    controls, whose difference is expected to be about zero.
    """
    rng = np.random.default_rng(seed)
    pairs = raw.loc[(raw.target != raw.source) & np.isfinite(raw.in_effect)]
    rows = []
    for kind, entries in (("hypothesis", plan.get("hypotheses", [])),
                          ("negative control", plan.get("negative_controls", []))):
        named = {(e["target"], e["source"], e["domain"]) for e in entries}
        key = list(zip(pairs.target, pairs.source, pairs.endpoint_group))
        chosen = pairs.loc[[k in named for k in key]]
        if chosen.empty:
            rows.append({"kind": kind, "n_pairs": 0, "status": "not estimable"})
            continue
        others = pairs.loc[[k not in named for k in key]]
        pools = {d: others.loc[others.endpoint_group == d, "in_effect"].to_numpy()
                 for d in chosen.endpoint_group.unique()}
        counts = chosen.endpoint_group.value_counts()
        if any(len(pools[d]) < n for d, n in counts.items()):
            rows.append({"kind": kind, "n_pairs": len(chosen),
                         "status": "not estimable: insufficient domain-matched reference pairs"})
            continue
        reference = np.mean([pools[d].mean() for d in chosen.endpoint_group])
        observed = chosen.in_effect.mean() - reference
        domains_of = chosen.endpoint_group.tolist()
        counts = pd.Series(domains_of).value_counts()
        null = np.empty(permutations)
        for i in range(permutations):
            draw = [rng.choice(pools[d], size=n, replace=False) for d, n in counts.items()]
            null[i] = np.concatenate(draw).mean() - reference
        rows.append({"kind": kind, "n_pairs": len(chosen),
                     "mean_effect": chosen.in_effect.mean(), "mean_other": reference,
                     "difference": observed,
                     "p_value": float((1 + (null >= observed).sum()) / (1 + permutations)),
                     "permutations": permutations, "status": "estimated"})
    return pd.DataFrame(rows)


def adaptive_shrinkage(y, se, *, iterations=500):
    """Posterior mean and sd under a fitted zero-centred normal scale mixture.

    Adaptive shrinkage (Stephens, Biostatistics 2017): the prior is a mixture
    of N(0, sigma_k^2) on a fixed grid (sigma_0 = 0, a point mass at no effect)
    with weights fitted by maximum likelihood (EM). Many near-null pairs are
    pulled hard to zero while large, precise effects are kept, which a single
    normal prior cannot do (its fitted spread collapses when most pairs are
    null, pulling the real ones in too).
    """
    y, se = np.asarray(y, float), np.asarray(se, float)
    largest = np.sqrt(max(float(np.max(y ** 2 - se ** 2)), float(np.min(se)) ** 2))
    grid = np.concatenate([[0.0], np.geomspace(np.min(se) / 10, 2 * largest, 20)])
    variance = grid[None, :] ** 2 + se[:, None] ** 2
    likelihood = np.exp(-0.5 * y[:, None] ** 2 / variance) / np.sqrt(2 * np.pi * variance)
    weights = np.full(len(grid), 1 / len(grid))
    for _ in range(iterations):
        joint = likelihood * weights
        responsibility = joint / joint.sum(1, keepdims=True).clip(1e-300)
        updated = responsibility.mean(0)
        if np.abs(updated - weights).max() < 1e-8:
            weights = updated
            break
        weights = updated
    joint = likelihood * weights
    responsibility = joint / joint.sum(1, keepdims=True).clip(1e-300)
    factor = grid[None, :] ** 2 / variance  # component posterior: factor * y
    means = factor * y[:, None]
    variances = factor * se[:, None] ** 2
    mean = (responsibility * means).sum(1)
    second = (responsibility * (variances + means ** 2)).sum(1)
    return mean, np.sqrt(np.maximum(second - mean ** 2, 0.0)), float(
        np.sqrt((weights * grid ** 2).sum()))


def shrink(table):
    """Adaptive shrinkage of the off-diagonal pairs, per endpoint group.

    Every off-diagonal pair of a domain shares one fitted prior centred on no
    borrowing. Lineage is deliberately not used, so any lineage pattern comes
    from the data, not from the pooling. Own-disease rows and domains with
    fewer than three pairs stay unshrunk; the unshrunk values are kept.
    """
    table = table.copy()
    table["raw_effect"], table["raw_lower"], table["raw_upper"] = (
        table.effect, table.lower, table.upper)
    table["pool"] = np.where(table.target == table.source, "own", "pairs")
    table["prior_sd"] = np.nan
    for _, rows in table.groupby(["endpoint_group", "pool"]):
        rows = rows.loc[np.isfinite(rows.effect) & (rows.se > 0)]
        if len(rows) < 3 or rows.pool.iloc[0] == "own":
            continue
        mean, sd, prior = adaptive_shrinkage(rows.effect, rows.se)
        table.loc[rows.index, "effect"] = mean
        table.loc[rows.index, "lower"] = mean - 1.96 * sd
        table.loc[rows.index, "upper"] = mean + 1.96 * sd
        table.loc[rows.index, "prior_sd"] = prior
    return table


def reading(in_lower, out_lower):
    """The leave-one-in / leave-one-out pattern of one pair."""
    alone, needed = in_lower > 0, out_lower > 0
    if alone and needed:
        return "main source"
    if alone:
        return "useful, redundant with other sources"
    if needed:
        return "useful in combination"
    return "contribution not shown"


def hypothesis_table(raw, plan):
    """Prespecified pairs and negative controls, on unshrunk estimates.

    ``raw`` has ``in_*`` (leave-one-in, the claim) and ``out_*`` columns. The
    practical margin is the plan's ``margin_points`` (AUC points).
    """
    margin = float(plan.get("margin_points", MARGIN_POINTS))
    rows = []
    for kind, entries in (("hypothesis", plan.get("hypotheses", [])),
                          ("negative control", plan.get("negative_controls", []))):
        for entry in entries:
            match = raw.loc[(raw.target == entry["target"]) & (raw.source == entry["source"])
                            & (raw.endpoint_group == entry["domain"])]
            record = {"target": entry["target"], "source": entry["source"],
                      "domain": entry["domain"], "kind": kind,
                      "strength": entry.get("strength", ""),
                      "rationale": entry.get("rationale", ""),
                      "evidence": "; ".join(entry.get("evidence", []))}
            if match.empty or not np.isfinite(match.iloc[0].in_effect):
                rows.append({**record, "effect": np.nan, "lower": np.nan, "upper": np.nan,
                             "p_value": np.nan, "out_effect": np.nan, "out_lower": np.nan,
                             "out_upper": np.nan, "n_cells": 0, "claim": "not estimable",
                             "reading": "not estimable"})
                continue
            m = match.iloc[0]
            rows.append({**record, "effect": m.in_effect, "lower": m.in_lower,
                         "upper": m.in_upper, "p_value": m.in_p_value,
                         "out_effect": m.out_effect, "out_lower": m.out_lower,
                         "out_upper": m.out_upper, "n_cells": m.in_n_cells,
                         "ipa_effect": m.get("in_ipa_effect", np.nan),
                         "ipa_lower": m.get("in_ipa_lower", np.nan),
                         "ipa_upper": m.get("in_ipa_upper", np.nan),
                         # "better": the source alone raises the target's AUC.
                         "claim": claim_label(m.in_effect, m.in_lower, m.in_upper,
                                              margin),
                         "reading": reading(m.in_lower, m.out_lower)})
    return pd.DataFrame(rows)


def borrowing_map(raw, diseases):
    """Map rows: shrunk leave-one-in as ``effect``; leave-one-out alongside."""
    table = raw.assign(target_group=raw.target.map(diseases),
                       source_group=raw.source.map(diseases),
                       effect=raw.in_effect, lower=raw.in_lower, upper=raw.in_upper,
                       se=raw.in_se, n_cells=raw.in_n_cells)
    table = shrink(table)
    table["reading"] = [reading(a, b) for a, b in zip(table.in_lower, table.out_lower)]
    return table


LEVEL_LABELS = {
    "dependence": "Borrowing switched off (loss)",
    "trained_borrowing": "Over the own-disease router",
    "routing": "Routing (over the joint model)",
}


def _points(effect, lower, upper):
    if not np.isfinite(effect):
        return "Unavailable"
    if not (np.isfinite(lower) and np.isfinite(upper)):
        return f"{effect:.2f}"
    return f"{effect:.2f} ({lower:.2f} to {upper:.2f})"


def ipa_summary(levels, hypotheses):
    """AUC and IPA side by side (points, 95% intervals), for the supplement.

    Rows: the three levels over all diseases; the loss without borrowing by
    clinical domain; and each prespecified pair's gain from the source alone.
    """
    rows = []
    overall = levels.loc[levels.target == ALL_DISEASES]
    for contrast, label in LEVEL_LABELS.items():
        for domain in [ALL_OUTCOMES] + ([] if contrast != "dependence" else sorted(
                set(overall.endpoint_group) - {ALL_OUTCOMES})):
            part = overall.loc[(overall.contrast == contrast)
                               & (overall.endpoint_group == domain)]
            if part.empty:
                continue
            values = {m: part.loc[part.metric == m].iloc[0] for m in METRICS
                      if (part.metric == m).any()}
            rows.append({
                "section": "All diseases" if domain == ALL_OUTCOMES else "By clinical domain",
                "comparison": label if domain == ALL_OUTCOMES else f"{label}: {domain}",
                "auc": _points(*(values["auc"][k] for k in ("effect", "lower", "upper")))
                if "auc" in values else "Unavailable",
                "ipa": _points(*(values["ipa"][k] for k in ("effect", "lower", "upper")))
                if "ipa" in values else "Unavailable",
            })
    for row in hypotheses.itertuples():
        rows.append({
            "section": "Candidate pair" if row.kind == "hypothesis" else "Negative control",
            "comparison": f"{row.source} to {row.target}, {row.domain}",
            "auc": _points(row.effect, row.lower, row.upper),
            "ipa": _points(getattr(row, "ipa_effect", np.nan), getattr(row, "ipa_lower", np.nan),
                           getattr(row, "ipa_upper", np.nan)),
        })
    return pd.DataFrame(rows, columns=["section", "comparison", "auc", "ipa"])


def validate_plan(plan, diseases, domains):
    for entry in [*plan.get("hypotheses", []), *plan.get("negative_controls", [])]:
        unknown = {entry["target"], entry["source"]} - set(diseases)
        if unknown:
            raise ValueError(f"Unknown diseases in borrowing hypotheses: {sorted(unknown)}")
        if entry["domain"] != ALL_OUTCOMES and entry["domain"] not in set(domains.values()):
            raise ValueError(f"Unknown clinical domain: {entry['domain']!r}")


def analysis_provenance(directory, *, row, model, split, ablation, replicates,
                        membership, hypotheses, display_groups, registry, references):
    """Bind CPU caches to code, configuration and input file versions.

    Large prediction files use size/mtime signatures to avoid repeatedly hashing
    all source-removal exports. Small code/config/marker files use SHA256.
    """
    def digest(path):
        path = Path(path)
        return hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None
    def version(path):
        path = Path(path).resolve()
        if not path.is_file():
            return {'path': str(path), 'missing': True}
        stat = path.stat()
        return {'path': str(path), 'bytes': stat.st_size, 'mtime_ns': stat.st_mtime_ns}
    code = [Path(__file__), REPO / 'opera/evaluation/secondary_analysis.py',
            REPO / 'opera/run/router_decision.py']
    files = sorted(Path(directory).rglob('*.parquet'))
    return dict(schema=1, model=model, scientific_hash=row['scientific_hash'],
                run_directory=row['run_directory'], split=split, ablation=ablation,
                replicates=replicates, bootstrap_method='multiplier', bootstrap_seed=2026,
                estimand='cif_ipcw', minimum_cases=MINIMUM, minimum_controls=MINIMUM,
                horizons=list(HORIZONS),
                code_sha256={str(p.relative_to(REPO)): digest(p) for p in code},
                config_sha256={str(p): digest(p) for p in (hypotheses, display_groups, registry)},
                evaluation_marker_sha256=digest(Path(directory) / 'evaluation_complete.json'),
                inputs=[version(p) for p in [membership, *files]],
                references={name: version(p) for name, p in references.items()},
                prediction_signature='path_size_mtime_ns; not a content hash')


def validate_cache_provenance(output, provenance):
    """Never accept legacy per-source caches merely because a CSV exists."""
    path = Path(output) / 'analysis_provenance.json'
    if path.is_file():
        if json.loads(path.read_text()) != provenance:
            raise ValueError('Borrowing cache provenance changed; rerun CPU borrowing-map '
                             'in a fresh analysis root. Existing results were preserved.')
    elif any((Path(output) / 'per_source').glob('*.csv')):
        raise ValueError('Legacy borrowing cache has no analysis provenance; rerun CPU '
                         'borrowing-map in a fresh analysis root, reusing verified inference.')
    else:
        path.write_text(json.dumps(provenance, indent=2) + '\n')


def main():
    from opera.run.paper_prediction_queue import (
        bundle_runs, canonical_models, removal_directory)

    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--bundle-dir", type=Path, action="append", required=True,
                        help="Bundles holding the locked router and its references "
                             "(J_L, RDH_L); repeatable")
    parser.add_argument("--prediction-dir", type=Path, required=True)
    parser.add_argument("--removal-dir", type=Path,
                        help="Fresh borrowing-design export; reference predictions remain in --prediction-dir")
    parser.add_argument("--membership", type=Path, required=True,
                        help="Population file with subject_id and cohort_fine")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split", choices=("tuning", "held_out"), default="held_out")
    parser.add_argument("--model", default="R_L",
                        help="the primary routed fit (R_AB_L after the 2026-10-27 amendment)")
    parser.add_argument("--own-reference", default="RDH_L",
                        help="the same router with its own sources only (RBH_AB_L for R_AB_L)")
    parser.add_argument("--hypotheses", type=Path, default=HYPOTHESES)
    parser.add_argument("--display-groups", type=Path, default=DISPLAY_GROUPS)
    parser.add_argument("--registry", type=Path, default=REGISTRY)
    parser.add_argument("--replicates", type=int, default=1000)
    parser.add_argument("--ablation", choices=ABLATIONS, default="mean",
                        help="mean (primary) or zero (sensitivity; written to "
                             "<split>/zero_ablation)")
    parser.add_argument("--shard-count", type=int, default=1,
                        help="Split the per-source bootstraps over processes")
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--combine-only", action="store_true",
                        help="Only build the map from finished per-source results")
    args = parser.parse_args()

    diseases = disease_groups(args.registry)
    domains = endpoint_groups(args.display_groups)
    plan = yaml.safe_load(args.hypotheses.read_text())
    validate_plan(plan, diseases, domains)
    fits = {}
    for bundle_dir in args.bundle_dir:
        bundle, rows = bundle_runs(bundle_dir)
        for candidate, _, candidate_config in rows:
            for alias in canonical_models(candidate, candidate_config, bundle):
                fits.setdefault(alias, {})[candidate["scientific_hash"]] = (
                    candidate, candidate_config)
    if len(fits.get(args.model, {})) != 1:
        raise ValueError(f"Expected one fit for {args.model}, "
                         f"found {len(fits.get(args.model, {}))}")
    row, config = next(iter(fits[args.model].values()))
    references = {}
    for name, alias in {**REFERENCES, "RDH": args.own_reference}.items():
        if len(fits.get(alias, {})) == 1:
            reference = next(iter(fits[alias].values()))[0]
            references[name] = (args.prediction_dir / reference["run_directory"]
                                / args.split / "direct_cr_predictions.parquet")
    sources = list(config["competing_risk"].get("fine_names") or [])
    directory = args.removal_dir or removal_directory(args.prediction_dir, row["run_directory"], args.split)
    output = args.output_dir / args.split
    if args.ablation != "mean":
        output = output / f"{args.ablation}_ablation"
    (output / "per_source").mkdir(parents=True, exist_ok=True)
    provenance = analysis_provenance(
        directory, row=row, model=args.model, split=args.split, ablation=args.ablation,
        replicates=args.replicates, membership=args.membership, hypotheses=args.hypotheses,
        display_groups=args.display_groups, registry=args.registry, references=references)
    validate_cache_provenance(output, provenance)

    if not args.combine_only:
        if not (directory / "evaluation_complete.json").is_file():
            raise FileNotFoundError(f"No completed borrowing design in {directory}; run "
                                    "`run_paper_experiments.sh source-removal` first")
        metadata = (pd.read_parquet(args.membership)
                    if args.membership.suffix in {".parquet", ".pq"}
                    else pd.read_csv(args.membership))
        metadata["subject_id"] = metadata.subject_id.astype(str)
        metadata = metadata.drop_duplicates("subject_id")
        for unit in [LEVELS, *sources][args.shard_index :: args.shard_count]:
            done = output / "per_source" / f"{unit}.csv"
            if done.is_file() and (unit == LEVELS or
                    (output / "per_source" / f"{unit}_draws.npz").is_file()):
                print(f"{unit}: done")
                continue
            if unit == LEVELS:
                missing_references = [n for n, p in references.items() if not p.is_file()]
                if missing_references or len(references) < len(REFERENCES):
                    print(f"levels: no {args.split} predictions for "
                          f"{missing_references or sorted(set(REFERENCES) - set(references))};"
                          " run `evaluate` for them to add routing and trained borrowing")
                pooled, cells = borrowing_levels(directory, metadata, domains,
                                                 split=args.split, references=references,
                                                 replicates=args.replicates,
                                                 ablation=args.ablation)
            else:
                pooled, cells, draws = source_effects(directory, metadata, unit, domains,
                                                      split=args.split,
                                                      replicates=args.replicates,
                                                      ablation=args.ablation)
                keys = sorted(draws)
                np.savez_compressed(
                    output / "per_source" / f"{unit}_draws.npz",
                    targets=np.array([k[0] for k in keys]),
                    domains=np.array([k[1] for k in keys]),
                    draws=np.vstack([draws[k] for k in keys]) if keys
                    else np.empty((0, args.replicates)))
            cells.to_csv(output / "per_source" / f"{unit}_cells.csv", index=False)
            pooled.to_csv(done.with_suffix(".tmp"), index=False)
            done.with_suffix(".tmp").rename(done)
            print(f"{unit}: {len(pooled)} target x domain rows")

    finished = {unit: output / "per_source" / f"{unit}.csv" for unit in [LEVELS, *sources]}
    missing = [unit for unit, path in finished.items() if not path.is_file()]
    if missing:
        print(json.dumps({"status": "incomplete", "missing": missing}, indent=2))
        return
    sizes = disease_sizes(args.registry)
    levels = pd.read_csv(finished[LEVELS])
    levels.assign(target_group=levels.target.map(diseases),
                  n_patients=levels.target.map(sizes)).to_csv(
        output / "borrowing_levels.csv", index=False)
    raw = pd.concat([pd.read_csv(finished[s]) for s in sources], ignore_index=True)
    borrowing_map(raw, diseases).to_csv(output / "borrowing_map.csv", index=False)
    hypotheses = hypothesis_table(raw, plan)
    hypotheses["analysis_role"] = "exploratory candidate pair"
    hypotheses.to_csv(output / "borrowing_hypotheses.csv", index=False)
    pair_draws = {}
    for source in sources:
        saved = np.load(output / "per_source" / f"{source}_draws.npz")
        for target, domain, values in zip(saved["targets"], saved["domains"], saved["draws"]):
            pair_draws[(str(target), source, str(domain))] = values
    raw_all = raw.loc[raw.endpoint_group == ALL_OUTCOMES]
    cells_per_target = raw_all.groupby("target").in_n_cells.max()
    sparse = set(cells_per_target.index[cells_per_target.fillna(0) < SPARSE_CELLS])
    pair_effects = {(t, so, d): e for t, so, d, e in
                    zip(raw.target, raw.source, raw.endpoint_group, raw.in_effect)}
    lineage_summary(pair_effects, pair_draws, diseases, exclude=sparse).to_csv(
        output / "borrowing_lineage.csv", index=False)
    test = hypothesis_test(raw, plan)
    test["analysis_role"] = "exploratory; not a locked confirmatory test"
    test.to_csv(output / "borrowing_hypothesis_test.csv", index=False)
    ipa_summary(levels, hypotheses).to_csv(output / "borrowing_ipa_table.csv", index=False)
    (output / "source_removal.json").write_text(json.dumps({
        "model": args.model, "fit": row["name"], "split": args.split,
        "sources": sources, "replicates": args.replicates, "horizons": list(HORIZONS),
        "ablation": args.ablation,
        "removal": ("Other-disease terms replaced by tuning means" if args.ablation == "mean"
                    else "Other-disease terms zeroed") + "; outcome bank and shared encoder retained; weights not renormalised",
        "analysis_role": "exploratory candidate pairs; clinician review/analysis lock not established",
        "provenance": "analysis_provenance.json",
        "references": {name: str(path) for name, path in references.items()},
        "effects": {"dependence": "AUC(full) - AUC(none)",
                    "routing": f"AUC({args.model}) - AUC(J_L)",
                    "trained_borrowing": f"AUC({args.model}) - AUC({args.own_reference})",
                    "in": "AUC(own + source) - AUC(none); map and claims",
                    "out": "AUC(full) - AUC(all but source)"},
        "units": "AUC points",
        "margin_points": float(plan.get("margin_points", MARGIN_POINTS)),
        "estimable_cell_minimum": MINIMUM, "hypotheses": str(args.hypotheses),
    }, indent=2) + "\n")
    summary = hypotheses[["kind", "target", "source", "domain", "effect", "lower",
                          "upper", "out_effect", "claim", "reading"]]
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.2f}"))
    print(test.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print(f"Wrote borrowing_levels.csv, borrowing_map.csv, borrowing_hypotheses.csv, "
          f"borrowing_hypothesis_test.csv and borrowing_lineage.csv "
          f"to {output}")


if __name__ == "__main__":
    main()
