"""Render every explicitly registered score/population from locked native CIFs.

Preview validates configuration only; --execute creates a fresh server-side
output directory. Clinical applicability/cutoffs must be locked before test use.
"""

from __future__ import annotations
import argparse
import json
import os
import re
from pathlib import Path
import numpy as np
import pandas as pd
import yaml
from opera.run.analyse_locked_predictions import file_hash, read_artifact, read_table
from opera.run.clinical_figure_provenance import render_receipt


def _score_rank(spec):
    """Ordinal score from the category order when no numeric score exists."""
    return {group: rank for rank, group in enumerate(spec["group_order"], start=1)}


def high_score_threshold(spec):
    """Score at or above which the clinical index calls a patient high risk."""
    if spec.get("score_column"):
        return spec["score_high_threshold"]
    return _score_rank(spec)[spec.get("score_high_group", spec["group_order"][-1])]


def score_patients(predictions, metadata, spec, split):
    """Patients of one score population with their model risk and score group.

    Metadata may carry a numeric score with its own date, or, as in the
    DALY-CARE release, only the index used (``index_column``) and its risk
    category (``group_column``); the category's rank then serves as the score.
    Patients scored with a different index are not part of this population.
    """
    selected = predictions.loc[
        (predictions.model == spec["model"])
        & (predictions.seed == spec["seed"])
        & (predictions.split == split)
        & (predictions.outcome == spec["outcome"])
        & (predictions.horizon_days == spec["horizon_days"])
        & predictions.cohort.isin(spec["cohorts"])
    ].copy()
    if selected.subject_id.duplicated().any():
        raise ValueError("Duplicate score-evaluation patient IDs")
    optional = [spec.get(key) for key in ("score_column", "score_date_column",
                                          "index_date_column", "index_column")]
    needed = ["subject_id", spec["group_column"], spec["age_column"],
              *[column for column in optional if column]]
    if not set(needed) <= set(metadata):
        raise ValueError("Missing registered score/date/age metadata columns")
    data = selected.merge(
        metadata[list(dict.fromkeys(needed))],
        on="subject_id",
        validate="one_to_one",
        how="left",
    )
    available = data[spec["group_column"]].notna()
    if spec.get("index_column"):
        available &= data[spec["index_column"]].isin(spec["index_values"])
    if spec.get("score_column"):
        available &= data[spec["score_column"]].notna()
    if spec.get("score_date_column"):
        score_date = pd.to_datetime(data[spec["score_date_column"]], errors="coerce")
        index_date = pd.to_datetime(data[spec["index_date_column"]], errors="coerce")
        available &= score_date.notna() & index_date.notna() & (score_date <= index_date)
    subset = data.loc[available].copy()
    subset = subset.rename(
        columns={
            spec["group_column"]: "ipi_group",
            spec["age_column"]: "age",
            "probability": "model_risk",
            **({spec["score_column"]: "ipi_score"} if spec.get("score_column") else {}),
        }
    )
    # Registry categories vary in case and padding ("Medium ", "high").
    group_map = {str(k).strip().lower(): v for k, v in spec["group_map"].items()}
    subset["ipi_group"] = subset.ipi_group.astype(str).str.strip().str.lower().map(group_map)
    if subset.ipi_group.isna().any():
        raise ValueError("Unmapped score category; do not silently drop it")
    if not spec.get("score_column"):
        subset["ipi_score"] = subset.ipi_group.map(_score_rank(spec))
    if not {"time_days", "event"} <= set(subset):
        raise ValueError("Score survival plots require full native time/event records")
    if not np.isfinite(
        subset[["time_days", "event", "ipi_score", "model_risk", "age"]].to_numpy(float)
    ).all():
        raise ValueError("Missing/nonfinite score, follow-up or age")
    return subset, {
        "eligible_n": len(data),
        "score_available_at_index_n": len(subset),
        "unavailable_or_late_n": int((~available).sum()),
    }


CUTOFF_RULES = ("size_matched", "risk_matched_low")


def clinical_weights(patients, horizon):
    """CIF labels and disease-specific censoring weights on the score population."""
    from opera.evaluation.secondary_analysis import attach_ipcw_weights
    data = patients.copy().reset_index(drop=True)
    for key, value in dict(model="clinical", seed=0, split="clinical", cohort="all",
                           outcome="clinical", horizon_days=horizon).items():
        if key not in data:
            data[key] = value
    data["horizon_days"] = horizon
    return attach_ipcw_weights(data)


def classification_at_boundaries(patients, spec, cuts, evaluation_split):
    """Both ordinary known-status and IPCW confusion metrics at frozen cutoffs."""
    from opera.evaluation.operating_points import threshold_metrics
    data = clinical_weights(patients, spec["horizon_days"])
    labels = data.label.to_numpy(bool)
    rank = data.ipi_group.map({g: i for i, g in enumerate(spec["group_order"])}).to_numpy()
    rows = []
    for i, (group, cut) in enumerate(zip(spec["group_order"][1:], cuts), 1):
        for system, flag in (("model", data.model_risk.to_numpy() >= cut),
                             ("index", rank >= i)):
            for estimand, weights in (("known_status", data.known.to_numpy(float)),
                                      ("cif_ipcw", data.ipcw_weight.to_numpy(float))):
                tp, fp, fn, tn = [float(weights[mask].sum()) for mask in
                    (flag & labels, flag & ~labels, ~flag & labels, ~flag & ~labels)]
                metrics = {k: float(v) for k,v in threshold_metrics(tp,fp,fn,tn).items()}
                denominator = tp+fp+fn+tn
                rows.append(dict(score_id=spec["id"], model=spec["model"], split=evaluation_split,
                    horizon_days=spec["horizon_days"], threshold_split=spec.get("threshold_split", "train"),
                    system=system, boundary=group, estimand=estimand,
                    threshold=float(cut) if system == "model" else i,
                    threshold_units="predicted_risk" if system == "model" else "zero_based_group_rank",
                    n_patients=len(data), n_known=int(data.known.sum()), tp=tp,fp=fp,fn=fn,tn=tn,
                    recall=metrics["sensitivity"], precision=metrics["ppv"],
                    accuracy=(tp+tn)/denominator if denominator else np.nan,
                    f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else np.nan, **metrics))
    return rows


def select_model_cutoffs(tuning, spec, *, minimum_group=20):
    """Model risk-group cutoffs developed on the designated development split.

    ``risk_matched_low`` (default; as in the earlier DLBCL study) anchors both
    ends of the stratification to the index:

    * low cutoff: the largest model low-risk group whose observed risk is no
      higher than the index's low-risk group (at least ``minimum_group``);
    * high cutoff: the model calls high risk at the specificity of the index's
      high-risk group, so both flag the same share of patients without events;
    * middle cutoffs (e.g. NCCN-IPI's two intermediate groups): the patients
      between the two are split in the proportions of the index's middle groups.

    ``size_matched`` gives the model groups exactly the index's group sizes.
    """
    from opera.visualization.paper_figures import _known_status

    rule = spec.get("cutoff_rule", "risk_matched_low")
    if rule not in CUTOFF_RULES:
        raise ValueError(f"cutoff_rule must be one of {CUTOFF_RULES}")
    order = list(spec["group_order"])
    shares = tuning.ipi_group.value_counts(normalize=True).reindex(order).fillna(0)
    if tuning.empty:
        raise ValueError("No threshold-development predictions available")
    risk = tuning.model_risk.to_numpy(float)
    if rule == "size_matched":
        cuts = [float(np.quantile(risk, q)) for q in shares.cumsum().to_numpy()[:-1]]
    else:
        weighted = clinical_weights(tuning, spec["horizon_days"])
        known = weighted.loc[weighted.known].copy()
        model_risk = known.model_risk.to_numpy(float)
        events = known.label.to_numpy()
        weights = known.ipcw_weight.to_numpy(float)
        if len(np.unique(events)) != 2:
            raise ValueError("Threshold development requires cases and controls")
        low_group = (known.ipi_group == order[0]).to_numpy()
        if not low_group.any():
            raise ValueError("No index low-risk patients for threshold development")
        low_rate = float(np.average(events[low_group], weights=weights[low_group]))
        candidates = np.unique(np.quantile(model_risk, np.linspace(0.01, 0.99, 197)))
        feasible = [c for c in candidates
                    if (model_risk < c).sum() >= minimum_group
                    and np.average(events[model_risk < c], weights=weights[model_risk < c]) <= low_rate]
        if not feasible:
            raise ValueError("No model low-risk cutoff matches the training index event risk")
        low = float(max(feasible))
        high_group = (known.ipi_group == order[-1]).to_numpy()
        if not high_group.any():
            raise ValueError("No index highest-risk patients for threshold development")
        specificity = float(np.average(~high_group[events == 0], weights=weights[events == 0]))
        control_risk, control_weights = model_risk[events == 0], weights[events == 0]
        sort = np.argsort(control_risk)
        cumulative = np.cumsum(control_weights[sort]) / control_weights.sum()
        high = float(control_risk[sort[min(np.searchsorted(cumulative, specificity), len(sort)-1)]])
        middle = shares.iloc[1:-1].to_numpy()
        between = np.sort(risk[(risk >= low) & (risk < high)])
        inner = ([float(np.quantile(between, q)) for q in np.cumsum(middle)[:-1] / middle.sum()]
                 if len(middle) > 1 and len(between) and middle.sum() > 0 else [])
        cuts = [low, *inner, high]
    rounded = [round(float(c), 6) for c in cuts]
    if (len(rounded) != len(order) - 1 or not np.isfinite(rounded).all()
            or any(right <= left for left, right in zip(rounded, rounded[1:]))):
        raise ValueError("Model risk-group cutoffs are not estimable: matched anchors "
                         "cross or intermediate thresholds coincide; no artificial separation applied")
    assigned = np.digitize(risk, rounded)
    if any(not np.any(assigned == i) for i in range(len(order))):
        raise ValueError("Model risk-group cutoffs produce an empty development group; "
                         "stratification is not estimable under the specified matching rule")
    return rounded


def stratification_summary(patients, spec, cuts, *, top_fractions=(0.05, 0.10)):
    """How each system stratifies held-out patients with known horizon status.

    Per group of the index and of the model: size, share, observed risk, and
    the sensitivity and specificity of calling this group or higher "at risk".
    The cutoffs fix the comparison at the index's operating points; the
    ``top`` rows show what the model finds beyond them, the observed risk in
    its highest-risk 5% and 10% of patients (the index cannot rank within
    its top group, so its row is the high-risk group itself).
    """
    from opera.visualization.paper_figures import _known_status

    known, labels = _known_status(patients, spec["horizon_days"])
    events = labels.to_numpy().astype(bool)
    order = list(spec["group_order"])
    model_group = pd.cut(known.model_risk, [-np.inf, *cuts, np.inf],
                         labels=order, right=False)
    rows = []
    for system, groups in (("index", known.ipi_group), ("model", model_group)):
        rank = groups.map({g: i for i, g in enumerate(order)}).astype(float).to_numpy()
        for i, group in enumerate(order):
            inside, at_or_above = rank == i, rank >= i
            rows.append(dict(
                score_id=spec["id"], system=system, group=group,
                event_rate=float(events.mean()),
                n=int(inside.sum()), share=float(inside.mean()),
                observed_risk=float(events[inside].mean()) if inside.any() else np.nan,
                sensitivity_at_or_above=float(at_or_above[events].mean()),
                specificity_at_or_above=float((~at_or_above[~events]).mean())))
    risk = known.model_risk.to_numpy(float)
    for fraction in top_fractions:
        top = risk >= np.quantile(risk, 1 - fraction)
        rows.append(dict(
            score_id=spec["id"], system="model", group=f"top {fraction:.0%}",
            n=int(top.sum()), share=float(top.mean()),
            observed_risk=float(events[top].mean()),
            sensitivity_at_or_above=float(top[events].mean()),
            specificity_at_or_above=float((~top[~events]).mean())))
    return rows


def score_availability(metadata, *, cohort_column, index_column, group_column):
    """Aggregate counts of the recorded index and category in each disease."""
    data = metadata[[cohort_column, index_column, group_column]].fillna("not recorded")
    return (data.groupby([cohort_column, index_column, group_column]).size()
            .rename("n_patients").reset_index())


MANUSCRIPT_NAMES = {"risk_groups": "fig_risk_groups",
                    "clinical_score": "fig_clinical_score",
                    "clinical_score_age": "supp_clinical_score_age",
                    "decision_curve": "fig_decision_curve"}
DEFAULT_DCA_THRESHOLDS = [round(0.05 + 0.01 * i, 2) for i in range(36)]  # 5% to 40%


def with_comparators(held, predictions, spec, models, split="held_out"):
    """Held-out score patients with each comparator model's risk as a column.

    Restricted to patients every comparator predicted, so all strategies are
    compared on the same patients; returns the frame and how many were dropped.
    """
    data = held
    for model in models:
        other = predictions.loc[
            (predictions.model == model) & (predictions.seed == spec["seed"])
            & (predictions.split == split) & (predictions.outcome == spec["outcome"])
            & (predictions.horizon_days == spec["horizon_days"])
            & predictions.cohort.isin(spec["cohorts"]),
            ["subject_id", "probability"]].rename(columns={"probability": model})
        if other.subject_id.duplicated().any():
            raise ValueError(f"Duplicate {model} predictions for score patients")
        data = data.merge(other, on="subject_id", how="inner", validate="one_to_one")
    return data, len(held) - len(data)


def native_entries(specs, native):
    """A second comparison per score on the endpoint the index was built for.

    ``native`` maps an index name (its ``index_values`` entry or score label)
    to the closest panel outcome and the original endpoint, e.g.
    ``{"IPI": {outcome: overall_survival, outcome_label: Overall survival,
    death_in_endpoint: true, original_endpoint: "overall survival, 5 years"}}``.
    The horizon is the longest of ``horizons_days`` the follow-up supports.
    """
    entries = []
    for spec in specs:
        keys = [str(v) for v in spec.get("index_values") or []] + [str(spec.get("score_label"))]
        mapping = next((native[key] for key in keys if key in native), None)
        if mapping is None:
            continue
        entries.append({
            **spec,
            **{key: value for key, value in mapping.items() if key != "horizons_days"},
            "id": f"{spec['id']}_native",
            "endpoint_role": "native",
            "horizon_days": max(mapping.get("horizons_days", [730])),
            "horizons_days": sorted(mapping.get("horizons_days", [730]), reverse=True),
            "manuscript_figures": False,
        })
    return entries


FOLLOWUP_SHARE = 1 / 3  # lock amendment 2026-10-26


def followup_horizon(held, spec):
    """Longest prespecified horizon that at least a third of these patients could
    be followed for (lock amendment 2026-10-26; IPCW handles the censoring, so
    the criterion is precision).

    Uses only follow-up (reverse Kaplan-Meier on the overall-survival record),
    never outcomes; returns the horizon and the follow-up summary.
    """
    from opera.evaluation.followup import median_followup, reverse_km_quantiles

    summary = median_followup(held.time_days, held.event)
    # The time by which two thirds have reached the end of their follow-up.
    limit = reverse_km_quantiles(held.time_days, held.event,
                                 quantiles=(1 - FOLLOWUP_SHARE,))[1 - FOLLOWUP_SHARE]
    summary["followup_one_third_days"] = limit
    options = spec.get("horizons_days") or [spec["horizon_days"]]
    if not np.isfinite(limit):
        return max(options), summary  # more than a third still followed at the end
    supported = [h for h in options if h <= limit]
    return (max(supported) if supported else min(options)), summary


def population_followup(predictions, held, spec):
    """Median follow-up of a score population from its overall-survival records."""
    from opera.evaluation.followup import median_followup

    survival = predictions.loc[
        (predictions.outcome == "overall_survival") & (predictions.model == spec["model"])
        & (predictions.seed == spec["seed"]) & predictions.subject_id.isin(held.subject_id)
    ].drop_duplicates("subject_id")
    if survival.empty or not {"time_days", "event"} <= set(survival):
        return {"median_followup_days": np.nan, "followup_source": "unavailable"}
    follow = median_followup(survival.time_days, survival.event)
    return {"median_followup_days": follow["median_days"],
            "followup_q25_days": follow["q25_days"],
            "followup_q75_days": follow["q75_days"],
            "followup_source": "overall_survival_reverse_km"}


def expand_index_scores(config, metadata):
    """One score entry per recorded prognostic index and disease.

    ``all_indices`` in the score config holds a ``template`` (model, endpoint,
    group mapping, provenance, ...) and optional per-index ``overrides``. Every
    value of the index column recorded for at least ``minimum_patients``
    patients of a disease becomes an entry, unless an explicit entry already
    covers that index and disease. ``manuscript`` names the index and disease
    that supply the main-text figures.
    """
    block = config.get("all_indices")
    explicit = []
    for spec in config.get("scores", []):
        if spec.get("cohorts") == "all" and spec.get("group_column") not in metadata:
            # The score was not computed for this metadata (e.g. the registry
            # IPI not merged yet): kept, and reported as missing per score below.
            spec = {**spec, "cohorts": []}
        elif spec.get("cohorts") == "all":
            # Every subtype with at least one patient scored.
            scored = metadata.loc[metadata[spec["group_column"]].notna(), "cohort_fine"]
            spec = {**spec, "cohorts": sorted(scored.dropna().unique()),
                    "population_label": spec.get("population_label", "all subtypes")}
        explicit.append(spec)
    if not block:
        return explicit
    index_column = block.get("index_column", "index")
    cohort_column = block.get("cohort_column", "cohort_fine")
    group_column = block["template"].get("group_column", "IPI")
    if not {index_column, group_column, cohort_column} <= set(metadata):
        return explicit  # no recorded index in this metadata: explicit scores only
    # ``pooled`` (default true) adds one comparison per index across every
    # subtype in which it is recorded.
    covered = {(str(v), c) for spec in explicit
               for v in (spec.get("index_values") or []) for c in spec["cohorts"]}
    recorded = metadata.loc[metadata[index_column].notna() & metadata[group_column].notna()]
    counts = recorded.groupby([index_column, cohort_column]).size()
    manuscript = block.get("manuscript") or {}
    generated = []
    per_cohort = block.get("per_cohort", True)
    for (value, cohort), n in (counts.items() if per_cohort else []):
        if n < block.get("minimum_patients", 50) or (str(value), cohort) in covered:
            continue
        slug = re.sub(r"[^a-z0-9]+", "_", f"{value}_{cohort}".lower()).strip("_")
        spec = {**block["template"], **(block.get("overrides", {}).get(str(value), {}))}
        spec.update(id=slug, score_label=str(value), cohorts=[cohort],
                    index_column=index_column, index_values=[value],
                    group_column=group_column,
                    manuscript_figures=(str(value) == str(manuscript.get("index"))
                                        and cohort == manuscript.get("cohort")))
        generated.append(spec)
    if block.get("pooled", True):
        # Each index used in several subtypes is also compared once across all
        # of them (e.g. the IPI across the aggressive and rare lymphomas),
        # including subtypes too small for their own comparison.
        totals = counts.groupby(level=0).sum()
        for value, cohorts in counts.groupby(level=0):
            names = sorted(cohorts.index.get_level_values(1))
            if (per_cohort and len(names) < 2) or totals[value] < block.get("minimum_patients", 50):
                continue
            slug = re.sub(r"[^a-z0-9]+", "_", f"{value}_all_subtypes".lower()).strip("_")
            spec = {**block["template"], **(block.get("overrides", {}).get(str(value), {}))}
            spec.update(id=slug, score_label=str(value), cohorts=names,
                        population_label="all subtypes", index_column=index_column,
                        index_values=[value], group_column=group_column,
                        manuscript_figures=(not per_cohort and
                                            str(value) == str(manuscript.get("index"))))
            generated.append(spec)
    specs = explicit + generated
    native = native_entries(specs, config.get("native_endpoints") or {})
    if config.get("index_endpoint", "native") == "native":
        # Indices are judged on the endpoint they were developed for: the native
        # comparison replaces the shared endpoint and takes over the main figures.
        replaced = {entry["id"].removesuffix("_native") for entry in native}
        flags = {spec["id"]: spec.get("manuscript_figures", False) for spec in specs}
        for entry in native:
            entry["manuscript_figures"] = flags[entry["id"].removesuffix("_native")]
        return [spec for spec in specs if spec["id"] not in replaced] + native
    return specs + native


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-config", type=Path, required=True)
    parser.add_argument("--score-config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split", choices=("held_out", "tuning"), default="held_out",
                        help="tuning only for rehearsing the pipeline")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    analysis = yaml.safe_load(args.analysis_config.read_text())
    config = yaml.safe_load(args.score_config.read_text())
    metadata = read_table(analysis["patient_metadata"])
    metadata.subject_id = metadata.subject_id.astype(str)
    metadata = metadata.rename(columns=analysis.get("patient_metadata_column_map", {}))
    specs = expand_index_scores(config, metadata)
    if not specs:
        raise ValueError("Register every score/population in score config first")
    names = [s["id"] for s in specs]
    if len(set(names)) != len(names) or any(
        Path(n).name != n or n in {".", ".."} for n in names
    ):
        raise ValueError("Score IDs must be unique safe directory names")
    for spec in specs:
        required = [
            "model",
            "cohorts",
            "outcome",
            "group_column",
            "age_column",
            "group_map",
            "group_order",
            "score_label",
            "outcome_label",
        ]
        if any(not spec.get(key) for key in required):
            raise ValueError(
                "Resolve all score/population/outcome/date mappings before rendering"
            )
        if spec.get("index_column") and not spec.get("index_values"):
            raise ValueError("Name the index values that define this score population")
        if spec.get("score_date_column") and not spec.get("index_date_column"):
            raise ValueError("A score date needs the index date to compare against")
        if not spec.get("score_date_column") and not spec.get("score_timing"):
            raise ValueError(
                "Without a score date, state when the score is recorded (score_timing)"
            )
        if not 2 <= len(spec["group_order"]) <= 4:
            raise ValueError(
                "Currently supports original score populations with 2–4 groups"
            )
        if not spec.get("lock_provenance") or not spec.get("applicability_review"):
            raise ValueError(
                "Each score needs cutoff-lock and applicability provenance"
            )
        if spec.get("cutoff_rule"):
            if spec["cutoff_rule"] not in CUTOFF_RULES:
                raise ValueError(f"cutoff_rule must be one of {CUTOFF_RULES}")
            continue  # cutoffs are chosen on the declared development split during rendering
        cuts = spec["model_cutoffs"]
        if (
            len(cuts) != len(spec["group_order"]) - 1
            or not all(0 < x < 1 for x in cuts)
            or cuts != sorted(set(cuts))
        ):
            raise ValueError("Cutoffs must be increasing, with one fewer than groups")
    if not args.execute:
        print(
            json.dumps(
                {
                    "score_jobs": names,
                    "output_dir": str(args.output_dir),
                    "preview": True,
                }
            )
        )
        return
    if args.output_dir.exists():
        raise FileExistsError("Use a fresh output directory for this locked report")
    if metadata.subject_id.duplicated().any():
        raise ValueError("Patient metadata must be unique")
    tables = []
    hashes = {}
    wanted = {s["model"] for s in specs} | {
        m for s in specs for m in (s.get("decision_curve") or {}).get("comparators", [])}
    for artifact in [*analysis["artifacts"], *analysis.get("clinical_threshold_artifacts", [])]:
        if artifact["model"] not in wanted:
            continue
        frame, _ = read_artifact(
            artifact,
            metadata,
            "cif_ipcw",
        )
        tables.append(frame)
        hashes[str(artifact["path"])] = file_hash(artifact["path"])
    if not tables:
        raise ValueError("No registered model prediction artifacts")
    predictions = pd.concat(tables, ignore_index=True)
    from opera.visualization import paper_figures as figures
    from opera.visualization.style import save_fig as _save_fig

    mark = os.environ.get("OPERA_PAPER_FIGURE_MARK")
    layout_records = []

    def save_fig(fig, path):
        """Save, with the preview mark (OPERA_PAPER_FIGURE_MARK) drawn once."""
        if mark and not getattr(fig, "_opera_marked", False):
            figures.watermark(fig, mark)
            fig._opera_marked = True
        _save_fig(fig, path)
        from opera.visualization.layout_audit import audit_layout
        layout_records.append({'file': str(Path(path).relative_to(args.output_dir)),
                               **audit_layout(fig)})
    import matplotlib.pyplot as plt

    args.output_dir.mkdir(parents=True)
    records = []
    group_rows = []
    stratification = []
    classification = []
    overview = []
    decision_tables = []
    for spec in specs:
        record = {"score_id": spec["id"]}
        try:
            if spec.get("endpoint_role") == "native":
                # Native endpoint: the longest prespecified horizon the test
                # patients' follow-up supports (outcome-blind).
                longest = {**spec, "horizon_days": max(spec["horizons_days"])}
                held, _ = score_patients(predictions, metadata, longest, args.split)
                horizon, follow = followup_horizon(held, spec)
                spec = {**spec, "horizon_days": horizon}
                record.update(endpoint_role="native", horizon_days=horizon,
                              original_endpoint=spec.get("original_endpoint"),
                              median_followup_days=follow["median_days"],
                              followup_q25_days=follow["q25_days"],
                              followup_q75_days=follow["q75_days"])
            held, audit = score_patients(predictions, metadata, spec, args.split)
            development_split = spec.get("threshold_split", "train")
            if development_split == args.split:
                raise ValueError("Threshold development and evaluation must use different splits")
            tuning, _ = score_patients(predictions, metadata, spec, development_split)
            if set(held.subject_id) & set(tuning.subject_id):
                raise ValueError("Threshold development and evaluation patients overlap")
            record.update(threshold_split=development_split, threshold_n=len(tuning),
                          threshold_estimand="cif_ipcw", threshold_prediction_kind="apparent_training" if development_split == "train" else "development_split")
            record.update(audit)
            if "median_followup_days" not in record:
                record.update(population_followup(predictions, held, spec))
            if len(held) < spec.get("minimum_patients", 20):
                record["status"] = "insufficient_support"
                records.append(record)
                continue
            cuts = (select_model_cutoffs(tuning, spec) if spec.get("cutoff_rule")
                    else list(spec["model_cutoffs"]))
            record["model_cutoffs"] = json.dumps(cuts)
            record["cutoff_rule"] = spec.get("cutoff_rule", "fixed_in_config")
            classification.extend(classification_at_boundaries(held, spec, cuts, args.split))
            weighted_development = clinical_weights(tuning, spec["horizon_days"])
            known = weighted_development.loc[weighted_development.known]
            labels = known.label
            if labels.nunique() != 2:
                raise ValueError(
                    "Training operating-point selection requires cases/controls"
                )
            specificity = float(
                (
                    known.loc[labels == 0, "ipi_score"] < high_score_threshold(spec)
                ).astype(float).dot(known.loc[labels == 0, "ipcw_weight"])
                / known.loc[labels == 0, "ipcw_weight"].sum()
            )
            low_development = known.loc[known.ipi_group == spec["group_order"][0]]
            record.update(development_n_known=len(known),
                development_n_cases=int(labels.sum()),
                development_n_controls=int((labels == 0).sum()),
                development_low_risk=float(np.average(low_development.label,
                    weights=low_development.ipcw_weight)) if len(low_development) else None)
            threshold = cuts[-1]
            directory = args.output_dir / spec["id"]
            directory.mkdir()
            common = dict(
                outcome_label=spec["outcome_label"],
                horizon_days=spec["horizon_days"],
                thresholds=tuple(cuts),
                ipi_label=spec["score_label"],
                ipi_order=tuple(spec["group_order"]),
                event_free=spec["death_in_endpoint"],
            )
            for name, fn in [
                ("risk_groups", figures.figure_risk_groups),
                ("clinical_score", figures.figure_clinical_score),
                ("clinical_score_age", figures.figure_clinical_score_age),
            ]:
                extra = (
                    dict(
                        model_threshold=threshold, ipi_high=high_score_threshold(spec)
                    )
                    if name == "clinical_score"
                    else {}
                )
                fig = fn(held, **common, **extra)
                save_fig(fig, str(directory / (name + ".png")))
                if spec.get("manuscript_figures"):
                    # The score shown in the main text, under its manuscript key.
                    save_fig(fig, str(args.output_dir / (MANUSCRIPT_NAMES[name] + ".png")))
                plt.close(fig)
            held["model_group"] = pd.cut(
                held.model_risk,
                [-np.inf, *cuts, np.inf],
                labels=spec["group_order"],
                right=False,
            )
            for system in ["model_group", "ipi_group"]:
                for group in spec["group_order"]:
                    part = held.loc[held[system] == group]
                    value = (
                        figures.cumulative_incidence(
                            part.time_days.to_numpy(),
                            part.event.to_numpy(),
                            [spec["horizon_days"]],
                        )[0]
                        if len(part)
                        else np.nan
                    )
                    group_rows.append(
                        dict(
                            score_id=spec["id"],
                            system=system,
                            group=group,
                            n=len(part),
                            at_risk_horizon_n=int(
                                (part.time_days >= spec["horizon_days"]).sum()
                            ),
                            estimate=1 - value if spec["death_in_endpoint"] else value,
                        )
                    )
            if spec.get("decision_curve"):
                # Net benefit of acting on the model, the index and comparators,
                # at thresholds and for an action fixed before testing.
                from opera.evaluation.decision_curves import (
                    decision_curves, figure_decision_curve, operating_point_rows)

                dca = spec["decision_curve"]
                comparators = list(dca.get("comparators", []))
                paired, dropped = with_comparators(held, predictions, spec, comparators,
                                                   split=args.split)
                # The model's risk-group cutoffs (fixed on tuning data) are the
                # prespecified operating points: net benefit is computed there.
                points = {"Low-risk cutoff": float(cuts[0]),
                          "High-risk cutoff": float(cuts[-1])}
                thresholds = sorted(set(map(float, dca.get(
                    "thresholds", DEFAULT_DCA_THRESHOLDS))) | set(points.values()))
                curves = decision_curves(
                    paired, tuning, spec["horizon_days"], thresholds,
                    comparators=comparators, n_boot=dca.get("n_boot", 1000))
                curves.insert(0, "score_id", spec["id"])
                curves.to_csv(directory / "decision_curve.csv", index=False)
                operating_point_rows(curves, points).to_csv(
                    directory / "decision_curve_operating_points.csv", index=False)
                decision_tables.append(curves)
                record.update(dca_patients=len(paired), dca_dropped_no_comparator=dropped,
                              dca_action=dca.get("action", ""))
                fig = figure_decision_curve(
                    curves, index_label=spec["score_label"],
                    labels=dca.get("labels", {}),
                    outcome_label=dca.get("outcome_label", spec["outcome_label"].lower()),
                    horizon_days=spec["horizon_days"], points=points)
                save_fig(fig, str(directory / "decision_curve.png"))
                if spec.get("manuscript_figures"):
                    save_fig(fig, str(args.output_dir / (MANUSCRIPT_NAMES["decision_curve"] + ".png")))
                plt.close(fig)
            stratification.extend(stratification_summary(
                held, spec, cuts,
                top_fractions=tuple(spec.get("top_fractions", (0.05, 0.10)))))
            overview.append({
                "title": f"{spec['score_label']} · "
                         f"{spec.get('population_label', ', '.join(spec['cohorts']))}",
                "endpoint": spec["outcome_label"], "patients": held,
                "thresholds": tuple(cuts),
                "ipi_order": tuple(spec["group_order"]),
                "horizon_days": spec["horizon_days"],
                "event_free": spec["death_in_endpoint"]})
            record.update(
                status="rendered",
                development_operating_threshold=threshold,
                development_target_specificity=specificity,
            )
        except (ValueError, KeyError) as exc:
            record.update(status="blocked", reason=str(exc))
        records.append(record)
    # One detailed plate per rendered index for the supplement; the manuscript
    # inputs this file from figures/results/clinical_scores/ when it exists.
    plates = [r"% Generated by opera.run.render_clinical_score_panels; do not edit."]
    for spec, record in zip(specs, records):
        if record.get("status") != "rendered":
            continue
        population = spec.get("population_label", ", ".join(spec["cohorts"]))
        years = spec["horizon_days"] // 365
        plates.append(
            "\n\\begin{figure}[p]\n"
            f"  \\resultfigure{{clinical_scores/{spec['id']}/clinical_score}}"
            f"{{{spec['score_label']} in {population}}}{{}}\n"
            f"  \\caption{{\\textbf{{The model against the {spec['score_label']} in "
            f"{population}.}} As \\cref{{fig:clinical-score}}, for "
            f"{spec['outcome_label'].lower()} within {years} years in patients with the "
            f"{spec['score_label']} recorded at diagnosis.}}\n"
            f"  \\label{{fig:clinical-score-{spec['id'].replace('_', '-')}}}\n"
            "\\end{figure}")
    (args.output_dir / "supplement_clinical_scores.tex").write_text(
        "\n".join(plates) + "\n")
    if overview:
        # Every rendered index on one page, for the supplement.
        # Four indices per page leave room for native four-category labels.
        for page, start in enumerate(range(0, len(overview), 4), start=1):
            fig = figures.figure_score_overview(overview[start:start + 4])
            suffix = "" if page == 1 else f"_{page}"
            save_fig(fig, str(args.output_dir / f"supp_score_overview{suffix}.png"))
            plt.close(fig)
    pd.DataFrame(classification).to_csv(args.output_dir / "classification_at_thresholds.csv", index=False)
    (args.output_dir / 'layout_report.json').write_text(json.dumps(layout_records, indent=2))
    pd.DataFrame(records).to_csv(args.output_dir / "score_inventory.csv", index=False)
    availability = config.get("availability")
    if availability:
        score_availability(metadata, **availability).to_csv(
            args.output_dir / "score_availability.csv", index=False)
    pd.DataFrame(group_rows).to_csv(
        args.output_dir / "risk_group_point_estimates.csv", index=False
    )
    # Group sizes, observed risk, sensitivity and specificity per system, and
    # the model's top-risk fractions (supplement table).
    pd.DataFrame(stratification).to_csv(
        args.output_dir / "stratification_diagnostics.csv", index=False)
    # One row per index for the headline figure: events captured by the
    # high-risk group of the index and of the model (at the index's specificity).
    headline = []
    strata = pd.DataFrame(stratification)
    for spec, record in zip(specs, records):
        if record.get("status") != "rendered":
            continue
        high = strata.loc[(strata.score_id == spec["id"])
                          & (strata.group == spec["group_order"][-1])].set_index("system")
        low = strata.loc[(strata.score_id == spec["id"])
                         & (strata.group == spec["group_order"][0])].set_index("system")
        rate = float(high.loc["index", "event_rate"])
        gained = (high.loc["model", "sensitivity_at_or_above"]
                  - high.loc["index", "sensitivity_at_or_above"])
        headline.append({
            "score_id": spec["id"],
            "model": spec["model"], "split": args.split,
            "threshold_split": spec.get("threshold_split", "train"),
            "group_order": json.dumps(spec['group_order']),
            "classification_estimand": "known_status",
            "n_known": int(strata.loc[(strata.score_id == spec['id']) &
                                       strata.system.eq('index'), 'n'].sum()),
            "n_events": int(round(rate * strata.loc[(strata.score_id == spec['id']) &
                                                     strata.system.eq('index'), 'n'].sum())),
            "label": f"{spec['score_label']} · "
                     f"{spec.get('population_label', ', '.join(spec['cohorts']))}",
            "index_sensitivity": high.loc["index", "sensitivity_at_or_above"],
            "model_sensitivity": high.loc["model", "sensitivity_at_or_above"],
            "specificity": high.loc["index", "specificity_at_or_above"],
            "model_specificity": high.loc["model", "specificity_at_or_above"],
            "event_rate": rate,
            # Per 1,000 patients at the frozen training-derived thresholds:
            # patients with the event that the model's high-risk group adds.
            "extra_events_flagged_per_1000": 1000 * rate * gained,
            # Low-risk groups: size per 1,000 and observed risk in each.
            "index_low_per_1000": 1000 * low.loc["index", "share"],
            "model_low_per_1000": 1000 * low.loc["model", "share"],
            "index_low_observed_risk": low.loc["index", "observed_risk"],
            "model_low_observed_risk": low.loc["model", "observed_risk"],
            "manuscript": bool(spec.get("manuscript_figures"))})
    pd.DataFrame(headline).to_csv(args.output_dir / "headline_scores.csv", index=False)
    if decision_tables:
        pd.concat(decision_tables).to_csv(args.output_dir / "decision_curves.csv", index=False)
    (args.output_dir / "report_manifest.json").write_text(
        json.dumps(
            dict(
                **render_receipt(args.output_dir, specs, args.split),
                analysis_config_sha=file_hash(args.analysis_config),
                score_config_sha=file_hash(args.score_config),
                metadata_sha=file_hash(analysis["patient_metadata"]),
                prediction_hashes=hashes,
            ),
            indent=2,
        )
    )
    if any(r["status"] == "blocked" for r in records):
        raise SystemExit("Some scores blocked; see score_inventory.csv")


if __name__ == "__main__":
    main()
