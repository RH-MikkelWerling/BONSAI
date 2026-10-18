"""Aggregate the internal attribution tables into exportable summaries.

Reads the output directory of ``opera.run.explain_direct_competing_risk`` and
the locked ``attribution_analysis`` config, and writes only aggregated tables
(patient-bootstrap intervals, suppression below ``min_count``):

``attribution_quality.csv``       completeness pass rate per explained outcome
``attribution_variance_sources.csv``, ``attribution_variance_windows.csv``
                                  every outcome: share of the differences in
                                  predicted risk between patients from each
                                  source / time window, and from the timing of
                                  care itself (log-odds)
``attribution_outcome_similarity.csv``  which outcomes rest on the same episodes
``attribution_concepts.csv``      every finding x explained outcome; the cells with a
                                  prespecified direction carry a verdict
``attribution_lab_response.csv``  latest-result curves by value bin
``attribution_top_concepts.csv``  data-driven concepts that most raise or lower risk
``attribution_faithfulness.csv``  removal checks (replace and delete)
``attribution_sanity.csv``        randomisation and second-reference agreement
``attribution_numbers.json``      scalars for the manuscript

Atlas tables (variance, findings across outcomes) are on log-odds so that
outcomes are comparable; laboratory curves, checks and explanations are in
percentage points of the predicted risk at the horizon (``scale`` column).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import warnings
from pathlib import Path

import pandas as pd
import yaml

from opera.interpretability.attribution_summaries import (
    TOKEN_COLUMNS,
    completeness_summary,
    concept_effects,
    faithfulness_summary,
    headline_numbers,
    lab_response,
    label_tokens,
    long_tokens,
    sanity_summary,
    top_concepts,
)
from opera.interpretability.code_meaning import CodeDictionary
from opera.run.explain_direct_competing_risk import column_name, parse_column, sha256

VALUE_COLUMNS = ("mean_pp", "lower_pp", "upper_pp", "median_pp")


def read_tokens(directory: Path, columns):
    import pyarrow.parquet as pq

    parts = sorted((directory / "token_attributions_INTERNAL").glob("part-*.parquet"))
    if not parts:
        raise FileNotFoundError(f"No token attribution shards under {directory}")
    names = pq.read_schema(parts[0]).names
    wanted = [c for c in (*TOKEN_COLUMNS, *columns) if c in names]
    return pd.concat(
        [pd.read_parquet(part, columns=wanted) for part in parts], ignore_index=True
    )


def _unmatched(concepts, codes):
    missing = []
    for concept in concepts:
        prefixes = tuple(concept["codes"])
        if not any(str(code).startswith(prefixes) for code in codes):
            missing.append(concept["name"])
    return missing


def on_scale(frame, column):
    """Record the unit; log-odds columns are returned in log-odds, not x100."""
    kind = parse_column(column)[0]
    frame = frame.assign(scale=kind)
    if kind == "logit":
        for name in VALUE_COLUMNS:
            if name in frame:
                frame[name] = frame[name] / 100
    return frame


def _concat(frames, columns):
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=columns)


def run(input_dir: Path, config: dict, output_dir: Path, *, horizon_days=None):
    if config.get("concept_mapping_reviewed") is False:
        warnings.warn(
            "Clinical code mappings are unreviewed: omitting named clinical concepts and lab curves"
        )
        config = {**config, "concepts": [], "lab_tests": []}
    settings = dict(config.get("summary", {}))
    # Names: reviewed labels (only once reviewed), official code tables, then
    # the built-in structure (opera/resources/code_systems).
    labels = CodeDictionary.from_config(config)
    draws = int(settings.get("bootstrap_draws", 1000))
    seed = int(settings.get("seed", 2026))
    min_count = int(settings.get("min_count", 20))
    edges = tuple(settings.get("time_windows_days", (30, 365)))
    run_info = json.loads((input_dir / "attribution_run.json").read_text())
    for name, digest in run_info.get("output_sha256", {}).items():
        if sha256(input_dir / name) != digest:
            raise ValueError(f"Attribution export changed: {name}")
    if run_info.get("complete") is False:
        raise ValueError(
            "Attribution pilot is incomplete; resume the planned sample before population summaries"
        )
    counts = json.loads((input_dir / "lab_bin_counts.json").read_text())
    explained = [c for c in run_info["columns"] if parse_column(c)[0] in {"risk", "logit"}]
    horizons = sorted({parse_column(c)[2] for c in explained})
    horizon = horizon_days if horizon_days is not None else settings.get("horizon_days")
    if horizon is None:
        if len(horizons) != 1:
            raise ValueError("Select one --horizon-days per summary; horizons must not be pooled")
        horizon = horizons[0]
    if horizon not in horizons:
        raise ValueError("Requested summary horizon was not explained")
    columns = [c for c in explained if parse_column(c)[2] == horizon]

    def column_for(outcome):
        """Absolute risk where it was explained, otherwise log-odds."""
        for kind in ("risk", "logit"):
            if column_name(kind, outcome, horizon) in columns:
                return column_name(kind, outcome, horizon)
        return None

    # Log-odds columns for comparisons across outcomes (absolute risk if a run
    # explained no log-odds).
    atlas_columns = [c for c in columns if parse_column(c)[0] == "logit"] or \
        [c for c in columns if parse_column(c)[0] == "risk"]
    tokens = read_tokens(input_dir, columns)
    patients = pd.read_parquet(input_dir / "patient_attributions_INTERNAL.parquet")
    patients = patients.loc[patients.column.isin(columns)]
    output_dir.mkdir(parents=True, exist_ok=False)
    tolerance = run_info.get("tolerance", 0.05)
    passed = (patients.completeness_ok.astype(bool) if "completeness_ok" in patients
              else patients.completeness_gap.abs() <= tolerance)
    quality = patients.assign(completeness_ok=passed)
    quality.groupby(["column", "outcome"]).completeness_ok.agg(
        n="size", passed="sum").reset_index().to_csv(
        output_dir / "attribution_quality.csv", index=False)
    valid = quality.loc[quality.completeness_ok, ["subject_id", "column"]]

    def long(column):
        accepted = valid.loc[valid.column == column, "subject_id"]
        frame = tokens.loc[tokens.subject_id.isin(accepted)]
        return label_tokens(long_tokens(frame, column, outcome=parse_column(column)[1]),
                            edges=edges)

    # Findings x outcomes. A cell with a prespecified direction is a test.
    findings = list(config.get("concepts", []))
    missing = _unmatched(findings, tokens.code.unique())
    if missing:
        warnings.warn(f"Concepts matching no attributed code: {missing}")
    rows = []
    for column in atlas_columns:
        outcome = parse_column(column)[1]
        frame = long(column)
        specs = [{**c, "outcome": outcome,
                  "expected": (c.get("expected") or {}).get(outcome)} for c in findings]
        if specs:
            rows.append(concept_effects(frame, specs, counts, draws=draws, seed=seed,
                                        min_count=min_count).pipe(on_scale, column))
    concepts = _concat(rows, ["concept", "label", "outcome", "group", "expected",
                              "n_carriers", "verdict", "mean_pp", "lower_pp", "upper_pp"])
    concepts.to_csv(output_dir / "attribution_concepts.csv", index=False)

    # What makes patients differ, by source and by time (every outcome).
    from opera.interpretability.attribution_patterns import REFERENCE, variance_decomposition

    variance = {"source": [], "window": []}
    for column in atlas_columns:
        frame = long(column)
        rows_c = quality.loc[quality.completeness_ok & (quality.column == column)]
        for by in variance:
            table = variance_decomposition(frame, rows_c, by=by, draws=draws, seed=seed,
                                           min_count=min_count)
            if len(table):
                variance[by].append(table.assign(outcome=parse_column(column)[1],
                                                 scale=parse_column(column)[0]))
    variance = {by: _concat(parts, ["group", "share", "outcome"]) for by, parts in variance.items()}
    variance["source"].to_csv(output_dir / "attribution_variance_sources.csv", index=False)
    variance["window"].to_csv(output_dir / "attribution_variance_windows.csv", index=False)

    # Which outcomes rest on the same episodes (pairs that passed completeness).
    from opera.interpretability.attribution_patterns import outcome_similarity

    masked = tokens.copy()
    for column in atlas_columns:
        accepted = valid.loc[valid.column == column, "subject_id"]
        masked.loc[~masked.subject_id.isin(accepted), column] = float("nan")
    similarity = outcome_similarity(masked, atlas_columns,
                                    gap_days=float(settings.get("episode_gap_days", 1.0)),
                                    min_count=min_count)
    similarity = similarity.assign(
        outcome_a=similarity.column_a.map(lambda c: parse_column(c)[1]),
        outcome_b=similarity.column_b.map(lambda c: parse_column(c)[1]))
    similarity.to_csv(output_dir / "attribution_outcome_similarity.csv", index=False)

    # Laboratory value curves.
    rows = []
    for test in config.get("lab_tests", []):
        column = column_for(test["outcome"])
        if column:
            rows.append(lab_response(
                long(column), [test], counts,
                within_days=float(settings.get("lab_within_days", 90)), draws=draws,
                seed=seed, min_count=min_count).pipe(on_scale, column))
    _concat(rows, ["test", "outcome", "bin", "bin_position", "mean_pp", "lower_pp",
                   "upper_pp"]).to_csv(output_dir / "attribution_lab_response.csv", index=False)

    # What else the model leans on, for each explained outcome (a table for review).
    from opera.interpretability.attribution_patterns import concept_label

    rows = []
    for column in atlas_columns:
        top = top_concepts(long(column), counts, n=int(settings.get("top_n", 10)),
                           draws=draws, seed=seed, min_count=min_count)
        if len(top):
            rows.append(top.assign(
                outcome=parse_column(column)[1],
                label=top.concept.map(lambda c: concept_label(c, labels)),
                group=top.concept.map(labels.group),
                chapter=top.concept.map(labels.chapter)).pipe(on_scale, column))
    _concat(rows, ["concept", "label", "group", "chapter", "outcome", "mean_pp"]).to_csv(
        output_dir / "attribution_top_concepts.csv", index=False)

    # Checks.
    faith = pd.read_parquet(input_dir / "faithfulness_INTERNAL.parquet")
    if len(faith):
        faith = faith.loc[faith.column.isin(columns)].merge(
            valid, on=["subject_id", "column"], validate="many_to_one")
    faithfulness = (faithfulness_summary(faith, patients, draws=draws, seed=seed,
                                         min_count=min_count)
                    if len(faith) else pd.DataFrame(columns=[
                        "column", "outcome", "perturbation", "k", "n_patients",
                        "gap_vs_random", "gap_lower", "gap_upper"]))
    faithfulness.to_csv(output_dir / "attribution_faithfulness.csv", index=False)
    sanity_rows = pd.read_parquet(input_dir / "sanity_INTERNAL.parquet")
    if len(sanity_rows):
        sanity_rows = sanity_rows.loc[sanity_rows.column.isin(columns)]
    sanity = (sanity_summary(sanity_rows) if len(sanity_rows) else
              pd.DataFrame(columns=["check", "outcome", "median", "q25", "q75", "n_patients"]))
    sanity.to_csv(output_dir / "attribution_sanity.csv", index=False)

    completeness = completeness_summary(patients, tolerance)
    numbers = headline_numbers(pd.DataFrame(), concepts, faithfulness, sanity, completeness,
                               verdict_k=int(settings.get("faithfulness_k", 5)))
    if len(variance["source"]):
        shares = variance["source"].pivot_table(index="outcome", columns="group", values="share")
        for group, key in ((REFERENCE, "ig_var_reference"), ("Laboratory", "ig_var_laboratory"),
                           ("Medication", "ig_var_medication"), ("Diagnoses", "ig_var_diagnoses")):
            if group in shares:
                numbers[key] = float(shares[group].median())
    if len(variance["window"]):
        recent = variance["window"].loc[variance["window"].group == "Last 30 days"]
        if len(recent):
            numbers["ig_var_recent"] = float(recent.share.median())
    from opera.visualization.paper_taxonomy import outcome_endpoint_groups

    domains = outcome_endpoint_groups()
    pairs = similarity.loc[(similarity.outcome_a < similarity.outcome_b)
                           & similarity.spearman.notna()]
    if len(pairs):
        same = pairs.outcome_a.map(domains) == pairs.outcome_b.map(domains)
        if same.any():
            numbers["ig_similarity_within"] = float(pairs.loc[same].spearman.median())
        if (~same).any():
            numbers["ig_similarity_between"] = float(pairs.loc[~same].spearman.median())
    numbers.update(ig_patients=int(patients.subject_id.nunique()),
                   ig_outcomes=int(patients.outcome.nunique()))
    (output_dir / "attribution_numbers.json").write_text(json.dumps({
        "numbers": numbers, "completeness": completeness,
        "run": {k: run_info.get(k) for k in ("split", "n_patients", "steps", "baseline")},
    }, indent=2) + "\n")
    (output_dir / "summary_manifest.json").write_text(json.dumps({
        "attribution_run_sha256": sha256(input_dir / "attribution_run.json"),
        "horizon_days": horizon,
        "config_sha256": hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest(),
        "output_sha256": {p.name: sha256(p) for p in output_dir.glob("*.csv")},
    }, indent=2))
    return numbers


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--horizon-days", type=float,
                        help="Required when explaining multiple horizons")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    print(json.dumps(run(args.input_dir, config, args.output_dir,
                         horizon_days=args.horizon_days), indent=2))


if __name__ == "__main__":
    main()
