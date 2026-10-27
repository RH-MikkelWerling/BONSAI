"""Turn locked analysis outputs into the plot tables the paper figures read.

Every table is a translation of an existing analysis output: contrast codes
from the contrast manifest become figure names (P1 -> "D-S", T2 -> "J-X", ...),
model codes become readout names (J_L -> "J"), and registry identifiers become
clinical display names. Nothing is estimated here except the reshaping, and
held-out results enter only where the analysis already placed them.

A table whose source has not been produced yet is skipped and listed in
``plot_tables_manifest.json`` with the missing input, so a partial analysis
renders the figures it can support and reports the rest.

    python -m opera.run.build_paper_plot_tables --analysis-root <analysis_root> \
        --cell-metadata <cell_metadata.csv> --output-dir <figure_inputs/plot_tables>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from opera.evaluation.secondary_analysis import ALL_HORIZONS, claim_label
import yaml

from opera.run.analyse_locked_predictions import file_hash, read_table
from opera.visualization.outcome_names import outcome_display_name
from opera.visualization.paper_taxonomy import (
    disease_group_name,
    disease_name,
    fine_cohort_groups,
    outcome_endpoint_groups,
    to_display,
)

CONTRAST_CODES = {
    "P1": "D-S", "P2": "O-S", "P3": "J-O", "P4": "J-D", "P5": "J-S", "P6": "interaction",
    "T1": "R-X", "T2": "J-X", "T3": "S-X", "T4": "X-LR",
    "T6": "J-XP", "T7": "XP-X", "T8": "D-XP", "T9": "J-XR", "T10": "XR-X", "P0": "L-RI", "P7": "L-FZ",
    "N10": "EA-L:F10", "N25": "EA-L:F25", "N50": "EA-L:F50",
    **{f"LS{f}": f"L:F{f}-S" for f in (10, 25, 50)},
    **{f"LSE{f}": f"EA:F{f}-S" for f in (10, 25, 50)},
    **{f"FD{f}": f"EA:F{f}-L" for f in (10, 25, 50)},
    **{f"C{i}": f"C{i}" for i in range(1, 7)}, "RS1": "RS1",
    "O1": "OPERA-X", "O2": "OPERA-J", "O3": "OPERA-S",
    "R1_orig": "Rorig-J", "T1_orig": "Rorig-X", "R_orig": "Rorig-R",
    "E1": "E-L:J", "E2": "E-L:R", "A1": "EA-E:J", "A2": "EA-E:R",
    "E3": "EA-L:J", "E4": "EA-L:R",
    **{f"R{i}": f"R{i}" for i in range(8)},
}
MODEL_CODES = {"S_L": "S", "D_L": "D", "O_L": "O", "J_L": "J", "R_L": "R",
               "OPERA": "OPERA", "X": "X", "LR": "LR"}
ORIGIN_MODELS = {f"{readout}_{origin}": (origin, readout)
                 for origin in ("L", "E", "EA") for readout in ("J", "R")}
DEFAULT_CHOICES = Path(__file__).resolve().parents[1] / "configs/paper_figure_choices.yaml"


class Missing(Exception):
    """A source table the analysis has not produced yet."""


def _read(path: Path, sources: dict) -> pd.DataFrame:
    if not path.exists():
        raise Missing(str(path))
    try:
        frame = read_table(path)
    except pd.errors.EmptyDataError:
        # The analysis writes an empty file when nothing was estimable (e.g. a
        # tuning preview without every model arm): report it, do not crash.
        raise Missing(f"{path} (empty)") from None
    sources[str(path)] = file_hash(path)
    return frame


# Split the figures describe: held_out for the paper, tuning for a preview.
EVALUATION_SPLIT = "held_out"


def _select(frame: pd.DataFrame, *, split=None, seed=None) -> pd.DataFrame:
    split = split or EVALUATION_SPLIT
    if "split" in frame:
        frame = frame.loc[frame.split == split]
    if seed is not None and "seed" in frame:
        frame = frame.loc[frame.seed.astype(str) == str(seed)]
    if "seed" in frame and frame.seed.nunique() > 1:
        raise ValueError("Several training seeds present; pass --seed")
    return frame


def _cell_groups(metadata: pd.DataFrame) -> pd.DataFrame:
    keys = ["cohort", "outcome", "horizon_days"]
    columns = [c for c in ("cohort_group", "endpoint_group", "n_events_train",
                           "n_patients_train") if c in metadata]
    return metadata[keys + columns].drop_duplicates(keys)


def _with_cells(frame: pd.DataFrame, metadata: pd.DataFrame) -> pd.DataFrame:
    """Attach train-only cell metadata unless the analysis already did."""
    missing = [c for c in ("cohort_group", "endpoint_group", "n_events_train",
                           "n_patients_train") if c not in frame and c in metadata]
    if not missing:
        return frame
    keys = ["cohort", "outcome", "horizon_days"]
    groups = _cell_groups(metadata)[keys + missing]
    return frame.merge(groups, on=keys, how="left", validate="many_to_one")


def paired_cells(contrasts, metadata, *, seed=None):
    data = _select(contrasts, seed=seed)
    data = data.loc[data.contrast.isin(CONTRAST_CODES)].copy()
    data["contrast"] = data.contrast.map(CONTRAST_CODES)
    data = _with_cells(data, metadata)
    columns = ["cohort", "cohort_group", "outcome", "endpoint_group", "horizon_days",
               "contrast", "metric", "gain", "lower", "upper", "n_events_train",
               "n_patients_train", "n_cases", "n_controls", "n_valid_bootstrap",
               "interval_status"]
    return to_display(data[[c for c in columns if c in data]])


def panel_contrasts(summaries, codes, *, seed=None):
    data = _select(summaries, seed=seed)
    data = data.loc[(data.stratum == "all_cells") & data.contrast.isin(codes)].copy()
    data["contrast"] = data.contrast.map(CONTRAST_CODES)
    return data[["contrast", "metric", "horizon_days", "gain", "lower", "upper",
                 "n_cells"]]


def support_curve(curves):
    data = pd.concat(curves, ignore_index=True)
    data = data.loc[data.contrast.isin(CONTRAST_CODES)].copy()
    data["contrast"] = data.contrast.map(CONTRAST_CODES)
    return data.rename(columns={"posterior_median": "mean", "posterior_lower_95": "lower",
                                "posterior_upper_95": "upper"})[
        ["contrast", "training_support", "mean", "lower", "upper"]]


FALLBACK_CONTRASTS = ("J-S", "D-S", "O-S", "J-X", "OPERA-X", "S-X", "J-XP")


def descriptive_support_curve(cells, *, contrasts=FALLBACK_CONTRASTS, bandwidth=0.35,
                              n_boot=200, points=60, seed=2026):
    """Fallback when the hierarchical posterior curve is missing (e.g. no PyMC):
    a Gaussian kernel smoother of cell AUC gain over log10 training events,
    each cell pooled over its horizons, with a 95% bootstrap band over cells.
    Descriptive only; ``source`` says so, and the figures' captions must too."""
    import numpy as np

    rng = np.random.default_rng(seed)
    rows = []
    data = cells.loc[(cells.metric == "auc") & (cells.horizon_days == ALL_HORIZONS)
                     & cells.gain.notna() & (cells.n_events_train > 0)]
    for contrast in contrasts:
        part = data.loc[data.contrast == contrast]
        if len(part) < 20:
            continue
        x = np.log10(part.n_events_train.to_numpy(float))
        y = part.gain.to_numpy(float)
        grid = np.linspace(*np.quantile(x, [0.02, 0.98]), points)

        def smooth(xs, ys):
            weights = np.exp(-0.5 * ((grid[:, None] - xs[None, :]) / bandwidth) ** 2)
            return (weights * ys).sum(1) / weights.sum(1)

        centre = smooth(x, y)
        draws = np.array([smooth(x[i], y[i]) for i in
                          (rng.integers(0, len(x), len(x)) for _ in range(n_boot))])
        low, high = np.quantile(draws, [0.025, 0.975], axis=0)
        rows.append(pd.DataFrame({"contrast": contrast, "training_support": 10 ** grid,
                                  "mean": centre, "lower": low, "upper": high,
                                  "source": "descriptive_fallback"}))
    if not rows:
        raise Missing("no cells for a descriptive support curve")
    return pd.concat(rows, ignore_index=True)


def origin_estimates(panel, *, seed=None):
    data = _select(panel, seed=seed)
    data = data.loc[data.model.isin(ORIGIN_MODELS)].copy()
    data[["origin", "readout"]] = data.model.map(ORIGIN_MODELS).tolist()
    return data[["origin", "readout", "metric", "horizon_days", "estimate", "lower",
                 "upper", "n_cells"]]


def calibration_tables(summary, reliability, metadata, *, seed=None):
    out = []
    for frame in (summary, reliability):
        data = _select(frame, seed=seed)
        data = data.loc[data.model.isin(MODEL_CODES)].copy()
        data["model"] = data.model.map(MODEL_CODES)
        out.append(to_display(_with_cells(data, metadata)))
    return out


# Model codes in estimates.csv -> short figure names.
MODEL_NAMES = {"S_L": "S", "D_L": "D", "O_L": "O", "J_L": "J", "R_L": "R", "OPERA": "OPERA",
               "X": "X", "XR": "XR", "XP": "XP", "LR": "LR", "J_EA": "J_EA",
               "J_E": "J_E",
               **{f"J_{o}_F{f}": f"J_{o}_F{f}" for o in ("L", "EA") for f in (10, 25, 50)}}


def model_cells(estimates, metadata, *, seed=None, metric="auc"):
    """One row per cell with each model's held-out estimate (absolute, not a gain)."""
    data = _select(estimates, seed=seed)
    data = data.loc[(data.metric == metric) & data.model.isin(MODEL_NAMES)]
    wide = (data.pivot_table(index=["cohort", "outcome", "horizon_days"], columns="model",
                             values="estimate", aggfunc="first")
            .rename(columns=MODEL_NAMES).reset_index())
    wide.columns.name = None
    # Each cell also pooled over its evaluated horizons (horizon_days ==
    # ALL_HORIZONS), with the one-year training support as for paired cells.
    pooled = (wide.drop(columns="horizon_days").groupby(["cohort", "outcome"]).mean()
              .reset_index().assign(horizon_days=ALL_HORIZONS))
    year = metadata.loc[metadata.horizon_days == 365].assign(horizon_days=ALL_HORIZONS)
    return to_display(pd.concat([_with_cells(wide, metadata), _with_cells(pooled, year)],
                                ignore_index=True))


def strategy_tables(estimates, selection, metadata, *, seed=None, metric="auc"):
    data = _select(estimates, seed=seed)
    data = data.loc[(data.metric == metric)
                    & data.model.isin(["S_L", "J_L", "BEST_TUNING"])]
    wide = (data.pivot_table(index=["cohort", "outcome", "horizon_days"], columns="model",
                             values="estimate", aggfunc="first")
            .rename(columns={"S_L": "specialist", "J_L": "joint",
                             "BEST_TUNING": "best_tuning"})
            .dropna().reset_index())
    wide = _with_cells(wide, metadata)
    chosen = _select(selection, seed=seed)
    chosen = chosen.loc[chosen.status == "selected_on_tuning"].copy()
    chosen["selected_model"] = chosen.selected_model.str.removesuffix("_L")
    groups = outcome_endpoint_groups()
    chosen["endpoint_group"] = chosen.outcome.map(groups)
    return (to_display(wide),
            to_display(chosen[["cohort", "outcome", "endpoint_group", "selected_model"]]))


def rescue_cells(contrasts, metadata, *, seed=None):
    keys = ["cohort", "outcome", "horizon_days", "metric"]
    parts = {
        "tuning_joint_vs_specialist": ("tuning", "P5"),
        "joint_vs_specialist": (EVALUATION_SPLIT, "P5"),
        "routed_vs_specialist": (EVALUATION_SPLIT, "router_vs_specialist"),
    }
    wide = None
    for column, (split, code) in parts.items():
        data = _select(contrasts, split=split, seed=seed)
        data = data.loc[data.contrast == code, keys + ["gain"]].rename(
            columns={"gain": column})
        wide = data if wide is None else wide.merge(data, on=keys, how="inner")
    return to_display(_with_cells(wide.dropna(), metadata))


def hierarchy_cells(contrasts, metadata, *, seed=None, designs=("AB", "O")):
    """Per cell: the router decomposition on the evaluation split (own
    corrections, borrowing, both) beside the tuning joint-vs-specialist gain.
    ``designs``: preference order (paper_figure_choices.yaml,
    router_decomposition_design); the first one analysed is used."""
    keys = ["cohort", "outcome", "horizon_days", "metric"]
    present = set(contrasts.contrast)
    design = next((tag for tag in designs if f"router_both_{tag}" in present), None)
    if design is None:
        raise Missing("no router decomposition in the analysis (analyse the "
                         "router-fix fits: inventory and analyse at stage 4)")
    tuning = _select(contrasts, split="tuning", seed=seed)
    wide = tuning.loc[tuning.contrast == "P5", keys + ["gain"]].rename(
        columns={"gain": "tuning_joint_vs_specialist"})
    evaluation = _select(contrasts, split=EVALUATION_SPLIT, seed=seed)
    for column in ("own", "borrow", "both"):
        part = evaluation.loc[evaluation.contrast == f"router_{column}_{design}",
                              keys + ["gain", "lower", "upper"]]
        part = part.rename(columns={"gain": column, "lower": f"{column}_lower",
                                    "upper": f"{column}_upper"})
        wide = wide.merge(part, on=keys, how="inner")
    return to_display(_with_cells(wide.dropna(subset=["own", "borrow", "both"]),
                                  metadata).assign(router_design=design))


def coverage_table(metadata, contrast_coverage, model_coverage=None, *, horizon=365,
                   seed=None, registry_outcomes=None):
    """Why each disease–outcome cell was or was not compared at ``horizon``.

    Cells absent from the training panel are structurally not applicable. A
    specialist without predictions is a failed fit when it had training events
    and an unsupported one otherwise; a test set without both classes cannot
    be evaluated.
    """
    cells = metadata.loc[metadata.horizon_days == horizon]
    outcomes = sorted(registry_outcomes or cells.outcome.unique())
    grid = pd.MultiIndex.from_product([sorted(fine_cohort_groups()), outcomes],
                                      names=["cohort", "outcome"]).to_frame(index=False)
    grid = grid.merge(cells[["cohort", "outcome", "n_events_train"]],
                      on=["cohort", "outcome"], how="left", indicator=True)
    paired = _select(contrast_coverage, seed=seed)
    paired = paired.loc[(paired.contrast == "P5") & (paired.horizon_days == horizon)]
    paired = paired.set_index(["cohort", "outcome"]).status
    predictions = None
    if model_coverage is not None:
        predictions = _select(model_coverage, seed=seed)
        predictions = predictions.loc[predictions.horizon_days == horizon]

    def status(row):
        if row._merge == "left_only":
            return "not_applicable"
        if not row.n_events_train:
            return "no_training_events"
        if predictions is not None:
            mine = predictions.loc[(predictions.cohort == row.cohort)
                                   & (predictions.outcome == row.outcome)]
            if (mine.prediction_status == "single_class").any():
                return "single_class_test"
            specialist = mine.loc[mine.model == "S_L"]
            if specialist.empty or (specialist.prediction_status == "absent").any():
                return "fit_failed"
        if paired.get((row.cohort, row.outcome)) == "paired":
            return "evaluated"
        return "fit_failed"

    grid["status"] = grid.apply(status, axis=1)
    grid["endpoint_group"] = grid.outcome.map(outcome_endpoint_groups())
    return to_display(grid[["cohort", "outcome", "endpoint_group", "status"]])


def atlas_table(pooled_metrics, metadata):
    events = (metadata.groupby(["outcome", "horizon_days"]).n_events_train.sum()
              .reset_index())
    data = pooled_metrics.merge(events, on=["outcome", "horizon_days"], how="left")
    data["endpoint_group"] = data.outcome.map(outcome_endpoint_groups())
    data = data[["outcome", "endpoint_group", "horizon_days", "n_events_train", "cif_auc"]]
    return to_display(data)


def route_table(routes, *, seed=None, model="R_L"):
    data = _select(routes, seed=seed)
    data = data.loc[data.model == model].copy()
    data["source"] = data.source.astype(str).str.split(":").str[-1]
    data["source"] = [disease_name(s) if bank == "disease"
                      else outcome_display_name(s) if bank == "outcome" else s
                      for s, bank in zip(data.source, data.bank)]
    return to_display(data[["cohort", "outcome", "bank", "source", "mean"]])


SUBGROUP_FIGURES = {
    "patient_subgroups": {
        "contrasts": {"P5": ("#2D3A8C", "Joint vs specialist"),
                      "T2": ("#408579", "Joint vs XGBoost"),
                      "O1": ("#8974AC", "OPERA vs XGBoost")},
        "covariates": {"history_event_count": "Recorded history before treatment",
                       "age_at_index": "Age", "sex": "Sex",
                       "predominant_region": "Region",
                       "IPI": "Clinical risk score"},
    },
    "national_subgroups": {
        "contrasts": {"E3": ("#2D3A8C", "Adapted national vs local"),
                      "E1": ("#5C7AEA", "National vs local")},
        "covariates": {"national_vocabulary_coverage":
                       "History covered by national vocabulary",
                       "support_bin": "Target training events",
                       "index_year": "Year of treatment"},
    },
}


# Danish region codes (DST/CPR) as they appear in the national data.
REGION_CODES = {"1081": "Nordjylland", "1082": "Midtjylland", "1083": "Syddanmark",
                "1084": "Hovedstaden", "1085": "Sjælland"}
# Named subgroups in reading order; anything unlisted sorts after, "Not recorded" last.
NAMED_ORDER = ["Female", "Male", "Low", "Low-intermediate", "Intermediate",
               "High-intermediate", "High", "Very high", *REGION_CODES.values()]


def _number(value: float) -> str:
    return f"{int(value):,}" if float(value).is_integer() else f"{value:g}"


def subgroup_label(label: str) -> str:
    """Readable labels for pd.cut intervals: '(49.0, 64.0]' -> '50–64'."""
    text = str(label).strip()
    if text.removesuffix(".0") in REGION_CODES:
        return REGION_CODES[text.removesuffix(".0")]
    if not (text[:1] in "([" and text[-1:] in ")]" and "," in text):
        named = {"missing": "Not recorded", "nan": "Not recorded", "f": "Female",
                 "m": "Male", "female": "Female", "male": "Male"}
        return named.get(text.lower(), text[:1].upper() + text[1:])
    low, high = (float(part) for part in text[1:-1].split(","))
    if low < 0 or (text[0] == "[" and low == 0) or low == float("-inf"):
        return f"≤ {_number(high)}"
    if high == float("inf"):
        return f"> {_number(low)}"
    start = low + 1 if float(low).is_integer() and float(high).is_integer() else low
    return f"{_number(start)}–{_number(high)}"


def _subgroup_sort_key(label) -> tuple:
    """Intervals by their lower bound, named groups in NAMED_ORDER, missing last."""
    text = str(label).strip()
    shown = subgroup_label(text)
    if shown == "Not recorded":
        return (3, 0.0, "")
    if text[:1] in "([" and text[-1:] in ")]" and "," in text:
        return (0, float(text[1:-1].split(",")[0]), "")
    if shown in NAMED_ORDER:
        return (1, float(NAMED_ORDER.index(shown)), "")
    return (2, 0.0, shown)


def subgroup_table(panels, summaries, spec, *, seed=None, horizon=ALL_HORIZONS,
                   metric="brier_gain"):
    """Panel subgroup effects for one figure, with display labels and styles."""
    data = _select(panels, seed=seed)
    if horizon == ALL_HORIZONS and not (data.horizon_days == ALL_HORIZONS).any():
        horizon = 365  # an analysis from before subgroups were pooled over horizons
    data = data.loc[(data.metric == metric) & (data.horizon_days == horizon)
                    & data.contrast.isin(spec["contrasts"])
                    & data.covariate.isin(spec["covariates"])].copy()
    rows = [data[["covariate", "subgroup", "contrast", "gain", "lower", "upper"]]]
    if summaries is not None and "support_bin" in spec["covariates"]:
        cells = _select(summaries, seed=seed)
        cells = cells.loc[(cells.stratum == "support_bin") & (cells.metric == "brier")
                          & (cells.horizon_days == horizon)
                          & cells.contrast.isin(spec["contrasts"])]
        rows.append(cells.assign(covariate="support_bin", subgroup=cells.stratum_value)[
            ["covariate", "subgroup", "contrast", "gain", "lower", "upper"]])
    table = pd.concat(rows, ignore_index=True)
    table["sort_key"] = table.subgroup.map(_subgroup_sort_key)
    table["subgroup"] = table.subgroup.map(subgroup_label)
    table = (table.sort_values(["sort_key"], kind="stable")
             .drop(columns="sort_key").reset_index(drop=True))
    order = {name: title for name, title in spec["covariates"].items()
             if name in set(table.covariate)}
    styles = {code: list(style) for code, style in spec["contrasts"].items()}
    return table, styles, order


def subgroup_auc_table(effects, spec, *, seed=None, horizon=ALL_HORIZONS):
    """Equal-cell AUC gains recomputed within prespecified patient groups."""
    panels = effects.rename(columns={"stratum_value": "subgroup"}).copy()
    panels = panels.loc[panels.metric == "auc"].copy()
    return subgroup_table(panels, None, spec, seed=seed, horizon=horizon, metric="auc")[0]


def group_maps():
    diseases = {disease_name(fine): disease_group_name(group)
                for fine, group in fine_cohort_groups().items()}
    outcomes = {outcome_display_name(outcome): group
                for outcome, group in outcome_endpoint_groups().items()}
    return diseases, outcomes


def calibration_examples(choices, metadata=None, *, horizon=365):
    """Reliability-curve examples: a fixed list, or (``calibration_example_rule``)
    per listed clinical domain the cell with the most training events among
    those common enough to calibrate visibly. Train-only counts: the rule never
    sees test results, so it can be fixed now and applied unchanged later."""
    rule = choices.get("calibration_example_rule")
    if rule is None or metadata is None or not {"n_events_train", "n_patients_train"} <= set(metadata):
        items = choices.get("calibration_examples", [])
    else:
        cells = metadata.loc[metadata.horizon_days == horizon].drop_duplicates(
            ["cohort", "outcome"]).copy()
        if "endpoint_group" not in cells:
            cells["endpoint_group"] = cells.outcome.map(outcome_endpoint_groups())
        share = cells.n_events_train / cells.n_patients_train
        eligible = cells.loc[(cells.n_events_train >= rule.get("min_events_train", 200))
                             & (share >= rule.get("min_event_share", 0.10))
                             & (share <= rule.get("max_event_share", 1.0))]
        items = []
        for domain in rule["domains"]:
            part = eligible.loc[eligible.endpoint_group == domain]
            if len(part):
                best = part.sort_values(["n_events_train", "cohort", "outcome"],
                                        ascending=[False, True, True]).iloc[0]
                items.append({"cohort": best.cohort, "outcome": best.outcome})
    return {f"{outcome_display_name(item['outcome'])} ({disease_name(item['cohort'])})":
            [disease_name(item["cohort"]), outcome_display_name(item["outcome"])]
            for item in items}


def region_tables(effects, models, patients=None, *, seed=None):
    """Per-region panel effects, absolute metrics and pre-index data coverage."""
    effects = _select(effects, seed=seed).copy()
    effects["contrast"] = effects.contrast.map(lambda c: CONTRAST_CODES.get(c, c))
    models = _select(models, seed=seed).copy()
    models["model"] = models.model.map(lambda m: MODEL_NAMES.get(m, m))
    coverage = None
    if patients is not None and "predominant_region" in patients:
        columns = [c for c in ("history_event_count", "laboratory_event_count") if c in patients]
        coverage = (patients.dropna(subset=["predominant_region"])
                    .groupby("predominant_region")[columns]
                    .agg(["median", lambda x: x.quantile(0.25), lambda x: x.quantile(0.75)]))
        coverage.columns = [f"{c}_{s}" for c in columns for s in ("median", "q1", "q3")]
        coverage = coverage.reset_index().rename(columns={"predominant_region": "region"})
        coverage["n_patients"] = (patients.predominant_region.value_counts()
                                  .reindex(coverage.region).to_numpy())
    return effects, models, coverage


# Rows of Table 2 (panel-level comparisons), in reading order.
PANEL_TABLE_ROWS = (
    ("P5", "Joint vs specialist"), ("P1", "Disease-pooled vs specialist"),
    ("P2", "Outcome-pooled vs specialist"), ("T2", "Joint vs per-task XGBoost"),
    ("T3", "Specialist vs per-task XGBoost"), ("T7", "Pooled vs per-task XGBoost"),
    ("T6", "Joint vs XGBoost pooled across diseases"),
    ("T9", "Joint vs XGBoost with registry variables"), ("R1", "Routed vs joint"),
    ("O1", "OPERA vs per-task XGBoost"),
    ("R_orig", "Originally locked router vs primary router (sensitivity)"),
    ("IPW_R", "Routed model, IPW event sampling vs natural (sensitivity)"),
    ("E1", "National vs local encoder (joint)"),
    ("E3", "Adapted national vs local encoder (joint)"),
)
PRACTICAL_MARGIN = 0.01  # AUC and IPA, the locked practical margin
TABLE_MODELS = {"J_L": "Joint", "R_L": "Routed", "S_L": "Specialist", "X": "XGBoost"}


def _interval(low, high, digits=3):
    if pd.isna(low) or pd.isna(high):
        return None
    return f"{low:.{digits}f} to {high:.{digits}f}"


def panel_effects(summaries, contrasts, *, horizon=ALL_HORIZONS, seed=None):
    """Table 2: pooled paired differences per comparison, with share of cells better.

    ``horizon`` = ALL_HORIZONS pools every disease-outcome-horizon cell (the
    main table); a number gives the table at that horizon (supplement).
    """
    summary = _select(summaries, seed=seed)
    summary = summary.loc[summary.horizon_days == horizon]
    rare = summary.loc[(summary.get("stratum", "") == "subtype_size")
                       & summary.get("stratum_value", pd.Series("", index=summary.index))
                       .astype(str).str.startswith("(-1")]
    summary = summary.loc[summary.get("stratum", "all_cells") == "all_cells"]
    cells = _select(contrasts, seed=seed)
    if horizon != ALL_HORIZONS:
        cells = cells.loc[cells.horizon_days == horizon]
    rows = []
    for code, label in PANEL_TABLE_ROWS:
        row = {"comparison": label, "contrast": code}
        for metric in ("auc", "ipa"):
            part = summary.loc[(summary.contrast == code) & (summary.metric == metric)]
            if len(part):
                record = part.iloc[0]
                row[f"{metric}_gain"] = round(float(record.gain), 3)
                row[f"{metric}_interval"] = _interval(record.lower, record.upper)
                row[f"{metric}_p_value"] = (float(record.p_value)
                                            if "p_value" in record and pd.notna(record.p_value)
                                            else None)
                row["cells"] = int(record.n_cells)
                if metric == "auc":
                    row["label"] = claim_label(float(record.gain), float(record.lower),
                                               float(record.upper), PRACTICAL_MARGIN)
        part = rare.loc[(rare.contrast == code) & (rare.metric == "auc")]
        if len(part):
            record = part.iloc[0]
            row["rare_auc_gain"] = round(float(record.gain), 3)
            row["rare_auc_interval"] = _interval(record.lower, record.upper)
        gains = cells.loc[(cells.contrast == code) & (cells.metric == "auc"), "gain"].dropna()
        row["cells_better"] = f"{(gains > 0).mean():.0%}" if len(gains) else None
        if "cells" in row:
            rows.append(row)
    return pd.DataFrame(rows)


def paired_domain_summary(contrasts, metadata, *, horizon=365, seed=None):
    """Supplementary table: median paired gain (IQR) by comparison and clinical domain."""
    data = _with_cells(_select(contrasts, seed=seed), metadata)
    data = data.loc[(data.horizon_days == horizon) & data.metric.isin(["auc", "ipa"])
                    & data.contrast.isin([code for code, _ in PANEL_TABLE_ROWS])]
    labels = dict(PANEL_TABLE_ROWS)
    rows = []
    for (code, group, metric), part in data.groupby(["contrast", "endpoint_group", "metric"]):
        gains = part.gain.dropna()
        if not len(gains):
            continue
        q1, median, q3 = gains.quantile([0.25, 0.5, 0.75])
        rows.append({"contrast": labels[code], "endpoint_group": group, "metric": metric.upper(),
                     "gain": round(float(median), 3), "interval": _interval(q1, q3),
                     "common_cells": len(gains)})
    return pd.DataFrame(rows)


def calibration_domain_summary(calibration, metadata, *, horizon=365, seed=None):
    """Supplementary table: median calibration by model and clinical domain."""
    data = _with_cells(_select(calibration, seed=seed), metadata)
    data = data.loc[(data.horizon_days == horizon) & data.model.isin(TABLE_MODELS)]
    rows = []
    for (model, group), part in data.groupby(["model", "endpoint_group"]):
        estimable = part.dropna(subset=[c for c in ("calibration_slope",) if c in part])
        rows.append({
            "model": TABLE_MODELS[model], "endpoint_group": group, "horizon_days": horizon,
            "calibration_in_the_large": (round(float(estimable.calibration_in_the_large.median()), 3)
                                         if "calibration_in_the_large" in estimable and len(estimable) else None),
            "calibration_slope": (round(float(estimable.calibration_slope.median()), 2)
                                  if "calibration_slope" in estimable and len(estimable) else None),
            "status": f"{len(estimable)} of {len(part)} cells estimable"})
    return pd.DataFrame(rows)


def outcome_support_summary(estimates, metadata, *, seed=None, model="J_L"):
    """Supplementary table: patients, events and cells by clinical domain and horizon."""
    data = _with_cells(_select(estimates, seed=seed), metadata)
    data = data.loc[(data.model == model) & (data.metric == "auc")]
    rows = []
    counts = {"patient_cells": "n", "events": "n_cases", "non_events": "n_controls"}
    for (group, horizon), part in data.groupby(["endpoint_group", "horizon_days"]):
        rows.append({"endpoint_group": group, "horizon_days": int(horizon), "cells": len(part),
                     **{name: int(part[column].sum()) for name, column in counts.items()
                        if column in part}})
    return pd.DataFrame(rows)


def operating_point_summary(points, *, horizon=365, seed=None):
    """Supplementary table: median threshold metrics across cells, by model and rule."""
    data = points.loc[(points.horizon_days == horizon) & (points.status == "estimated")
                      & points.model.isin(TABLE_MODELS)]
    if seed is not None and "seed" in data:
        data = data.loc[data.seed.astype(str) == str(seed)]
    rows = []
    for (model, rule), part in data.groupby(["model", "rule"]):
        rows.append({"model": TABLE_MODELS[model],
                     "rule": rule.replace("tuning_", "").replace("_", " "),
                     **{metric: round(float(part[metric].median()), 2)
                        for metric in ("sensitivity", "specificity", "ppv", "mcc")}})
    return pd.DataFrame(rows)


def build(analysis_root: Path, cell_metadata: Path, output_dir: Path, *, seed=None,
          atlas_metrics: Path | None = None, choices: Path = DEFAULT_CHOICES,
          bayes_variant="support_spline", registry_outcomes=None,
          patient_metadata: Path | None = None):
    analysis = analysis_root / "prediction_analysis"
    bayes = analysis_root / "bayesian"
    output_dir.mkdir(parents=True, exist_ok=True)
    sources: dict[str, str] = {}
    metadata = _read(cell_metadata, sources)
    choice_spec = yaml.safe_load(Path(choices).read_text()) if Path(choices).exists() else {}
    report = {}

    def table(name, build_fn):
        try:
            result = build_fn()
        except Missing as missing:
            report[name] = {"status": "missing_input", "input": str(missing)}
            return
        if isinstance(result, pd.DataFrame):
            result.to_csv(output_dir / name, index=False)
            report[name] = {"status": "written", "rows": len(result)}
        else:
            (output_dir / name).write_text(json.dumps(result, indent=2, ensure_ascii=False))
            report[name] = {"status": "written", "entries": len(result)}

    contrasts = lambda: _read(analysis / "contrasts.csv", sources)  # noqa: E731

    def cell_contrasts():
        """Per-cell contrasts plus, where the analysis made them, each cell
        pooled over its horizons (horizon_days == ALL_HORIZONS) for the
        cell-level figures; panel tables keep the per-horizon cells only."""
        path = analysis / "contrasts_all_horizons.csv"
        if not path.exists():
            return contrasts()
        per_horizon = contrasts()
        pooled = _read(path, sources)
        if "n_events_train" in per_horizon:  # the analysis annotated its cells
            pooled = _with_cells(pooled, cell_metadata_all)
        return pd.concat([per_horizon, pooled], ignore_index=True)

    # Pooled cells carry the one-year training support (a proxy for ordering
    # and support plots; captions say so).
    cell_metadata_all = pd.concat(
        [metadata, metadata.loc[metadata.horizon_days == 365].assign(horizon_days=ALL_HORIZONS)],
        ignore_index=True)
    summaries = lambda: _read(analysis / "effect_summaries.csv", sources)  # noqa: E731
    table("paired_cells.csv", lambda: paired_cells(cell_contrasts(), cell_metadata_all,
                                                    seed=seed))
    def curve():
        try:
            return support_curve([
                _read(bayes / family / "auc" / bayes_variant / "posterior_curve.csv", sources)
                for family in ("pooling", "tabular")])
        except Missing:
            # No posterior curve yet (no PyMC): the descriptive smoother, labelled.
            return descriptive_support_curve(
                paired_cells(cell_contrasts(), cell_metadata_all, seed=seed))

    table("support_curve.csv", curve)
    if isinstance(report.get("support_curve.csv"), dict) and \
            (output_dir / "support_curve.csv").exists() and \
            "source" in pd.read_csv(output_dir / "support_curve.csv", nrows=1):
        report["support_curve.csv"]["source"] = "descriptive_fallback (no posterior curve)"
    table("external_panel.csv", lambda: panel_contrasts(
        summaries(), ["E1", "E2", "A1", "A2", "E3", "E4"], seed=seed))
    table("gate_effects.csv", lambda: panel_contrasts(
        summaries(), [f"R{i}" for i in range(8)], seed=seed))
    table("origin_estimates.csv", lambda: origin_estimates(
        _read(analysis / "panel_estimates.csv", sources), seed=seed))

    def calibration():
        summary, reliability = calibration_tables(
            _read(analysis / "calibration_summary.csv", sources),
            _read(analysis / "calibration.csv", sources), metadata, seed=seed)
        reliability.to_csv(output_dir / "reliability.csv", index=False)
        report["reliability.csv"] = {"status": "written", "rows": len(reliability)}
        return summary

    table("calibration_summary.csv", calibration)
    table("calibration_examples.json", lambda: calibration_examples(choice_spec, metadata))

    def strategy():
        cells, selection = strategy_tables(
            _read(analysis / "estimates.csv", sources),
            _read(analysis / "strategy_selection.csv", sources), metadata, seed=seed)
        selection.to_csv(output_dir / "selected_strategy.csv", index=False)
        report["selected_strategy.csv"] = {"status": "written", "rows": len(selection)}
        return cells

    table("strategy_cells.csv", strategy)
    # Printed tables (Table 2 and the supplementary tables).
    table("panel_effects.csv", lambda: panel_effects(summaries(), contrasts(), seed=seed))
    # Supplement: Table 2 at each horizon.
    table("panel_effects_by_horizon.csv", lambda: pd.concat(
        [panel_effects(summaries(), contrasts(), horizon=h, seed=seed).assign(horizon_days=h)
         for h in sorted(set(summaries().horizon_days) - {ALL_HORIZONS})], ignore_index=True))
    table("paired_domain_summary.csv",
          lambda: paired_domain_summary(contrasts(), metadata, seed=seed))
    table("calibration_domain_summary.csv", lambda: calibration_domain_summary(
        _read(analysis / "calibration_summary.csv", sources), metadata, seed=seed))
    table("outcome_support_summary.csv", lambda: outcome_support_summary(
        _read(analysis / "estimates.csv", sources), metadata, seed=seed))
    table("operating_point_summary.csv", lambda: operating_point_summary(
        _read(analysis / "operating_points.csv", sources), seed=seed))
    table("model_cells.csv", lambda: model_cells(
        _read(analysis / "estimates.csv", sources), metadata, seed=seed))
    table("rescue_cells.csv", lambda: rescue_cells(cell_contrasts(), cell_metadata_all,
                                                    seed=seed))
    design = choice_spec.get("router_decomposition_design")
    table("hierarchy_cells.csv", lambda: hierarchy_cells(
        cell_contrasts(), cell_metadata_all, seed=seed,
        designs=(design, "AB", "O") if design else ("AB", "O")))

    def coverage():
        model_coverage = analysis / "model_coverage.csv"
        return coverage_table(
            metadata, _read(analysis / "contrast_coverage.csv", sources),
            _read(model_coverage, sources) if model_coverage.exists() else None,
            seed=seed, registry_outcomes=registry_outcomes)

    table("coverage.csv", coverage)
    if atlas_metrics is not None:
        table("atlas.csv", lambda: atlas_table(_read(Path(atlas_metrics), sources),
                                               metadata))
    else:
        report["atlas.csv"] = {"status": "missing_input",
                               "input": "--atlas-metrics (selected joint run, held out)"}
    table("route_summaries.csv", lambda: route_table(
        _read(analysis / "route_weights.csv", sources), seed=seed))
    def subgroups(name, spec):
        def build_fn():
            panels = _read(analysis / "patient_subgroup_panels.csv", sources)
            effect = analysis / "effect_summaries.csv"
            table, styles, order = subgroup_table(
                panels, _read(effect, sources) if effect.exists() else None, spec,
                seed=seed)
            for suffix, payload in (("styles", styles), ("order", order)):
                (output_dir / f"{name}_{suffix}.json").write_text(
                    json.dumps(payload, indent=2, ensure_ascii=False))
            return table
        return build_fn

    for name, spec in SUBGROUP_FIGURES.items():
        table(f"{name}.csv", subgroups(name, spec))

    def subgroup_auc():
        spec = SUBGROUP_FIGURES["patient_subgroups"]
        paths = [analysis / f"stratified_{column}_effects.csv"
                 for column in spec["covariates"]]
        parts = [_read(path, sources) for path in paths if path.is_file()]
        if not parts:
            raise Missing("No stratified patient-group effects; rerun analyse")
        return subgroup_auc_table(pd.concat(parts, ignore_index=True), spec, seed=seed)

    table("patient_subgroups_auc.csv", subgroup_auc)

    def regions():
        effects, models, coverage = region_tables(
            _read(analysis / "stratified_predominant_region_effects.csv", sources),
            _read(analysis / "stratified_predominant_region_models.csv", sources),
            _read(patient_metadata, sources) if patient_metadata else None, seed=seed)
        models.to_csv(output_dir / "region_models.csv", index=False)
        report["region_models.csv"] = {"status": "written", "rows": len(models)}
        if coverage is not None:
            coverage.to_csv(output_dir / "region_coverage.csv", index=False)
            report["region_coverage.csv"] = {"status": "written", "rows": len(coverage)}
        return effects

    table("region_effects.csv", regions)
    diseases, outcomes = group_maps()
    table("disease_groups.json", lambda: diseases)
    table("outcome_groups.json", lambda: outcomes)
    for name in ("patient_coordinates.csv",
                 "calibration_map_outcomes.json", "matched_patient_embeddings.csv"):
        report.setdefault(name, {"status": "pending_upstream_builder"})
    manifest = {"analysis_root": str(analysis_root), "seed": seed,
                "bayes_variant": bayes_variant, "tables": report,
                "input_sha256": sources}
    (output_dir / "plot_tables_manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-root", type=Path, required=True)
    parser.add_argument("--cell-metadata", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed")
    parser.add_argument("--atlas-metrics", type=Path,
                        help="direct_cr_pooled_metrics.csv of the selected joint run")
    parser.add_argument("--choices", type=Path, default=DEFAULT_CHOICES)
    parser.add_argument("--bayes-variant", default="support_spline")
    parser.add_argument("--patient-metadata", type=Path,
                        help="Pre-index patient metadata (regional data coverage).")
    parser.add_argument("--split", choices=("held_out", "tuning"), default="held_out",
                        help="tuning: a preview of the figures before the test set is opened.")
    args = parser.parse_args()
    global EVALUATION_SPLIT
    EVALUATION_SPLIT = args.split
    manifest = build(args.analysis_root, args.cell_metadata, args.output_dir,
                     seed=args.seed, atlas_metrics=args.atlas_metrics,
                     choices=args.choices, bayes_variant=args.bayes_variant,
                     patient_metadata=args.patient_metadata)
    print(json.dumps({name: item["status"] for name, item in manifest["tables"].items()},
                     indent=2))


if __name__ == "__main__":
    main()
