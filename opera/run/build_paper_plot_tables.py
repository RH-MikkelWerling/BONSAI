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
    "T6": "J-XP", "T7": "XP-X", "T8": "D-XP", "P0": "L-RI",
    "E1": "E-L:J", "E2": "E-L:R", "A1": "EA-E:J", "A2": "EA-E:R",
    "E3": "EA-L:J", "E4": "EA-L:R",
    **{f"R{i}": f"R{i}" for i in range(8)},
}
MODEL_CODES = {"S_L": "S", "D_L": "D", "O_L": "O", "J_L": "J", "R_L": "R",
               "X": "X", "LR": "LR"}
ORIGIN_MODELS = {f"{readout}_{origin}": (origin, readout)
                 for origin in ("L", "E", "EA") for readout in ("J", "R")}
DEFAULT_CHOICES = Path(__file__).resolve().parents[1] / "configs/paper_figure_choices.yaml"


class Missing(Exception):
    """A source table the analysis has not produced yet."""


def _read(path: Path, sources: dict) -> pd.DataFrame:
    if not path.exists():
        raise Missing(str(path))
    sources[str(path)] = file_hash(path)
    return read_table(path)


def _select(frame: pd.DataFrame, *, split="held_out", seed=None) -> pd.DataFrame:
    if "split" in frame:
        frame = frame.loc[frame.split == split]
    if seed is not None and "seed" in frame:
        frame = frame.loc[frame.seed.astype(str) == str(seed)]
    if "seed" in frame and frame.seed.nunique() > 1:
        raise ValueError("Several training seeds present; pass --seed")
    return frame


def _cell_groups(metadata: pd.DataFrame) -> pd.DataFrame:
    keys = ["cohort", "outcome", "horizon_days"]
    columns = [c for c in ("cohort_group", "endpoint_group", "n_events_train")
               if c in metadata]
    return metadata[keys + columns].drop_duplicates(keys)


def _with_cells(frame: pd.DataFrame, metadata: pd.DataFrame) -> pd.DataFrame:
    """Attach train-only cell metadata unless the analysis already did."""
    missing = [c for c in ("cohort_group", "endpoint_group", "n_events_train")
               if c not in frame and c in metadata]
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
               "contrast", "metric", "gain", "lower", "upper", "n_events_train"]
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
        "joint_vs_specialist": ("held_out", "P5"),
        "routed_vs_specialist": ("held_out", "router_vs_specialist"),
    }
    wide = None
    for column, (split, code) in parts.items():
        data = _select(contrasts, split=split, seed=seed)
        data = data.loc[data.contrast == code, keys + ["gain"]].rename(
            columns={"gain": column})
        wide = data if wide is None else wide.merge(data, on=keys, how="inner")
    return to_display(_with_cells(wide.dropna(), metadata))


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
                      "T2": ("#408579", "Joint vs XGBoost")},
        "covariates": {"history_event_count": "Recorded history before treatment",
                       "age_at_index": "Age", "sex": "Sex",
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


def _number(value: float) -> str:
    return f"{int(value):,}" if float(value).is_integer() else f"{value:g}"


def subgroup_label(label: str) -> str:
    """Readable labels for pd.cut intervals: '(49.0, 64.0]' -> '50–64'."""
    text = str(label).strip()
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


def subgroup_table(panels, summaries, spec, *, seed=None, horizon=365):
    """Panel subgroup effects for one figure, with display labels and styles."""
    data = _select(panels, seed=seed)
    data = data.loc[(data.metric == "brier_gain") & (data.horizon_days == horizon)
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
    table["subgroup"] = table.subgroup.map(subgroup_label)
    order = {name: title for name, title in spec["covariates"].items()
             if name in set(table.covariate)}
    styles = {code: list(style) for code, style in spec["contrasts"].items()}
    return table, styles, order


def group_maps():
    diseases = {disease_name(fine): disease_group_name(group)
                for fine, group in fine_cohort_groups().items()}
    outcomes = {outcome_display_name(outcome): group
                for outcome, group in outcome_endpoint_groups().items()}
    return diseases, outcomes


def calibration_examples(choices):
    return {outcome_display_name(item["outcome"]): [disease_name(item["cohort"]),
                                                     outcome_display_name(item["outcome"])]
            for item in choices.get("calibration_examples", [])}


def build(analysis_root: Path, cell_metadata: Path, output_dir: Path, *, seed=None,
          atlas_metrics: Path | None = None, choices: Path = DEFAULT_CHOICES,
          bayes_variant="support_spline", registry_outcomes=None):
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
    summaries = lambda: _read(analysis / "effect_summaries.csv", sources)  # noqa: E731
    table("paired_cells.csv", lambda: paired_cells(contrasts(), metadata, seed=seed))
    table("support_curve.csv", lambda: support_curve([
        _read(bayes / family / "auc" / bayes_variant / "posterior_curve.csv", sources)
        for family in ("pooling", "tabular")]))
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
    table("calibration_examples.json", lambda: calibration_examples(choice_spec))

    def strategy():
        cells, selection = strategy_tables(
            _read(analysis / "estimates.csv", sources),
            _read(analysis / "strategy_selection.csv", sources), metadata, seed=seed)
        selection.to_csv(output_dir / "selected_strategy.csv", index=False)
        report["selected_strategy.csv"] = {"status": "written", "rows": len(selection)}
        return cells

    table("strategy_cells.csv", strategy)
    table("rescue_cells.csv", lambda: rescue_cells(contrasts(), metadata, seed=seed))

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
    args = parser.parse_args()
    manifest = build(args.analysis_root, args.cell_metadata, args.output_dir,
                     seed=args.seed, atlas_metrics=args.atlas_metrics,
                     choices=args.choices, bayes_variant=args.bayes_variant)
    print(json.dumps({name: item["status"] for name, item in manifest["tables"].items()},
                     indent=2))


if __name__ == "__main__":
    main()
