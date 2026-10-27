"""Analysis outputs become figure-ready plot tables without new estimation."""

import json

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest
import pandas as pd

from opera.evaluation.secondary_analysis import panel_estimates
from opera.run import build_paper_plot_tables as plot_tables
from opera.run.build_paper_cell_metadata import build_cell_metadata
from opera.visualization import paper_figures as figures
from opera.visualization.outcome_names import outcome_display_name

CELLS = [("DLBCL", "treatment_failure"), ("SolM", "anemia_g3plus"),
         ("MM", "sepsis"), ("CLL", "neutropenia_g3plus")]


def cell_metadata():
    return pd.DataFrame([
        {"cohort": c, "cohort_group": g, "outcome": o, "horizon_days": 365,
         "endpoint_group": e, "n_events_train": n}
        for (c, o), g, e, n in zip(
            CELLS, ["DLBCL_like", "MM", "MM", "CLL_SLL"],
            ["Disease course and later malignancy", "Erythroid findings and transfusion",
             "Infection and antimicrobial use", "Neutrophil, white-cell and platelet findings"],
            [400, 0, 60, 120])
    ])


def analysis_root(tmp_path):
    root = tmp_path / "analysis"
    out = root / "prediction_analysis"
    out.mkdir(parents=True)
    rng = np.random.default_rng(0)
    contrasts = []
    for split in ("tuning", "held_out"):
        for cohort, outcome in CELLS:
            for code in ("P1", "P2", "P5", "P6", "T2", "E3", "R1", "router_vs_specialist",
                         "I1"):
                for metric in ("auc", "brier"):
                    gain = rng.normal(0.01, 0.01)
                    contrasts.append({"split": split, "seed": 42, "cohort": cohort,
                                      "outcome": outcome, "horizon_days": 365,
                                      "contrast": code, "metric": metric, "gain": gain,
                                      "lower": gain - 0.01, "upper": gain + 0.01})
    pd.DataFrame(contrasts).to_csv(out / "contrasts.csv", index=False)
    summaries = [{"split": "held_out", "seed": 42, "contrast": code, "metric": "auc",
                  "horizon_days": 365, "stratum": "all_cells", "stratum_value": "all",
                  "gain": 0.01, "lower": 0.0, "upper": 0.02, "n_cells": 4}
                 for code in ("E1", "E2", "A1", "A2", "E3", "E4", "R0", "R2", "R3")]
    pd.DataFrame(summaries).to_csv(out / "effect_summaries.csv", index=False)
    estimates = [{"split": "held_out", "seed": 42, "cohort": c, "outcome": o,
                  "horizon_days": 365, "model": m, "metric": "auc",
                  "estimate": 0.7 + 0.01 * i}
                 for c, o in CELLS for i, m in enumerate(["S_L", "J_L", "BEST_TUNING"])]
    pd.DataFrame(estimates).to_csv(out / "estimates.csv", index=False)
    pd.DataFrame([{"seed": 42, "cohort": c, "outcome": o, "status": "selected_on_tuning",
                   "selected_model": "D_L"} for c, o in CELLS]).to_csv(
        out / "strategy_selection.csv", index=False)
    pd.DataFrame([{"split": "held_out", "seed": 42, "cohort": c, "outcome": o,
                   "horizon_days": 365, "contrast": "P5",
                   "status": "paired" if c != "SolM" else "missing_models"}
                  for c, o in CELLS]).to_csv(out / "contrast_coverage.csv", index=False)
    pd.DataFrame([{"split": "held_out", "seed": 42, "model": m, "cohort": c, "outcome": o,
                   "horizon_days": 365, "status": "planned",
                   "prediction_status": "absent" if (c, m) == ("SolM", "S_L")
                   else "single_class" if c == "CLL" else "two_class"}
                  for c, o in CELLS for m in ("S_L", "J_L")]).to_csv(
        out / "model_coverage.csv", index=False)
    calibration = [{"split": "held_out", "seed": 42, "model": m, "cohort": c, "outcome": o,
                    "horizon_days": 365, "status": "estimated", "calibration_slope": 1.0,
                    "calibration_in_the_large": 0.0}
                   for c, o in CELLS for m in ("J_L", "R_L", "S_L", "X", "N_L")]
    pd.DataFrame(calibration).to_csv(out / "calibration_summary.csv", index=False)
    reliability = [{"split": "held_out", "seed": 42, "model": m, "cohort": c, "outcome": o,
                    "horizon_days": 365, "probability_bin": b, "n": 50,
                    "predicted_weighted_mean": b / 10, "observed_fraction": b / 10}
                   for c, o in CELLS for m in ("J_L", "X") for b in range(5)]
    pd.DataFrame(reliability).to_csv(out / "calibration.csv", index=False)
    panel = [{"split": "held_out", "seed": 42, "horizon_days": 365, "metric": "auc",
              "group": "encoder_origin", "model": m, "estimate": 0.75, "lower": 0.74,
              "upper": 0.76, "n_cells": 4}
             for m in ("J_L", "R_L", "J_E", "R_E", "J_EA", "R_EA")]
    pd.DataFrame(panel).to_csv(out / "panel_estimates.csv", index=False)
    return root


def test_plot_tables_translate_codes_and_names_and_report_missing_inputs(tmp_path):
    metadata = tmp_path / "cell_metadata.csv"
    cell_metadata().to_csv(metadata, index=False)
    output = tmp_path / "plot_tables"
    output.mkdir()
    (output / "support_curve.csv").write_text("stale,previous_run\n1,2\n")
    (output / "atlas.csv").write_text("stale\n1\n")
    manifest = plot_tables.build(analysis_root(tmp_path), metadata, output, seed=42)
    tables = manifest["tables"]
    assert tables["paired_cells.csv"]["status"] == "written"
    assert tables["support_curve.csv"]["status"] == "missing_input"
    assert tables["support_curve.csv"]["removed_stale_output"]
    assert not (output / "support_curve.csv").exists()
    assert not (output / "atlas.csv").exists()
    assert tables["patient_coordinates.csv"]["status"] == "pending_upstream_builder"

    cells = pd.read_csv(output / "paired_cells.csv")
    assert {"D-S", "O-S", "J-S", "interaction", "J-X", "EA-L:J", "R1"} == set(cells.contrast)
    assert "Solitary plasmacytoma" in set(cells.cohort)
    assert "Myeloma" in set(cells.cohort_group)
    assert not cells.outcome.str.contains("_").any()  # display names, not identifiers
    assert len(pd.read_csv(output / "external_panel.csv")) == 6

    coverage = pd.read_csv(output / "coverage.csv").set_index(["cohort", "outcome"]).status
    names = {o: outcome_display_name(o) for _, o in CELLS}
    assert coverage[("DLBCL", names["treatment_failure"])] == "evaluated"
    assert coverage[("Solitary plasmacytoma", names["anemia_g3plus"])] == "no_training_events"
    assert coverage[("CLL", names["neutropenia_g3plus"])] == "single_class_test"
    assert coverage[("HL", names["sepsis"])] == "not_applicable"

    summary = pd.read_csv(output / "calibration_summary.csv")
    assert set(summary.model) == {"J", "R", "S", "X"}  # source-bank variants dropped
    examples = json.loads((output / "calibration_examples.json").read_text())
    assert len(examples) == 4

    rendered = [
        figures.figure_pooling_cells(cells),
        figures.figure_external(pd.read_csv(output / "origin_estimates.csv"), cells,
                                pd.read_csv(output / "external_panel.csv")),
        figures.figure_strategy(pd.read_csv(output / "strategy_cells.csv"),
                                pd.read_csv(output / "selected_strategy.csv")),
        figures.figure_rescue(pd.read_csv(output / "rescue_cells.csv")),
        figures.figure_coverage(pd.read_csv(output / "coverage.csv")),
    ]
    for fig in rendered:
        plt.close(fig)


def test_panel_estimates_use_only_cells_where_every_model_was_evaluated():
    rows, draws = [], []
    for cohort in ("A", "B", "C"):
        for model, value in (("J_L", 0.7), ("J_E", 0.8)):
            if cohort == "C" and model == "J_E":
                continue  # C is not on the common panel
            identity = {"split": "held_out", "seed": 42, "cohort": cohort, "outcome": "o",
                        "horizon_days": 365, "model": model, "metric": "auc"}
            rows.append({**identity, "estimate": value + (0.1 if cohort == "C" else 0)})
            draws.append({**identity, "estimate_draws": np.full(50, value)})
    panel = panel_estimates(pd.DataFrame(rows), pd.DataFrame(draws),
                            {"origin": ["J_L", "J_E"]})
    assert panel.set_index("model").estimate.to_dict() == {"J_L": 0.7, "J_E": 0.8}
    assert set(panel.n_cells) == {2}
    assert np.allclose(panel.lower, panel.estimate)


def test_cell_metadata_uses_train_labels_and_clinical_groups():
    calls = []

    def fake_audit(registry, *, outcomes, horizon_days, include_held_out, training_config):
        calls.append(include_held_out)
        return pd.DataFrame([{"cohort_grouped": "DLBCL_like", "cohort": "DLBCL",
                              "outcome": o, "horizon_days": horizon_days,
                              "n_primary_events_train": 7} for o in outcomes])

    cells = build_cell_metadata({}, {"outcomes": {"sepsis": {}, "MACE_composite": {}}},
                                horizons=(90, 365), audit=fake_audit)
    assert calls == [False, False]  # held-out labels are never read
    assert len(cells) == 4 and set(cells.n_events_train) == {7}
    assert cells.set_index("outcome").endpoint_group.to_dict() == {
        "sepsis": "Infection and antimicrobial use",
        "MACE_composite": "Cardiovascular and thrombotic events"}


def test_model_cells_hold_each_models_absolute_estimate():
    import pandas as pd

    from opera.run.build_paper_plot_tables import model_cells

    estimates = pd.DataFrame({
        "split": "held_out", "seed": 42, "cohort": "DLBCL", "outcome": "sepsis",
        "horizon_days": 365, "metric": "auc", "model": ["J_L", "X", "XR"],
        "estimate": [0.8, 0.75, 0.77]})
    metadata = pd.DataFrame({"cohort": ["DLBCL"], "outcome": ["sepsis"], "horizon_days": [365],
                             "cohort_group": ["DLBCL_like"], "endpoint_group": ["Infection"],
                             "n_events_train": [40]})
    wide = model_cells(estimates, metadata, seed=42)
    assert wide.loc[0, ["J", "X", "XR"]].tolist() == [0.8, 0.75, 0.77]
    assert wide.loc[0, "n_events_train"] == 40


def test_stable_panel_keeps_intervals_when_a_small_cell_fails_some_replicates():
    rows, draws = [], []
    for cohort in ("A", "B", "C"):
        for model, value in (("J_L", 0.7), ("X", 0.6)):
            identity = {"split": "held_out", "seed": 42, "cohort": cohort, "outcome": "o",
                        "horizon_days": 365, "model": model, "metric": "auc"}
            rows.append({**identity, "estimate": value})
            sampled = np.full(50, value)
            if cohort == "C" and model == "X":
                sampled[::3] = np.nan  # a small cell: one class in some resamples
            draws.append({**identity, "estimate_draws": sampled})
    rows, draws = pd.DataFrame(rows), pd.DataFrame(draws)
    strict = panel_estimates(rows, draws, {"region": ["J_L", "X"]})
    # One unstable cell voids that model's interval.
    assert strict.set_index("model").lower.isna().to_dict() == {"J_L": False, "X": True}
    stable = panel_estimates(rows, draws, {"region": ["J_L", "X"]}, stable_cells_only=True)
    assert stable.lower.notna().all()
    assert set(stable.n_cells) == {2}  # C dropped for both models alike


def test_effect_summaries_stable_panel_reports_dropped_cells():
    from opera.evaluation.secondary_analysis import effect_summaries
    contrasts, draws = [], []
    for cohort in ("A", "B", "C"):
        cell = {"split": "held_out", "seed": 42, "cohort": cohort, "outcome": "o",
                "horizon_days": 365, "contrast": "T2", "metric": "auc"}
        contrasts.append({**cell, "gain": 0.02})
        sampled = np.full(50, 0.02)
        if cohort == "C":
            sampled[::3] = np.nan
        draws.append({**cell, "gain_draws": sampled})
    metadata = pd.DataFrame({"cohort": ["A", "B", "C"], "outcome": "o", "horizon_days": 365,
                             "no_strata": "all"})
    args = (pd.DataFrame(contrasts), pd.DataFrame(draws), metadata)
    _, strict = effect_summaries(*args, strata=["no_strata"])
    _, stable = effect_summaries(*args, strata=["no_strata"], stable_cells_only=True)
    pick = lambda t: t.loc[(t.stratum == "all_cells") & (t.horizon_days == 365)].iloc[0]  # noqa: E731
    assert np.isnan(pick(strict).lower)
    assert pick(stable).lower == pytest.approx(0.02)
    assert pick(stable).n_cells == 2 and pick(stable).n_cells_unstable == 1


def test_panel_estimates_can_pool_every_horizon():
    rows, draws = [], []
    for horizon, value in ((365, 0.7), (730, 0.6)):
        identity = {"split": "held_out", "seed": 42, "cohort": "A", "outcome": "o",
                    "horizon_days": horizon, "model": "J_L", "metric": "auc"}
        rows.append({**identity, "estimate": value})
        draws.append({**identity, "estimate_draws": np.full(50, value)})
    panel = panel_estimates(pd.DataFrame(rows), pd.DataFrame(draws), {"g": ["J_L"]},
                            pooled_horizons=True)
    pooled = panel.loc[panel.horizon_days == 0].iloc[0]
    assert pooled.estimate == pytest.approx(0.65) and pooled.n_cells == 2
    assert set(panel.horizon_days) == {0, 365, 730}


def test_cells_pool_over_horizons_with_shared_draws():
    from opera.evaluation.secondary_analysis import pool_cells_over_horizons
    rng = np.random.default_rng(0)
    contrasts, draws = [], []
    for cohort in ("A", "B"):
        for horizon, shift in ((365, 0.01), (730, 0.03)):
            cell = {"split": "held_out", "seed": 42, "cohort": cohort, "outcome": "o",
                    "contrast": "T2", "metric": "auc", "horizon_days": horizon}
            sampled = shift + rng.normal(0, 0.01, 200)
            if cohort == "B" and horizon == 730:
                sampled[:100] = np.nan  # half the replicates fail at one horizon
            contrasts.append({**cell, "gain": shift})
            draws.append({**cell, "gain_draws": sampled})
    pooled = pool_cells_over_horizons(pd.DataFrame(contrasts), pd.DataFrame(draws))
    pooled = pooled.set_index("cohort")
    assert set(pooled.horizon_days) == {0} and set(pooled.n_horizons) == {2}
    assert pooled.gain["A"] == pytest.approx(0.02)
    expected = np.mean([d["gain_draws"] for d in draws[:2]], axis=0)
    assert pooled.lower["A"] == pytest.approx(np.quantile(expected, 0.025))
    assert np.isnan(pooled.lower["B"])  # too few finite pooled replicates


def test_descriptive_support_curve_fallback():
    from opera.run.build_paper_plot_tables import descriptive_support_curve
    rng = np.random.default_rng(0)
    events = rng.integers(5, 3000, 300)
    cells = pd.DataFrame({"contrast": "J-S", "metric": "auc", "horizon_days": 0,
                          "n_events_train": events,
                          "gain": 0.04 - 0.01 * np.log10(events) + rng.normal(0, 0.01, 300)})
    curve = descriptive_support_curve(cells, n_boot=50)
    assert set(curve.source) == {"descriptive_fallback"} and set(curve.contrast) == {"J-S"}
    assert (curve.lower <= curve["mean"]).all() and (curve["mean"] <= curve.upper).all()
    assert curve["mean"].iloc[0] > curve["mean"].iloc[-1]  # the gain falls with support


def test_calibration_fit_returns_nan_on_separation_not_a_wild_slope():
    import warnings
    from opera.evaluation.secondary_analysis import _calibration_fit
    rng = np.random.default_rng(0)
    p = rng.uniform(0.05, 0.6, 2000)
    y = (rng.uniform(size=2000) < p).astype(float)
    a, b = _calibration_fit(y, p, np.ones(2000))
    assert abs(a) < 0.3 and abs(b - 1) < 0.2  # well calibrated by construction
    separated = np.r_[np.full(50, 0.01), np.full(50, 0.02)]
    labels = np.r_[np.zeros(50), np.ones(50)]  # perfectly ordered
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # no overflow warnings either
        assert np.isnan(_calibration_fit(labels, separated, np.ones(100))).all()


def test_subgroup_auc_uses_pooled_auc_only_and_matches_brier_labels():
    spec = plot_tables.SUBGROUP_FIGURES["patient_subgroups"]
    effects = pd.DataFrame([
        {"split": "held_out", "seed": 42, "horizon_days": h, "metric": m,
         "covariate": "age_at_index", "stratum_value": "(50.0, 65.0]",
         "contrast": "P5", "gain": g, "lower": g - 0.01, "upper": g + 0.01}
        for h, m, g in [(0, "auc", 0.03), (365, "auc", 0.09), (0, "brier", 0.004)]
    ])
    table = plot_tables.subgroup_auc_table(effects, spec, seed=42)
    assert len(table) == 1
    assert table.iloc[0].gain == 0.03
    assert table.iloc[0].subgroup == "51–65"


def test_who_benefits_shares_rows_even_when_auc_is_not_estimable(tmp_path):
    from opera.visualization.style import align_panel_titles, save_fig
    brier = pd.DataFrame([
        {"covariate": "sex", "subgroup": s, "contrast": "P5", "gain": 0.01,
         "lower": 0.0, "upper": 0.02} for s in ["Female", "Male"]])
    auc = brier.iloc[:1].copy()
    fig = figures.figure_who_benefits(
        brier, auc, contrast_styles={"P5": ("blue", "Joint vs specialist")},
        covariate_order={"sex": "Sex"}, mark="SYNTHETIC")
    assert [t.get_text() for t in fig.axes[0].get_yticklabels()] == ["Female", "Male"]
    assert len(fig.axes[0].collections) == 1
    assert len(fig.axes[1].collections) == 2
    save_fig(fig, str(tmp_path / "who.png"))
    positions = [ax._left_title.get_position() for ax in fig.axes]
    align_panel_titles(fig)
    assert positions == [ax._left_title.get_position() for ax in fig.axes]
    assert (tmp_path / "who.pdf").is_file()
    plt.close(fig)


def test_missing_calibration_invalidates_both_existing_outputs(tmp_path):
    metadata = tmp_path / 'cells.csv'
    cell_metadata().to_csv(metadata, index=False)
    root = analysis_root(tmp_path)
    (root / 'prediction_analysis/calibration.csv').unlink()
    output = tmp_path / 'plots'
    output.mkdir()
    for name in ['calibration_summary.csv', 'reliability.csv']:
        (output / name).write_text('old\n1\n')
    report = plot_tables.build(root, metadata, output, seed=42)
    for name in ['calibration_summary.csv', 'reliability.csv']:
        assert not (output / name).exists()
        assert report['tables'][name]['removed_stale_output']
