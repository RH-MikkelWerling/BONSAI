"""Every clinical index at both ends: derivation, figure, table and wiring."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
import yaml

from opera.evaluation.clinical_index_summary import (
    across_index_counts, summarise_clinical_indices, verdict)

REPO = Path(__file__).resolve().parents[1]
THREE = ["Low", "Intermediate", "High"]


def _inventory(*rows):
    base = dict(model="J_OA_OWN_L", group_order=json.dumps(THREE), status="rendered",
                reason=None, cutoff_failure_code=None, outcome_label="Overall survival",
                horizon_days=730, population_label=None)
    return pd.DataFrame([{**base, **row} for row in rows])


def _sizes(score_id, low, high, *, estimand="km_survival"):
    """Rows as cohort_group_comparisons writes them: difference = model - index."""
    rows = []
    for group, (diff, lo, hi) in zip(THREE, (low, (0.0, -0.01, 0.01), high)):
        rows.append(dict(score_id=score_id, group=group, horizon_days=730, estimand=estimand,
                         difference=diff, lower=lo, upper=hi, model_n=10, index_n=10))
    return rows


@pytest.fixture
def tables():
    inventory = _inventory(
        dict(score_id="mm_iss", score_label="ISS", cohorts='["MM"]',
             score_available_at_index_n=400),
        dict(score_id="ipi_all", score_label="IPI", cohorts='["DLBCL", "RT"]',
             population_label="all subtypes", score_available_at_index_n=900),
        dict(score_id="ipi_dlbcl", score_label="IPI", cohorts='["DLBCL"]',
             score_available_at_index_n=600),
        dict(score_id="pit_ptcl", score_label="PIT", cohorts='["PTCL"]', status="not_estimable",
             cutoff_failure_code="crossed_anchors",
             reason="Model risk-group cutoffs are not estimable: low cutoff 0.2 >= high 0.1",
             score_available_at_index_n=70),
        dict(score_id="fl_flipi", score_label="FLIPI", cohorts='["FL"]',
             outcome_label="Treatment failure", score_available_at_index_n=300))
    sizes = pd.DataFrame(
        _sizes("ipi_dlbcl", (0.04, 0.01, 0.07), (-0.06, -0.11, -0.01))
        + _sizes("mm_iss", (-0.02, -0.05, 0.01), (0.03, -0.01, 0.07))
        + _sizes("ipi_all", (0.01, -0.02, 0.04), (0.02, 0.0, 0.04))
        # Competing-risk endpoint: cumulative incidence, lower is better.
        + _sizes("fl_flipi", (-0.03, -0.05, -0.01), (0.05, 0.02, 0.08),
                 estimand="cumulative_incidence"))
    km = pd.DataFrame([
        dict(score_id="ipi_dlbcl", status="estimable", additional_low_per_1000=52.0,
             additional_low_per_1000_lower=21.0, additional_low_per_1000_upper=83.0,
             model_low_n=231, index_low_n=200, n_patients=600),
        dict(score_id="mm_iss", status="estimable", additional_low_per_1000=-10.0,
             additional_low_per_1000_lower=-40.0, additional_low_per_1000_upper=20.0,
             model_low_n=96, index_low_n=100, n_patients=400),
        dict(score_id="ipi_all", status="not_estimable",
             reason="No supported model Low group reaches native Low KM survival"),
        dict(score_id="fl_flipi", status="not_applicable",
             reason="KM Low matching is for death-inclusive endpoints")])
    detection = pd.DataFrame([
        dict(score_id=s, label=l, estimand=e, additional_events_per_1000=v + (100 if e == "known_status" else 0),
             lower=v - 5, upper=v + 5)
        for s, l, v in (("ipi_dlbcl", "IPI", 18.0), ("mm_iss", "ISS", -3.0), ("ipi_all", "IPI", 9.0),
                        ("fl_flipi", "FLIPI", 2.0))
        for e in ("known_status", "cif_ipcw")])
    return inventory, sizes, km, detection


def test_each_comparison_is_oriented_so_positive_favours_the_model(tables):
    summary = summarise_clinical_indices(*tables).set_index("score_id")
    ipi = summary.loc["ipi_dlbcl"]
    # (a) model lowest minus index lowest survival, percentage points.
    assert ipi.low_size == pytest.approx(4.0)
    assert (ipi.low_size_lower, ipi.low_size_upper) == pytest.approx((1.0, 7.0))
    # (b) index highest minus model highest: the model's highest group is worse off.
    assert ipi.high_size == pytest.approx(6.0)
    assert (ipi.high_size_lower, ipi.high_size_upper) == pytest.approx((1.0, 11.0))
    # (c) additional low-risk patients as % with the count; (d) IPCW events per 1,000.
    assert ipi.low_risk == pytest.approx(5.2) and ipi.low_risk_count == 31
    assert (ipi.low_risk_lower, ipi.low_risk_upper) == pytest.approx((2.1, 8.3))
    assert ipi.high_risk == pytest.approx(18.0) and ipi.high_risk_source.endswith("cif_ipcw")
    assert ipi.disease == "DLBCL" and ipi.n_patients == 600 and ipi.endpoint == "Overall survival"
    # Cumulative incidence: lower incidence in the model's lowest group is better.
    fl = summary.loc["fl_flipi"]
    assert fl.low_size == pytest.approx(3.0) and fl.low_size_lower == pytest.approx(1.0)
    assert fl.high_size == pytest.approx(5.0) and fl.high_size_lower == pytest.approx(2.0)
    known = summarise_clinical_indices(*tables, estimand="known_status").set_index("score_id")
    assert known.loc["ipi_dlbcl", "high_risk"] == pytest.approx(118.0)


def test_lock_rule_and_non_estimable_rows_are_kept_with_reasons(tables):
    assert verdict(0.0, -1.0) == "at_least_as_good"
    assert verdict(0.5, 0.1) == "better"
    assert verdict(-0.1, -0.5) == "not_at_least_as_good"
    assert verdict(np.nan, np.nan) == "not_estimable"
    assert verdict(2.0, np.nan) == "at_least_as_good"  # no interval, no "better"
    summary = summarise_clinical_indices(*tables).set_index("score_id")
    assert summary.loc["ipi_dlbcl", "low_size_verdict"] == "better"
    assert summary.loc["mm_iss", "low_size_verdict"] == "not_at_least_as_good"
    assert summary.loc["ipi_all", "high_size_verdict"] == "not_at_least_as_good"
    assert summary.loc["mm_iss", "both_ends_size_verdict"] == "not_at_least_as_good"
    pit = summary.loc["pit_ptcl"]
    assert pit.status == "not_estimable" and pit.reason == "low and high training cutoffs crossed"
    assert all(pit[f"{m}_verdict"] == "not_estimable"
               for m in ("low_size", "high_size", "low_risk", "high_risk"))
    assert summary.loc["ipi_all", "low_risk_verdict"] == "not_estimable"
    assert "no supported model" in summary.loc["ipi_all", "low_risk_reason"]
    counts = across_index_counts(summary.reset_index())
    assert counts["low_size"] == dict(total=5, estimable=4, better=2, at_least_as_good=3,
                                      not_at_least_as_good=1, not_estimable=1)
    assert counts["both_ends_size"]["at_least_as_good"] == 2  # IPI DLBCL, FLIPI


def test_rows_are_ordered_by_lineage_then_subtype_with_pooled_last(tables):
    summary = summarise_clinical_indices(*tables)
    assert summary.score_id.tolist() == ["ipi_dlbcl", "fl_flipi", "pit_ptcl", "mm_iss", "ipi_all"]
    assert summary.iloc[-1].disease == "All subtypes (2)"


def test_derivation_reads_the_real_clinical_comparison_outputs():
    """Rows produced by the pipeline's own functions, not hand-built schemas."""
    from opera.run.render_clinical_score_panels import (
        clinical_detection_counts, cohort_group_comparisons)
    rng = np.random.default_rng(7)
    n = 300
    risk = rng.uniform(0.02, 0.9, n)
    group = np.array(THREE)[np.digitize(risk + rng.normal(0, 0.25, n), [0.33, 0.66])]
    event = rng.binomial(1, risk)
    time = np.where(event == 1, rng.integers(20, 700, n), rng.integers(400, 1200, n))
    patients = pd.DataFrame(dict(subject_id=[str(i) for i in range(n)], time_days=time,
                                 event=event, model_risk=risk, ipi_group=group,
                                 ipi_score=pd.Series(group).map({g: i + 1 for i, g in enumerate(THREE)})))
    spec = dict(id="ipi_dlbcl", score_label="IPI", outcome="overall_survival", horizon_days=365,
                group_order=THREE, death_in_endpoint=True, score_high_group="High")
    sizes = pd.DataFrame(cohort_group_comparisons(patients, spec, draws=30))
    detection = pd.DataFrame(clinical_detection_counts(patients, spec, dict(high_cutoff=0.6),
                                                       draws=50))
    inventory = _inventory(dict(score_id="ipi_dlbcl", score_label="IPI", cohorts='["DLBCL"]',
                                score_available_at_index_n=n, horizon_days=365))
    row = summarise_clinical_indices(inventory, sizes, pd.DataFrame(), detection).iloc[0]
    low = sizes.loc[sizes.group.eq("Low")].iloc[0]
    high = sizes.loc[sizes.group.eq("High")].iloc[0]
    assert row.low_size == pytest.approx(100 * low.difference)
    assert row.high_size == pytest.approx(-100 * high.difference)
    assert row.high_size_lower == pytest.approx(-100 * high.upper)
    ipcw = detection.loc[detection.estimand.eq("cif_ipcw")].iloc[0]
    assert row.high_risk == pytest.approx(ipcw.additional_events_per_1000)
    assert row.low_risk_verdict == "not_estimable"


def test_risk_matched_recipe_falls_back_to_headline_points_without_intervals():
    inventory = _inventory(dict(score_id="ipi_dlbcl", score_label="IPI", cohorts='["DLBCL"]',
                                score_available_at_index_n=500))
    headline = pd.DataFrame([dict(score_id="ipi_dlbcl", grouping_rule="risk_matched_low",
                                  model_low_per_1000=300.0, index_low_per_1000=250.0,
                                  n_known=400, extra_events_flagged_per_1000=12.0)])
    row = summarise_clinical_indices(inventory, pd.DataFrame(), pd.DataFrame(), pd.DataFrame(),
                                     headline).iloc[0]
    assert row.low_risk == pytest.approx(5.0) and row.low_risk_count == 20
    assert np.isnan(row.low_risk_lower) and row.low_risk_verdict == "at_least_as_good"
    assert row.high_risk == pytest.approx(12.0) and row.high_risk_source == "headline_scores:known_status"
    assert row.low_size_verdict == "not_estimable"
    assert "risk_matched_low" in row.low_size_reason


def test_figure_keeps_every_row_greys_the_unestimable_and_highlights_none(tables):
    from opera.visualization.paper_figures import figure_clinical_index_summary
    from opera.visualization.layout_audit import audit_layout
    fig = figure_clinical_index_summary(*tables, mark="SYNTHETIC")
    texts = [t.get_text() for t in fig.findobj(matplotlib.text.Text)]
    assert "Not estimable: low and high training cutoffs crossed" in texts
    labels = [t.get_text() for t in fig.axes[1].get_yticklabels()]
    assert labels == ["IPI (n=600)", "FLIPI (n=300)", "PIT (n=70)", "ISS (n=400)", "IPI (n=900)"]
    titles = [ax.get_title(loc="left") for ax in fig.axes]
    assert {"Lowest group", "Highest group", "Low-risk group", "High-risk group"} <= set(titles)
    # Bold panel letters come from align_panel_titles.
    assert {"A", "B", "C", "D"} <= {t.get_text() for t in fig.findobj(matplotlib.text.Text)
                                    if t.get_fontweight() == "bold"}
    assert "SYNTHETIC" in texts
    assert not audit_layout(fig)["text_collision_candidates"]
    # No single index is emphasised: every tick label uses the same weight.
    assert {t.get_fontweight() for t in fig.axes[1].get_yticklabels()} == {"normal"}
    plt.close(fig)


def test_numbers_builder_writes_table_rows_and_across_index_macros(tables, tmp_path):
    from opera.run.build_paper_numbers import build
    inventory, sizes, km, detection = tables
    clinical = tmp_path / "clinical_scores"
    clinical.mkdir()
    inventory.to_csv(clinical / "score_inventory.csv", index=False)
    sizes.to_csv(clinical / "size_matched_survival_comparisons.csv", index=False)
    km.to_csv(clinical / "km_matched_low_groups.csv", index=False)
    detection.to_csv(clinical / "clinical_operating_point_counts.csv", index=False)
    pd.DataFrame([]).to_csv(clinical / "headline_scores.csv", index=False)  # empty, as written
    numbers = build(tmp_path, None, tmp_path / "numbers", clinical_scores=clinical)
    assert numbers["clinical_indices_total"] == "5"
    assert numbers["clinical_indices_not_estimable"] == "1"
    assert numbers["clinical_low_size_atleast"] == "3 of 5"
    assert numbers["clinical_low_size_better"] == "2 of 5"
    assert numbers["clinical_both_size_atleast"] == "2 of 5"
    assert numbers["clinical_high_risk_median"] == "6 (range $-$3 to 18)"
    rows = (tmp_path / "numbers" / "clinical_index_rows.tex").read_text()
    assert rows.rstrip().endswith("\\bottomrule")
    assert rows.count("\\\\") == 5
    assert "\\textbf{4.0 (1.0 to 7.0)}" in rows and "+31" in rows
    assert "Not estimable: low and high training cutoffs crossed" in rows
    macros = (tmp_path / "numbers" / "results_macros.tex").read_text()
    assert "opera@res@clinical_low_size_atleast}{3 of 5}" in macros
    summary = pd.read_csv(tmp_path / "numbers" / "clinical_index_summary.csv")
    assert "subject_id" not in summary and len(summary) == 5


def test_recipe_registers_the_plate_and_the_renderer_reads_empty_tables(tmp_path):
    import inspect
    from opera.run.render_paper_results import read_input_table
    from opera.visualization import paper_figures
    recipe = yaml.safe_load((REPO / "opera/configs/paper_results_figures.full.example.yaml").read_text())
    plate = next(p for p in recipe["figures"] if p["id"] == "supp_clinical_index_summary")
    assert plate["id"] in recipe["required_figures"]
    assert all("/clinical_scores/" in source for source in plate["tables"].values())
    function = getattr(paper_figures, plate["function"])
    inspect.signature(function).bind(**plate.get("options", {}),
                                      **{key: None for key in plate["tables"]}, mark=None)
    empty = tmp_path / "empty.csv"
    pd.DataFrame([]).to_csv(empty, index=False)
    assert read_input_table(empty).empty
    supplement = (REPO / "paper/opera_supplement_figures.tex").read_text()
    assert "\\resultfigure{supp_clinical_index_summary}" in supplement
    assert "\\label{fig:clinical-index-summary}" in supplement
    assert "\\input{tables/clinical_index_summary}" in supplement
    table = (REPO / "paper/tables/clinical_index_summary.tex").read_text()
    for label in ("tab:clinical-index-summary", "tab:clinical-index-sizes", "tab:clinical-index-matched"):
        assert f"\\label{{{label}}}" in table
