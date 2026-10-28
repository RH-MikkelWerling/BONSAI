import pandas as pd

from opera.run.analyse_locked_predictions import run
from opera.run.build_paper_numbers import build as build_numbers, percent_ci
from opera.run.build_paper_plot_tables import build as build_tables
from test_secondary_analysis import predictions


def test_numbers_and_table_rows_come_from_the_aggregate_outputs(tmp_path):
    frame = predictions()
    frame.to_csv(tmp_path / "p.csv", index=False)
    pd.DataFrame([{"cohort": "CLL", "cohort_group": "CLL", "outcome": o, "horizon_days": 365,
                   "n_events_train": 12, "n_patients_train": 120,
                   "endpoint_group": "Infection", "outcome_family": "t"}
                  for o in ["a", "b"]]).to_csv(tmp_path / "c.csv", index=False)
    run({"estimand": "binary_known_status_unweighted",
         "artifacts": [{"path": str(tmp_path / "p.csv")}],
         "contrast_manifest": "opera/configs/manifests/final_contrasts_20261014.yaml",
         "cell_metadata": str(tmp_path / "c.csv"), "bootstrap_replicates": 30,
         "cell_groups": {"subtype_size": {"column": "n_patients_train", "provenance": "train_only",
                                          "bins": [-1, 499, float("inf")]},
                         "no_index_subtype": {"column": "cohort", "provenance": "train_only",
                                              "label": "no_index", "values": ["CLL"]}}},
        tmp_path / "analysis" / "prediction_analysis")
    build_tables(tmp_path / "analysis", tmp_path / "c.csv", tmp_path / "plot")
    numbers = build_numbers(tmp_path / "analysis", tmp_path / "c.csv", tmp_path / "out",
                            plot_tables=tmp_path / "plot")
    assert numbers["P5_better"].endswith("of 2")
    assert "95\\% CI" in numbers["P5_auc"]
    assert numbers["cells_low_support"] == "2"
    # Prespecified strata: every cell here is a rare subtype without an index.
    assert numbers["P5_auc_rare"] == numbers["P5_auc"] == numbers["P5_auc_noindex"]
    macros = (tmp_path / "out" / "results_macros.tex").read_text()
    assert r"\@namedef{opera@res@P5_auc}" in macros
    rows = (tmp_path / "out" / "panel_effects_rows.tex").read_text()
    assert rows.rstrip().endswith(r"\bottomrule")  # tables cannot end with \IfFileExists
    assert "\\opera{} (local) vs per-task XGBoost & -- " in rows  # absent comparisons stay explicit
    assert rows.splitlines()[0].startswith(r"\rlap{\textit{")  # grouped by question
    assert rows.splitlines()[1].count("&") == 6  # seven columns, rare subtypes included


def test_percent_ci_is_a_wilson_interval():
    assert percent_ci(20, 100) == "20\\% (95\\% CI 13--29)"
    assert percent_ci(0, 0) is None


def test_gap_restriction_numbers_are_available_for_manuscript(tmp_path):
    from opera.run.build_paper_numbers import restriction_numbers
    path = tmp_path / 'restrictions.csv'
    pd.DataFrame([dict(group='All outcomes', restriction='Last 14 days hidden',
                       mean_delta=-.3, lower=-.5, upper=-.1)]).to_csv(path, index=False)
    sources = {}
    result = restriction_numbers(path, sources)
    assert 'restrict_gap_14d' in result


def test_selected_and_joint_comparisons_keep_distinct_labels():
    from opera.run.build_paper_numbers import table_rows, decision_numbers
    panel = pd.DataFrame([
        dict(contrast=code, cells=12, auc_gain=value, auc_interval='interval')
        for code, value in [('O1', .02), ('T2', -.01)]])
    rows = table_rows(panel)
    assert '\\opera{} (local) vs per-task XGBoost & 12 & 0.020' in rows
    assert 'Joint vs per-task XGBoost & 12 & $-$0.010' in rows
    points = pd.DataFrame([dict(model=model, horizon_days=365, status='estimated',
        rule='tuning_specificity:0.90', sensitivity=sens, mcc=.2)
        for model, sens in [('OPERA', .7), ('J_L', .4), ('X', .5)]])
    result = decision_numbers(points)
    assert result['dp90_sens_opera'] == '0.70'
    assert result['dp90_sens_joint'] == '0.40'


def test_national_opera_rows_do_not_use_joint_model_effects():
    from opera.run.build_paper_numbers import table_rows
    panel = pd.DataFrame([dict(contrast=code,cells=12,auc_gain=value,auc_interval='interval')
        for code,value in [('E1',.01),('E2',.02),('E3',.03),('E4',.04),('T5_router',.05)]])
    rows = table_rows(panel)
    assert '\\opera{} (national) vs \\opera{} (local) & 12 & 0.020' in rows
    assert '\\opera{} (national + adaptation) vs \\opera{} (local) & 12 & 0.040' in rows
    assert '\\opera{} (national + adaptation) vs per-task XGBoost & 12 & 0.050' in rows
    assert 'National vs cohort encoder & 12 & 0.010' in rows
