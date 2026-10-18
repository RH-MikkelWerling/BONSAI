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
    assert "Joint vs per-task XGBoost & -- " in rows  # absent comparisons stay explicit
    assert rows.splitlines()[0].count("&") == 6  # seven columns, rare subtypes included


def test_percent_ci_is_a_wilson_interval():
    assert percent_ci(20, 100) == "20\\% (95\\% CI 13--29)"
    assert percent_ci(0, 0) is None
