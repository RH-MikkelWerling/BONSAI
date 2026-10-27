import pandas as pd

from opera.run.build_paper_plot_tables import _subgroup_sort_key, subgroup_label
from opera.visualization.paper_figures import _region_name


def test_region_codes_become_names():
    assert subgroup_label("1084") == "Hovedstaden"
    assert subgroup_label("1081.0") == "Nordjylland"
    assert _region_name("1085") == "Sjælland"
    assert _region_name("Region Midtjylland") == "Midtjylland"


def test_subgroups_sort_in_reading_order():
    labels = ["High", "missing", "Intermediate", "Low", "Very high", "(65.0, 80.0]",
              "(-inf, 50.0]"]
    ordered = sorted(labels, key=_subgroup_sort_key)
    assert [subgroup_label(x) for x in ordered] == [
        "≤ 50", "66–80", "Low", "Intermediate", "High", "Very high", "Not recorded"]


def test_scatter_drops_imprecise_cells():
    from opera.visualization.paper_figures import estimable_cells
    cells = pd.DataFrame({"gain": [0.02, -0.9, 0.5, None],
                          "n_events_train": [50, 3, 200, 80],
                          "lower": [-0.01, -1.0, -0.2, 0], "upper": [0.05, 0.1, 0.9, 0]})
    assert estimable_cells(cells).gain.tolist() == [0.02]


def test_calibration_examples_skip_rare_cells():
    from opera.run.build_paper_plot_tables import calibration_examples
    metadata = pd.DataFrame({
        "cohort": ["DLBCL", "MM", "CLL"], "outcome": ["sepsis", "sepsis", "sepsis"],
        "horizon_days": 365, "endpoint_group": "Infection and antimicrobial use",
        "n_events_train": [900, 400, 150], "n_patients_train": [8000, 2000, 500]})
    rule = {"domains": ["Infection and antimicrobial use"], "min_events_train": 200,
            "min_event_share": 0.10}
    examples = calibration_examples({"calibration_example_rule": rule}, metadata)
    # DLBCL: most events, share 900/8000 = 11%
    assert [v[0] for v in examples.values()] == ["DLBCL"]
    rule["min_event_share"] = 0.15
    examples = calibration_examples({"calibration_example_rule": rule}, metadata)
    assert [v[0] for v in examples.values()] == ["MM"]  # CLL: too few events
