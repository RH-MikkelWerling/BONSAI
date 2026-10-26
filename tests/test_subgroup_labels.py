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
