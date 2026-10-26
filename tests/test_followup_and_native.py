import numpy as np
import pandas as pd
import pytest

from opera.evaluation.followup import median_followup, reverse_km_quantiles
from opera.run.render_clinical_score_panels import followup_horizon, native_entries


def test_reverse_km_ignores_deaths_when_summarising_followup():
    # Ten patients censored at 100..1000 days; deaths early do not shorten it.
    time = np.r_[np.arange(100, 1001, 100), [10, 20, 30]]
    event = np.r_[np.zeros(10), np.ones(3)]
    with_deaths = median_followup(time, event)
    without = median_followup(time[:10], event[:10])
    assert with_deaths["median_days"] == without["median_days"] == 500
    # The naive median of observed time is pulled down by the deaths.
    assert np.median(time) < with_deaths["median_days"]


def test_quantile_not_reached_is_nan():
    assert np.isnan(reverse_km_quantiles([100, 200], [1, 1])[0.5])


def test_native_entries_and_followup_horizon():
    specs = [{"id": "ipi_dlbcl", "index_values": ["IPI"], "score_label": "IPI",
              "outcome": "tf", "horizon_days": 730, "manuscript_figures": True}]
    native = {"IPI": {"outcome": "overall_survival", "outcome_label": "Overall survival",
                      "death_in_endpoint": True, "original_endpoint": "OS, 5 years",
                      "horizons_days": [365, 730]}}
    entry, = native_entries(specs, native)
    assert entry["id"] == "ipi_dlbcl_native" and entry["outcome"] == "overall_survival"
    assert entry["horizon_days"] == 730 and not entry["manuscript_figures"]
    # Two years needs at least a third of the patients followed past 730 days.
    enough = pd.DataFrame({"time_days": np.r_[np.full(60, 500), np.full(40, 800)],
                           "event": 0})
    horizon, follow = followup_horizon(enough, entry)
    assert follow["median_days"] == 500 and horizon == 730  # 40% reach two years
    short = pd.DataFrame({"time_days": np.r_[np.full(70, 500), np.full(30, 800)],
                          "event": 0})
    assert followup_horizon(short, entry)[0] == 365  # only 30% do
    long = enough.assign(time_days=900)
    assert followup_horizon(long, entry)[0] == 730


def test_native_endpoint_replaces_the_shared_one_and_keeps_the_main_figure():
    from opera.run.render_clinical_score_panels import expand_index_scores
    metadata = pd.DataFrame({"index": ["IPI"] * 60, "IPI": ["low"] * 60,
                             "cohort_fine": ["DLBCL"] * 60})
    config = {"all_indices": {"template": {"outcome": "tf", "horizon_days": 730},
                              "minimum_patients": 50, "pooled": False,
                              "manuscript": {"index": "IPI", "cohort": "DLBCL"}},
              "native_endpoints": {"IPI": {"outcome": "overall_survival",
                                           "horizons_days": [730, 365]}}}
    specs = expand_index_scores(config, metadata)
    assert [s["outcome"] for s in specs] == ["overall_survival"]
    assert specs[0]["manuscript_figures"]
    shared = expand_index_scores({**config, "index_endpoint": "shared"}, metadata)
    assert sorted(s["outcome"] for s in shared) == ["overall_survival", "tf"]
