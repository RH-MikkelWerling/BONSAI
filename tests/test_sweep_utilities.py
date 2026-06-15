"""Unit tests for the pure utility functions in ``opera.run.sweep``.

These cover the results-aggregation helpers that turn per-cell ``metrics.json``
dicts into the wide comparison table and LaTeX output for the paper, plus the
rank-normalization used by the IPI baseline. All functions here are pure and
require no data, model, or subprocess.
"""

import numpy as np
import pandas as pd

from opera.run.sweep import (
    build_results_table,
    flatten_metrics,
    rank_normalize_scores,
    to_latex_table,
)


# ── flatten_metrics ─────────────────────────────────────────────────────────


def test_flatten_metrics_empty_returns_empty_dict():
    assert flatten_metrics({}) == {}


def test_flatten_metrics_discrimination_fields():
    metrics = {
        "discrimination": {
            "auroc": 0.81,
            "auprc": 0.42,
            "sensitivity": 0.7,
            "specificity": 0.6,
            "f1": 0.5,
            "n_total": 120,
            "n_positive": 30,
            "prevalence": 0.25,
        }
    }
    flat = flatten_metrics(metrics)
    assert flat["auroc"] == 0.81
    assert flat["auprc"] == 0.42
    assert flat["sensitivity"] == 0.7
    assert flat["specificity"] == 0.6
    assert flat["f1"] == 0.5
    assert flat["n_total"] == 120
    assert flat["n_positive"] == 30
    assert flat["prevalence"] == 0.25


def test_flatten_metrics_bootstrap_ci_fields():
    metrics = {
        "bootstrap_ci": {
            "auroc": {"lower": 0.75, "upper": 0.87},
            "auprc": {"lower": 0.30, "upper": 0.55},
        }
    }
    flat = flatten_metrics(metrics)
    assert flat["auroc_lower"] == 0.75
    assert flat["auroc_upper"] == 0.87
    assert flat["auprc_lower"] == 0.30
    assert flat["auprc_upper"] == 0.55


def test_flatten_metrics_survival_per_horizon():
    metrics = {
        "survival": {
            "concordance_index": 0.68,
            "n_total": 200,
            "per_horizon": {
                "365d": {"ipcw_auc": 0.72, "ipcw_brier": 0.18},
                "730d": {"ipcw_auc": 0.70, "ipcw_brier": 0.20},
            },
        }
    }
    flat = flatten_metrics(metrics)
    assert flat["concordance_index"] == 0.68
    assert flat["n_total_survival"] == 200
    assert flat["ipcw_auc_365d"] == 0.72
    assert flat["ipcw_brier_365d"] == 0.18
    assert flat["ipcw_auc_730d"] == 0.70
    assert flat["ipcw_brier_730d"] == 0.20


def test_flatten_metrics_nan_values_preserved():
    # A survival block present but missing concordance_index yields NaN, not a
    # dropped key — downstream tables rely on the column existing.
    metrics = {"survival": {"n_total": 50}}
    flat = flatten_metrics(metrics)
    assert "concordance_index" in flat
    assert np.isnan(flat["concordance_index"])


def test_flatten_metrics_prefix_applied():
    metrics = {"discrimination": {"auroc": 0.9}}
    flat = flatten_metrics(metrics, prefix="opera_")
    assert flat == {"opera_auroc": 0.9}


# ── build_results_table ─────────────────────────────────────────────────────


def test_build_results_table_empty_returns_empty_df():
    df = build_results_table([])
    assert isinstance(df, pd.DataFrame)
    assert df.empty


def test_build_results_table_single_row():
    results = [
        {
            "cohort": "dlbcl",
            "outcome": "mortality_1y",
            "variant": "opera",
            "metrics": {"discrimination": {"auroc": 0.8}},
        }
    ]
    df = build_results_table(results)
    assert len(df) == 1
    assert df.iloc[0]["cohort"] == "dlbcl"
    assert df.iloc[0]["outcome"] == "mortality_1y"
    assert df.iloc[0]["opera__auroc"] == 0.8


def test_build_results_table_pivots_variants_to_columns():
    results = [
        {
            "cohort": "dlbcl",
            "outcome": "mortality_1y",
            "variant": "opera",
            "metrics": {"discrimination": {"auroc": 0.82}},
        },
        {
            "cohort": "dlbcl",
            "outcome": "mortality_1y",
            "variant": "dapt",
            "metrics": {"discrimination": {"auroc": 0.78}},
        },
    ]
    df = build_results_table(results)
    # One row per (cohort, outcome); one auroc column per variant.
    assert len(df) == 1
    assert "opera__auroc" in df.columns
    assert "dapt__auroc" in df.columns
    assert df.iloc[0]["opera__auroc"] == 0.82
    assert df.iloc[0]["dapt__auroc"] == 0.78


def test_build_results_table_multiple_cohorts():
    results = [
        {
            "cohort": "dlbcl",
            "outcome": "mortality_1y",
            "variant": "opera",
            "metrics": {"discrimination": {"auroc": 0.82}},
        },
        {
            "cohort": "cll",
            "outcome": "mortality_1y",
            "variant": "opera",
            "metrics": {"discrimination": {"auroc": 0.74}},
        },
    ]
    df = build_results_table(results)
    assert len(df) == 2
    assert set(df["cohort"]) == {"dlbcl", "cll"}
    by_cohort = df.set_index("cohort")["opera__auroc"].to_dict()
    assert by_cohort["dlbcl"] == 0.82
    assert by_cohort["cll"] == 0.74


# ── to_latex_table ──────────────────────────────────────────────────────────


def _latex_table_df():
    results = [
        {
            "cohort": "dlbcl",
            "outcome": "mortality_1y",
            "variant": "opera",
            "metrics": {"discrimination": {"auroc": 0.85}},
        },
        {
            "cohort": "dlbcl",
            "outcome": "mortality_1y",
            "variant": "dapt",
            "metrics": {"discrimination": {"auroc": 0.79}},
        },
    ]
    return build_results_table(results)


def test_to_latex_table_contains_toprule():
    latex = to_latex_table(_latex_table_df(), metric="auroc")
    assert "\\toprule" in latex
    assert "\\bottomrule" in latex
    assert "\\begin{tabular}" in latex


def test_to_latex_table_best_value_bolded():
    latex = to_latex_table(_latex_table_df(), metric="auroc")
    # opera (0.85) is the best of the two variants and must be bolded.
    assert "\\textbf{0.850}" in latex
    # dapt (0.79) is not the best and is rendered plainly.
    assert "0.790" in latex
    assert "\\textbf{0.790}" not in latex


def test_to_latex_table_missing_value_shown_as_dash():
    results = [
        {
            "cohort": "dlbcl",
            "outcome": "mortality_1y",
            "variant": "opera",
            "metrics": {"discrimination": {"auroc": 0.85}},
        },
        {
            "cohort": "cll",
            "outcome": "mortality_1y",
            "variant": "dapt",
            "metrics": {"discrimination": {"auroc": 0.70}},
        },
    ]
    df = build_results_table(results)
    latex = to_latex_table(df, metric="auroc")
    # dlbcl has no dapt cell and cll has no opera cell → NaN cells render as "--".
    assert "--" in latex


# ── rank_normalize_scores ───────────────────────────────────────────────────


def test_rank_normalize_scores_numeric_range_0_1():
    scores = rank_normalize_scores(pd.Series([1.0, 2.0, 3.0, 4.0, 5.0]))
    assert scores.min() == 0.0
    assert scores.max() == 1.0
    assert np.all((scores >= 0.0) & (scores <= 1.0))
    # Monotone increasing input → monotone increasing rank-normalized output.
    assert np.all(np.diff(scores) > 0)


def test_rank_normalize_scores_constant_returns_half():
    scores = rank_normalize_scores(pd.Series([3.0, 3.0, 3.0]))
    assert np.allclose(scores, 0.5)


def test_rank_normalize_scores_two_values():
    scores = rank_normalize_scores(pd.Series([10.0, 20.0]))
    assert np.allclose(sorted(scores), [0.0, 1.0])
