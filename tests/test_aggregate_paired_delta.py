"""
Task 6 — Patient-paired delta in the default aggregate output.

Tests that ``run_pairwise_comparisons`` produces per-cell deltas with
bootstrap/DeLong confidence intervals and BH-corrected p-values when given a
small synthetic predictions directory, and that ``aggregate_results.main``
writes the paired delta CSV when ``--predictions_dir`` is supplied.
"""

import sys
from pathlib import Path

import numpy as np


def _write_predictions(directory: Path, labels, probs, subject_ids) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    np.savez(
        directory / "predictions.npz",
        labels=labels,
        probabilities=probs,
        subject_ids=subject_ids,
    )


def _make_predictions_tree(tmp_path: Path):
    """Two cohorts × one outcome × two models with 120 aligned subjects each."""
    rng = np.random.default_rng(0)
    n = 120
    labels = (rng.uniform(size=n) > 0.7).astype(int)
    subject_ids = np.arange(n)

    for cohort in ("cohort_a", "cohort_b"):
        for outcome in ("mortality",):
            base_probs = np.clip(labels * 0.45 + rng.uniform(0, 0.3, n), 0.0, 1.0)
            opera_probs = np.clip(base_probs + 0.10 * labels, 0.0, 1.0)
            _write_predictions(
                tmp_path / cohort / outcome / "opera",
                labels,
                opera_probs,
                subject_ids,
            )
            _write_predictions(
                tmp_path / cohort / outcome / "tabular",
                labels,
                base_probs,
                subject_ids,
            )
    return tmp_path


def test_paired_delta_has_ci_and_fdr_corrected_pvalues(tmp_path):
    """Per-cell output carries CI bounds and BH-adjusted p-values."""
    from opera.evaluation.significance import run_pairwise_comparisons

    preds_dir = _make_predictions_tree(tmp_path / "preds")
    result = run_pairwise_comparisons(
        str(preds_dir),
        contrasts=[("opera", "tabular", "opera_vs_tabular")],
        metrics=["auroc"],
        n_bootstrap=500,
    )

    assert not result.empty, "Expected comparison rows"
    # Two cohort/outcome cells
    assert len(result) == 2

    # CI bounds present and ordered
    assert "delta_lower" in result.columns
    assert "delta_upper" in result.columns
    assert (result["delta_lower"] <= result["delta_mean"] + 1e-9).all()
    assert (result["delta_mean"] - 1e-9 <= result["delta_upper"]).all()

    # FDR-corrected p-values present
    assert "p_adjusted" in result.columns
    assert result["p_adjusted"].notna().all()

    # BH correction never makes p smaller
    assert (result["p_adjusted"] >= result["p_value"] - 1e-9).all()

    # Subject-ID alignment was used (n_patients should match fixture size)
    assert "n_patients" in result.columns
    assert (result["n_patients"] == 120).all()


def test_aggregate_results_writes_paired_delta_csv(tmp_path, monkeypatch):
    """aggregate_results.main writes paired_delta_*.csv when --predictions_dir given."""
    preds_dir = _make_predictions_tree(tmp_path / "preds")
    output_dir = tmp_path / "out"
    output_dir.mkdir()

    # Minimal result.jsonl so aggregate_results doesn't abort early
    import json

    (tmp_path / "results").mkdir()
    rows = [
        {
            "cohort": "cohort_a",
            "outcome": "mortality",
            "outcome_window_hours": 8760,
            "split": "held_out",
            "seed": 42,
            "training_fraction": 1.0,
            "model_family": "opera",
            "auroc": 0.75,
            "rarity_mode": "real",
            "evaluation_subset": "full",
        },
        {
            "cohort": "cohort_a",
            "outcome": "mortality",
            "outcome_window_hours": 8760,
            "split": "held_out",
            "seed": 42,
            "training_fraction": 1.0,
            "model_family": "tabular",
            "auroc": 0.65,
            "rarity_mode": "real",
            "evaluation_subset": "full",
        },
    ]
    with open(tmp_path / "results" / "result.jsonl", "w") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "aggregate_results",
            "--results_dir",
            str(tmp_path / "results"),
            "--output_dir",
            str(output_dir),
            "--baseline",
            "tabular",
            "--comparator",
            "opera",
            "--predictions_dir",
            str(preds_dir),
            "--paired_n_bootstrap",
            "200",
            "--diagnostic_plots",
        ],
    )

    from opera.run.aggregate_results import main

    main()

    paired_csv = output_dir / "paired_delta_opera_minus_tabular.csv"
    assert paired_csv.exists(), f"Expected {paired_csv}"
    import pandas as pd

    df = pd.read_csv(paired_csv)
    assert "delta_mean" in df.columns
    assert "delta_lower" in df.columns
    assert "p_adjusted" in df.columns
    assert (output_dir / "seed_stability_auroc.png").exists()
