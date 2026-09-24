from __future__ import annotations

import json

import pandas as pd
import pytest

from opera.run.compare_direct_cr_tabular import (
    _tabular_artifacts,
    main as compare_direct_cr_tabular_main,
)


def _write_tabular_artifact(
    root,
    *,
    cohort: str,
    outcome: str = "anemia_g3plus",
    horizon: int = 365,
    model_family: str = "tabular_ehr",
    model_name: str = "xgboost",
) -> None:
    output_dir = root / cohort / model_name
    output_dir.mkdir(parents=True, exist_ok=True)
    prediction_path = output_dir / f"{outcome}_{horizon}d_predictions.csv"
    pd.DataFrame(
        {
            "subject_id": [1, 2, 3, 4, 5, 6],
            "probability": [0.9, 0.2, 0.8, 0.3, 0.1, 0.4],
        }
    ).to_csv(prediction_path, index=False)
    metadata = {
        "model_family": model_family,
        "model_name": model_name,
        "cohort": cohort,
        "outcome_name": outcome,
        "prediction_file": str(prediction_path),
    }
    (output_dir / f"{outcome}_{horizon}d_metadata.json").write_text(
        json.dumps(metadata), encoding="utf-8"
    )


def test_tabular_artifacts_are_keyed_by_cohort_and_found_recursively(tmp_path):
    _write_tabular_artifact(tmp_path, cohort="DLBCL")
    _write_tabular_artifact(tmp_path, cohort="AML")

    artifacts = _tabular_artifacts(
        tmp_path, model_family="tabular_ehr", model_name="xgboost"
    )

    assert set(artifacts) == {
        ("AML", "anemia_g3plus", 365),
        ("DLBCL", "anemia_g3plus", 365),
    }


def test_tabular_artifact_model_filter_prevents_ambiguous_duplicates(tmp_path):
    _write_tabular_artifact(tmp_path, cohort="DLBCL", model_name="xgboost")
    _write_tabular_artifact(
        tmp_path,
        cohort="DLBCL",
        model_name="logistic_regression",
    )

    with pytest.raises(ValueError, match="Duplicate tabular artifact"):
        _tabular_artifacts(tmp_path, model_family="tabular_ehr")

    artifacts = _tabular_artifacts(
        tmp_path, model_family="tabular_ehr", model_name="xgboost"
    )
    assert list(artifacts) == [("DLBCL", "anemia_g3plus", 365)]


def test_comparison_keeps_fine_cohorts_separate(tmp_path, monkeypatch):
    tabular_dir = tmp_path / "tabular"
    _write_tabular_artifact(tabular_dir, cohort="DLBCL")
    _write_tabular_artifact(tabular_dir, cohort="AML")
    direct_path = tmp_path / "direct.parquet"
    rows = []
    for cohort, offset in (("DLBCL", 0.0), ("AML", 0.05)):
        for subject_id, time_days, event, risk in (
            (1, 20.0, 1, 0.85),
            (2, 500.0, 0, 0.15),
            (3, 90.0, 1, 0.75),
            (4, 600.0, 0, 0.25),
            (5, 100.0, 2, 0.20),
            (6, 450.0, 0, 0.35),
        ):
            rows.append(
                {
                    "subject_id": subject_id,
                    "cohort": cohort,
                    "outcome": "anemia_g3plus",
                    "time_days": time_days,
                    "event": event,
                    "cif_365d": min(risk + offset, 0.99),
                }
            )
    pd.DataFrame(rows).to_parquet(direct_path, index=False)
    output_dir = tmp_path / "comparison"
    monkeypatch.setattr(
        "sys.argv",
        [
            "compare_direct_cr_tabular",
            "--direct-predictions",
            str(direct_path),
            "--tabular-dir",
            str(tabular_dir),
            "--output-dir",
            str(output_dir),
            "--n-bootstrap",
            "5",
        ],
    )

    compare_direct_cr_tabular_main()

    results = pd.read_csv(output_dir / "paired_comparison.csv")
    draws = pd.read_parquet(output_dir / "paired_bootstrap_draws.parquet")
    assert set(results["cohort"]) == {"AML", "DLBCL"}
    assert results.groupby("cohort")["n_paired"].first().to_dict() == {
        "AML": 6,
        "DLBCL": 6,
    }
    assert set(draws["cohort"]) == {"AML", "DLBCL"}
    assert {
        "opera_cif_auc_lower",
        "opera_cif_auc_upper",
        "tabular_cif_auc_lower",
        "tabular_cif_auc_upper",
        "delta_cif_auc_lower",
        "delta_cif_auc_upper",
    }.issubset(results.columns)
