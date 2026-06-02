import numpy as np
import pandas as pd
import pytest

from opera.evaluation.subgroups import (
    build_subgroup_delta_table,
    compute_subgroup_metrics,
)


def test_compute_subgroup_metrics_keeps_sparse_groups_visible():
    subgroup_df = pd.DataFrame(
        {
            "subject_id": [1, 2, 3, 4],
            "sex": ["F", "F", "M", "M"],
        }
    )

    result = compute_subgroup_metrics(
        subject_ids=np.array([1, 2, 3, 4]),
        labels=np.array([0, 1, 0, 0]),
        probabilities=np.array([0.2, 0.8, 0.1, 0.3]),
        subgroup_df=subgroup_df,
        columns=["sex"],
    )

    female = result[result["subgroup_value"] == "F"].iloc[0]
    male = result[result["subgroup_value"] == "M"].iloc[0]

    assert female["n_total"] == 2
    assert female["auroc"] == 1.0
    assert "calibration_intercept" in result.columns
    assert "calibration_slope" in result.columns
    assert male["n_total"] == 2
    assert "auroc" in result.columns
    assert pd.isna(male["auroc"])


def test_build_subgroup_delta_table_compares_opera_to_tabular():
    rows = pd.DataFrame(
        [
            {
                "cohort": "dlbcl",
                "outcome": "mortality_1y",
                "split": "held_out",
                "seed": 42,
                "evaluation_subset": "full",
                "subgroup_column": "sex",
                "subgroup_value": "F",
                "model_family": "tabular_ehr",
                "n_total": 10,
                "n_positive": 4,
                "prevalence": 0.4,
                "auroc": 0.70,
            },
            {
                "cohort": "dlbcl",
                "outcome": "mortality_1y",
                "split": "held_out",
                "seed": 42,
                "evaluation_subset": "full",
                "subgroup_column": "sex",
                "subgroup_value": "F",
                "model_family": "opera",
                "n_total": 10,
                "n_positive": 4,
                "prevalence": 0.4,
                "auroc": 0.78,
            },
        ]
    )

    result = build_subgroup_delta_table(rows, metrics=["auroc"])

    assert len(result) == 1
    assert result["delta_auroc_vs_tabular_ehr"].iloc[0] == pytest.approx(0.08)
