import numpy as np
import pandas as pd

from opera.evaluation.patient_transfer import (
    build_patient_transfer_table,
    compute_embedding_neighbor_support,
    compute_patient_prediction_gain,
    summarize_beneficiary_profile,
)


def test_compute_patient_prediction_gain_positive_when_comparator_improves():
    baseline = pd.DataFrame(
        {
            "subject_id": [1, 2, 3],
            "label": [1, 0, 1],
            "probability": [0.4, 0.6, 0.7],
        }
    )
    comparator = pd.DataFrame(
        {
            "subject_id": [1, 2, 3],
            "probability": [0.8, 0.2, 0.9],
        }
    )

    result = compute_patient_prediction_gain(baseline, comparator)

    assert len(result) == 3
    assert result["brier_gain"].mean() > 0
    assert result["logloss_gain"].mean() > 0


def test_embedding_neighbor_support_handles_missing_optional_features():
    embeddings = pd.DataFrame(
        {
            "subject_id": [1, 2, 3, 4],
            "e0": [1.0, 0.9, -1.0, -0.9],
            "e1": [0.0, 0.1, 0.0, -0.1],
        }
    )
    metadata = pd.DataFrame(
        {
            "subject_id": [1, 2, 3, 4],
            "cohort": ["dlbcl", "cll", "dlbcl", "cll"],
            "label": [1, 1, 0, 0],
            "age": [70, 72, 55, 57],
        }
    )

    result = compute_embedding_neighbor_support(
        embeddings,
        metadata,
        feature_cols=["age", "missing_rkkp_col"],
        k=1,
    )

    assert len(result) == 4
    assert "cross_cohort_neighbor_fraction" in result
    assert "neighbor_mean_age" in result
    assert "neighbor_mean_missing_rkkp_col" not in result
    assert np.isfinite(result["mean_neighbor_similarity"]).all()


def test_build_patient_transfer_table_and_profile_summary():
    baseline = pd.DataFrame(
        {"subject_id": [1, 2], "label": [1, 0], "probability": [0.2, 0.8]}
    )
    comparator = pd.DataFrame({"subject_id": [1, 2], "probability": [0.9, 0.1]})
    embeddings = pd.DataFrame(
        {"subject_id": [1, 2], "e0": [1.0, 0.0], "e1": [0.0, 1.0]}
    )
    metadata = pd.DataFrame(
        {"subject_id": [1, 2], "cohort": ["a", "b"], "label": [1, 0]}
    )

    table = build_patient_transfer_table(baseline, comparator, embeddings, metadata, k=1)
    summary = summarize_beneficiary_profile(table, top_fraction=0.5)

    assert len(table) == 2
    assert table["brier_gain"].mean() > 0
    assert not summary.empty
