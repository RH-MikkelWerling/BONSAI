import json

import numpy as np
import pytest
import pandas as pd

from opera.functional.tabular_features import prepare_feature_matrix


def test_sequence_matched_profile_prepares_and_reuses_pickle(tmp_path):
    source = tmp_path / "feature_matrix_all.pkl"
    population = tmp_path / "population_metadata.csv"
    output_dir = tmp_path / "tabular"
    pd.DataFrame(
        {
            "patientid": [1, 2, 3],
            "timestamp": ["a", "b", "c"],
            "prediction_time_uuid": ["x", "y", "z"],
            "numeric__pred_SDS_lab": [1.0, 2.0, 3.0],
            "numeric__pred_RKKP_stage": [1.0, 2.0, 3.0],
            "numeric__pred_adverse_events_grade": [0.0, 1.0, 0.0],
            "empty": [np.nan, np.nan, np.nan],
            "constant": [7, 7, 7],
        }
    ).to_pickle(source)
    pd.DataFrame(
        {
            "subject_id": [1, 2, 3],
            "birth_date": ["1980-01-01", "1990-01-01", "2000-01-01"],
            "index_date": ["2020-01-01", "2020-01-01", "2020-01-01"],
            "sex": ["F", "M", "F"],
        }
    ).to_csv(population, index=False)

    target, manifest = prepare_feature_matrix(
        source,
        profile="sequence_matched",
        population_path=population,
        output_dir=output_dir,
    )
    result = pd.read_parquet(target)
    assert set(result.columns) == {
        "subject_id",
        "numeric__pred_SDS_lab",
        "age_at_index",
        "matched_sex",
    }
    assert manifest["removed_all_missing_columns"] == ["empty"]
    assert manifest["removed_constant_columns"] == ["constant"]
    assert manifest["age_source"] == "index_date-birth_date"
    assert target.name == "feature_matrix_all__sequence_matched.parquet"

    second_target, second_manifest = prepare_feature_matrix(
        source,
        profile="sequence_matched",
        population_path=population,
        output_dir=output_dir,
    )
    assert second_target == target
    assert second_manifest == manifest
    with open(target.with_suffix(target.suffix + ".manifest.json")) as handle:
        assert json.load(handle) == manifest


def test_sequence_matched_profile_uses_existing_age_and_sex_keywords(tmp_path):
    source = tmp_path / "cohort.pkl"
    population = tmp_path / "population.parquet"
    pd.DataFrame({"patientid": ["1", "2"], "lab": [1, 2]}).to_pickle(source)
    pd.DataFrame(
        {
            "subject_id": ["1", "2"],
            "age_years": [55, 66],
            "biological_sex": ["F", "M"],
        }
    ).to_parquet(population, index=False)
    target, manifest = prepare_feature_matrix(
        source,
        profile="sequence_matched",
        population_path=population,
    )
    result = pd.read_parquet(target)
    assert result["age_at_index"].tolist() == [55, 66]
    assert result["matched_sex"].tolist() == ["F", "M"]
    assert manifest["age_source"] == "age_years"
    assert manifest["sex_source"] == "biological_sex"


def test_sealed_membership_uses_firstline_date_and_joined_demographics(tmp_path):
    """The sealed membership file has firstline_date and no age or sex; those
    come from the Table 1 demographics export."""
    source = tmp_path / "DLBCL.pkl"
    pd.DataFrame({"patientid": [1, 2], "timestamp": ["a", "b"],
                  "numeric__pred_lab": [1.0, 2.0]}).to_pickle(source)
    membership = tmp_path / "population_metadata.parquet"
    pd.DataFrame({"subject_id": [1, 2, 3], "cohort_fine": ["DLBCL"] * 3,
                  "firstline_date": ["2020-01-01", "2021-06-01", None]}
                 ).to_parquet(membership)
    demographics = tmp_path / "table1_demographics.parquet"
    pd.DataFrame({"subject_id": [1, 2, 3], "sex": ["F", "M", "F"],
                  "birth_date": pd.to_datetime(["1950-01-01", "1960-06-01", "1970-01-01"])}
                 ).to_parquet(demographics)

    target, manifest = prepare_feature_matrix(
        source, profile="sequence_matched", population_path=membership,
        output_dir=tmp_path / "out", demographics_path=demographics)
    result = pd.read_parquet(target).set_index("subject_id")
    assert manifest["age_source"] == "index_date-birth_date"
    assert result.loc[1, "age_at_index"] == pytest.approx(70.0, abs=0.01)
    assert result.loc[2, "matched_sex"] == "M"
    assert "demographics" in manifest
