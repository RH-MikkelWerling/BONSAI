"""Contract tests for generic prediction-origin embedding extraction."""

import pandas as pd
import pytest

from opera.run.extract_outcome_transfer_embeddings import (
    OutcomeTransferExtractionError,
)
from opera.run.extract_patient_embeddings import prepare_prediction_origins


def _index_frame():
    return pd.DataFrame(
        {
            "patientid": [1, 2, 3],
            "first_line_date": ["2021-01-01", "2022-06-01", "2024-01-01"],
            "partition": ["development", "temporal_validation", "test"],
        }
    )


def test_prepare_prediction_origins_supports_explicit_column_and_split_names():
    reference, keys = prepare_prediction_origins(
        _index_frame(),
        subject_col="patientid",
        index_date_col="first_line_date",
        split_col="partition",
        split_keys={
            "train": "development",
            "tuning": "temporal_validation",
            "held_out": "test",
        },
    )

    assert list(reference.columns) == [
        "subject_id",
        "split",
        "index_date",
        "censor_abspos",
    ]
    assert reference["subject_id"].tolist() == [1, 2, 3]
    assert reference["censor_abspos"].notna().all()
    assert keys["held_out"] == "test"


def test_prepare_prediction_origins_rejects_duplicate_patients():
    frame = pd.concat([_index_frame(), _index_frame().iloc[[0]]], ignore_index=True)

    with pytest.raises(OutcomeTransferExtractionError, match="one row per patient"):
        prepare_prediction_origins(
            frame,
            subject_col="patientid",
            index_date_col="first_line_date",
            split_col="partition",
            split_keys={
                "train": "development",
                "tuning": "temporal_validation",
                "held_out": "test",
            },
        )


def test_prepare_prediction_origins_requires_all_prospective_splits():
    frame = _index_frame().iloc[:2]

    with pytest.raises(OutcomeTransferExtractionError, match="absent"):
        prepare_prediction_origins(
            frame,
            subject_col="patientid",
            index_date_col="first_line_date",
            split_col="partition",
            split_keys={
                "train": "development",
                "tuning": "temporal_validation",
                "held_out": "test",
            },
        )


def test_prepare_prediction_origins_derives_prospective_splits_from_index_date():
    frame = _index_frame().drop(columns="partition")

    reference, keys = prepare_prediction_origins(
        frame,
        subject_col="patientid",
        index_date_col="first_line_date",
        split_col="partition",
        split_keys={
            "train": "development",
            "tuning": "temporal_validation",
            "held_out": "test",
        },
        tuning_start_date="2022-01-01",
        held_out_start_date="2023-01-01",
    )

    assert reference["split"].tolist() == [
        "development",
        "temporal_validation",
        "test",
    ]
    assert keys["held_out"] == "test"
