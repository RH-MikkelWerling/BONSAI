from datetime import datetime

import polars as pl
import pytest

from bonsai.functional.create_data import create_combined_binning_value_tokens


def test_combined_binning_expands_ehr2meds_numeric_columns():
    time = datetime(2025, 1, 1)
    frame = pl.DataFrame(
        {
            "subject_id": [1, 1],
            "time": [time, time],
            "code": ["LAB/A", "DIAG/B"],
            "numeric_value_normalized": [0.75, None],
            "numeric_value_bin": [3, None],
            "numeric_value_binned": [0.7, None],
            "numeric_value_present": [True, False],
        }
    )

    result = create_combined_binning_value_tokens(frame).sort("row_idx")

    assert result["code"].to_list() == ["LAB/A", "[VAL]", "DIAG/B"]
    assert result["value_present"].to_list() == [False, True, False]
    assert result["value_bin"].to_list() == [None, 3, None]
    assert result["value_normalized"].to_list() == [None, 0.7, None]


def test_combined_binning_is_noop_without_numeric_columns():
    frame = pl.DataFrame(
        {"subject_id": [1], "time": [datetime(2025, 1, 1)], "code": ["A"]}
    )
    assert create_combined_binning_value_tokens(frame).equals(frame)


def test_combined_binning_rejects_incomplete_present_values():
    frame = pl.DataFrame(
        {
            "subject_id": [1],
            "time": [datetime(2025, 1, 1)],
            "code": ["LAB/A"],
            "numeric_value_bin": [2],
            "numeric_value_present": [True],
        }
    )
    with pytest.raises(ValueError, match="normalized bin representative"):
        create_combined_binning_value_tokens(frame)
