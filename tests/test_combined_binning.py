from datetime import datetime

import polars as pl
import pytest

from bonsai.functional.create_data import create_combined_binning_value_tokens
from bonsai.modules.tokenizer.tokenizer import EHRTokenizer


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


def test_same_time_row_order_and_value_adjacency_survive_sep_insertion():
    frame = pl.DataFrame(
        {
            "subject_id": [1, 1, 1],
            "time": [datetime(2025, 1, 2)] * 3,
            "code": ["EARLY", "LAB/A", "LATE"],
            "row_idx": [10, 20, 30],
            "numeric_value_bin": [None, 2, None],
            "numeric_value_binned": [None, 0.4, None],
            "numeric_value_present": [False, True, False],
        }
    )
    expanded = create_combined_binning_value_tokens(frame).with_columns(
        abspos=pl.lit(1.0),
        segment=pl.when(pl.col("code") == "LATE").then(1).otherwise(0),
    )
    tokenizer = EHRTokenizer(sep_tokens=True)

    result = tokenizer(expanded)
    inverse = {token_id: token for token, token_id in tokenizer.vocabulary.items()}

    assert [inverse[token_id] for token_id in result["code"]] == [
        "EARLY",
        "LAB/A",
        "[VAL]",
        "[SEP]",
        "LATE",
    ]


def test_hot_vocabulary_excludes_codes_at_or_after_cutoff():
    tokenizer = EHRTokenizer(sep_tokens=False, vocabulary_cutoff_abspos=2.0)
    features = pl.DataFrame(
        {
            "subject_id": [1, 1, 1],
            "code": ["PAST", "AT_CUTOFF", "FUTURE"],
            "abspos": [1.0, 2.0, 3.0],
            "segment": [0, 1, 2],
        }
    )

    result = tokenizer(features)

    assert "PAST" in tokenizer.vocabulary
    assert "AT_CUTOFF" not in tokenizer.vocabulary
    assert "FUTURE" not in tokenizer.vocabulary
    assert result["code"].to_list()[1:] == [
        tokenizer.vocabulary["[UNK]"],
        tokenizer.vocabulary["[UNK]"],
    ]
