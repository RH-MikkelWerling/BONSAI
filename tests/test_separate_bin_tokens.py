from datetime import datetime

import polars as pl
import pytest

from bonsai.functional.create_data import create_separate_bin_tokens
from bonsai.functional.features import create_features
from bonsai.functional.truncation import infer_background_length, truncate_subject
from bonsai.modules.tokenizer.tokenizer import EHRTokenizer


def _frame() -> pl.DataFrame:
    time = datetime(2025, 1, 1)
    return pl.DataFrame(
        {
            "subject_id": [1, 1, 1, 1],
            "time": [datetime(1980, 1, 1), time, time, time],
            "code": ["DOB", "LAB/NPU_A", "LAB/NPU_B", "DX/C"],
            "numeric_value_bin": [None, 7, 7, None],
            "numeric_value_binned": [None, 0.75, 0.75, None],
            "numeric_value_present": [False, True, True, False],
        }
    )


def test_separate_bins_are_shared_ordinary_tokens_without_numeric_payloads():
    result = create_separate_bin_tokens(_frame())

    assert result["code"].to_list() == [
        "DOB",
        "LAB/NPU_A",
        "BIN_7",
        "LAB/NPU_B",
        "BIN_7",
        "DX/C",
    ]
    assert not any("value" in column for column in result.columns)
    assert result["row_idx"].to_list() == [0, 2, 3, 4, 5, 6]


def test_separator_is_not_inserted_between_lab_and_bin():
    features = create_features(create_separate_bin_tokens(_frame()))
    tokenizer = EHRTokenizer(sep_tokens=True)
    tokenized = tokenizer(features)
    inverse = {token_id: token for token, token_id in tokenizer.vocabulary.items()}
    codes = [inverse[token_id] for token_id in tokenized["code"]]

    assert codes[codes.index("LAB/NPU_A") + 1] == "BIN_7"
    assert codes[codes.index("LAB/NPU_B") + 1] == "BIN_7"


def test_separate_bin_token_rejects_invalid_present_bin():
    frame = pl.DataFrame(
        {
            "subject_id": [1],
            "time": [datetime(2025, 1, 1)],
            "code": ["LAB/A"],
            "numeric_value_bin": [None],
            "numeric_value_present": [True],
        }
    )

    with pytest.raises(ValueError, match="non-negative integer bin"):
        create_separate_bin_tokens(frame)


def test_segment_aware_truncation_cannot_orphan_bin_token():
    frame = _frame().with_columns(
        time=pl.Series(
            [
                datetime(1980, 1, 1),
                datetime(2025, 1, 1),
                datetime(2025, 1, 2),
                datetime(2025, 1, 3),
            ]
        )
    )
    tokenizer = EHRTokenizer(sep_tokens=False)
    tokenized = tokenizer(create_features(create_separate_bin_tokens(frame)))
    subject = {
        column: tokenized[column].to_torch()
        for column in ("code", "age", "abspos", "segment")
    }
    truncated = truncate_subject(
        subject,
        max_len=3,
        background_length=infer_background_length(subject),
        strategy="tail",
    )
    inverse = {token_id: token for token, token_id in tokenizer.vocabulary.items()}
    codes = [inverse[token_id.item()] for token_id in truncated["code"]]

    # The two-token lab/bin groups are indivisible; the one-token tail event is
    # retained without an orphaned value token.
    assert codes == ["DOB", "DX/C"]
