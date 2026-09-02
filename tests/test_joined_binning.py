import polars as pl

from bonsai.functional.create_data import prepare_joined_binning_tokens


def test_joined_binning_preserves_codes_and_drops_numeric_payloads():
    frame = pl.DataFrame(
        {
            "subject_id": [1, 1],
            "code": ["LAB/NPU//bin_3", "DX/ABC"],
            "numeric_value": [4.2, None],
            "numeric_value_normalized": [0.6, None],
            "numeric_value_bin": [3, None],
            "numeric_value_binned": [0.625, None],
            "numeric_value_present": [True, False],
            "row_idx": [0, 1],
        }
    )

    result = prepare_joined_binning_tokens(frame)

    assert result["code"].to_list() == ["LAB/NPU//bin_3", "DX/ABC"]
    assert result["row_idx"].to_list() == [0, 1]
    assert not any("value" in column for column in result.columns)


def test_joined_binning_is_noop_without_numeric_payloads():
    frame = pl.DataFrame({"subject_id": [1], "code": ["LAB/NPU//bin_3"]})

    assert prepare_joined_binning_tokens(frame).equals(frame)
