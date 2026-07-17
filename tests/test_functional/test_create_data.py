import polars as pl

from bonsai.functional.create_data import drop_duplicates


def test_drop_duplicates_keeps_same_time_rows_with_distinct_numeric_payloads():
    df = pl.DataFrame(
        {
            "subject_id": [1, 1, 1],
            "code": ["LAB", "LAB", "LAB"],
            "time": [10, 10, 10],
            "row_idx": [5, 6, 6],
            "value_bin": [2, 3, 3],
            "value_normalized": [0.2, 0.3, 0.3],
        }
    )

    deduped = drop_duplicates(df)

    assert deduped["row_idx"].to_list() == [5, 6]
