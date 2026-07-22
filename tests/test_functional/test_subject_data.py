import unittest
from unittest.mock import patch, MagicMock
from bonsai.functional.subject_data import filter_subject_data, prepare_subject_data
from pathlib import Path
import polars as pl


class TestSubjectData(unittest.TestCase):
    def test_filter_subject_data(self):
        subject_data = [
            {"subject_id": 1, "code": [1]},
            {"subject_id": 2, "code": [2]},
            {"subject_id": 3, "code": [3]},
        ]
        cohort = [1, 3]
        filtered = filter_subject_data(subject_data, cohort)
        self.assertEqual(len(filtered), 2)
        self.assertEqual(filtered[0]["subject_id"], 1)
        self.assertEqual(filtered[1]["subject_id"], 3)

    @patch("bonsai.functional.subject_data.pl.read_parquet")
    def test_prepare_subject_data(self, mock_read_parquet):
        # Mock the dataframe returned by read_parquet
        df = pl.from_dict(
            {
                "subject_id": [1, 1, 2, 1],
                "code": [11, 10, 20, 12],
                "abspos": [1, 0, 0, 1],
                "row_idx": [3, 1, 1, 2],
                "segment": [0, 0, 1, 0],
                "age": [31, 30, 40, 31],
            }
        )
        mock_read_parquet.return_value = df

        # Mock Path.glob to return a list with a single 'file'
        mock_path = MagicMock(spec=Path)
        mock_path.glob.return_value = ["fake.parquet"]

        result = prepare_subject_data(mock_path)
        self.assertEqual(len(result), 2)
        result_by_subject = {s["subject_id"]: s for s in result}

        # ehr2meds owns event ordering; BONSAI preserves the input order.
        self.assertEqual(result_by_subject[1]["code"].tolist(), [11, 10, 12])
        self.assertEqual(len(result_by_subject[2]["code"]), 1)
