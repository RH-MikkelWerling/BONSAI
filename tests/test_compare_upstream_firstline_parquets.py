import pandas as pd

from opera.run.compare_upstream_firstline_parquets import compare


def test_detects_smaller_treatment_table_and_changed_mm_seed(tmp_path):
    old_tx = tmp_path / "old_tx.parquet"
    new_tx = tmp_path / "new_tx.parquet"
    old_mm = tmp_path / "old_mm.parquet"
    new_mm = tmp_path / "new_mm.parquet"
    pd.DataFrame({"patientid": [1, 1, 2], "date_start": ["2020-01-01"] * 3,
                  "source": ["A", "A", "B"], "tx_code": ["X", "Y", "Z"]}).to_parquet(old_tx)
    pd.DataFrame({"patientid": [1, 3], "date_start": ["2020-01-01", "2021-01-01"],
                  "source": ["A", "C"], "tx_code": ["X", "W"]}).to_parquet(new_tx)
    pd.DataFrame({"patientid": [1, 2], "line": [1, 1],
                  "date_start_line": ["2020-01-01", "2020-02-01"],
                  "regimen": ["A", "B"]}).to_parquet(old_mm)
    pd.DataFrame({"patientid": [1, 3, 3], "line": [1, 1, 2],
                  "date_start_line": ["2020-01-10", "2020-03-01", "2020-04-01"],
                  "regimen": ["A", "C", "D"]}).to_parquet(new_mm)
    result = compare(old_tx, new_tx, old_mm, new_mm)
    assert result["treatment"]["old"]["n_rows"] == 3
    assert result["treatment"]["new"]["n_rows"] == 2
    assert result["treatment"]["n_patients_shared"] == 1
    assert result["treatment"]["n_patients_old_only"] == 1
    assert result["treatment"]["n_patients_new_only"] == 1
    assert result["mm_seed"]["n_line_one_dates_changed_shared"] == 1
    assert result["mm_seed"]["n_line_one_patients_new_only"] == 1


def test_treatment_only_when_old_mm_is_unavailable(tmp_path):
    old_tx = tmp_path / "old_tx.parquet"
    new_tx = tmp_path / "new_tx.parquet"
    frame = pd.DataFrame({"patientid": [1], "date_start": ["2020-01-01"],
                          "source": ["A"], "tx_code": ["X"]})
    frame.to_parquet(old_tx)
    frame.to_parquet(new_tx)
    result = compare(old_tx, new_tx)
    assert result["treatment"]["n_patients_shared"] == 1
    assert "mm_seed" not in result


def test_mm_only_skips_treatment_comparison(tmp_path):
    old_mm = tmp_path / "old_mm.parquet"
    new_mm = tmp_path / "new_mm.parquet"
    pd.DataFrame({"patientid": [1], "line": [1],
                  "date_start_line": ["2020-01-01"]}).to_parquet(old_mm)
    pd.DataFrame({"patientid": [1, 1, 1], "line": [1, 1, 2],
                  "date_start": ["2020-01-05", "2020-01-02", "2020-02-01"],
                  "row_acc": [True, False, True]}).to_parquet(new_mm)
    result = compare(old_mm=old_mm, new_mm=new_mm)
    assert "treatment" not in result
    assert result["mm_seed"]["n_line_one_dates_changed_shared"] == 1
    assert result["mm_seed"]["new"]["date_derivation"] == (
        "minimum_date_start_among_line_1_rows_per_patient")
    assert result["mm_seed"]["date_shift_all_line_one_rows"]["median_signed_days"] == 1
    assert result["mm_seed"]["date_shift_row_acc_true_only"]["median_signed_days"] == 4
