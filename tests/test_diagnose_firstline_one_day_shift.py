import pandas as pd

from opera.run.diagnose_firstline_one_day_shift import diagnose


def test_identifies_record_level_minus_one_day_shift(tmp_path):
    old_mapped, new_mapped = (tmp_path / name for name in ("old.csv", "new.csv"))
    old_tx, new_tx = (tmp_path / name for name in ("old.parquet", "new.parquet"))
    pd.DataFrame({"patientid": [1], "firstline_date": ["2020-01-02"]}).to_csv(
        old_mapped, index=False)
    pd.DataFrame({"patientid": [1], "firstline_date": ["2020-01-01"]}).to_csv(
        new_mapped, index=False)
    pd.DataFrame({"patientid": [1], "source": ["RKKP"], "tx_code": ["A"],
                  "date_start": ["2020-01-02"]}).to_parquet(old_tx)
    pd.DataFrame({"patientid": [1], "source": ["RKKP"], "tx_code": ["A"],
                  "date_start": ["2020-01-01"]}).to_parquet(new_tx)
    result = diagnose(old_mapped, new_mapped, old_tx, new_tx)
    assert result["n_equal_multiplicity_matched_rows"] == 1
    assert result["source_date_shifts_INTERNAL"][0]["n_minus_one_day"] == 1
