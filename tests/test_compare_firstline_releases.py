import pandas as pd

from opera.run.compare_firstline_releases import compare


def test_comparison_reports_overlap_and_changed_dates(tmp_path):
    old = tmp_path / "old.csv"
    new = tmp_path / "new.csv"
    pd.DataFrame({"patientid": [1, 2], "firstline_date": ["2020-01-01", "2020-01-01"],
                  "cohort_grouped": ["A", "B"], "cohort_fine": ["A1", "B1"],
                  "arm_primary": ["X", "Y"]}).to_csv(old, index=False)
    pd.DataFrame({"patientid": [1, 3], "firstline_date": ["2020-03-01", "2020-01-01"],
                  "cohort_grouped": ["A", "C"], "cohort_fine": ["A1", "C1"],
                  "arm_primary": ["Z", "Y"]}).to_csv(new, index=False)
    result = compare(old, new)
    assert (result["shared_n"], result["old_only_n"], result["new_only_n"]) == (1, 1, 1)
    assert result["fields"]["firstline_date"]["changed_by_over_30d_n"] == 1
    assert result["fields"]["firstline_date"]["date_shift_summary"]["median_absolute_days"] == 60
    assert result["fields"]["arm_primary"]["changed_shared_n"] == 1
    assert result["old_only_profile_INTERNAL"]["cohort_fine"] == {"B1": 1}
    assert result["fields"]["firstline_date"]["changed_by_old_cohort_fine_INTERNAL"] == {"A1": 1}
