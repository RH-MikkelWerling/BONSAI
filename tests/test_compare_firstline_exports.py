import pandas as pd

from opera.run.compare_firstline_exports import compare


def test_reports_only_counts_of_changed_first_lines():
    before = pd.DataFrame({"patientid": ["1", "2", "3"],
                           "firstline_date": ["2020-01-01", "2021-05-01", "2019-03-03"],
                           "cohort_fine": ["DLBCL", "FL", "CLL"],
                           "cohort_grouped": ["DLBCL_like", "Indolent_B_NHL", "CLL_SLL"],
                           "arm_primary": ["R-CHOP", "R", "FCR"]})
    after = before.copy()
    after.loc[1, "firstline_date"] = "2021-04-01"
    after = pd.concat([after.iloc[:2], pd.DataFrame({"patientid": ["4"], "firstline_date": ["2022-01-01"],
                                                      "cohort_fine": ["MM"], "cohort_grouped": ["MM"],
                                                      "arm_primary": ["VRd"]})])
    report = compare(before, after)
    assert report["only_before"] == 1 and report["only_after"] == 1 and report["in_both"] == 2
    assert report["changed_firstline_date"] == 1 and report["changed_cohort_fine"] == 0
    assert report["identical"] is False
    assert compare(before, before)["identical"] is True
