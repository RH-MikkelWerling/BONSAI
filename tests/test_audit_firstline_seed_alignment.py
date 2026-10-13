import pandas as pd

from opera.run.audit_firstline_seed_alignment import audit


def test_row_level_mm_v5_uses_earliest_line_one_date(tmp_path):
    from opera.run.audit_firstline_seed_alignment import _seeds

    mm = tmp_path / "mm_v5.parquet"
    pd.DataFrame({"patientid": [1, 1, 1], "line": [1, 1, 2],
                  "date_start": ["2020-01-05", "2020-01-02", "2020-02-01"]}).to_parquet(mm)
    seeds = _seeds(mm, "MM")
    assert len(seeds) == 1
    assert str(seeds.iloc[0]["seed_date"].date()) == "2020-01-02"


def test_seed_crosses_nonmatching_cohort(tmp_path):
    mapped, cll, mm, old = (tmp_path / name for name in
                            ("mapped.csv", "cll.parquet", "mm.parquet", "old.csv"))
    pd.DataFrame({"patientid": [1, 2, 3], "firstline_date": ["2020-01-10"] * 3,
                  "cohort_fine": ["CLL", "MCL", "MCL"],
                  "seed_date": ["2020-01-10", "2020-01-10", None],
                  "firstline_source": ["clinical_seed", "clinical_seed", "inferred"]}).to_csv(mapped, index=False)
    pd.DataFrame({"patientid": [1, 2], "line": [1, 1],
                  "date_start_line": ["2020-01-10"] * 2}).to_parquet(cll, index=False)
    pd.DataFrame({"patientid": [], "line": [], "date_start_line": []}).to_parquet(mm, index=False)
    pd.DataFrame({"patientid": [1, 2, 3],
                  "firstline_date": ["2020-01-10", "2020-02-10", "2020-01-10"],
                  "seed_date": ["2020-01-10", "2019-12-01", None],
                  "cohort_fine": ["CLL", "MCL", "MCL"],
                  "firstline_source": ["clinical_seed", "inferred", "inferred"]}).to_csv(old, index=False)
    result = audit(mapped, cll, mm, old)
    assert result["rows"][0]["n_seed_cohort_mismatch"] == 1
    assert result["rows"][0]["n_seed_cohort_mismatch_and_clinical_source"] == 1
    assert result["old_new_change_overlap_INTERNAL"][0]["n_changed_date_shared"] == 1
    assert result["old_new_change_overlap_INTERNAL"][0]["n_changed_date_seed_cohort_mismatch_shared"] == 1
    assert result["old_new_change_overlap_INTERNAL"][0]["n_seed_date_changed_shared"] == 1
    assert result["old_new_change_overlap_INTERNAL"][0]["source_transitions_INTERNAL"] == [
        {"old_source": "inferred", "new_source": "clinical_seed", "n": 1}]
    assert result["old_new_change_overlap_INTERNAL"][2]["n_seed_date_changed_shared"] == 0
