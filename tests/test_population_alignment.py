import pandas as pd
import pytest

from opera.run.diagnose_population_alignment import (
    audit_alignment,
    inspect_missing_patient_audit,
)


def test_alignment_reports_both_directions_and_conflicts_without_changing_scope():
    population = pd.DataFrame(
        {
            "subject_id": [1, 2, 3],
            "index_date": ["2020-01-01"] * 3,
            "cohort_fine": ["A", "A", "B"],
        }
    )
    survival = pd.DataFrame(
        {
            "subject_id": [1, 2, 4],
            "index_date": ["2020-01-01", "2020-02-01", None],
            "split": ["train", None, "held_out"],
        }
    )
    report, missing, extra = audit_alignment(population, survival)
    assert report["shared_patients"] == 2
    assert report["population_only"] == report["survival_only"] == 1
    assert report["shared_index_date_conflicts"] == 1
    assert report["shared_invalid_or_missing_splits"] == 1
    assert missing.subject_id.tolist() == [3]
    assert extra.subject_id.tolist() == [4]
    assert len(population) == 3


def test_duplicate_ids_fail_before_comparing_sets():
    frame = pd.DataFrame({"subject_id": [1, 1]})
    with pytest.raises(ValueError, match="duplicate"):
        audit_alignment(frame, frame)


def test_audit_evidence_is_limited_to_missing_patients_and_reports_dates():
    missing = pd.DataFrame({"subject_id": [2, 3]})
    audit = pd.DataFrame(
        {
            "subject_id": [1, 2, 3],
            "eligible": [True, False, False],
            "eligibility_reason": ["ok", "no_followup", None],
            "index_date": ["2020-01-01"] * 3,
            "censor_date": ["2021-01-01", "2019-01-01", None],
        }
    )
    report, selected = inspect_missing_patient_audit(missing, audit)
    assert selected.subject_id.tolist() == [2, 3]
    assert report["matched_patients"] == 2
    assert report["field_counts"]["eligibility_reason"] == {
        "no_followup": 1,
        "<missing>": 1,
    }
    assert report["date_checks"]["censor_date"]["missing_or_invalid"] == 1
    assert report["date_checks"]["censor_date_relative_to_index"]["before"] == 1
