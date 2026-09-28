import pandas as pd
import pytest
from opera.run.plot_censoring_aware_map import align_map_data


def test_alignment_uses_ids_and_filters_split_and_cohort():
    coords = pd.DataFrame({"subject_id": [2, 1], "component_1": [20, 10], "component_2": [0, 0]})
    predictions = pd.DataFrame({"subject_id": [1, 2, 3], "outcome": ["a"] * 3,
                                "cohort": ["DLBCL", "DLBCL", "MM"], "split": ["tuning"] * 3,
                                "time_days": [100] * 3, "event": [0] * 3, "cif_30d": [.2] * 3})
    result = align_map_data(coords, predictions, outcome="a", cohort="DLBCL", horizon=30)
    assert result.component_1.tolist() == [10, 20]
    with pytest.raises(ValueError, match="lack coordinates"):
        align_map_data(coords, predictions, outcome="a", horizon=30)
    with pytest.raises(ValueError, match="Duplicate"):
        align_map_data(pd.concat([coords, coords]), predictions, outcome="a", cohort="DLBCL", horizon=30)


def test_fine_cohorts_can_come_from_metadata_when_predictions_have_null_fine():
    coords = pd.DataFrame({"subject_id": [1, 2], "component_1": [10, 20], "component_2": [0, 0]})
    predictions = pd.DataFrame({"subject_id": [1, 2], "outcome": ["a"] * 2,
                                "cohort_grouped": ["DLBCL_like"] * 2, "cohort_fine": [None] * 2,
                                "split": ["tuning"] * 2, "time_days": [100] * 2,
                                "event": [0] * 2, "cif_30d": [.2] * 2})
    membership = pd.DataFrame({"subject_id": [1, 2], "cohort_fine": ["DLBCL", "RT"]})
    result = align_map_data(coords, predictions, outcome="a", cohort_level="cohort_fine",
                            cohort="DLBCL", membership=membership, horizon=30)
    assert result.subject_id.tolist() == [1]
    assert len(align_map_data(coords, predictions, outcome="a", cohort_level="all", horizon=30)) == 2
    with pytest.raises(ValueError, match="Missing cohort_fine"):
        align_map_data(coords, predictions, outcome="a", cohort_level="cohort_fine", cohort="DLBCL", horizon=30)
    with pytest.raises(ValueError, match="omit"):
        align_map_data(coords, predictions, outcome="a", cohort_level="all", cohort="DLBCL", horizon=30)
