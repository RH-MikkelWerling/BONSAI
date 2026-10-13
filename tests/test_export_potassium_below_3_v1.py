import pandas as pd

from opera.run.export_potassium_below_3_v1 import NAME, build


def test_numeric_v6_boundary_and_censoring() -> None:
    old = pd.DataFrame({
        "subject_id": [1, 2, 3], "split": ["train"] * 3,
        "index_date": pd.to_datetime(["2020-01-01"] * 3),
        "outcome_date": pd.to_datetime([None, None, None]),
        "censor_date": pd.to_datetime(["2020-06-01"] * 3),
    })
    survival = old.copy()
    audit = pd.DataFrame({
        "subject_id": [1, 2, 3], "eligible": [True] * 3,
        "last_measurement_date": pd.to_datetime(["2020-01-05"] * 3),
    })
    labs = pd.DataFrame({
        "subject_id": [1, 2, 3, 3, 2], "analyte": ["potassium"] * 5,
        "sample_date": ["2020-01-02", "2020-01-02", "2020-01-02", "2019-12-31", "2020-01-03"],
        "value": [2.99, 3.0, 2.49, 2.0, None], "unit": ["mmol/L"] * 5,
    })
    result = build(labs, old, survival, audit).set_index("subject_id")
    assert NAME == "potassium_below_3p0_mmol_l"
    assert result.loc[1, "outcome_date"] == pd.Timestamp("2020-01-02")
    assert pd.isna(result.loc[2, "outcome_date"])
    assert result.loc[3, "outcome_date"] == pd.Timestamp("2020-01-02")
    assert result.loc[2, "censor_date"] == pd.Timestamp("2020-06-01")
