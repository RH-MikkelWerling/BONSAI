"""Contract checks for the isolated TLS candidate labels."""

import pandas as pd

from opera.run.export_candidate_tls_v1 import NAMES, STRICT_NAMES, _validate_labs, build_tls


def _lab_audit(survival: pd.DataFrame, eligible: list[bool] | None = None) -> pd.DataFrame:
    return pd.DataFrame({
        "subject_id": survival["subject_id"],
        "split": survival["split"],
        "index_date": survival["index_date"],
        "eligible": eligible if eligible is not None else [True] * len(survival),
        "last_measurement_date": survival["index_date"] + pd.Timedelta(days=7),
    })


def test_urate_mmol_is_converted_to_micromol_without_changing_other_axes() -> None:
    units = {
        "urate": "mmol/L", "potassium": "mmol/L", "phosphate": "mmol/L",
        "calcium": "mmol/L", "creatinine": "μmol/L",
    }
    frame = pd.DataFrame({
        "subject_id": [1] * 5,
        "analyte": list(units),
        "sample_date": ["2022-01-03"] * 5,
        "value": [0.5, 6.1, 1.2, 2.2, 100.0],
        "unit": list(units.values()),
        "ref_upper": [0.4, 5.0, 1.45, 2.6, 90.0],
    })
    cleaned = _validate_labs(frame).set_index("analyte")
    assert cleaned.loc["urate", "value"] == 500.0
    assert cleaned.loc["urate", "unit"] == "µmol/l"
    assert cleaned.loc["potassium", "value"] == 6.1
    assert cleaned.loc["creatinine", "unit"] == "µmol/l"
    assert cleaned.loc["creatinine", "ref_upper"] == 90.0


def test_tls_and_kidney_are_nested_and_require_same_day_panel() -> None:
    index = pd.Timestamp("2022-01-01")
    survival = pd.DataFrame({
        "subject_id": [1, 2, 3], "split": ["train"] * 3,
        "index_date": [index] * 3,
        "censor_date": [index + pd.Timedelta(days=90)] * 3,
    })
    rows = []
    for patient, day, urate, potassium, creatinine in (
        (1, 2, 500.0, 6.2, 120.0),
        (2, 3, 500.0, 6.2, 80.0),
        (3, 2, 500.0, 6.2, 120.0),
    ):
        for analyte, value in (
            ("urate", urate), ("potassium", potassium),
            ("phosphate", 1.0), ("calcium", 2.2), ("creatinine", creatinine),
        ):
            if patient == 3 and analyte == "calcium":
                continue
            rows.append({"subject_id": patient, "analyte": analyte,
                         "sample_date": index + pd.Timedelta(days=day),
                         "value": value, "ref_upper": 100.0 if analyte == "creatinine" else 10.0})
    results, audit = build_tls(pd.DataFrame(rows), survival, _lab_audit(survival))
    lab = results[NAMES[0]].set_index("subject_id")
    kidney = results[NAMES[1]].set_index("subject_id")
    assert set(lab.index) == {1, 2, 3}
    assert set(kidney.index) == {1, 2, 3}
    assert lab["outcome_date"].notna().sum() == 3
    assert kidney["outcome_date"].notna().sum() == 2
    assert set(results[STRICT_NAMES[0]]["subject_id"]) == {1, 2}
    assert results[STRICT_NAMES[0]]["outcome_date"].notna().sum() == 2
    assert audit.loc[0, "n_complete_four_axis_same_day"] == 2
    assert audit.loc[0, "n_tls_signal_before_coverage_filter"] == 3
    assert lab.loc[1, "censor_date"] == index + pd.Timedelta(days=30)


def test_tls_event_requires_complete_panel_on_event_day() -> None:
    index = pd.Timestamp("2022-01-01")
    survival = pd.DataFrame({
        "subject_id": [1], "split": ["train"], "index_date": [index],
        "censor_date": [index + pd.Timedelta(days=90)],
    })
    rows = []
    for day, values in (
        (2, {"urate": 500.0, "potassium": 6.2, "creatinine": 120.0}),
        (3, {"urate": 200.0, "potassium": 4.0, "phosphate": 1.0,
             "calcium": 2.2, "creatinine": 80.0}),
    ):
        for analyte, value in values.items():
            rows.append({
                "subject_id": 1, "analyte": analyte,
                "sample_date": index + pd.Timedelta(days=day),
                "value": value,
                "ref_upper": 100.0 if analyte == "creatinine" else 10.0,
            })
    results, audit = build_tls(pd.DataFrame(rows), survival, _lab_audit(survival))
    assert results[NAMES[0]]["outcome_date"].notna().all()
    assert results[NAMES[1]]["outcome_date"].notna().all()
    assert results[STRICT_NAMES[0]]["outcome_date"].isna().all()
    assert results[STRICT_NAMES[1]]["outcome_date"].isna().all()
    assert audit.loc[0, "n_complete_four_axis_same_day"] == 1


def test_shared_gate_excludes_uncovered_signal_but_strict_sensitivity_is_separate() -> None:
    index = pd.Timestamp("2022-01-01")
    survival = pd.DataFrame({
        "subject_id": [1, 2], "split": ["train", "train"],
        "index_date": [index, index],
        "censor_date": [index + pd.Timedelta(days=90)] * 2,
    })
    rows = []
    for subject_id in [1, 2]:
        for analyte, value in [("urate", 500.0), ("potassium", 6.2),
                               ("phosphate", 1.0), ("calcium", 2.2), ("creatinine", 80.0)]:
            rows.append({"subject_id": subject_id, "analyte": analyte,
                         "sample_date": index + pd.Timedelta(days=2),
                         "value": value, "ref_upper": 100.0})
    results, _ = build_tls(pd.DataFrame(rows), survival,
                           _lab_audit(survival, eligible=[True, False]))
    assert results[NAMES[0]]["subject_id"].tolist() == [1]
    assert set(results[STRICT_NAMES[0]]["subject_id"]) == {1, 2}
