"""Boundary and observation-contract checks for isolated candidate outcomes."""

import pandas as pd

from opera.run.export_candidate_outcomes_v1 import (
    _candidate_flags, _event_overlap, _gradable_baseline, build_candidate,
)


def test_numeric_grade_thresholds_are_strict_and_nested() -> None:
    neutrophils = pd.DataFrame({"value": [1.0, 0.99, 0.5, 0.49]})
    g2, g3 = _candidate_flags(neutrophils, "neutrophils")
    assert g2.tolist() == [False, True, True, True]
    assert g3.tolist() == [False, False, False, True]

    alt = pd.DataFrame({
        "value": [3.0, 3.01, 5.0, 5.01, 3.0, 3.01, 4.0, 4.01],
        "baseline_value": [0.5] * 4 + [2.0] * 4,
        "baseline_ref_upper": [1.0] * 8,
        "ref_upper": [1.0] * 8,
    })
    g2, g3 = _candidate_flags(alt, "ALT")
    assert g2.tolist() == [False, True, True, True, False, True, True, True]
    assert g3.tolist() == [False, False, False, True, False, False, False, True]


def test_creatinine_ctcae_and_relative_rise_are_distinct() -> None:
    creatinine = pd.DataFrame({
        "value": [74.9, 75.0, 150.0, 150.1, 180.0, 301.0],
        "baseline_value": [50.0, 50.0, 100.0, 100.0, 120.0, 120.0],
        "baseline_ref_lower": [60.0] * 6,
        "ref_upper": [100.0] * 6,
    })
    g2, g3 = _candidate_flags(creatinine, "creatinine")
    relative, = _candidate_flags(creatinine, "creatinine_relative")
    assert g2.tolist() == [False, False, False, True, True, True]
    assert g3.tolist() == [False, False, False, False, False, True]
    assert relative.tolist() == [False, False, False, True, False, True]

    # Relative rise can be positive without CTCAE G2+: 100 -> 151 with ULN 110.
    different = pd.DataFrame({
        "value": [151.0], "baseline_value": [100.0],
        "baseline_ref_lower": [60.0], "ref_upper": [110.0],
    })
    assert not _candidate_flags(different, "creatinine")[0].iloc[0]
    assert _candidate_flags(different, "creatinine_relative")[0].iloc[0]


def test_relative_creatinine_does_not_require_reference_range() -> None:
    frame = pd.DataFrame({
        "baseline_value": [100.0, 100.0, 0.0],
        "baseline_ref_lower": [None, 60.0, 60.0],
    })
    assert _gradable_baseline(frame, "creatinine_relative").tolist() == [True, True, False]
    assert _gradable_baseline(frame, "creatinine").tolist() == [False, True, False]


def test_relative_creatinine_uses_preindex_draw_and_own_eligibility() -> None:
    index = pd.Timestamp("2022-01-10")
    admin = pd.Timestamp("2022-12-31")
    old = pd.DataFrame({
        "subject_id": [1, 2], "split": ["train", "train"],
        "index_date": [index, index], "outcome_date": [pd.NaT, pd.NaT],
        "censor_date": [admin, admin],
    })
    audit = pd.DataFrame({
        "subject_id": [1, 2], "eligible": [True, True],
        "last_measurement_date": [index + pd.Timedelta(days=10)] * 2,
    })
    survival = old.copy()
    labs = pd.DataFrame({
        "subject_id": [1, 1, 1, 2], "analyte": ["creatinine"] * 4,
        "sample_date": [index - pd.Timedelta(days=4),
                        index - pd.Timedelta(days=1),
                        index + pd.Timedelta(days=2), index + pd.Timedelta(days=2)],
        "value": [80.0, 100.0, 151.0, 151.0],
        "ref_lower": [60.0, 60.0, None, None],
        "ref_upper": [110.0, 110.0, None, None],
    })
    frames, summary = build_candidate("creatinine_relative", labs, old, audit, survival)
    result = frames["creatinine_rise_gt1p5x_preindex_baseline"]
    assert result["subject_id"].tolist() == [1]
    assert result.iloc[0]["outcome_date"] == index + pd.Timedelta(days=2)
    assert summary["excluded_missing_baseline_or_reference"] == 1

    bilirubin = pd.DataFrame({
        "value": [1.5, 1.51, 3.0, 3.01, 3.0, 3.01, 5.0, 5.01],
        "baseline_value": [0.5] * 4 + [2.0] * 4,
        "baseline_ref_upper": [1.0] * 8, "ref_upper": [1.0] * 8,
    })
    g2, g3 = _candidate_flags(bilirubin, "bilirubin")
    assert g2.tolist() == [False, True, True, True, False, True, True, True]
    assert g3.tolist() == [False, False, False, True, False, False, False, True]


def test_missing_baseline_is_not_treated_as_event_free() -> None:
    frame = pd.DataFrame({
        "baseline_value": [None, 0.0, 1.0], "baseline_ref_upper": [1.0] * 3,
    })
    assert _gradable_baseline(frame, "ALT").tolist() == [False, False, True]


def test_candidate_recomputes_event_free_censor_after_label_loss() -> None:
    index = pd.Timestamp("2022-01-01")
    admin = pd.Timestamp("2022-12-31")
    old = pd.DataFrame({
        "subject_id": [1, 2], "split": ["tuning", "tuning"],
        "index_date": [index, index],
        "outcome_date": [index + pd.Timedelta(days=5), pd.NaT],
        "censor_date": [admin, index + pd.Timedelta(days=210)],
    })
    audit = pd.DataFrame({
        "subject_id": [1, 2], "eligible": [True, True],
        "last_measurement_date": [index + pd.Timedelta(days=10), index + pd.Timedelta(days=30)],
    })
    survival = old.copy()
    survival["censor_date"] = admin
    labs = pd.DataFrame({
        "subject_id": [1, 2], "analyte": ["neutrophils"] * 2,
        "sample_date": [index + pd.Timedelta(days=5), index + pd.Timedelta(days=20)],
        "value": [0.7, 0.4], "ref_lower": [2.0] * 2, "ref_upper": [8.0] * 2,
    })
    results, summary = build_candidate("neutrophils", labs, old, audit, survival)
    g3 = results["neutropenia_g3plus"].set_index("subject_id")
    assert pd.isna(g3.loc[1, "outcome_date"])
    assert g3.loc[1, "censor_date"] == index + pd.Timedelta(days=190)
    assert g3.loc[2, "outcome_date"] == index + pd.Timedelta(days=20)
    assert g3.loc[2, "censor_date"] == admin
    assert summary["excluded_missing_baseline_or_reference"] == 0


def test_event_overlap_separates_eligibility_loss_from_changed_labels() -> None:
    index = pd.Timestamp("2022-01-01")
    old = pd.DataFrame({
        "subject_id": [1, 2, 3, 4], "split": ["tuning"] * 4,
        "index_date": [index] * 4,
        "outcome_date": [index + pd.Timedelta(days=5),
                         index + pd.Timedelta(days=10), pd.NaT,
                         index + pd.Timedelta(days=3)],
    })
    candidate = pd.DataFrame({
        "subject_id": [1, 2, 3], "split": ["tuning"] * 3,
        "index_date": [index] * 3,
        "outcome_date": [index + pd.Timedelta(days=7), pd.NaT,
                         index + pd.Timedelta(days=20)],
    })
    row = next(r for r in _event_overlap(old, candidate, "ALT_increased_g2plus")
               if r["split"] == "tuning" and r["horizon_days"] == 30)
    assert (row["both_event_on_shared"], row["old_only_event_on_shared"],
            row["candidate_only_event_on_shared"]) == (1, 1, 1)
    assert row["lost_eligibility"] == 1
    assert row["old_events_on_lost_eligibility"] == 1
    assert row["jaccard_event_overlap_on_shared"] == 1 / 3
