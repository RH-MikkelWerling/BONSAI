import pandas as pd
import pytest

from opera.functional.table1_meds import summarize_meds_batches
from opera.functional.table1_release import release_table
from opera.run.export_table1 import build_tables, summarize


def test_sex_has_no_redundant_binary_row():
    table = summarize(pd.DataFrame({"sex": ["Male", "Female", None]}))
    assert "female" not in set(table.variable)
    assert set(table.loc[table.variable == "sex", "category"]) == {"Male", "Female"}
    assert "female" not in set(summarize(pd.DataFrame({"age": [50]})).variable)


def test_history_is_preindex_unbinned_distinct_and_missing_not_zero():
    pop = pd.DataFrame({"subject_id": [1, 2, 3], "index_date": ["2020-01-10"] * 3})
    events = pd.DataFrame(
        {
            "subject_id": [1, 1, 1, 1, 1, 2, 3],
            "time": [
                "2020-01-01",
                "2020-01-02",
                "2020-01-03",
                "2020-01-10",
                "2020-01-11",
                "2020-01-09",
                None,
            ],
            "code": ["RX//A", "RX//A", "LAB//X", "RX//B", "DX//A", "DX//B", "RX//A"],
        }
    )
    prefixes = {
        "distinct_medications": ["RX//"],
        "distinct_diagnoses": ["DX//"],
        "laboratory_measurements": ["LAB//"],
    }
    values = summarize_meds_batches(
        pop, [events.iloc[:2], events.iloc[2:]], prefixes
    ).set_index("subject_id")
    assert values.loc[1, "recorded_events"] == 3
    assert values.loc[1, "distinct_medications"] == 1
    assert values.loc[1, "distinct_diagnoses"] == 0
    assert values.loc[1, "laboratory_measurements"] == 1
    assert values.loc[2, "distinct_medications"] == 0
    assert pd.isna(values.loc[3, "recorded_events"])
    window = summarize_meds_batches(pop, [events], prefixes, 2).set_index("subject_id")
    assert pd.isna(window.loc[1, "recorded_events"])
    with pytest.raises(ValueError, match="explicit"):
        summarize_meds_batches(pop, [events], {})


def test_demographic_tokens_do_not_make_history_span_equal_age():
    pop = pd.DataFrame({"subject_id": [1, 2], "index_date": ["2020-01-01"] * 2})
    events = pd.DataFrame(
        {
            "subject_id": [1, 1, 2],
            "time": ["1950-01-01", "2019-01-01", "1960-01-01"],
            "code": ["DOB", "DIAGNOSIS//X", "DOB"],
        }
    )
    prefixes = {
        "distinct_medications": ["MEDICATION//"],
        "distinct_diagnoses": ["DIAGNOSIS//"],
        "laboratory_measurements": ["LABTEST//"],
    }
    result = summarize_meds_batches(pop, [events], prefixes).set_index("subject_id")
    assert result.loc[1, "observable_history_years"] == pytest.approx(1, abs=0.01)
    assert result.loc[1, "recorded_events"] == 1
    assert pd.isna(result.loc[2, "observable_history_years"])


def test_contact_context_is_derived_from_preindex_meds_tokens():
    pop = pd.DataFrame({"subject_id": [1, 2, 3], "index_date": ["2020-01-10"] * 3})
    events = pd.DataFrame({
        "subject_id": [1, 1, 1, 1, 1, 2, 2, 2],
        "time": ["2020-01-01", "2020-01-02", "2020-01-03", "2020-01-04",
                 "2020-01-10", "2020-01-01", "2020-01-02", "2020-01-03"],
        "code": ["SPEC//A", "SPEC//B", "REG//X", "REG//X", "REG//Y",
                 "REG//X", "REG//Y", "DIAGNOSIS//C"],
    })
    prefixes = {
        "distinct_medications": ["MEDICATION//"],
        "distinct_diagnoses": ["DIAGNOSIS//"],
        "laboratory_measurements": ["LABTEST//"],
        "n_unique_specialties": ["SPEC//"],
        "predominant_region": ["REG//"],
    }
    result = summarize_meds_batches(pop, [events], prefixes).set_index("subject_id")
    assert result.loc[1, "n_unique_specialties"] == 2
    assert result.loc[1, "predominant_region"] == "REG//X"
    assert result.loc[1, "predominant_region_status"] == "unique_mode"
    assert pd.isna(result.loc[2, "predominant_region"])
    assert result.loc[2, "predominant_region_status"] == "tied_mode"
    assert result.loc[3, "predominant_region_status"] == "no_preindex_history"
    with pytest.raises(ValueError, match="No pre-index MEDS events matched"):
        summarize_meds_batches(pop, [events], {**prefixes, "n_unique_specialties": ["BAD//"]})


def test_bundle_wide_suppression_hides_complements_percentages_and_denominators():
    pop = pd.DataFrame(
        {
            "subject_id": range(30),
            "split": ["train"] * 10 + ["tuning"] * 10 + ["held_out"] * 10,
            "sex": ["Female"] * 2 + ["Male"] * 28,
            "age": [50] * 30,
        }
    )
    raw = build_tables(pop)
    released, policy = release_table(raw, policy="conservative")
    sex = released.loc[released.variable == "sex"]
    assert sex.suppressed.all()
    assert sex["count"].isna().all() and sex.percent.isna().all()
    assert "<5" in set(sex.display) and "Suppressed" in set(sex.display)
    assert not {"n_total", "n_observed", "n_missing"} & set(released)
    assert raw.loc[raw.variable == "sex", "count"].notna().all()
    assert policy["review_required"]


def test_small_missingness_and_small_binary_complement_are_protected():
    pop = pd.DataFrame(
        {
            "subject_id": range(20),
            "split": ["train"] * 20,
            "sex": ["Male"] * 19 + [None],
            "death_event": [1] * 19 + [0],
        }
    )
    released, _ = release_table(build_tables(pop), policy="conservative")
    assert released.loc[
        released.variable.isin(["sex", "death_event"]), "suppressed"
    ].all()


def test_tiny_cohorts_hide_totals_and_cross_level_cohort_distributions():
    pop = pd.DataFrame(
        {
            "subject_id": range(20),
            "split": ["train"] * 20,
            "cohort_fine": ["rare"] * 2 + ["common"] * 18,
            "cohort_grouped": ["A"] * 20,
            "age": [50] * 20,
        }
    )
    released, _ = release_table(build_tables(pop), policy="conservative")
    assert released.loc[
        released.variable.isin(["patients", "cohort_fine", "cohort_grouped"]),
        "suppressed",
    ].all()
    assert released.loc[released.cohort == "rare", "suppressed"].all()


def test_targeted_suppression_keeps_large_unaffected_sex_cells():
    pop = pd.DataFrame(
        {
            "subject_id": range(300),
            "split": ["train"] * 100 + ["tuning"] * 100 + ["held_out"] * 100,
            "sex": ["Female"] * 2 + ["Male"] * 98 + ["Female", "Male"] * 100,
            "arm_primary": ["Rare"] * 2 + ["Common"] * 298,
            "arm_day1_primary": ["Not used"] * 300,
        }
    )
    raw = build_tables(pop)
    assert "arm_day1_primary" not in set(raw.variable)
    release, _ = release_table(raw)
    tuning_sex = release.loc[(release.variable == "sex") & (release.split == "tuning")]
    assert not tuning_sex.suppressed.any()
    assert set(tuning_sex["count"]) == {50}
    rare = release.loc[
        (release.variable == "arm_primary") & (release.category == "Rare")
    ]
    assert rare.suppressed.all()
    assert set(rare.display) == {"<5"}
    common_overall = release.loc[
        (release.variable == "arm_primary")
        & (release.category == "Common")
        & (release.split == "Overall")
    ]
    assert common_overall.suppressed.all()


def test_primary_only_review_does_not_spread_suppression():
    pop = pd.DataFrame(
        {
            "subject_id": range(20),
            "split": ["train"] * 20,
            "arm_primary": ["Rare"] * 2 + ["Common"] * 18,
            "sex": ["Male", "Female"] * 10,
        }
    )
    result, policy = release_table(build_tables(pop), policy="primary-only")
    rare = result.loc[(result.variable == "arm_primary") & (result.category == "Rare")]
    common = result.loc[
        (result.variable == "arm_primary") & (result.category == "Common")
    ]
    assert set(rare.display) == {"<5"}
    assert rare["count"].isna().all() and rare.percent.isna().all()
    assert not common.suppressed.any()
    assert not result.loc[result.variable == "sex", "suppressed"].any()
    assert policy["disclosure_clearance"] is False
