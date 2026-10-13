import pandas as pd
import pytest
from opera.run.export_table1 import (
    join_metadata,
    build_tables,
    summarize,
    prepare_survival_table,
)
from opera.run.export_table1 import (
    restrict_to_survival_population,
    derive_baseline_fields,
)


def test_metadata_does_not_expand_population_and_duplicates_fail():
    population = pd.DataFrame({"subject_id": [1, 2], "split": ["train", "tuning"]})
    source = pd.DataFrame({"subject_id": [1, 3], "age": [50, 90]})
    merged = join_metadata(population, [source])
    assert len(merged) == 2
    assert merged.age.iloc[0] == 50
    assert pd.isna(merged.age.iloc[1])
    with pytest.raises(ValueError, match="one nonmissing"):
        join_metadata(population, [pd.concat([source, source])])


def test_observed_denominators_and_missingness_are_explicit():
    frame = pd.DataFrame({"age": [50, 70, None], "female": [1, 0, None]})
    table = summarize(frame).set_index("variable")
    assert table.loc["female", "percent"] == 50
    assert table.loc["female", "n_missing"] == 1
    assert table.loc["age", "median"] == 60
    assert table.loc["recorded_events", "display"] == "Not available"


def test_baseline_contact_context_uses_observed_denominators():
    frame = pd.DataFrame(
        {
            "predominant_region": ["North", "North", None],
            "predominant_region_status": ["unique_mode", "unique_mode", "tied_mode"],
            "n_unique_specialties": [2, 3, None],
        }
    )
    table = summarize(frame)
    region = table.loc[
        (table.variable == "predominant_region") & (table.category == "North")
    ].iloc[0]
    assert region["count"] == 2
    assert region["n_observed"] == 2
    assert region["n_missing"] == 1
    specialty = table.loc[table.variable == "n_unique_specialties"].iloc[0]
    assert specialty["median"] == 2.5
    assert specialty["n_missing"] == 1


def test_disease_and_split_tables_are_separate_and_no_split_inference():
    frame = pd.DataFrame(
        {
            "subject_id": [1, 2, 3],
            "split": ["train", "tuning", "held_out"],
            "cohort_grouped": ["DLBCL_like"] * 3,
            "cohort_fine": ["DLBCL", "DLBCL", "RT"],
        }
    )
    table = build_tables(frame)
    counts = table.loc[table.variable == "patients"]
    assert (
        counts.loc[
            (counts.cohort_level == "all") & (counts.split == "Overall"), "count"
        ].iloc[0]
        == 3
    )
    assert set(counts.cohort_level) == {"all", "cohort_grouped", "cohort_fine"}
    with pytest.raises(ValueError, match="frozen split"):
        build_tables(frame.drop(columns="split"))


def test_followup_requires_overall_survival_not_an_arbitrary_endpoint():
    survival = pd.DataFrame(
        {
            "subject_id": [1],
            "time_days": [365.25],
            "event": [1],
            "outcome": ["overall_survival"],
        }
    )
    converted = prepare_survival_table(survival)
    table = summarize(converted).set_index("variable")
    assert table.loc["followup_days", "median"] == 1
    assert table.loc["death_event", "count"] == 1
    with pytest.raises(ValueError, match="Only overall_survival"):
        prepare_survival_table(survival.assign(outcome="infection"))


def test_population_metadata_retains_distinct_scores_and_treatment_groups():
    frame = pd.DataFrame(
        {
            "Treatment Group": ["A", "B", None],
            "nccn_ipi": [1, 3, None],
            "cll_ipi": [None, None, 2],
            "Performance Status": [0, 2, None],
        }
    )
    table = summarize(frame)
    assert {"Treatment Group", "nccn_ipi", "cll_ipi", "Performance Status"} <= set(
        table.variable
    )
    nccn = table.loc[table.variable == "nccn_ipi"]
    assert set(nccn.n_observed) == {2}
    assert set(nccn.percent) == {50.0}
    cll = table.loc[table.variable == "cll_ipi"]
    assert cll.iloc[0]["n_observed"] == 1
    assert cll.iloc[0]["percent"] == 100.0


def test_split_failure_reports_unknown_and_missing_assignments():
    frame = pd.DataFrame({"subject_id": [1, 2], "split": ["validation", None]})
    with pytest.raises(ValueError, match="validation.*<missing>"):
        build_tables(frame)


def test_index_identifies_prognostic_score_system_not_date():
    frame = pd.DataFrame({"IPI": [2, 2, None], "index": ["IPI", "FLIPI", "IPI"]})
    table = summarize(frame)
    systems = table.loc[table.variable == "index"]
    assert set(systems.characteristic) == {"Prognostic scoring system"}
    paired = table.loc[table.variable == "prognostic_score_by_system"]
    assert set(paired.category) == {"IPI: 2.0", "FLIPI: 2.0"}
    assert set(paired.n_observed) == {2}
    assert set(paired.n_missing) == {1}


def test_explicit_survival_scope_does_not_add_patients_or_hide_invalid_splits():
    population = pd.DataFrame({"subject_id": [1, 2, 3]})
    survival = pd.DataFrame(
        {"subject_id": [1, 2, 4], "split": ["train", "tuning", None]}
    )
    selected, audit = restrict_to_survival_population(population, survival)
    assert selected.subject_id.tolist() == [1, 2]
    assert audit["excluded_population_patients"] == 1
    assert audit["survival_only_patients"] == 1
    build_tables(join_metadata(selected, [survival]))
    with pytest.raises(ValueError, match="<missing>"):
        build_tables(
            join_metadata(selected, [survival.assign(split=["train", None, None])])
        )


def test_actual_metadata_fields_and_age_from_dates():
    frame = pd.DataFrame(
        {
            "PS": [0],
            "IPI": [2],
            "Disease": ["DLBCL"],
            "arm_primary": ["R-CHOP"],
            "arm_day1_primary": ["R-CHOP"],
            "birth_date": ["1960-01-01"],
            "index_date": ["2020-01-01"],
        }
    )
    derived = derive_baseline_fields(frame)
    assert derived.age.iloc[0] == pytest.approx(60, abs=0.01)
    assert {"PS", "IPI", "Disease", "arm_primary"} <= set(summarize(derived).variable)
    assert "arm_day1_primary" not in set(summarize(derived).variable)
    assert derive_baseline_fields(frame.assign(age=55)).age.iloc[0] == 55
    with pytest.raises(ValueError, match="Birth date"):
        derive_baseline_fields(frame.assign(birth_date="2021-01-01"))
