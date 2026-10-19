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


def test_manuscript_table_renders_released_display_values_verbatim():
    from opera.run.format_table1_release import pooled_table_latex

    pop = pd.DataFrame(
        {
            "subject_id": range(300),
            "split": ["train"] * 100 + ["tuning"] * 100 + ["held_out"] * 100,
            "sex": ["Female"] * 2 + ["Male"] * 98 + ["Female", "Male"] * 100,
            "age": list(range(40, 90)) * 6,
            "arm_primary": ["Rare"] * 2 + ["Common_arm"] * 298,
        }
    )
    release, _ = release_table(build_tables(pop))
    tex = pooled_table_latex(release, source_sha256="abc123")
    assert "export sha256 abc123" in tex and "Review version" in tex
    assert r"\quad Common\_arm" in tex  # escaped, indented category row
    assert "$<$5" in tex  # suppressed small cell kept as released
    pooled = release.loc[(release.cohort_level == "all") & (release.kind != "continuous")]
    for value in pooled.display.dropna().astype(str):
        if value == "Not available":  # rows without data anywhere are left out
            assert value not in tex
            continue
        assert value.replace("<", "$<$").replace("%", r"\%") in tex
    # Continuous rows are reformatted from the exact median/IQR (age in whole years).
    assert "65 (52–77)" in tex
    assert tex.count(r"\begin{table*}") == 1
    assert "Review version" not in pooled_table_latex(
        release, source_sha256="abc123", status="approved"
    )


def test_contact_days_and_common_specialties_are_summarised():
    from opera.functional.table1_meds import summarize_meds_batches

    pop = pd.DataFrame({"subject_id": [1, 2], "index_date": ["2020-01-10", "2020-01-10"]})
    events = pd.DataFrame({
        "subject_id": [1, 1, 1, 2, 2],
        "time": ["2020-01-01 08:00", "2020-01-01 15:00", "2020-01-05 00:00", "2020-01-02 00:00", "2020-01-20 00:00"],
        "code": ["SPEC//Haematology", "SPEC//Haematology", "SPEC//Cardiology",
                 "SPEC//Haematology", "SPEC//Cardiology"]})
    prefixes = {"distinct_medications": ["MED//"], "distinct_diagnoses": ["DX//"],
                "laboratory_measurements": ["LAB//"], "hospital_encounters": ["SPEC//"],
                "n_unique_specialties": ["SPEC//"]}
    result = summarize_meds_batches(pop, [events], prefixes).set_index("subject_id")
    assert result.loc[1, "hospital_encounters"] == 2  # two distinct contact days
    assert result.loc[2, "hospital_encounters"] == 1  # post-index contact excluded
    assert result.loc[1, "contact_specialty__Cardiology"] == 1
    assert result.loc[2, "contact_specialty__Cardiology"] == 0


def test_small_cells_shown_as_less_than_five_in_both_manuscript_tables():
    from opera.run.format_table1_release import disease_table_latex, pooled_table_latex

    pop = pd.DataFrame({
        "subject_id": range(300),
        "split": ["train"] * 100 + ["tuning"] * 100 + ["held_out"] * 100,
        "sex": ["Female"] * 2 + ["Male"] * 98 + ["Female", "Male"] * 100,
        "age": list(range(40, 90)) * 6,
        "female_flag": 0,
        "cohort_grouped": ["DLBCL_like"] * 150 + ["MM"] * 150,
        "contact_specialty__Cardiology": [1] * 3 + [0] * 297,
    })
    table = build_tables(pop)
    release, _ = release_table(table, policy="primary-only")
    tex = pooled_table_latex(release, source_sha256="abc", policy="primary-only")
    assert "$<$5" in tex and "Counts of 1--4 are shown as $<$5" in tex
    assert "Contact with specialty" not in tex  # kept to the supplementary table
    assert r"\quad 1 &" not in tex  # binary rows are single rows
    supplement = disease_table_latex(release, source_sha256="abc", policy="primary-only")
    assert "DLBCL-like" in supplement and "Myeloma" in supplement
    assert "Contact with specialty" in supplement and r"\quad Cardiology" in supplement



def test_subtypes_are_nested_under_their_disease_group():
    from opera.run.format_table1_release import pooled_table_latex

    pop = pd.DataFrame({
        "subject_id": range(60), "split": "train", "sex": ["Female", "Male"] * 30,
        "cohort_fine": ["RT"] * 10 + ["DLBCL"] * 20 + ["HL"] * 30,
        "cohort_grouped": ["DLBCL_like"] * 30 + ["HL"] * 30})
    release, _ = release_table(build_tables(pop), policy="primary-only")
    tex = pooled_table_latex(release, source_sha256="x", policy="primary-only")
    lines = tex.splitlines()
    group = next(i for i, line in enumerate(lines)
                 if line.startswith(r"\quad \textbf{DLBCL and related"))
    # Subtypes by size, full names; RT has no index in routine use.
    assert lines[group + 1].startswith(r"\qquad Diffuse large B-cell lymphoma (DLBCL) &")
    assert lines[group + 2].startswith(r"\qquad Richter transformation (RT)$^{\dagger}$")
    # A single-subtype group is one bold row under the subtype's name, after the
    # aggressive B-cell group (clinical, not alphabetical, order).
    hodgkin = next(i for i, line in enumerate(lines)
                   if line.startswith(r"\quad \textbf{Hodgkin lymphoma (HL)}"))
    assert hodgkin > group and not any(line.startswith(r"\qquad Hodgkin") for line in lines)
    assert "No dedicated, externally validated prognostic" in tex


def test_layout_preview_shows_every_subtype_without_data():
    from opera.run.format_table1_release import layout_table_latex
    from opera.visualization.paper_taxonomy import FINE_LONG, fine_cohort_groups

    tex = layout_table_latex()
    for fine in fine_cohort_groups():
        assert FINE_LONG[fine].replace("_", r"\_") in tex
    assert "Layout preview" in tex and "Review version" not in tex
    assert not any(ch.isdigit() for line in tex.splitlines()
                   if line.startswith(r"\q") for ch in line.split("&", 1)[1])


def test_no_index_stratum_is_one_list_of_registry_identifiers():
    """The analysis stratum, the Table 1 dagger and the paper's list agree."""
    from pathlib import Path

    import yaml

    from opera.visualization.paper_taxonomy import NO_ROUTINE_INDEX, fine_cohort_groups

    repo = Path(__file__).resolve().parents[1]
    config = yaml.safe_load((repo / "opera/configs/locked_prediction_analysis.example.yaml")
                            .read_text())
    strata = config["cohort_strata"] if "cohort_strata" in config else config
    found = {}

    def walk(node):
        if isinstance(node, dict):
            for key, value in node.items():
                if key.startswith("no_index") and isinstance(value, dict) and "values" in value:
                    found[key] = set(value["values"])
                walk(value)
    walk(strata)
    registry = set(fine_cohort_groups())
    assert found["no_index_subtype"] == NO_ROUTINE_INDEX
    for values in found.values():
        assert values <= registry  # registry identifiers, not display names
    assert found["no_index_family_level"] == NO_ROUTINE_INDEX - {"NMZL"}
    assert found["no_index_guideline_use"] == NO_ROUTINE_INDEX | {"AITL", "BL"}


def test_supervised_population_keeps_indexed_split_patients_and_counts_the_rest():
    import pandas as pd
    import pytest
    from opera.run.export_table1 import supervised_population

    population = pd.DataFrame({
        "subject_id": [1, 2, 3, 4, 5],
        "index_date": ["2020-01-01", None, "2022-03-01", "2023-05-01", "2019-02-02"],
        "split": ["train", "train", "tuning", "held_out", None],
    })
    kept, audit = supervised_population(population)
    assert kept.subject_id.tolist() == [1, 3, 4]
    assert audit["excluded_without_index_date"] == 1
    assert audit["excluded_with_index_but_no_split"] == 1
    assert audit["by_split"] == {"train": 1, "tuning": 1, "held_out": 1}
    duplicated = pd.concat([population, population.iloc[[0]]])
    with pytest.raises(ValueError, match="more than one row"):
        supervised_population(duplicated)


def test_supervised_scope_uses_index_and_split_from_the_joined_sources():
    import pandas as pd
    from opera.run.export_table1 import join_metadata, supervised_population

    population = pd.DataFrame({"subject_id": [1, 2, 3], "diagnosis": ["DLBCL", "FL", "CLL"]})
    survival = pd.DataFrame({"subject_id": [1, 3], "index_date": ["2020-01-01", "2023-02-01"],
                             "split": ["train", "held_out"]})
    kept, audit = supervised_population(join_metadata(population, [survival]))
    assert kept.subject_id.tolist() == [1, 3]
    assert audit["excluded_without_index_date"] == 1
