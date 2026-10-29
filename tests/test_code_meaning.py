"""From model tokens to clinical meaning: names, groups, chapters and the audit."""

import pandas as pd
import pytest

from opera.interpretability.attribution_patterns import _code_label, concept_label
from opera.interpretability.attribution_summaries import source_of
from opera.interpretability.code_meaning import CodeDictionary, normalise
from opera.run.audit_vocabulary import audit, token_format

REVIEWED = {"DIAGNOSIS//DJ18": "Pneumonia", "MEDICATION//J01C": "Penicillins",
            "LABTEST//NPU19673//BIN": "Albumin"}


@pytest.mark.parametrize("system,code,expected", [
    ("icd10", "DC833", "C833"), ("icd10", "C83.3", "C833"), ("icd10", "D50", "D50"),
    ("icd10", "DD509", "D509"), ("atc", "j01ca04", "J01CA04"), ("atc", "MJ01CA04", "J01CA04"),
])
def test_codes_are_normalised_from_sks_and_who_forms(system, code, expected):
    assert normalise(system, code) == expected


def test_sources_cover_every_register_and_demographic_format():
    for token, source in [("DIAGNOSIS//DC833", "Diagnoses"), ("SEX//F", "Demographics"),
                          ("BG_SEX//M", "Demographics"), ("AGE//71", "Demographics"),
                          ("RC_SPEC//HAEM", "Hospital contacts"), ("LABTEST//NPU1//BIN_2",
                                                                    "Laboratory"),
                          ("[PAD]", "Other"), ("FOO//BAR", "Other")]:
        assert source_of(token) == source


def test_names_come_from_the_finest_named_level_reviewed_first():
    labels = CodeDictionary(REVIEWED, {"atc": {"J01CA04": "Phenoxymethylpenicillin"}})
    assert labels.resolve("MEDICATION//J01CA04") == ("Phenoxymethylpenicillin", "table")
    assert labels.resolve("MEDICATION//J01CE02") == ("Penicillins", "reviewed")
    assert labels.resolve("MEDICATION//N05BA01") == ("Psycholeptics", "built-in")
    assert labels.resolve("DIAGNOSIS//DJ189") == ("Pneumonia", "reviewed")
    assert labels.resolve("DIAGNOSIS//DC833") == ("Non-Hodgkin lymphoma", "group")
    assert labels.resolve("DIAGNOSIS//DZ001") == ("ICD-10 Z00.1", "code")
    assert labels.name("PATHOLOGY//M96803") == "Morphology codes M96"
    assert labels.name("SEX//F") == "Female"
    assert labels.name("RC_SPEC//HAEM") == "Specialty haem"
    # ehr2meds formats: untimed codes, procedure types, text lab results.
    assert labels.name("BACKGROUND//SEX//M") == "Male"
    assert source_of("PROCEDURE_TYPE//A") == "Procedures"
    assert source_of("DOB") == "Demographics"
    assert labels.name("LABTEST//NPU19673//NOT_DETECTED") == "Albumin not detected"


def test_groups_follow_the_outcome_definitions_and_chapters_the_classification():
    labels = CodeDictionary()
    assert labels.group("DIAGNOSIS//DA419") == "Sepsis"
    assert labels.group("DIAGNOSIS//DJ151") == "Pneumonia"  # range J12-J18
    assert labels.group("DIAGNOSIS//DI509") == "Heart failure"
    assert labels.group("MEDICATION//L03AA02") == "Colony stimulating factors"  # longest wins
    assert labels.group("PROCEDURE//BOQA05") == "Blood transfusion"
    assert labels.chapter("DIAGNOSIS//DD700") == "Blood and immune system"
    assert labels.chapter("DIAGNOSIS//DC833") == "Neoplasms"
    assert labels.chapter("MEDICATION//J01CA04") == "Antiinfectives for systemic use"
    assert labels.chapter("MEDICATION//BOHJ11") == "Treatment and care"


def test_dictionary_drops_into_the_attribution_labels():
    labels = CodeDictionary(REVIEWED)
    counts = {"NPU19673": {"min": 0, "max": 8}}
    assert concept_label("LABTEST//NPU19673//low", labels) == "Albumin, lower bins"
    assert concept_label("LABTEST//NPU19673//mid", labels) == "Albumin, middle bins"
    assert _code_label("LABTEST//NPU19673//BIN_8", labels, counts) == "Albumin, upper bins"
    assert _code_label("MEDICATION//J01CA04", labels) == "Penicillins"
    assert concept_label("DIAGNOSIS//DI50", labels) == "Heart failure"
    assert concept_label("FOO//BAR", labels) == "Foo BAR"  # unknown formats stay readable


def test_official_tables_are_read_and_npu_definitions_shortened(tmp_path):
    path = tmp_path / "npu.csv"
    pd.DataFrame({"NPU code": ["NPU19673", "NPU03958"],
                  "Short definition": ["P—Albumin; mass c. = ? g/L",
                                       "U—Albumin; mass c. = ? mg/L"]}).to_csv(path, index=False)
    config = {"code_tables": [{"system": "npu", "path": str(path), "code": "NPU code",
                               "label": "Short definition"}]}
    labels = CodeDictionary.from_config(config)
    assert labels.name("LABTEST//NPU19673//BIN_2") == "Albumin"
    assert labels.name("LABTEST//NPU03958//BIN_2") == "Albumin (urine)"
    with pytest.raises(FileNotFoundError):
        CodeDictionary.from_config({"code_tables": [{"system": "atc", "path": "missing.csv"}]})


def test_reviewed_labels_wait_for_review():
    config = {"concept_mapping_reviewed": False, "summary": {"concept_labels": REVIEWED}}
    assert CodeDictionary.from_config(config).name("DIAGNOSIS//DJ18") == "Pneumonia"  # group
    assert CodeDictionary.from_config(config).resolve("DIAGNOSIS//DJ18")[1] == "group"
    config["concept_mapping_reviewed"] = True
    assert CodeDictionary.from_config(config).resolve("DIAGNOSIS//DJ18")[1] == "reviewed"


def test_audit_reports_formats_coverage_and_what_to_review():
    vocabulary = {t: i for i, t in enumerate([
        "[PAD]", "SEX//F", "DIAGNOSIS//DC833", "DIAGNOSIS//DZ001", "MEDICATION//J01CA04",
        "LABTEST//NPU19673//BIN_1", "LABTEST//NPU19673//BIN_2", "NEWREG//X1"])}
    importance = pd.DataFrame({"concept": ["DIAGNOSIS//DZ00"], "outcome": ["death"],
                               "mean_pp": [0.4]})
    meanings, formats, coverage, review = audit(vocabulary, CodeDictionary(REVIEWED), importance)
    assert len(meanings) == len(vocabulary)
    assert "NEWREG//<A9>" in set(formats.loc[formats.source == "Other", "format"])
    assert token_format("LABTEST//NPU19673//BIN_1") == "LABTEST//<A9>//BIN_<9>"
    assert review.concept.iloc[0] == "DIAGNOSIS//DZ00"
    assert "Laboratory" in set(coverage.source)


def test_concept_coverage_flags_findings_that_match_nothing():
    from opera.run.audit_vocabulary import concept_coverage

    vocabulary = {t: i for i, t in enumerate([
        "LABTEST//NPU19673//BIN_0", "LABTEST//NPU19673//BIN_7", "DIAGNOSIS//DI509"])}
    config = {"concepts": [{"name": "hf", "label": "Heart failure", "codes": ["DIAGNOSIS//DI50"]},
                           {"name": "crp", "label": "High CRP",
                            "codes": ["LABTEST//NPU19748//BIN_"]}],
              "lab_tests": [{"npu": "NPU19673", "label": "Albumin"}]}
    table = concept_coverage(vocabulary, config, CodeDictionary(REVIEWED)).set_index("name")
    assert table.loc["hf", "status"] == "OK" and table.loc["crp", "status"] == "NO MATCH"
    assert table.loc["NPU19673", "bins"] == "0-7"
    assert table.loc["NPU19673", "dictionary_names"] == "Albumin"


def test_molecular_pathology_codes_are_not_merged_into_families():
    from opera.interpretability.attribution_summaries import rollup

    assert rollup("PATHOLOGY//M96803") == "PATHOLOGY//M96"  # morphology family
    assert rollup("PATHOLOGY//T06000") == "PATHOLOGY//T06"  # topography family
    assert rollup("PATHOLOGY//F29890") == "PATHOLOGY//F29890"  # e.g. a molecular finding


def test_findings_without_codes_yet_match_nothing_and_are_flagged():
    import pandas as pd

    from opera.interpretability.attribution_summaries import concept_effects
    from opera.run.audit_vocabulary import concept_coverage

    tokens = pd.DataFrame({"subject_id": [1, 2], "code": ["PATHOLOGY//F1", "PATHOLOGY//F2"],
                           "attribution": [0.1, 0.2], "outcome": "death",
                           "days_before_index": [3.0, 4.0], "position": [0, 0]})
    table = concept_effects(tokens, [{"name": "tp53", "codes": []}], {}, draws=10, min_count=1)
    assert table.verdict.iloc[0] == "too few patients"
    coverage = concept_coverage({"PATHOLOGY//F1": 0}, {"concepts": [
        {"name": "tp53", "label": "TP53 aberration", "codes": []}]}, CodeDictionary())
    assert coverage.status.iloc[0] == "NO MATCH"
