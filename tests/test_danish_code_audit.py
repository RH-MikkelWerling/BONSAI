"""Guard the distinctions established by the public Danish code audit."""
from pathlib import Path

import pandas as pd
import yaml

from opera.interpretability.attribution_summaries import concept_tokens, rollup
from opera.interpretability.code_meaning import CodeDictionary
from opera.run.audit_danish_code_assignments import PatobankTable, read_sks
from opera.run.audit_vocabulary import (concept_coverage, adaptive_concept_coverage,
                                        laboratory_bin_definitions)

ROOT = Path(__file__).resolve().parents[1]


def test_molecular_concepts_include_only_the_specified_positive_results():
    config = yaml.safe_load((ROOT / "opera/configs/attribution_analysis.example.yaml").read_text())
    concepts = {c["name"]: c for c in config["concepts"]}
    cases = {
        "tp53_aberration": (["FE16E3", "FE16E4", "FK17D0", "M632C0"],
                            ["FE16E0", "FE16E1", "FE16E2", "FE16E5", "FE16E9", "FK17D1"]),
        "myc_rearrangement": (["FE18UA", "M636D0", "J3N8U1", "J3N8U3", "JFR8U1"],
                              ["FE18UB", "FE18U3", "FE18U5", "J08140", "J08141", "J3N8U0"]),
        "ighv_unmutated": (["FE13N8"], ["FE13N0", "FE13N3", "FE13N7"]),
    }
    for name, (positive, other) in cases.items():
        codes = [f"PATHOLOGY//{c}" for c in positive + other + [positive[0] + "0"]]
        codes += [f"MEDICATION//{positive[0]}"]  # namespace is part of the meaning
        frame = pd.DataFrame({"code": codes})
        hits = concept_tokens(frame, concepts[name], {}).code.tolist()
        assert hits == [f"PATHOLOGY//{c}" for c in positive]
        coverage = concept_coverage(dict.fromkeys(codes, 0), {"concepts": [concepts[name]]},
                                    CodeDictionary()).iloc[0]
        assert coverage.tokens == len(positive)
        assert coverage.match_mode == "exact"


def test_broad_code_families_are_not_presented_as_narrow_diagnoses():
    labels = CodeDictionary()
    assert labels.group("PROCEDURE//BOQA05") == "Blood transfusion"
    assert labels.group("DIAGNOSIS//DI480") == "Atrial fibrillation or flutter"
    assert "thrombophlebitis" in labels.group("DIAGNOSIS//DI800")
    assert labels.group("DIAGNOSIS//DB448A") == "Selected fungal diagnoses"
    assert labels.group("DIAGNOSIS//DB599") == "Pneumocystosis"
    assert labels.group("DIAGNOSIS//DB485") == "Pneumocystosis"
    assert labels.group("DIAGNOSIS//DB484") != "Pneumocystosis"
    assert labels.group("MEDICATION//MH02CA01") == "Antiadrenal preparations"
    assert labels.name("PATHOLOGY//M97423") == "Morphology codes M97"
    assert labels.name("PATHOLOGY//M95911") == "Morphology codes M95"
    # A code table, rather than prefix arithmetic, supplies the exact diagnosis.
    official = CodeDictionary(tables={"snomed": {"M95911": "Monoclonal B-cell lymphocytosis"}})
    assert official.name("PATHOLOGY//M95911") == "Monoclonal B-cell lymphocytosis"


def test_sks_and_who_tokens_have_the_same_concept_membership_and_rollup():
    drugs = pd.DataFrame({"code": ["MEDICATION//J01CA04", "MEDICATION//MJ01CA04",
                                   "MEDICATION//J02AC01", "PATHOLOGY//J08141"]})
    assert len(concept_tokens(drugs, {"codes": ["MEDICATION//J01"]}, {})) == 2
    assert rollup(drugs.code.iloc[0]) == rollup(drugs.code.iloc[1]) == "MEDICATION//J01C"
    diagnoses = pd.DataFrame({"code": ["DIAGNOSIS//DI509", "DIAGNOSIS//I50.9",
                                       "DIAGNOSIS//I489"]})
    assert len(concept_tokens(diagnoses, {"codes": ["DIAGNOSIS//DI50"]}, {})) == 2
    assert rollup(diagnoses.code.iloc[0]) == rollup(diagnoses.code.iloc[1]) == "DIAGNOSIS//DI50"


def test_sks_parser_retains_historical_records_and_non_ascii_text(tmp_path):
    def row(start, end, label):
        return f"SKS{'DI48':<20}{start}20261001{end}{label:<120}{'':47}"
    file = tmp_path / "SKScomplete.txt"
    file.write_bytes((row("19940101", "20120101", "Atrieflimren") + "\r\n" +
                      row("20120101", "25000101", "Atrieflagren og atrieflimren") + "\r\n")
                     .encode("cp1252"))
    result = read_sks(file)
    assert result.code.tolist() == ["DI48", "DI48"]
    assert result.valid_to.tolist() == ["20120101", "25000101"]
    assert result.label.iloc[1] == "Atrieflagren og atrieflimren"


def test_patobank_table_preserves_result_polarity():
    table = PatobankTable()
    table.feed('<table><tr><th>Code</th><th>Label</th></tr>'
               '<tr><td>FE18UA</td><td>MYC translokation påvist</td></tr>'
               '<tr><td>FE18UB</td><td>MYC translokation <b>ikke</b> påvist</td></tr></table>')
    assert table.rows == [("FE18UA", "MYC translokation påvist"),
                          ("FE18UB", "MYC translokation ikke påvist")]


def test_adaptive_mapping_exposes_mixed_positive_and_negative_ancestors():
    mapping = pd.DataFrame({'code': ['PATHOLOGY//FE16E3', 'PATHOLOGY//FE16E0',
                                     'PATHOLOGY//M632C0'],
                            'adaptive/mapped_code': ['PATHOLOGY//FE16E',
                                                      'PATHOLOGY//FE16E',
                                                      'PATHOLOGY//M632C0']})
    config = {'concepts': [{'name': 'tp53', 'exact_codes': True,
                            'codes': ['PATHOLOGY//FE16E3', 'PATHOLOGY//M632C0']}]}
    result = adaptive_concept_coverage(mapping,
        {'PATHOLOGY//FE16E': 1, 'PATHOLOGY//M632C0': 2}, config).set_index('source_code')
    assert result.loc['PATHOLOGY//FE16E3', 'status'] == 'collapsed_mixed_codes'
    assert result.loc['PATHOLOGY//FE16E3', 'other_original_codes'] == 'PATHOLOGY//FE16E0'
    assert result.loc['PATHOLOGY//M632C0', 'status'] == 'retained_exact'
    assert rollup('PATHOLOGY//M632C0') == 'PATHOLOGY//M632C0'
    assert rollup('PATHOLOGY//M636D0') == 'PATHOLOGY//M636D0'


def test_bin_audit_recovers_zero_based_physical_intervals_and_clipped_tails():
    import numpy as np
    metadata = pd.DataFrame([{'code': 'LABTEST//NPU02319', 'numeric/bin_1': 5.,
                              'numeric/bin_99': 15., 'numeric/bin_edges': [0., .5, 1.]},
                             {'code': 'LABTEST//constant', 'numeric/bin_1': 7.,
                              'numeric/bin_99': 7., 'numeric/bin_edges': []}])
    result = laboratory_bin_definitions(metadata, {'LABTEST//NPU02319//BIN_1': 1})
    bins = result.loc[result.code == 'LABTEST//NPU02319'].set_index('bin_index')
    assert bins.reachable_after_clipping.tolist() == [False, True, True, True]
    np.testing.assert_equal(bins.lower_inclusive.to_numpy(), [np.nan, -np.inf, 10., 15.])
    np.testing.assert_equal(bins.upper_exclusive.to_numpy(), [np.nan, 10., 15., np.inf])
    assert bins.token_in_vocabulary.tolist() == [False, True, False, False]
    # Confirm every raw value maps to the reported interval after clipping.
    for value in [-100., 5., 9.9, 10., 14.9, 15., 100.]:
        index = sum(edge <= np.clip((value-5)/10, 0, 1) for edge in [0., .5, 1.])
        assert bins.loc[index, 'lower_inclusive'] <= value < bins.loc[index, 'upper_exclusive']
    constant = result.loc[result.code == 'LABTEST//constant'].iloc[0]
    assert constant.bin_index == 0 and constant.reachable_after_clipping
    assert constant.lower_inclusive == -np.inf and constant.upper_exclusive == np.inf
