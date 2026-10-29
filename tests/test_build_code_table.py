import pandas as pd

from opera.interpretability.code_meaning import CodeDictionary
from opera.run.build_code_table import build_table, count_pairs


def test_daly_care_table_names_codes_and_drops_one_off_text(tmp_path):
    lab = pd.DataFrame({
        "analysiscode": ["NPU02319"] * 6 + ["NPU19658"] * 5 + ["NPU99999"],
        "analysisname": ["B-Hæmoglobin"] * 5 + ["Hgb (B)"]
                        + ["P-Laktatdehydrogenase"] * 5 + ["free text once"],
    })
    path = tmp_path / "lab.parquet"
    lab.to_parquet(path)
    table = build_table(count_pairs([path], "analysiscode", "analysisname"), "npu", min_count=5)
    assert table.set_index("code").label.to_dict() == {
        "NPU02319": "B-Hæmoglobin", "NPU19658": "P-Laktatdehydrogenase"}
    assert table.set_index("code").names_seen["NPU02319"] == 2
    out = tmp_path / "npu.csv"
    table.to_csv(out, index=False)
    dictionary = CodeDictionary.from_config(
        {"code_tables": [{"system": "npu", "path": str(out), "code": "code", "label": "label"}]})
    assert "Laktatdehydrogenase" in dictionary.describe("LABTEST//NPU19658//BIN_2")["name"]


def test_sks_prefixed_atc_codes_are_normalised(tmp_path):
    path = tmp_path / "med.csv"
    pd.DataFrame({"atc": ["MJ01CA04"] * 5, "drug": ["Amoxicillin"] * 5}).to_csv(path, index=False)
    table = build_table(count_pairs([path], "atc", "drug"), "atc")
    assert table.code.tolist() == ["J01CA04"]
