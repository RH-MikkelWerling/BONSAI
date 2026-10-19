import json

import pandas as pd

from opera.run.flatten_ipi_records import flatten


def _records(*rows):
    return json.dumps([dict(zip(("index", "IPI", "PS", "Disease"), r)) for r in rows])


def test_one_row_per_patient_with_an_explicit_rule_for_several_records():
    metadata = pd.DataFrame({
        "subject_id": [1, 2, 3, 4, 5],
        "ipi_records_json": [
            _records(("IPI", "high", "2", "DLBCL")),
            _records(("FLIPI", "low", "0", "FL"), ("FLIPI", "low", "0", "FL")),  # agreeing
            _records(("IPI", "low", "1", "DLBCL"), ("IPI", "high", "1", "DLBCL")),  # conflict
            "[]",
            _records((None, None, "4", "MM")),
        ],
    })
    table = flatten(metadata).set_index("subject_id")
    assert table.loc[1, ["index", "IPI", "PS"]].tolist() == ["IPI", "high", "2"]
    assert table.loc[1, "risk_category"] == "High" and table.loc[1, "performance_status"] == "2–4"
    assert table.loc[2, "ipi_resolution"] == "agreeing" and table.loc[2, "risk_category"] == "Low"
    assert table.loc[3, "ipi_resolution"] == "conflicting"
    assert table.loc[3, "risk_category"] == "Not recorded"
    assert table.loc[3, "performance_status"] == "0–1"  # PS agrees across the records
    assert table.loc[4, "ipi_resolution"] == "none"
    assert table.loc[4, ["risk_category", "performance_status"]].tolist() == ["Not recorded"] * 2
    assert table.loc[5, "risk_category"] == "Not recorded" and table.loc[5, "performance_status"] == "2–4"
    assert table.loc[2, "IPI"] == "low" and pd.isna(table.loc[3, "IPI"])


def test_computed_scores_join_one_to_one(tmp_path):
    import subprocess, sys
    metadata = pd.DataFrame({"subject_id": [1, 2], "cohort_fine": ["DLBCL", "FL"],
                             "ipi_records_json": [_records(("IPI", "high", "2", "DLBCL")), "[]"]})
    metadata.to_parquet(tmp_path / "meta.parquet")
    pd.DataFrame({"subject_id": [1], "nccn_ipi_score": [5], "nccn_ipi_group": ["high-intermediate"]}
                 ).to_parquet(tmp_path / "scores.parquet")
    out = tmp_path / "clinical.parquet"
    subprocess.run([sys.executable, "-m", "opera.run.flatten_ipi_records", "--metadata",
                    str(tmp_path / "meta.parquet"), "--scores", str(tmp_path / "scores.parquet"),
                    "--output", str(out)], check=True, capture_output=True)
    table = pd.read_parquet(out).set_index("subject_id")
    assert table.loc[1, "nccn_ipi_group"] == "high-intermediate"
    assert table.loc[1, "nccn_ipi_risk"] == "Intermediate" and pd.isna(table.loc[2, "nccn_ipi_risk"])
    assert pd.isna(table.loc[2, "nccn_ipi_score"]) and table.loc[2, "risk_category"] == "Not recorded"


def test_nccn_ipi_is_collapsed_and_kept_for_dlbcl_only():
    from opera.run.flatten_ipi_records import nccn_risk, risk_category

    groups = pd.Series(["low", "low-intermediate", "high-intermediate", "high", None, "high"])
    cohorts = pd.Series(["DLBCL"] * 5 + ["FL"])
    assert nccn_risk(groups, cohorts).tolist()[:5] == [
        "Low", "Intermediate", "Intermediate", "High", "Not recorded"]
    assert pd.isna(nccn_risk(groups, cohorts).iloc[5])
    assert risk_category("very high") == "High" and risk_category("medium") == "Intermediate"


def test_dlbcl_risk_group_comes_from_the_computed_nccn_ipi(tmp_path):
    import subprocess, sys
    metadata = pd.DataFrame({
        "subject_id": [1, 2, 3], "cohort_fine": ["DLBCL", "DLBCL", "FL"],
        "ipi_records_json": [_records(("IPI", "low", "0", "DLBCL")),
                             _records(("IPI", "high", "1", "DLBCL")),
                             _records(("FLIPI", "high", "1", "FL"))]})
    metadata.to_parquet(tmp_path / "meta.parquet")
    pd.DataFrame({"subject_id": [1, 2, 3], "nccn_ipi_group": ["high-intermediate", None, "low"]}
                 ).to_parquet(tmp_path / "scores.parquet")
    out = tmp_path / "clinical.parquet"
    subprocess.run([sys.executable, "-m", "opera.run.flatten_ipi_records", "--metadata",
                    str(tmp_path / "meta.parquet"), "--scores", str(tmp_path / "scores.parquet"),
                    "--output", str(out)], check=True, capture_output=True)
    t = pd.read_parquet(out).set_index("subject_id")
    assert (t.loc[1, "risk_category"], t.loc[1, "risk_source"]) == ("Intermediate", "NCCN-IPI (computed)")
    assert (t.loc[2, "risk_category"], t.loc[2, "risk_source"]) == ("High", "IPI")  # not computable
    assert (t.loc[3, "risk_category"], t.loc[3, "risk_source"]) == ("High", "FLIPI")  # not DLBCL
