"""Preflight of a new MEDS release: what it accepts and what it refuses."""

from datetime import datetime

import pandas as pd
import polars as pl
import pytest

from opera.run import check_meds_release as check


def release(tmp_path, codes, *, held_out=False, numeric=None):
    rows = []
    for subject in (1, 2):
        rows.append({"subject_id": subject, "time": None, "code": "SEX//F",
                     "numeric_value": None})
        for k, code in enumerate(codes):
            rows.append({"subject_id": subject, "time": datetime(2020, 1, 1 + k), "code": code,
                         "numeric_value": (numeric or {}).get(code)})
    for split in ("train", "tuning") + (("held_out",) if held_out else ()):
        (tmp_path / "data" / split).mkdir(parents=True)
        pl.DataFrame(rows, schema={"subject_id": pl.Int64, "time": pl.Datetime,
                                   "code": pl.String, "numeric_value": pl.Float64}
                     ).write_parquet(tmp_path / "data" / split / "0.parquet")
    (tmp_path / "metadata").mkdir()
    (tmp_path / "metadata" / "numeric_metadata.json").write_text("{}")
    return tmp_path


def run(root, **kwargs):
    report = check.Report()
    check.check_layout(root, report)
    files, columns, codes, subjects, numeric, bare = check.scan(root)
    check.check_representation(columns, codes, numeric, bare, report)
    check.check_formats(codes, report)
    check.check_order(files, report)
    check.check_identities(subjects, kwargs.get("membership"), kwargs.get("outcomes"), report)
    return {r["check"]: r["status"] for r in report.rows}


def test_a_joined_release_passes(tmp_path):
    root = release(tmp_path, ["DIAGNOSIS//DC833", "LABTEST//NPU19673//BIN_0",
                              "LABTEST//NPU19748//NEGATIVE", "MEDICATION//J01CA04"],
                   numeric={"LABTEST//NPU19673//BIN_0": 31.0})
    status = run(root)
    assert all(v != "FAIL" for v in status.values()), status


@pytest.mark.parametrize("codes,failing", [
    (["LABTEST//NPU19673"], "labs:numeric_without_bin"),       # numeric, no bin fitted
    (["LABTEST//NPU19673//bin_2"], "labs:suffix_case"),
])
def test_unjoined_or_lower_case_bins_fail(tmp_path, codes, failing):
    root = release(tmp_path, codes, numeric={codes[0]: 31.0})
    assert run(root)[failing] in {"FAIL", "WARN"}
    if failing == "labs:numeric_without_bin":  # a whole release unbinned is not "minor"
        assert run(root)["labs:bins_joined"] == "PASS"


def test_default_split_and_missing_patients_fail(tmp_path):
    root = release(tmp_path, ["LABTEST//NPU1//BIN_0"], held_out=True)
    members = tmp_path / "population.csv"
    pd.DataFrame({"subject_id": [1, 2, 3]}).to_csv(members, index=False)
    status = run(root, membership=members)
    assert status["layout:no_held_out"] == "FAIL"
    assert status["ids:membership_in_release"] == "FAIL"


def test_changed_subject_ids_between_runs_fail(tmp_path):
    old, new = tmp_path / "old.csv", tmp_path / "new.csv"
    pd.DataFrame({"pnr": ["a", "b", "c"], "mapping": [0, 1, 2]}).to_csv(old, index=False)
    pd.DataFrame({"pnr": ["a", "aa", "b", "c"], "mapping": [0, 1, 2, 3]}).to_csv(new,
                                                                                  index=False)
    report = check.Report()
    check.check_mapping(old, new, report)
    assert report.rows[0]["status"] == "FAIL" and report.rows[0]["changed"] == 2


def test_a_bin_that_was_not_joined_fails(tmp_path):
    rows = [{"subject_id": 1, "time": datetime(2020, 1, 1), "code": "LABTEST//NPU1",
             "numeric_value": 3.0, "numeric_value_bin": 2},
            {"subject_id": 1, "time": None, "code": "SEX//F", "numeric_value": None,
             "numeric_value_bin": None}]
    for split in ("train", "tuning"):
        (tmp_path / "data" / split).mkdir(parents=True)
        pl.DataFrame(rows, schema={"subject_id": pl.Int64, "time": pl.Datetime,
                                   "code": pl.String, "numeric_value": pl.Float64,
                                   "numeric_value_bin": pl.Int64}
                     ).write_parquet(tmp_path / "data" / split / "0.parquet")
    assert run(tmp_path)["labs:bins_joined"] == "FAIL"
