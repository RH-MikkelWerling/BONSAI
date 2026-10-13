import pytest

from opera.run.audit_firstline_input_files import _describe, audit


def test_file_inventory_has_hash_and_size(tmp_path):
    source = tmp_path / "seed.parquet"
    source.write_bytes(b"test")
    result = _describe(source)
    assert result["n_files"] == 1
    assert result["total_bytes"] == 4
    assert len(result["files"][0]["sha256"]) == 64
    with pytest.raises(ValueError, match="Missing"):
        _describe(tmp_path / "missing.parquet")


def test_sql_inputs_are_declared_but_not_treated_as_frozen_files(tmp_path, monkeypatch):
    monkeypatch.setattr("opera.run.audit_firstline_input_files._describe", lambda path: {"path": str(path)})
    report = audit(tmp_path, tmp_path / "firstline.R", tmp_path / "loader.R", {},
                   {"patient": "db.patient", "t_dalycare_diagnoses": "db.diagnoses"})
    assert report["status"] == "loader_sources_unresolved"
    assert report["unresolved_loader_inputs"] == ["icd10_codebook"]
    report = audit(tmp_path, tmp_path / "firstline.R", tmp_path / "loader.R", {},
                   {"patient": "db.patient", "t_dalycare_diagnoses": "db.diagnoses",
                    "icd10_codebook": "db.icd10"})
    assert report["status"] == "sql_live_unfrozen"
