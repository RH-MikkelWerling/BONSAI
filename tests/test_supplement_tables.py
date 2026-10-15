import json
import sys
import pandas as pd
import pytest
import yaml
from opera.run.render_supplement_tables import main


def test_reviewed_tables_preserve_provenance_and_escape_clinical_names(tmp_path, monkeypatch):
    source = tmp_path / "summary.csv"
    pd.DataFrame({"score": ["ISS & R-ISS"], "auc": [.8]}).to_csv(source, index=False)
    recipe = tmp_path / "tables.yaml"
    recipe.write_text(yaml.safe_dump({"tables": [{"id": "scores", "source": "summary.csv",
                                                   "caption": "Score comparison", "columns": ["score", "auc"]}]}))
    output = tmp_path / "out"
    monkeypatch.setattr(sys, "argv", ["tables", "--config", str(recipe), "--output-dir", str(output), "--execute"])
    main()
    assert r"ISS \& R-ISS" in (output / "supplement_tables.tex").read_text()
    manifest = json.loads((output / "table_manifest.json").read_text())
    assert len(manifest["tables"][0]["source_sha"]) == 64


def test_empty_templates_cannot_be_rendered_as_results(tmp_path, monkeypatch):
    pd.DataFrame(columns=["score"]).to_csv(tmp_path / "empty.csv", index=False)
    recipe = tmp_path / "tables.yaml"
    recipe.write_text(yaml.safe_dump({"tables": [{"id": "scores", "source": "empty.csv",
                                                   "caption": "Scores", "columns": ["score"]}]}))
    monkeypatch.setattr(sys, "argv", ["tables", "--config", str(recipe), "--output-dir", str(tmp_path / "out"), "--execute"])
    with pytest.raises(ValueError, match="header-only"):
        main()
    assert not (tmp_path / "out").exists()
