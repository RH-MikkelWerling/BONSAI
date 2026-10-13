import json

import pytest

from opera.run.snapshot_release_inputs import run


def test_snapshot_is_opt_in_and_verifies_copy(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "input.txt").write_text("frozen", encoding="utf-8")
    spec = tmp_path / "spec.json"
    spec.write_text(json.dumps({"version": 1, "items": [
        {"name": "source", "path": str(source)}]}), encoding="utf-8")
    output = tmp_path / "snapshot"
    dry = run(spec, output)
    assert dry["status"] == "dry_run"
    assert not output.exists()
    copied = run(spec, output, execute=True)
    assert copied["status"] == "copied_and_verified"
    assert copied["files"][0]["source_sha256"] == copied["files"][0]["copy_sha256"]
    assert (output / "source" / "input.txt").read_text(encoding="utf-8") == "frozen"
    with pytest.raises(ValueError, match="overwrite"):
        run(spec, output, execute=True)


def test_snapshot_rejects_nested_source(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "input.txt").write_text("data", encoding="utf-8")
    spec = tmp_path / "spec.json"
    spec.write_text(json.dumps({"version": 1, "items": [
        {"name": "source", "path": str(source)}]}), encoding="utf-8")
    with pytest.raises(ValueError, match="must not be nested"):
        run(spec, source / "snapshot")
