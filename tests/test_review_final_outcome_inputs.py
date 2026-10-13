from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

from opera.run.review_final_outcome_inputs import _manifest_inputs, _support


def test_support_requires_each_split_once_and_uses_correct_horizon() -> None:
    frame = pd.DataFrame({
        "outcome": ["example"] * 6,
        "split": ["train", "tuning", "held_out"] * 2,
        "horizon_days": [30] * 3 + [365] * 3,
        "n_events": [2, 1, 0, 100, 20, 999],
    })
    assert _support(frame, "example", 365) == {"train": 100, "tuning": 20, "held_out": 999}
    with pytest.raises(ValueError, match="Incomplete/duplicate"):
        _support(frame.loc[frame["split"].ne("tuning")], "example", 365)


def test_candidate_manifest_requires_same_release_and_all_inputs_unchanged() -> None:
    survival = Path("/sealed/overall_survival.parquet")
    lab = Path("/sealed/labs.parquet")
    manifest = {"input_sha256": {
        str(survival): "survival_hash", str(lab): "lab_hash",
    }}
    issues: list[dict] = []
    with patch.object(Path, "is_file", return_value=True), patch(
        "opera.run.review_final_outcome_inputs.sha256_file",
        side_effect=lambda path: "survival_hash" if path == survival else "lab_hash",
    ):
        _manifest_inputs(manifest, {survival}, issues, "lab")
        assert issues == []
    with patch.object(Path, "is_file", return_value=True), patch(
        "opera.run.review_final_outcome_inputs.sha256_file", return_value="changed"
    ):
        _manifest_inputs(manifest, {survival}, issues, "lab")
        assert any(item["problem"] == "file hash changed since audit" for item in issues)
    issues.clear()
    with patch.object(Path, "is_file", return_value=True), patch(
        "opera.run.review_final_outcome_inputs.sha256_file",
        side_effect=lambda path: "survival_hash" if path == survival else "lab_hash",
    ):
        _manifest_inputs(manifest, {Path("/sealed/other_survival.parquet")}, issues, "lab")
        assert any(item["problem"] == "candidate uses a different outcome release" for item in issues)
