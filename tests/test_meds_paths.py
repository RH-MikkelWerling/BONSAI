from pathlib import Path

from bonsai.functional.meds import resolve_meds_data_dir


def test_resolve_current_meds_cohort_layout(tmp_path: Path):
    data = tmp_path / "data"
    (data / "train").mkdir(parents=True)

    assert resolve_meds_data_dir(tmp_path, ["train", "tuning"]) == data


def test_resolve_legacy_flat_layout(tmp_path: Path):
    (tmp_path / "train").mkdir()

    assert resolve_meds_data_dir(tmp_path, ["train"]) == tmp_path
