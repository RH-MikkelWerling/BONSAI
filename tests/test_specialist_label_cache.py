"""Train/tuning specialist labels can be reused without changing their contract."""

from pathlib import Path

import pytest

from opera.functional.outcome_label_cache import cache_contract, load_or_build_labels


def _contract(source: Path, split: str = "train") -> dict:
    return cache_contract(
        cohort_name="fine_a", outcome_name="infection", split=split,
        cohort_config={"cohort_fine_value": "fine_a"},
        outcome_config={"n_hours_end_include": 8760},
        eligibility_scope="ascertainment", require_min_followup=False,
        source_paths={"outcome": str(source)},
    )


def test_specialist_label_cache_reuses_and_invalidates(tmp_path: Path) -> None:
    source = tmp_path / "outcomes.parquet"
    source.write_bytes(b"release-v1")
    calls = []

    def build():
        calls.append(1)
        return {123: {"event": 1, "time_days": 30.0}}

    cache = tmp_path / "secure_cache"
    first = load_or_build_labels(cache, _contract(source), build)
    second = load_or_build_labels(cache, _contract(source), build)
    assert first == second
    assert len(calls) == 1
    load_or_build_labels(cache, _contract(source, "tuning"), build)
    assert len(calls) == 2
    source.write_bytes(b"release-v2-longer")
    load_or_build_labels(cache, _contract(source), build)
    assert len(calls) == 3


def test_specialist_label_cache_refuses_held_out() -> None:
    with pytest.raises(ValueError, match="train/tuning"):
        _contract(Path("outcomes.parquet"), "held_out")
