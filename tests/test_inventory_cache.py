import pandas as pd

from opera.run import paper_analysis_inventory as inventory


def _task(tmp_path, prediction, destination, cache):
    job = {"cohort": "CLL", "outcome": "infection", "horizon_days": 365, "population": None,
           "outcome_path": None, "competing_path": None, "eligibility": None}
    return (str(prediction), job, "tuning", "cif_ipcw", str(destination), str(cache), "code-v1")


def test_tabular_tables_are_cached_and_rebuilt_when_their_input_changes(tmp_path, monkeypatch):
    calls = []

    def fake_canonical(path, job, split, estimand):
        calls.append(path)
        return pd.read_csv(path).assign(split=split)

    monkeypatch.setattr(inventory, "canonical_tabular", fake_canonical)
    prediction = tmp_path / "xgboost_tuning_predictions.csv"
    prediction.write_text("subject_id,probability\n1,0.2\n2,0.7\n")
    cache = tmp_path / "cache"

    first = tmp_path / "a" / "table.parquet"
    assert inventory._tabular_task(_task(tmp_path, prediction, first, cache)) == "built"
    second = tmp_path / "b" / "table.parquet"
    assert inventory._tabular_task(_task(tmp_path, prediction, second, cache)) == "cached"
    assert len(calls) == 1
    pd.testing.assert_frame_equal(pd.read_parquet(first), pd.read_parquet(second))

    # A refitted cell (new prediction file) is never served from the cache.
    prediction.write_text("subject_id,probability\n1,0.3\n2,0.6\n")
    third = tmp_path / "c" / "table.parquet"
    assert inventory._tabular_task(_task(tmp_path, prediction, third, cache)) == "built"
    assert pd.read_parquet(third).probability.tolist() == [0.3, 0.6]


def test_the_cache_key_covers_job_split_estimand_and_code(tmp_path):
    prediction = tmp_path / "p.csv"
    prediction.write_text("subject_id,probability\n1,0.2\n")
    job = {"cohort": "CLL", "outcome": "infection", "horizon_days": 365}
    base = inventory.tabular_cache_key(prediction, job, "tuning", "cif_ipcw", "v1")
    assert base != inventory.tabular_cache_key(prediction, {**job, "horizon_days": 730}, "tuning", "cif_ipcw", "v1")
    assert base != inventory.tabular_cache_key(prediction, job, "held_out", "cif_ipcw", "v1")
    assert base != inventory.tabular_cache_key(prediction, job, "tuning", "binary_known_status_unweighted", "v1")
    assert base != inventory.tabular_cache_key(prediction, job, "tuning", "cif_ipcw", "v2")
