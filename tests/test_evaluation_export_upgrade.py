import pandas as pd

from opera.run.analyse_locked_predictions import read_table
from opera.run.paper_prediction_queue import adds_exports_only


BASE = {"schema": 1, "checkpoint_sha256": "a", "config_sha256": "b", "split": "tuning",
        "horizons": "365", "export_embeddings": False, "export_routes": False}


def test_only_gaining_exports_counts_as_an_upgrade():
    assert adds_exports_only(BASE, {**BASE, "export_routes": True, "export_embeddings": True})
    assert not adds_exports_only(BASE, BASE)
    assert not adds_exports_only({**BASE, "export_routes": True}, BASE)
    assert not adds_exports_only(BASE, {**BASE, "export_routes": True, "checkpoint_sha256": "c"})
    assert not adds_exports_only(BASE, {**BASE, "export_routes": True, "split": "held_out"})


def test_read_table_recognises_parquet_whatever_the_suffix(tmp_path):
    path = tmp_path / "patient_metadata.csv"
    pd.DataFrame({"subject_id": ["1", "2"], "age": [60, 70]}).to_parquet(path)
    assert read_table(path).age.tolist() == [60, 70]


def test_an_evaluation_with_more_exports_serves_a_plainer_request():
    from opera.run.paper_prediction_queue import covers
    richer = {**BASE, "export_routes": True, "export_embeddings": True}
    assert covers(richer, BASE)
    assert covers(BASE, BASE)
    assert not covers(BASE, richer)
    assert not covers({**richer, "checkpoint_sha256": "z"}, BASE)
