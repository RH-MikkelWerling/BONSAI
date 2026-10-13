import csv

from bonsai.run.compare_pretraining_runs import epoch_metrics, metrics_path


def test_pretraining_metric_reader_keeps_validation_rows_and_latest_resume(tmp_path):
    version = tmp_path / "version_0"
    version.mkdir()
    path = version / "metrics.csv"
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["epoch", "val/loss", "val/Precision@1"])
        writer.writeheader()
        writer.writerows(
            [
                {"epoch": "0", "val/loss": "", "val/Precision@1": ""},
                {"epoch": "0", "val/loss": "2.0", "val/Precision@1": "0.2"},
                {"epoch": "1", "val/loss": "1.8", "val/Precision@1": "0.3"},
                {"epoch": "1", "val/loss": "1.7", "val/Precision@1": "0.31"},
            ]
        )
    assert metrics_path(str(tmp_path)) == path
    assert epoch_metrics(path) == {
        1: {"val/loss": 2.0, "val/Precision@1": 0.2},
        2: {"val/loss": 1.7, "val/Precision@1": 0.31},
    }
