import pandas as pd

from opera.run.monitor_training import resolve_metrics_csv, summarize_metrics


def test_monitor_collapses_sparse_lightning_rows_by_epoch(tmp_path):
    frame = pd.DataFrame(
        {
            "epoch": [None, 0, 0, None, 1, 1],
            "step": [0, 1, 1, 2, 2, 2],
            "train/loss_epoch": [None, 5.0, None, None, 4.0, None],
            "val/loss": [None, None, 5.5, None, None, 4.5],
            "lr-AdamW/pg1": [1e-5, None, None, 3e-5, None, None],
        }
    )

    summary = summarize_metrics(frame)

    assert summary["epoch"].tolist() == [0, 1]
    assert summary["train/loss_epoch"].tolist() == [5.0, 4.0]
    assert summary["val/loss"].tolist() == [5.5, 4.5]
    assert summary["lr-AdamW/pg1"].tolist() == [1e-5, 3e-5]


def test_monitor_selects_newest_nested_metrics_file(tmp_path):
    older = tmp_path / "version_0" / "metrics.csv"
    newer = tmp_path / "version_1" / "metrics.csv"
    older.parent.mkdir()
    newer.parent.mkdir()
    older.write_text("epoch,step\n0,0\n", encoding="utf-8")
    newer.write_text("epoch,step\n1,1\n", encoding="utf-8")
    older.touch()
    newer.touch()
    # Make ordering deterministic even on coarse timestamp filesystems.
    older_mtime = older.stat().st_mtime
    import os
    os.utime(newer, (older_mtime + 2, older_mtime + 2))

    assert resolve_metrics_csv(tmp_path) == newer
