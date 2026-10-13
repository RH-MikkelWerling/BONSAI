from opera.run.check_router_source_arms import latest_metrics


def test_latest_metrics_ignores_sparse_lightning_rows(tmp_path):
    path = tmp_path / "metrics.csv"
    path.write_text(
        "epoch,val/loss,val/cr/route_shared_weight,train/loss\n"
        "0,0.7,1.0,\n"
        "0,,,0.6\n"
        "1,0.5,0.8,\n", encoding="utf-8",
    )
    metrics = latest_metrics(path)
    assert metrics["val/loss"] == ("1", "0.5")
    assert metrics["val/cr/route_shared_weight"] == ("1", "0.8")
    assert metrics["train/loss"] == ("0", "0.6")
