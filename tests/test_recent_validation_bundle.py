import csv
import io
import json
import tarfile
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

from opera.run.bundle_recent_validation import build_bundle, sanitized_metrics


def test_metric_columns_seal_test_and_unknown_fields():
    clean, rows, fields = sanitized_metrics(
        "epoch,step,val/auc,train/loss,test/auc,subject_id\n1,3,.76,.2,.9,123\n"
    )
    assert fields == ["epoch", "step", "val/auc", "train/loss"]
    assert "subject_id" not in clean and "test/auc" not in clean
    assert rows[0]["val/auc"] == ".76"


def test_bundle_includes_support_and_seals_predictions():
    # Native temporary directories avoid Windows pytest fixture ACL issues.
    with TemporaryDirectory() as folder:
        root = Path(folder)
        run = root / "suite" / "routergroups"
        (run / ".hydra").mkdir(parents=True)
        (run / ".hydra/config.yaml").write_text("model:\n  freeze_encoder: true\n")
        (run / "checkpoint_metadata.json").write_text("{}")
        (run / "training_complete.json").write_text("{}")
        (run / "metrics.csv").write_text("epoch,val/auc,test/auc\n1,.76,.9\n")
        (run / "source_training_support.csv").write_text(
            "source_name,training_events\ndisease:A,10\n"
        )
        (run / "held_out").mkdir()
        (run / "held_out/metrics.csv").write_text("epoch,val/auc\n1,.99\n")
        (run / "tuning_predictions.csv").write_text("subject_id\n123\n")
        (run / "x_metadata.json").write_text(
            json.dumps({"evaluation_split": "held_out"})
        )
        output = root / "bundle.tar.gz"
        note = build_bundle([root / "suite"], output, root)
        assert note["runs"] == 1
        with tarfile.open(output) as archive:
            names = archive.getnames()
            assert any(n.endswith("checkpoint_metadata.json") for n in names)
            assert any(n.endswith("source_training_support.csv") for n in names)
            assert not any(
                "held_out/" in n or n.endswith("predictions.csv") for n in names
            )
            assert not any(n.endswith("x_metadata.json") for n in names)
            history = (
                archive.extractfile(
                    "recent_validation/runs/root0/routergroups/metrics.csv"
                )
                .read()
                .decode()
            )
            assert "test/auc" not in history
            inventory = list(
                csv.DictReader(
                    io.StringIO(
                        archive.extractfile("recent_validation/run_inventory.csv")
                        .read()
                        .decode()
                    )
                )
            )
            assert inventory[0]["complete"] == "True"
        with pytest.raises(FileExistsError):
            build_bundle([root / "suite"], output, root)
